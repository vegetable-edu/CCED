from torch.nn.functional import gelu
import torch.nn.functional as F
import math
import torch

from collections import OrderedDict
from transformers import RobertaModel, RobertaTokenizer, RobertaForSequenceClassification
from chinese_roberta_model import *
from utils import contrastive_mmd
import torch.nn as nn
import torch.nn.functional as F



def functional_modifiedroberta(fast_weights, prev_fast_weights, config, input_ids=None,
                               attention_mask=None, token_type_ids=None, position_ids=None,
                               head_mask=None, inputs_embeds=None, encoder_hidden_states=None,
                               encoder_attention_mask=None,
                               class_labels=None, trigger_labels=None, scaling_trigger=1.0, zero_shot=False,
                               scaling_contrastive=1.0,
                               output_attentions=None, output_hidden_states=None, is_train=True):

    weight1 = nn.Parameter(torch.tensor(0.5), requires_grad=True)
    weight2 = nn.Parameter(torch.tensor(0.5), requires_grad=True)
    weight3 = nn.Parameter(torch.tensor(0.5), requires_grad=True)
    weight4 = nn.Parameter(torch.tensor(0.5), requires_grad=True)
    weight5 = nn.Parameter(torch.tensor(0.5), requires_grad=True)
    T = nn.Parameter(torch.tensor(3.0), requires_grad=True)

    positive_indices = class_labels == 1
    negative_indices = class_labels == 0
    positive_samples = input_ids[positive_indices]
    negative_samples = input_ids[negative_indices]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # input_ids = input_ids.to(device)
    distill_criterion = nn.CosineEmbeddingLoss()

    outputs = functional_roberta(fast_weights, config, input_ids=input_ids,
                                 attention_mask=attention_mask, token_type_ids=token_type_ids,
                                 position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds,
                                 encoder_hidden_states=encoder_hidden_states,
                                 encoder_attention_mask=encoder_attention_mask, output_attentions=True,
                                 output_hidden_states=output_hidden_states,class_labels=class_labels,
                                 is_train=is_train)

    if prev_fast_weights is not None:
        pre_outputs = functional_roberta(prev_fast_weights, config, input_ids=input_ids,
                                         attention_mask=attention_mask, token_type_ids=token_type_ids,
                                         position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds,
                                         encoder_hidden_states=encoder_hidden_states,
                                         encoder_attention_mask=encoder_attention_mask, output_attentions=True,
                                         output_hidden_states=output_hidden_states,class_labels=class_labels,
                                         is_train=is_train)
    sequence_output = outputs[0]

    split_results = sliding_window_split(sequence_output, 4)
    log_width_differences = calculate_log_width_difference(split_results)
    log_centers = calculate_log_center(split_results)   
    ST = list(zip(log_width_differences, log_centers))
    ST2i = sin_embedding(ST, 5)
    sequence_output = merge_ST2i_sequence_output(sequence_output, ST2i)

    if positive_samples.size(0) & negative_samples.size(0) != 0:
        positive_sequence = sequence_output[class_labels==1]

        negative_sequence = sequence_output[class_labels==0]

        size = input_ids.size(0)
        repeat_whole = size // positive_sequence.size(0)
        extra_samples_needed = size % positive_sequence.size(0)

        repeat_samples = positive_sequence.repeat(repeat_whole, 1,1)

        if extra_samples_needed > 0:
            extra_samples = positive_sequence[:extra_samples_needed]
            positive_samples = torch.cat((repeat_samples, extra_samples), dim=0)
        else:
            positive_samples = repeat_samples

        num_repeats2 = size - negative_sequence.size(0)
        indices_to_repeat = torch.randint(0, negative_sequence.size(0), (num_repeats2,))
        additional_negative_samples = negative_sequence[indices_to_repeat]
        negative_samples = torch.cat((negative_sequence, additional_negative_samples), dim=0)
        positive_reps_emb = F.normalize(positive_samples.view(-1, positive_samples.size(2)), p=2, dim=1)
        negative_reps_emb = F.normalize(negative_samples.view(-1, negative_samples.size(2)), p=2, dim=1)
        normalized_reps_emb = F.normalize(sequence_output.view(-1, sequence_output.size(2)), p=2, dim=1)


 
    if prev_fast_weights is not None:
        pre_sequence_output = pre_outputs[0]
        normalized_prev_reps_emb = F.normalize(pre_sequence_output.view(-1, pre_sequence_output.size(2)), p=2, dim=1)

        assert normalized_reps_emb.shape == normalized_prev_reps_emb.shape, "Input tensors must have the same shape for cosine embedding loss"
        # print(input_ids.shape)
        batch_size = normalized_prev_reps_emb.size(0)
        target = torch.ones(batch_size, dtype=torch.float32)
        #
        target = target.to(normalized_reps_emb.device)
        feature_distill_loss = distill_criterion(normalized_reps_emb, normalized_prev_reps_emb, target)

    pre_trigger_logits = None
    pre_class_logits = None
    
 
    trigger_logits = F.linear(sequence_output,
                              fast_weights['base_model.classifier.weight'],
                              fast_weights['base_model.classifier.bias'])

    # 因为现在的是 outputs = (sequence_output,1,2, pooled_outputs,prev_sequence_output) + encoder_outputs[1:]+ prev_encoder_outputs[1:]
    if prev_fast_weights is not None:
        attention_weights = torch.mean(torch.sum(outputs[3][-1].detach(), dim=-2), dim=1)
    else:
        attention_weights = torch.mean(torch.sum(outputs[-1][-1].detach(), dim=-2), dim=1)
    context_start_pose = (input_ids[0] == 2).max(dim=-1)[-1] + 2
    context_end_poses = (input_ids == 1).max(dim=-1)[-1] - 1
    context_end_poses[context_end_poses == -1] = input_ids.shape[1] - 1
    attention_weights[:, :context_start_pose] = -1e9
    for i in range(input_ids.shape[0]):
        attention_weights[i, context_end_poses[i]:] = -1e9
    # attention_weights = torch.softmax(attention_weights, dim=-1)
    trigger_weights = torch.softmax(trigger_logits, dim=-1)[:, :, 1] * attention_weights

    #
    trigger_probs = torch.softmax(trigger_weights, dim=-1).unsqueeze(-1)

    trigger_probs_logits = torch.softmax(trigger_weights / 2, dim=-1).unsqueeze(-1)

    trigger_rep = (sequence_output * trigger_probs).sum(dim=1)

    mask_pos_ids = (input_ids == 50264).nonzero()[:, 1]  # 50264 is the id for <mask>
    assert len(input_ids) == len(mask_pos_ids)
    x = F.linear(sequence_output[torch.arange(len(input_ids)), mask_pos_ids],
                 fast_weights['base_model.lm_head.dense.weight'],
                 fast_weights['base_model.lm_head.dense.bias'])
    x = gelu(x)
    x = F.layer_norm(x, [config.hidden_size],
                     weight=fast_weights['base_model.lm_head.layer_norm.weight'],
                     bias=fast_weights['base_model.lm_head.layer_norm.bias'],
                     eps=config.layer_norm_eps)
    prediction_scores = F.linear(x,
                                 fast_weights['base_model.lm_head.decoder_weight'],
                                 fast_weights['base_model.lm_head.bias'])
    trigger_aware_logits = torch.cat((prediction_scores, trigger_rep), dim=-1)

    # event的预测
    class_logits = F.linear(gelu(trigger_aware_logits),
                            fast_weights['base_model.verbalizer.weight'],
                            fast_weights['base_model.verbalizer.bias'])

    log_class_logits = F.log_softmax(class_logits / 2, dim=1)

    # scaled_logits = selected_logits / 2
    # log_class_logits = F.log_softmax(scaled_logits, dim=1)

    if prev_fast_weights is not None:
        pre_trigger_logits = F.linear(sequence_output,
                                      prev_fast_weights['base_model.classifier.weight'],
                                      prev_fast_weights['base_model.classifier.bias'])

        # 因为现在的是 outputs = (sequence_output, pooled_outputs,prev_sequence_output) + encoder_outputs[1:]+ prev_encoder_outputs[1:]
        pre_attention_weights = torch.mean(torch.sum(pre_outputs[-1][-1].detach(), dim=-2), dim=1)
        context_start_pose = (input_ids[0] == 2).max(dim=-1)[-1] + 2
        context_end_poses = (input_ids == 1).max(dim=-1)[-1] - 1
        context_end_poses[context_end_poses == -1] = input_ids.shape[1] - 1
        pre_attention_weights[:, :context_start_pose] = -1e9
        for i in range(input_ids.shape[0]):
            pre_attention_weights[i, context_end_poses[i]:] = -1e9
        # attention_weights = torch.softmax(attention_weights, dim=-1)
        pre_trigger_weights = torch.softmax(pre_trigger_logits, dim=-1)[:, :, 1] * pre_attention_weights
        #
        pre_trigger_probs = torch.softmax(pre_trigger_weights / 2, dim=1).unsqueeze(-1)


    total_loss = None
    # 计算损失

    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

    if class_labels is not None:

        loss_fct = CrossEntropyLoss()
        total_loss = loss_fct(class_logits.view(-1, config.num_labels), class_labels.view(-1)) ** 2

        if trigger_labels is not None:
            loss_fct = CrossEntropyLoss()
            triger_loss = scaling_trigger * loss_fct(trigger_logits.view(-1, 2), trigger_labels.view(-1))
            if prev_fast_weights is not None:
                prediction_distill_loss = -torch.mean(torch.sum(pre_trigger_probs * trigger_probs_logits, dim=1))
                prediction_distill_loss /= T
        if zero_shot:
            total_loss += scaling_contrastive * contrastive_mmd(trigger_aware_logits, class_labels)
            if positive_samples.size(0) & negative_samples.size(0) != 0:
                total_loss += triplet_loss(normalized_reps_emb, positive_reps_emb, negative_reps_emb) ** 2
    if prev_fast_weights is not None:
        total_loss += feature_distill_loss ** 2 + triger_loss ** 2 + prediction_distill_loss ** 2

    output = (class_logits, trigger_logits, trigger_aware_logits)

    # 返回所有的损失
    return ((total_loss,) + output) if total_loss is not None else output

def merge_ST2i_sequence_output(sequence_output, ST2i):
 
    batch_size, max_len, hidden_size = sequence_output.shape
    num_pairs = len(ST2i[0])

    # 初始化合并后的张量
    merged_output = torch.zeros(batch_size, max_len, hidden_size + 10)

    # 合并 sequence_output 和 ST2i
    for batch in range(batch_size):
        for idx in range(max_len):
            # 直接复制 sequence_output 的值
            merged_output[batch, idx, :hidden_size] = sequence_output[batch, idx]
            # 复制 ST2i 的值
            for pair_idx in range(num_pairs):
                merged_output[batch, idx, hidden_size + pair_idx*2:(pair_idx+1)*2] = torch.tensor(ST2i[batch][pair_idx])

    return merged_output
 
def sliding_window_split(sequence_output, n):

    batchsize, maxlen, _ = sequence_output.shape
    split_results = []

    for batch in range(batchsize):
        current_batch_splits = []
        for window_size in range(0, n + 1):
            for start_idx in range(0, maxlen):
                end_idx = start_idx + window_size
                if end_idx <= maxlen:
                    slice_ = sequence_output[batch, start_idx:end_idx]
                    # 存储切片和其长度
                    current_batch_splits.append((slice_, window_size))
        split_results.append(current_batch_splits)

    return split_results

def calculate_log_width_difference(split_results):

    log_width_differences = []

    for batch_splits in split_results:
        batch_log_width_differences = []
        for i in range(len(batch_splits)):
            for j in range(i+1, len(batch_splits)):
                w = abs(batch_splits[i][1] - batch_splits[j][1])
                # 确保宽度差大于0
                if w > 0:
                    log_w = math.log(w)
                    batch_log_width_differences.append(log_w)
                else:
                    # 如果宽度差为0，则可以考虑设定一个默认值
                    # 例如，可以将对数宽度设置为负无穷或一个大的负数
                    batch_log_width_differences.append(float('-inf'))
        log_width_differences.append(batch_log_width_differences)

    return log_width_differences

def calculate_log_center(split_results):

    log_centers = []

    for batch_splits in split_results:
        batch_log_centers = []
        for i in range(len(batch_splits)):
            for j in range(i+1, len(batch_splits)):
                w1 = batch_splits[i][1]
                w2 = batch_splits[j][1]
                # 确保宽度不为0
                if w1 > 0 and w2 > 0:
                    center = math.log(w1 / w2)
                    batch_log_centers.append(center)
                else:
                    # 如果任一宽度为0，则可以考虑设定一个默认值
                    # 例如，可以将center设置为负无穷或一个大的负数
                    batch_log_centers.append(float('-inf'))
        log_centers.append(batch_log_centers)

    return log_centers

def sin_embedding(ST, d):

    ST2i = []

    # 遍历ST中的每个元素
    for st in ST:
        embedded = []
        # 计算每个维度的嵌入
        for i in range(d):
            factor = 10000 ** (2 * i / d)
            # 对于ST中的每个元素(st)，都包含两个值(width, center)，我们需要分别处理
            embedded.append(math.sin(st[0] / factor))
            embedded.append(math.sin(st[1] / factor))
        ST2i.append(embedded)

    return ST2i



def functional_robertaforclassification(fast_weights, config, input_ids=None, attention_mask=None, token_type_ids=None,
                                        position_ids=None,
                                        head_mask=None, inputs_embeds=None, encoder_hidden_states=None,
                                        encoder_attention_mask=None,
                                        labels=None, output_attentions=None, output_hidden_states=None, is_train=True):
    outputs = functional_roberta(fast_weights, config, input_ids=input_ids, attention_mask=attention_mask,
                                 token_type_ids=token_type_ids,
                                 position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds,
                                 encoder_hidden_states=encoder_hidden_states,
                                 encoder_attention_mask=encoder_attention_mask, output_attentions=output_attentions,
                                 output_hidden_states=output_hidden_states,class_labels=labels,
                                 is_train=is_train)

    sequence_output = outputs[0]

    dropout_rate = config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
    x = sequence_output[:, 0, :]
    x = F.dropout(x, dropout_rate, training=is_train)
    x = F.linear(x,
                 fast_weights['base_model.classifier.dense.weight'],
                 fast_weights['base_model.classifier.dense.bias'])
    x = torch.tanh(x)
    x = F.dropout(x, dropout_rate, training=is_train)
    logits = F.linear(x,
                      fast_weights['base_model.classifier.out_proj.weight'],
                      fast_weights['base_model.classifier.out_proj.bias'])

    loss = None
    if labels is not None:
        if config.num_labels == 1:
            # We are doing regression
            loss = F.mse_loss(logits.view(-1), labels.view(-1))
        else:
            loss = F.cross_entropy(logits.view(-1, config.num_labels), labels.view(-1))

    output = (logits,) + outputs[1:]
    return ((loss,) + output) if loss is not None else output


def functional_roberta(fast_weights, config, input_ids=None, attention_mask=None,
                       token_type_ids=None,
                       position_ids=None, head_mask=None, inputs_embeds=None, encoder_hidden_states=None,
                       encoder_attention_mask=None, output_attentions=None, output_hidden_states=None, class_labels=None,is_train=True):
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        input_shape = input_ids.size()
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")
        # 用这些索引来获取正样本和负样本

    device = input_ids.device if input_ids is not None else inputs_embeds.device

    if attention_mask is None:
        attention_mask = torch.ones(input_shape, device=device)
    if token_type_ids is None:
        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        if config.is_decoder:
            batch_size, seq_length = input_shape
            seq_ids = torch.arange(seq_length, device=device)
            causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
            causal_mask = causal_mask.to(torch.long)
            extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
        else:
            extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError("Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(input_shape,
                                                                                                    attention_mask.shape))

    extended_attention_mask = extended_attention_mask.to(
        dtype=next((p for p in fast_weights.values())).dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

    # 如果提供了encoder_hidden_states，也会为其生成相应的encoder_extended_attention_mask。
    # 主要是为了使用跨注意力机制
    if config.is_decoder and encoder_hidden_states is not None:
        encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        elif encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                    encoder_hidden_shape,
                    encoder_attention_mask.shape))
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(
            dtype=next((p for p in fast_weights.values())).dtype)
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
    else:
        encoder_extended_attention_mask = None

    if head_mask is not None:
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(config.num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        head_mask = head_mask.to(dtype=next((p for p in fast_weights.values())).dtype)
    else:
        head_mask = [None] * config.num_hidden_layers


    # 获得embedding
    embedding_output = functional_embedding(fast_weights, config, input_ids, position_ids,
                                            token_type_ids, inputs_embeds, is_train=is_train)

    # 最终输出包括最后一层的隐藏状态、（可选的）所有层的隐藏状态和注意力权重。
    encoder_outputs = functional_encoder(fast_weights, config, embedding_output, attention_mask=extended_attention_mask,
                                         head_mask=head_mask,
                                         encoder_hidden_states=encoder_hidden_states,
                                         encoder_attention_mask=encoder_extended_attention_mask,
                                         output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                                         is_train=is_train)


    prev_sequence_output = None
    prev_encoder_outputs = None

    # 最后一层隐藏层状态
    sequence_output = encoder_outputs[0]

    pooled_outputs = None
    if 'roberta.pooler.dense.weight' in fast_weights:
        # pooled_outputs = functional_pooler(fast_weights, sequence_output)
        pooled_outputs = functional_pooler_with_dynamic_pooling(fast_weights, sequence_output, 2)

    outputs = (sequence_output, pooled_outputs, prev_sequence_output) + encoder_outputs[1:]
    return outputs


def functional_embedding(fast_weights, config, input_ids, position_ids,
                         token_type_ids, inputs_embeds=None, is_train=True):
    if input_ids is not None:
        input_shape = input_ids.size()
    else:
        input_shape = inputs_embeds.size()[:-1]

    seq_length = input_shape[1]
    device = input_ids.device if input_ids is not None else inputs_embeds.device
    if position_ids is None:
        # position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        # position_ids = position_ids.unsqueeze(0).expand(input_shape)
        position_ids = create_position_ids_from_input_ids(input_ids, config.pad_token_id, 0)

    if token_type_ids is None:
        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

    if inputs_embeds is None:
        inputs_embeds = F.embedding(input_ids, fast_weights['base_model.roberta.embeddings.word_embeddings.weight'],
                                    padding_idx=0)

    position_embeddings = F.embedding(position_ids, fast_weights['base_model.roberta.embeddings.position_embeddings'
                                                                 '.weight'])
    token_type_embeddings = F.embedding(token_type_ids, fast_weights['base_model.roberta.embeddings'
                                                                     '.token_type_embeddings.weight'])
    # 获得一个完整的embedding表示，输入入计算词嵌入、位置嵌入和token类型嵌入
    embeddings = inputs_embeds + position_embeddings + token_type_embeddings

    # 选择是否使用归一层或者dropout层
    embeddings = F.layer_norm(embeddings, [config.hidden_size],
                              weight=fast_weights['base_model.roberta.embeddings.LayerNorm.weight'],
                              bias=fast_weights['base_model.roberta.embeddings.LayerNorm.bias'],
                              eps=config.layer_norm_eps)

    embeddings = F.dropout(embeddings, p=config.hidden_dropout_prob, training=is_train)

    return embeddings


def functional_encoder(fast_weights, config, hidden_states, attention_mask, head_mask, encoder_hidden_states,
                       encoder_attention_mask, output_attentions=False, output_hidden_states=False, is_train=True):
    all_hidden_states = () if output_hidden_states else None
    all_self_attentions = () if output_attentions else None

    for i in range(0, config.num_hidden_layers):
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        layer_outputs = functional_layer(fast_weights, config, str(i), hidden_states,
                                         attention_mask, head_mask[i], encoder_hidden_states,
                                         encoder_attention_mask, output_attentions, is_train)
        hidden_states = layer_outputs[0]

        if output_attentions:
            all_self_attentions = all_self_attentions + (layer_outputs[1],)

    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    outputs = (hidden_states, all_hidden_states, all_self_attentions)

    return tuple(v for v in [
        hidden_states,
        all_hidden_states,
        all_self_attentions, ] if v is not None)


def functional_pooler(fast_weights, sequence_output):
    first_token_tensor = sequence_output[:, 0]
    pooled_output = F.linear(first_token_tensor,
                             fast_weights['base_model.roberta.pooler.dense.weight'],
                             fast_weights['base_model.roberta.pooler.dense.bias'])
    pooled_output = torch.tanh(pooled_output)

    return pooled_output


def dynamic_k_max_pooling(sequence_output, k):
    # sequence_output的形状假设为(batch_size, seq_length, hidden_size)
    # 我们要从序列长度维度(seq_length)选择k个最大的特征
    # 使用topk方法获取最大值及其索引
    topk_pooled, _ = sequence_output.topk(k, dim=1)
    # 然后将这些最大值沿着序列长度维度拼接起来
    # 这会给我们一个形状为(batch_size, k, hidden_size)的tensor
    # 为了匹配原始的pooled_output形状，我们需要将最后两个维度合并
    return topk_pooled.view(sequence_output.size(0), -1)


def functional_pooler_with_dynamic_pooling(fast_weights, sequence_output, k=1):
    # 使用dynamic k-max pooling代替取第一个token
    pooled_output = dynamic_k_max_pooling(sequence_output, k)
    # 我们假设fast_weights中有一个适用于新形状的权重和偏置
    pooled_output = F.linear(pooled_output,
                             fast_weights['base_model.roberta.pooler.dense.weight'],
                             fast_weights['base_model.roberta.pooler.dense.bias'])
    pooled_output = torch.tanh(pooled_output)

    return pooled_output


def functional_layer(fast_weights, config, layer_idx, hidden_states, attention_mask, head_mask,
                     encoder_hidden_states, encoder_attention_mask, output_attentions=False, is_train=True):
    self_attention_outputs = functional_attention(fast_weights, config, layer_idx, hidden_states,
                                                  attention_mask, head_mask, encoder_hidden_states,
                                                  encoder_attention_mask, output_attentions, is_train)
    attention_output = self_attention_outputs[0]
    outputs = self_attention_outputs[1:]

    intermediate_output = functional_intermediate(fast_weights, config, layer_idx, attention_output, is_train)
    layer_output = functional_output(fast_weights, config, layer_idx,
                                     intermediate_output, attention_output, is_train)
    outputs = (layer_output,) + outputs

    return outputs


def functional_attention(fast_weights, config, layer_idx, hidden_states,
                         attention_mask=None, head_mask=None, encoder_hidden_states=None,
                         encoder_attention_mask=None, output_attentions=False, is_train=True):
    self_outputs = functional_self_attention(fast_weights, config, layer_idx, hidden_states,
                                             attention_mask, head_mask, encoder_hidden_states,
                                             encoder_attention_mask, output_attentions, is_train)

    attention_output = functional_out_attention(fast_weights, config, layer_idx,
                                                self_outputs[0], hidden_states, is_train)
    outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them

    return outputs


def functional_intermediate(fast_weights, config, layer_idx, hidden_states, is_train=True):
    weight_name = 'base_model.roberta.encoder.layer.' + layer_idx + '.intermediate.dense.weight'
    bias_name = 'base_model.roberta.encoder.layer.' + layer_idx + '.intermediate.dense.bias'
    hidden_states = F.linear(hidden_states, fast_weights[weight_name], fast_weights[bias_name])
    hidden_states = gelu(hidden_states)

    return hidden_states


def functional_output(fast_weights, config, layer_idx, hidden_states, input_tensor, is_train=True):
    hidden_states = F.linear(hidden_states,
                             fast_weights['base_model.roberta.encoder.layer.' + layer_idx + '.output.dense.weight'],
                             fast_weights['base_model.roberta.encoder.layer.' + layer_idx + '.output.dense.bias'])

    hidden_states = F.dropout(hidden_states, p=config.hidden_dropout_prob, training=is_train)
    hidden_states = F.layer_norm(hidden_states + input_tensor, [config.hidden_size],
                                 weight=fast_weights[
                                     'base_model.roberta.encoder.layer.' + layer_idx + '.output.LayerNorm.weight'],
                                 bias=fast_weights[
                                     'base_model.roberta.encoder.layer.' + layer_idx + '.output.LayerNorm.bias'],
                                 eps=config.layer_norm_eps)

    return hidden_states


def functional_self_attention(fast_weights, config, layer_idx, hidden_states,
                              attention_mask, head_mask, encoder_hidden_states,
                              encoder_attention_mask, output_attentions=False, is_train=True):
    attention_head_size = int(config.hidden_size / config.num_attention_heads)
    all_head_size = config.num_attention_heads * attention_head_size

    mixed_query_layer = F.linear(hidden_states,
                                 fast_weights[
                                     'base_model.roberta.encoder.layer.' + layer_idx + '.attention.self.query.weight'],
                                 fast_weights[
                                     'base_model.roberta.encoder.layer.' + layer_idx + '.attention.self.query.bias'])

    if encoder_hidden_states is not None:
        mixed_key_layer = F.linear(encoder_hidden_states,
                                   fast_weights[
                                       'base_model.roberta.encoder.layer.' + layer_idx + '.attention.self.key.weight'],
                                   fast_weights[
                                       'base_model.roberta.encoder.layer.' + layer_idx + '.attention.self.key.bias'])
        mixed_value_layer = F.linear(encoder_hidden_states,
                                     fast_weights[
                                         'base_model.roberta.encoder.layer.' + layer_idx + '.attention.self.value.weight'],
                                     fast_weights[
                                         'base_model.roberta.encoder.layer.' + layer_idx + '.attention.self.value.bias'])
        attention_mask = encoder_attention_mask
    else:
        mixed_key_layer = F.linear(hidden_states,
                                   fast_weights[
                                       'base_model.roberta.encoder.layer.' + layer_idx + '.attention.self.key.weight'],
                                   fast_weights[
                                       'base_model.roberta.encoder.layer.' + layer_idx + '.attention.self.key.bias'])
        mixed_value_layer = F.linear(hidden_states,
                                     fast_weights[
                                         'base_model.roberta.encoder.layer.' + layer_idx + '.attention.self.value.weight'],
                                     fast_weights[
                                         'base_model.roberta.encoder.layer.' + layer_idx + '.attention.self.value.bias'])

    query_layer = transpose_for_scores(config, mixed_query_layer)
    key_layer = transpose_for_scores(config, mixed_key_layer)
    value_layer = transpose_for_scores(config, mixed_value_layer)

    # Take the dot product between "query" and "key" to get the raw attention scores.
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    attention_scores = attention_scores / math.sqrt(attention_head_size)
    if attention_mask is not None:
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

    attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)

    if is_train:
        attention_probs = F.dropout(attention_probs, p=config.attention_probs_dropout_prob)

    # Mask heads if we want to
    if head_mask is not None:
        attention_probs = attention_probs * head_mask

    context_layer = torch.matmul(attention_probs, value_layer)

    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (all_head_size,)
    context_layer = context_layer.view(*new_context_layer_shape)

    outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

    return outputs


def functional_out_attention(fast_weights, config, layer_idx,
                             hidden_states, input_tensor,
                             is_train=True):
    hidden_states = F.linear(hidden_states,
                             fast_weights[
                                 'base_model.roberta.encoder.layer.' + layer_idx + '.attention.output.dense.weight'],
                             fast_weights[
                                 'base_model.roberta.encoder.layer.' + layer_idx + '.attention.output.dense.bias'])

    hidden_states = F.dropout(hidden_states, p=config.hidden_dropout_prob, training=is_train)
    hidden_states = F.layer_norm(hidden_states + input_tensor, [config.hidden_size],
                                 weight=fast_weights[
                                     'base_model.roberta.encoder.layer.' + layer_idx + '.attention.output.LayerNorm.weight'],
                                 bias=fast_weights[
                                     'base_model.roberta.encoder.layer.' + layer_idx + '.attention.output.LayerNorm.bias'],
                                 eps=config.layer_norm_eps)

    return hidden_states


def transpose_for_scores(config, x):
    new_x_shape = x.size()[:-1] + (config.num_attention_heads, int(config.hidden_size / config.num_attention_heads))
    x = x.view(*new_x_shape)

    return x.permute(0, 2, 1, 3)


def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask

    return incremental_indices.long() + padding_idx