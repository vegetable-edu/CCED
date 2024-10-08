import re
import json
import itertools
import numpy as np
from tqdm import tqdm, trange
from collections import OrderedDict

import torch
import torch.nn.functional as F

from torch.utils.data._utils.collate import default_collate
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from chinese_roberta_model import *
from roberta_method import *

import random
import argparse
from datetime import datetime
from dataloader_csed import get_csed_loaders
from collections import defaultdict
from sklearn.metrics import f1_score
from utils import *
from copy import deepcopy
from peft import TaskType, get_peft_model, PromptTuningInit, PromptTuningConfig, PeftModel

def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sample_set(query_set, batch_size=50):
    permute_ids = np.random.permutation(len(query_set['input_ids']))[:batch_size]
    batch = defaultdict(list)
    for key in query_set:
        for idx in permute_ids:
            batch[key].append(query_set[key][idx])
    return [batch]  # one sampled batch only


def split_set(query_set, batch_size=50):
    batches = []
    for start_idx in range(0, len(query_set[list(query_set)[0]]), batch_size):
        batch = defaultdict(list)
        for key in query_set:
            batch[key] = query_set[key][start_idx:start_idx+batch_size]
        batches.append(batch)
    return batches


def to_tensor_to_device(inputs, device):
    for k in inputs:
        inputs[k] = torch.tensor(inputs[k]).to(device)
    return inputs


def evaluations(args, model, scaler, inner_loop_optimizer, eval_iter):
    print('Evaluating...')
    support_set, query_set = next(eval_iter)
    if len(support_set) == 1:
        support_set, query_set = support_set[0], query_set[0]
    assert len(support_set) == len(query_set)

    f1_micro = []
    with torch.autocast(device_type=args.device, dtype=torch.float16):
        for task in range(len(support_set)):
            gts, preds = [], []
            model.verbalizer_uniform_init()
            fast_weights = OrderedDict(model.named_parameters())  # start from original params

            def remove_base_model_from_keys(fast_weights):
                new_fast_weights = {}
                for key in fast_weights.keys():
                    # 使用正则表达式替换第一个出现的'base_model.'子字符串
                    new_key = re.sub(r'^base_model\.', '', key)
                    new_fast_weights[new_key] = fast_weights[key]
                return new_fast_weights

            fast_weights = remove_base_model_from_keys(fast_weights)
            for name, param in fast_weights.items():
                if not param.requires_grad:
                    param.requires_grad_(True)

            prev_fast_weights = None
            #每一次内部任务
            for k in range(args.task_updates):
                batches = sample_set(support_set[task], batch_size=args.max_batchsize)[:args.max_batch]
                for idx, support in enumerate(batches):
                    data = to_tensor_to_device(support, args.device)
                    trigger_logits, event_logits, task_grads = None, None, None
                    input_ids, attention_mask = data['input_ids'], data['attention_mask']
                    labels, trigger_mask = data['label'], data['trigger_mask']

                    outputs = functional_modifiedroberta(fast_weights,prev_fast_weights, model.config,
                                                         input_ids=input_ids,
                                                         attention_mask=attention_mask,
                                                         class_labels=labels,
                                                         trigger_labels=trigger_mask,
                                                         scaling_trigger=args.scaling_trigger,
                                                         is_train=True)
                    task_loss = outputs[0]
                    task_loss = task_loss.requires_grad_(True)
                    scaled_grads = torch.autograd.grad(scaler.scale(task_loss), fast_weights.values(),allow_unused=True)

                    scaled_grads = [grad if grad is not None else torch.zeros_like(param) for grad, param in
                                    zip(scaled_grads, fast_weights.values())]
                    inv_scale = 1. / scaler.get_scale()
                    grads = [p * inv_scale for p in scaled_grads]  # manually unscale task gradients
                    clip_task_grad_norm(grads, args.max_grad_norm)
                    if any([False in torch.isfinite(g) for g in grads]):
                        print('Invalid task gradients, adjust scale and zero out gradients')
                        if scaler.get_scale() * scaler.get_backoff_factor() >= 1.:
                            scaler.update(scaler.get_scale() * scaler.get_backoff_factor())
                        for g in grads: g.zero_()

                    if not task_grads:
                        task_grads = grads
                    else:
                        for f_g, g in zip(task_grads, grads): f_g.add_(g)

                model.zero_grad()
                for g in task_grads: g.div_(len(batches))
                fast_weights = inner_loop_optimizer.update_params(fast_weights, task_grads, k)
                del outputs

            with torch.no_grad():
                batches = split_set(query_set[task], batch_size=args.max_batchsize)
                if len(batches) == 0: continue
                for query in batches:
                    data = to_tensor_to_device(query, args.device)

                    input_ids, attention_mask, labels = data['input_ids'], data['attention_mask'], data['label']
                    outputs = functional_modifiedroberta(fast_weights, prev_fast_weights, model.config,
                                                         input_ids=input_ids,
                                                         attention_mask=attention_mask,
                                                         is_train=False)
                    gts.extend(labels.cpu().numpy().tolist())
                    preds.extend(outputs[0].argmax(-1).cpu().numpy().tolist())

            f1_micro.append(f1_score(gts, preds, average='micro'))

    return np.mean(f1_micro)

def evaluation(args, model, scaler, inner_loop_optimizer, eval_iter):
    print('Evaluating...')
    support_set, query_set = next(eval_iter)
    if len(support_set) == 1:
        support_set, query_set = support_set[0], query_set[0]
    assert len(support_set) == len(query_set)

    f1_micro = []
    with torch.autocast(device_type=args.device, dtype=torch.float16):
        temp_protos = {}
        for task in range(len(support_set)):
            gts, preds = [], []
            model.verbalizer_uniform_init()
            fast_weights = OrderedDict(model.named_parameters())  # start from original params
            for name, param in fast_weights.items():
                if not param.requires_grad:
                    param.requires_grad_(True)

            prev_fast_weights = None
            #每一次内部任务
            for k in range(args.task_updates):
                batches = sample_set(support_set[task], batch_size=args.max_batchsize)[:args.max_batch]
                for idx, support in enumerate(batches):
                    data = to_tensor_to_device(support, args.device)
                    trigger_logits, event_logits, task_grads = None, None, None
                    input_ids, attention_mask = data['input_ids'], data['attention_mask']
                    labels, trigger_mask = data['label'], data['trigger_mask']
                    
                    

                    outputs = functional_modifiedroberta(fast_weights,prev_fast_weights, model.config,
                                                         input_ids=input_ids,
                                                         attention_mask=attention_mask,
                                                         class_labels=labels,
                                                         trigger_labels=trigger_mask,
                                                         scaling_trigger=args.scaling_trigger,
                                                         is_train=True)
                    task_loss = outputs[0]
                    task_loss = task_loss.requires_grad_(True)
                    scaled_grads = torch.autograd.grad(scaler.scale(task_loss), fast_weights.values(),allow_unused=True)

                    scaled_grads = [grad if grad is not None else torch.zeros_like(param) for grad, param in
                                    zip(scaled_grads, fast_weights.values())]
                    inv_scale = 1. / scaler.get_scale()
                    grads = [p * inv_scale for p in scaled_grads]  # manually unscale task gradients
                    clip_task_grad_norm(grads, args.max_grad_norm)
                    if any([False in torch.isfinite(g) for g in grads]):
                        print('Invalid task gradients, adjust scale and zero out gradients')
                        if scaler.get_scale() * scaler.get_backoff_factor() >= 1.:
                            scaler.update(scaler.get_scale() * scaler.get_backoff_factor())
                        for g in grads: g.zero_()

                    if not task_grads:
                        task_grads = grads
                    else:
                        for f_g, g in zip(task_grads, grads): f_g.add_(g)

                model.zero_grad()
                for g in task_grads: g.div_(len(batches))
                fast_weights = inner_loop_optimizer.update_params(fast_weights, task_grads, k)
                del outputs

            with torch.no_grad():
                batches = split_set(query_set[task], batch_size=args.max_batchsize)
                if len(batches) == 0: continue
                for query in batches:
                    data = to_tensor_to_device(query, args.device)

                    input_ids, attention_mask, labels = data['input_ids'], data['attention_mask'], data['label']
                    outputs = functional_modifiedroberta(fast_weights, prev_fast_weights, model.config,
                                                         input_ids=input_ids,
                                                         attention_mask=attention_mask,
                                                         is_train=False)
                    gts.extend(labels.cpu().numpy().tolist())
                    preds.extend(outputs[0].argmax(-1).cpu().numpy().tolist())

            f1_micro.append(f1_score(gts, preds, average='micro'))

    return np.mean(f1_micro)


def train_fewshot(args):
    fix_random_seed_as(args.seed)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not args.output_dir:
        args.output_dir = datetime.now().strftime("%Y%m%d%H%M%S")
    export_root = os.path.join(args.experiment_dir, args.output_dir)

    if not os.path.exists(export_root):
        os.makedirs(export_root)

    print(args)

    tokenizer = RobertaTokenizer.from_pretrained('../roberta-base')
    train_loader, val_loader, test_loader, vocab = get_csed_loaders(args, args.N, args.K, args.max_len,
                                                                     tokenizer, prompt_template=args.template,
                                                                     train_size=args.num_task)
    train_iter, val_iter, test_iter = iter(train_loader), iter(val_loader), iter(test_loader)

    model = ModifiedRoberta.from_pretrained('../roberta-base', num_labels=2).to(args.device)  
    params = sum(param.numel() for param in model.parameters())

    print(params)
    configs = PromptTuningConfig(task_type=TaskType.CAUSAL_LM, prompt_tuning_init=PromptTuningInit.TEXT,
                                  prompt_tuning_init_text="This type of event is",
                                  num_virtual_tokens=len(tokenizer("This type of event is")["input_ids"]),
                                  tokenizer_name_or_path="../roberta-base"
                                  )

    # 调用 get_peft_model 函数修改模型
    model = get_peft_model(model, configs).to(args.device)

    print(model.print_trainable_parameters())
   # print(model)


    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr_meta, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                           T_max=args.train_iterations,
                                                           eta_min=args.lr_meta * args.lr_decay_meta)

    inner_loop_optimizer = LSLRGradientDescentLearningRule(device=args.device,
                                                           total_num_inner_loop_steps=args.task_updates,
                                                           init_learning_rate=args.lr_learner,
                                                           clf_lr_multiplier=args.clf_lr_multiplier,
                                                           use_learnable_lr=True,
                                                           lr_of_lr=args.lr_of_lr,
                                                           max_grad_norm=args.max_grad_norm)
    inner_loop_optimizer.initialize(names_weights_dict=model.named_parameters())

    # training
    losses = []
    max_f1_micro = 0.
    scaler = torch.cuda.amp.GradScaler()
    tqdm_iterations = trange(args.train_iterations)

    for iteration in tqdm_iterations:
        model.train()
        support_set, query_set = next(train_iter)
        assert len(support_set) == len(query_set)
        with torch.autocast(device_type=args.device, dtype=torch.float16):
            task_accs = []
            meta_gradients = []
            for task in range(len(support_set)):
                model.verbalizer_uniform_init()
                fast_weights = OrderedDict(model.named_parameters())

                # 检查当前上下文中梯度计算是否被激活
               # print(f"Is gradient computation enabled? {torch.is_grad_enabled()}")
                for name, param in fast_weights.items():
                    if not param.requires_grad:
                        param.requires_grad_(True) 
                prev_fast_weights = None
                pre_outputs = None
  
                for k in range(args.task_updates):
                    batches = sample_set(support_set[task], batch_size=args.max_batchsize)[:args.max_batch]
                    task_grads = None
                    for idx, support in enumerate(batches):
                        data = to_tensor_to_device(support, args.device)
                        input_ids, attention_mask = data['input_ids'], data['attention_mask']
                        labels, trigger_mask = data['label'], data['trigger_mask']
        
                        outputs = functional_modifiedroberta(fast_weights, prev_fast_weights, model.config,
                                                             input_ids=input_ids,
                                                             attention_mask=attention_mask,
                                                             class_labels=labels,
                                                             trigger_labels=trigger_mask,
                                                             scaling_trigger=args.scaling_trigger,
                                                             is_train=True)
                        task_loss = outputs[0]

                        scaled_grads = torch.autograd.grad(scaler.scale(task_loss), fast_weights.values(), allow_unused=True)

                        scaled_grads = [grad if grad is not None else torch.zeros_like(param) for grad, param in zip(scaled_grads, fast_weights.values())]
                        inv_scale = 1. / scaler.get_scale()

                        grads = [p * inv_scale for p in scaled_grads]  # manually unscale task gradients
                        clip_task_grad_norm(grads, args.max_grad_norm)
                        if any([False in torch.isfinite(g) for g in grads]):
                            print('Invalid task gradients, adjust scale and zero out gradients')
                            if scaler.get_scale() * scaler.get_backoff_factor() >= 1.:
                                scaler.update(scaler.get_scale() * scaler.get_backoff_factor())
                            for g in grads: g.zero_()

                        if not task_grads:
                            task_grads = grads
                        else:
                            for f_g, g in zip(task_grads, grads): f_g.add_(g)

                    model.zero_grad()
                    for g in task_grads: g.div_(len(batches))

                    #prev_fast_weights = deepcopy(fast_weights)
                    prev_fast_weights = {name: param.clone() for name, param in fast_weights.items()}
                   # pre_outputs = tuple(o.clone().detach() for o in outputs if isinstance(o, torch.Tensor))
                    fast_weights = inner_loop_optimizer.update_params(fast_weights, task_grads, k)


                meta_grads = None
                sum_query_loss = 0.
                batches = split_set(query_set[task], batch_size=args.max_batchsize)[:args.max_batch]
                for idx, query in enumerate(batches):
                    data = to_tensor_to_device(query, args.device)
                    input_ids, attention_mask = data['input_ids'], data['attention_mask']
                    labels, trigger_mask = data['label'], data['trigger_mask']

                    outputs = functional_modifiedroberta(fast_weights, prev_fast_weights, model.config,
                                                         input_ids=input_ids,
                                                         attention_mask=attention_mask,
                                                         class_labels=labels,
                                                         trigger_labels=trigger_mask,
                                                         scaling_trigger=args.scaling_trigger,
                                                         is_train=True)
                    query_loss = outputs[0]
                    sum_query_loss += query_loss.item()
                    task_accs.append((outputs[1].argmax(-1) == labels).float().mean().item())
                    # scaled_grads = torch.autograd.grad(scaler.scale(query_loss), fast_weights.values())
                    if idx == 0:
                        scaled_grads = torch.autograd.grad(scaler.scale(query_loss), fast_weights.values(),
                                                           retain_graph=True, allow_unused=True)
                        scaled_grads = [grad if grad is not None else torch.zeros_like(param) for grad, param in
                                        zip(scaled_grads, fast_weights.values())]
                        inner_loop_optimizer.update_lrs(query_loss, scaler)
                        #Invalid LR gradients, adjust scale and zero out gradients
                    else:
                        scaled_grads = torch.autograd.grad(scaler.scale(query_loss), fast_weights.values(),allow_unused=True)
                        scaled_grads = [grad if grad is not None else torch.zeros_like(param) for grad, param in
                                        zip(scaled_grads, fast_weights.values())]

                    if any([False in torch.isfinite(g) for g in scaled_grads]):
                        print('Invalid task gradients, adjust scale and zero out gradients')
                        if scaler.get_scale() * scaler.get_backoff_factor() >= 1.:
                            scaler.update(scaler.get_scale() * scaler.get_backoff_factor())
                        for g in scaled_grads: g.zero_()

                    if not meta_grads:
                        meta_grads = scaled_grads
                    else:
                        for m_g, g in zip(meta_grads, scaled_grads): m_g.add_(g)

                losses.append(sum_query_loss / len(batches))
                for g in meta_grads: g.div_(len(batches))
                meta_gradients.append(meta_grads)

            for weights in zip(model.parameters(), *meta_gradients):
                for k in range(len(meta_gradients)):
                    if k == 0:
                        weights[0].grad = weights[k + 1] / len(meta_gradients)
                    else:
                        weights[0].grad += weights[k + 1] / len(meta_gradients)

            # notice the above meta gradients are scaled
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            if scaler.get_scale() < 1.:
                scaler.update(1.)

            tqdm_iterations.set_description('Training query loss {:.4f}, cur acc {:.4f}, scale {:.0f}'.format(
                np.mean(losses), np.mean(task_accs), scaler.get_scale()))

        if (iteration + 1) % args.eval_span == 0:
            val_f1_micro = evaluation(args, model, scaler, inner_loop_optimizer, val_iter)
            # if val_f1_micro <= max_f1_micro:
            #     print('Validation f1 micro: {:.4f}'.format(val_f1_micro))
            # else:
            #     max_f1_micro = val_f1_micro
            #     print('Validation f1 micro: {:.4f}, saving best model...'.format(val_f1_micro))
            #     #model = PeftModel.save_pretrained(model=model, model_id=export_root ,safe_serialization=False)
            #     model.save_pretrained(export_root ,safe_serialization=False)
            #     tokenizer.save_pretrained(export_root)
            #     output_args_file = os.path.join(export_root, 'training_args.bin')
            #     torch.save(args, output_args_file)
        if (iteration + 1) % args.eval_span == 0:
            val_f1_micro, val_f1_macro = evaluation(args, model, scaler, inner_loop_optimizer, val_iter)
            if val_f1_micro <= max_f1_micro:    
                print('Validation f1 micro: {:.4f}, f1 macro: {:.4f}'.format(val_f1_micro, val_f1_macro))
            else:
                max_f1_micro = val_f1_micro
                max_f1_macro = val_f1_macro
                print('Validation f1 micro: {:.4f}, f1 macro: {:.4f}, saving best model...'.format(val_f1_micro, val_f1_macro))
                model.save_pretrained(export_root)
                tokenizer.save_pretrained(export_root)
                output_args_file = os.path.join(export_root, 'training_args.bin')
                torch.save(args, output_args_file)

    # for filename in os.listdir(export_root):
    #     if filename.startswith("adapter_"):
    #         old_file = os.path.join(export_root, filename)
    #         new_file = os.path.join(export_root, filename.replace("adapter_", "", 1))
    #         os.rename(old_file, new_file)
    #         print(f"Renamed {old_file} to {new_file}")


    model = PeftModel.from_pretrained(model=model, model_id=export_root, from_flax=True).to(args.device)


    test_f1_micro = evaluations(args, model, scaler, inner_loop_optimizer, test_iter)

    if args.del_model:
        print('Finished evaluation, deleting model...')
        for filename in os.listdir(export_root):
            os.remove(os.path.join(export_root, filename))

    # only save result file
    print('Result f1 micro: {:.4f}'.format(test_f1_micro))
    with open(os.path.join(export_root, 'test_metrics.json'), 'w') as f:
        json.dump([test_f1_micro], f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int, help='random seed for the experiments')
    parser.add_argument('--csed_data_dir', default='../data/csed', type=str, help='data path for CSED dataset')
    parser.add_argument('--csed_train_file', default='csed_train.json', type=str, help='train file for CSED dataset')
    parser.add_argument('--csed_dev_file', default='csed_dev.json', type=str, help='dev file for CSED dataset')
    parser.add_argument('--csed_test_file', default='csed_test.json', type=str, help='test file for CSED dataset')
    parser.add_argument('--experiment_dir', default='experiments', type=str, help='root for saving the experiments')
    parser.add_argument('--output_dir', default=None, type=str, help='path for saving the model state dict')
    parser.add_argument('--device', default='cuda', type=str, help='device for training and evaluation')
    parser.add_argument('--del_model', default=True, type=bool, help='whether to delete model after training')
    parser.add_argument('--template', default='A <mask> event </s></s> {}', type=str, help='prompt template for inference')
    parser.add_argument('--N', default=10, type=int, help='number of classes (both training and evaluation)')
    parser.add_argument('--K', default=10, type=int, help='number of support instances per class (i.e., shot)')
    parser.add_argument('--train_iterations', default=500, type=int, help='total number of training iterations to perform')
    parser.add_argument('--eval_span', default=25, type=int, help='iterations between each validation round')
    parser.add_argument('--num_task', default=3, type=int, help='number of tasks in each iteration')
    parser.add_argument('--max_batch', default=2, type=int, help='maximum number of batch for each forward pass')
    parser.add_argument('--max_batchsize', default=50, type=int, help='maximum batch size for each forward pass')
    parser.add_argument('--max_len', default=64, type=int, help='maximum input length for event detection')
    parser.add_argument('--lr_meta', default=1e-5, type=float, help='the initial meta learning rate for AdamW')
    parser.add_argument('--lr_decay_meta', default=0.1, type=float, help='the decay factor for cosine annealing')
    parser.add_argument('--weight_decay', default=0., type=float, help='the weight decay value for AdamW')
    parser.add_argument('--lr_learner', default=1e-3, type=float, help='the intial learner learning rate (Subtask)')
    parser.add_argument('--clf_lr_multiplier', default=10., type=float, help='the intial learner learning rate (Subtask)')
    parser.add_argument('--lr_of_lr', default=1e-4, type=float, help='the learning rate of learning rate (Subtask)')
    parser.add_argument('--task_updates', default=50, type=int, help='number of updates in the inner loop for training')
    parser.add_argument('--scaling_trigger', default=1., type=float, help='scaling factor for trigger span loss')
    parser.add_argument('--zero_shot', default=False, type=bool, help='whether to perform zero-shot training')
    parser.add_argument('--scaling_contrastive', default=1., type=float, help='scaling factor for contrastive loss')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='maximum gradient norm in updating model weights')

    args = parser.parse_args()
    train_fewshot(args)