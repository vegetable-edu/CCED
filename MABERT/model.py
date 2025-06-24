import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import uuid

class MABERT(nn.Module):
    def __init__(self, bert_model_name, lexicon_size, num_event_types, num_roles, k=3):
        super(MABERT, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.hidden_size = self.bert.config.hidden_size
        self.k = k  # Number of layers with matched-words mask-attention
        self.num_layers = 12  # Total transformer layers
        self.word_embedding = nn.Embedding(lexicon_size, self.hidden_size)
        self.W1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.W2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.trigger_classifier = nn.Linear(2 * self.hidden_size + 50, num_event_types)
        self.argument_classifier = nn.Linear(2 * self.hidden_size, num_roles)

    def embed_char_words(self, sequence, lexicon):
        # Match words and create char-words sequence
        tokens, heads, tails = [], [], []
        for i, char in enumerate(sequence):
            tokens.append(char)
            heads.append(i)
            tails.append(i)
            for word in lexicon:
                if sequence[i:i+len(word)] == word:
                    tokens.append(word)
                    heads.append(i)
                    tails.append(i + len(word) - 1)
        # Compute embeddings
        embeddings = []
        for token, head, tail in zip(tokens, heads, tails):
            if len(token) == 1:  # Character
                token_emb = self.bert.embeddings.word_embeddings(torch.tensor([self.bert_tokenizer.convert_tokens_to_ids(token)]))
            else:  # Word
                word_emb = self.word_embedding(torch.tensor([lexicon.index(token)]))
                token_emb = self.W2(self.relu(self.W1(word_emb)))
            pos_emb = self.bert.embeddings.position_embeddings(torch.tensor([head, tail])).sum(dim=0)
            seg_emb = self.bert.embeddings.token_type_embeddings(torch.tensor([0]))
            embeddings.append(token_emb + pos_emb + seg_emb)
        return torch.stack(embeddings)

    def generate_mask_matrices(self, X_cw):
        n = len([t for t in X_cw if len(t[0]) == 1])  # Number of characters
        m = len(X_cw)  # Total tokens
        M_w = torch.zeros(m, m)
        M_c = torch.ones(m, m)
        for i, (token_i, hp_i, tp_i) in enumerate(X_cw):
            for j, (token_j, hp_j, tp_j) in enumerate(X_cw):
                if i <= n and hp_j <= hp_i <= tp_i <= tp_j:
                    M_w[i, j] = 1
                if i > n and j <= n and hp_i <= hp_j <= tp_j <= tp_i:
                    M_w[i, j] = 1
                if len(token_j) > 1:
                    M_c[i, j] = 0
        return M_w, M_c

    def mask_attention_transformer(self, H, M):
        # Simplified mask-attention implementation
        Q = H @ self.bert.encoder.layer[0].attention.self.query.weight
        K = H @ self.bert.encoder.layer[0].attention.self.key.weight
        V = H @ self.bert.encoder.layer[0].attention.self.value.weight
        scores = (Q @ K.transpose(-2, -1)) / (self.hidden_size ** 0.5)
        scores = scores.masked_fill(M == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        return attn @ V

    def forward(self, sequence, spans=None, trigger_span=None, entity_spans=None, event_type=None):
        # Embed char-words sequence
        X_cw = [(token, head, tail) for token, head, tail in self.embed_char_words(sequence, lexicon)]
        embeddings = self.embed_char_words(sequence, lexicon)
        
        # Generate mask matrices
        M_w, M_c = self.generate_mask_matrices(X_cw)
        
        # Transformer layers
        H = embeddings
        for i in range(self.num_layers):
            M = M_w if i < self.k else M_c
            H = self.mask_attention_transformer(H, M)
            H = self.bert.encoder.layer[i].output.dense(H)
        
        # Trigger classification
        if spans is not None:
            span_reps = []
            for s in spans:
                start, end = s
                span_rep = torch.cat([H[start], H[end], torch.zeros(50)])  # Simplified span length embedding
                span_reps.append(span_rep)
            span_reps = torch.stack(span_reps)
            trigger_scores = self.trigger_classifier(span_reps)
            return trigger_scores
        
        # Argument classification
        if trigger_span and entity_spans and event_type:
            X_prime = self.insert_type_markers(sequence, trigger_span, entity_spans, event_type)
            H_prime = self.embed_char_words(X_prime, lexicon)
            for i in range(self.num_layers):
                M = M_w if i < self.k else M_c
                H_prime = self.mask_attention_transformer(H_prime, M)
                H_prime = self.bert.encoder.layer[i].output.dense(H_prime)
            trigger_rep = torch.max(H_prime[trigger_span[0]-1:trigger_span[1]+1], dim=0)[0]
            arg_reps = [torch.max(H_prime[s[0]-1:s[1]+1], dim=0)[0] for s in entity_spans]
            pair_reps = [torch.cat([trigger_rep, arg_rep]) for arg_rep in arg_reps]
            pair_reps = torch.stack(pair_reps)
            role_scores = self.argument_classifier(pair_reps)
            M_r = self.create_event_schema_mask([event_type], num_roles=36)
            role_scores = role_scores * M_r
            return role_scores

    def insert_type_markers(self, sequence, trigger_span, entity_spans, event_type):
        # Simplified type marker insertion
        X_prime = list(sequence)
        X_prime.insert(trigger_span[0], f"<T:{event_type}>")
        X_prime.insert(trigger_span[1]+1, f"</T:{event_type}>")
        for i, (start, end, entity_type) in enumerate(entity_spans):
            X_prime.insert(start + 2*i, f"<E:{entity_type}>")
            X_prime.insert(end + 2*i + 1, f"</E:{entity_type}>")
        return X_prime

    def create_event_schema_mask(self, event_types, num_roles):
        # Simplified event schema mask
        M_r = torch.zeros(len(event_types), num_roles)
        for i, et in enumerate(event_types):
            valid_roles = EVENT_SCHEMA.get(et, [])
            for role in valid_roles:
                M_r[i, role_idx[role]] = 1
        return M_r