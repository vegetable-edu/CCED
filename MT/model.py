import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import AutoModel, AutoTokenizer
import numpy as np

class MeshTransformer(nn.Module):
    def __init__(self, entity_type_vocab_size: int, num_labels: int = None, 
                 event_type_vocab: Dict[str, int] = None, roberta_model_name: str = 'hfl/chinese-roberta-wwm-ext',
                 char_embed_dim: int = 768, entity_embed_dim: int = 20, num_heads: int = 10, 
                 num_layers: int = 4, dropout: float = 0.2):
        """
        Mesh Transformer model for Chinese Event Detection with RoBERTa-wwm-ext.

        Args:
            entity_type_vocab_size: Size of entity type vocabulary.
            num_labels: Number of BIOES labels (for few-shot); None for zero-shot.
            event_type_vocab: Dictionary of event types to IDs (for zero-shot).
            roberta_model_name: Hugging Face model name (e.g., 'hfl/chinese-roberta-wwm-ext').
            char_embed_dim: Dimension of RoBERTa embeddings (default 768).
            entity_embed_dim: Dimension of entity type embeddings (20, as per paper).
            num_heads: Number of attention heads (10, as per paper).
            num_layers: Number of transformer layers (4, as per paper).
            dropout: Dropout rate for transformer layers (0.2, as per paper).
        """
        super(MeshTransformer, self).__init__()
        
        # Load RoBERTa model and tokenizer
        self.roberta = AutoModel.from_pretrained(roberta_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(roberta_model_name)
        
        # Entity type embedding
        self.entity_type_embedding = nn.Embedding(entity_type_vocab_size, entity_embed_dim)
        
        # Event type embedding for zero-shot
        self.event_type_vocab = event_type_vocab
        if event_type_vocab:
            self.event_type_embedding = nn.Embedding(len(event_type_vocab), entity_embed_dim)
        
        # Projection layer to adjust RoBERTa output dimension if needed
        self.projection = nn.Linear(char_embed_dim, char_embed_dim)
        
        # Mesh Transformer layers
        self.transformer_layers = nn.ModuleList([
            MeshTransformerLayer(char_embed_dim + entity_embed_dim, 
                                char_embed_dim + entity_embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Relay node
        self.relay_node = nn.Parameter(torch.zeros(char_embed_dim + entity_embed_dim))
        
        # Decoder
        self.num_labels = num_labels
        if num_labels:
            # Few-shot: CRF decoder
            self.linear = nn.Linear(char_embed_dim + entity_embed_dim, num_labels)
            self.crf = CRF(num_labels, batch_first=True)
        else:
            # Zero-shot: Similarity-based classifier
            self.linear = nn.Linear(char_embed_dim + entity_embed_dim, entity_embed_dim)
        
        self.dropout = nn.Dropout(dropout)

    def get_roberta_embeddings(self, tokens: List[str], word_positions: List[Tuple[int, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get RoBERTa embeddings for characters and words.

        Args:
            tokens: List of character tokens.
            word_positions: List of (start, end) positions for words.

        Returns:
            char_embeds: [batch, seq_len, char_embed_dim]
            word_embeds: [batch, num_words, char_embed_dim]
        """
        # Tokenize input sentence (convert tokens to RoBERTa input format)
        input_text = ''.join(tokens)
        inputs = self.tokenizer(input_text, return_tensors='pt', add_special_tokens=False, 
                               truncation=True, max_length=len(tokens) + 2)
        input_ids = inputs['input_ids'].to(self.roberta.device)
        
        # Get RoBERTa embeddings
        with torch.no_grad():
            outputs = self.roberta(input_ids)
        char_embeds = outputs.last_hidden_state[:, 1:-1, :]  # Remove [CLS] and [SEP]
        
        # Aggregate word embeddings by averaging character embeddings
        word_embeds = []
        for start, end in word_positions:
            word_embed = char_embeds[:, start:end + 1, :].mean(dim=1)  # Average over characters
            word_embeds.append(word_embed)
        word_embeds = torch.stack(word_embeds, dim=1) if word_embeds else torch.zeros_like(char_embeds)
        
        return char_embeds, word_embeds

    def forward(self, char_ids: torch.Tensor, tokens: List[str], word_positions: torch.Tensor, 
                entity_type_ids: torch.Tensor, seq_len: torch.Tensor, 
                event_type_embed: torch.Tensor = None):
        """
        Forward pass of the Mesh Transformer.

        Args:
            char_ids: [batch, seq_len] (not used with RoBERTa, kept for compatibility).
            tokens: List of character tokens for RoBERTa input.
            word_positions: [batch, num_words, 2] (start, end positions).
            entity_type_ids: [batch, seq_len] entity type IDs.
            seq_len: [batch] sequence lengths.
            event_type_embed: [num_event_types, entity_embed_dim] for zero-shot.

        Returns:
            Decoded labels (few-shot) or similarity scores (zero-shot).
        """
        # Get RoBERTa embeddings
        char_embeds, word_embeds = self.get_roberta_embeddings(tokens, word_positions)
        
        # Get entity type embeddings
        entity_type_embeds = self.entity_type_embedding(entity_type_ids)
        
        # Combine character and entity embeddings
        char_inputs = self.projection(char_embeds) + entity_type_embeds
        
        # Initialize relay node
        batch_size, seq_len, embed_dim = char_inputs.size()
        relay_inputs = self.relay_node.unsqueeze(0).expand(batch_size, -1)
        
        # Mesh Transformer Encoder
        for layer in self.transformer_layers:
            char_inputs, word_embeds, relay_inputs = layer(char_inputs, word_embeds, relay_inputs, 
                                                         word_positions, seq_len)
        
        # Apply dropout
        char_outputs = self.dropout(char_inputs)
        
        # Decoder
        if self.num_labels:
            # Few-shot: CRF decoder
            emissions = self.linear(char_outputs)
            mask = (torch.arange(seq_len.max()).to(emissions.device) < seq_len.unsqueeze(1)).to(emissions.device)
            return self.crf.decode(emissions, mask=mask)
        else:
            # Zero-shot: Similarity-based classifier
            char_outputs = self.linear(char_outputs)  # [batch, seq_len, entity_embed_dim]
            similarities = torch.matmul(char_outputs, event_type_embed.transpose(0, 1))  # [batch, seq_len, num_event_types]
            return similarities.argmax(dim=-1)  # Predict event type with highest similarity

class MeshTransformerLayer(nn.Module):
    def __init__(self, char_dim: int, word_dim: int, num_heads: int, dropout: float):
        super(MeshTransformerLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(char_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(char_dim, char_dim * 4),
            nn.ReLU(),
            nn.Linear(char_dim * 4, char_dim)
        )
        self.norm1 = nn.LayerNorm(char_dim)
        self.norm2 = nn.LayerNorm(char_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, char_inputs: torch.Tensor, word_inputs: torch.Tensor, 
                relay_inputs: torch.Tensor, word_positions: torch.Tensor, seq_len: torch.Tensor):
        """
        Forward pass of a Mesh Transformer layer.

        Args:
            char_inputs: [batch, seq_len, char_dim]
            word_inputs: [batch, num_words, word_dim]
            relay_inputs: [batch, char_dim]
            word_positions: [batch, num_words, 2]
            seq_len: [batch]

        Returns:
            Updated char_inputs, word_inputs, relay_inputs
        """
        # Character node update
        batch_size, seq_len, char_dim = char_inputs.size()
        char_outputs = []
        for i in range(seq_len):
            # Adjacent characters
            adj_chars = []
            if i > 0:
                adj_chars.append(char_inputs[:, i-1])
            adj_chars.append(char_inputs[:, i])
            if i < seq_len - 1:
                adj_chars.append(char_inputs[:, i+1])
            adj_chars = torch.stack(adj_chars, dim=1) if adj_chars else char_inputs[:, i:i+1]
            
            # Self-matched words
            word_mask = (word_positions[:, :, 0] <= i) & (word_positions[:, :, 1] >= i)
            matched_words = word_inputs * word_mask.unsqueeze(-1).float()
            
            # Relay node
            inputs = torch.cat([adj_chars, matched_words, relay_inputs.unsqueeze(1)], dim=1)
            char_output = self.multi_head_attention(char_inputs[:, i], inputs)
            char_output = self.norm1(char_inputs[:, i] + self.dropout(char_output))
            char_output = self.norm2(char_output + self.dropout(self.feed_forward(char_output)))
            char_outputs.append(char_output)
        
        char_inputs = torch.stack(char_outputs, dim=1)
        
        # Word node update
        word_outputs = []
        for b in range(batch_size):
            for start, end in word_positions[b]:
                if start < seq_len[b]:
                    composing_chars = char_inputs[b, start:end+1]
                    inputs = torch.cat([composing_chars, word_inputs[b, start:end+1], relay_inputs[b:b+1]], dim=0)
                    word_output = self.multi_head_attention(word_inputs[b, start:end+1].mean(dim=0), inputs)
                    word_output = self.norm1(word_inputs[b, start:end+1].mean(dim=0) + self.dropout(word_output))
                    word_output = self.norm2(word_output + self.dropout(self.feed_forward(word_output)))
                    word_outputs.append(word_output)
        word_inputs = torch.stack(word_outputs, dim=0).view(batch_size, -1, char_dim) if word_outputs else word_inputs
        
        # Relay node update
        all_inputs = torch.cat([char_inputs, word_inputs, relay_inputs.unsqueeze(1)], dim=1)
        relay_output = self.multi_head_attention(relay_inputs, all_inputs)
        relay_inputs = self.norm1(relay_inputs + self.dropout(relay_output))
        relay_inputs = self.norm2(relay_inputs + self.dropout(self.feed_forward(relay_inputs)))
        
        return char_inputs, word_inputs, relay_inputs

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        
        # Relative position encoding parameters
        self.rel_pos_weight = nn.Linear(4 * self.head_dim, self.head_dim)
        self.u = nn.Parameter(torch.randn(num_heads, self.head_dim))
        self.v = nn.Parameter(torch.randn(num_heads, self.head_dim))

    def get_relative_position_encoding(self, word_positions: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Compute relative position encodings (Section 3.2.2).

        Args:
            word_positions: [batch, num_words, 2]
            seq_len: Maximum sequence length.

        Returns:
            Relative position encodings.
        """
        # Placeholder: Implement as per Section 3.2.2
        # Compute d_{b_k,e_j}, d_{b_k,b_j}, d_{e_k,b_j}, d_{e_k,e_j} and apply sinusoids
        return torch.zeros(word_positions.size(0), word_positions.size(1), 4 * self.head_dim)

    def forward(self, query: torch.Tensor, keys: torch.Tensor, word_positions: torch.Tensor = None):
        batch_size, seq_len, embed_dim = keys.size()
        q = self.query(query).view(-1, self.num_heads, self.head_dim)
        k = self.key(keys).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.value(keys).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Attention scores
        scores = torch.einsum('bnh,bsnh->bns', q, k) / (self.head_dim ** 0.5)
        
        # Add relative position encoding if word_positions provided
        if word_positions is not None:
            rel_pos = self.get_relative_position_encoding(word_positions, seq_len)
            rel_scores = torch.einsum('bnh,bnsd->bns', q, rel_pos)
            scores = scores + rel_scores
        
        # Add u and v terms (Section 3.2.2)
        u_scores = torch.einsum('nh,bsnh->bns', self.u, k)
        v_scores = torch.einsum('nh,bnsd->bns', self.v, rel_pos) if word_positions is not None else 0
        scores = scores + u_scores + v_scores
        
        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.einsum('bns,bsnh->bnh', attn_weights, v).view(-1, embed_dim)
        return self.out(output)