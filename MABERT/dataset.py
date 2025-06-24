import json
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import random
import uuid

class CSEDataset(Dataset):
    def __init__(self, json_path, bert_model_name='bert-base-chinese', lexicon_path=None, 
                 max_seq_length=300, max_span_length=4, num_shots=None, seen_event_types=None, 
                 seed=42):
        """
        Initialize the CSED dataset with few-shot or zero-shot settings.
        
        Args:
            json_path (str): Path to the JSON dataset file.
            bert_model_name (str): Name of the BERT model for tokenization.
            lexicon_path (str): Path to the Chinese lexicon file (optional).
            max_seq_length (int): Maximum sequence length for padding.
            max_span_length (int): Maximum span length for trigger classification.
            num_shots (int, optional): Number of samples per event type for few-shot learning.
            seen_event_types (list, optional): List of event types to include (for zero-shot).
            seed (int): Random seed for reproducibility in few-shot sampling.
        """
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.max_seq_length = max_seq_length
        self.max_span_length = max_span_length
        self.lexicon = self.load_lexicon(lexicon_path) if lexicon_path else []
        self.num_shots = num_shots
        self.seen_event_types = seen_event_types
        self.seed = seed
        
        # Load JSON data
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Filter data for few-shot or zero-shot
        self.data = self.filter_data()
        
        # Event type to ID mapping
        self.event_type_to_id = {item['eventype']: item['eventype id'] for item in self.data}
        self.num_event_types = len(set(item['eventype id'] for item in self.data)) + 1  # +1 for 'Other'

    def load_lexicon(self, lexicon_path):
        """
        Load the Chinese lexicon from a file.
        
        Args:
            lexicon_path (str): Path to the lexicon file.
        
        Returns:
            list: List of words in the lexicon.
        """
        with open(lexicon_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

    def filter_data(self):
        """
        Filter dataset for few-shot or zero-shot settings.
        
        Returns:
            list: Filtered dataset.
        """
        data = self.data
        
        # Zero-shot: Filter by seen event types
        if self.seen_event_types is not None:
            data = [item for item in data if item['eventype'] in self.seen_event_types]
        
        # Few-shot: Sample K shots per event type
        if self.num_shots is not None:
            random.seed(self.seed)
            event_type_groups = {}
            for item in data:
                event_type = item['eventype']
                if event_type not in event_type_groups:
                    event_type_groups[event_type] = []
                event_type_groups[event_type].append(item)
            
            filtered_data = []
            for event_type, items in event_type_groups.items():
                if len(items) > self.num_shots:
                    filtered_data.extend(random.sample(items, self.num_shots))
                else:
                    filtered_data.extend(items)
            data = filtered_data
        
        return data

    def preprocess_sequence(self, sentence, tokens):
        """
        Convert sentence to char-words sequence and prepare embeddings.
        
        Args:
            sentence (str): Input sentence.
            tokens (list): Pre-segmented tokens from the dataset.
        
        Returns:
            tuple: (X_cw, char_indices), where X_cw is the char-words sequence
                   with (token, head, tail) tuples, and char_indices maps characters
                   to their positions.
        """
        chars = list(sentence)
        char_indices = list(range(len(chars)))
        X_cw = [(char, i, i) for i, char in enumerate(chars)]
        for i in range(len(chars)):
            for word in self.lexicon:
                if ''.join(chars[i:i+len(word)]) == word:
                    X_cw.append((word, i, i + len(word) - 1))
        return X_cw, char_indices

    def generate_mask_matrices(self, X_cw, char_indices):
        """
        Generate matched-words and character-sequence mask matrices.
        
        Args:
            X_cw (list): Char-words sequence with (token, head, tail) tuples.
            char_indices (list): Indices of characters in the original sequence.
        
        Returns:
            tuple: (M_w, M_c), matched-words and character-sequence mask matrices.
        """
        n = len(char_indices)
        m = len(X_cw)
        M_w = torch.zeros(m, m)
        M_c = torch.ones(m, m)
        
        for i, (token_i, hp_i, tp_i) in enumerate(X_cw):
            for j, (token_j, hp_j, tp_j) in enumerate(X_cw):
                if i < n and hp_j <= hp_i <= tp_i <= tp_j:
                    M_w[i, j] = 1
                if i >= n and j < n and hp_i <= hp_j <= tp_j <= tp_i:
                    M_w[i, j] = 1
                if len(token_j) > 1:
                    M_c[i, j] = 0
        return M_w, M_c

    def generate_spans(self, sequence_length):
        """
        Generate all possible spans for trigger classification.
        
        Args:
            sequence_length (int): Length of the character sequence.
        
        Returns:
            list: List of (start, end) tuples for spans of length 1 to max_span_length.
        """
        spans = []
        for length in range(1, self.max_span_length + 1):
            for start in range(sequence_length - length + 1):
                end = start + length - 1
                spans.append((start, end))
        return spans

    def insert_type_markers(self, sequence, trigger_span, entity_spans, event_type):
        """
        Insert event and entity type markers into the sequence for EAE.
        
        Args:
            sequence (list): List of characters in the sentence.
           土地 trigger_span (tuple): (start, end) positions of the trigger.
            entity_spans (list): List of (start, end, entity_type) tuples.
            event_type (str): Event type (e.g., '漏洞发现').
        
        Returns:
            list: Modified sequence with type markers.
        """
        X_prime = list(sequence)
        X_prime.insert(trigger_span[0], f"<T:{event_type}>")
        X_prime.insert(trigger_span[1] + 1, f"</T:{event_type}>")
        offset = 2
        for i, (start, end, entity_type) in enumerate(entity_spans):
            X_prime.insert(start + offset, f"<E:{entity_type}>")
            X_prime.insert(end + offset + 1, f"</E:{entity_type}>")
            offset += 2
        return X_prime

    def __getitem__(self, idx):
        """
        Get a preprocessed data sample.
        
        Args:
            idx (int): Index of the data sample.
        
        Returns:
            dict: Dictionary containing input_ids, attention_mask, spans, labels,
                  mask matrices, and other metadata.
        """
        item = self.data[idx]
        sentence = item['sentence']
        tokens = item['tokens']
        trigger = item['trigger']
        trigger_positions = item['trigger_positions']
        event_type = item['eventype']
        event_type_id = item['eventype id']
        
        X_cw, char_indices = self.preprocess_sequence(sentence, tokens)
        M_w, M_c = self.generate_mask_matrices(X_cw, char_indices)
        
        input_tokens = [t[0] for t in X_cw]
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
        if len(input_ids) > self.max_seq_length:
            input_ids = input_ids[:self.max_seq_length]
            M_w = M_w[:self.max_seq_length, :self.max_seq_length]
            M_c = M_c[:self.max_seq_length, :self.max_seq_length]
        else:
            padding_length = self.max_seq_length - len(input_ids)
            input_ids += [self.tokenizer.pad_token_id] * padding_length
            M_w = torch.cat([M_w, torch.zeros(M_w.size(0), padding_length)], dim=1)
            M_w = torch.cat([M_w, torch.zeros(padding_length, M_w.size(1))], dim=0)
            M_c = torch.cat([M_c, torch.ones(M_c.size(0), padding_length)], dim=1)
            M_c = torch.cat([M_c, torch.ones(padding_length, M_c.size(1))], dim=0)
        
        attention_mask = [1] * len(input_tokens) + [0] * padding_length
        
        spans = self.generate_spans(len(char_indices))
        labels = torch.zeros(len(spans), dtype=torch.long)
        for i, (start, end) in enumerate(spans):
            if start == trigger_positions[0] and end == trigger_positions[1]:
                labels[i] = event_type_id
            else:
                labels[i] = 0
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'M_w': M_w,
            'M_c': M_c,
            'spans': spans,
            'labels': labels,
            'sentence': sentence,
            'trigger_positions': trigger_positions,
            'event_type': event_type
        }

    def __len__(self):
        """
        Return the size of the dataset.
        
        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data)