import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import random
import StanfordCoreNLP
from gensim.models import KeyedVectors
import numpy as np

class CSEDDataset(Dataset):
    def __init__(self, json_file: str, lexicon_file: str, char_embedding_file: str, 
                 word_embedding_file: str, entity_type_vocab: Dict[str, int], 
                 max_len: int = 128, k_shot: int = None, zero_shot_types: Set[str] = None, 
                 mode: str = 'train'):
        """
        Initialize the CSED dataset for few-shot or zero-shot learning.

        Args:
            json_file: Path to the JSON dataset file.
            lexicon_file: Path to the lexicon file.
            char_embedding_file: Path to pre-trained character embeddings.
            word_embedding_file: Path to pre-trained word embeddings.
            entity_type_vocab: Dictionary mapping entity types to IDs.
            max_len: Maximum sequence length.
            k_shot: Number of examples per event type for few-shot learning (None for full dataset).
            zero_shot_types: Set of event types to exclude from training (for zero-shot).
            mode: 'train', 'dev', or 'test' to determine data filtering.
        """
        self.data = self._load_json(json_file)
        self.lexicon = self._load_lexicon(lexicon_file)
        self.char_embeddings = KeyedVectors.load(char_embedding_file)
        self.word_embeddings = KeyedVectors.load(word_embedding_file)
        self.entity_type_vocab = entity_type_vocab
        self.max_len = max_len
        self.stnlp = StanfordCoreNLP('path_to_stanford_corenlp')
        self.k_shot = k_shot
        self.zero_shot_types = zero_shot_types or set()
        self.mode = mode
        self.label_vocab = self._build_label_vocab()
        
        # Filter data for few-shot or zero-shot
        self.data = self._filter_data()

    def _load_json(self, json_file: str) -> List[Dict]:
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_lexicon(self, lexicon_file: str) -> set:
        with open(lexicon_file, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f)
    
    def _build_label_vocab(self) -> Dict[str, int]:
        event_types = set(sample['eventype'] for sample in self.data)
        labels = ['O']
        for event in event_types:
            labels.extend([f'B_{event}', f'I_{event}', f'E_{event}', f'S_{event}'])
        return {label: idx for idx, label in enumerate(labels)}
    
    def _filter_data(self) -> List[Dict]:
        """Filter data for few-shot or zero-shot settings."""
        if self.mode == 'train' and (self.k_shot is not None or self.zero_shot_types):
            # Group data by event type
            event_groups = defaultdict(list)
            for sample in self.data:
                event_groups[sample['eventype']].append(sample)
            
            filtered_data = []
            for event_type, samples in event_groups.items():
                if event_type in self.zero_shot_types:
                    continue  # Skip zero-shot event types for training
                if self.k_shot is not None:
                    # Sample k examples per event type
                    random.shuffle(samples)
                    filtered_data.extend(samples[:self.k_shot])
                else:
                    filtered_data.extend(samples)
            return filtered_data
        elif self.mode == 'test' and self.zero_shot_types:
            # For zero-shot testing, include only zero-shot event types
            return [sample for sample in self.data if sample['eventype'] in self.zero_shot_types]
        return self.data
    
    def _get_entity_mentions(self, sentence: str) -> List[Tuple[str, str, Tuple[int, int]]]:
        # Same as original implementation
        annotations = self.stnlp.ner(sentence)
        entities = []
        current_entity = []
        current_type = None
        start_pos = None
        for i, (char, ner_tag) in enumerate(annotations):
            if ner_tag != 'O':
                if ner_tag.startswith('B-'):
                    if current_entity:
                        entities.append((''.join(current_entity), current_type, (start_pos, i-1)))
                        current_entity = []
                    current_entity.append(char)
                    current_type = ner_tag[2:]
                    start_pos = i
                elif ner_tag.startswith('I-'):
                    current_entity.append(char)
                elif ner_tag.startswith('E-'):
                    current_entity.append(char)
                    entities.append((''.join(current_entity), current_type, (start_pos, i)))
                    current_entity = []
                    current_type = None
            else:
                if current_entity:
                    entities.append((''.join(current_entity), current_type, (start_pos, i-1)))
                    current_entity = []
                    current_type = None
        return entities
    
    def _get_self_matched_words(self, sentence: str, tokens: List[str], entities: List[Tuple[str, str, Tuple[int, int]]]) -> Dict[int, List[Tuple[str, Tuple[int, int]]]]:
        # Same as original implementation
        self_matched_words = {i: [] for i in range(len(tokens))}
        for entity_text, _, (start_pos, end_pos) in entities:
            for i in range(start_pos, end_pos + 1):
                self_matched_words[i].append((entity_text, (start_pos, end_pos)))
        for i in range(len(tokens)):
            if not any(start_pos <= i <= end_pos for _, _, (start_pos, end_pos) in entities):
                for j in range(i, len(tokens)):
                    candidate = ''.join(tokens[i:j+1])
                    if candidate in self.lexicon:
                        self_matched_words[i].append((candidate, (i, j)))
        return self_matched_words
    
    def _get_bioes_labels(self, tokens: List[str], trigger_positions: List[int], event_type: str) -> List[str]:
        # Same as original implementation
        labels = ['O'] * len(tokens)
        start, end = trigger_positions
        if start == end:
            labels[start] = f'S_{event_type}'
        else:
            labels[start] = f'B_{event_type}'
            for i in range(start + 1, end):
                labels[i] = f'I_{event_type}'
            labels[end] = f'E_{event_type}'
        return labels
    
    def _get_event_type_embedding(self, event_type: str) -> np.ndarray:
        """Generate embedding for event type (for zero-shot)."""
        # Placeholder: Use pre-trained model or description-based embedding
        # For simplicity, return random embedding; replace with actual implementation
        return np.random.randn(20)  # 20-dim, consistent with entity type embedding
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        sentence = sample['sentence']
        tokens = sample['tokens']
        trigger_positions = sample['trigger positions']
        event_type = sample['eventype']
        
        entities = self._get_entity_mentions(sentence)
        self_matched_words = self._get_self_matched_words(sentence, tokens, entities)
        labels = self._get_bioes_labels(tokens, trigger_positions, event_type)
        
        char_ids = []
        char_embeds = []
        for token in tokens[:self.max_len]:
            char_ids.append(self.char_embeddings.key_to_index.get(token, 0))
            try:
                char_embeds.append(self.char_embeddings[token])
            except KeyError:
                char_embeds.append(np.zeros(200))
        
        word_ids = []
        word_embeds = []
        word_positions = []
        for i in range(len(tokens[:self.max_len])):
            for word, (start, end) in self_matched_words[i]:
                if start < self.max_len:
                    word_ids.append(self.word_embeddings.key_to_index.get(word, 0))
                    try:
                        word_embeds.append(self.word_embeddings[word])
                    except KeyError:
                        word_embeds.append(np.zeros(200))
                    word_positions.append((start, end))
        
        entity_type_ids = []
        entity_type_embeds = []
        for i in range(len(tokens[:self.max_len])):
            entity_type = 'O'
            for entity_text, entity_type_, (start, end) in entities:
                if start <= i <= end:
                    entity_type = entity_type_
                    break
            entity_type_ids.append(self.entity_type_vocab.get(entity_type, 0))
            entity_type_embeds.append(np.random.randn(20) if entity_type == 'O' else np.random.randn(20))
        
        label_ids = [self.label_vocab[label] for label in labels[:self.max_len]]
        
        # Add event type embedding for zero-shot
        event_type_embed = self._get_event_type_embedding(event_type) if self.mode == 'test' and event_type in self.zero_shot_types else np.zeros(20)
        
        seq_len = min(len(tokens), self.max_len)
        char_ids = char_ids + [0] * (self.max_len - seq_len)
        char_embeds = char_embeds + [np.zeros(200)] * (self.max_len - seq_len)
        entity_type_ids = entity_type_ids + [0] * (self.max_len - seq_len)
        entity_type_embeds = entity_type_embeds + [np.zeros(20)] * (self.max_len - seq_len)
        label_ids = label_ids + [self.label_vocab['O']] * (self.max_len - seq_len)
        
        # In CSEDDataset.__getitem__
        return {
            'tokens': tokens,  # Replace 'char_ids'
            'char_embeds': torch.tensor(char_embeds, dtype=torch.float),  # Optional, for fallback
            'word_ids': torch.tensor(word_ids, dtype=torch.long),
            'word_embeds': torch.tensor(word_embeds, dtype=torch.float),
            'word_positions': torch.tensor(word_positions, dtype=torch.long),
            'entity_type_ids': torch.tensor(entity_type_ids, dtype=torch.long),
            'entity_type_embeds': torch.tensor(entity_type_embeds, dtype=torch.float),
            'label_ids': torch.tensor(label_ids, dtype=torch.long),
            'seq_len': torch.tensor(seq_len, dtype=torch.long),
            'event_type_embed': torch.tensor(event_type_embed, dtype=torch.float)
        }
def get_dataloader(json_file: str, lexicon_file: str, char_embedding_file: str, 
                  word_embedding_file: str, entity_type_vocab: Dict[str, int], 
                  batch_size: int = 32, shuffle: bool = True, k_shot: int = None, 
                  zero_shot_types: Set[str] = None, mode: str = 'train') -> DataLoader:
    dataset = CSEDDataset(json_file, lexicon_file, char_embedding_file, word_embedding_file, 
                          entity_type_vocab, k_shot=k_shot, zero_shot_types=zero_shot_types, mode=mode)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)