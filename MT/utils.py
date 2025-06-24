import torch
import os
from typing import Dict, Any
import logging
from transformers import AutoTokenizer, AutoModel
import numpy as np
from tqdm import tqdm

def load_entity_type_vocab(file_path: str) -> Dict[str, int]:
    """
    Load entity type vocabulary from a text file.
    
    Args:
        file_path: Path to entity type vocabulary file (one type per line).
    
    Returns:
        Dictionary mapping entity types to IDs.
    """
    if not os.path.exists(file_path):
        return {'O': 0}  # Default vocabulary with 'O' for non-entities
    with open(file_path, 'r', encoding='utf-8') as f:
        return {line.strip(): i for i, line in enumerate(f) if line.strip()}

def load_event_type_vocab(file_path: str) -> Dict[str, int]:
    """
    Load event type vocabulary from a text file.
    
    Args:
        file_path: Path to event type vocabulary file (one type per line).
    
    Returns:
        Dictionary mapping event types to IDs.
    """
    if not os.path.exists(file_path):
        return {}  # Empty if no file provided
    with open(file_path, 'r', encoding='utf-8') as f:
        return {line.strip(): i for i, line in enumerate(f) if line.strip()}

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                    epoch: int, f1: float, filename: str):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model.
        optimizer: PyTorch optimizer.
        epoch: Current epoch number.
        f1: Current F1-score.
        filename: Path to save checkpoint.
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'f1': f1
    }, filename)
    logging.info(f"Saved checkpoint to {filename}")

def log_metrics(metrics: Dict[str, Any], log_file: str = 'training.log'):
    """
    Log training metrics to file and console.
    
    Args:
        metrics: Dictionary of metrics (e.g., {'epoch': 1, 'train_loss': 0.5, 'dev_f1': 0.8}).
        log_file: Path to log file.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.info(metrics)

def get_event_type_embedding(event_type: str, tokenizer: AutoTokenizer, 
                            model: AutoModel, device: str = 'cuda') -> np.ndarray:
    """
    Generate embedding for an event type using RoBERTa.
    
    Args:
        event_type: Event type string (e.g., '漏洞利用').
        tokenizer: Hugging Face tokenizer.
        model: Hugging Face model (e.g., RoBERTa).
        device: Device to run the model on.
    
    Returns:
        NumPy array of event type embedding (768-dim).
    """
    inputs = tokenizer(event_type, return_tensors='pt', padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()  # [CLS] embedding

def augment_data(sample: Dict, lexicon: set, max_replacements: int = 2) -> Dict:
    """
    Augment a data sample by replacing tokens with synonyms from the lexicon.
    
    Args:
        sample: Data sample dictionary with 'sentence' and 'tokens'.
        lexicon: Set of words for synonym replacement.
        max_replacements: Maximum number of tokens to replace.
    
    Returns:
        Augmented sample dictionary.
    """
    import random
    augmented = sample.copy()
    tokens = augmented['tokens']
    new_tokens = tokens.copy()
    indices = list(range(len(tokens)))
    random.shuffle(indices)
    replacements = 0
    
    for idx in indices:
        if replacements >= max_replacements:
            break
        token = tokens[idx]
        # Simple augmentation: Replace token with a random lexicon word
        # In practice, use a synonym dictionary or embeddings for better matches
        candidate = random.choice(list(lexicon))
        if len(candidate) == 1:  # Ensure single-character replacement
            new_tokens[idx] = candidate
            replacements += 1
    
    augmented['tokens'] = new_tokens
    augmented['sentence'] = ''.join(new_tokens)
    return augmented