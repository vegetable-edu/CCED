import torch
import json
from typing import List, Tuple, Dict

def load_lexicon(lexicon_path: str) -> List[str]:
    """
    Load the Chinese lexicon from a file.
    
    Args:
        lexicon_path (str): Path to the lexicon file.
    
    Returns:
        List[str]: List of words in the lexicon.
    """
    try:
        with open(lexicon_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Lexicon file {lexicon_path} not found. Returning empty lexicon.")
        return []

def compute_metrics(predictions: torch.Tensor, labels: torch.Tensor, 
                    num_event_types: int) -> Dict[str, float]:
    """
    Compute precision, recall, and F1 scores for event detection.
    
    Args:
        predictions (torch.Tensor): Predicted event type IDs for spans (batch_size, num_spans).
        labels (torch.Tensor): Ground-truth event type IDs for spans (batch_size, num_spans).
        num_event_types (int): Number of event types (including 'Other').
    
    Returns:
        Dict[str, float]: Dictionary with precision, recall, and F1 scores.
    """
    # Flatten predictions and labels
    preds = predictions.view(-1).cpu().numpy()
    true = labels.view(-1).cpu().numpy()
    
    # Compute metrics for non-'Other' (ID 0) predictions
    correct = 0
    pred_count = 0
    true_count = 0
    
    for pred, label in zip(preds, true):
        if pred != 0:  # Non-'Other' prediction
            pred_count += 1
            if pred == label:
                correct += 1
        if label != 0:  # Non-'Other' ground truth
            true_count += 1
    
    precision = correct / pred_count if pred_count > 0 else 0.0
    recall = correct / true_count if true_count > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def compute_zero_shot_metrics(predictions: torch.Tensor, labels: torch.Tensor, 
                              unseen_event_ids: List[int]) -> Dict[str, float]:
    """
    Compute metrics for zero-shot evaluation on unseen event types.
    
    Args:
        predictions (torch.Tensor): Predicted event type IDs for spans.
        labels (torch.Tensor): Ground-truth event type IDs for spans.
        unseen_event_ids (List[int]): IDs of unseen event types.
    
    Returns:
        Dict[str, float]: Metrics for unseen event types.
    """
    preds = predictions.view(-1).cpu().numpy()
    true = labels.view(-1).cpu().numpy()
    
    correct = 0
    pred_count = 0
    true_count = 0
    
    for pred, label in zip(preds, true):
        if label in unseen_event_ids:  # Ground truth is an unseen event type
            true_count += 1
            if pred == label:
                correct += 1
        if pred in unseen_event_ids:
            pred_count += 1
    
    precision = correct / pred_count if pred_count > 0 else 0.0
    recall = correct / true_count if true_count > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }