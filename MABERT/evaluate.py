import torch
from torch.utils.data import DataLoader
from utils import compute_metrics, compute_zero_shot_metrics
from typing import Dict, List

def evaluate(model, dataloader: DataLoader, device: str, 
             unseen_event_ids: List[int] = None) -> Dict[str, float]:
    """
    Evaluate the MABERT model on a dataset.
    
    Args:
        model: The MABERT model instance.
        dataloader (DataLoader): DataLoader for the evaluation dataset.
        device (str): Device to run the model on ('cuda' or 'cpu').
        unseen_event_ids (List[int], optional): IDs of unseen event types for zero-shot evaluation.
    
    Returns:
        Dict[str, float]: Evaluation metrics (precision, recall, f1).
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move inputs to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            M_w = batch['M_w'].to(device)
            M_c = batch['M_c'].to(device)
            spans = batch['spans']  # List of spans
            labels = batch['labels'].to(device)
            
            # Prepare sequence for model
            sequences = [''.join(dataloader.dataset[idx]['sentence']) 
                        for idx in batch['indices']]  # Reconstruct sentences
            
            # Forward pass for trigger classification
            outputs = []
            for seq, span_list in zip(sequences, spans):
                output = model(sequence=list(seq), spans=span_list)
                outputs.append(output)
            outputs = torch.stack(outputs)
            
            # Get predictions
            predictions = torch.argmax(outputs, dim=-1)  # Shape: (batch_size, num_spans)
            
            all_predictions.append(predictions)
            all_labels.append(labels)
    
    # Concatenate all predictions and labels
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Compute metrics
    if unseen_event_ids:
        # Zero-shot evaluation
        metrics = compute_zero_shot_metrics(all_predictions, all_labels, unseen_event_ids)
        metrics['type'] = 'zero_shot'
    else:
        # Few-shot or standard evaluation
        metrics = compute_metrics(all_predictions, all_labels, 
                                 num_event_types=dataloader.dataset.num_event_types)
        metrics['type'] = 'few_shot'
    
    return metrics