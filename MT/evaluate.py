from sklearn.metrics import precision_recall_fscore_support
import torch
from typing import Dict, List
from utils import log_metrics

def evaluate(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, 
             mode: str = 'few_shot', device: str = 'cuda') -> Dict[str, float]:
    """
    Evaluate the Mesh Transformer model on trigger identification and classification.
    
    Args:
        model: MeshTransformer instance.
        data_loader: DataLoader for evaluation data.
        mode: 'few_shot' or 'zero_shot'.
        device: Device to run the model on.
    
    Returns:
        Dictionary with micro-averaged precision, recall, and F1-score.
    """
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    
    with torch.no_grad():
        for batch in data_loader:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            # Forward pass
            preds = model(
                None,  # char_ids not used with RoBERTa
                batch['tokens'],
                batch['word_positions'],
                batch['entity_type_ids'],
                batch['seq_len'],
                batch['event_type_embed'] if mode == 'zero_shot' else None
            )
            
            # Process predictions
            if mode == 'few_shot':
                # CRF decode returns a list of lists (per sequence)
                preds_flat = []
                labels_flat = []
                for seq_preds, seq_labels, seq_len in zip(preds, batch['label_ids'], batch['seq_len']):
                    preds_flat.extend(seq_preds[:seq_len.item()])
                    labels_flat.extend(seq_labels[:seq_len.item()].cpu().numpy())
                all_preds.extend(preds_flat)
                all_labels.extend(labels_flat)
            else:
                # Zero-shot: Similarity-based predictions
                preds = preds.argmax(dim=-1)  # [batch, seq_len]
                mask = (torch.arange(preds.size(1)).to(device) < 
                        batch['seq_len'].unsqueeze(1)).to(device)
                preds_flat = preds[mask].cpu().numpy()
                labels_flat = batch['label_ids'][mask].cpu().numpy()
                all_preds.extend(preds_flat)
                all_labels.extend(labels_flat)
    
    # Compute micro-averaged metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='micro', zero_division=0
    )
    
    metrics = {'precision': precision, 'recall': recall, 'f1': f1}
    log_metrics(metrics, log_file=f'evaluation_{mode}.log')
    
    return metrics