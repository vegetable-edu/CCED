import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional

def train_ed(model, dataloader: DataLoader, optimizer: torch.optim.Optimizer, 
             device: str) -> float:
    """
    Train the MABERT model for event detection.
    
    Args:
        model: The MABERT model instance.
        dataloader (DataLoader): DataLoader for the training dataset.
        optimizer: Optimizer for updating model parameters.
        device (str): Device to run the model on ('cuda' or 'cpu').
    
    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore 'Other' label (ID 0)
    
    for batch in dataloader:
        optimizer.zero_grad()
        
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
        
        # Forward pass
        outputs = []
        for seq, span_list in zip(sequences, spans):
            output = model(sequence=list(seq), spans=span_list)
            outputs.append(output)
        outputs = torch.stack(outputs)  # Shape: (batch_size, num_spans, num_event_types)
        
        # Compute loss
        loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0