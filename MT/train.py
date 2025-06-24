import torch
from torch.optim import Adam
import torch.nn.functional as F
from utils import save_checkpoint, log_metrics

def train(model, train_loader, dev_loader, config, device='cuda'):
    """
    Train the Mesh Transformer model.

    Args:
        model: MeshTransformer instance.
        train_loader: DataLoader for training data.
        dev_loader: DataLoader for development data.
        config: Configuration dictionary.
        device: Device to run the model on.
    """
    # Initialize optimizer
    optimizer = Adam(model.parameters(), lr=config['learning_rate'])

    # Move model to device
    model.to(device)

    best_dev_f1 = 0.0
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()

            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}

            # Forward pass
            if model.num_labels:
                # Few-shot: Compute CRF emissions
                char_embeds, word_embeds = model.get_roberta_embeddings(
                    batch['tokens'], batch['word_positions'])
                entity_type_embeds = model.entity_type_embedding(batch['entity_type_ids'])
                char_inputs = model.projection(char_embeds) + entity_type_embeds
                
                # Run through transformer layers (simplified)
                for layer in model.transformer_layers:
                    char_inputs, word_embeds, relay_inputs = layer(
                        char_inputs, word_embeds, model.relay_node.expand(char_inputs.size(0), -1),
                        batch['word_positions'], batch['seq_len'])
                
                emissions = model.linear(model.dropout(char_inputs))
                mask = (torch.arange(emissions.size(1)).to(device) < 
                        batch['seq_len'].unsqueeze(1)).to(device)
                loss = -model.crf(emissions, batch['label_ids'], mask=mask)
            else:
                # Zero-shot: Compute similarity scores
                outputs = model(
                    None,  # char_ids not used
                    batch['tokens'],
                    batch['word_positions'],
                    batch['entity_type_ids'],
                    batch['seq_len'],
                    batch['event_type_embed']
                )
                loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), 
                                      batch['label_ids'].view(-1))

            # Backward pass
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        
        # Evaluate on dev set
        if dev_loader:
            dev_metrics = evaluate(model, dev_loader, mode='few_shot' if model.num_labels else 'zero_shot', 
                                  device=device)
            dev_f1 = dev_metrics['f1']
            print(f"Epoch {epoch+1}/{config['epochs']}, Loss: {avg_loss:.4f}, "
                  f"Dev F1: {dev_f1:.4f}")
            
            # Save best model
            if dev_f1 > best_dev_f1:
                best_dev_f1 = dev_f1
                save_checkpoint(model, optimizer, epoch, dev_f1, 'best_model.pt')
        
        else:
            print(f"Epoch {epoch+1}/{config['epochs']}, Loss: {avg_loss:.4f}")

        # Log metrics
        log_metrics({'epoch': epoch+1, 'train_loss': avg_loss, 'dev_f1': dev_f1 if dev_loader else None})

    # Save final model
    save_checkpoint(model, optimizer, config['epochs'], best_dev_f1, 'final_model.pt')