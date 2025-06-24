import torch
from torch.utils.data import DataLoader
from dataset import CSEDataset
from model import MABERT
from train import train_ed
from evaluate import evaluate
from config import Config

def main():
    config = Config()
    
    # Initialize datasets
    train_dataset = CSEDataset(
        json_path=config.train_json_path,
        bert_model_name=config.bert_model_name,
        lexicon_path=config.lexicon_path,
        max_seq_length=config.max_seq_length,
        max_span_length=config.max_span_length,
        num_shots=config.num_shots,  # Few-shot setting
        seen_event_types=config.seen_event_types,  # Zero-shot setting
        seed=config.seed
    )
    
    dev_dataset = CSEDataset(
        json_path=config.dev_json_path,
        bert_model_name=config.bert_model_name,
        lexicon_path=config.lexicon_path,
        max_seq_length=config.max_seq_length,
        max_span_length=config.max_span_length
    )
    
    test_dataset = CSEDataset(
        json_path=config.test_json_path,
        bert_model_name=config.bert_model_name,
        lexicon_path=config.lexicon_path,
        max_seq_length=config.max_seq_length,
        max_span_length=config.max_span_length
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Initialize model
    model = MABERT(
        bert_model_name=config.bert_model_name,
        lexicon_size=len(train_dataset.lexicon),
        num_event_types=train_dataset.num_event_types,
        num_roles=36,  # Adjust based on your dataset
        k=config.k
    ).to(config.device)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Training loop
    for epoch in range(config.num_epochs):
        train_loss = train_ed(model, train_loader, optimizer, config.device)
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}")
        
        # Evaluate on dev set
        dev_metrics = evaluate(model, dev_loader, config.device)
        print(f"Dev Metrics: {dev_metrics}")
    
    # Evaluate on test set (includes unseen event types for zero-shot)
    test_metrics = evaluate(model, test_loader, config.device)
    print(f"Test Metrics: {test_metrics}")

if __name__ == "__main__":
    main()