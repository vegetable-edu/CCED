import argparse
import torch
from dataset import get_dataloader, CSEDDataset
from model import MeshTransformer
from train import train
from evaluate import evaluate
from config import configs
from utils import load_entity_type_vocab, load_event_type_vocab

def main():
    """
    Main script for training and evaluating the Mesh Transformer on the CSED dataset.
    Supports few-shot and zero-shot learning modes.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Mesh Transformer for Chinese Event Detection")
    parser.add_argument('--mode', choices=['few_shot', 'zero_shot'], default='few_shot',
                        help='Learning mode: few_shot or zero_shot')
    parser.add_argument('--k_shot', type=int, default=5,
                        help='Number of examples per event type for few-shot (e.g., 5)')
    parser.add_argument('--zero_shot_types', type=str, default='',
                        help='Comma-separated list of event types for zero-shot (e.g., 漏洞利用)')
    parser.add_argument('--config', type=str, default='few_shot_5',
                        choices=['few_shot_5', 'zero_shot'],
                        help='Configuration to use from config.py')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run the model on')
    args = parser.parse_args()

    # Load configuration
    config = configs[args.config]
    if args.k_shot is not None:
        config['k_shot'] = args.k_shot
    if args.zero_shot_types:
        config['zero_shot_types'] = set(args.zero_shot_types.split(','))

    # Load vocabularies
    entity_type_vocab = load_entity_type_vocab('entity_types.txt')  # Assumed file
    event_type_vocab = load_event_type_vocab('event_types.txt') if args.mode == 'zero_shot' else None

    # Initialize data loaders
    train_loader = get_dataloader(
        json_file='csed.json',
        lexicon_file='lexicon.txt',
        char_embedding_file='char_embeds.bin',  # Optional, for fallback
        word_embedding_file='word_embeds.bin',  # Optional, for fallback
        entity_type_vocab=entity_type_vocab,
        batch_size=config['batch_size'],
        shuffle=True,
        k_shot=config['k_shot'],
        zero_shot_types=config['zero_shot_types'],
        mode='train'
    )

    dev_loader = get_dataloader(
        json_file='csed.json',
        lexicon_file='lexicon.txt',
        char_embedding_file='char_embeds.bin',
        word_embedding_file='word_embeds.bin',
        entity_type_vocab=entity_type_vocab,
        batch_size=config['batch_size'],
        shuffle=False,
        mode='dev'
    )

    test_loader = get_dataloader(
        json_file='csed.json',
        lexicon_file='lexicon.txt',
        char_embedding_file='char_embeds.bin',
        word_embedding_file='word_embeds.bin',
        entity_type_vocab=entity_type_vocab,
        batch_size=config['batch_size'],
        shuffle=False,
        zero_shot_types=config['zero_shot_types'],
        mode='test'
    )

    # Get label vocabulary from training dataset
    train_dataset = train_loader.dataset
    label_vocab = train_dataset.label_vocab

    # Initialize model
    model = MeshTransformer(
        entity_type_vocab_size=len(entity_type_vocab),
        num_labels=len(label_vocab) if args.mode == 'few_shot' else None,
        event_type_vocab=event_type_vocab,
        roberta_model_name=config['roberta_model_name'],
        char_embed_dim=config['char_embed_dim'],
        entity_embed_dim=config['entity_embed_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(args.device)

    # Move RoBERTa to device
    model.roberta.to(args.device)

    # Train model
    train(model, train_loader, dev_loader, config, device=args.device)

    # Evaluate model
    metrics = evaluate(model, test_loader, mode=args.mode, device=args.device)
    print(f"Test Metrics: Precision={metrics['precision']:.4f}, "
          f"Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")

if __name__ == "__main__":
    main()