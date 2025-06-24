configs = {
    'few_shot_5': {
        'k_shot': 5,
        'zero_shot_types': set(),
        'roberta_model_name': 'hfl/chinese-roberta-wwm-ext',
        'char_embed_dim': 768,  # RoBERTa output dimension
        'entity_embed_dim': 20,  # As per Section 4.1
        'learning_rate': 2e-5,  # As per Section 4.1
        'batch_size': 16,  # Smaller for few-shot
        'epochs': 50,  # More epochs for few-shot
        'dropout_input': 0.5,  # As per Section 4.1
        'dropout': 0.2,  # Encoder output, Section 4.1
        'num_heads': 10,  # As per Section 4.1
        'num_layers': 4,  # As per Section 4.1
        'max_len': 128,  # Maximum sequence length
        'gradient_accumulation_steps': 1  # Optional for small GPUs
    },
    'zero_shot': {
        'k_shot': None,
        'zero_shot_types': {'漏洞利用'},  # Example event type
        'roberta_model_name': 'hfl/chinese-roberta-wwm-ext',
        'char_embed_dim': 768,
        'entity_embed_dim': 20,
        'learning_rate': 2e-5,
        'batch_size': 32,
        'epochs': 20,
        'dropout_input': 0.5,
        'dropout': 0.2,
        'num_heads': 10,
        'num_layers': 4,
        'max_len': 128,
        'gradient_accumulation_steps': 1
    }
}