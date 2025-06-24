class Config:
    # General model settings
    bert_model_name = 'bert-base-chinese'
    lexicon_path = 'chinese_lexicon.txt'  # Path to your Chinese lexicon
    max_seq_length = 300
    max_span_length = 4
    num_transformer_layers = 12
    k = 3  # Number of layers with matched-words mask-attention
    learning_rate = 2e-5
    batch_size = 16
    num_epochs = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Dataset paths
    train_json_path = 'csed_train.json'
    dev_json_path = 'csed_dev.json'
    test_json_path = 'csed_test.json'
    
    # All event types
    event_types = [
        '数据泄露', '网络钓鱼', '赎金', 'DDoS攻击', '恶意软件',
        '供应链', '漏洞影响', '漏洞发现', '漏洞补丁'
    ]
    
    # Few-shot settings
    num_shots = 5  # Options: 1, 5, 10, etc.
    seed = 42  # Random seed for reproducibility
    
    # Zero-shot settings
    seen_event_types = [
        '数据泄露', '网络钓鱼', '赎金', 'DDoS攻击', '恶意软件'
    ]  # Event types for training
    unseen_event_types = [
        '供应链', '漏洞影响', '漏洞发现', '漏洞补丁'
    ]  # Event types for testing