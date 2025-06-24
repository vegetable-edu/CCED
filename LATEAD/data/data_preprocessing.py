import os
import json
import pandas as pd
from transformers import AutoTokenizer
import yaml

def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

def preprocess_data(dataset_name, raw_dir, processed_dir):
    config = load_config()
    tokenizer = AutoTokenizer.from_pretrained(config["slm"]["model_name"])
    
    # Load CSED JSON dataset
    raw_path = os.path.join(raw_dir, dataset_name)
    processed_path = os.path.join(processed_dir, dataset_name.split('.')[0])
    os.makedirs(processed_path, exist_ok=True)
    
    with open(raw_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to DataFrame
    processed_data = []
    for item in data:
        sentence = item["sentence"]
        event_type = item["eventype"]
        trigger = item["trigger"]
        inputs = tokenizer(sentence, max_length=config["slm"]["max_length"], truncation=True, padding="max_length")
        processed_data.append({
            "id": item["id"],
            "sentence": sentence,
            "trigger": trigger,
            "event_type": event_type,
            "event_type_id": item["eventype id"],
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        })
    
    # Save processed data
    df = pd.DataFrame(processed_data)
    df.to_pickle(os.path.join(processed_path, "processed.pkl"))
    print(f"Processed CSED data saved to {processed_path}")

if __name__ == "__main__":
    config = load_config()
    raw_dir = config["data"]["raw_dir"]
    processed_dir = config["data"]["processed_dir"]
    preprocess_data(config["data"]["datasets"]["csed"], raw_dir, processed_dir)