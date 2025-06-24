import pandas as pd
from pipeline.inference import LSLAEDInference
from evaluation.metrics import evaluate_predictions
import yaml

def run_few_shot(config):
    inference = LSLAEDInference(config)
    dataset = pd.read_pickle(os.path.join(config["data"]["processed_dir"], "csed/processed.pkl"))
    
    results = {}
    for ratio in config["experiments"]["few_shot"]["train_ratios"]:
        subset = dataset.sample(frac=ratio, random_state=config["seed"])
        true_labels = []
        pred_labels = []
        for _, row in subset.iterrows():
            sentence = row["sentence"]
            true_label = row["event_type"]
            pred_label = inference.predict(sentence)
            true_labels.append(true_label)
            pred_labels.append(pred_label)
        
        metrics = evaluate_predictions(true_labels, pred_labels)
        results[ratio] = metrics
        print(f"Few-shot Metrics (Ratio {ratio}):", metrics)
    
    return results

if __name__ == "__main__":
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    results = run_few_shot(config)