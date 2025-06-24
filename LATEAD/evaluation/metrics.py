from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_predictions(true_labels, pred_labels):
    precision = precision_score(true_labels, pred_labels, average="macro")
    recall = recall_score(true_labels, pred_labels, average="macro")
    f1 = f1_score(true_labels, pred_labels, average="macro")
    return {"precision": precision, "recall": recall, "f1": f1}

if __name__ == "__main__":
    true_labels = ["Vulnerability Discovery", "Vulnerability Impact"]
    pred_labels = ["Vulnerability Discovery", "Vulnerability Impact"]
    metrics = evaluate_predictions(true_labels, pred_labels)
    print("Metrics:", metrics)