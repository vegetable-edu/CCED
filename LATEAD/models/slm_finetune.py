import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
import yaml
from torch.utils.data import DataLoader, Dataset

class EventDetectionDataset(Dataset):
    def __init__(self, data, event_types):
        self.data = data
        self.event_types = event_types
        self.label_map = {et: idx for idx, et in enumerate(self.event_types)}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"]),
            "attention_mask": torch.tensor(item["attention_mask"]),
            "labels": torch.tensor(self.label_map[item["event_type"]])
        }

class SLMEventDetector(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # [CLS] token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

def fine_tune_slm(config):
    device = torch.device(config["device"])
    tokenizer = AutoTokenizer.from_pretrained(config["slm"]["model_name"])
    model = SLMEventDetector(config["slm"]["model_name"], len(config["event_types"])).to(device)
    
    # Load processed CSED data
    data = pd.read_pickle(os.path.join(config["data"]["processed_dir"], "csed/processed.pkl"))
    dataset = EventDetectionDataset(data, config["event_types"])
    dataloader = DataLoader(dataset, batch_size=config["slm"]["batch_size"], shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["slm"]["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(config["slm"]["epochs"]):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")
    
    # Save model
    torch.save(model.state_dict(), "models/slm_finetuned.pt")
    print("SLM fine-tuned and saved.")

if __name__ == "__main__":
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    fine_tune_slm(config)