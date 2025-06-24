import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer
from models.slm_finetune import SLMEventDetector
import yaml
import spacy

class LatentAnswerGenerator:
    def __init__(self, config):
        self.device = torch.device(config["device"])
        self.slm = SLMEventDetector(config["slm"]["model_name"], len(config["event_types"])).to(self.device)
        self.slm.load_state_dict(torch.load("models/slm_finetuned.pt"))
        self.slm.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(config["slm"]["model_name"])
        self.sentence_encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.event_types = config["event_types"]
        self.nlp = spacy.load("en_core_web_sm")  # For structural analysis
    
    def _extract_potential_trigger(self, sentence):
        """Extract potential trigger using spaCy for dependency parsing."""
        doc = self.nlp(sentence)
        # Heuristic: Select the main verb or noun phrase as potential trigger
        for token in doc:
            if token.dep_ in ["ROOT", "nsubj", "dobj"] and token.pos_ in ["VERB", "NOUN"]:
                return token.text
        return sentence.split()[0]  # Fallback to first word
    
    def generate_answer_candidates(self, sentence, k=8, prob_threshold=0.05):
        inputs = self.tokenizer(sentence, return_tensors="pt", max_length=128, truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            logits = self.slm(inputs["input_ids"], inputs["attention_mask"])
            probs = torch.softmax(logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, min(5, len(self.event_types)), dim=-1)
        candidates = [(self.event_types[idx], prob.item()) for idx, prob in zip(top_k_indices[0], top_k_probs[0]) if prob.item() >= prob_threshold]
        return candidates
    
    def generate_answer_aware_examples(self, sentence, dataset, m=5):
        sentence embedding = self.sentence_encoder.encode(sentence, convert_to_tensor=True)
        dataset_embeddings = self.sentence_encoder.encode(dataset["sentence"].tolist(), convert_to_tensor=True)
        cos_scores = util.cos_sim(sentence_embedding, dataset_embeddings)[0]
        top_m_indices = torch.topk(cos_scores, min(m, len(dataset))).indices
        return dataset.iloc[top_m_indices][["sentence", "event_type", "trigger"]].to_dict("records")
    
    def generate_structure_aware_examples(self, sentence, dataset, n=5):
        # Use spaCy to analyze sentence structure (dependency tree similarity)
        input_doc = self.nlp(sentence)
        input_trigger = self._extract_potential_trigger(sentence)
        input_trigger_embedding = self.sentence_encoder.encode(input_trigger, convert_to_tensor=True)
        
        dataset["trigger_embedding"] = dataset["trigger"].apply(lambda x: self.sentence_encoder.encode(x, convert_to_tensor=True))
        trigger_similarities = dataset["trigger_embedding"].apply(lambda x: util.cos_sim(input_trigger_embedding, x)[0].item())
        
        # Combine with sentence length similarity
        input_length = len(sentence.split())
        dataset["sentence_length"] = dataset["sentence"].apply(lambda x: len(x.split()))
        length_similarities = 1 / (1 + abs(dataset["sentence_length"] - input_length))
        
        # Weighted similarity (70% trigger, 30% length)
        combined_similarities = 0.7 * trigger_similarities + 0.3 * length_similarities
        top_n_indices = combined_similarities.argsort()[-min(n, len(dataset)):][::-1]
        
        return dataset.iloc[top_n_indices][["sentence", "event_type", "trigger"]].to_dict("records")

if __name__ == "__main__":
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    generator = LatentAnswerGenerator(config)
    dataset = pd.read_pickle(os.path.join(config["data"]["processed_dir"], "csed/processed.pkl"))
    sentence = "有人能解释下报告安全缺陷具体是什么原理吗？"
    candidates = generator.generate_answer_candidates(sentence)
    answer_aware = generator.generate_answer_aware_examples(sentence, dataset)
    structure_aware = generator.generate_structure_aware_examples(sentence, dataset)
    print("Candidates:", candidates)
    print("Answer-aware examples:", answer_aware)
    print("Structure-aware examples:", structure_aware)
