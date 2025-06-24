import yaml

class PromptConstructor:
    def __init__(self, config):
        self.event_types = config["event_types"]
    
    def construct_prompt(self, sentence, answer_aware_examples, structure_aware_examples, answer_candidates):
        prompt = "Task: Classify the cybersecurity event type in the following sentence from the given list of event types.\n\n"
        prompt += f"Event Types: {', '.join(self.event_types)}\n\n"
        
        # Add answer-aware examples
        prompt += "Answer-aware Examples:\n"
        for example in answer_aware_examples:
            prompt += f"Sentence: {example['sentence']}\nTrigger: {example['trigger']}\nEvent Type: {example['event_type']}\n\n"
        
        # Add structure-aware examples
        prompt += "Structure-aware Examples:\n"
        for example in structure_aware_examples:
            prompt += f"Sentence: {example['sentence']}\nTrigger: {example['trigger']}\nEvent Type: {example['event_type']}\n\n"
        
        # Add answer candidates
        prompt += "Answer Candidates (with confidence scores):\n"
        for candidate, score in answer_candidates:
            prompt += f"{candidate}: {score:.2f}\n"
        
        # Add test sentence
        prompt += f"\nTest Sentence: {sentence}\n"
        prompt += "Predicted Event Type: "
        
        return prompt

if __name__ == "__main__":
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    constructor = PromptConstructor(config)
    sentence = "有人能解释下报告安全缺陷具体是什么原理吗？"
    answer_aware = [{"sentence": "发现系统漏洞。", "trigger": "发现漏洞", "event_type": "Vulnerability Discovery"}]
    structure_aware = [{"sentence": "报告网络安全问题。", "trigger": "报告问题", "event_type": "Vulnerability Discovery"}]
    answer_candidates = [("Vulnerability Discovery", 0.85), ("Vulnerability Impact", 0.10), ("Malware", 0.05)]
    prompt = constructor.construct_prompt(sentence, answer_aware, structure_aware, answer_candidates)
    print(prompt)