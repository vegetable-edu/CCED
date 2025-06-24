import pandas as pd
from pipeline.latent_answer_generation import LatentAnswerGenerator
from pipeline.prompt_construction import PromptConstructor
from models.llm_interface import LLMInterface
import yaml

class LSLAEDInference:
    def __init__(self, config):
        self.latent_generator = LatentAnswerGenerator(config)
        self.prompt_constructor = PromptConstructor(config)
        self.llm = LLMInterface(config)
        self.dataset = pd.read_pickle(os.path.join(config["data"]["processed_dir"], "csed/processed.pkl"))
    
    def predict(self, sentence):
        candidates = self.latent_generator.generate_answer_candidates(sentence)
        answer_aware = self.latent_generator.generate_answer_aware_examples(sentence, self.dataset)
        structure_aware = self.latent_generator.generate_structure_aware_examples(sentence, self.dataset)
        prompt = self.prompt_constructor.construct_prompt(sentence, answer_aware, structure_aware, candidates)
        prediction = self.llm.generate(prompt)
        return prediction.strip()

if __name__ == "__main__":
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    inference = LSLAEDInference(config)
    sentence = "有人能解释下报告安全缺陷具体是什么原理吗？"
    prediction = inference.predict(sentence)
    print(f"Predicted Event Type: {prediction}")