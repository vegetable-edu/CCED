from openai import OpenAI
import yaml

class LLMInterface:
    def __init__(self, config):
        self.model = config["llm"]["model"]
        self.temperature = config["llm"]["temperature"]
        self.max_tokens = config["llm"]["max_tokens"]
        self.client = OpenAI(api_key="YOUR_API_KEY")  # Replace with actual API key
    
    def generate(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content

if __name__ == "__main__":
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    llm = LLMInterface(config)
    prompt = "Classify the event type in this sentence: "
    print(llm.generate(prompt))