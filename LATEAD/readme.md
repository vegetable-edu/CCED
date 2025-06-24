#### 3. `config/config.yaml`
存储超参数、文件路径和其他配置。

```yaml
# General settings
seed: 42
device: cuda  # or cpu

# Dataset paths
data:
  raw_dir: data/raw/
  processed_dir: data/processed/
  datasets:
    ace2005: LDC2006T06
    ere_en: LDC2020T18

# SLM settings
slm:
  model_name: roberta-base
  embedding: glove
  max_length: 128
  batch_size: 16
  learning_rate: 2e-54
  epochs: 5

# LLM settings
llm:
  model: openai/gpt-3.5-turbo  # or local LLM like LLaMA
  temperature: 0.5
  max_tokens: 512

# Latent answer generation
latent_answers:
  answer_aware_examples: 5
  structure_aware_examples: 5
  answer_candidates: 8

# Experiment settings
experiments:
  full_shot:
    train_ratio: 1.0
  few_shot:
    train_ratios: [0.1, 0.2, 0.4, 0.6, 0.8]