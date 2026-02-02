# ExPara: Explanation Paraphrase Consistency Benchmark
## ACL 2026 Research Project

## 🚀 100% Open Source - NO API KEYS NEEDED!

This project uses local HuggingFace models for all generation tasks.
No OpenAI, Anthropic, or other API keys required.

---

## Overview

ExPara is a benchmark and evaluation framework for measuring semantic consistency 
of chain-of-thought explanations across paraphrased inputs.

**Research Question:** When LLMs give the same answer to paraphrased questions, 
are their explanations semantically equivalent? Or are they post-hoc rationalizations?

---

## Installation

```bash
git clone <repo>
cd expara
pip install -r requirements.txt
```

---

## Recommended Models (by GPU memory)

| VRAM | Recommended Model | Notes |
|------|-------------------|-------|
| 4GB | `google/gemma-2-2b-it` | Use with `load_in_4bit=True` |
| 8GB | `Qwen/Qwen2.5-3B-Instruct` | Good balance |
| 12GB | `Qwen/Qwen2.5-7B-Instruct` | Excellent reasoning |
| 16GB+ | `Qwen/Qwen2.5-14B-Instruct` | Best quality |
| CPU only | Use `--use_mock` for testing | Or small models with patience |

---

## Quick Start

### Option 1: Test with mock data (no GPU needed)
```bash
python scripts/run_full_pipeline.py --use_mock --num_samples 50
```

### Option 2: Run with open-source models
```bash
# For 8GB+ GPU
python scripts/run_full_pipeline.py \
    --models Qwen/Qwen2.5-7B-Instruct \
    --num_samples 100

# For limited memory (4-bit quantization)  
python scripts/run_full_pipeline.py \
    --models Qwen/Qwen2.5-7B-Instruct \
    --quantize \
    --num_samples 100
```

### Run individual stages:
```bash
# 1. Generate paraphrases
python scripts/generate_paraphrases.py \
    --dataset gsm8k \
    --num_samples 100 \
    --model Qwen/Qwen2.5-3B-Instruct

# 2. Generate explanations  
python scripts/generate_explanations.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --input data/paraphrases/

# 3. Evaluate consistency
python scripts/evaluate_consistency.py \
    --explanations data/explanations/ \
    --output results/
```

---

## Python Usage

```python
from expara.data import DatasetLoader, ParaphraseGenerator
from expara.evaluation import ExplanationGenerator, GenerationConfig
from expara.metrics import ExParaSimScore

# Load data
loader = DatasetLoader()
questions = loader.load_gsm8k(num_samples=100)

# Generate paraphrases (using open-source model)
para_gen = ParaphraseGenerator(
    model_name="Qwen/Qwen2.5-3B-Instruct",
    load_in_4bit=True  # For limited memory
)
paraphrases = para_gen.generate_paraphrases_batch(questions)

# Generate explanations
exp_gen = ExplanationGenerator(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    load_in_4bit=True
)
config = GenerationConfig(prompt_type="zero_shot_cot", temperature=0.0)
explanations = exp_gen.generate_explanations_batch(questions, config=config)

# Evaluate consistency
scorer = ExParaSimScore(device="cuda", use_llm_judge=False)
score, metrics = scorer.compute(explanation1.reasoning, explanation2.reasoning)
print(f"ExPara-Sim Score: {score:.4f}")
```

---

## Project Structure

```
expara/
├── configs/           # Configuration files
├── data/              # Data loading, paraphrase generation
│   ├── loader.py      # Load GSM8K, StrategyQA, ARC, LogiQA
│   ├── paraphrase_generator.py  # Generate paraphrases
│   └── models.py      # Data structures
├── metrics/           # ExPara-Sim metrics
│   └── expara_sim.py  # All 5 metrics + composite score
├── evaluation/        # Evaluation pipeline
│   ├── explanation_generator.py  # CoT generation
│   └── pipeline.py    # Full evaluation pipeline
├── training/          # Consistency training
│   └── consistency_training.py  # Fine-tuning & DPO
├── utils/             # Utilities
└── scripts/           # CLI scripts
    ├── generate_paraphrases.py
    ├── generate_explanations.py
    ├── evaluate_consistency.py
    └── run_full_pipeline.py
```

---

## Metrics

ExPara-Sim combines 5 metrics:

| Metric | Description | Weight |
|--------|-------------|--------|
| **Semantic-Sim** | Cosine similarity of embeddings | 0.25 |
| **Entailment-Sym** | Bidirectional NLI score | 0.25 |
| **Step-Align** | Optimal transport alignment | 0.20 |
| **Fact-Overlap** | Jaccard of factual claims | 0.15 |
| **LLM-Judge** | Model-based judgment (optional) | 0.15 |

---

## Citation

```bibtex
@inproceedings{expara2026,
  title={Beyond Answer Consistency: Measuring Explanation Stability Under Query Paraphrase},
  author={...},
  booktitle={Proceedings of ACL 2026},
  year={2026}
}
```

---

## License

MIT License
