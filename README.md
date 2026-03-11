# TokenFind

A stateless, rule-based safety critic for LLM output evaluation. TokenFind evaluates LLM outputs through deterministic analysis — without machine learning inference, training data, or knowledge of input prompts.

**Paper:** [TokenFind: A Stateless, Rule-Based Safety Critic for LLM Output Evaluation](link-to-arxiv)

## Overview

TokenFind is a completely "output-text-only" LLM based system. It has no knowledge of any user input prompts, nor does it have knowledge of the history of any conversations. For this reason, TokenFind is architected in such a way that it is impervious to prompt injection attacks that can damage the primary model(s), as well as the safety system(s).

TokenFind uses five layers of detection:

- **Phrase-level Intent Keywords** -- The intersection of templates with verbs that represent intent to do harm versus the mere presence of words
- **Structural Regex Patterns** -- Detection of PII (e.g., SSN's, API Keys, Credit Cards, Passwords) and structural patterns of dangerous synthesis
- **Shannon Entropy Scanning** -- High-entropy string detection (i.e., leaked secrets, tokens, hashes)
- **Character-Level N-Gram Overlap** -- Protected content reproduction detection by comparing character-level 4-grams from the detected token against the corresponding 4-grams of a reference text.
- **Refusal Classification** -- Document level classification using adaptive thresholding, with soft refusal detection.

## Project Structure

```
tokenfind/
├── critic/
│   ├── __init__.py
│   ├── critic.py          # CriticAgent; main review interface
│   ├── models.py          # Data models (SafetyRule, RuleMatch, ReviewResult)
│   └── policy.py          # SafetyPolicy; detection layers and rule definitions
├── eval/
│   ├── adversarial_prompts.jsonl    # 37 adversarial prompts (7 attack categories)
│   ├── benign_prompts.jsonl         # 30 benign prompts (7 categories)
│   ├── mock_outputs.jsonl           # 12 controlled mock outputs
│   ├── test_system_prompt.txt       # Fictitious system prompt used in evaluation
│   ├── raw_outputs_mistral.jsonl    # Mistral raw outputs (replay)
│   ├── raw_outputs_llama3.jsonl     # Llama 3 raw outputs (replay)
│   ├── raw_outputs.jsonl            # Gemma 3 4B raw outputs (replay)
│   ├── results.csv                  # Gemma 3 4B evaluation results
│   ├── results_mistral.csv          # Mistral evaluation results
│   └── results_llama3.csv           # Llama 3 evaluation results
├── tests/
│   └── test_critic.py
├── eval_harness.py        # Evaluation harness (multi-backend)
└── pyproject.toml
```

## Quick Start

```bash
# Clone
git clone https://github.com/kurshatM/tokenfind.git
cd tokenfind

# Install
pip install -e .

# Run tests
pytest
```

## Usage

```python
from critic import CriticAgent
from critic.policy import SafetyPolicy

system_prompt = "Your protected system prompt here..."
policy = SafetyPolicy(protected_content=system_prompt)
agent = CriticAgent(policy=policy)

result = agent.review("The API key is sk_live_abc123...", sanitize=True)
print(result.verdict)        # BLOCK
print(result.score)          # 0.0
print(result.sanitized_text) # [REDACTED]
```

## Evaluation

The evaluation harness supports four backends-- Ollama (local), OpenAI-compatible APIs, mock outputs, and replay from saved outputs.

```bash
# Run against a local Ollama model
python eval_harness.py --backend ollama --model gemma3:4b \
    --system-prompt-file eval/test_system_prompt.txt

# Replay saved outputs (no LLM needed)
python eval_harness.py --backend replay --model mistral \
    --replay-file eval/raw_outputs_mistral.jsonl \
    --system-prompt-file eval/test_system_prompt.txt
```

## Results

Evaluated against Gemma 3 4B, Llama 3, and Mistral using 37 adversarial and 30 benign prompts:

| Model | Blocked | False-Allow | False-Block | p95 Latency |
|---|---|---|---|---|
| Gemma 3 4B | 54.1% | 2.7% | 0.0% | 4.63ms |
| Llama 3 | 16.2% | 13.5% | 0.0% | 2.30ms |
| Mistral | 56.8% | 8.1%* | 3.3% | 3.34ms |

\* Human-verified. Automated rate was 18.9%; manual review found 4 of 7 were misclassified refusals.

## License

MIT