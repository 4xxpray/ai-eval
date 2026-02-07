# Benchmarks Guide

AI Eval includes support for running standardized LLM benchmarks to compare model performance.

## Available Benchmarks

### MMLU (Massive Multitask Language Understanding)

A multiple-choice benchmark covering 57 subjects across STEM, humanities, social sciences, and more.

```bash
# Run with default sample size
eval benchmark --dataset mmlu

# Run with specific sample size
eval benchmark --dataset mmlu --sample-size 100

# Run with specific provider/model
eval benchmark --dataset mmlu --provider claude --model claude-sonnet-4-5-20250929
```

**What it tests:**
- General knowledge
- Reasoning ability
- Domain expertise

### GSM8K (Grade School Math)

Math word problems requiring multi-step reasoning.

```bash
eval benchmark --dataset gsm8k --sample-size 100
```

**What it tests:**
- Mathematical reasoning
- Multi-step problem solving
- Arithmetic accuracy

### HumanEval

Code generation benchmark with function completion tasks.

```bash
# Requires enabling code execution
AI_EVAL_ENABLE_CODE_EXEC=1 eval benchmark --dataset humaneval --sample-size 50
```

**What it tests:**
- Code generation
- Algorithm implementation
- Function correctness

**Sandboxing:** By default HumanEval runs untrusted Python inside a Docker sandbox (`AI_EVAL_SANDBOX_MODE=docker`). Ensure Docker is installed and the image is available:

```bash
docker pull python:3.11-slim
```

To opt out (UNSAFE) and run on the host:

```bash
AI_EVAL_ENABLE_CODE_EXEC=1 AI_EVAL_SANDBOX_MODE=host eval benchmark --dataset humaneval --sample-size 50
```

Even in Docker, enable with caution.

## Comparing Models

Run the same benchmark with different providers to compare:

```bash
# Claude
eval benchmark --dataset mmlu --provider claude --sample-size 100

# OpenAI
eval benchmark --dataset mmlu --provider openai --model gpt-4o --sample-size 100

# View comparison
eval leaderboard --dataset mmlu
```

## Viewing Results

### Leaderboard

```bash
# View top results for a dataset
eval leaderboard --dataset mmlu

# Limit to top N
eval leaderboard --dataset mmlu --top 10

# JSON output
eval leaderboard --dataset mmlu --format json
```

**Example output:**

```
RANK  MODEL                        PROVIDER  SCORE   ACCURACY  LAT(ms)  DATE
1     claude-sonnet-4-5-20250929   claude    1.0000  1.0000    14444    2026-02-07
2     gpt-4o                       openai    0.9800  0.9800    18234    2026-02-07
3     claude-haiku-3-5-20250929    claude    0.9200  0.9200    8123     2026-02-07
```

## Custom Datasets

You can provide your own benchmark data in JSONL format.

### MMLU Format

```json
{"question": "What is the capital of France?", "choices": ["London", "Paris", "Berlin", "Madrid"], "answer": "B", "subject": "geography"}
```

Set the data path:

```bash
export AI_EVAL_MMLU_PATH="/path/to/your/mmlu.jsonl"
eval benchmark --dataset mmlu
```

### GSM8K Format

```json
{"question": "If John has 5 apples and gives 2 to Mary, how many does he have left?", "answer": "3"}
```

```bash
export AI_EVAL_GSM8K_PATH="/path/to/your/gsm8k.jsonl"
eval benchmark --dataset gsm8k
```

### HumanEval Format

```json
{"task_id": "HumanEval/0", "prompt": "def add(a, b):\n    \"\"\"Return sum of a and b.\"\"\"\n", "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n\ncheck(add)\n", "entry_point": "add"}
```

```bash
export AI_EVAL_HUMANEVAL_PATH="/path/to/your/humaneval.jsonl"
AI_EVAL_ENABLE_CODE_EXEC=1 eval benchmark --dataset humaneval
```

## Benchmark Configuration

### Sample Size

Control how many questions to run:

```bash
# Quick test (10 questions)
eval benchmark --dataset mmlu --sample-size 10

# Full benchmark
eval benchmark --dataset mmlu --sample-size 0  # 0 = all available
```

### Provider Selection

Override the default provider:

```bash
# Use OpenAI instead of default Claude
eval benchmark --dataset gsm8k --provider openai --model gpt-4o

# Use specific Claude model
eval benchmark --dataset gsm8k --provider claude --model claude-opus-4-6-20250929
```

## Interpreting Results

### Score vs Accuracy

- **Score**: Normalized score (0.0-1.0)
- **Accuracy**: Percentage of correct answers

For multiple-choice benchmarks (MMLU), these are typically the same.

### Latency

Total time in milliseconds for all questions. Lower is better for cost efficiency.

### Tokens

Total tokens used (input + output). Consider this for cost estimation.

## Best Practices

1. **Use consistent sample sizes** when comparing models
2. **Run multiple times** to account for variance
3. **Consider cost** - larger sample sizes cost more
4. **Check leaderboard** before running to avoid duplicates
5. **Use appropriate models** - don't waste expensive models on simple benchmarks

## Troubleshooting

### HumanEval shows 0% accuracy

Make sure code execution is enabled:

```bash
AI_EVAL_ENABLE_CODE_EXEC=1 eval benchmark --dataset humaneval
```

### Benchmark fails with timeout

Increase the timeout in config:

```yaml
evaluation:
  timeout: 300s  # 5 minutes
```

### Provider not found

Ensure the provider is configured in `configs/config.yaml` and API key is set.
