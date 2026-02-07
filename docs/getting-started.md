# Getting Started

This guide will help you set up AI Eval and run your first prompt evaluation.

## Prerequisites

- Go 1.21 or later
- An API key for Claude (Anthropic) or OpenAI

## Installation

### From Source

```bash
git clone https://github.com/stellarlinkco/ai-eval.git
cd ai-eval
go build -o eval ./cmd/eval
```

### Using Go Install

```bash
go install github.com/stellarlinkco/ai-eval/cmd/eval@latest
```

## Configuration

### 1. Create Configuration File

```bash
cp configs/config.yaml.example configs/config.yaml
```

### 2. Set API Keys

Set your API keys as environment variables:

```bash
# For Claude (Anthropic)
export ANTHROPIC_API_KEY="your-anthropic-api-key"
# Or use auth token
export ANTHROPIC_AUTH_TOKEN="your-auth-token"

# For OpenAI
export OPENAI_API_KEY="your-openai-api-key"
```

### 3. (Optional) Customize Configuration

Edit `configs/config.yaml` to customize:

```yaml
llm:
  default_provider: "claude"  # or "openai"
  providers:
    claude:
      api_key: ""  # Uses ANTHROPIC_API_KEY env var
      # base_url: "https://api.anthropic.com/v1"  # Custom endpoint
      # model: "claude-sonnet-4-5-20250929"  # Specific model
    openai:
      api_key: ""  # Uses OPENAI_API_KEY env var
      # base_url: "https://api.openai.com/v1"
      # model: "gpt-4o"

evaluation:
  trials: 3          # Number of trials per test case
  threshold: 0.8     # Pass threshold (0.0-1.0)
  output_format: "table"  # table, json, or github
```

## Your First Evaluation

### 1. Create a Prompt

Create a file `prompts/hello.yaml`:

```yaml
name: hello
version: "1.0"
is_system_prompt: true
template: |
  You are a helpful assistant. Always be concise and accurate.

  User request: {{.user_request}}
```

### 2. Create Test Cases

Create a file `tests/hello.yaml`:

```yaml
prompt: hello
suite: hello-tests
cases:
  - id: greeting
    input:
      user_request: "Say hello"
    evaluators:
      - type: contains
        expected: "hello"

  - id: math
    input:
      user_request: "What is 2 + 2?"
    evaluators:
      - type: contains
        expected: "4"
      - type: llm_judge
        criteria: "Response should be accurate and concise"
```

### 3. Run the Evaluation

```bash
./eval run --prompt hello
```

You should see output like:

```
Prompt: hello

Suite: hello-tests PASS
Cases: 2 passed=2 failed=0 pass_rate=1.00 avg_score=1.00
CASE      RESULT  SCORE  PASS@K  LAT(ms)  TOKENS  ERROR
greeting  PASS    1.000  1.000   1234     150
math      PASS    1.000  1.000   2345     200

Summary: suites=1 cases=2 passed=2 failed=0
Overall: PASS
```

## Next Steps

- [Evaluators Guide](evaluators.md) - Learn about different evaluator types
- [Benchmarks Guide](benchmarks.md) - Run standardized benchmarks
- [Configuration Reference](configuration.md) - Detailed configuration options
- [API Reference](api.md) - Web API documentation
