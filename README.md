# AI Eval

A comprehensive prompt evaluation and optimization system for LLM applications.

## Features

- **Multi-provider LLM support**: Claude (Anthropic), OpenAI, and compatible APIs
- **Rich evaluator suite**:
  - `llm_judge` - LLM-based response quality scoring
  - `regex`, `contains`, `exact` - Pattern matching evaluators
  - `factuality`, `similarity` - Semantic evaluators
  - `safety`, `agent`, `rag` - Specialized evaluators
- **Benchmark datasets**: MMLU, GSM8K, HumanEval
- **CLI tools**: run, compare, benchmark, optimize, diagnose, leaderboard
- **Web API server** with evaluation endpoints
- **SQLite storage** for results and leaderboard tracking

## Architecture

```mermaid
graph TB
    subgraph CLI["CLI (cmd/eval)"]
        RUN[eval run]
        BENCH[eval benchmark]
        CMP[eval compare]
        OPT[eval optimize]
        LB[eval leaderboard]
    end

    subgraph API["Web API (api/)"]
        REST[REST Endpoints]
        WEB[Web UI]
    end

    subgraph Core["Core Engine"]
        RUNNER[Runner]
        LOADER[Prompt Loader]
        TC[Test Case Loader]
    end

    subgraph LLM["LLM Providers (internal/llm)"]
        CLAUDE[Claude / Anthropic]
        OPENAI[OpenAI / Compatible]
    end

    subgraph Evaluators["Evaluators (internal/evaluator)"]
        direction LR
        BASIC["Basic\ncontains | exact | regex"]
        SEMANTIC["Semantic\nllm_judge | factuality | similarity"]
        SAFETY["Safety\ntoxicity | bias | hallucination"]
        AGENT["Agent\ntool_selection | efficiency"]
        RAG["RAG\nfaithfulness | relevancy | precision"]
    end

    subgraph Benchmark["Benchmarks (internal/benchmark)"]
        MMLU[MMLU]
        GSM8K[GSM8K]
        HUMANEVAL[HumanEval]
    end

    subgraph Storage["Storage"]
        SQLITE[(SQLite)]
    end

    CLI --> Core
    API --> Core
    Core --> LLM
    Core --> Evaluators
    BENCH --> Benchmark
    Benchmark --> LLM
    LB --> SQLITE
    BENCH --> SQLITE
    RUNNER --> LOADER
    RUNNER --> TC

    style CLI fill:#e1f5fe,stroke:#01579b
    style API fill:#e8f5e9,stroke:#1b5e20
    style Core fill:#fff3e0,stroke:#e65100
    style LLM fill:#f3e5f5,stroke:#4a148c
    style Evaluators fill:#fce4ec,stroke:#880e4f
    style Benchmark fill:#e0f2f1,stroke:#004d40
    style Storage fill:#f5f5f5,stroke:#616161
```

### Evaluation Flow

```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant Runner
    participant LLM as LLM Provider
    participant Eval as Evaluators
    participant DB as SQLite

    User->>CLI: eval run --prompt example
    CLI->>Runner: Load prompt + test cases

    loop For each test case
        loop For each trial (1..N)
            Runner->>LLM: Send prompt + input
            LLM-->>Runner: Response
            Runner->>Eval: Evaluate response
            Eval-->>Runner: Score + Pass/Fail
        end
    end

    Runner->>DB: Store results
    Runner-->>CLI: Suite results
    CLI-->>User: Pass/Fail report
```

### Benchmark Flow

```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant BM as Benchmark Runner
    participant DS as Dataset
    participant LLM as LLM Provider
    participant LB as Leaderboard

    User->>CLI: eval benchmark --dataset mmlu
    CLI->>DS: Load questions
    DS-->>BM: Questions[]

    loop For each question
        BM->>LLM: Question prompt
        LLM-->>BM: Answer
        BM->>DS: Evaluate answer
        DS-->>BM: Score
    end

    BM->>LB: Save result
    BM-->>User: Accuracy report
```

## Installation

```bash
go install github.com/stellarlinkco/ai-eval/cmd/eval@latest
```

Or build from source:

```bash
git clone https://github.com/stellarlinkco/ai-eval.git
cd ai-eval
go build -o eval ./cmd/eval
```

## Configuration

1. Copy the example config:
```bash
cp configs/config.yaml.example configs/config.yaml
```

2. Set API keys via environment variables:
```bash
# Claude
export ANTHROPIC_API_KEY="your-key"
# or
export ANTHROPIC_AUTH_TOKEN="your-token"

# OpenAI
export OPENAI_API_KEY="your-key"
```

3. (Optional) Customize `configs/config.yaml` for base URLs, models, etc.

## Usage

### Run Prompt Evaluation

```bash
# Run evaluation for a specific prompt
eval run --prompt example

# Run with custom trials
eval run --prompt example --trials 5

# Run all prompts
eval run --all
```

### Run Benchmarks

```bash
# MMLU benchmark
eval benchmark --dataset mmlu --sample-size 100

# GSM8K math benchmark
eval benchmark --dataset gsm8k --sample-size 100

# HumanEval code benchmark (requires AI_EVAL_ENABLE_CODE_EXEC=1; defaults to Docker sandbox)
AI_EVAL_ENABLE_CODE_EXEC=1 eval benchmark --dataset humaneval --sample-size 50
AI_EVAL_ENABLE_CODE_EXEC=1 AI_EVAL_SANDBOX_MODE=host eval benchmark --dataset humaneval --sample-size 50 # UNSAFE

# Compare providers
eval benchmark --dataset mmlu --provider claude --model claude-sonnet-4-5-20250929
eval benchmark --dataset mmlu --provider openai --model gpt-4o
```

### View Leaderboard

```bash
eval leaderboard --dataset mmlu
eval leaderboard --dataset gsm8k --top 10
eval leaderboard --dataset humaneval --format json
```

### Compare Prompt Versions

```bash
eval compare --prompt example
```

### Optimize Prompts

```bash
eval optimize --prompt example
```

## Documentation

- [Getting Started](docs/getting-started.md) - Installation and first evaluation
- [Evaluators Guide](docs/evaluators.md) - All evaluator types and usage
- [Benchmarks Guide](docs/benchmarks.md) - Running standardized benchmarks
- [Configuration Reference](docs/configuration.md) - Detailed config options
- [API Reference](docs/api.md) - REST API documentation

## Project Structure

```
ai-eval/
├── cmd/
│   ├── eval/           # CLI application
│   └── server/         # Web API server
├── configs/
│   └── config.yaml.example
├── internal/
│   ├── benchmark/      # MMLU, GSM8K, HumanEval datasets
│   ├── evaluator/      # Evaluator implementations
│   │   ├── agent/      # Agent-specific evaluators
│   │   ├── rag/        # RAG evaluators
│   │   └── safety/     # Safety evaluators
│   ├── llm/            # LLM provider implementations
│   ├── optimizer/      # Prompt optimization
│   ├── runner/         # Test runner
│   └── ...
├── prompts/            # Prompt definitions (YAML)
├── tests/              # Test case definitions (YAML)
└── web/static/         # Web UI
```

## Prompt Definition Format

```yaml
# prompts/example.yaml
name: example
version: "1.0"
is_system_prompt: true
template: |
  You are a helpful assistant.

  User request: {{.user_task}}
```

## Test Case Format

```yaml
# tests/example.yaml
prompt: example
suite: example
cases:
  - id: test-1
    input:
      user_task: "What is 2+2?"
    expected:
      contains:
        - "4"
    evaluators:
      - type: llm_judge
        criteria: "Response should be accurate and concise"
```

## License

This project is licensed under the [GNU Affero General Public License v3.0 (AGPL-3.0)](LICENSE).

This means:
- You can use, modify, and distribute this software
- If you modify and deploy this software as a network service, you must make your source code available
- Any derivative work must also be licensed under AGPL-3.0

### Commercial License

For commercial licensing options that don't require AGPL compliance, please contact:

**Email:** support@stellarlink.co
