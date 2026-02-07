# Configuration Reference

Complete reference for AI Eval configuration options.

## Configuration File

The default configuration file is `configs/config.yaml`. Use `--config` flag to specify an alternative:

```bash
eval run --config /path/to/custom-config.yaml --prompt example
```

## Full Configuration Example

```yaml
llm:
  # Default provider for evaluations
  default_provider: "claude"

  providers:
    claude:
      # API key (prefer environment variable ANTHROPIC_API_KEY)
      api_key: ""
      # Custom API endpoint (optional)
      base_url: "https://api.anthropic.com/v1"
      # Default model (optional)
      model: "claude-sonnet-4-5-20250929"

    openai:
      api_key: ""
      base_url: "https://api.openai.com/v1"
      model: "gpt-4o"

evaluation:
  # Number of trials per test case (for pass@k calculation)
  trials: 3
  # Pass threshold (0.0-1.0)
  threshold: 0.8
  # Output format: table, json, github
  output_format: "table"
  # Maximum concurrent evaluations
  concurrency: 4
  # Request timeout
  timeout: 120s

storage:
  # Storage type: sqlite or memory
  type: "sqlite"
  # Database file path (for sqlite)
  path: "data/ai-eval.db"
```

## LLM Configuration

### Providers

#### Claude (Anthropic)

```yaml
llm:
  providers:
    claude:
      api_key: ""  # Or set ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN
      base_url: "https://api.anthropic.com/v1"  # Optional
      model: "claude-sonnet-4-5-20250929"  # Optional
```

**Available models:**
- `claude-opus-4-6-20250929` - Most capable
- `claude-sonnet-4-5-20250929` - Balanced performance/cost
- `claude-haiku-3-5-20250929` - Fastest, lowest cost

#### OpenAI

```yaml
llm:
  providers:
    openai:
      api_key: ""  # Or set OPENAI_API_KEY
      base_url: "https://api.openai.com/v1"  # Optional
      model: "gpt-4o"  # Optional
```

**Available models:**
- `gpt-4o` - Latest GPT-4 Omni
- `gpt-4-turbo` - GPT-4 Turbo
- `gpt-3.5-turbo` - Faster, lower cost

### Custom/Proxy Endpoints

To use a proxy or self-hosted endpoint:

```yaml
llm:
  providers:
    openai:
      api_key: "your-proxy-key"
      base_url: "https://your-proxy.com/v1"
      model: "gpt-4o"
```

## Evaluation Configuration

### trials

Number of times to run each test case. Higher values give better pass@k estimates.

```yaml
evaluation:
  trials: 3  # Default
```

- `1`: Single run, no variance estimation
- `3`: Good balance of accuracy and cost
- `5+`: Higher confidence, higher cost

### threshold

Minimum pass rate threshold (0.0-1.0).

```yaml
evaluation:
  threshold: 0.8  # 80% of trials must pass
```

### output_format

Output format for results.

```yaml
evaluation:
  output_format: "table"  # Human-readable table
  # output_format: "json"  # JSONL for parsing
  # output_format: "github"  # GitHub Actions annotations
```

### concurrency

Maximum concurrent API calls.

```yaml
evaluation:
  concurrency: 4  # Default
```

Higher values speed up evaluation but may hit rate limits.

### timeout

Timeout for each API request.

```yaml
evaluation:
  timeout: 120s  # Default: 2 minutes
```

Increase for complex prompts or slow endpoints.

## Storage Configuration

### SQLite (Default)

```yaml
storage:
  type: "sqlite"
  path: "data/ai-eval.db"
```

### In-Memory

For testing or ephemeral use:

```yaml
storage:
  type: "memory"
```

## Environment Variables

| Variable | Description | Priority |
|----------|-------------|----------|
| `ANTHROPIC_API_KEY` | Claude API key | High |
| `ANTHROPIC_AUTH_TOKEN` | Claude auth token | Medium |
| `OPENAI_API_KEY` | OpenAI API key | High |
| `AI_EVAL_API_KEY` | API server key for `X-API-Key` authentication | High |
| `AI_EVAL_DISABLE_AUTH` | Disable API server authentication when set to `true` | Medium |
| `AI_EVAL_CORS_ORIGINS` | Enable CORS for the API server (`*` or comma-separated origins) | - |
| `AI_EVAL_MMLU_PATH` | Custom MMLU dataset path | - |
| `AI_EVAL_GSM8K_PATH` | Custom GSM8K dataset path | - |
| `AI_EVAL_HUMANEVAL_PATH` | Custom HumanEval dataset path | - |
| `AI_EVAL_ENABLE_CODE_EXEC` | Enable code execution (set to "1") | - |
| `AI_EVAL_SANDBOX_MODE` | HumanEval sandbox mode: `docker` (default), `host` (UNSAFE), `disabled` | - |

## CLI Flags

CLI flags override configuration file values:

```bash
# Override config file
eval run --config custom.yaml --prompt example

# Override trials
eval run --prompt example --trials 5

# Override threshold
eval run --prompt example --threshold 0.9

# Override output format
eval run --prompt example --output json

# Override provider/model (benchmark)
eval benchmark --dataset mmlu --provider openai --model gpt-4o
```

## Configuration Precedence

1. CLI flags (highest)
2. Environment variables
3. Configuration file
4. Default values (lowest)
