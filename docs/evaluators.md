# Evaluators Guide

AI Eval evaluates each test case using two independent mechanisms:

- `expected`: deterministic assertions (exact/contains/regex/json_schema/not_contains/tool_calls)
- `evaluators`: optional LLM-based evaluators (judge, similarity, safety, RAG, agent)

A test case must define at least one `expected` assertion or at least one `evaluators` entry.

## Where this lives

These fields are part of `tests/*.yaml` suites:

```yaml
suite: hello-tests
prompt: hello
cases:
  - id: greeting
    input:
      user_request: "Say hello"
    expected:
      contains:
        - "hello"
    evaluators:
      - type: llm_judge
        criteria: "Response should be friendly and concise"
        score_threshold: 0.8
```

## Expected assertions (`expected`)

### exact_match

Exact string match.

```yaml
expected:
  exact_match: "Hello, world!"
```

### contains

Require the response to include **all** substrings.

```yaml
expected:
  contains:
    - "hello"
    - "world"
```

### not_contains

Require the response to **not** include any forbidden substrings.

```yaml
expected:
  not_contains:
    - "I can't"
    - "I won't"
```

### regex

Require the response to match **all** regex patterns (Go `regexp` syntax).

```yaml
expected:
  regex:
    - "^\\\\d{4}-\\\\d{2}-\\\\d{2}$"
```

### json_schema

Require the response to be valid JSON matching a minimal schema object.

```yaml
expected:
  json_schema:
    type: object
    required: ["name", "age"]
    properties:
      name: {type: string}
      age: {type: integer}
```

### tool_calls

Assert tool calls made by the model.

```yaml
expected:
  tool_calls:
    - name: get_weather
      args_match:
        location: "San Francisco"
      order: 1
      required: true
```

Notes:
- `order` is 1-based. Use `0` (or omit) for unordered matching.
- `args_match` is a subset match (only the provided keys must match).
- String values in `args_match` may use `regex:<pattern>` to match tool arguments by regex.

Example:

```yaml
expected:
  tool_calls:
    - name: search
      args_match:
        query: "regex:^golang\\\\s+error\\\\s+wrapping"
      required: true
```

## Evaluators (`evaluators`)

Each evaluator entry has:
- `type` (required)
- type-specific configuration fields
- `score_threshold` (optional): pass threshold (meaning depends on evaluator type)

Supported `type` values:
`exact`, `contains`, `regex`, `json_schema`, `llm_judge`, `similarity`, `factuality`, `tool_call`,
`faithfulness`, `relevancy`, `precision`,
`task_completion`, `tool_selection`, `efficiency`,
`hallucination`, `toxicity`, `bias`.

For the simple types (`exact`, `contains`, `regex`, `json_schema`) the expected values still come from the
`expected` block. In most cases you should configure them via `expected` only.

### llm_judge

Scores a response against free-form criteria using an LLM.

```yaml
evaluators:
  - type: llm_judge
    criteria: |
      The response should:
      - Be accurate
      - Be concise
    rubric:
      - "Accuracy"
      - "Clarity"
    score_scale: 5
    score_threshold: 0.8
```

`criteria` is required. `score_threshold` is the minimum normalized score (0.0-1.0).

### similarity

Semantic similarity against a reference answer.

```yaml
evaluators:
  - type: similarity
    reference: "The capital of France is Paris."
    score_threshold: 0.8
```

`reference` is required. `score_threshold` is the minimum similarity score (0.0-1.0).

### factuality

Fact-check against ground truth.

```yaml
evaluators:
  - type: factuality
    ground_truth: |
      Python was created by Guido van Rossum.
```

`ground_truth` is required.

### faithfulness

RAG faithfulness to retrieved context.

```yaml
evaluators:
  - type: faithfulness
    context: "Returns accepted within 30 days."
    score_threshold: 0.8
```

`context` is required. `score_threshold` is the minimum score (0.0-1.0).

### relevancy

RAG/query relevancy.

```yaml
evaluators:
  - type: relevancy
    question: "What is the return policy?"
    score_threshold: 0.8
```

`question` is required. `score_threshold` is the minimum score (0.0-1.0).

### precision

Retrieval precision for a question/context pair.

```yaml
evaluators:
  - type: precision
    question: "What is the return policy?"
    context: "..."
    score_threshold: 0.8
```

`question` and `context` are required. Set `score_threshold` to enforce a minimum score.

### task_completion

Agent task completion.

```yaml
evaluators:
  - type: task_completion
    task: "Write a short summary of the document"
    criteria_list:
      - "Mentions the main conclusion"
      - "Under 120 words"
    score_threshold: 0.7
```

`task` is required.

### tool_selection

Score whether the agent used expected tools.

```yaml
evaluators:
  - type: tool_selection
    expected_tools: ["search", "calculate"]
    score_threshold: 0.8
```

### efficiency

Score against step/token budgets.

```yaml
evaluators:
  - type: efficiency
    max_steps: 5
    max_tokens: 2000
    score_threshold: 0.8
```

At least one of `max_steps` or `max_tokens` should be set.

### hallucination

Detect hallucinations against ground truth.

```yaml
evaluators:
  - type: hallucination
    ground_truth: "The meeting is Tuesday at 3pm."
    score_threshold: 0.9
```

`ground_truth` is required. `score_threshold` is the minimum consistency score (0.0-1.0).

### toxicity

Detect harmful/toxic content.

```yaml
evaluators:
  - type: toxicity
    score_threshold: 0.1
```

`score_threshold` is the **maximum allowed** toxicity (0.0-1.0). Lower is stricter.

### bias

Detect biased or discriminatory content.

```yaml
evaluators:
  - type: bias
    categories: ["gender", "race"]
    score_threshold: 0.1
```

`score_threshold` is the **maximum allowed** bias (0.0-1.0). Lower is stricter.

### tool_call (threshold for `expected.tool_calls`)

If you use `expected.tool_calls`, you can optionally control pass/fail using a match-score threshold:

```yaml
evaluators:
  - type: tool_call
    score_threshold: 0.8
```

## Combining checks

All configured assertions and evaluators must pass for the case to pass.
