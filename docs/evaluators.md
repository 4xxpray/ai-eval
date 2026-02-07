# Evaluators Guide

AI Eval provides a rich set of evaluators to assess LLM responses. This guide covers each evaluator type and how to use them effectively.

## Basic Evaluators

### exact

Checks for an exact string match.

```yaml
evaluators:
  - type: exact
    expected: "Hello, World!"
```

**Options:**
- `expected` (string): The exact string to match
- `case_sensitive` (bool, default: true): Whether to perform case-sensitive matching

### contains

Checks if the response contains a substring.

```yaml
evaluators:
  - type: contains
    expected: "hello"
```

**Options:**
- `expected` (string or array): Substring(s) to find
- `case_sensitive` (bool, default: false): Case sensitivity
- `all` (bool, default: false): When expected is an array, require all substrings

### not_contains

Checks that the response does NOT contain certain strings.

```yaml
evaluators:
  - type: not_contains
    expected:
      - "error"
      - "failed"
```

### regex

Matches response against a regular expression.

```yaml
evaluators:
  - type: regex
    pattern: "\\d{4}-\\d{2}-\\d{2}"  # Date format YYYY-MM-DD
```

**Options:**
- `pattern` (string): Regular expression pattern
- `expected` (string): Alternative to `pattern`
- `flags` (string): Regex flags (e.g., "i" for case-insensitive)

## Semantic Evaluators

### llm_judge

Uses an LLM to evaluate response quality against specified criteria.

```yaml
evaluators:
  - type: llm_judge
    criteria: |
      The response should:
      - Be accurate and factually correct
      - Be concise (under 100 words)
      - Use professional tone
    score_threshold: 0.8
```

**Options:**
- `criteria` (string): Evaluation criteria description
- `rubric` (array): Specific scoring dimensions
- `score_scale` (int, default: 5): Rating scale (1-5 or 1-10)
- `score_threshold` (float, default: 0.6): Minimum normalized score to pass
- `context` (string): Additional context about the original question

**Example with rubric:**

```yaml
evaluators:
  - type: llm_judge
    criteria: "Evaluate the code review quality"
    rubric:
      - "Identifies actual bugs or issues"
      - "Provides constructive suggestions"
      - "Explains reasoning clearly"
      - "Follows coding best practices"
    score_scale: 10
    score_threshold: 0.7
```

### factuality

Evaluates factual accuracy of the response.

```yaml
evaluators:
  - type: factuality
    expected:
      facts:
        - "Python was created by Guido van Rossum"
        - "Python was first released in 1991"
```

### similarity

Compares semantic similarity to a reference response.

```yaml
evaluators:
  - type: similarity
    expected:
      reference: "The capital of France is Paris."
      threshold: 0.8
```

## Structured Output Evaluators

### json_schema

Validates that response is valid JSON matching a schema.

```yaml
evaluators:
  - type: json_schema
    expected:
      type: object
      required:
        - name
        - age
      properties:
        name:
          type: string
        age:
          type: integer
          minimum: 0
```

### tool_call

Evaluates tool/function call responses.

```yaml
evaluators:
  - type: tool_call
    expected:
      name: "get_weather"
      arguments:
        location: "San Francisco"
```

**Options:**
- `name` (string): Expected tool name
- `arguments` (object): Expected arguments (partial match)
- `strict` (bool): Require exact argument match

## Safety Evaluators

### safety/toxicity

Detects toxic or harmful content.

```yaml
evaluators:
  - type: toxicity
    threshold: 0.1  # Max acceptable toxicity score
```

### safety/bias

Detects biased content.

```yaml
evaluators:
  - type: bias
    categories:
      - gender
      - race
      - religion
```

### safety/hallucination

Detects hallucinated or fabricated information.

```yaml
evaluators:
  - type: hallucination
    expected:
      context: "The meeting is scheduled for Tuesday at 3pm."
```

## Agent Evaluators

### agent/tool_selection

Evaluates whether the agent selected appropriate tools.

```yaml
evaluators:
  - type: agent/tool_selection
    expected:
      required_tools:
        - search
        - calculate
      forbidden_tools:
        - delete
```

### agent/efficiency

Evaluates agent efficiency (steps, token usage).

```yaml
evaluators:
  - type: agent/efficiency
    expected:
      max_steps: 5
      max_tokens: 2000
```

## RAG Evaluators

### rag/relevancy

Evaluates retrieval relevancy.

```yaml
evaluators:
  - type: rag/relevancy
    expected:
      query: "What is the return policy?"
      threshold: 0.7
```

### rag/faithfulness

Evaluates faithfulness to retrieved context.

```yaml
evaluators:
  - type: rag/faithfulness
    expected:
      context: "Returns accepted within 30 days."
```

### rag/precision

Evaluates precision of retrieved information.

```yaml
evaluators:
  - type: rag/precision
```

## Combining Evaluators

You can use multiple evaluators on a single test case. All evaluators must pass for the case to pass.

```yaml
cases:
  - id: comprehensive-test
    input:
      question: "Summarize the document"
    evaluators:
      # Basic validation
      - type: not_contains
        expected:
          - "I don't know"
          - "I cannot"

      # Length check via regex
      - type: regex
        pattern: "^.{100,500}$"  # Between 100-500 chars

      # Quality check
      - type: llm_judge
        criteria: |
          Summary should:
          - Capture main points
          - Be coherent and readable
          - Not include irrelevant information
```

## Custom Score Thresholds

Many evaluators support custom pass thresholds:

```yaml
evaluators:
  - type: llm_judge
    criteria: "High quality response"
    score_threshold: 0.9  # Stricter than default 0.6

  - type: similarity
    expected:
      reference: "Expected answer"
      threshold: 0.95  # Very high similarity required
```
