# API Reference

AI Eval provides a REST API for programmatic access to evaluation features.

## Starting the Server

```bash
go run ./cmd/server
# or
./eval-server
```

Default port: `8080`

## Endpoints

### Health Check

```
GET /health
```

**Response:**

```json
{"status": "ok"}
```

### Run Evaluation

```
POST /api/v1/evaluate
```

**Request Body:**

```json
{
  "prompt": "example",
  "input": {
    "user_request": "Hello world"
  },
  "evaluators": [
    {
      "type": "contains",
      "expected": "hello"
    },
    {
      "type": "llm_judge",
      "criteria": "Response should be friendly"
    }
  ]
}
```

**Response:**

```json
{
  "passed": true,
  "score": 1.0,
  "response": "Hello! How can I help you today?",
  "evaluations": [
    {
      "type": "contains",
      "passed": true,
      "score": 1.0,
      "message": "Found expected substring"
    },
    {
      "type": "llm_judge",
      "passed": true,
      "score": 0.95,
      "message": "Response is friendly and helpful"
    }
  ],
  "latency_ms": 1234,
  "tokens_used": 150
}
```

### Run Test Suite

```
POST /api/v1/suite
```

**Request Body:**

```json
{
  "prompt": "example",
  "suite": "example-tests",
  "trials": 3
}
```

**Response:**

```json
{
  "suite": "example-tests",
  "total_cases": 5,
  "passed_cases": 4,
  "failed_cases": 1,
  "pass_rate": 0.8,
  "avg_score": 0.85,
  "results": [
    {
      "case_id": "test-1",
      "passed": true,
      "score": 1.0,
      "pass_at_k": 1.0
    }
  ]
}
```

### List Prompts

```
GET /api/v1/prompts
```

**Response:**

```json
{
  "prompts": [
    {
      "name": "example",
      "version": "1.0",
      "is_system_prompt": true
    },
    {
      "name": "code-review",
      "version": "2.1",
      "is_system_prompt": true
    }
  ]
}
```

### Get Prompt

```
GET /api/v1/prompts/{name}
```

**Response:**

```json
{
  "name": "example",
  "version": "1.0",
  "is_system_prompt": true,
  "template": "You are a helpful assistant...\n\nUser: {{.user_request}}"
}
```

### List Test Suites

```
GET /api/v1/suites
```

**Response:**

```json
{
  "suites": [
    {
      "name": "example-tests",
      "prompt": "example",
      "case_count": 5
    }
  ]
}
```

### Leaderboard

```
GET /api/v1/leaderboard?dataset=mmlu&top=10
```

**Query Parameters:**
- `dataset` (required): Benchmark dataset name
- `top` (optional, default: 20): Number of entries

**Response:**

```json
{
  "dataset": "mmlu",
  "entries": [
    {
      "rank": 1,
      "model": "claude-sonnet-4-5-20250929",
      "provider": "claude",
      "score": 1.0,
      "accuracy": 1.0,
      "latency_ms": 14444,
      "date": "2026-02-07T05:08:34Z"
    }
  ]
}
```

### Run Benchmark

```
POST /api/v1/benchmark
```

**Request Body:**

```json
{
  "dataset": "mmlu",
  "provider": "claude",
  "model": "claude-sonnet-4-5-20250929",
  "sample_size": 100
}
```

**Response:**

```json
{
  "id": 1,
  "dataset": "mmlu",
  "provider": "claude",
  "model": "claude-sonnet-4-5-20250929",
  "score": 0.98,
  "accuracy": 0.98,
  "total_time_ms": 145678,
  "total_tokens": 12345
}
```

### Optimize Prompt

```
POST /api/v1/optimize
```

**Request Body:**

```json
{
  "prompt": "example",
  "iterations": 3,
  "strategy": "iterative"
}
```

**Response:**

```json
{
  "original_score": 0.75,
  "optimized_score": 0.92,
  "iterations": 3,
  "suggestions": [
    "Add explicit output format instructions",
    "Include example responses"
  ],
  "optimized_template": "..."
}
```

## Error Responses

All endpoints return errors in this format:

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "Missing required field: prompt"
  }
}
```

**Error Codes:**
- `INVALID_REQUEST`: Bad request parameters
- `NOT_FOUND`: Resource not found
- `PROVIDER_ERROR`: LLM provider error
- `TIMEOUT`: Request timeout
- `INTERNAL_ERROR`: Internal server error

## Authentication

The API server does not require authentication by default. For production use, consider:

1. Running behind a reverse proxy with authentication
2. Implementing API key middleware
3. Using network-level access controls

## Rate Limiting

No built-in rate limiting. LLM provider rate limits apply to all requests.

## CORS

CORS is enabled for all origins by default. Configure in production as needed.

## Example: cURL

```bash
# Run single evaluation
curl -X POST http://localhost:8080/api/v1/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "example",
    "input": {"user_request": "Hello"},
    "evaluators": [{"type": "contains", "expected": "hello"}]
  }'

# Get leaderboard
curl "http://localhost:8080/api/v1/leaderboard?dataset=mmlu&top=5"
```

## Example: Python

```python
import requests

base_url = "http://localhost:8080/api/v1"

# Run evaluation
response = requests.post(f"{base_url}/evaluate", json={
    "prompt": "example",
    "input": {"user_request": "Hello"},
    "evaluators": [
        {"type": "contains", "expected": "hello"}
    ]
})
print(response.json())

# Get leaderboard
response = requests.get(f"{base_url}/leaderboard", params={
    "dataset": "mmlu",
    "top": 10
})
print(response.json())
```
