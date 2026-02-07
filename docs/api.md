# API Reference

AI Eval exposes a REST API under the `/api` prefix.

## Starting the Server

The server listens on `:8080` by default and supports:
- `-addr` (default `:8080`)
- `-config` (default `configs/config.yaml`)

For local development, explicitly disable API auth:

```bash
AI_EVAL_DISABLE_AUTH=true go run ./cmd/server -addr :8080
```

Or build a standalone binary:

```bash
go build -o eval-server ./cmd/server
AI_EVAL_DISABLE_AUTH=true ./eval-server -addr :8080
```

Note: the API reads/writes `prompts/` and `tests/` relative to the process working directory.

## Authentication

The API server **requires** one of the following at startup:

- Set `AI_EVAL_API_KEY` to enable API key auth (send `X-API-Key: <key>` on requests)
- Or set `AI_EVAL_DISABLE_AUTH=true` to explicitly disable authentication (recommended only for local/trusted environments)

If neither is set, the server refuses to start.

Unauthorized requests return:

```json
{"error":"unauthorized"}
```

## CORS

CORS headers are **disabled by default**. Enable them by setting `AI_EVAL_CORS_ORIGINS`:

- Allow any origin: `AI_EVAL_CORS_ORIGINS="*"`
- Allow a fixed list (comma-separated): `AI_EVAL_CORS_ORIGINS="http://localhost:3000, https://example.com"`

When enabled, the server allows:
- Methods: `GET,POST,DELETE,OPTIONS`
- Headers: `Content-Type, X-API-Key`

## Error Responses

All API errors use this format:

```json
{"error":"<message>"}
```

## JSON Field Naming

Some endpoints return Go structs without explicit `json` tags, so response keys are exported Go field names
(e.g. `Name`, `Suite`, `StartedAt`).

## Endpoints

### Health

```
GET /api/health
```

**Response:**

```json
{"status":"ok","time":"2026-02-07T05:08:34Z"}
```

### Prompts

List prompts:

```
GET /api/prompts
```

Optional query parameters:
- `name`: case-insensitive exact match filter

Get one prompt:

```
GET /api/prompts/{name}
```

Upsert a prompt (writes `prompts/<name>.yaml`):

```
POST /api/prompts
```

**Request Body (keys are case-insensitive):**

```json
{
  "name": "example",
  "version": "v1",
  "description": "example prompt",
  "template": "hello"
}
```

Delete a prompt:

```
DELETE /api/prompts/{name}
```

Returns `204 No Content` on success.

### Tests

List test suites:

```
GET /api/tests
```

Optional query parameters:
- `prompt`: case-insensitive exact match filter

Get one suite:

```
GET /api/tests/{suite}
```

Responses are `TestSuite` objects (including `Cases`).

### Runs

Start a run (runs suites from `tests/` against prompts in `prompts/`):

```
POST /api/runs
```

**Request Body:**

```json
{"prompt":"example","trials":3,"threshold":0.8,"concurrency":4}
```

Or run all prompts:

```json
{"all":true}
```

**Response:** `201 Created`

```json
{
  "run": {
    "ID": "run_20260207T050834Z_deadbeef...",
    "StartedAt": "2026-02-07T05:08:34Z",
    "FinishedAt": "2026-02-07T05:08:40Z",
    "TotalSuites": 1,
    "PassedSuites": 1,
    "FailedSuites": 0,
    "Config": {"trials":3,"threshold":0.8,"concurrency":4,"all":false,"prompts":["example"]}
  },
  "summary": {
    "total_suites": 1,
    "total_cases": 10,
    "passed_cases": 10,
    "failed_cases": 0,
    "total_latency_ms": 12345,
    "total_tokens": 6789
  }
}
```

List runs:

```
GET /api/runs
```

Query parameters:
- `limit` (default `20`, must be `> 0`)
- `since` / `until`: RFC3339 or `YYYY-MM-DD`
- `prompt` / `version`: filters

Get one run:

```
GET /api/runs/{id}
```

Get suite results for a run:

```
GET /api/runs/{id}/results
```

### History

```
GET /api/history/{prompt}
```

Optional query parameters:
- `limit` (default `20`, must be `> 0`)

Returns a list of `SuiteRecord` entries.

### Compare

```
POST /api/compare
```

**Request Body:**

```json
{"prompt":"example","v1":"v1","v2":"v2"}
```

Returns a `VersionComparison` object.

### Leaderboard

Get leaderboard entries:

```
GET /api/leaderboard?dataset=mmlu&limit=20
```

Query parameters:
- `dataset` (required)
- `limit` (optional, default `20`, max `100`)

Returns a JSON array of entries.

Get model history:

```
GET /api/leaderboard/history?model=m1&dataset=gsm8k
```

### Optimize

```
POST /api/optimize
```

**Request Body:**

```json
{"prompt_content":"...","prompt_name":"example","num_cases":5}
```

Returns fields like `analysis`, `suggestions`, `eval_results`, `optimized_prompt`, `changes`,
and `optimization_summary`.

### Diagnose

```
POST /api/diagnose
```

**Request Body:**

```json
{"prompt_content":"...","tests_yaml":"<YAML>","max_suggestions":5}
```

Returns `suites` (summary) and `diagnosis` (details).

## Example: cURL

```bash
# No auth (only if AI_EVAL_DISABLE_AUTH=true)
curl "http://localhost:8080/api/health"

# API key auth (only if AI_EVAL_API_KEY is set on the server)
curl "http://localhost:8080/api/health" -H "X-API-Key: $AI_EVAL_API_KEY"

# Start a run
curl -X POST "http://localhost:8080/api/runs" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $AI_EVAL_API_KEY" \
  -d '{"prompt":"example","trials":3,"threshold":0.8,"concurrency":4}'
```
