package evaluator

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stellarlinkco/ai-eval/internal/llm"
)

func TestLLMJudgeEvaluator_Evaluate(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/messages" {
			http.Error(w, "bad path", http.StatusBadRequest)
			return
		}
		defer r.Body.Close()

		b, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, "read", http.StatusBadRequest)
			return
		}

		var req map[string]any
		if err := json.Unmarshal(b, &req); err != nil {
			http.Error(w, "json", http.StatusBadRequest)
			return
		}
		msgs, _ := req["messages"].([]any)
		if len(msgs) != 1 {
			http.Error(w, "messages", http.StatusBadRequest)
			return
		}
		m0, _ := msgs[0].(map[string]any)
		m0c, _ := m0["content"].([]any)
		if len(m0c) != 1 {
			http.Error(w, "messages", http.StatusBadRequest)
			return
		}
		b0, _ := m0c[0].(map[string]any)
		content, _ := b0["text"].(string)
		if !strings.Contains(content, "## Evaluation Criteria\nBe strict.") {
			http.Error(w, "criteria missing", http.StatusBadRequest)
			return
		}
		if !strings.Contains(content, "## Original Question/Context\nQ") {
			http.Error(w, "context missing", http.StatusBadRequest)
			return
		}
		if !strings.Contains(content, "## Scoring Dimensions\n- Correctness\n- Clarity\n") {
			http.Error(w, "rubric missing", http.StatusBadRequest)
			return
		}
		if !strings.Contains(content, "Rate the response on a scale of 1-5.") {
			http.Error(w, "scale missing", http.StatusBadRequest)
			return
		}

		w.Header().Set("content-type", "application/json")
		_ = json.NewEncoder(w).Encode(messageResponse(
			"msg_1",
			req["model"].(string),
			"end_turn",
			[]map[string]any{textBlock(`{"score": 4, "reasoning": "Solid overall."}`)},
			1,
			1,
		))
	}))
	t.Cleanup(srv.Close)

	provider := llm.NewClaudeProvider("k", srv.URL+"/v1", "")
	e := &LLMJudgeEvaluator{Provider: provider}

	res, err := e.Evaluate(context.Background(), "A", map[string]any{
		"criteria":    "Be strict.",
		"context":     "Q",
		"rubric":      []string{"Correctness", "Clarity"},
		"score_scale": 5,
	})
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if res == nil {
		t.Fatalf("Evaluate: nil result")
	}
	if !res.Passed {
		t.Fatalf("Passed: got false want true (score=%v)", res.Score)
	}
	if res.Score != 0.75 {
		t.Fatalf("Score: got %v want %v", res.Score, 0.75)
	}
	if !strings.Contains(res.Message, "Solid") {
		t.Fatalf("Message: got %q", res.Message)
	}
}

func TestLLMJudgeEvaluator_InvalidJSON(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("content-type", "application/json")
		_ = json.NewEncoder(w).Encode(messageResponse(
			"msg_1",
			"m",
			"end_turn",
			[]map[string]any{textBlock("not json")},
			1,
			1,
		))
	}))
	t.Cleanup(srv.Close)

	provider := llm.NewClaudeProvider("k", srv.URL+"/v1", "")
	e := &LLMJudgeEvaluator{Provider: provider}

	res, err := e.Evaluate(context.Background(), "A", map[string]any{
		"criteria":    "Be strict.",
		"context":     "Q",
		"score_scale": 5,
	})
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if res == nil {
		t.Fatalf("Evaluate: nil result")
	}
	if res.Passed || res.Score != 0.0 {
		t.Fatalf("got passed=%v score=%v want false/0.0", res.Passed, res.Score)
	}
	if res.Message == "" {
		t.Fatalf("Message: empty")
	}
}

func messageResponse(id, model, stopReason string, content []map[string]any, inputTokens, outputTokens int) map[string]any {
	return map[string]any{
		"id":            id,
		"type":          "message",
		"role":          "assistant",
		"content":       content,
		"model":         model,
		"stop_reason":   stopReason,
		"stop_sequence": "",
		"usage": map[string]any{
			"input_tokens":                inputTokens,
			"output_tokens":               outputTokens,
			"cache_creation":              map[string]any{"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 0},
			"cache_creation_input_tokens": 0,
			"cache_read_input_tokens":     0,
			"server_tool_use":             map[string]any{"web_search_requests": 0},
			"service_tier":                "standard",
		},
	}
}

func textBlock(text string) map[string]any {
	return map[string]any{
		"type": "text",
		"text": text,
	}
}
