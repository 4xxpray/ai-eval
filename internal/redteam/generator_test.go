package redteam

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stellarlinkco/ai-eval/internal/llm"
)

func TestGenerator_Generate(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = r.Body.Close()

		payload := "```json\n{" +
			"\"cases\":[" +
			"{\"id\":\"Case 1\",\"category\":\"\",\"attack\":\" attack1 \",\"description\":\"\"}," +
			"{\"id\":\"\",\"category\":\"HARMFUL\",\"attack\":\"attack2\",\"description\":\"desc\"}" +
			"]}" +
			"\n```"

		w.Header().Set("content-type", "application/json")
		_ = json.NewEncoder(w).Encode(messageResponse(
			"msg_1",
			"default",
			"end_turn",
			[]map[string]any{textBlock(payload)},
			1,
			1,
		))
	}))
	t.Cleanup(srv.Close)

	provider := llm.NewClaudeProvider("k", srv.URL+"/v1", "")
	g := &Generator{Provider: provider}

	cases, err := g.Generate(context.Background(), "system", []Category{CategoryJailbreak, CategoryHarmful})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if len(cases) != 2 {
		t.Fatalf("len(cases): got %d want %d", len(cases), 2)
	}

	first := cases[0]
	if first.ID != "case1" {
		t.Fatalf("cases[0].ID: got %q want %q", first.ID, "case1")
	}
	if first.Description != "redteam category=jailbreak" {
		t.Fatalf("cases[0].Description: got %q", first.Description)
	}
	if got := first.Input["attack"]; got != "attack1" {
		t.Fatalf("cases[0].Input.attack: got %v want %q", got, "attack1")
	}
	if got := first.Input["category"]; got != "jailbreak" {
		t.Fatalf("cases[0].Input.category: got %v want %q", got, "jailbreak")
	}

	second := cases[1]
	if second.ID != "harmful_02" {
		t.Fatalf("cases[1].ID: got %q want %q", second.ID, "harmful_02")
	}
	if second.Description != "desc" {
		t.Fatalf("cases[1].Description: got %q want %q", second.Description, "desc")
	}
	if got := second.Input["attack"]; got != "attack2" {
		t.Fatalf("cases[1].Input.attack: got %v want %q", got, "attack2")
	}
	if got := second.Input["category"]; got != "harmful" {
		t.Fatalf("cases[1].Input.category: got %v want %q", got, "harmful")
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
