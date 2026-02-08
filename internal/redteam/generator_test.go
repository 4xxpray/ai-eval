package redteam

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/stellarlinkco/ai-eval/internal/llm"
)

type mockProvider struct {
	name     string
	response *llm.Response
	err      error
}

func (m *mockProvider) Name() string { return m.name }
func (m *mockProvider) Complete(_ context.Context, _ *llm.Request) (*llm.Response, error) {
	return m.response, m.err
}
func (m *mockProvider) CompleteWithTools(_ context.Context, _ *llm.Request) (*llm.EvalResult, error) {
	return nil, nil
}

func textResponse(text string) *llm.Response {
	return &llm.Response{
		Content: []llm.ContentBlock{{Type: "text", Text: text}},
	}
}

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

func TestGenerator_Generate_Errors(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		g         *Generator
		ctx       context.Context
		template  string
		cats      []Category
		wantError string
	}{
		{
			name:      "nil generator",
			g:         nil,
			ctx:       context.Background(),
			template:  "system",
			cats:      []Category{CategoryJailbreak},
			wantError: "redteam: nil generator",
		},
		{
			name:      "nil context",
			g:         &Generator{Provider: &mockProvider{name: "m", response: textResponse(`{"cases":[{"attack":"a"}]}`)}},
			ctx:       nil,
			template:  "system",
			cats:      []Category{CategoryJailbreak},
			wantError: "redteam: nil context",
		},
		{
			name:      "nil provider",
			g:         &Generator{Provider: nil},
			ctx:       context.Background(),
			template:  "system",
			cats:      []Category{CategoryJailbreak},
			wantError: "redteam: nil llm provider",
		},
		{
			name:      "empty template",
			g:         &Generator{Provider: &mockProvider{name: "m"}},
			ctx:       context.Background(),
			template:  "   ",
			cats:      []Category{CategoryJailbreak},
			wantError: "redteam: empty prompt template",
		},
		{
			name:      "unknown category",
			g:         &Generator{Provider: &mockProvider{name: "m"}},
			ctx:       context.Background(),
			template:  "system",
			cats:      []Category{"nope"},
			wantError: "unknown category",
		},
		{
			name:      "no categories after normalization",
			g:         &Generator{Provider: &mockProvider{name: "m"}},
			ctx:       context.Background(),
			template:  "system",
			cats:      []Category{""},
			wantError: "redteam: no categories",
		},
		{
			name:      "provider error",
			g:         &Generator{Provider: &mockProvider{name: "m", err: errors.New("boom")}},
			ctx:       context.Background(),
			template:  "system",
			cats:      []Category{CategoryJailbreak},
			wantError: "redteam: generate: llm: boom",
		},
		{
			name:      "nil llm response",
			g:         &Generator{Provider: &mockProvider{name: "m", response: nil}},
			ctx:       context.Background(),
			template:  "system",
			cats:      []Category{CategoryJailbreak},
			wantError: "redteam: generate: nil llm response",
		},
		{
			name:      "parse output error",
			g:         &Generator{Provider: &mockProvider{name: "m", response: textResponse("not json")}},
			ctx:       context.Background(),
			template:  "system",
			cats:      []Category{CategoryJailbreak},
			wantError: "redteam: generate: parse output",
		},
		{
			name:      "no cases returned",
			g:         &Generator{Provider: &mockProvider{name: "m", response: textResponse(`{"cases":[]}`)}},
			ctx:       context.Background(),
			template:  "system",
			cats:      []Category{CategoryJailbreak},
			wantError: "redteam: generate: no cases returned",
		},
		{
			name:      "all cases empty",
			g:         &Generator{Provider: &mockProvider{name: "m", response: textResponse(`{"cases":[{"id":"x","category":"jailbreak","attack":"   "}]}`)}},
			ctx:       context.Background(),
			template:  "system",
			cats:      []Category{CategoryJailbreak},
			wantError: "redteam: generate: all cases empty",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			_, err := tt.g.Generate(tt.ctx, tt.template, tt.cats)
			if err == nil {
				t.Fatalf("Generate: expected error")
			}
			if tt.wantError != "" && err.Error() != tt.wantError && !contains(err.Error(), tt.wantError) {
				t.Fatalf("Generate: got %v want %q", err, tt.wantError)
			}
		})
	}
}

func TestGenerator_Generate_IDFallbackAndDedup(t *testing.T) {
	t.Parallel()

	g := &Generator{
		Provider: &mockProvider{
			name: "m",
			response: textResponse(`{
				"cases": [
					{"id":"!!!","category":"","attack":"a"},
					{"id":"A","category":"jailbreak","attack":"b"},
					{"id":"A","category":"jailbreak","attack":"c"}
				]
			}`),
		},
	}

	cases, err := g.Generate(context.Background(), "system", []Category{CategoryJailbreak})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if len(cases) != 3 {
		t.Fatalf("len(cases): got %d want %d", len(cases), 3)
	}
	if cases[0].ID != "jailbreak_01" {
		t.Fatalf("cases[0].ID: got %q want %q", cases[0].ID, "jailbreak_01")
	}
	if cases[1].ID != "a" || cases[2].ID != "a_2" {
		t.Fatalf("dedup: got %q, %q", cases[1].ID, cases[2].ID)
	}
}

func TestNormalizeCategories_Default(t *testing.T) {
	t.Parallel()

	got, err := normalizeCategories(nil)
	if err != nil {
		t.Fatalf("normalizeCategories: %v", err)
	}
	want := []Category{CategoryJailbreak, CategoryInjection, CategoryPII}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("normalizeCategories: got %v want %v", got, want)
	}
}

func TestSanitizeCaseID(t *testing.T) {
	t.Parallel()

	tests := []struct {
		in   string
		want string
	}{
		{in: "", want: ""},
		{in: " Case 1 ", want: "case1"},
		{in: "__a--b__", want: "a_b"},
		{in: "!!!", want: ""},
	}

	for _, tt := range tests {
		if got := sanitizeCaseID(tt.in); got != tt.want {
			t.Fatalf("sanitizeCaseID(%q): got %q want %q", tt.in, got, tt.want)
		}
	}
}

func contains(s, sub string) bool {
	for i := 0; i <= len(s)-len(sub); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
