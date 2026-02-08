package llm

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/stellarlinkco/ai-eval/internal/claude"
)

func TestToClaudeRequest(t *testing.T) {
	t.Parallel()

	if _, err := toClaudeRequest(nil); err == nil {
		t.Fatalf("toClaudeRequest(nil): expected error")
	}

	got, err := toClaudeRequest(&Request{
		Messages: []Message{
			{Role: " ", Content: "a"},
			{Role: "assistant", Content: "b"},
		},
		System:      "sys",
		MaxTokens:   7,
		Temperature: 0.5,
		Tools: []ToolDefinition{
			{Name: " ", Description: "ignored"},
			{Name: " t1 ", Description: " d1 ", InputSchema: nil},
		},
	})
	if err != nil {
		t.Fatalf("toClaudeRequest: %v", err)
	}
	if got == nil {
		t.Fatalf("toClaudeRequest: nil request")
	}
	if len(got.Messages) != 2 {
		t.Fatalf("len(Messages): got %d want %d", len(got.Messages), 2)
	}
	if got.Messages[0].Role != "user" || got.Messages[0].Content != "a" {
		t.Fatalf("Messages[0]: %#v", got.Messages[0])
	}
	if got.Messages[1].Role != "assistant" || got.Messages[1].Content != "b" {
		t.Fatalf("Messages[1]: %#v", got.Messages[1])
	}
	if got.System != "sys" || got.MaxTokens != 7 || got.Temperature != 0.5 {
		t.Fatalf("fields: %#v", got)
	}
	if len(got.Tools) != 1 {
		t.Fatalf("len(Tools): got %d want %d", len(got.Tools), 1)
	}
	if got.Tools[0].Name != "t1" || got.Tools[0].Description != "d1" {
		t.Fatalf("Tools[0]: %#v", got.Tools[0])
	}
	if got.Tools[0].InputSchema == nil {
		t.Fatalf("Tools[0].InputSchema: nil")
	}
}

func TestFromClaudeResponse(t *testing.T) {
	t.Parallel()

	if got := fromClaudeResponse(nil); got != nil {
		t.Fatalf("fromClaudeResponse(nil): got %#v", got)
	}

	out := fromClaudeResponse(&claude.Response{
		StopReason: "end_turn",
		Usage:      claude.Usage{InputTokens: 1, OutputTokens: 2},
		Content: []claude.ContentBlock{
			{Type: "text", Text: "a"},
			{Type: "tool_use", ID: "id", Name: "t", Input: map[string]any{"k": "v"}},
			{Type: "tool_result", ToolUseID: "ignored"},
		},
	})
	if out == nil {
		t.Fatalf("fromClaudeResponse: nil")
	}
	if out.StopReason != "end_turn" {
		t.Fatalf("StopReason: got %q", out.StopReason)
	}
	if out.Usage.InputTokens != 1 || out.Usage.OutputTokens != 2 {
		t.Fatalf("Usage: %#v", out.Usage)
	}
	if len(out.Content) != 2 {
		t.Fatalf("len(Content): got %d want %d", len(out.Content), 2)
	}
	if out.Content[0].Type != "text" || out.Content[0].Text != "a" {
		t.Fatalf("Content[0]: %#v", out.Content[0])
	}
	if out.Content[1].Type != "tool_use" || out.Content[1].Name != "t" {
		t.Fatalf("Content[1]: %#v", out.Content[1])
	}
}

func TestFromClaudeEvalResult(t *testing.T) {
	t.Parallel()

	if got := fromClaudeEvalResult(nil); got != nil {
		t.Fatalf("fromClaudeEvalResult(nil): got %#v", got)
	}

	out := fromClaudeEvalResult(&claude.EvalResult{
		Response:     &claude.Response{StopReason: "end_turn"},
		TextContent:  "x",
		LatencyMs:    3,
		InputTokens:  4,
		OutputTokens: 5,
		Error:        errors.New("e"),
		ToolCalls: []claude.ToolUse{
			{ID: "id", Name: "t", Input: map[string]any{"k": "v"}},
		},
	})
	if out == nil || out.Response == nil {
		t.Fatalf("fromClaudeEvalResult: nil result/response")
	}
	if out.TextContent != "x" || out.LatencyMs != 3 || out.InputTokens != 4 || out.OutputTokens != 5 {
		t.Fatalf("fields: %#v", out)
	}
	if out.Error == nil || out.Error.Error() != "e" {
		t.Fatalf("Error: %#v", out.Error)
	}
	if len(out.ToolCalls) != 1 || out.ToolCalls[0].Name != "t" {
		t.Fatalf("ToolCalls: %#v", out.ToolCalls)
	}
}

func TestFromClaudeMultiTurnResult(t *testing.T) {
	t.Parallel()

	if got := fromClaudeMultiTurnResult(nil); got != nil {
		t.Fatalf("fromClaudeMultiTurnResult(nil): got %#v", got)
	}

	out := fromClaudeMultiTurnResult(&claude.MultiTurnResult{
		FinalResponse:     &claude.Response{StopReason: "end_turn"},
		TotalLatencyMs:    1,
		TotalInputTokens:  2,
		TotalOutputTokens: 3,
		Steps:             2,
		AllToolCalls:      []claude.ToolUse{{ID: "id", Name: "t", Input: map[string]any{"k": "v"}}},
		AllResponses:      []*claude.Response{{StopReason: "a"}, {StopReason: "b"}},
	})
	if out == nil || out.FinalResponse == nil {
		t.Fatalf("fromClaudeMultiTurnResult: nil result/final response")
	}
	if out.Steps != 2 || out.TotalInputTokens != 2 || out.TotalOutputTokens != 3 {
		t.Fatalf("fields: %#v", out)
	}
	if len(out.AllToolCalls) != 1 || out.AllToolCalls[0].Name != "t" {
		t.Fatalf("AllToolCalls: %#v", out.AllToolCalls)
	}
	if len(out.AllResponses) != 2 || out.AllResponses[0].StopReason != "a" {
		t.Fatalf("AllResponses: %#v", out.AllResponses)
	}
}

func TestClaudeProvider_CompleteAndWithTools(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/messages" {
			http.NotFound(w, r)
			return
		}
		_ = r.Body.Close()

		w.Header().Set("content-type", "application/json")
		_ = json.NewEncoder(w).Encode(claudeMessageResponse(
			"msg_1",
			"test-model",
			"end_turn",
			[]map[string]any{
				claudeTextBlock("a"),
				claudeToolUseBlock("toolu_1", "search", map[string]any{"q": "x"}),
				claudeTextBlock("b"),
			},
			1,
			2,
		))
	}))
	t.Cleanup(srv.Close)

	p := NewClaudeProvider("k", srv.URL+"/v1", "m")
	resp, err := p.Complete(context.Background(), &Request{
		Messages:  []Message{{Role: "user", Content: "hi"}},
		MaxTokens: 7,
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if resp == nil {
		t.Fatalf("Complete: nil response")
	}
	if Text(resp) != "ab" {
		t.Fatalf("Text(resp): got %q want %q", Text(resp), "ab")
	}
	if len(resp.Content) != 3 {
		t.Fatalf("len(Content): got %d want %d", len(resp.Content), 3)
	}
	if resp.Content[1].Type != "tool_use" || resp.Content[1].Name != "search" {
		t.Fatalf("Content[1]: %#v", resp.Content[1])
	}

	res, err := p.CompleteWithTools(context.Background(), &Request{
		Messages:  []Message{{Role: "user", Content: "hi"}},
		MaxTokens: 7,
	})
	if err != nil {
		t.Fatalf("CompleteWithTools: %v", err)
	}
	if res == nil || res.Response == nil {
		t.Fatalf("CompleteWithTools: nil result/response")
	}
	if res.TextContent != "ab" {
		t.Fatalf("TextContent: got %q want %q", res.TextContent, "ab")
	}
	if len(res.ToolCalls) != 1 || res.ToolCalls[0].Name != "search" {
		t.Fatalf("ToolCalls: %#v", res.ToolCalls)
	}

	var pnil *ClaudeProvider
	if _, err := pnil.Complete(context.Background(), &Request{}); err == nil {
		t.Fatalf("Complete(nil provider): expected error")
	}
}

func TestClaudeProvider_CompleteMultiTurn(t *testing.T) {
	t.Parallel()

	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/messages" {
			http.NotFound(w, r)
			return
		}
		_ = r.Body.Close()
		n := atomic.AddInt32(&calls, 1)

		w.Header().Set("content-type", "application/json")
		switch n {
		case 1:
			_ = json.NewEncoder(w).Encode(claudeMessageResponse(
				"msg_1",
				"test-model",
				"tool_use",
				[]map[string]any{
					claudeToolUseBlock("toolu_1", "git", map[string]any{"cmd": "status"}),
				},
				1,
				1,
			))
		default:
			_ = json.NewEncoder(w).Encode(claudeMessageResponse(
				"msg_2",
				"test-model",
				"end_turn",
				[]map[string]any{claudeTextBlock("done")},
				2,
				3,
			))
		}
	}))
	t.Cleanup(srv.Close)

	p := NewClaudeProvider("k", srv.URL+"/v1", "m")
	out, err := p.CompleteMultiTurn(context.Background(), &Request{
		Messages:  []Message{{Role: "user", Content: "hi"}},
		MaxTokens: 7,
	}, func(tu ToolUse) (string, error) {
		if tu.ID != "toolu_1" || tu.Name != "git" || tu.Input["cmd"] != "status" {
			return "", errors.New("bad tool use")
		}
		return "ok", nil
	}, 2)
	if err != nil {
		t.Fatalf("CompleteMultiTurn: %v", err)
	}
	if out == nil || out.FinalResponse == nil {
		t.Fatalf("CompleteMultiTurn: nil result/final response")
	}
	if out.Steps != 2 {
		t.Fatalf("Steps: got %d want %d", out.Steps, 2)
	}
	if len(out.AllToolCalls) != 1 || out.AllToolCalls[0].Name != "git" {
		t.Fatalf("AllToolCalls: %#v", out.AllToolCalls)
	}

	atomic.StoreInt32(&calls, 0)
	_, err = p.CompleteMultiTurn(context.Background(), &Request{
		Messages: []Message{{Role: "user", Content: "hi"}},
	}, nil, 1)
	if err == nil || !strings.Contains(err.Error(), "nil tool executor") {
		t.Fatalf("CompleteMultiTurn(nil executor): got %v", err)
	}
}

func TestClaudeProvider_ErrorBranches(t *testing.T) {
	t.Parallel()

	p := NewClaudeProvider("k", " ", " ")

	if _, err := p.Complete(context.Background(), nil); err == nil || !strings.Contains(err.Error(), "nil request") {
		t.Fatalf("Complete(nil req): %v", err)
	}
	if _, err := p.CompleteWithTools(context.Background(), nil); err == nil || !strings.Contains(err.Error(), "nil request") {
		t.Fatalf("CompleteWithTools(nil req): %v", err)
	}
	if _, err := p.CompleteMultiTurn(context.Background(), nil, nil, 1); err == nil || !strings.Contains(err.Error(), "nil request") {
		t.Fatalf("CompleteMultiTurn(nil req): %v", err)
	}

	var pnil *ClaudeProvider
	if _, err := pnil.CompleteWithTools(context.Background(), &Request{}); err == nil || !strings.Contains(err.Error(), "nil client") {
		t.Fatalf("CompleteWithTools(nil provider): %v", err)
	}
	if _, err := pnil.CompleteMultiTurn(context.Background(), &Request{}, nil, 1); err == nil || !strings.Contains(err.Error(), "nil client") {
		t.Fatalf("CompleteMultiTurn(nil provider): %v", err)
	}
}

func claudeMessageResponse(id, model, stopReason string, content []map[string]any, inTok, outTok int) map[string]any {
	return map[string]any{
		"id":            id,
		"type":          "message",
		"role":          "assistant",
		"content":       content,
		"model":         model,
		"stop_reason":   stopReason,
		"stop_sequence": "",
		"usage": map[string]any{
			"input_tokens":                inTok,
			"output_tokens":               outTok,
			"cache_creation":              map[string]any{"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 0},
			"cache_creation_input_tokens": 0,
			"cache_read_input_tokens":     0,
			"server_tool_use":             map[string]any{"web_search_requests": 0},
			"service_tier":                "standard",
		},
	}
}

func claudeTextBlock(text string) map[string]any {
	return map[string]any{
		"type": "text",
		"text": text,
	}
}

func claudeToolUseBlock(id, name string, input map[string]any) map[string]any {
	return map[string]any{
		"type":  "tool_use",
		"id":    id,
		"name":  name,
		"input": input,
	}
}
