package llm

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	openai "github.com/sashabaranov/go-openai"
)

func TestNormalizeOpenAIRole(t *testing.T) {
	t.Parallel()

	tests := []struct {
		in   string
		want string
	}{
		{"system", openai.ChatMessageRoleSystem},
		{"user", openai.ChatMessageRoleUser},
		{"assistant", openai.ChatMessageRoleAssistant},
		{"tool", openai.ChatMessageRoleTool},
		{"developer", openai.ChatMessageRoleDeveloper},
		{"  USER ", openai.ChatMessageRoleUser},
		{"unknown", openai.ChatMessageRoleUser},
		{"", openai.ChatMessageRoleUser},
	}
	for _, tt := range tests {
		tt := tt
		t.Run(tt.in, func(t *testing.T) {
			t.Parallel()

			if got := normalizeOpenAIRole(tt.in); got != tt.want {
				t.Fatalf("normalizeOpenAIRole(%q): got %q want %q", tt.in, got, tt.want)
			}
		})
	}
}

func TestOpenAIHelpers(t *testing.T) {
	t.Parallel()

	if got := clampMaxTokens(-1); got != 0 {
		t.Fatalf("clampMaxTokens(-1): got %d want %d", got, 0)
	}
	if got := clampMaxTokens(3); got != 3 {
		t.Fatalf("clampMaxTokens(3): got %d want %d", got, 3)
	}

	if got := toOpenAITools(nil); got != nil {
		t.Fatalf("toOpenAITools(nil): expected nil")
	}

	tools := toOpenAITools([]ToolDefinition{
		{Name: " ", Description: "ignored"},
		{Name: " fn ", Description: " d ", InputSchema: nil},
	})
	if len(tools) != 1 {
		t.Fatalf("len(tools): got %d want %d", len(tools), 1)
	}
	if tools[0].Type != openai.ToolTypeFunction {
		t.Fatalf("tools[0].Type: got %q want %q", tools[0].Type, openai.ToolTypeFunction)
	}
	if tools[0].Function == nil || tools[0].Function.Name != "fn" {
		t.Fatalf("tools[0].Function: got %#v", tools[0].Function)
	}
	if tools[0].Function.Description != "d" {
		t.Fatalf("tools[0].Function.Description: got %q want %q", tools[0].Function.Description, "d")
	}
	if tools[0].Function.Parameters == nil {
		t.Fatalf("tools[0].Function.Parameters: nil")
	}

	if got := parseToolArguments(" "); got != nil {
		t.Fatalf("parseToolArguments(empty): got %#v want nil", got)
	}
	if got := parseToolArguments(`{"x":1}`); got["x"] != float64(1) {
		t.Fatalf("parseToolArguments(valid): got %#v", got)
	}
	if got := parseToolArguments("not-json"); got["_raw"] != "not-json" {
		t.Fatalf("parseToolArguments(invalid): got %#v", got)
	}

	if got := toolUsesFromOpenAIMessage(openai.ChatCompletionMessage{}); got != nil {
		t.Fatalf("toolUsesFromOpenAIMessage(empty): got %#v want nil", got)
	}
}

func TestOpenAIToResponse_Nil(t *testing.T) {
	t.Parallel()

	if got := openAIToResponse(nil, nil); got != nil {
		t.Fatalf("openAIToResponse(nil,nil): got %#v", got)
	}
}

func TestOpenAIProvider_Complete_Errors(t *testing.T) {
	t.Parallel()

	var pnil *OpenAIProvider
	if _, err := pnil.Complete(context.Background(), &Request{}); err == nil {
		t.Fatalf("Complete(nil provider): expected error")
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = r.Body.Close()
		w.Header().Set("content-type", "application/json")
		_ = json.NewEncoder(w).Encode(openai.ChatCompletionResponse{
			ID:                "id",
			Object:            "chat.completion",
			Created:           time.Now().Unix(),
			Model:             openai.GPT4o,
			Choices:           nil,
			Usage:             openai.Usage{PromptTokensDetails: &openai.PromptTokensDetails{}, CompletionTokensDetails: &openai.CompletionTokensDetails{}},
			SystemFingerprint: "fp",
		})
	}))
	t.Cleanup(srv.Close)

	p := NewOpenAIProvider("k", srv.URL+"/v1", openai.GPT4o)
	if _, err := p.Complete(nil, &Request{}); err == nil || !strings.Contains(err.Error(), "nil context") {
		t.Fatalf("Complete(nil ctx): got %v", err)
	}
	if _, err := p.Complete(context.Background(), nil); err == nil || !strings.Contains(err.Error(), "nil request") {
		t.Fatalf("Complete(nil req): got %v", err)
	}

	_, err := p.Complete(context.Background(), &Request{Messages: []Message{{Role: "user", Content: "hi"}}})
	if err == nil || !strings.Contains(err.Error(), "empty choices") {
		t.Fatalf("Complete(empty choices): got %v", err)
	}

	srvErr := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = r.Body.Close()
		http.Error(w, "boom", http.StatusInternalServerError)
	}))
	t.Cleanup(srvErr.Close)

	pErr := NewOpenAIProvider("k", srvErr.URL+"/v1", openai.GPT4o)
	if _, err := pErr.Complete(context.Background(), &Request{Messages: []Message{{Role: "user", Content: "hi"}}}); err == nil {
		t.Fatalf("Complete(http err): expected error")
	}
}

func TestOpenAIProvider_Complete_BasicAndToolCalls(t *testing.T) {
	t.Parallel()

	var gotPath string
	var gotBody []byte

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		if r.Body != nil {
			defer r.Body.Close()
		}
		b, _ := io.ReadAll(r.Body)
		gotBody = b

		w.Header().Set("content-type", "application/json")
		_ = json.NewEncoder(w).Encode(openai.ChatCompletionResponse{
			ID:      "chatcmpl_1",
			Object:  "chat.completion",
			Created: time.Now().Unix(),
			Model:   openai.GPT4o,
			Choices: []openai.ChatCompletionChoice{{
				Index:        0,
				FinishReason: openai.FinishReasonStop,
				Message: openai.ChatCompletionMessage{
					Role:    openai.ChatMessageRoleAssistant,
					Content: "hello",
					ToolCalls: []openai.ToolCall{
						{
							ID:   " call_1 ",
							Type: openai.ToolTypeFunction,
							Function: openai.FunctionCall{
								Name:      " tool ",
								Arguments: `{"x":1}`,
							},
						},
						{
							ID:   "call_2",
							Type: openai.ToolTypeFunction,
							Function: openai.FunctionCall{
								Name:      "bad_args",
								Arguments: "not-json",
							},
						},
					},
				},
			}},
			Usage: openai.Usage{
				PromptTokens:            10,
				CompletionTokens:        20,
				TotalTokens:             30,
				PromptTokensDetails:     &openai.PromptTokensDetails{},
				CompletionTokensDetails: &openai.CompletionTokensDetails{},
			},
			SystemFingerprint: "fp",
		})
	}))
	t.Cleanup(srv.Close)

	p := NewOpenAIProvider("k", srv.URL+"/v1", openai.GPT4o)
	resp, err := p.Complete(context.Background(), &Request{
		System:    " sys ",
		MaxTokens: -1,
		Messages: []Message{
			{Role: "unknown", Content: "u"},
			{Role: "assistant", Content: "a"},
		},
		Tools: []ToolDefinition{
			{Name: " fn ", Description: " d "},
		},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if gotPath != "/v1/chat/completions" {
		t.Fatalf("path: got %q want %q", gotPath, "/v1/chat/completions")
	}
	if len(gotBody) == 0 {
		t.Fatalf("request body: empty")
	}

	if resp == nil {
		t.Fatalf("Complete: nil response")
	}
	if resp.StopReason != string(openai.FinishReasonStop) {
		t.Fatalf("StopReason: got %q want %q", resp.StopReason, string(openai.FinishReasonStop))
	}
	if resp.Usage.InputTokens != 10 || resp.Usage.OutputTokens != 20 {
		t.Fatalf("usage: got in=%d out=%d", resp.Usage.InputTokens, resp.Usage.OutputTokens)
	}

	if len(resp.Content) != 3 {
		t.Fatalf("len(Content): got %d want %d", len(resp.Content), 3)
	}
	if resp.Content[0].Type != "text" || resp.Content[0].Text != "hello" {
		t.Fatalf("Content[0]: %#v", resp.Content[0])
	}
	if resp.Content[1].Type != "tool_use" || resp.Content[1].ID != "call_1" || resp.Content[1].Name != "tool" {
		t.Fatalf("Content[1]: %#v", resp.Content[1])
	}
	if resp.Content[1].Input["x"] != float64(1) {
		t.Fatalf("Content[1].Input: %#v", resp.Content[1].Input)
	}
	if resp.Content[2].Type != "tool_use" || resp.Content[2].Name != "bad_args" {
		t.Fatalf("Content[2]: %#v", resp.Content[2])
	}
	if resp.Content[2].Input["_raw"] != "not-json" {
		t.Fatalf("Content[2].Input: %#v", resp.Content[2].Input)
	}
}

func TestOpenAIProvider_CompleteWithTools(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = r.Body.Close()
		w.Header().Set("content-type", "application/json")
		_ = json.NewEncoder(w).Encode(openai.ChatCompletionResponse{
			ID:      "id",
			Object:  "chat.completion",
			Created: time.Now().Unix(),
			Model:   openai.GPT4o,
			Choices: []openai.ChatCompletionChoice{{
				Index:        0,
				FinishReason: openai.FinishReasonStop,
				Message: openai.ChatCompletionMessage{
					Role:    openai.ChatMessageRoleAssistant,
					Content: "a",
					ToolCalls: []openai.ToolCall{
						{
							ID:   "call_1",
							Type: openai.ToolTypeFunction,
							Function: openai.FunctionCall{
								Name:      "t1",
								Arguments: `{"k":"v"}`,
							},
						},
					},
				},
			}},
			Usage: openai.Usage{
				PromptTokens:            1,
				CompletionTokens:        2,
				TotalTokens:             3,
				PromptTokensDetails:     &openai.PromptTokensDetails{},
				CompletionTokensDetails: &openai.CompletionTokensDetails{},
			},
			SystemFingerprint: "fp",
		})
	}))
	t.Cleanup(srv.Close)

	p := NewOpenAIProvider("k", srv.URL+"/v1", openai.GPT4o)
	res, err := p.CompleteWithTools(context.Background(), &Request{
		Messages: []Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("CompleteWithTools: %v", err)
	}
	if res == nil || res.Response == nil {
		t.Fatalf("CompleteWithTools: nil result/response")
	}
	if res.TextContent != "a" {
		t.Fatalf("TextContent: got %q want %q", res.TextContent, "a")
	}
	if len(res.ToolCalls) != 1 || res.ToolCalls[0].Name != "t1" {
		t.Fatalf("ToolCalls: %#v", res.ToolCalls)
	}
	if res.InputTokens != 1 || res.OutputTokens != 2 {
		t.Fatalf("tokens: got in=%d out=%d", res.InputTokens, res.OutputTokens)
	}
	if res.LatencyMs < 0 {
		t.Fatalf("LatencyMs: got %d want >= 0", res.LatencyMs)
	}

	var pnil *OpenAIProvider
	if _, err := pnil.CompleteWithTools(context.Background(), &Request{}); err == nil {
		t.Fatalf("CompleteWithTools(nil provider): expected error")
	}
}

func TestOpenAIProvider_CompleteMultiTurn_Success(t *testing.T) {
	t.Parallel()

	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = r.Body.Close()
		n := atomic.AddInt32(&calls, 1)

		w.Header().Set("content-type", "application/json")
		switch n {
		case 1:
			_ = json.NewEncoder(w).Encode(openai.ChatCompletionResponse{
				ID:      "id1",
				Object:  "chat.completion",
				Created: time.Now().Unix(),
				Model:   openai.GPT4o,
				Choices: []openai.ChatCompletionChoice{{
					Index:        0,
					FinishReason: openai.FinishReasonStop,
					Message: openai.ChatCompletionMessage{
						Role: "", // cover default assistant role fallback
						ToolCalls: []openai.ToolCall{
							{
								ID:   "call_1",
								Type: openai.ToolTypeFunction,
								Function: openai.FunctionCall{
									Name:      "t1",
									Arguments: `{"x":1}`,
								},
							},
						},
					},
				}},
				Usage: openai.Usage{
					PromptTokens:            1,
					CompletionTokens:        1,
					TotalTokens:             2,
					PromptTokensDetails:     &openai.PromptTokensDetails{},
					CompletionTokensDetails: &openai.CompletionTokensDetails{},
				},
				SystemFingerprint: "fp",
			})
		default:
			_ = json.NewEncoder(w).Encode(openai.ChatCompletionResponse{
				ID:      "id2",
				Object:  "chat.completion",
				Created: time.Now().Unix(),
				Model:   openai.GPT4o,
				Choices: []openai.ChatCompletionChoice{{
					Index:        0,
					FinishReason: openai.FinishReasonStop,
					Message: openai.ChatCompletionMessage{
						Role:    openai.ChatMessageRoleAssistant,
						Content: "done",
					},
				}},
				Usage: openai.Usage{
					PromptTokens:            2,
					CompletionTokens:        3,
					TotalTokens:             5,
					PromptTokensDetails:     &openai.PromptTokensDetails{},
					CompletionTokensDetails: &openai.CompletionTokensDetails{},
				},
				SystemFingerprint: "fp",
			})
		}
	}))
	t.Cleanup(srv.Close)

	p := NewOpenAIProvider("k", srv.URL+"/v1", openai.GPT4o)
	out, err := p.CompleteMultiTurn(context.Background(), &Request{
		Messages: []Message{{Role: "user", Content: "hi"}},
		Tools:    []ToolDefinition{{Name: "t1", InputSchema: map[string]any{"type": "object"}}},
	}, func(tu ToolUse) (string, error) {
		if tu.Name != "t1" || tu.ID != "call_1" {
			return "", errors.New("bad tool use")
		}
		return "", errors.New("tool failed")
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
	if len(out.AllToolCalls) != 1 || out.AllToolCalls[0].Name != "t1" {
		t.Fatalf("AllToolCalls: %#v", out.AllToolCalls)
	}
	if len(out.AllResponses) != 2 || Text(out.AllResponses[1]) != "done" {
		t.Fatalf("AllResponses: %#v", out.AllResponses)
	}
	if out.TotalInputTokens != 3 || out.TotalOutputTokens != 4 {
		t.Fatalf("tokens: got in=%d out=%d", out.TotalInputTokens, out.TotalOutputTokens)
	}
}

func TestOpenAIProvider_CompleteMultiTurn_Errors(t *testing.T) {
	t.Parallel()

	p := &OpenAIProvider{}
	if _, err := p.CompleteMultiTurn(context.Background(), &Request{}, func(ToolUse) (string, error) { return "", nil }, 1); err == nil {
		t.Fatalf("CompleteMultiTurn(nil client): expected error")
	}

	p = NewOpenAIProvider("k", "http://example.test/v1", openai.GPT4o)
	if _, err := p.CompleteMultiTurn(nil, &Request{Tools: []ToolDefinition{{Name: "t", InputSchema: map[string]any{"type": "object"}}}}, func(ToolUse) (string, error) { return "", nil }, 1); err == nil || !strings.Contains(err.Error(), "nil context") {
		t.Fatalf("CompleteMultiTurn(nil ctx): %v", err)
	}
	if _, err := p.CompleteMultiTurn(context.Background(), nil, func(ToolUse) (string, error) { return "", nil }, 1); err == nil || !strings.Contains(err.Error(), "nil request") {
		t.Fatalf("CompleteMultiTurn(nil req): %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := p.CompleteMultiTurn(ctx, &Request{Tools: []ToolDefinition{{Name: "t"}}}, func(ToolUse) (string, error) { return "", nil }, 1); err == nil {
		t.Fatalf("CompleteMultiTurn(canceled): expected error")
	}

	srvChoicesEmpty := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = r.Body.Close()
		w.Header().Set("content-type", "application/json")
		_ = json.NewEncoder(w).Encode(openai.ChatCompletionResponse{
			ID:                "id",
			Object:            "chat.completion",
			Created:           time.Now().Unix(),
			Model:             openai.GPT4o,
			Choices:           nil,
			Usage:             openai.Usage{PromptTokensDetails: &openai.PromptTokensDetails{}, CompletionTokensDetails: &openai.CompletionTokensDetails{}},
			SystemFingerprint: "fp",
		})
	}))
	t.Cleanup(srvChoicesEmpty.Close)

	p = NewOpenAIProvider("k", srvChoicesEmpty.URL+"/v1", openai.GPT4o)
	if _, err := p.CompleteMultiTurn(context.Background(), &Request{}, func(ToolUse) (string, error) { return "", nil }, 1); err == nil || !strings.Contains(err.Error(), "tool loop requires tools") {
		t.Fatalf("CompleteMultiTurn(no tools): got %v", err)
	}
	_, err := p.CompleteMultiTurn(context.Background(), &Request{Tools: []ToolDefinition{{Name: "t", InputSchema: map[string]any{"type": "object"}}}}, nil, 1)
	if err == nil || !strings.Contains(err.Error(), "empty choices") {
		t.Fatalf("CompleteMultiTurn(empty choices): got %v", err)
	}

	srvNoTools := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = r.Body.Close()
		w.Header().Set("content-type", "application/json")
		_ = json.NewEncoder(w).Encode(openai.ChatCompletionResponse{
			ID:      "id",
			Object:  "chat.completion",
			Created: time.Now().Unix(),
			Model:   openai.GPT4o,
			Choices: []openai.ChatCompletionChoice{{
				Index:        0,
				FinishReason: openai.FinishReasonStop,
				Message: openai.ChatCompletionMessage{
					Role:    openai.ChatMessageRoleAssistant,
					Content: "done",
				},
			}},
			Usage: openai.Usage{
				PromptTokens:            1,
				CompletionTokens:        1,
				TotalTokens:             2,
				PromptTokensDetails:     &openai.PromptTokensDetails{},
				CompletionTokensDetails: &openai.CompletionTokensDetails{},
			},
			SystemFingerprint: "fp",
		})
	}))
	t.Cleanup(srvNoTools.Close)

	p = NewOpenAIProvider("k", srvNoTools.URL+"/v1", openai.GPT4o)
	out, err := p.CompleteMultiTurn(context.Background(), &Request{
		System:   "sys",
		Messages: []Message{{Role: "user", Content: "hi"}},
		Tools:    []ToolDefinition{{Name: "t1", InputSchema: map[string]any{"type": "object"}}},
	}, func(ToolUse) (string, error) { return "", nil }, 0)
	if err != nil || out == nil || out.Steps != 1 || Text(out.FinalResponse) != "done" {
		t.Fatalf("CompleteMultiTurn(default maxSteps): out=%#v err=%v", out, err)
	}

	srvHTTPError := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = r.Body.Close()
		http.Error(w, "boom", http.StatusInternalServerError)
	}))
	t.Cleanup(srvHTTPError.Close)

	p = NewOpenAIProvider("k", srvHTTPError.URL+"/v1", openai.GPT4o)
	if _, err := p.CompleteMultiTurn(context.Background(), &Request{
		Messages: []Message{{Role: "user", Content: "hi"}},
		Tools:    []ToolDefinition{{Name: "t1", InputSchema: map[string]any{"type": "object"}}},
	}, func(ToolUse) (string, error) { return "", nil }, 1); err == nil {
		t.Fatalf("CompleteMultiTurn(http err): expected error")
	}

	var calls int32
	srvToolCalls := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = r.Body.Close()
		atomic.AddInt32(&calls, 1)
		w.Header().Set("content-type", "application/json")
		_ = json.NewEncoder(w).Encode(openai.ChatCompletionResponse{
			ID:      "id",
			Object:  "chat.completion",
			Created: time.Now().Unix(),
			Model:   openai.GPT4o,
			Choices: []openai.ChatCompletionChoice{{
				Index:        0,
				FinishReason: openai.FinishReasonStop,
				Message: openai.ChatCompletionMessage{
					Role: openai.ChatMessageRoleAssistant,
					ToolCalls: []openai.ToolCall{
						{
							ID:   "call_1",
							Type: openai.ToolTypeFunction,
							Function: openai.FunctionCall{
								Name:      "t1",
								Arguments: `{}`,
							},
						},
					},
				},
			}},
			Usage: openai.Usage{
				PromptTokens:            1,
				CompletionTokens:        1,
				TotalTokens:             2,
				PromptTokensDetails:     &openai.PromptTokensDetails{},
				CompletionTokensDetails: &openai.CompletionTokensDetails{},
			},
			SystemFingerprint: "fp",
		})
	}))
	t.Cleanup(srvToolCalls.Close)

	p = NewOpenAIProvider("k", srvToolCalls.URL+"/v1", openai.GPT4o)
	_, err = p.CompleteMultiTurn(context.Background(), &Request{
		Messages: []Message{{Role: "user", Content: "hi"}},
		Tools:    []ToolDefinition{{Name: "t1", InputSchema: map[string]any{"type": "object"}}},
	}, nil, 1)
	if err == nil || !strings.Contains(err.Error(), "nil tool executor") {
		t.Fatalf("CompleteMultiTurn(nil executor): got %v", err)
	}

	_, err = p.CompleteMultiTurn(context.Background(), &Request{
		Messages: []Message{{Role: "user", Content: "hi"}},
		Tools:    []ToolDefinition{{Name: "t1", InputSchema: map[string]any{"type": "object"}}},
	}, func(ToolUse) (string, error) { return "ok", nil }, 2)
	if err == nil || !strings.Contains(err.Error(), "max steps (2) reached") {
		t.Fatalf("CompleteMultiTurn(max steps): got %v", err)
	}
}
