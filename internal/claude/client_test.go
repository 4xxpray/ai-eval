package claude

import (
	"bytes"
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
)

func TestComplete_DefaultModelAndHeaders(t *testing.T) {
	t.Parallel()

	reqCh := make(chan map[string]any, 1)
	hdrCh := make(chan http.Header, 1)
	pathCh := make(chan string, 1)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Body != nil {
			defer r.Body.Close()
		}

		b, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, "read body", http.StatusBadRequest)
			return
		}

		var gotReq map[string]any
		if err := json.Unmarshal(b, &gotReq); err != nil {
			http.Error(w, "bad json", http.StatusBadRequest)
			return
		}

		reqCh <- gotReq
		hdrCh <- r.Header.Clone()
		pathCh <- r.URL.Path

		w.Header().Set("content-type", "application/json")
		model, _ := gotReq["model"].(string)
		_ = json.NewEncoder(w).Encode(messageResponse(
			"msg_1",
			model,
			"end_turn",
			[]map[string]any{textBlock("ok")},
			1,
			2,
		))
	}))
	t.Cleanup(srv.Close)

	c := NewClient("test-key", WithBaseURL(srv.URL+"/v1/"))
	resp, err := c.Complete(context.Background(), &Request{
		Messages:  []Message{{Role: "user", Content: "hi"}},
		MaxTokens: 12,
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if resp == nil {
		t.Fatalf("Complete: nil response")
	}
	if resp.Content[0].Text != "ok" {
		t.Fatalf("Content[0].Text: got %q want %q", resp.Content[0].Text, "ok")
	}

	gotReq := <-reqCh
	gotHdr := <-hdrCh
	gotPath := <-pathCh

	if gotPath != "/v1/messages" {
		t.Fatalf("path: got %q want %q", gotPath, "/v1/messages")
	}
	if gotReq["model"] != defaultModel {
		t.Fatalf("model: got %v want %q", gotReq["model"], defaultModel)
	}
	if gotReq["max_tokens"] != float64(12) {
		t.Fatalf("max_tokens: got %v want %d", gotReq["max_tokens"], 12)
	}
	msgs, _ := gotReq["messages"].([]any)
	if len(msgs) != 1 {
		t.Fatalf("messages: got %d want %d", len(msgs), 1)
	}
	m0, _ := msgs[0].(map[string]any)
	if m0["role"] != "user" {
		t.Fatalf("messages[0].role: got %v want %q", m0["role"], "user")
	}
	content, _ := m0["content"].([]any)
	if len(content) != 1 {
		t.Fatalf("messages[0].content: got %d want %d", len(content), 1)
	}
	b0, _ := content[0].(map[string]any)
	if b0["type"] != "text" || b0["text"] != "hi" {
		t.Fatalf("messages[0].content[0]: got %#v", b0)
	}
	if gotHdr.Get("x-api-key") != "test-key" {
		t.Fatalf("x-api-key: got %q want %q", gotHdr.Get("x-api-key"), "test-key")
	}
	if gotHdr.Get("anthropic-version") != apiVersionHeader {
		t.Fatalf("anthropic-version: got %q want %q", gotHdr.Get("anthropic-version"), apiVersionHeader)
	}
	if got := gotHdr.Get("content-type"); !strings.Contains(got, "application/json") {
		t.Fatalf("content-type: got %q", got)
	}
}

func TestCompleteWithTools_ParsesTextAndToolUse(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("content-type", "application/json")
		_ = json.NewEncoder(w).Encode(messageResponse(
			"msg_2",
			defaultModel,
			"tool_use",
			[]map[string]any{
				textBlock("a"),
				toolUseBlock("toolu_1", "search", map[string]any{"q": "x"}),
				textBlock("b"),
			},
			3,
			4,
		))
	}))
	t.Cleanup(srv.Close)

	c := NewClient("k", WithBaseURL(srv.URL+"/v1"))
	res, err := c.CompleteWithTools(context.Background(), &Request{
		Messages:  []Message{{Role: "user", Content: "hi"}},
		MaxTokens: 12,
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
	if len(res.ToolCalls) != 1 {
		t.Fatalf("len(ToolCalls): got %d want %d", len(res.ToolCalls), 1)
	}
	if res.ToolCalls[0].Name != "search" {
		t.Fatalf("ToolCalls[0].Name: got %q want %q", res.ToolCalls[0].Name, "search")
	}
	if res.InputTokens != 3 || res.OutputTokens != 4 {
		t.Fatalf("tokens: got in=%d out=%d want in=%d out=%d", res.InputTokens, res.OutputTokens, 3, 4)
	}
	if res.LatencyMs < 0 {
		t.Fatalf("LatencyMs: got %d want >= 0", res.LatencyMs)
	}
}

func TestComplete_APIError(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("content-type", "application/json")
		w.Header().Set("request-id", "rid_123")
		w.WriteHeader(http.StatusBadRequest)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"type": "error",
			"error": map[string]any{
				"type":    "invalid_request_error",
				"message": "bad",
			},
		})
	}))
	t.Cleanup(srv.Close)

	c := NewClient("k", WithBaseURL(srv.URL+"/v1"))
	_, err := c.Complete(context.Background(), &Request{
		Messages:  []Message{{Role: "user", Content: "hi"}},
		MaxTokens: 1,
	})
	if err == nil {
		t.Fatalf("Complete: expected error")
	}

	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("error type: got %T want *APIError", err)
	}
	if apiErr.StatusCode != http.StatusBadRequest {
		t.Fatalf("StatusCode: got %d want %d", apiErr.StatusCode, http.StatusBadRequest)
	}
	if apiErr.Type != "invalid_request_error" {
		t.Fatalf("Type: got %q want %q", apiErr.Type, "invalid_request_error")
	}
	if apiErr.Message != "bad" {
		t.Fatalf("Message: got %q want %q", apiErr.Message, "bad")
	}
	if apiErr.RequestID != "rid_123" {
		t.Fatalf("RequestID: got %q want %q", apiErr.RequestID, "rid_123")
	}
	if !strings.Contains(err.Error(), "invalid_request_error") {
		t.Fatalf("Error(): got %q", err.Error())
	}
}

func TestComplete_RetryOn5xx(t *testing.T) {
	t.Parallel()

	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = r.Body.Close()
		n := atomic.AddInt32(&calls, 1)
		if n < 3 {
			writeAPIError(w, http.StatusInternalServerError, "overloaded_error", "server")
			return
		}
		w.Header().Set("content-type", "application/json")
		_ = json.NewEncoder(w).Encode(messageResponse(
			"msg_retry",
			defaultModel,
			"end_turn",
			[]map[string]any{textBlock("ok")},
			1,
			1,
		))
	}))
	t.Cleanup(srv.Close)

	c := NewClient("k", WithBaseURL(srv.URL+"/v1"), WithRetry(3))
	c.retryBase = time.Millisecond
	resp, err := c.Complete(context.Background(), &Request{
		Messages:  []Message{{Role: "user", Content: "hi"}},
		MaxTokens: 1,
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if resp == nil {
		t.Fatalf("Complete: nil response")
	}
	if got := atomic.LoadInt32(&calls); got != 3 {
		t.Fatalf("calls: got %d want %d", got, 3)
	}
}

func TestComplete_NoRetryOn4xx(t *testing.T) {
	t.Parallel()

	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = r.Body.Close()
		atomic.AddInt32(&calls, 1)
		writeAPIError(w, http.StatusBadRequest, "invalid_request_error", "bad")
	}))
	t.Cleanup(srv.Close)

	c := NewClient("k", WithBaseURL(srv.URL+"/v1"), WithRetry(3))
	c.retryBase = time.Millisecond
	_, err := c.Complete(context.Background(), &Request{
		Messages:  []Message{{Role: "user", Content: "hi"}},
		MaxTokens: 1,
	})
	if err == nil {
		t.Fatalf("Complete: expected error")
	}
	if got := atomic.LoadInt32(&calls); got != 1 {
		t.Fatalf("calls: got %d want %d", got, 1)
	}
}

type timeoutError struct{}

func (timeoutError) Error() string   { return "timeout" }
func (timeoutError) Timeout() bool   { return true }
func (timeoutError) Temporary() bool { return true }

type timeoutRoundTripper struct {
	calls int32
}

func (rt *timeoutRoundTripper) RoundTrip(r *http.Request) (*http.Response, error) {
	if r.Body != nil {
		_ = r.Body.Close()
	}
	n := atomic.AddInt32(&rt.calls, 1)
	if n < 3 {
		return nil, timeoutError{}
	}

	payload, _ := json.Marshal(messageResponse(
		"msg_timeout",
		defaultModel,
		"end_turn",
		[]map[string]any{textBlock("ok")},
		1,
		1,
	))
	body := io.NopCloser(bytes.NewReader(payload))
	return &http.Response{
		StatusCode: http.StatusOK,
		Status:     "200 OK",
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body:       body,
	}, nil
}

func TestComplete_RetryOnTimeout(t *testing.T) {
	t.Parallel()

	rt := &timeoutRoundTripper{}
	c := NewClient("k", WithRetry(2))
	c.retryBase = time.Millisecond
	c.httpClient = &http.Client{Transport: rt}

	resp, err := c.Complete(context.Background(), &Request{
		Messages:  []Message{{Role: "user", Content: "hi"}},
		MaxTokens: 1,
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if resp == nil {
		t.Fatalf("Complete: nil response")
	}
	if got := atomic.LoadInt32(&rt.calls); got != 3 {
		t.Fatalf("calls: got %d want %d", got, 3)
	}
}

func TestOptions(t *testing.T) {
	t.Parallel()

	c := NewClient("k",
		WithBaseURL("http://example.com/v1/"),
		WithModel("custom-model"),
		WithTimeout(5*time.Second),
		WithRetry(2),
	)

	if c.baseURL != "http://example.com/v1" {
		t.Fatalf("baseURL: got %q want %q", c.baseURL, "http://example.com/v1")
	}
	if c.model != "custom-model" {
		t.Fatalf("model: got %q want %q", c.model, "custom-model")
	}
	if c.httpClient.Timeout != 5*time.Second {
		t.Fatalf("timeout: got %v want %v", c.httpClient.Timeout, 5*time.Second)
	}
	if c.retryMax != 2 {
		t.Fatalf("retryMax: got %d want %d", c.retryMax, 2)
	}
}

func TestCompleteMultiTurn_ToolUseFlow(t *testing.T) {
	t.Parallel()

	var calls int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++

		if r.Body != nil {
			defer r.Body.Close()
		}
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
		switch calls {
		case 1:
			if req["model"] != defaultModel {
				http.Error(w, "model", http.StatusBadRequest)
				return
			}
			if len(msgs) != 1 {
				http.Error(w, "messages len", http.StatusBadRequest)
				return
			}
			m0, _ := msgs[0].(map[string]any)
			m0c, _ := m0["content"].([]any)
			if m0["role"] != "user" || len(m0c) != 1 {
				http.Error(w, "bad message", http.StatusBadRequest)
				return
			}
			b0, _ := m0c[0].(map[string]any)
			if b0["type"] != "text" || b0["text"] != "hi" {
				http.Error(w, "bad message", http.StatusBadRequest)
				return
			}

			w.Header().Set("content-type", "application/json")
			_ = json.NewEncoder(w).Encode(messageResponse(
				"msg_1",
				defaultModel,
				"tool_use",
				[]map[string]any{
					toolUseBlock("toolu_1", "git", map[string]any{"cmd": "status"}),
				},
				1,
				1,
			))
		case 2:
			if len(msgs) != 3 {
				http.Error(w, "messages len 2", http.StatusBadRequest)
				return
			}

			m1, _ := msgs[1].(map[string]any)
			m1c, _ := m1["content"].([]any)
			if m1["role"] != "assistant" || len(m1c) != 1 {
				http.Error(w, "assistant message", http.StatusBadRequest)
				return
			}
			b1, _ := m1c[0].(map[string]any)
			if b1["type"] != "tool_use" || b1["id"] != "toolu_1" || b1["name"] != "git" {
				http.Error(w, "tool_use block", http.StatusBadRequest)
				return
			}

			m2, _ := msgs[2].(map[string]any)
			m2c, _ := m2["content"].([]any)
			if m2["role"] != "user" || len(m2c) != 1 {
				http.Error(w, "tool_result message", http.StatusBadRequest)
				return
			}
			b2, _ := m2c[0].(map[string]any)
			if b2["type"] != "tool_result" || b2["tool_use_id"] != "toolu_1" {
				http.Error(w, "tool_result block", http.StatusBadRequest)
				return
			}
			b2c, _ := b2["content"].([]any)
			if len(b2c) != 1 {
				http.Error(w, "tool_result content", http.StatusBadRequest)
				return
			}
			b2t, _ := b2c[0].(map[string]any)
			if b2t["type"] != "text" || b2t["text"] != "ok" {
				http.Error(w, "tool_result block", http.StatusBadRequest)
				return
			}

			w.Header().Set("content-type", "application/json")
			_ = json.NewEncoder(w).Encode(messageResponse(
				"msg_2",
				defaultModel,
				"end_turn",
				[]map[string]any{textBlock("done")},
				2,
				3,
			))
		default:
			http.Error(w, "unexpected call", http.StatusBadRequest)
		}
	}))
	t.Cleanup(srv.Close)

	c := NewClient("k", WithBaseURL(srv.URL+"/v1"))
	res, err := c.CompleteMultiTurn(context.Background(), &Request{
		Messages:  []Message{{Role: "user", Content: "hi"}},
		MaxTokens: 12,
	}, func(toolUse ToolUse) (string, error) {
		if toolUse.Name != "git" {
			return "", errors.New("unexpected tool")
		}
		return "ok", nil
	}, 3)
	if err != nil {
		t.Fatalf("CompleteMultiTurn: %v", err)
	}
	if res == nil || res.FinalResponse == nil {
		t.Fatalf("CompleteMultiTurn: nil result/final response")
	}
	if res.FinalResponse.StopReason != "end_turn" {
		t.Fatalf("StopReason: got %q want %q", res.FinalResponse.StopReason, "end_turn")
	}
	if res.Steps != 2 {
		t.Fatalf("Steps: got %d want %d", res.Steps, 2)
	}
	if len(res.AllResponses) != 2 {
		t.Fatalf("len(AllResponses): got %d want %d", len(res.AllResponses), 2)
	}
	if len(res.AllToolCalls) != 1 || res.AllToolCalls[0].Name != "git" {
		t.Fatalf("AllToolCalls: got %#v", res.AllToolCalls)
	}
	if len(res.AllToolResults) != 1 || res.AllToolResults[0].Content != "ok" || res.AllToolResults[0].IsError {
		t.Fatalf("AllToolResults: got %#v", res.AllToolResults)
	}
	if res.TotalInputTokens != 3 || res.TotalOutputTokens != 4 {
		t.Fatalf("tokens: got in=%d out=%d want in=%d out=%d", res.TotalInputTokens, res.TotalOutputTokens, 3, 4)
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

func toolUseBlock(id, name string, input map[string]any) map[string]any {
	return map[string]any{
		"type":  "tool_use",
		"id":    id,
		"name":  name,
		"input": input,
	}
}

func writeAPIError(w http.ResponseWriter, status int, typ, message string) {
	w.Header().Set("content-type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"type": "error",
		"error": map[string]any{
			"type":    typ,
			"message": message,
		},
	})
}
