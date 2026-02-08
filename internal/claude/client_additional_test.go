package claude

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
)

func TestNewClient_EnvFallback(t *testing.T) {
	t.Setenv("ANTHROPIC_BASE_URL", "http://example.com/v1/")
	t.Setenv("ANTHROPIC_API_KEY", "env-key")
	t.Setenv("ANTHROPIC_AUTH_TOKEN", "")

	c := NewClient(" ")
	if c.baseURL != "http://example.com/v1" {
		t.Fatalf("baseURL: got %q want %q", c.baseURL, "http://example.com/v1")
	}
	if c.apiKey != "env-key" {
		t.Fatalf("apiKey: got %q want %q", c.apiKey, "env-key")
	}
	if c.authToken != "" {
		t.Fatalf("authToken: got %q want empty", c.authToken)
	}

	t.Setenv("ANTHROPIC_API_KEY", "")
	t.Setenv("ANTHROPIC_AUTH_TOKEN", "env-token")
	c2 := NewClient("")
	if c2.apiKey != "" || c2.authToken != "env-token" {
		t.Fatalf("env token: apiKey=%q authToken=%q", c2.apiKey, c2.authToken)
	}
}

func TestComplete_Guards(t *testing.T) {
	if _, err := (*Client)(nil).Complete(context.Background(), &Request{}); err == nil {
		t.Fatalf("Complete(nil client): expected error")
	}

	c := NewClient("k")
	if _, err := c.Complete(nil, &Request{}); err == nil {
		t.Fatalf("Complete(nil ctx): expected error")
	}
	if _, err := c.Complete(context.Background(), nil); err == nil {
		t.Fatalf("Complete(nil req): expected error")
	}

	c2 := NewClient("k")
	c2.httpClient = nil
	if _, err := c2.Complete(context.Background(), &Request{}); err == nil {
		t.Fatalf("Complete(nil http client): expected error")
	}

	t.Setenv("ANTHROPIC_API_KEY", "")
	t.Setenv("ANTHROPIC_AUTH_TOKEN", "")
	c3 := NewClient(" ")
	if _, err := c3.Complete(context.Background(), &Request{
		Messages:  []Message{{Role: "user", Content: "hi"}},
		MaxTokens: 1,
	}); err == nil {
		t.Fatalf("Complete(missing auth): expected error")
	}
}

func TestDo_DefaultRetryBase(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Body != nil {
			_ = r.Body.Close()
		}
		w.Header().Set("content-type", "application/json")
		_ = json.NewEncoder(w).Encode(messageResponse(
			"msg_default_retry",
			defaultModel,
			"end_turn",
			[]map[string]any{textBlock("ok")},
			1,
			1,
		))
	}))
	t.Cleanup(srv.Close)

	c := NewClient("k", WithBaseURL(srv.URL+"/v1"))
	c.retryBase = 0
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
	if c.retryBase != retryBaseDelay {
		t.Fatalf("retryBase: got %v want %v", c.retryBase, retryBaseDelay)
	}
}

func TestDo_CancelDuringBackoff(t *testing.T) {
	sig := make(chan struct{}, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Body != nil {
			_ = r.Body.Close()
		}
		writeAPIError(w, http.StatusInternalServerError, "overloaded_error", "server")
		select {
		case sig <- struct{}{}:
		default:
		}
	}))
	t.Cleanup(srv.Close)

	c := NewClient("k", WithBaseURL(srv.URL+"/v1"), WithRetry(3))
	c.retryBase = 2 * time.Second

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		<-sig
		cancel()
	}()

	_, err := c.Complete(ctx, &Request{
		Messages:  []Message{{Role: "user", Content: "hi"}},
		MaxTokens: 1,
	})
	if err == nil || !errors.Is(err, context.Canceled) {
		t.Fatalf("Complete: got err=%v want context canceled", err)
	}
}

func TestCompleteWithTools_Error(t *testing.T) {
	if _, err := (*Client)(nil).CompleteWithTools(context.Background(), &Request{}); err == nil {
		t.Fatalf("CompleteWithTools(nil client): expected error")
	}
}

func TestCompleteMultiTurn_Guards(t *testing.T) {
	exec := func(ToolUse) (string, error) { return "ok", nil }

	if _, err := (*Client)(nil).CompleteMultiTurn(context.Background(), &Request{}, exec, 1); err == nil {
		t.Fatalf("CompleteMultiTurn(nil client): expected error")
	}

	c := NewClient("k")
	if _, err := c.CompleteMultiTurn(nil, &Request{}, exec, 1); err == nil {
		t.Fatalf("CompleteMultiTurn(nil ctx): expected error")
	}
	if _, err := c.CompleteMultiTurn(context.Background(), nil, exec, 1); err == nil {
		t.Fatalf("CompleteMultiTurn(nil req): expected error")
	}

	c2 := NewClient("k")
	c2.httpClient = nil
	if _, err := c2.CompleteMultiTurn(context.Background(), &Request{}, exec, 1); err == nil {
		t.Fatalf("CompleteMultiTurn(nil http client): expected error")
	}

	t.Setenv("ANTHROPIC_API_KEY", "")
	t.Setenv("ANTHROPIC_AUTH_TOKEN", "")
	c3 := NewClient(" ")
	if _, err := c3.CompleteMultiTurn(context.Background(), &Request{}, exec, 1); err == nil {
		t.Fatalf("CompleteMultiTurn(missing auth): expected error")
	}
}

func TestCompleteMultiTurn_ContextErr(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Body != nil {
			_ = r.Body.Close()
		}
		http.Error(w, "unexpected", http.StatusBadRequest)
	}))
	t.Cleanup(srv.Close)

	c := NewClient("k", WithBaseURL(srv.URL+"/v1"))
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	res, err := c.CompleteMultiTurn(ctx, &Request{
		Messages:  []Message{{Role: "user", Content: "hi"}},
		MaxTokens: 1,
	}, func(ToolUse) (string, error) { return "ok", nil }, 1)
	if err == nil || !errors.Is(err, context.Canceled) {
		t.Fatalf("CompleteMultiTurn: got err=%v want context canceled", err)
	}
	if res == nil {
		t.Fatalf("CompleteMultiTurn: nil result")
	}
}

func TestCompleteMultiTurn_MaxStepsDefault(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Body != nil {
			_ = r.Body.Close()
		}
		w.Header().Set("content-type", "application/json")
		_ = json.NewEncoder(w).Encode(messageResponse(
			"msg_1",
			defaultModel,
			"end_turn",
			[]map[string]any{textBlock("ok")},
			1,
			1,
		))
	}))
	t.Cleanup(srv.Close)

	c := NewClient("k", WithBaseURL(srv.URL+"/v1"))
	res, err := c.CompleteMultiTurn(context.Background(), &Request{
		Messages:  []Message{{Role: "user", Content: "hi"}},
		MaxTokens: 1,
	}, nil, 0)
	if err != nil {
		t.Fatalf("CompleteMultiTurn: %v", err)
	}
	if res == nil || res.Steps != 1 {
		t.Fatalf("CompleteMultiTurn: %#v", res)
	}
}

func TestCompleteMultiTurn_StopReasonToolUseWithoutToolCalls(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Body != nil {
			_ = r.Body.Close()
		}
		w.Header().Set("content-type", "application/json")
		_ = json.NewEncoder(w).Encode(messageResponse(
			"msg_1",
			defaultModel,
			"tool_use",
			[]map[string]any{textBlock("no tools")},
			1,
			1,
		))
	}))
	t.Cleanup(srv.Close)

	c := NewClient("k", WithBaseURL(srv.URL+"/v1"))
	_, err := c.CompleteMultiTurn(context.Background(), &Request{
		Messages:  []Message{{Role: "user", Content: "hi"}},
		MaxTokens: 1,
	}, func(ToolUse) (string, error) { return "ok", nil }, 1)
	if err == nil || !strings.Contains(err.Error(), "stop_reason tool_use") {
		t.Fatalf("CompleteMultiTurn: %v", err)
	}
}

func TestCompleteMultiTurn_NilToolExecutor(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Body != nil {
			_ = r.Body.Close()
		}
		w.Header().Set("content-type", "application/json")
		_ = json.NewEncoder(w).Encode(messageResponse(
			"msg_1",
			defaultModel,
			"tool_use",
			[]map[string]any{toolUseBlock("toolu_1", "git", map[string]any{"cmd": "status"})},
			1,
			1,
		))
	}))
	t.Cleanup(srv.Close)

	c := NewClient("k", WithBaseURL(srv.URL+"/v1"))
	_, err := c.CompleteMultiTurn(context.Background(), &Request{
		Messages:  []Message{{Role: "user", Content: "hi"}},
		MaxTokens: 1,
	}, nil, 1)
	if err == nil || !strings.Contains(err.Error(), "nil tool executor") {
		t.Fatalf("CompleteMultiTurn: %v", err)
	}
}

func TestCompleteMultiTurn_ExecutorErrorAndRoleDefault(t *testing.T) {
	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Body != nil {
			_ = r.Body.Close()
		}
		w.Header().Set("content-type", "application/json")
		n := atomic.AddInt32(&calls, 1)
		if n == 1 {
			resp := messageResponse(
				"msg_1",
				defaultModel,
				"tool_use",
				[]map[string]any{toolUseBlock("toolu_1", "git", map[string]any{"cmd": "status"})},
				1,
				1,
			)
			resp["role"] = ""
			_ = json.NewEncoder(w).Encode(resp)
			return
		}
		_ = json.NewEncoder(w).Encode(messageResponse(
			"msg_2",
			defaultModel,
			"end_turn",
			[]map[string]any{textBlock("done")},
			1,
			1,
		))
	}))
	t.Cleanup(srv.Close)

	c := NewClient("k", WithBaseURL(srv.URL+"/v1"))
	res, err := c.CompleteMultiTurn(context.Background(), &Request{
		Messages:  []Message{{Role: "user", Content: "hi"}},
		MaxTokens: 1,
	}, func(ToolUse) (string, error) {
		return "", errors.New("boom")
	}, 3)
	if err != nil {
		t.Fatalf("CompleteMultiTurn: %v", err)
	}
	if res == nil || len(res.AllToolResults) != 1 {
		t.Fatalf("CompleteMultiTurn: %#v", res)
	}
	if !res.AllToolResults[0].IsError || res.AllToolResults[0].Content != "boom" {
		t.Fatalf("ToolResults: %#v", res.AllToolResults)
	}
}

func TestCompleteMultiTurn_DoErrorStopsLoop(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Body != nil {
			_ = r.Body.Close()
		}
		writeAPIError(w, http.StatusBadRequest, "invalid_request_error", "bad")
	}))
	t.Cleanup(srv.Close)

	c := NewClient("k", WithBaseURL(srv.URL+"/v1"))
	_, err := c.CompleteMultiTurn(context.Background(), &Request{
		Messages:  []Message{{Role: "user", Content: "hi"}},
		MaxTokens: 1,
	}, func(ToolUse) (string, error) { return "ok", nil }, 2)
	if err == nil {
		t.Fatalf("CompleteMultiTurn: expected error")
	}
}

func TestCompleteMultiTurn_MaxStepsReached(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Body != nil {
			_ = r.Body.Close()
		}
		w.Header().Set("content-type", "application/json")
		_ = json.NewEncoder(w).Encode(messageResponse(
			"msg_1",
			defaultModel,
			"tool_use",
			[]map[string]any{toolUseBlock("toolu_1", "git", map[string]any{"cmd": "status"})},
			1,
			1,
		))
	}))
	t.Cleanup(srv.Close)

	c := NewClient("k", WithBaseURL(srv.URL+"/v1"))
	res, err := c.CompleteMultiTurn(context.Background(), &Request{
		Messages:  []Message{{Role: "user", Content: "hi"}},
		MaxTokens: 1,
	}, func(ToolUse) (string, error) { return "ok", nil }, 1)
	if err == nil || !strings.Contains(err.Error(), "max steps") {
		t.Fatalf("CompleteMultiTurn: %v", err)
	}
	if res == nil || res.Steps != 1 {
		t.Fatalf("CompleteMultiTurn result: %#v", res)
	}
}

func TestClient_HelperBranches(t *testing.T) {
	t.Parallel()

	if toolUses(nil) != nil {
		t.Fatalf("toolUses(nil): expected nil")
	}
	if normalizeError(nil) != nil {
		t.Fatalf("normalizeError(nil): expected nil")
	}
	if apiErrorFromSDK(nil) != nil {
		t.Fatalf("apiErrorFromSDK(nil): expected nil")
	}

	apiErr := apiErrorFromSDK(&anthropic.Error{StatusCode: http.StatusTooManyRequests})
	if apiErr == nil || !strings.Contains(apiErr.Status, "429") {
		t.Fatalf("apiErrorFromSDK status: %#v", apiErr)
	}

	c := &Client{baseURL: defaultBaseURL, httpClient: &http.Client{}, authToken: "tok"}
	if c.newSDKClient() == nil {
		t.Fatalf("newSDKClient: expected non-nil")
	}

	if got := contentBlocksToSDK(nil); got != nil {
		t.Fatalf("contentBlocksToSDK(nil): %#v", got)
	}
	if got := toSDKToolInputSchema(nil); got.Type != "object" {
		t.Fatalf("toSDKToolInputSchema(nil): %#v", got)
	}
}
