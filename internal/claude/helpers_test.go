package claude

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"net"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
)

func TestOptions_NilReceiverAndValidation(t *testing.T) {
	t.Parallel()

	WithBaseURL("http://example.com")(nil)
	WithModel("m")(nil)
	WithRetry(1)(nil)
	WithTimeout(time.Second)(nil)

	c := &Client{}
	WithBaseURL(" ")(c)
	WithModel(" ")(c)
	WithRetry(-1)(c)
	WithTimeout(250 * time.Millisecond)(c)

	if c.retryMax != 0 {
		t.Fatalf("retryMax: got %d want %d", c.retryMax, 0)
	}
	if c.httpClient == nil || c.httpClient.Timeout != 250*time.Millisecond {
		t.Fatalf("httpClient timeout: %#v", c.httpClient)
	}
}

func TestAPIError_ErrorFormatting(t *testing.T) {
	t.Parallel()

	if got := (*APIError)(nil).Error(); got != "claude: api error <nil>" {
		t.Fatalf("Error(nil): got %q", got)
	}

	e := &APIError{Status: "400 Bad Request", Type: "invalid", Message: "bad"}
	if got := e.Error(); !strings.Contains(got, "invalid: bad") {
		t.Fatalf("Error(): got %q", got)
	}

	e = &APIError{Status: "400 Bad Request", Message: "bad"}
	if got := e.Error(); !strings.Contains(got, "400 Bad Request") || !strings.Contains(got, ": bad") {
		t.Fatalf("Error(): got %q", got)
	}

	e = &APIError{Status: "400 Bad Request", Body: []byte(" body ")}
	if got := e.Error(); !strings.Contains(got, ": body") {
		t.Fatalf("Error(): got %q", got)
	}

	e = &APIError{Status: "400 Bad Request"}
	if got := e.Error(); got != "claude: api error (400 Bad Request)" {
		t.Fatalf("Error(): got %q", got)
	}
}

func TestEnsureAuth_EnvFallbacks(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "")
	t.Setenv("ANTHROPIC_AUTH_TOKEN", "")

	if err := (*Client)(nil).ensureAuth(); err == nil {
		t.Fatalf("ensureAuth(nil): expected error")
	}

	c := &Client{}
	if err := c.ensureAuth(); err == nil {
		t.Fatalf("ensureAuth: expected error")
	}

	t.Setenv("ANTHROPIC_API_KEY", "k")
	c = &Client{}
	if err := c.ensureAuth(); err != nil {
		t.Fatalf("ensureAuth(api key): %v", err)
	}
	if c.apiKey != "k" {
		t.Fatalf("apiKey: got %q want %q", c.apiKey, "k")
	}

	t.Setenv("ANTHROPIC_API_KEY", "")
	t.Setenv("ANTHROPIC_AUTH_TOKEN", "t")
	c = &Client{}
	if err := c.ensureAuth(); err != nil {
		t.Fatalf("ensureAuth(auth token): %v", err)
	}
	if c.authToken != "t" {
		t.Fatalf("authToken: got %q want %q", c.authToken, "t")
	}
}

func TestSDKHelpers(t *testing.T) {
	t.Parallel()

	if got := sdkBaseURL("http://example.com/v1/"); got != "http://example.com" {
		t.Fatalf("sdkBaseURL: got %q want %q", got, "http://example.com")
	}

	tools := toSDKTools([]ToolDefinition{{
		Name:        "t",
		Description: "desc",
		InputSchema: map[string]any{
			"type":       "object",
			"properties": map[string]any{"a": map[string]any{"type": "string"}},
			"required":   []any{"a", 123},
			"extra":      true,
		},
	}})
	if len(tools) != 1 || tools[0].OfTool == nil || tools[0].OfTool.Name != "t" {
		t.Fatalf("toSDKTools: %#v", tools)
	}

	schema := toSDKToolInputSchema(map[string]any{
		"properties": map[string]any{"x": map[string]any{"type": "string"}},
		"required":   []string{"x"},
		"extra":      1,
	})
	if len(schema.Required) != 1 || schema.Required[0] != "x" {
		t.Fatalf("schema.Required: %#v", schema.Required)
	}
	if schema.Properties == nil || schema.ExtraFields == nil || schema.ExtraFields["extra"] != 1 {
		t.Fatalf("schema: %#v", schema)
	}

	if got := toStringSlice("bad"); got != nil {
		t.Fatalf("toStringSlice(default): got %#v want nil", got)
	}
	if got := toStringSlice([]any{"a", 1, "b"}); len(got) != 2 || got[0] != "a" || got[1] != "b" {
		t.Fatalf("toStringSlice([]any): got %#v", got)
	}
}

func TestDecodeToolInput(t *testing.T) {
	t.Parallel()

	if got := decodeToolInput(nil); got != nil {
		t.Fatalf("decodeToolInput(nil): got %#v want nil", got)
	}
	if got := decodeToolInput([]byte("not json")); got != nil {
		t.Fatalf("decodeToolInput(invalid): got %#v want nil", got)
	}
	if got := decodeToolInput([]byte(`{"a": 1}`)); got == nil || got["a"] != float64(1) {
		t.Fatalf("decodeToolInput: got %#v", got)
	}
}

type tempNetErr struct{}

func (tempNetErr) Error() string   { return "timeout" }
func (tempNetErr) Timeout() bool   { return true }
func (tempNetErr) Temporary() bool { return true }

func TestRetryHelpers(t *testing.T) {
	t.Parallel()

	if got := clampRetryMax(-1); got != 0 {
		t.Fatalf("clampRetryMax(-1): %d", got)
	}
	if got := clampRetryMax(999); got != maxRetryMax {
		t.Fatalf("clampRetryMax(999): %d", got)
	}
	if got := retryBackoff(0, 1); got != 0 {
		t.Fatalf("retryBackoff(base<=0): %v", got)
	}
	if got := retryBackoff(time.Second, -1); got != 0 {
		t.Fatalf("retryBackoff(attempt<0): %v", got)
	}
	if got := retryBackoff(time.Second, 2); got != 4*time.Second {
		t.Fatalf("retryBackoff: got %v want %v", got, 4*time.Second)
	}

	if shouldRetry(nil) {
		t.Fatalf("shouldRetry(nil): expected false")
	}
	if !shouldRetry(&APIError{StatusCode: http.StatusInternalServerError}) {
		t.Fatalf("shouldRetry(5xx): expected true")
	}
	if shouldRetry(&APIError{StatusCode: http.StatusBadRequest}) {
		t.Fatalf("shouldRetry(4xx): expected false")
	}
	if !shouldRetry(tempNetErr{}) {
		t.Fatalf("shouldRetry(timeout): expected true")
	}

	sdkErr := &anthropic.Error{StatusCode: http.StatusServiceUnavailable}
	if !shouldRetry(sdkErr) {
		t.Fatalf("shouldRetry(anthropic.Error): expected true")
	}
}

func TestSleepWithContext(t *testing.T) {
	t.Parallel()

	if err := sleepWithContext(context.Background(), 0); err != nil {
		t.Fatalf("sleepWithContext(0): %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := sleepWithContext(ctx, time.Second); !errors.Is(err, context.Canceled) {
		t.Fatalf("sleepWithContext(canceled): %v", err)
	}

	if err := sleepWithContext(context.Background(), time.Millisecond); err != nil {
		t.Fatalf("sleepWithContext: %v", err)
	}
}

func TestContentBlocksToSDK(t *testing.T) {
	t.Parallel()

	blocks := contentBlocksToSDK([]ContentBlock{
		{Type: "text", Text: "a"},
		{Type: "tool_use", ID: "id", Name: "t", Input: map[string]any{"k": "v"}},
		{Type: "tool_result", ToolUseID: "id", Content: "ok", IsError: true},
		{Type: "unknown"},
	})
	if len(blocks) != 3 {
		t.Fatalf("contentBlocksToSDK: got %d want %d", len(blocks), 3)
	}
}

func TestFromSDKMessage_EmptyAndToolUse(t *testing.T) {
	t.Parallel()

	if got := fromSDKMessage(nil); got != nil {
		t.Fatalf("fromSDKMessage(nil): got %#v want nil", got)
	}

	msg := &anthropic.Message{ID: "m"}
	msg.Usage.InputTokens = 1
	msg.Usage.OutputTokens = 2
	got := fromSDKMessage(msg)
	if got == nil || got.ID != "m" || got.Usage.InputTokens != 1 || got.Usage.OutputTokens != 2 {
		t.Fatalf("fromSDKMessage: %#v", got)
	}

	var msg2 anthropic.Message
	if err := json.Unmarshal([]byte(`{
		"id":"m2",
		"type":"message",
		"role":"assistant",
		"model":"x",
		"stop_reason":"end_turn",
		"stop_sequence":"",
		"usage":{
			"cache_creation":{},
			"cache_creation_input_tokens":0,
			"cache_read_input_tokens":0,
			"input_tokens":1,
			"output_tokens":2,
			"server_tool_use":{},
			"service_tier":"standard"
		},
		"content":[
			{"type":"text","text":"a"},
			{"type":"tool_use","id":"toolu_1","name":"t","input":{"a":1}}
		]
	}`), &msg2); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	got = fromSDKMessage(&msg2)
	if got == nil || len(got.Content) != 2 {
		t.Fatalf("fromSDKMessage: %#v", got)
	}
	if got.Content[0].Type != "text" || got.Content[0].Text != "a" {
		t.Fatalf("content[0]: %#v", got.Content[0])
	}
	if got.Content[1].Type != "tool_use" || got.Content[1].Name != "t" {
		t.Fatalf("content[1]: %#v", got.Content[1])
	}
}

func TestBuildMessageParams_SystemTempTools(t *testing.T) {
	t.Parallel()

	req := &Request{
		Model:       "m",
		MaxTokens:   10,
		System:      "sys",
		Temperature: math.SmallestNonzeroFloat64,
		Tools: []ToolDefinition{{
			Name:        "t",
			InputSchema: map[string]any{"type": "object"},
		}},
	}
	params := buildMessageParams(req, nil)
	if len(params.System) != 1 || params.System[0].Text != "sys" {
		t.Fatalf("System: %#v", params.System)
	}
	if len(params.Tools) != 1 {
		t.Fatalf("Tools: %#v", params.Tools)
	}
}

func TestShouldRetry_NonTimeoutNetError(t *testing.T) {
	t.Parallel()

	err := &net.DNSError{IsTimeout: false}
	if shouldRetry(err) {
		t.Fatalf("shouldRetry(non-timeout net error): expected false")
	}
}
