package claude

import (
	"net/http"
	"time"
)

// Client holds configuration for Claude API requests.
type Client struct {
	apiKey     string
	authToken  string
	baseURL    string
	httpClient *http.Client
	model      string
	retryMax   int
	retryBase  time.Duration
}

// Message represents a single role/content message.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ToolDefinition describes a tool exposed to Claude.
type ToolDefinition struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	InputSchema map[string]any `json:"input_schema"`
}

// ToolUse represents a tool invocation requested by Claude.
type ToolUse struct {
	ID    string         `json:"id"`
	Name  string         `json:"name"`
	Input map[string]any `json:"input"`
}

// ToolResult carries the output of a tool call.
type ToolResult struct {
	ToolUseID string `json:"tool_use_id"`
	Content   string `json:"content"`
	IsError   bool   `json:"is_error,omitempty"`
}

// MultiTurnMessage represents a message in a multi-turn tool exchange.
type MultiTurnMessage struct {
	Role       string      `json:"role"`
	Content    any         `json:"content"` // string or []ContentBlock
	ToolResult *ToolResult `json:"-"`       // For tool_result messages
}

// Request defines a Claude messages API request payload.
type Request struct {
	Model       string           `json:"model"`
	Messages    []Message        `json:"messages"`
	MaxTokens   int              `json:"max_tokens"`
	System      string           `json:"system,omitempty"`
	Tools       []ToolDefinition `json:"tools,omitempty"`
	Temperature float64          `json:"temperature,omitempty"`
}

// Response represents a Claude messages API response.
type Response struct {
	ID         string         `json:"id"`
	Type       string         `json:"type"`
	Role       string         `json:"role"`
	Content    []ContentBlock `json:"content"`
	Model      string         `json:"model"`
	StopReason string         `json:"stop_reason"`
	Usage      Usage          `json:"usage"`
}

// ContentBlock represents a single content item in a response.
type ContentBlock struct {
	Type      string         `json:"type"` // "text", "tool_use", or "tool_result"
	Text      string         `json:"text,omitempty"`
	ID        string         `json:"id,omitempty"`
	Name      string         `json:"name,omitempty"`
	Input     map[string]any `json:"input,omitempty"`
	ToolUseID string         `json:"tool_use_id,omitempty"`
	Content   string         `json:"content,omitempty"`
	IsError   bool           `json:"is_error,omitempty"`
}

// Usage reports token usage for a response.
type Usage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// EvalResult aggregates a response with parsed text, tools, and usage.
type EvalResult struct {
	Response     *Response
	TextContent  string
	ToolCalls    []ToolUse
	LatencyMs    int64
	InputTokens  int
	OutputTokens int
	Error        error
}
