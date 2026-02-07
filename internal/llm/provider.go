package llm

import "context"

type Provider interface {
	Name() string
	Complete(ctx context.Context, req *Request) (*Response, error)
	CompleteWithTools(ctx context.Context, req *Request) (*EvalResult, error)
}

// ToolLoopProvider is an optional interface for providers that can execute a
// multi-turn tool-calling loop by repeatedly calling the model and sending tool
// results back.
type ToolLoopProvider interface {
	CompleteMultiTurn(
		ctx context.Context,
		req *Request,
		toolExecutor func(ToolUse) (string, error),
		maxSteps int,
	) (*MultiTurnResult, error)
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ToolDefinition struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	InputSchema map[string]any `json:"input_schema"`
}

type ToolUse struct {
	ID    string         `json:"id"`
	Name  string         `json:"name"`
	Input map[string]any `json:"input"`
}

type Request struct {
	Messages    []Message
	System      string
	MaxTokens   int
	Temperature float64
	Tools       []ToolDefinition
}

type ContentBlock struct {
	Type  string         `json:"type"` // "text" or "tool_use"
	Text  string         `json:"text,omitempty"`
	ID    string         `json:"id,omitempty"`
	Name  string         `json:"name,omitempty"`
	Input map[string]any `json:"input,omitempty"`
}

type Usage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

type Response struct {
	Content    []ContentBlock
	Usage      Usage
	StopReason string
}

type EvalResult struct {
	Response     *Response
	TextContent  string
	ToolCalls    []ToolUse
	LatencyMs    int64
	InputTokens  int
	OutputTokens int
	Error        error
}

type MultiTurnResult struct {
	FinalResponse     *Response
	AllResponses      []*Response
	AllToolCalls      []ToolUse
	TotalLatencyMs    int64
	TotalInputTokens  int
	TotalOutputTokens int
	Steps             int
}
