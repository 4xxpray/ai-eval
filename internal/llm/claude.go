package llm

import (
	"context"
	"errors"
	"strings"

	"github.com/stellarlinkco/ai-eval/internal/claude"
)

type ClaudeProvider struct {
	client *claude.Client
}

func NewClaudeProvider(apiKey string, baseURL string, model string) *ClaudeProvider {
	opts := make([]claude.Option, 0, 2)
	if v := strings.TrimSpace(baseURL); v != "" {
		opts = append(opts, claude.WithBaseURL(v))
	}
	if v := strings.TrimSpace(model); v != "" {
		opts = append(opts, claude.WithModel(v))
	}
	return &ClaudeProvider{
		client: claude.NewClient(strings.TrimSpace(apiKey), opts...),
	}
}

func (p *ClaudeProvider) Name() string {
	return "claude"
}

func (p *ClaudeProvider) Complete(ctx context.Context, req *Request) (*Response, error) {
	if p == nil || p.client == nil {
		return nil, errors.New("llm: claude: nil client")
	}
	cReq, err := toClaudeRequest(req)
	if err != nil {
		return nil, err
	}
	resp, err := p.client.Complete(ctx, cReq)
	return fromClaudeResponse(resp), err
}

func (p *ClaudeProvider) CompleteWithTools(ctx context.Context, req *Request) (*EvalResult, error) {
	if p == nil || p.client == nil {
		return nil, errors.New("llm: claude: nil client")
	}
	cReq, err := toClaudeRequest(req)
	if err != nil {
		return nil, err
	}
	res, err := p.client.CompleteWithTools(ctx, cReq)
	return fromClaudeEvalResult(res), err
}

func (p *ClaudeProvider) CompleteMultiTurn(
	ctx context.Context,
	req *Request,
	toolExecutor func(ToolUse) (string, error),
	maxSteps int,
) (*MultiTurnResult, error) {
	if p == nil || p.client == nil {
		return nil, errors.New("llm: claude: nil client")
	}

	cReq, err := toClaudeRequest(req)
	if err != nil {
		return nil, err
	}

	var cToolExecutor func(claude.ToolUse) (string, error)
	if toolExecutor != nil {
		cToolExecutor = func(tu claude.ToolUse) (string, error) {
			return toolExecutor(ToolUse{
				ID:    tu.ID,
				Name:  tu.Name,
				Input: tu.Input,
			})
		}
	}

	res, err := p.client.CompleteMultiTurn(ctx, cReq, cToolExecutor, maxSteps)
	return fromClaudeMultiTurnResult(res), err
}

func toClaudeRequest(req *Request) (*claude.Request, error) {
	if req == nil {
		return nil, errors.New("llm: claude: nil request")
	}

	msgs := make([]claude.Message, 0, len(req.Messages))
	for _, m := range req.Messages {
		role := strings.TrimSpace(m.Role)
		if role == "" {
			role = "user"
		}
		msgs = append(msgs, claude.Message{
			Role:    role,
			Content: m.Content,
		})
	}

	tools := make([]claude.ToolDefinition, 0, len(req.Tools))
	for _, t := range req.Tools {
		name := strings.TrimSpace(t.Name)
		if name == "" {
			continue
		}
		schema := t.InputSchema
		if schema == nil {
			schema = map[string]any{"type": "object"}
		}
		tools = append(tools, claude.ToolDefinition{
			Name:        name,
			Description: strings.TrimSpace(t.Description),
			InputSchema: schema,
		})
	}

	return &claude.Request{
		Messages:    msgs,
		MaxTokens:   req.MaxTokens,
		System:      req.System,
		Temperature: req.Temperature,
		Tools:       tools,
	}, nil
}

func fromClaudeResponse(resp *claude.Response) *Response {
	if resp == nil {
		return nil
	}

	out := &Response{
		StopReason: resp.StopReason,
		Usage: Usage{
			InputTokens:  resp.Usage.InputTokens,
			OutputTokens: resp.Usage.OutputTokens,
		},
	}

	if len(resp.Content) == 0 {
		return out
	}

	out.Content = make([]ContentBlock, 0, len(resp.Content))
	for _, b := range resp.Content {
		switch b.Type {
		case "text":
			out.Content = append(out.Content, ContentBlock{
				Type: "text",
				Text: b.Text,
			})
		case "tool_use":
			out.Content = append(out.Content, ContentBlock{
				Type:  "tool_use",
				ID:    b.ID,
				Name:  b.Name,
				Input: b.Input,
			})
		}
	}

	return out
}

func fromClaudeEvalResult(res *claude.EvalResult) *EvalResult {
	if res == nil {
		return nil
	}

	out := &EvalResult{
		Response:     fromClaudeResponse(res.Response),
		TextContent:  res.TextContent,
		LatencyMs:    res.LatencyMs,
		InputTokens:  res.InputTokens,
		OutputTokens: res.OutputTokens,
		Error:        res.Error,
	}

	if len(res.ToolCalls) > 0 {
		out.ToolCalls = make([]ToolUse, 0, len(res.ToolCalls))
		for _, tc := range res.ToolCalls {
			out.ToolCalls = append(out.ToolCalls, ToolUse{
				ID:    tc.ID,
				Name:  tc.Name,
				Input: tc.Input,
			})
		}
	}

	return out
}

func fromClaudeMultiTurnResult(res *claude.MultiTurnResult) *MultiTurnResult {
	if res == nil {
		return nil
	}

	out := &MultiTurnResult{
		FinalResponse:     fromClaudeResponse(res.FinalResponse),
		TotalLatencyMs:    res.TotalLatencyMs,
		TotalInputTokens:  res.TotalInputTokens,
		TotalOutputTokens: res.TotalOutputTokens,
		Steps:             res.Steps,
	}

	if len(res.AllToolCalls) > 0 {
		out.AllToolCalls = make([]ToolUse, 0, len(res.AllToolCalls))
		for _, tc := range res.AllToolCalls {
			out.AllToolCalls = append(out.AllToolCalls, ToolUse{
				ID:    tc.ID,
				Name:  tc.Name,
				Input: tc.Input,
			})
		}
	}

	if len(res.AllResponses) > 0 {
		out.AllResponses = make([]*Response, 0, len(res.AllResponses))
		for _, r := range res.AllResponses {
			out.AllResponses = append(out.AllResponses, fromClaudeResponse(r))
		}
	}

	return out
}
