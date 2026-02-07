package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	openai "github.com/sashabaranov/go-openai"
)

type OpenAIProvider struct {
	client *openai.Client
	model  string
}

func NewOpenAIProvider(apiKey string, baseURL string, model string) *OpenAIProvider {
	cfg := openai.DefaultConfig(strings.TrimSpace(apiKey))
	if v := strings.TrimSpace(baseURL); v != "" {
		cfg.BaseURL = strings.TrimRight(v, "/")
	}

	m := strings.TrimSpace(model)
	if m == "" {
		m = "gpt-4o"
	}

	return &OpenAIProvider{
		client: openai.NewClientWithConfig(cfg),
		model:  m,
	}
}

func (p *OpenAIProvider) Name() string {
	return "openai"
}

func (p *OpenAIProvider) Complete(ctx context.Context, req *Request) (*Response, error) {
	if p == nil || p.client == nil {
		return nil, errors.New("llm: openai: nil client")
	}
	if ctx == nil {
		return nil, errors.New("llm: openai: nil context")
	}
	if req == nil {
		return nil, errors.New("llm: openai: nil request")
	}

	msgs := make([]openai.ChatCompletionMessage, 0, len(req.Messages)+1)
	if system := strings.TrimSpace(req.System); system != "" {
		msgs = append(msgs, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleSystem,
			Content: system,
		})
	}
	for _, m := range req.Messages {
		msgs = append(msgs, openai.ChatCompletionMessage{
			Role:    normalizeOpenAIRole(m.Role),
			Content: m.Content,
		})
	}

	tools := toOpenAITools(req.Tools)

	r := openai.ChatCompletionRequest{
		Model:               strings.TrimSpace(p.model),
		Messages:            msgs,
		MaxTokens: clampMaxTokens(req.MaxTokens),
		Temperature:         float32(req.Temperature),
		Tools:               tools,
	}
	if len(tools) > 0 {
		r.ToolChoice = "auto"
	}

	resp, err := p.client.CreateChatCompletion(ctx, r)
	if err != nil {
		return nil, err
	}
	if len(resp.Choices) == 0 {
		return nil, errors.New("llm: openai: empty choices")
	}

	choice := resp.Choices[0]
	out := &Response{
		StopReason: string(choice.FinishReason),
		Usage: Usage{
			InputTokens:  resp.Usage.PromptTokens,
			OutputTokens: resp.Usage.CompletionTokens,
		},
	}

	msg := choice.Message
	if strings.TrimSpace(msg.Content) != "" {
		out.Content = append(out.Content, ContentBlock{
			Type: "text",
			Text: msg.Content,
		})
	}

	for _, tc := range msg.ToolCalls {
		out.Content = append(out.Content, ContentBlock{
			Type:  "tool_use",
			ID:    strings.TrimSpace(tc.ID),
			Name:  strings.TrimSpace(tc.Function.Name),
			Input: parseToolArguments(tc.Function.Arguments),
		})
	}

	return out, nil
}

func (p *OpenAIProvider) CompleteWithTools(ctx context.Context, req *Request) (*EvalResult, error) {
	start := time.Now()
	resp, err := p.Complete(ctx, req)
	latency := time.Since(start).Milliseconds()

	out := &EvalResult{
		Response:  resp,
		LatencyMs: latency,
		Error:     err,
	}
	if resp == nil {
		if err != nil {
			return out, err
		}
		return out, errors.New("llm: openai: nil response")
	}

	out.InputTokens = resp.Usage.InputTokens
	out.OutputTokens = resp.Usage.OutputTokens

	var sb strings.Builder
	for _, b := range resp.Content {
		switch b.Type {
		case "text":
			sb.WriteString(b.Text)
		case "tool_use":
			out.ToolCalls = append(out.ToolCalls, ToolUse{
				ID:    b.ID,
				Name:  b.Name,
				Input: b.Input,
			})
		}
	}
	out.TextContent = sb.String()

	if err != nil {
		return out, err
	}
	return out, nil
}

func (p *OpenAIProvider) CompleteMultiTurn(
	ctx context.Context,
	req *Request,
	toolExecutor func(ToolUse) (string, error),
	maxSteps int,
) (*MultiTurnResult, error) {
	if p == nil || p.client == nil {
		return nil, errors.New("llm: openai: nil client")
	}
	if ctx == nil {
		return nil, errors.New("llm: openai: nil context")
	}
	if req == nil {
		return nil, errors.New("llm: openai: nil request")
	}
	if maxSteps <= 0 {
		maxSteps = 5
	}

	msgs := make([]openai.ChatCompletionMessage, 0, len(req.Messages)+maxSteps*2+1)
	if system := strings.TrimSpace(req.System); system != "" {
		msgs = append(msgs, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleSystem,
			Content: system,
		})
	}
	for _, m := range req.Messages {
		msgs = append(msgs, openai.ChatCompletionMessage{
			Role:    normalizeOpenAIRole(m.Role),
			Content: m.Content,
		})
	}

	tools := toOpenAITools(req.Tools)
	if len(tools) == 0 {
		return nil, errors.New("llm: openai: tool loop requires tools")
	}

	out := &MultiTurnResult{}

	for step := 0; step < maxSteps; step++ {
		if err := ctx.Err(); err != nil {
			return out, err
		}

		r := openai.ChatCompletionRequest{
			Model:               strings.TrimSpace(p.model),
			Messages:            msgs,
			MaxTokens: clampMaxTokens(req.MaxTokens),
			Temperature:         float32(req.Temperature),
			Tools:               tools,
			ToolChoice:          "auto",
		}

		start := time.Now()
		resp, err := p.client.CreateChatCompletion(ctx, r)
		latency := time.Since(start).Milliseconds()

		out.Steps = step + 1
		out.TotalLatencyMs += latency

		if err != nil {
			return out, err
		}
		if len(resp.Choices) == 0 {
			return out, errors.New("llm: openai: empty choices")
		}

		choice := resp.Choices[0]
		llmResp := openAIToResponse(&resp, &choice)
		out.AllResponses = append(out.AllResponses, llmResp)
		out.FinalResponse = llmResp
		out.TotalInputTokens += resp.Usage.PromptTokens
		out.TotalOutputTokens += resp.Usage.CompletionTokens

		assistantMsg := choice.Message
		if strings.TrimSpace(assistantMsg.Role) == "" {
			assistantMsg.Role = openai.ChatMessageRoleAssistant
		}
		msgs = append(msgs, assistantMsg)

		toolCalls := toolUsesFromOpenAIMessage(assistantMsg)
		if len(toolCalls) > 0 {
			out.AllToolCalls = append(out.AllToolCalls, toolCalls...)
		}

		if len(toolCalls) == 0 {
			return out, nil
		}
		if toolExecutor == nil {
			return out, errors.New("llm: openai: nil tool executor")
		}

		for _, call := range toolCalls {
			content, execErr := toolExecutor(call)
			if execErr != nil {
				content = execErr.Error()
			}
			msgs = append(msgs, openai.ChatCompletionMessage{
				Role:       openai.ChatMessageRoleTool,
				Content:    content,
				ToolCallID: call.ID,
			})
		}
	}

	return out, fmt.Errorf("llm: openai: max steps (%d) reached", maxSteps)
}

func normalizeOpenAIRole(role string) string {
	role = strings.ToLower(strings.TrimSpace(role))
	switch role {
	case openai.ChatMessageRoleSystem,
		openai.ChatMessageRoleUser,
		openai.ChatMessageRoleAssistant,
		openai.ChatMessageRoleTool,
		openai.ChatMessageRoleDeveloper:
		return role
	default:
		return openai.ChatMessageRoleUser
	}
}

func clampMaxTokens(n int) int {
	if n <= 0 {
		return 0
	}
	return n
}

func toOpenAITools(in []ToolDefinition) []openai.Tool {
	if len(in) == 0 {
		return nil
	}
	out := make([]openai.Tool, 0, len(in))
	for _, t := range in {
		name := strings.TrimSpace(t.Name)
		if name == "" {
			continue
		}
		schema := t.InputSchema
		if schema == nil {
			schema = map[string]any{"type": "object"}
		}
		out = append(out, openai.Tool{
			Type: openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{
				Name:        name,
				Description: strings.TrimSpace(t.Description),
				Parameters:  schema,
			},
		})
	}
	return out
}

func parseToolArguments(args string) map[string]any {
	args = strings.TrimSpace(args)
	if args == "" {
		return nil
	}
	var out map[string]any
	if err := json.Unmarshal([]byte(args), &out); err != nil {
		return map[string]any{"_raw": args}
	}
	return out
}

func toolUsesFromOpenAIMessage(msg openai.ChatCompletionMessage) []ToolUse {
	if len(msg.ToolCalls) == 0 {
		return nil
	}
	out := make([]ToolUse, 0, len(msg.ToolCalls))
	for _, tc := range msg.ToolCalls {
		out = append(out, ToolUse{
			ID:    strings.TrimSpace(tc.ID),
			Name:  strings.TrimSpace(tc.Function.Name),
			Input: parseToolArguments(tc.Function.Arguments),
		})
	}
	return out
}

func openAIToResponse(resp *openai.ChatCompletionResponse, choice *openai.ChatCompletionChoice) *Response {
	if resp == nil || choice == nil {
		return nil
	}

	out := &Response{
		StopReason: string(choice.FinishReason),
		Usage: Usage{
			InputTokens:  resp.Usage.PromptTokens,
			OutputTokens: resp.Usage.CompletionTokens,
		},
	}

	msg := choice.Message
	if strings.TrimSpace(msg.Content) != "" {
		out.Content = append(out.Content, ContentBlock{
			Type: "text",
			Text: msg.Content,
		})
	}
	for _, tc := range msg.ToolCalls {
		out.Content = append(out.Content, ContentBlock{
			Type:  "tool_use",
			ID:    strings.TrimSpace(tc.ID),
			Name:  strings.TrimSpace(tc.Function.Name),
			Input: parseToolArguments(tc.Function.Arguments),
		})
	}
	return out
}
