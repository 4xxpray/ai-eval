package claude

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/packages/param"
)

const (
	defaultBaseURL  = "https://api.anthropic.com/v1"
	defaultModel    = "claude-sonnet-4-5-20250929"
	defaultRetryMax = 3
	maxRetryMax     = 3
	retryBaseDelay  = time.Second

	apiVersionHeader = "2023-06-01"
)

// Option configures a Client.
type Option func(*Client)

// WithBaseURL sets the Claude API base URL.
func WithBaseURL(baseURL string) Option {
	return func(c *Client) {
		if c == nil {
			return
		}
		baseURL = strings.TrimSpace(baseURL)
		if baseURL == "" {
			return
		}
		c.baseURL = strings.TrimRight(baseURL, "/")
	}
}

// WithModel sets the default model name.
func WithModel(model string) Option {
	return func(c *Client) {
		if c == nil {
			return
		}
		model = strings.TrimSpace(model)
		if model == "" {
			return
		}
		c.model = model
	}
}

// WithTimeout sets the HTTP client timeout.
func WithTimeout(timeout time.Duration) Option {
	return func(c *Client) {
		if c == nil {
			return
		}
		if c.httpClient == nil {
			c.httpClient = &http.Client{}
		}
		c.httpClient.Timeout = timeout
	}
}

// WithRetry sets the max retry count for retryable failures.
func WithRetry(maxRetries int) Option {
	return func(c *Client) {
		if c == nil {
			return
		}
		c.retryMax = clampRetryMax(maxRetries)
	}
}

// NewClient constructs a Client with the given API key and options.
func NewClient(apiKey string, opts ...Option) *Client {
	apiKey = strings.TrimSpace(apiKey)
	c := &Client{
		apiKey:     apiKey,
		baseURL:    strings.TrimRight(defaultBaseURL, "/"),
		httpClient: &http.Client{},
		model:      defaultModel,
		retryMax:   defaultRetryMax,
		retryBase:  retryBaseDelay,
	}
	// Check ANTHROPIC_BASE_URL env var
	if envBaseURL := strings.TrimSpace(os.Getenv("ANTHROPIC_BASE_URL")); envBaseURL != "" {
		c.baseURL = strings.TrimRight(envBaseURL, "/")
	}
	if c.apiKey == "" {
		if envKey := strings.TrimSpace(os.Getenv("ANTHROPIC_API_KEY")); envKey != "" {
			c.apiKey = envKey
		} else if envToken := strings.TrimSpace(os.Getenv("ANTHROPIC_AUTH_TOKEN")); envToken != "" {
			c.authToken = envToken
		}
	}
	for _, opt := range opts {
		if opt != nil {
			opt(c)
		}
	}
	return c
}

// APIError represents a non-2xx response from the Claude API.
type APIError struct {
	StatusCode int
	Status     string
	RequestID  string
	Type       string
	Message    string
	Body       []byte
}

// Error formats the API error string.
func (e *APIError) Error() string {
	if e == nil {
		return "claude: api error <nil>"
	}

	msg := strings.TrimSpace(e.Message)
	if msg == "" && len(e.Body) > 0 {
		msg = strings.TrimSpace(string(e.Body))
	}

	switch {
	case e.Type != "" && msg != "":
		return fmt.Sprintf("claude: api error (%s): %s: %s", e.Status, e.Type, msg)
	case msg != "":
		return fmt.Sprintf("claude: api error (%s): %s", e.Status, msg)
	default:
		return fmt.Sprintf("claude: api error (%s)", e.Status)
	}
}

// Complete sends a messages API request and returns the response.
func (c *Client) Complete(ctx context.Context, req *Request) (*Response, error) {
	if c == nil {
		return nil, errors.New("claude: nil client")
	}
	if ctx == nil {
		return nil, errors.New("claude: nil context")
	}
	if req == nil {
		return nil, errors.New("claude: nil request")
	}
	if c.httpClient == nil {
		return nil, errors.New("claude: nil http client")
	}
	if err := c.ensureAuth(); err != nil {
		return nil, err
	}

	r := *req
	if strings.TrimSpace(r.Model) == "" {
		r.Model = c.model
	}

	messages := toSDKMessages(r.Messages)
	params := buildMessageParams(&r, messages)
	return c.do(ctx, params)
}

func (c *Client) do(ctx context.Context, params anthropic.MessageNewParams) (*Response, error) {
	retryMax := clampRetryMax(c.retryMax)
	if c.retryBase <= 0 {
		c.retryBase = retryBaseDelay
	}

	sdk := c.newSDKClient()
	for attempt := 0; ; attempt++ {
		msg, err := sdk.Messages.New(ctx, params)
		if err != nil {
			err = normalizeError(err)
			if !shouldRetry(err) || attempt >= retryMax {
				return nil, err
			}
			if err := sleepWithContext(ctx, retryBackoff(c.retryBase, attempt)); err != nil {
				return nil, err
			}
			continue
		}

		return fromSDKMessage(msg), nil
	}
}

// CompleteWithTools sends a request and extracts text and tool calls.
func (c *Client) CompleteWithTools(ctx context.Context, req *Request) (*EvalResult, error) {
	start := time.Now()
	resp, err := c.Complete(ctx, req)
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
		return out, errors.New("claude: nil response")
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

// MultiTurnResult captures responses, tool calls, and usage from a multi-turn run.
type MultiTurnResult struct {
	FinalResponse     *Response
	AllResponses      []*Response
	AllToolCalls      []ToolUse
	AllToolResults    []ToolResult
	TotalLatencyMs    int64
	TotalInputTokens  int
	TotalOutputTokens int
	Steps             int
}

// CompleteMultiTurn runs a multi-step tool loop until end_turn or max steps.
func (c *Client) CompleteMultiTurn(
	ctx context.Context,
	req *Request,
	toolExecutor func(toolUse ToolUse) (string, error),
	maxSteps int,
) (*MultiTurnResult, error) {
	if c == nil {
		return nil, errors.New("claude: nil client")
	}
	if ctx == nil {
		return nil, errors.New("claude: nil context")
	}
	if req == nil {
		return nil, errors.New("claude: nil request")
	}
	if c.httpClient == nil {
		return nil, errors.New("claude: nil http client")
	}
	if err := c.ensureAuth(); err != nil {
		return nil, err
	}
	if maxSteps <= 0 {
		maxSteps = 5
	}

	model := strings.TrimSpace(req.Model)
	if model == "" {
		model = c.model
	}

	messages := make([]anthropic.MessageParam, 0, len(req.Messages)+maxSteps*2)
	for _, m := range req.Messages {
		messages = append(messages, toSDKMessage(m.Role, []anthropic.ContentBlockParamUnion{
			anthropic.NewTextBlock(m.Content),
		}))
	}

	out := &MultiTurnResult{}

	for step := 0; step < maxSteps; step++ {
		if err := ctx.Err(); err != nil {
			return out, err
		}

		stepReq := *req
		stepReq.Model = model
		params := buildMessageParams(&stepReq, messages)

		start := time.Now()
		resp, err := c.do(ctx, params)
		latency := time.Since(start).Milliseconds()

		out.Steps = step + 1
		out.TotalLatencyMs += latency

		if resp != nil {
			out.AllResponses = append(out.AllResponses, resp)
			out.FinalResponse = resp
			out.TotalInputTokens += resp.Usage.InputTokens
			out.TotalOutputTokens += resp.Usage.OutputTokens
		}
		if err != nil {
			return out, err
		}

		role := strings.TrimSpace(resp.Role)
		if role == "" {
			role = "assistant"
		}
		messages = append(messages, toSDKMessage(role, contentBlocksToSDK(resp.Content)))

		toolCalls := toolUses(resp)
		if len(toolCalls) > 0 {
			out.AllToolCalls = append(out.AllToolCalls, toolCalls...)
		}

		if len(toolCalls) == 0 {
			if resp.StopReason == "tool_use" {
				return out, errors.New("claude: stop_reason tool_use but no tool calls")
			}
			return out, nil
		}

		if toolExecutor == nil {
			return out, errors.New("claude: nil tool executor")
		}

		blocks := make([]anthropic.ContentBlockParamUnion, 0, len(toolCalls))
		for _, call := range toolCalls {
			content, execErr := toolExecutor(call)
			tr := ToolResult{
				ToolUseID: call.ID,
				Content:   content,
			}
			if execErr != nil {
				tr.IsError = true
				tr.Content = execErr.Error()
			}
			out.AllToolResults = append(out.AllToolResults, tr)
			blocks = append(blocks, anthropic.NewToolResultBlock(tr.ToolUseID, tr.Content, tr.IsError))
		}

		messages = append(messages, toSDKMessage("user", blocks))
	}

	return out, fmt.Errorf("claude: max steps (%d) reached", maxSteps)
}

func toolUses(resp *Response) []ToolUse {
	if resp == nil {
		return nil
	}

	var out []ToolUse
	for _, b := range resp.Content {
		if b.Type != "tool_use" {
			continue
		}
		out = append(out, ToolUse{
			ID:    b.ID,
			Name:  b.Name,
			Input: b.Input,
		})
	}
	return out
}

type apiErrorEnvelope struct {
	Type  string `json:"type"`
	Error struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

func normalizeError(err error) error {
	if err == nil {
		return nil
	}

	var sdkErr *anthropic.Error
	if errors.As(err, &sdkErr) {
		return apiErrorFromSDK(sdkErr)
	}
	return err
}

func apiErrorFromSDK(err *anthropic.Error) *APIError {
	if err == nil {
		return nil
	}

	apiErr := &APIError{
		StatusCode: err.StatusCode,
		RequestID:  err.RequestID,
	}
	if err.Response != nil {
		apiErr.Status = err.Response.Status
	} else if err.StatusCode != 0 {
		apiErr.Status = fmt.Sprintf("%d %s", err.StatusCode, http.StatusText(err.StatusCode))
	}

	raw := strings.TrimSpace(err.RawJSON())
	if raw != "" {
		apiErr.Body = []byte(raw)
		var env apiErrorEnvelope
		if json.Unmarshal([]byte(raw), &env) == nil {
			apiErr.Type = env.Error.Type
			apiErr.Message = env.Error.Message
		}
	}

	return apiErr
}

func (c *Client) ensureAuth() error {
	if c == nil {
		return errors.New("claude: nil client")
	}
	if strings.TrimSpace(c.apiKey) != "" || strings.TrimSpace(c.authToken) != "" {
		return nil
	}
	if envKey := strings.TrimSpace(os.Getenv("ANTHROPIC_API_KEY")); envKey != "" {
		c.apiKey = envKey
		return nil
	}
	if envToken := strings.TrimSpace(os.Getenv("ANTHROPIC_AUTH_TOKEN")); envToken != "" {
		c.authToken = envToken
		return nil
	}
	return errors.New("claude: missing api key")
}

func (c *Client) newSDKClient() *anthropic.Client {
	opts := make([]option.RequestOption, 0, 4)
	if base := strings.TrimSpace(c.baseURL); base != "" {
		opts = append(opts, option.WithBaseURL(sdkBaseURL(base)))
	}
	if c.httpClient != nil {
		opts = append(opts, option.WithHTTPClient(c.httpClient))
	}
	if strings.TrimSpace(c.apiKey) != "" {
		opts = append(opts, option.WithAPIKey(c.apiKey))
	} else if strings.TrimSpace(c.authToken) != "" {
		opts = append(opts, option.WithAuthToken(c.authToken))
	}
	opts = append(opts, option.WithMaxRetries(0))
	opts = append(opts, option.WithHeader("anthropic-version", apiVersionHeader))

	client := anthropic.NewClient(opts...)
	return &client
}

func sdkBaseURL(base string) string {
	base = strings.TrimSpace(strings.TrimRight(base, "/"))
	if strings.HasSuffix(base, "/v1") {
		base = strings.TrimSuffix(base, "/v1")
	}
	return base
}

func buildMessageParams(req *Request, messages []anthropic.MessageParam) anthropic.MessageNewParams {
	params := anthropic.MessageNewParams{
		Model:     anthropic.Model(req.Model),
		MaxTokens: int64(req.MaxTokens),
		Messages:  messages,
	}

	if system := strings.TrimSpace(req.System); system != "" {
		params.System = []anthropic.TextBlockParam{{
			Text: system,
			Type: "text",
		}}
	}
	if req.Temperature != 0 {
		params.Temperature = param.NewOpt(req.Temperature)
	}
	if len(req.Tools) > 0 {
		params.Tools = toSDKTools(req.Tools)
	}
	return params
}

func toSDKMessages(msgs []Message) []anthropic.MessageParam {
	out := make([]anthropic.MessageParam, 0, len(msgs))
	for _, m := range msgs {
		out = append(out, toSDKMessage(m.Role, []anthropic.ContentBlockParamUnion{
			anthropic.NewTextBlock(m.Content),
		}))
	}
	return out
}

func toSDKMessage(role string, blocks []anthropic.ContentBlockParamUnion) anthropic.MessageParam {
	return anthropic.MessageParam{
		Role:    toSDKRole(role),
		Content: blocks,
	}
}

func toSDKRole(role string) anthropic.MessageParamRole {
	if strings.EqualFold(strings.TrimSpace(role), "assistant") {
		return anthropic.MessageParamRoleAssistant
	}
	return anthropic.MessageParamRoleUser
}

func contentBlocksToSDK(blocks []ContentBlock) []anthropic.ContentBlockParamUnion {
	if len(blocks) == 0 {
		return nil
	}

	out := make([]anthropic.ContentBlockParamUnion, 0, len(blocks))
	for _, b := range blocks {
		switch b.Type {
		case "text":
			out = append(out, anthropic.NewTextBlock(b.Text))
		case "tool_use":
			out = append(out, anthropic.NewToolUseBlock(b.ID, b.Input, b.Name))
		case "tool_result":
			out = append(out, anthropic.NewToolResultBlock(b.ToolUseID, b.Content, b.IsError))
		}
	}
	return out
}

func toSDKTools(tools []ToolDefinition) []anthropic.ToolUnionParam {
	out := make([]anthropic.ToolUnionParam, 0, len(tools))
	for _, t := range tools {
		tool := anthropic.ToolParam{
			Name:        t.Name,
			InputSchema: toSDKToolInputSchema(t.InputSchema),
		}
		if desc := strings.TrimSpace(t.Description); desc != "" {
			tool.Description = param.NewOpt(desc)
		}
		out = append(out, anthropic.ToolUnionParam{OfTool: &tool})
	}
	return out
}

func toSDKToolInputSchema(schema map[string]any) anthropic.ToolInputSchemaParam {
	out := anthropic.ToolInputSchemaParam{Type: "object"}
	if schema == nil {
		return out
	}

	if props, ok := schema["properties"]; ok {
		out.Properties = props
	}
	if required, ok := schema["required"]; ok {
		out.Required = toStringSlice(required)
	}

	extra := make(map[string]any)
	for k, v := range schema {
		if k == "properties" || k == "required" || k == "type" {
			continue
		}
		extra[k] = v
	}
	if len(extra) > 0 {
		out.ExtraFields = extra
	}

	return out
}

func toStringSlice(v any) []string {
	switch value := v.(type) {
	case []string:
		return value
	case []any:
		out := make([]string, 0, len(value))
		for _, item := range value {
			if s, ok := item.(string); ok {
				out = append(out, s)
			}
		}
		return out
	default:
		return nil
	}
}

func fromSDKMessage(msg *anthropic.Message) *Response {
	if msg == nil {
		return nil
	}

	resp := &Response{
		ID:         msg.ID,
		Type:       string(msg.Type),
		Role:       string(msg.Role),
		Model:      string(msg.Model),
		StopReason: string(msg.StopReason),
		Usage: Usage{
			InputTokens:  int(msg.Usage.InputTokens),
			OutputTokens: int(msg.Usage.OutputTokens),
		},
	}

	if len(msg.Content) == 0 {
		return resp
	}

	resp.Content = make([]ContentBlock, 0, len(msg.Content))
	for _, block := range msg.Content {
		switch block.Type {
		case "text":
			text := block.AsText()
			resp.Content = append(resp.Content, ContentBlock{
				Type: "text",
				Text: text.Text,
			})
		case "tool_use":
			tool := block.AsToolUse()
			resp.Content = append(resp.Content, ContentBlock{
				Type:  "tool_use",
				ID:    tool.ID,
				Name:  tool.Name,
				Input: decodeToolInput(tool.Input),
			})
		}
	}

	return resp
}

func decodeToolInput(raw json.RawMessage) map[string]any {
	if len(raw) == 0 {
		return nil
	}

	var out map[string]any
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil
	}
	return out
}

func clampRetryMax(maxRetries int) int {
	if maxRetries < 0 {
		return 0
	}
	if maxRetries > maxRetryMax {
		return maxRetryMax
	}
	return maxRetries
}

func retryBackoff(base time.Duration, attempt int) time.Duration {
	if base <= 0 {
		return 0
	}
	if attempt < 0 {
		return 0
	}
	return base * time.Duration(1<<attempt)
}

func shouldRetry(err error) bool {
	if err == nil {
		return false
	}

	var apiErr *APIError
	if errors.As(err, &apiErr) {
		return apiErr.StatusCode >= 500 && apiErr.StatusCode <= 599
	}

	var sdkErr *anthropic.Error
	if errors.As(err, &sdkErr) {
		return sdkErr.StatusCode >= 500 && sdkErr.StatusCode <= 599
	}

	var netErr net.Error
	return errors.As(err, &netErr) && netErr.Timeout()
}

func sleepWithContext(ctx context.Context, d time.Duration) error {
	if d <= 0 {
		return nil
	}
	timer := time.NewTimer(d)
	defer timer.Stop()

	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}
