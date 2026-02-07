package evaluator

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"text/template"

	"github.com/stellarlinkco/ai-eval/internal/llm"
)

// LLMJudgeEvaluator scores responses against criteria using an LLM provider.
type LLMJudgeEvaluator struct {
	Provider       llm.Provider
	Criteria       string   // Evaluation criteria description
	Rubric         []string // Scoring dimensions (optional)
	ScoreScale     int      // 1-5 or 1-10, default 5
	ScoreThreshold float64  // Pass threshold (0.0-1.0), default 0.6
}

// Name returns the evaluator identifier.
func (LLMJudgeEvaluator) Name() string {
	return "llm_judge"
}

const llmJudgePromptTemplate = `You are an expert evaluator. Assess the AI response based on the given criteria.

## Evaluation Criteria
{{.Criteria}}

{{if .Rubric}}
## Scoring Dimensions
{{range .Rubric}}- {{.}}
{{end}}
{{end}}

## Original Question/Context
{{.Context}}

## AI Response to Evaluate
{{.Response}}

## Instructions
Rate the response on a scale of 1-{{.ScoreScale}}.
- 1: Completely fails to meet criteria
- {{.ScoreScale}}: Perfectly meets all criteria

Output ONLY valid JSON in this exact format:
{"score": <integer 1-{{.ScoreScale}}>, "reasoning": "<brief explanation>"}`

var llmJudgePromptTmpl = template.Must(template.New("llm_judge").Parse(llmJudgePromptTemplate))

type llmJudgePromptData struct {
	Criteria   string
	Rubric     []string
	Context    string
	Response   string
	ScoreScale int
}

type llmJudgeOutput struct {
	Score     int    `json:"score"`
	Reasoning string `json:"reasoning"`
}

// Evaluate uses Claude to score the response and normalizes the result.
func (e *LLMJudgeEvaluator) Evaluate(ctx context.Context, response string, expected any) (*Result, error) {
	if e == nil {
		return nil, errors.New("llm_judge: nil evaluator")
	}
	if e.Provider == nil {
		return nil, errors.New("llm_judge: nil llm provider")
	}

	criteria := strings.TrimSpace(e.Criteria)
	rubric := e.Rubric
	scoreScale := e.ScoreScale
	scoreThreshold := e.ScoreThreshold
	contextText := ""

	if scoreScale <= 0 {
		scoreScale = 5
	}
	if scoreScale < 2 {
		scoreScale = 2
	}
	if scoreThreshold <= 0 {
		scoreThreshold = 0.6
	}
	if scoreThreshold < 0 {
		scoreThreshold = 0
	}
	if scoreThreshold > 1 {
		scoreThreshold = 1
	}

	switch v := expected.(type) {
	case nil:
	case string:
		if s := strings.TrimSpace(v); s != "" {
			criteria = s
		}
	case map[string]any:
		if raw, ok := v["criteria"]; ok {
			s, ok := raw.(string)
			if !ok {
				return nil, fmt.Errorf("llm_judge: expected.criteria must be string, got %T", raw)
			}
			criteria = strings.TrimSpace(s)
		}
		if raw, ok := v["context"]; ok {
			s, ok := raw.(string)
			if !ok {
				return nil, fmt.Errorf("llm_judge: expected.context must be string, got %T", raw)
			}
			contextText = s
		}
		if raw, ok := v["rubric"]; ok {
			if raw == nil {
				rubric = nil
			} else {
				rr, err := asStringSlice(raw)
				if err != nil {
					return nil, fmt.Errorf("llm_judge: expected.rubric: %w", err)
				}
				rubric = rr
			}
		}
		if raw, ok := v["score_scale"]; ok {
			n, ok := asInt(raw)
			if !ok {
				return nil, fmt.Errorf("llm_judge: expected.score_scale must be number, got %T", raw)
			}
			scoreScale = n
		}
		if raw, ok := v["score_threshold"]; ok {
			f, ok := asFloat(raw)
			if !ok {
				return nil, fmt.Errorf("llm_judge: expected.score_threshold must be number, got %T", raw)
			}
			scoreThreshold = f
		}
	default:
		return nil, fmt.Errorf("llm_judge: expected must be string or map[string]any, got %T", expected)
	}

	if scoreScale <= 0 {
		scoreScale = 5
	}
	if scoreScale < 2 {
		scoreScale = 2
	}
	if scoreThreshold <= 0 {
		scoreThreshold = 0.6
	}
	if scoreThreshold < 0 {
		scoreThreshold = 0
	}
	if scoreThreshold > 1 {
		scoreThreshold = 1
	}
	if criteria == "" {
		return nil, errors.New("llm_judge: missing criteria")
	}

	var promptBuf bytes.Buffer
	if err := llmJudgePromptTmpl.Execute(&promptBuf, llmJudgePromptData{
		Criteria:   criteria,
		Rubric:     rubric,
		Context:    contextText,
		Response:   response,
		ScoreScale: scoreScale,
	}); err != nil {
		return nil, fmt.Errorf("llm_judge: render prompt: %w", err)
	}

	resp, err := e.Provider.Complete(ctx, &llm.Request{
		Messages:  []llm.Message{{Role: "user", Content: promptBuf.String()}},
		MaxTokens: 512,
	})
	if err != nil {
		return nil, fmt.Errorf("llm_judge: llm: %w", err)
	}
	if resp == nil {
		return nil, errors.New("llm_judge: nil llm response")
	}

	raw := strings.TrimSpace(llm.Text(resp))
	var out llmJudgeOutput
	if err := llm.ParseJSON(raw, &out); err != nil {
		return &Result{
			Passed:  false,
			Score:   0.0,
			Message: "invalid judge output",
			Details: map[string]any{
				"error":  err.Error(),
				"output": raw,
			},
		}, nil
	}

	if out.Score < 1 || out.Score > scoreScale {
		return &Result{
			Passed:  false,
			Score:   0.0,
			Message: "judge score out of range",
			Details: map[string]any{
				"score":       out.Score,
				"score_scale": scoreScale,
				"output":      raw,
			},
		}, nil
	}

	score := normalizeLikert(out.Score, scoreScale)
	passed := score >= scoreThreshold
	reasoning := strings.TrimSpace(out.Reasoning)
	if reasoning == "" {
		reasoning = "no reasoning provided"
	}

	return &Result{
		Passed:  passed,
		Score:   score,
		Message: reasoning,
		Details: map[string]any{
			"raw_score":       out.Score,
			"score_scale":     scoreScale,
			"score_threshold": scoreThreshold,
		},
	}, nil
}

func normalizeLikert(score int, scale int) float64 {
	if scale <= 1 {
		return 0
	}
	if score <= 1 {
		return 0
	}
	if score >= scale {
		return 1
	}
	return float64(score-1) / float64(scale-1)
}

func asInt(v any) (int, bool) {
	switch n := v.(type) {
	case int:
		return n, true
	case int8:
		return int(n), true
	case int16:
		return int(n), true
	case int32:
		return int(n), true
	case int64:
		return int(n), true
	case uint:
		return int(n), true
	case uint8:
		return int(n), true
	case uint16:
		return int(n), true
	case uint32:
		return int(n), true
	case uint64:
		return int(n), true
	case float32:
		return int(n), true
	case float64:
		return int(n), true
	case json.Number:
		i, err := n.Int64()
		if err != nil {
			return 0, false
		}
		return int(i), true
	default:
		return 0, false
	}
}

func asFloat(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case float32:
		return float64(n), true
	case int:
		return float64(n), true
	case int8:
		return float64(n), true
	case int16:
		return float64(n), true
	case int32:
		return float64(n), true
	case int64:
		return float64(n), true
	case uint:
		return float64(n), true
	case uint8:
		return float64(n), true
	case uint16:
		return float64(n), true
	case uint32:
		return float64(n), true
	case uint64:
		return float64(n), true
	case json.Number:
		f, err := n.Float64()
		if err != nil {
			return 0, false
		}
		return f, true
	default:
		return 0, false
	}
}
