package optimizer

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/stellarlinkco/ai-eval/internal/llm"
	"github.com/stellarlinkco/ai-eval/internal/runner"
)

// Optimizer optimizes prompts based on evaluation results.
type Optimizer struct {
	Provider llm.Provider
}

// OptimizeRequest contains the prompt and evaluation results.
type OptimizeRequest struct {
	OriginalPrompt string
	EvalResults    *runner.SuiteResult
	MaxIterations  int // Max optimization iterations (default: 1)
}

// OptimizeResult contains the optimized prompt and analysis.
type OptimizeResult struct {
	OptimizedPrompt  string
	Changes          []Change
	Summary          string
	ScoreImprovement float64
}

// Change describes a specific change made to the prompt.
type Change struct {
	Type        string // "add", "remove", "modify", "restructure"
	Description string
	Before      string
	After       string
}

const optimizePrompt = `You are a prompt engineering expert. Analyze the evaluation results and optimize the prompt.

## Original Prompt
<prompt>
{{PROMPT}}
</prompt>

## Evaluation Results
{{EVAL_RESULTS}}

## Your Task
1. Analyze why certain test cases failed
2. Identify weaknesses in the prompt
3. Optimize the prompt to address these issues while maintaining its core purpose

## Optimization Guidelines
- Be specific and actionable
- Maintain the original intent
- Add constraints where behavior is ambiguous
- Use clear, unambiguous language
- Add examples if helpful
- Structure with clear sections

## Output Format
Return a JSON object with optimized_prompt FIRST (this is the most important field):
{
  "optimized_prompt": "The complete optimized prompt (FULL text, not a diff)",
  "summary": "Brief summary of issues found and fixes applied",
  "changes": [
    {
      "type": "add|remove|modify|restructure",
      "description": "What was changed and why"
    }
  ]
}

IMPORTANT: Return ONLY valid JSON, no markdown code blocks. The optimized_prompt MUST be the complete rewritten prompt.`

// Optimize improves a prompt based on evaluation results.
func (o *Optimizer) Optimize(ctx context.Context, req *OptimizeRequest) (*OptimizeResult, error) {
	if o == nil || o.Provider == nil {
		return nil, errors.New("optimizer: nil provider")
	}
	if req == nil {
		return nil, errors.New("optimizer: nil request")
	}
	if strings.TrimSpace(req.OriginalPrompt) == "" {
		return nil, errors.New("optimizer: empty prompt")
	}

	evalSummary := formatEvalResults(req.EvalResults)

	prompt := strings.ReplaceAll(optimizePrompt, "{{PROMPT}}", req.OriginalPrompt)
	prompt = strings.ReplaceAll(prompt, "{{EVAL_RESULTS}}", evalSummary)

	resp, err := o.Provider.Complete(ctx, &llm.Request{
		Messages:  []llm.Message{{Role: "user", Content: prompt}},
		MaxTokens: 16384,
	})
	if err != nil {
		return nil, fmt.Errorf("optimizer: %w", err)
	}

	text := strings.TrimSpace(llm.Text(resp))
	text = strings.TrimPrefix(text, "```json")
	text = strings.TrimPrefix(text, "```")
	text = strings.TrimSuffix(text, "```")
	text = strings.TrimSpace(text)

	var parsed struct {
		Summary string `json:"summary"`
		Changes []struct {
			Type        string `json:"type"`
			Description string `json:"description"`
			Before      string `json:"before"`
			After       string `json:"after"`
		} `json:"changes"`
		OptimizedPrompt string `json:"optimized_prompt"`
	}

	if err := json.Unmarshal([]byte(text), &parsed); err != nil {
		// Fallback: extract optimized_prompt from truncated JSON
		extracted := extractJSONStringField(text, "optimized_prompt")
		if extracted != "" {
			summary := extractJSONStringField(text, "summary")
			return &OptimizeResult{
				OptimizedPrompt: extracted,
				Summary:         summary,
			}, nil
		}
		return nil, fmt.Errorf("optimizer: failed to parse response: %w (response length: %d)", err, len(text))
	}

	result := &OptimizeResult{
		OptimizedPrompt: parsed.OptimizedPrompt,
		Summary:         parsed.Summary,
		Changes:         make([]Change, 0, len(parsed.Changes)),
	}

	for _, c := range parsed.Changes {
		result.Changes = append(result.Changes, Change{
			Type:        c.Type,
			Description: c.Description,
			Before:      c.Before,
			After:       c.After,
		})
	}

	if req.EvalResults != nil {
		result.ScoreImprovement = 1.0 - req.EvalResults.AvgScore
	}

	return result, nil
}

// extractJSONStringField extracts a string value for a given key from
// potentially truncated JSON. It handles escaped characters in the value.
func extractJSONStringField(text string, key string) string {
	needle := `"` + key + `":`
	idx := strings.Index(text, needle)
	if idx < 0 {
		return ""
	}
	rest := text[idx+len(needle):]
	rest = strings.TrimSpace(rest)
	if len(rest) == 0 || rest[0] != '"' {
		return ""
	}
	var sb strings.Builder
	i := 1 // skip opening quote
	for i < len(rest) {
		ch := rest[i]
		if ch == '\\' && i+1 < len(rest) {
			next := rest[i+1]
			switch next {
			case '"':
				sb.WriteByte('"')
			case '\\':
				sb.WriteByte('\\')
			case 'n':
				sb.WriteByte('\n')
			case 't':
				sb.WriteByte('\t')
			case 'r':
				sb.WriteByte('\r')
			default:
				sb.WriteByte('\\')
				sb.WriteByte(next)
			}
			i += 2
			continue
		}
		if ch == '"' {
			return sb.String()
		}
		sb.WriteByte(ch)
		i++
	}
	return sb.String()
}

func formatEvalResults(results *runner.SuiteResult) string {
	if results == nil {
		return "No evaluation results available."
	}

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("Pass Rate: %.1f%% (%d/%d cases)\n", results.PassRate*100, results.PassedCases, results.TotalCases))
	sb.WriteString(fmt.Sprintf("Average Score: %.2f\n\n", results.AvgScore))

	sb.WriteString("## Case Results\n")
	for _, r := range results.Results {
		status := "✅ PASS"
		if !r.Passed {
			status = "❌ FAIL"
		}
		sb.WriteString(fmt.Sprintf("\n### %s: %s (Score: %.2f)\n", r.CaseID, status, r.Score))

		for _, trial := range r.Trials {
			if len(trial.Evaluations) > 0 {
				for _, eval := range trial.Evaluations {
					if !eval.Passed {
						sb.WriteString(fmt.Sprintf("- Failed: %s\n", eval.Message))
					}
				}
			}
		}
	}

	return sb.String()
}
