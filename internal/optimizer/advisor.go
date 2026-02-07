package optimizer

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strings"

	"github.com/stellarlinkco/ai-eval/internal/llm"
	"github.com/stellarlinkco/ai-eval/internal/runner"
)

// Advisor diagnoses prompt failures and proposes targeted fixes.
type Advisor struct {
	Provider llm.Provider
}

// DiagnoseRequest contains the prompt and evaluation results for diagnosis.
type DiagnoseRequest struct {
	PromptContent  string
	EvalResults    []*runner.SuiteResult
	MaxSuggestions int // default: 5
}

// DiagnoseResult is the diagnosis output.
type DiagnoseResult struct {
	FailurePatterns []string        `json:"failure_patterns"`
	RootCauses      []string        `json:"root_causes"`
	Suggestions     []FixSuggestion `json:"suggestions"`
}

// FixSuggestion describes a targeted fix.
type FixSuggestion struct {
	ID          string `json:"id"`
	Type        string `json:"type"`
	Description string `json:"description"`
	Before      string `json:"before,omitempty"`
	After       string `json:"after,omitempty"`
	Impact      string `json:"impact,omitempty"`
	Priority    int    `json:"priority"`
}

const diagnosePromptTemplate = `You are a prompt debugging advisor. Analyze failures and propose the smallest effective prompt changes.

## Prompt
<prompt>
{{PROMPT}}
</prompt>

## Known Failure Patterns (choose by id)
{{PATTERNS}}

## Heuristic Pattern Hints (optional, may be wrong)
{{HINTS}}

## Evaluation Results (failures only)
{{EVAL_RESULTS}}

## Your Task
1. Pick which failure patterns apply (ids only; pick from Known Failure Patterns).
2. Explain the root causes (short bullets, concrete).
3. Propose up to {{MAX_SUGGESTIONS}} fix suggestions.

## Suggestion Rules
- Prefer minimal prompt edits; keep original intent.
- Make output format requirements explicit when needed.
- Include at least one suggestion with type="rewrite_prompt" whose "after" is the FULL revised prompt.
- Each suggestion must include: id, type, description, before, after, impact, priority.
- priority: integer 1 (highest) to 5 (lowest).
- impact: low|medium|high.

## Output Format
Return ONLY valid JSON, no markdown, no code fences:
{
  "failure_patterns": ["missing_context", "output_format_unclear"],
  "root_causes": ["..."],
  "suggestions": [
    {
      "id": "S1",
      "type": "clarify_instruction|add_context|add_examples|resolve_constraints|specify_output_format|handle_edge_cases|rewrite_prompt",
      "description": "...",
      "before": "...",
      "after": "...",
      "impact": "low|medium|high",
      "priority": 1
    }
  ]
}`

// Diagnose analyzes evaluation failures and generates fix suggestions.
func (a *Advisor) Diagnose(ctx context.Context, req *DiagnoseRequest) (*DiagnoseResult, error) {
	if a == nil || a.Provider == nil {
		return nil, errors.New("advisor: nil provider")
	}
	if ctx == nil {
		return nil, errors.New("advisor: nil context")
	}
	if req == nil {
		return nil, errors.New("advisor: nil request")
	}

	promptContent := req.PromptContent
	if strings.TrimSpace(promptContent) == "" {
		return nil, errors.New("advisor: empty prompt_content")
	}

	maxSuggestions := req.MaxSuggestions
	if maxSuggestions <= 0 {
		maxSuggestions = 5
	}
	if maxSuggestions > 20 {
		maxSuggestions = 20
	}

	evalSummary := formatEvalResultsForDiagnosis(req.EvalResults)
	patternsText := formatPatternRules(CommonPatterns)
	hints := PatternMatcher{}.Match(promptContent, evalSummary)
	hintsText := "[]"
	if len(hints) > 0 {
		hintsText = strings.Join(hints, ", ")
	}

	prompt := strings.ReplaceAll(diagnosePromptTemplate, "{{PROMPT}}", promptContent)
	prompt = strings.ReplaceAll(prompt, "{{PATTERNS}}", patternsText)
	prompt = strings.ReplaceAll(prompt, "{{HINTS}}", hintsText)
	prompt = strings.ReplaceAll(prompt, "{{EVAL_RESULTS}}", evalSummary)
	prompt = strings.ReplaceAll(prompt, "{{MAX_SUGGESTIONS}}", fmt.Sprintf("%d", maxSuggestions))

	resp, err := a.Provider.Complete(ctx, &llm.Request{
		Messages:  []llm.Message{{Role: "user", Content: prompt}},
		MaxTokens: 8192,
	})
	if err != nil {
		return nil, fmt.Errorf("advisor: %w", err)
	}
	if resp == nil {
		return nil, errors.New("advisor: nil llm response")
	}

	raw := strings.TrimSpace(llm.Text(resp))
	var parsed DiagnoseResult
	if err := llm.ParseJSON(raw, &parsed); err != nil {
		return nil, fmt.Errorf("advisor: failed to parse response: %w (response length: %d)", err, len(raw))
	}

	parsed.FailurePatterns = normalizePatternIDs(parsed.FailurePatterns)
	parsed.RootCauses = trimStringSlice(parsed.RootCauses)
	parsed.Suggestions = normalizeSuggestions(parsed.Suggestions, maxSuggestions)

	return &parsed, nil
}

func normalizePatternIDs(ids []string) []string {
	seen := make(map[string]struct{}, len(ids))
	out := ids[:0]
	for _, id := range ids {
		id = strings.TrimSpace(id)
		if id == "" {
			continue
		}
		id = strings.ToLower(id)
		if _, ok := seen[id]; ok {
			continue
		}
		seen[id] = struct{}{}
		out = append(out, id)
	}
	sort.Strings(out)
	return out
}

func trimStringSlice(in []string) []string {
	out := in[:0]
	for _, s := range in {
		s = strings.TrimSpace(s)
		if s == "" {
			continue
		}
		out = append(out, s)
	}
	return out
}

func normalizeSuggestions(in []FixSuggestion, max int) []FixSuggestion {
	out := in[:0]
	for _, s := range in {
		s.ID = strings.TrimSpace(s.ID)
		s.Type = strings.TrimSpace(s.Type)
		s.Description = strings.TrimSpace(s.Description)
		s.Before = strings.TrimSpace(s.Before)
		s.After = strings.TrimSpace(s.After)
		s.Impact = strings.TrimSpace(s.Impact)
		if s.ID == "" || s.Type == "" || s.Description == "" {
			continue
		}
		if s.Priority <= 0 {
			s.Priority = 3
		}
		if s.Priority > 5 {
			s.Priority = 5
		}
		out = append(out, s)
		if max > 0 && len(out) >= max {
			break
		}
	}

	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Priority != out[j].Priority {
			return out[i].Priority < out[j].Priority
		}
		return out[i].ID < out[j].ID
	})

	return out
}

func formatPatternRules(rules []PatternRule) string {
	if len(rules) == 0 {
		return "(none)"
	}
	var sb strings.Builder
	for _, r := range rules {
		id := strings.TrimSpace(r.ID)
		if id == "" {
			continue
		}
		title := strings.TrimSpace(r.Title)
		desc := strings.TrimSpace(r.Description)
		if title == "" {
			title = id
		}
		if desc == "" {
			desc = "-"
		}
		sb.WriteString(fmt.Sprintf("- %s: %s\n  %s\n", id, title, desc))
	}
	return strings.TrimSpace(sb.String())
}

func formatEvalResultsForDiagnosis(results []*runner.SuiteResult) string {
	if len(results) == 0 {
		return "No evaluation results available."
	}

	var sb strings.Builder
	for _, res := range results {
		if res == nil {
			continue
		}
		sb.WriteString(fmt.Sprintf("## Suite: %s\n", strings.TrimSpace(res.Suite)))
		sb.WriteString(fmt.Sprintf("Pass Rate: %.1f%% (%d/%d cases)\n", res.PassRate*100, res.PassedCases, res.TotalCases))
		sb.WriteString(fmt.Sprintf("Average Score: %.2f\n\n", res.AvgScore))

		failed := 0
		for _, rr := range res.Results {
			if rr.Passed {
				continue
			}
			failed++
			sb.WriteString(fmt.Sprintf("### Case: %s (score=%.3f pass@k=%.3f)\n", rr.CaseID, rr.Score, rr.PassAtK))
			if rr.Error != nil {
				sb.WriteString(fmt.Sprintf("Error: %s\n", strings.TrimSpace(rr.Error.Error())))
			}

			// Include the first failing trial as evidence.
			for _, tr := range rr.Trials {
				if tr.Passed {
					continue
				}

				resp := strings.TrimSpace(tr.Response)
				if resp != "" {
					sb.WriteString("Response (sample):\n")
					sb.WriteString(indentBlock(truncate(resp, 1200), "  "))
					sb.WriteByte('\n')
				}

				for _, ev := range tr.Evaluations {
					if ev.Passed {
						continue
					}
					msg := strings.TrimSpace(ev.Message)
					if msg == "" {
						continue
					}
					sb.WriteString(fmt.Sprintf("- Failed: %s\n", msg))
				}
				break
			}
			sb.WriteByte('\n')
		}

		if failed == 0 {
			sb.WriteString("All cases passed.\n\n")
		}
	}

	out := strings.TrimSpace(sb.String())
	if out == "" {
		return "No evaluation results available."
	}
	return out
}

func truncate(s string, max int) string {
	if max <= 0 {
		return ""
	}
	if len(s) <= max {
		return s
	}
	return s[:max] + "...(truncated)"
}

func indentBlock(s string, prefix string) string {
	s = strings.ReplaceAll(s, "\r\n", "\n")
	lines := strings.Split(s, "\n")
	for i := range lines {
		lines[i] = prefix + lines[i]
	}
	return strings.Join(lines, "\n")
}
