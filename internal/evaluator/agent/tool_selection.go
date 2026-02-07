package agent

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strings"

	"github.com/stellarlinkco/ai-eval/internal/evaluator"
	"github.com/stellarlinkco/ai-eval/internal/llm"
)

// ToolSelectionEvaluator scores whether the agent selected expected tools.
type ToolSelectionEvaluator struct {
	Client llm.Provider
}

func (ToolSelectionEvaluator) Name() string {
	return "tool_selection"
}

func (e *ToolSelectionEvaluator) Evaluate(ctx context.Context, response string, expected any) (*evaluator.Result, error) {
	_ = ctx
	_ = response

	var expectedTools []string
	var toolCalls any

	switch v := expected.(type) {
	case nil:
	case map[string]any:
		if raw, ok := v["expected_tools"]; ok {
			ss, err := asStringSlice(raw)
			if err != nil {
				return nil, fmt.Errorf("tool_selection: expected.expected_tools: %w", err)
			}
			expectedTools = ss
		}
		if raw, ok := v["tool_calls"]; ok {
			toolCalls = raw
		}
	default:
		return nil, fmt.Errorf("tool_selection: expected must be map[string]any, got %T", expected)
	}

	actualTools, err := toolNamesFromAny(toolCalls)
	if err != nil {
		return nil, fmt.Errorf("tool_selection: expected.tool_calls: %w", err)
	}

	expectedSet := make(map[string]struct{}, len(expectedTools))
	for _, t := range expectedTools {
		name := strings.TrimSpace(t)
		if name == "" {
			continue
		}
		expectedSet[name] = struct{}{}
	}

	actualSet := make(map[string]struct{}, len(actualTools))
	for _, t := range actualTools {
		name := strings.TrimSpace(t)
		if name == "" {
			continue
		}
		actualSet[name] = struct{}{}
	}

	if len(expectedSet) == 0 {
		passed := len(actualSet) == 0
		score := 1.0
		if !passed {
			score = 0.0
		}
		return &evaluator.Result{
			Passed:  passed,
			Score:   score,
			Message: "no expected tools",
			Details: map[string]any{
				"unexpected_tools": setKeysSorted(actualSet),
			},
		}, nil
	}

	matched := 0
	missing := make([]string, 0)
	for name := range expectedSet {
		if _, ok := actualSet[name]; ok {
			matched++
			continue
		}
		missing = append(missing, name)
	}
	sort.Strings(missing)

	score := float64(matched) / float64(len(expectedSet))
	passed := matched == len(expectedSet)

	unexpected := make([]string, 0)
	for name := range actualSet {
		if _, ok := expectedSet[name]; ok {
			continue
		}
		unexpected = append(unexpected, name)
	}
	sort.Strings(unexpected)

	details := map[string]any{
		"matched": matched,
		"total":   len(expectedSet),
	}
	if len(missing) > 0 {
		details["missing_tools"] = missing
	}
	if len(unexpected) > 0 {
		details["unexpected_tools"] = unexpected
	}

	msg := fmt.Sprintf("matched %d/%d expected tools", matched, len(expectedSet))
	if passed {
		msg = "tool selection matches"
	}

	return &evaluator.Result{
		Passed:  passed,
		Score:   score,
		Message: msg,
		Details: details,
	}, nil
}

func toolNamesFromAny(v any) ([]string, error) {
	switch calls := v.(type) {
	case nil:
		return nil, errors.New("missing tool calls")
	case []llm.ToolUse:
		out := make([]string, 0, len(calls))
		for _, c := range calls {
			if name := strings.TrimSpace(c.Name); name != "" {
				out = append(out, name)
			}
		}
		return out, nil
	case []any:
		out := make([]string, 0, len(calls))
		for i, elem := range calls {
			switch tc := elem.(type) {
			case llm.ToolUse:
				if name := strings.TrimSpace(tc.Name); name != "" {
					out = append(out, name)
				}
			case map[string]any:
				raw, ok := tc["name"]
				if !ok {
					return nil, fmt.Errorf("tool_calls[%d]: missing name", i)
				}
				s, ok := raw.(string)
				if !ok {
					return nil, fmt.Errorf("tool_calls[%d].name: expected string, got %T", i, raw)
				}
				if name := strings.TrimSpace(s); name != "" {
					out = append(out, name)
				}
			default:
				return nil, fmt.Errorf("tool_calls[%d]: unsupported type %T", i, elem)
			}
		}
		return out, nil
	default:
		return nil, fmt.Errorf("unsupported type %T", v)
	}
}

func setKeysSorted(set map[string]struct{}) []string {
	if len(set) == 0 {
		return nil
	}
	out := make([]string, 0, len(set))
	for k := range set {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
