package evaluator

import (
	"encoding/json"
	"fmt"
	"reflect"
	"regexp"
	"strings"

	"github.com/stellarlinkco/ai-eval/internal/llm"
	"github.com/stellarlinkco/ai-eval/internal/testcase"
)

// ToolCallExpect aliases testcase.ToolCallExpect for tool call matching.
type ToolCallExpect = testcase.ToolCallExpect

// ToolCallEvaluator compares tool calls to expectations.
type ToolCallEvaluator struct {
	Expected []ToolCallExpect
}

type toolCallMismatch struct {
	Name   string
	Order  int
	Reason string
}

// Evaluate scores tool call matches against expectations.
func (e ToolCallEvaluator) Evaluate(got []llm.ToolUse) Result {
	total := countExpectedToolCalls(e.Expected)
	if total == 0 {
		return Result{
			Passed:  true,
			Score:   1,
			Message: "no expected tool calls",
		}
	}

	used := make([]bool, len(got))
	matched := 0
	missingRequired := make([]string, 0)
	mismatches := make([]toolCallMismatch, 0)

	for _, exp := range e.Expected {
		name := strings.TrimSpace(exp.Name)
		if name == "" || exp.Order <= 0 {
			continue
		}

		idx, ok, reason := matchAtOrder(got, used, exp)
		if !ok {
			if exp.Required {
				missingRequired = append(missingRequired, name)
			}
			mismatches = append(mismatches, toolCallMismatch{
				Name:   name,
				Order:  exp.Order,
				Reason: reason,
			})
			continue
		}

		used[idx] = true
		matched++
	}

	for _, exp := range e.Expected {
		name := strings.TrimSpace(exp.Name)
		if name == "" || exp.Order > 0 {
			continue
		}

		idx, reason := findUnorderedMatch(got, used, exp)
		if idx < 0 {
			if exp.Required {
				missingRequired = append(missingRequired, name)
			}
			mismatches = append(mismatches, toolCallMismatch{
				Name:   name,
				Order:  exp.Order,
				Reason: reason,
			})
			continue
		}

		used[idx] = true
		matched++
	}

	score := float64(matched) / float64(total)
	passed := len(missingRequired) == 0

	details := map[string]any{
		"matched": matched,
		"total":   total,
	}
	if len(missingRequired) > 0 {
		details["missing_required"] = missingRequired
	}
	if len(mismatches) > 0 {
		out := make([]map[string]any, 0, len(mismatches))
		for _, m := range mismatches {
			out = append(out, map[string]any{
				"name":   m.Name,
				"order":  m.Order,
				"reason": m.Reason,
			})
		}
		details["mismatches"] = out
	}

	msg := fmt.Sprintf("matched %d/%d tool calls", matched, total)
	if matched == total {
		msg = "tool calls match"
	}

	return Result{
		Passed:  passed,
		Score:   score,
		Message: msg,
		Details: details,
	}
}

func countExpectedToolCalls(expected []ToolCallExpect) int {
	total := 0
	for _, exp := range expected {
		if strings.TrimSpace(exp.Name) == "" {
			continue
		}
		total++
	}
	return total
}

func matchAtOrder(got []llm.ToolUse, used []bool, exp ToolCallExpect) (int, bool, string) {
	name := strings.TrimSpace(exp.Name)
	if name == "" {
		return -1, false, "missing name"
	}
	if exp.Order <= 0 {
		return -1, false, "order must be > 0"
	}

	idx := exp.Order - 1
	if idx < 0 || idx >= len(got) {
		return -1, false, fmt.Sprintf("missing tool call at order %d", exp.Order)
	}
	if idx < len(used) && used[idx] {
		return -1, false, fmt.Sprintf("tool call at order %d already matched", exp.Order)
	}

	call := got[idx]
	gotName := strings.TrimSpace(call.Name)
	if gotName != name {
		return -1, false, fmt.Sprintf("order %d: got name=%q want=%q", exp.Order, gotName, name)
	}

	ok, reason := toolArgsSubsetMatch(call.Input, exp.ArgsMatch)
	if !ok {
		return -1, false, reason
	}
	return idx, true, ""
}

func findUnorderedMatch(got []llm.ToolUse, used []bool, exp ToolCallExpect) (int, string) {
	name := strings.TrimSpace(exp.Name)
	if name == "" {
		return -1, "missing name"
	}

	var firstMismatch string
	for i, call := range got {
		if i < len(used) && used[i] {
			continue
		}
		if strings.TrimSpace(call.Name) != name {
			continue
		}

		ok, reason := toolArgsSubsetMatch(call.Input, exp.ArgsMatch)
		if ok {
			return i, ""
		}
		if firstMismatch == "" && reason != "" {
			firstMismatch = reason
		}
	}
	if firstMismatch != "" {
		return -1, firstMismatch
	}
	return -1, "tool not called"
}

func toolArgsSubsetMatch(got map[string]any, want map[string]any) (bool, string) {
	if len(want) == 0 {
		return true, ""
	}
	if got == nil {
		return false, "missing args"
	}

	for k, wantV := range want {
		gotV, ok := got[k]
		if !ok {
			return false, fmt.Sprintf("missing arg %q", k)
		}
		if ok, reason := matchValue(gotV, wantV, fmt.Sprintf("arg %q", k)); !ok {
			return false, reason
		}
	}
	return true, ""
}

func matchValue(got any, want any, path string) (bool, string) {
	if want == nil {
		if got == nil {
			return true, ""
		}
		return false, fmt.Sprintf("%s: got=%v want=nil", path, got)
	}

	if w, ok := want.(string); ok && strings.HasPrefix(w, "regex:") {
		pattern := strings.TrimPrefix(w, "regex:")
		re, err := regexp.Compile(pattern)
		if err != nil {
			return false, fmt.Sprintf("%s: invalid regex %q: %v", path, pattern, err)
		}
		s, ok := got.(string)
		if !ok {
			return false, fmt.Sprintf("%s: expected string to match regex %q, got %T", path, pattern, got)
		}
		if !re.MatchString(s) {
			return false, fmt.Sprintf("%s: regex %q did not match %q", path, pattern, s)
		}
		return true, ""
	}

	if equal, comparable := numericEqual(got, want); comparable {
		if equal {
			return true, ""
		}
		return false, fmt.Sprintf("%s: got=%v want=%v", path, got, want)
	}

	if wmap, ok := asStringAnyMap(want); ok {
		gmap, ok := asStringAnyMap(got)
		if !ok {
			return false, fmt.Sprintf("%s: expected object, got %T", path, got)
		}

		for k, wv := range wmap {
			gv, ok := gmap[k]
			if !ok {
				return false, fmt.Sprintf("%s.%s: missing", path, k)
			}
			if ok, reason := matchValue(gv, wv, path+"."+k); !ok {
				return false, reason
			}
		}
		return true, ""
	}

	if wslice, ok := asAnySlice(want); ok {
		gslice, ok := asAnySlice(got)
		if !ok {
			return false, fmt.Sprintf("%s: expected array, got %T", path, got)
		}
		if len(gslice) != len(wslice) {
			return false, fmt.Sprintf("%s: len=%d want=%d", path, len(gslice), len(wslice))
		}
		for i := range wslice {
			if ok, reason := matchValue(gslice[i], wslice[i], fmt.Sprintf("%s[%d]", path, i)); !ok {
				return false, reason
			}
		}
		return true, ""
	}

	if reflect.DeepEqual(got, want) {
		return true, ""
	}
	return false, fmt.Sprintf("%s: got=%v want=%v", path, got, want)
}

func asStringAnyMap(v any) (map[string]any, bool) {
	switch m := v.(type) {
	case map[string]any:
		return m, true
	case map[any]any:
		out := make(map[string]any, len(m))
		for k, v := range m {
			ks, ok := k.(string)
			if !ok {
				return nil, false
			}
			out[ks] = v
		}
		return out, true
	default:
		rv := reflect.ValueOf(v)
		if !rv.IsValid() || rv.Kind() != reflect.Map {
			return nil, false
		}
		if rv.Type().Key().Kind() != reflect.String {
			return nil, false
		}
		out := make(map[string]any, rv.Len())
		iter := rv.MapRange()
		for iter.Next() {
			out[iter.Key().String()] = iter.Value().Interface()
		}
		return out, true
	}
}

func asAnySlice(v any) ([]any, bool) {
	switch s := v.(type) {
	case []any:
		return s, true
	case []string:
		out := make([]any, len(s))
		for i := range s {
			out[i] = s[i]
		}
		return out, true
	default:
		rv := reflect.ValueOf(v)
		if !rv.IsValid() {
			return nil, false
		}
		if rv.Kind() != reflect.Slice && rv.Kind() != reflect.Array {
			return nil, false
		}
		out := make([]any, rv.Len())
		for i := 0; i < rv.Len(); i++ {
			out[i] = rv.Index(i).Interface()
		}
		return out, true
	}
}

func numericEqual(a any, b any) (equal bool, comparable bool) {
	af, ok := toFloat64(a)
	if !ok {
		return false, false
	}
	bf, ok := toFloat64(b)
	if !ok {
		return false, false
	}
	return af == bf, true
}

func toFloat64(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case float32:
		return float64(n), true
	case int:
		return float64(n), true
	case int64:
		return float64(n), true
	case int32:
		return float64(n), true
	case int16:
		return float64(n), true
	case int8:
		return float64(n), true
	case uint:
		return float64(n), true
	case uint64:
		return float64(n), true
	case uint32:
		return float64(n), true
	case uint16:
		return float64(n), true
	case uint8:
		return float64(n), true
	case json.Number:
		f, err := n.Float64()
		return f, err == nil
	default:
		return 0, false
	}
}
