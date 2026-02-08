package evaluator

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/stellarlinkco/ai-eval/internal/llm"
	"github.com/stellarlinkco/ai-eval/internal/testcase"
)

func TestToolCallEvaluator_NoExpectedToolCalls(t *testing.T) {
	e := ToolCallEvaluator{
		Expected: []testcase.ToolCallExpect{
			{Name: " \t\n "},
		},
	}
	res := e.Evaluate([]llm.ToolUse{{Name: "x"}})
	if !res.Passed || res.Score != 1 {
		t.Fatalf("res=%#v", res)
	}
	if res.Message != "no expected tool calls" {
		t.Fatalf("msg=%q", res.Message)
	}
}

func TestCountExpectedToolCalls_TrimsNames(t *testing.T) {
	if got := countExpectedToolCalls([]testcase.ToolCallExpect{
		{Name: " "},
		{Name: "a"},
	}); got != 1 {
		t.Fatalf("got=%d", got)
	}
}

func TestMatchAtOrder_EdgeCases(t *testing.T) {
	got := []llm.ToolUse{{Name: "a", Input: map[string]any{"x": 1}}}

	{
		_, ok, reason := matchAtOrder(got, nil, testcase.ToolCallExpect{Name: " ", Order: 1})
		if ok || reason != "missing name" {
			t.Fatalf("ok=%v reason=%q", ok, reason)
		}
	}
	{
		_, ok, reason := matchAtOrder(got, nil, testcase.ToolCallExpect{Name: "a", Order: 0})
		if ok || reason != "order must be > 0" {
			t.Fatalf("ok=%v reason=%q", ok, reason)
		}
	}
	{
		_, ok, reason := matchAtOrder(got, nil, testcase.ToolCallExpect{Name: "a", Order: 2})
		if ok || !strings.Contains(reason, "missing tool call") {
			t.Fatalf("ok=%v reason=%q", ok, reason)
		}
	}
	{
		used := []bool{true}
		_, ok, reason := matchAtOrder(got, used, testcase.ToolCallExpect{Name: "a", Order: 1})
		if ok || !strings.Contains(reason, "already matched") {
			t.Fatalf("ok=%v reason=%q", ok, reason)
		}
	}
	{
		_, ok, reason := matchAtOrder(got, nil, testcase.ToolCallExpect{Name: "b", Order: 1})
		if ok || !strings.Contains(reason, "got name") {
			t.Fatalf("ok=%v reason=%q", ok, reason)
		}
	}
	{
		_, ok, reason := matchAtOrder(got, nil, testcase.ToolCallExpect{Name: "a", Order: 1, ArgsMatch: map[string]any{"y": 2}})
		if ok || !strings.Contains(reason, "missing arg") {
			t.Fatalf("ok=%v reason=%q", ok, reason)
		}
	}
}

func TestFindUnorderedMatch_EdgeCases(t *testing.T) {
	got := []llm.ToolUse{{Name: "a", Input: map[string]any{"x": 1}}}

	{
		idx, reason := findUnorderedMatch(got, nil, testcase.ToolCallExpect{Name: " "})
		if idx != -1 || reason != "missing name" {
			t.Fatalf("idx=%d reason=%q", idx, reason)
		}
	}
	{
		idx, reason := findUnorderedMatch(got, nil, testcase.ToolCallExpect{Name: "b"})
		if idx != -1 || reason != "tool not called" {
			t.Fatalf("idx=%d reason=%q", idx, reason)
		}
	}
	{
		idx, reason := findUnorderedMatch(got, nil, testcase.ToolCallExpect{Name: "a", ArgsMatch: map[string]any{"x": 2}})
		if idx != -1 || !strings.Contains(reason, "got=1 want=2") {
			t.Fatalf("idx=%d reason=%q", idx, reason)
		}
	}
}

func TestMatchValue_AdditionalBranches(t *testing.T) {
	{
		ok, reason := matchValue(1, 2, "$")
		if ok || !strings.Contains(reason, "got=1 want=2") {
			t.Fatalf("reason=%q", reason)
		}
	}
	{
		ok, reason := matchValue("x", map[string]any{"a": 1}, "$")
		if ok || !strings.Contains(reason, "expected object") {
			t.Fatalf("reason=%q", reason)
		}
	}
	{
		ok, reason := matchValue(map[string]any{"a": 1}, map[string]any{"b": 2}, "$")
		if ok || !strings.Contains(reason, "$.b: missing") {
			t.Fatalf("reason=%q", reason)
		}
	}
	{
		ok, reason := matchValue(map[string]any{"a": map[string]any{"x": 1}}, map[string]any{"a": map[string]any{"x": 2}}, "$")
		if ok || !strings.Contains(reason, "got=1 want=2") {
			t.Fatalf("reason=%q", reason)
		}
	}
	{
		ok, reason := matchValue("x", []any{1}, "$")
		if ok || !strings.Contains(reason, "expected array") {
			t.Fatalf("reason=%q", reason)
		}
	}
	{
		ok, reason := matchValue([]any{1}, []any{1, 2}, "$")
		if ok || !strings.Contains(reason, "len=") {
			t.Fatalf("reason=%q", reason)
		}
	}
	{
		ok, reason := matchValue([]any{1}, []any{2}, "$")
		if ok || !strings.Contains(reason, "got=1 want=2") {
			t.Fatalf("reason=%q", reason)
		}
	}
	{
		ok, reason := matchValue("x", "x", "$")
		if !ok || reason != "" {
			t.Fatalf("ok=%v reason=%q", ok, reason)
		}
	}
	{
		ok, reason := matchValue([]any{1, "a"}, []any{1, "a"}, "$")
		if !ok || reason != "" {
			t.Fatalf("ok=%v reason=%q", ok, reason)
		}
	}
}

func TestAsStringAnyMap_AnyAnySuccess(t *testing.T) {
	m, ok := asStringAnyMap(map[any]any{"a": 1})
	if !ok || m["a"].(int) != 1 {
		t.Fatalf("m=%#v ok=%v", m, ok)
	}
}

func TestAsAnySlice_AnyCase(t *testing.T) {
	s, ok := asAnySlice([]any{"a"})
	if !ok || len(s) != 1 || s[0].(string) != "a" {
		t.Fatalf("s=%#v ok=%v", s, ok)
	}
}

func TestNumericEqual_SecondNotNumeric(t *testing.T) {
	_, comparable := numericEqual(1, "x")
	if comparable {
		t.Fatalf("expected comparable=false")
	}
}

func TestToFloat64_AllTypes(t *testing.T) {
	cases := []struct {
		in   any
		want float64
		ok   bool
	}{
		{in: float64(1.5), want: 1.5, ok: true},
		{in: float32(1.25), want: 1.25, ok: true},
		{in: int(1), want: 1, ok: true},
		{in: int8(2), want: 2, ok: true},
		{in: int16(3), want: 3, ok: true},
		{in: int32(4), want: 4, ok: true},
		{in: int64(5), want: 5, ok: true},
		{in: uint(6), want: 6, ok: true},
		{in: uint8(7), want: 7, ok: true},
		{in: uint16(8), want: 8, ok: true},
		{in: uint32(9), want: 9, ok: true},
		{in: uint64(10), want: 10, ok: true},
		{in: json.Number("11.5"), want: 11.5, ok: true},
		{in: json.Number("x"), want: 0, ok: false},
		{in: "x", want: 0, ok: false},
	}

	for _, tc := range cases {
		got, ok := toFloat64(tc.in)
		if ok != tc.ok {
			t.Fatalf("toFloat64(%T): ok=%v want %v", tc.in, ok, tc.ok)
		}
		if ok && got != tc.want {
			t.Fatalf("toFloat64(%T): got=%v want %v", tc.in, got, tc.want)
		}
	}
}
