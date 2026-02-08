package rag

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/stellarlinkco/ai-eval/internal/llm"
)

type fakeProvider struct {
	resp *llm.Response
	err  error
}

func (p fakeProvider) Name() string { return "fake" }
func (p fakeProvider) Complete(context.Context, *llm.Request) (*llm.Response, error) {
	return p.resp, p.err
}
func (p fakeProvider) CompleteWithTools(context.Context, *llm.Request) (*llm.EvalResult, error) {
	return nil, nil
}

func TestUtil(t *testing.T) {
	t.Parallel()

	if got := clamp01(-1); got != 0 {
		t.Fatalf("clamp01(-1): got %v want %v", got, 0.0)
	}
	if got := clamp01(0.5); got != 0.5 {
		t.Fatalf("clamp01(0.5): got %v want %v", got, 0.5)
	}
	if got := clamp01(2); got != 1 {
		t.Fatalf("clamp01(2): got %v want %v", got, 1.0)
	}

	numTests := []struct {
		in   any
		want float64
	}{
		{float64(1.5), 1.5},
		{float32(2.5), 2.5},
		{int(-3), -3},
		{int8(-4), -4},
		{int16(-5), -5},
		{int32(-6), -6},
		{int64(-7), -7},
		{uint(8), 8},
		{uint8(9), 9},
		{uint16(10), 10},
		{uint32(11), 11},
		{uint64(12), 12},
	}
	for _, tt := range numTests {
		got, ok := asFloat(tt.in)
		if !ok || got != tt.want {
			t.Fatalf("asFloat(%T): got %v ok=%v want %v", tt.in, got, ok, tt.want)
		}
	}

	if _, ok := asFloat("x"); ok {
		t.Fatalf("asFloat(string): expected not ok")
	}
	if _, ok := asFloat(json.Number("x")); ok {
		t.Fatalf("asFloat(invalid json.Number): expected not ok")
	}
	if got, ok := asFloat(json.Number("1.25")); !ok || got != 1.25 {
		t.Fatalf("asFloat(json.Number): got %v ok=%v", got, ok)
	}
}

func TestEvaluatorNames(t *testing.T) {
	t.Parallel()

	if (FaithfulnessEvaluator{}).Name() != "faithfulness" {
		t.Fatalf("FaithfulnessEvaluator.Name: unexpected")
	}
	if (PrecisionEvaluator{}).Name() != "precision" {
		t.Fatalf("PrecisionEvaluator.Name: unexpected")
	}
	if (RelevancyEvaluator{}).Name() != "relevancy" {
		t.Fatalf("RelevancyEvaluator.Name: unexpected")
	}
}

func TestFaithfulnessEvaluator(t *testing.T) {
	t.Parallel()

	var enil *FaithfulnessEvaluator
	if _, err := enil.Evaluate(context.Background(), "r", nil); err == nil {
		t.Fatalf("Evaluate(nil evaluator): expected error")
	}

	e := &FaithfulnessEvaluator{}
	if _, err := e.Evaluate(context.Background(), "r", map[string]any{"context": "c"}); err == nil {
		t.Fatalf("Evaluate(nil provider): expected error")
	}

	e.Client = fakeProvider{resp: &llm.Response{Content: []llm.ContentBlock{{Type: "text", Text: "not json"}}}}
	if _, err := e.Evaluate(context.Background(), "r", 1); err == nil {
		t.Fatalf("Evaluate(bad expected type): expected error")
	}
	if _, err := e.Evaluate(context.Background(), "r", map[string]any{"context": 1}); err == nil {
		t.Fatalf("Evaluate(context not string): expected error")
	}
	if _, err := e.Evaluate(context.Background(), "r", map[string]any{"context": "c", "threshold": "x"}); err == nil {
		t.Fatalf("Evaluate(threshold not number): expected error")
	}
	if _, err := e.Evaluate(context.Background(), "r", map[string]any{"context": "  "}); err == nil || !strings.Contains(err.Error(), "missing context") {
		t.Fatalf("Evaluate(missing context): got %v", err)
	}

	e.Client = fakeProvider{err: errors.New("upstream")}
	if _, err := e.Evaluate(context.Background(), "r", map[string]any{"context": "c"}); err == nil || !strings.Contains(err.Error(), "faithfulness: llm: upstream") {
		t.Fatalf("Evaluate(upstream): got %v", err)
	}

	e.Client = fakeProvider{resp: nil, err: nil}
	if _, err := e.Evaluate(context.Background(), "r", map[string]any{"context": "c"}); err == nil || !strings.Contains(err.Error(), "nil llm response") {
		t.Fatalf("Evaluate(nil resp): got %v", err)
	}

	e.Client = fakeProvider{resp: &llm.Response{Content: []llm.ContentBlock{{Type: "text", Text: "not json"}}}}
	res, err := e.Evaluate(context.Background(), "r", map[string]any{"context": "c"})
	if err != nil {
		t.Fatalf("Evaluate(invalid json): %v", err)
	}
	if res == nil || res.Passed || res.Score != 0 || !strings.Contains(res.Message, "invalid") {
		t.Fatalf("invalid json result: %#v", res)
	}
	if res.Details["output"] != "not json" {
		t.Fatalf("Details.output: %#v", res.Details)
	}

	e.Client = fakeProvider{resp: &llm.Response{Content: []llm.ContentBlock{{Type: "text", Text: `{"score": 1.5, "reasoning": " ", "unsupported_claims": ["x"]}`}}}}
	res, err = e.Evaluate(context.Background(), "r", map[string]any{"context": "c", "threshold": 2})
	if err != nil {
		t.Fatalf("Evaluate(success): %v", err)
	}
	if res == nil || !res.Passed || res.Score != 1 {
		t.Fatalf("result: %#v", res)
	}
	if res.Message != "no reasoning provided" {
		t.Fatalf("Message: got %q", res.Message)
	}
	if res.Details["threshold"] != float64(1) {
		t.Fatalf("Details.threshold: %#v", res.Details)
	}
	if _, ok := res.Details["unsupported_claims"]; !ok {
		t.Fatalf("Details.unsupported_claims: %#v", res.Details)
	}
}

func TestPrecisionEvaluator(t *testing.T) {
	t.Parallel()

	var enil *PrecisionEvaluator
	if _, err := enil.Evaluate(context.Background(), "r", nil); err == nil {
		t.Fatalf("Evaluate(nil evaluator): expected error")
	}

	e := &PrecisionEvaluator{}
	if _, err := e.Evaluate(context.Background(), "r", map[string]any{"context": "c", "question": "q"}); err == nil {
		t.Fatalf("Evaluate(nil provider): expected error")
	}

	e.Client = fakeProvider{resp: &llm.Response{Content: []llm.ContentBlock{{Type: "text", Text: `{"score":0.4,"reasoning":"ok"}`}}}}
	if _, err := e.Evaluate(context.Background(), "r", "x"); err == nil {
		t.Fatalf("bad expected type: expected error")
	}
	if _, err := e.Evaluate(context.Background(), "r", map[string]any{"context": 1, "question": "q"}); err == nil {
		t.Fatalf("context not string: expected error")
	}
	if _, err := e.Evaluate(context.Background(), "r", map[string]any{"context": "c", "question": 1}); err == nil {
		t.Fatalf("question not string: expected error")
	}
	if _, err := e.Evaluate(context.Background(), "r", map[string]any{"context": "c"}); err == nil || !strings.Contains(err.Error(), "missing question") {
		t.Fatalf("missing question: %v", err)
	}
	if _, err := e.Evaluate(context.Background(), "r", map[string]any{"question": "q"}); err == nil || !strings.Contains(err.Error(), "missing context") {
		t.Fatalf("missing context: %v", err)
	}
	if _, err := e.Evaluate(context.Background(), "r", map[string]any{"context": "c", "question": "q", "threshold": "x"}); err == nil {
		t.Fatalf("threshold not number: expected error")
	}

	res, err := e.Evaluate(context.Background(), "r", map[string]any{"context": "c", "question": "q"})
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if res == nil || !res.Passed || res.Score != 0.4 {
		t.Fatalf("result: %#v", res)
	}
	if len(res.Details) != 0 {
		t.Fatalf("Details: %#v", res.Details)
	}

	e.Client = fakeProvider{resp: &llm.Response{Content: []llm.ContentBlock{{Type: "text", Text: `{"score":-1,"reasoning":" "}`}}}}
	res, err = e.Evaluate(context.Background(), "r", map[string]any{"context": "c", "question": "q", "threshold": -1})
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if !res.Passed || res.Score != 0 {
		t.Fatalf("result: %#v", res)
	}
	if res.Message != "no reasoning provided" {
		t.Fatalf("Message: got %q", res.Message)
	}
	if len(res.Details) != 0 {
		t.Fatalf("threshold: %#v", res.Details)
	}

	e.Client = fakeProvider{resp: &llm.Response{Content: []llm.ContentBlock{{Type: "text", Text: `{"score":0.2,"reasoning":"ok"}`}}}}
	res, err = e.Evaluate(context.Background(), "r", map[string]any{"context": "c", "question": "q", "threshold": 0.3})
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if res.Passed || res.Details["threshold"] != float64(0.3) {
		t.Fatalf("threshold logic: %#v", res)
	}

	e.Client = fakeProvider{resp: &llm.Response{Content: []llm.ContentBlock{{Type: "text", Text: "not json"}}}}
	res, err = e.Evaluate(context.Background(), "r", map[string]any{"context": "c", "question": "q"})
	if err != nil {
		t.Fatalf("Evaluate(invalid json): %v", err)
	}
	if res == nil || res.Passed || res.Score != 0 || !strings.Contains(res.Message, "invalid") {
		t.Fatalf("invalid json result: %#v", res)
	}

	e.Client = fakeProvider{err: errors.New("upstream")}
	if _, err := e.Evaluate(context.Background(), "r", map[string]any{"context": "c", "question": "q"}); err == nil || !strings.Contains(err.Error(), "precision: llm: upstream") {
		t.Fatalf("upstream: %v", err)
	}
	e.Client = fakeProvider{resp: nil, err: nil}
	if _, err := e.Evaluate(context.Background(), "r", map[string]any{"context": "c", "question": "q"}); err == nil || !strings.Contains(err.Error(), "nil llm response") {
		t.Fatalf("nil resp: %v", err)
	}

	e.Client = fakeProvider{resp: &llm.Response{Content: []llm.ContentBlock{{Type: "text", Text: `{"score":0.5,"reasoning":"ok"}`}}}}
	res, err = e.Evaluate(context.Background(), "r", map[string]any{"context": "c", "question": "q", "threshold": 2})
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if res.Passed || res.Details["threshold"] != float64(1) {
		t.Fatalf("threshold clamp: %#v", res)
	}
}

func TestRelevancyEvaluator(t *testing.T) {
	t.Parallel()

	var enil *RelevancyEvaluator
	if _, err := enil.Evaluate(context.Background(), "r", nil); err == nil {
		t.Fatalf("Evaluate(nil evaluator): expected error")
	}

	e := &RelevancyEvaluator{}
	if _, err := e.Evaluate(context.Background(), "r", map[string]any{"question": "q"}); err == nil {
		t.Fatalf("Evaluate(nil provider): expected error")
	}

	e.Client = fakeProvider{resp: &llm.Response{Content: []llm.ContentBlock{{Type: "text", Text: `{"score":0.9,"reasoning":"ok"}`}}}}
	if _, err := e.Evaluate(context.Background(), "r", 1); err == nil {
		t.Fatalf("bad expected type: expected error")
	}
	if _, err := e.Evaluate(context.Background(), "r", map[string]any{"question": 1}); err == nil {
		t.Fatalf("question not string: expected error")
	}
	if _, err := e.Evaluate(context.Background(), "r", map[string]any{"question": " "}); err == nil || !strings.Contains(err.Error(), "missing question") {
		t.Fatalf("missing question: %v", err)
	}
	if _, err := e.Evaluate(context.Background(), "r", map[string]any{"question": "q", "threshold": "x"}); err == nil {
		t.Fatalf("threshold not number: expected error")
	}

	res, err := e.Evaluate(context.Background(), "r", map[string]any{"question": "q", "threshold": 0})
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if res == nil || !res.Passed || res.Score != 0.9 {
		t.Fatalf("result: %#v", res)
	}
	if res.Details["threshold"] != float64(0.8) {
		t.Fatalf("threshold: %#v", res.Details)
	}

	e.Client = fakeProvider{resp: &llm.Response{Content: []llm.ContentBlock{{Type: "text", Text: `{"score":1,"reasoning":" "}`}}}}
	res, err = e.Evaluate(context.Background(), "r", map[string]any{"question": "q"})
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if res == nil || !res.Passed || res.Message != "no reasoning provided" {
		t.Fatalf("reasoning default: %#v", res)
	}

	e.Client = fakeProvider{resp: nil, err: nil}
	if _, err := e.Evaluate(context.Background(), "r", map[string]any{"question": "q"}); err == nil || !strings.Contains(err.Error(), "nil llm response") {
		t.Fatalf("nil resp: %v", err)
	}
	e.Client = fakeProvider{err: errors.New("upstream")}
	if _, err := e.Evaluate(context.Background(), "r", map[string]any{"question": "q"}); err == nil || !strings.Contains(err.Error(), "relevancy: llm: upstream") {
		t.Fatalf("upstream: %v", err)
	}

	e.Client = fakeProvider{resp: &llm.Response{Content: []llm.ContentBlock{{Type: "text", Text: "not json"}}}}
	res, err = e.Evaluate(context.Background(), "r", map[string]any{"question": "q"})
	if err != nil {
		t.Fatalf("Evaluate(invalid json): %v", err)
	}
	if res == nil || res.Passed || res.Score != 0 || !strings.Contains(res.Message, "invalid") {
		t.Fatalf("invalid json result: %#v", res)
	}
}
