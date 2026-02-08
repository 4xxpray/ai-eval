package safety

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

	if _, err := asStringSlice(nil); err == nil {
		t.Fatalf("asStringSlice(nil): expected error")
	}
	if got, err := asStringSlice("x"); err != nil || len(got) != 1 || got[0] != "x" {
		t.Fatalf("asStringSlice(string): got=%v err=%v", got, err)
	}
	if got, err := asStringSlice([]string{"a", "b"}); err != nil || len(got) != 2 || got[1] != "b" {
		t.Fatalf("asStringSlice([]string): got=%v err=%v", got, err)
	}
	if got, err := asStringSlice([]any{"a", "b"}); err != nil || len(got) != 2 || got[1] != "b" {
		t.Fatalf("asStringSlice([]any): got=%v err=%v", got, err)
	}
	if _, err := asStringSlice([]any{"a", 2}); err == nil {
		t.Fatalf("asStringSlice([]any mix): expected error")
	}
	if _, err := asStringSlice(123); err == nil {
		t.Fatalf("asStringSlice(other): expected error")
	}
}

func TestEvaluatorNames(t *testing.T) {
	t.Parallel()

	if (BiasEvaluator{}).Name() != "bias" {
		t.Fatalf("BiasEvaluator.Name: unexpected")
	}
	if (HallucinationEvaluator{}).Name() != "hallucination" {
		t.Fatalf("HallucinationEvaluator.Name: unexpected")
	}
	if (ToxicityEvaluator{}).Name() != "toxicity" {
		t.Fatalf("ToxicityEvaluator.Name: unexpected")
	}
}

func TestBiasEvaluator(t *testing.T) {
	t.Parallel()

	var enil *BiasEvaluator
	if _, err := enil.Evaluate(context.Background(), "r", nil); err == nil {
		t.Fatalf("Evaluate(nil evaluator): expected error")
	}

	e := &BiasEvaluator{}
	if _, err := e.Evaluate(context.Background(), "r", map[string]any{}); err == nil {
		t.Fatalf("Evaluate(nil provider): expected error")
	}

	e.Client = fakeProvider{resp: &llm.Response{Content: []llm.ContentBlock{{Type: "text", Text: `{"bias":0.2,"reasoning":"ok","detected":[]}`}}}}
	if _, err := e.Evaluate(context.Background(), "r", 1); err == nil {
		t.Fatalf("bad expected type: expected error")
	}
	if _, err := e.Evaluate(context.Background(), "r", map[string]any{"categories": nil}); err == nil {
		t.Fatalf("categories nil: expected error")
	}
	if _, err := e.Evaluate(context.Background(), "r", map[string]any{"categories": []any{"a", 2}}); err == nil {
		t.Fatalf("categories bad elem: expected error")
	}
	if _, err := e.Evaluate(context.Background(), "r", map[string]any{"threshold": "x"}); err == nil {
		t.Fatalf("threshold not number: expected error")
	}

	e.Client = fakeProvider{resp: &llm.Response{Content: []llm.ContentBlock{{Type: "text", Text: "not json"}}}}
	res, err := e.Evaluate(context.Background(), "r", map[string]any{})
	if err != nil {
		t.Fatalf("Evaluate(invalid json): %v", err)
	}
	if res == nil || res.Passed || res.Score != 0 || !strings.Contains(res.Message, "invalid") {
		t.Fatalf("invalid json result: %#v", res)
	}

	e.Client = fakeProvider{err: errors.New("upstream")}
	if _, err := e.Evaluate(context.Background(), "r", map[string]any{}); err == nil || !strings.Contains(err.Error(), "bias: llm: upstream") {
		t.Fatalf("upstream: %v", err)
	}
	e.Client = fakeProvider{resp: nil, err: nil}
	if _, err := e.Evaluate(context.Background(), "r", map[string]any{}); err == nil || !strings.Contains(err.Error(), "nil llm response") {
		t.Fatalf("nil resp: %v", err)
	}

	e.Client = fakeProvider{resp: &llm.Response{Content: []llm.ContentBlock{{Type: "text", Text: `{"bias":2,"reasoning":" ","detected":["x"]}`}}}}
	res, err = e.Evaluate(context.Background(), "r", map[string]any{"categories": "a", "threshold": 0})
	if err != nil {
		t.Fatalf("Evaluate(success): %v", err)
	}
	if res == nil || res.Passed || res.Score != 0 {
		t.Fatalf("result: %#v", res)
	}
	if res.Message != "no reasoning provided" {
		t.Fatalf("Message: got %q", res.Message)
	}
	if res.Details["threshold"] != float64(0.1) || res.Details["bias"] != float64(1) {
		t.Fatalf("Details: %#v", res.Details)
	}
	if _, ok := res.Details["detected"]; !ok {
		t.Fatalf("Details.detected: %#v", res.Details)
	}

	e.Client = fakeProvider{resp: &llm.Response{Content: []llm.ContentBlock{{Type: "text", Text: `{"bias":0.5,"reasoning":"ok","detected":[]}`}}}}
	res, err = e.Evaluate(context.Background(), "r", map[string]any{"threshold": 2})
	if err != nil {
		t.Fatalf("Evaluate(threshold clamp): %v", err)
	}
	if res == nil || !res.Passed || res.Details["threshold"] != float64(1) {
		t.Fatalf("threshold clamp result: %#v", res)
	}
}

func TestHallucinationEvaluator(t *testing.T) {
	t.Parallel()

	var enil *HallucinationEvaluator
	if _, err := enil.Evaluate(context.Background(), "r", nil); err == nil {
		t.Fatalf("Evaluate(nil evaluator): expected error")
	}

	e := &HallucinationEvaluator{}
	if _, err := e.Evaluate(context.Background(), "r", map[string]any{"ground_truth": "x"}); err == nil {
		t.Fatalf("Evaluate(nil provider): expected error")
	}

	e.Client = fakeProvider{resp: &llm.Response{Content: []llm.ContentBlock{{Type: "text", Text: `{"score":0.2,"reasoning":"ok","hallucinations":[],"contradictions":[]}`}}}}
	if _, err := e.Evaluate(context.Background(), "r", 1); err == nil {
		t.Fatalf("bad expected type: expected error")
	}
	if _, err := e.Evaluate(context.Background(), "r", map[string]any{"ground_truth": 1}); err == nil {
		t.Fatalf("ground_truth not string: expected error")
	}
	if _, err := e.Evaluate(context.Background(), "r", map[string]any{"ground_truth": "x", "threshold": "x"}); err == nil {
		t.Fatalf("threshold not number: expected error")
	}
	if _, err := e.Evaluate(context.Background(), "r", map[string]any{"ground_truth": " "}); err == nil || !strings.Contains(err.Error(), "missing ground truth") {
		t.Fatalf("missing ground truth: %v", err)
	}

	e.Client = fakeProvider{resp: &llm.Response{Content: []llm.ContentBlock{{Type: "text", Text: "not json"}}}}
	res, err := e.Evaluate(context.Background(), "r", map[string]any{"ground_truth": "x"})
	if err != nil {
		t.Fatalf("Evaluate(invalid json): %v", err)
	}
	if res == nil || res.Passed || res.Score != 0 || !strings.Contains(res.Message, "invalid") {
		t.Fatalf("invalid json result: %#v", res)
	}

	e.Client = fakeProvider{err: errors.New("upstream")}
	if _, err := e.Evaluate(context.Background(), "r", map[string]any{"ground_truth": "x"}); err == nil || !strings.Contains(err.Error(), "hallucination: llm: upstream") {
		t.Fatalf("upstream: %v", err)
	}
	e.Client = fakeProvider{resp: nil, err: nil}
	if _, err := e.Evaluate(context.Background(), "r", map[string]any{"ground_truth": "x"}); err == nil || !strings.Contains(err.Error(), "nil llm response") {
		t.Fatalf("nil resp: %v", err)
	}

	e.Client = fakeProvider{resp: &llm.Response{Content: []llm.ContentBlock{{Type: "text", Text: `{"score":1.5,"reasoning":" ","hallucinations":["h"],"contradictions":["c"]}`}}}}
	res, err = e.Evaluate(context.Background(), "r", map[string]any{"ground_truth": "x", "threshold": 2})
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
		t.Fatalf("Details: %#v", res.Details)
	}
	if _, ok := res.Details["hallucinations"]; !ok {
		t.Fatalf("Details.hallucinations: %#v", res.Details)
	}
	if _, ok := res.Details["contradictions"]; !ok {
		t.Fatalf("Details.contradictions: %#v", res.Details)
	}

	e.Client = fakeProvider{resp: &llm.Response{Content: []llm.ContentBlock{{Type: "text", Text: `{"score":1,"reasoning":"ok","hallucinations":[],"contradictions":[]}`}}}}
	res, err = e.Evaluate(context.Background(), "r", map[string]any{"ground_truth": "x", "threshold": 0})
	if err != nil {
		t.Fatalf("Evaluate(default threshold): %v", err)
	}
	if res == nil || !res.Passed || res.Details["threshold"] != float64(0.9) {
		t.Fatalf("default threshold result: %#v", res)
	}
}

func TestToxicityEvaluator(t *testing.T) {
	t.Parallel()

	var enil *ToxicityEvaluator
	if _, err := enil.Evaluate(context.Background(), "r", nil); err == nil {
		t.Fatalf("Evaluate(nil evaluator): expected error")
	}

	e := &ToxicityEvaluator{}
	if _, err := e.Evaluate(context.Background(), "r", map[string]any{}); err == nil {
		t.Fatalf("Evaluate(nil provider): expected error")
	}

	e.Client = fakeProvider{resp: &llm.Response{Content: []llm.ContentBlock{{Type: "text", Text: `{"toxicity":0.2,"reasoning":"ok","categories":[]}`}}}}
	if _, err := e.Evaluate(context.Background(), "r", 1); err == nil {
		t.Fatalf("bad expected type: expected error")
	}
	if _, err := e.Evaluate(context.Background(), "r", map[string]any{"threshold": "x"}); err == nil {
		t.Fatalf("threshold not number: expected error")
	}

	e.Client = fakeProvider{resp: &llm.Response{Content: []llm.ContentBlock{{Type: "text", Text: "not json"}}}}
	res, err := e.Evaluate(context.Background(), "r", map[string]any{})
	if err != nil {
		t.Fatalf("Evaluate(invalid json): %v", err)
	}
	if res == nil || res.Passed || res.Score != 0 || !strings.Contains(res.Message, "invalid") {
		t.Fatalf("invalid json result: %#v", res)
	}

	e.Client = fakeProvider{err: errors.New("upstream")}
	if _, err := e.Evaluate(context.Background(), "r", map[string]any{}); err == nil || !strings.Contains(err.Error(), "toxicity: llm: upstream") {
		t.Fatalf("upstream: %v", err)
	}
	e.Client = fakeProvider{resp: nil, err: nil}
	if _, err := e.Evaluate(context.Background(), "r", map[string]any{}); err == nil || !strings.Contains(err.Error(), "nil llm response") {
		t.Fatalf("nil resp: %v", err)
	}

	e.Client = fakeProvider{resp: &llm.Response{Content: []llm.ContentBlock{{Type: "text", Text: `{"toxicity":2,"reasoning":" ","categories":["insult"]}`}}}}
	res, err = e.Evaluate(context.Background(), "r", map[string]any{"threshold": 0})
	if err != nil {
		t.Fatalf("Evaluate(success): %v", err)
	}
	if res == nil || res.Passed || res.Score != 0 {
		t.Fatalf("result: %#v", res)
	}
	if res.Message != "no reasoning provided" {
		t.Fatalf("Message: got %q", res.Message)
	}
	if res.Details["threshold"] != float64(0.1) || res.Details["toxicity"] != float64(1) {
		t.Fatalf("Details: %#v", res.Details)
	}
	if _, ok := res.Details["categories"]; !ok {
		t.Fatalf("Details.categories: %#v", res.Details)
	}

	e.Client = fakeProvider{resp: &llm.Response{Content: []llm.ContentBlock{{Type: "text", Text: `{"toxicity":0.5,"reasoning":"ok","categories":[]}`}}}}
	res, err = e.Evaluate(context.Background(), "r", map[string]any{"threshold": 2})
	if err != nil {
		t.Fatalf("Evaluate(threshold clamp): %v", err)
	}
	if res == nil || !res.Passed || res.Details["threshold"] != float64(1) {
		t.Fatalf("threshold clamp result: %#v", res)
	}
}
