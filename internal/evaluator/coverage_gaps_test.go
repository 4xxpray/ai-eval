package evaluator

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/stellarlinkco/ai-eval/internal/llm"
)

func TestContainsAndNotContains_ErrorPaths(t *testing.T) {
	{
		res, err := (ContainsEvaluator{}).Evaluate(context.Background(), "hello", "he")
		if err != nil {
			t.Fatalf("Evaluate: %v", err)
		}
		if res == nil || !res.Passed || res.Score != 1 {
			t.Fatalf("res=%#v", res)
		}
	}
	{
		_, err := (ContainsEvaluator{}).Evaluate(context.Background(), "hello", 123)
		if err == nil {
			t.Fatalf("expected error")
		}
	}
	{
		_, err := (NotContainsEvaluator{}).Evaluate(context.Background(), "hello", 123)
		if err == nil {
			t.Fatalf("expected error")
		}
	}
}

func TestFactualityEvaluator_MapGroundTruth(t *testing.T) {
	e := &FactualityEvaluator{
		Provider: &stubProvider{name: "p", fn: func(ctx context.Context, req *llm.Request) (*llm.Response, error) {
			_ = ctx
			_ = req
			return textResponse(`{"has_error": false, "errors": [], "reasoning": "ok"}`), nil
		}},
	}

	res, err := e.Evaluate(context.Background(), "x", map[string]any{"ground_truth": " gt "})
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if res == nil || !res.Passed {
		t.Fatalf("res=%#v", res)
	}
}

func TestJSONSchemaEvaluator_ExpectedTypeError(t *testing.T) {
	_, err := (JSONSchemaEvaluator{}).Evaluate(context.Background(), `{}`, 123)
	if err == nil {
		t.Fatalf("expected error")
	}
	if !strings.Contains(err.Error(), "expected map[string]any") {
		t.Fatalf("err=%q", err.Error())
	}
}

func TestValidateJSONSchema_PropertyMissingAndArraySuccess(t *testing.T) {
	{
		err := validateJSONSchema(map[string]any{}, map[string]any{
			"type": "object",
			"properties": map[string]any{
				"a": map[string]any{"type": "string"},
			},
		}, "$")
		if err != nil {
			t.Fatalf("err=%v", err)
		}
	}
	{
		err := validateJSONSchema([]any{json.Number("1")}, map[string]any{
			"type":  "array",
			"items": map[string]any{"type": "integer"},
		}, "$")
		if err != nil {
			t.Fatalf("err=%v", err)
		}
	}
}

func TestIsInteger_MoreBranches(t *testing.T) {
	if isInteger(json.Number("x")) {
		t.Fatalf("expected false")
	}
	if !isInteger(float64(2)) || isInteger(float64(1.2)) {
		t.Fatalf("float64 branch")
	}
	if !isInteger(int8(1)) {
		t.Fatalf("int branch")
	}
}

func TestLLMJudgeEvaluator_DefaultingAndClamping(t *testing.T) {
	provider := &stubProvider{name: "p", fn: func(ctx context.Context, req *llm.Request) (*llm.Response, error) {
		_ = ctx
		_ = req
		return textResponse(`{"score": 5, "reasoning": "ok"}`), nil
	}}

	{
		e := &LLMJudgeEvaluator{Provider: provider, Criteria: "c", ScoreScale: 5, ScoreThreshold: 2}
		res, err := e.Evaluate(context.Background(), "x", nil)
		if err != nil {
			t.Fatalf("Evaluate: %v", err)
		}
		if res == nil || !res.Passed {
			t.Fatalf("res=%#v", res)
		}
		if res.Details["score_threshold"].(float64) != 1 {
			t.Fatalf("threshold=%v", res.Details["score_threshold"])
		}
	}

	{
		e := &LLMJudgeEvaluator{Provider: provider}
		res, err := e.Evaluate(context.Background(), "x", map[string]any{
			"criteria":        "c",
			"score_scale":     0,
			"score_threshold": 0,
		})
		if err != nil {
			t.Fatalf("Evaluate: %v", err)
		}
		if res == nil {
			t.Fatalf("nil result")
		}
		if res.Details["score_scale"].(int) != 5 {
			t.Fatalf("score_scale=%v", res.Details["score_scale"])
		}
		if res.Details["score_threshold"].(float64) != 0.6 {
			t.Fatalf("score_threshold=%v", res.Details["score_threshold"])
		}
	}
}

func TestLLMJudge_AsInt_AsFloat_MoreTypes(t *testing.T) {
	if v, ok := asInt(int16(1)); !ok || v != 1 {
		t.Fatalf("asInt int16=%d ok=%v", v, ok)
	}
	if v, ok := asInt(int32(2)); !ok || v != 2 {
		t.Fatalf("asInt int32=%d ok=%v", v, ok)
	}
	if v, ok := asInt(uint8(3)); !ok || v != 3 {
		t.Fatalf("asInt uint8=%d ok=%v", v, ok)
	}
	if v, ok := asInt(uint16(4)); !ok || v != 4 {
		t.Fatalf("asInt uint16=%d ok=%v", v, ok)
	}
	if v, ok := asInt(uint32(5)); !ok || v != 5 {
		t.Fatalf("asInt uint32=%d ok=%v", v, ok)
	}
	if v, ok := asInt(float32(6.9)); !ok || v != 6 {
		t.Fatalf("asInt float32=%d ok=%v", v, ok)
	}

	if v, ok := asFloat(int8(1)); !ok || v != 1 {
		t.Fatalf("asFloat int8=%v ok=%v", v, ok)
	}
	if v, ok := asFloat(uint8(2)); !ok || v != 2 {
		t.Fatalf("asFloat uint8=%v ok=%v", v, ok)
	}
	if v, ok := asFloat(uint16(3)); !ok || v != 3 {
		t.Fatalf("asFloat uint16=%v ok=%v", v, ok)
	}
	if v, ok := asFloat(uint32(4)); !ok || v != 4 {
		t.Fatalf("asFloat uint32=%v ok=%v", v, ok)
	}
	if v, ok := asFloat(uint64(5)); !ok || v != 5 {
		t.Fatalf("asFloat uint64=%v ok=%v", v, ok)
	}
	if v, ok := asFloat(float64(1.25)); !ok || v != 1.25 {
		t.Fatalf("asFloat float64=%v ok=%v", v, ok)
	}
}

func TestLLMJudge_AsInt_AsFloat_RemainingTypes(t *testing.T) {
	if v, ok := asInt(int64(7)); !ok || v != 7 {
		t.Fatalf("asInt int64=%d ok=%v", v, ok)
	}
	if v, ok := asInt(uint(8)); !ok || v != 8 {
		t.Fatalf("asInt uint=%d ok=%v", v, ok)
	}

	if v, ok := asFloat(int32(9)); !ok || v != 9 {
		t.Fatalf("asFloat int32=%v ok=%v", v, ok)
	}
	if v, ok := asFloat(int64(10)); !ok || v != 10 {
		t.Fatalf("asFloat int64=%v ok=%v", v, ok)
	}
	if v, ok := asFloat(uint(11)); !ok || v != 11 {
		t.Fatalf("asFloat uint=%v ok=%v", v, ok)
	}
}

func TestRegexEvaluator_ToStringSliceAnySuccess(t *testing.T) {
	res, err := (RegexEvaluator{}).Evaluate(context.Background(), "x", []any{"^x$"})
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if res == nil || !res.Passed || res.Score != 1 {
		t.Fatalf("res=%#v", res)
	}
}

func TestSimilarityEvaluator_MoreBranches(t *testing.T) {
	t.Run("ExpectedStringOverrides", func(t *testing.T) {
		e := &SimilarityEvaluator{
			Provider: &stubProvider{name: "p", fn: func(ctx context.Context, req *llm.Request) (*llm.Response, error) {
				_ = ctx
				_ = req
				return textResponse(`{"score": 1, "reasoning": "ok"}`), nil
			}},
			Reference: "old",
		}
		res, err := e.Evaluate(context.Background(), "x", " new ")
		if err != nil {
			t.Fatalf("Evaluate: %v", err)
		}
		if res == nil || !res.Passed {
			t.Fatalf("res=%#v", res)
		}
	})

	t.Run("InitialMinScoreClamp", func(t *testing.T) {
		e := &SimilarityEvaluator{
			Provider: &stubProvider{name: "p", fn: func(ctx context.Context, req *llm.Request) (*llm.Response, error) {
				_ = ctx
				_ = req
				return textResponse(`{"score": 0.9, "reasoning": "ok"}`), nil
			}},
			Reference: "ref",
			MinScore:  2,
		}
		res, err := e.Evaluate(context.Background(), "x", nil)
		if err != nil {
			t.Fatalf("Evaluate: %v", err)
		}
		if res == nil || res.Details["min_score"].(float64) != 1 {
			t.Fatalf("res=%#v", res)
		}
	})

	t.Run("ScoreLowerBoundClamp", func(t *testing.T) {
		e := &SimilarityEvaluator{
			Provider: &stubProvider{name: "p", fn: func(ctx context.Context, req *llm.Request) (*llm.Response, error) {
				_ = ctx
				_ = req
				return textResponse(`{"score": -1, "reasoning": "ok"}`), nil
			}},
			Reference: "ref",
		}
		res, err := e.Evaluate(context.Background(), "x", nil)
		if err != nil {
			t.Fatalf("Evaluate: %v", err)
		}
		if res == nil || res.Score != 0 {
			t.Fatalf("res=%#v", res)
		}
	})
}
