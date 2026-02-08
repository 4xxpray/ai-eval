package evaluator

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/stellarlinkco/ai-eval/internal/llm"
)

type stubProvider struct {
	name string
	fn   func(ctx context.Context, req *llm.Request) (*llm.Response, error)
}

func (p *stubProvider) Name() string { return p.name }

func (p *stubProvider) Complete(ctx context.Context, req *llm.Request) (*llm.Response, error) {
	if p.fn == nil {
		return nil, nil
	}
	return p.fn(ctx, req)
}

func (p *stubProvider) CompleteWithTools(ctx context.Context, req *llm.Request) (*llm.EvalResult, error) {
	_ = ctx
	_ = req
	return nil, errors.New("not implemented")
}

func textResponse(s string) *llm.Response {
	return &llm.Response{
		Content: []llm.ContentBlock{{Type: "text", Text: s}},
	}
}

func mustPanic(t *testing.T, wantContains string, fn func()) {
	t.Helper()
	defer func() {
		t.Helper()
		r := recover()
		if r == nil {
			t.Fatalf("expected panic")
		}
		if wantContains == "" {
			return
		}
		msg := fmt.Sprint(r)
		if !strings.Contains(msg, wantContains) {
			t.Fatalf("panic=%q want contains %q", msg, wantContains)
		}
	}()
	fn()
}

type emptyNameEvaluator struct{}

func (emptyNameEvaluator) Name() string { return " \t\n " }
func (emptyNameEvaluator) Evaluate(ctx context.Context, response string, expected any) (*Result, error) {
	_ = ctx
	_ = response
	_ = expected
	return &Result{Passed: true, Score: 1}, nil
}

func TestEvaluator_Names(t *testing.T) {
	if (ContainsEvaluator{}).Name() != "contains" {
		t.Fatalf("contains name")
	}
	if (NotContainsEvaluator{}).Name() != "not_contains" {
		t.Fatalf("not_contains name")
	}
	if (RegexEvaluator{}).Name() != "regex" {
		t.Fatalf("regex name")
	}
	if (JSONSchemaEvaluator{}).Name() != "json_schema" {
		t.Fatalf("json_schema name")
	}
	if (LLMJudgeEvaluator{}).Name() != "llm_judge" {
		t.Fatalf("llm_judge name")
	}
	if (SimilarityEvaluator{}).Name() != "similarity" {
		t.Fatalf("similarity name")
	}
	if (FactualityEvaluator{}).Name() != "factuality" {
		t.Fatalf("factuality name")
	}
}

func TestRegistry_PanicsAndNilMap(t *testing.T) {
	t.Run("NilRegistry", func(t *testing.T) {
		mustPanic(t, "nil registry", func() {
			var r *Registry
			r.Register(ExactEvaluator{})
		})
	})

	t.Run("NilEvaluator", func(t *testing.T) {
		mustPanic(t, "nil evaluator", func() {
			r := NewRegistry()
			var e Evaluator
			r.Register(e)
		})
	})

	t.Run("EmptyName", func(t *testing.T) {
		mustPanic(t, "empty name", func() {
			r := NewRegistry()
			r.Register(emptyNameEvaluator{})
		})
	})

	t.Run("NilMap", func(t *testing.T) {
		r := &Registry{}
		r.Register(ExactEvaluator{})
		if _, ok := r.Get("exact"); !ok {
			t.Fatalf("expected exact evaluator")
		}
	})

	t.Run("GetNil", func(t *testing.T) {
		var r *Registry
		if e, ok := r.Get("x"); ok || e != nil {
			t.Fatalf("Get on nil registry: ok=%v e=%v", ok, e)
		}
	})
}

func TestContainsEvaluator_EdgeCases(t *testing.T) {
	e := ContainsEvaluator{}

	if _, err := e.Evaluate(context.Background(), "x", nil); err == nil {
		t.Fatalf("expected error")
	}

	if _, err := e.Evaluate(context.Background(), "x", []any{"a", 1}); err == nil {
		t.Fatalf("expected error")
	}

	res, err := e.Evaluate(context.Background(), "hello", []string{})
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if res == nil || !res.Passed || res.Score != 1 {
		t.Fatalf("res=%#v", res)
	}
}

func TestNotContainsEvaluator_EdgeCases(t *testing.T) {
	e := NotContainsEvaluator{}

	res, err := e.Evaluate(context.Background(), "hello", []string{})
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if res == nil || !res.Passed || res.Score != 1 {
		t.Fatalf("res=%#v", res)
	}
}

func TestRegexEvaluator_EdgeCases(t *testing.T) {
	e := RegexEvaluator{}

	if _, err := e.Evaluate(context.Background(), "x", nil); err == nil {
		t.Fatalf("expected error")
	}
	if _, err := e.Evaluate(context.Background(), "x", []any{"a", 1}); err == nil {
		t.Fatalf("expected error")
	}

	res, err := e.Evaluate(context.Background(), "x", []string{})
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if res == nil || !res.Passed || res.Score != 1 {
		t.Fatalf("res=%#v", res)
	}
}

func TestSchemaType_EdgeCases(t *testing.T) {
	if _, err := schemaType(nil); err == nil {
		t.Fatalf("expected error")
	}
	if _, err := schemaType(map[string]any{"type": 1}); err == nil {
		t.Fatalf("expected error")
	}
	if _, err := schemaType(map[string]any{"type": " "}); err == nil {
		t.Fatalf("expected error")
	}
	if got, err := schemaType(map[string]any{"properties": map[string]any{}}); err != nil || got != "object" {
		t.Fatalf("got=%q err=%v", got, err)
	}
	if got, err := schemaType(map[string]any{"required": []any{"a"}}); err != nil || got != "object" {
		t.Fatalf("got=%q err=%v", got, err)
	}
	if got, err := schemaType(map[string]any{"items": map[string]any{}}); err != nil || got != "array" {
		t.Fatalf("got=%q err=%v", got, err)
	}
	if _, err := schemaType(map[string]any{}); err == nil {
		t.Fatalf("expected error")
	}
}

func TestSchemaError_ErrorAndUnwrap(t *testing.T) {
	err := validateJSONSchema(map[string]any{}, map[string]any{"type": 1}, "$")

	var se *schemaError
	if !errors.As(err, &se) {
		t.Fatalf("expected schemaError, got %T", err)
	}
	if se.Error() == "" {
		t.Fatalf("empty error")
	}
	if errors.Unwrap(se) == nil {
		t.Fatalf("expected unwrap error")
	}
}

func TestValidateJSONSchema_Branches(t *testing.T) {
	{
		err := validateJSONSchema("x", map[string]any{"type": "object"}, "$")
		if err == nil || !strings.Contains(err.Error(), "expected object") {
			t.Fatalf("err=%v", err)
		}
	}
	{
		err := validateJSONSchema(map[string]any{}, map[string]any{
			"type":     "object",
			"required": []any{"a", 1},
		}, "$")
		if err == nil || !strings.Contains(err.Error(), "required") {
			t.Fatalf("err=%v", err)
		}
	}
	{
		err := validateJSONSchema(map[string]any{}, map[string]any{
			"type":     "object",
			"required": []string{"a"},
		}, "$")
		if err == nil || !strings.Contains(err.Error(), "missing required field") {
			t.Fatalf("err=%v", err)
		}
	}
	{
		err := validateJSONSchema(map[string]any{}, map[string]any{
			"type":       "object",
			"properties": 1,
		}, "$")
		if err == nil || !strings.Contains(err.Error(), "properties") {
			t.Fatalf("err=%v", err)
		}
	}
	{
		err := validateJSONSchema(map[string]any{"a": "x"}, map[string]any{
			"type":       "object",
			"properties": map[string]any{"a": 1},
		}, "$")
		if err == nil || !strings.Contains(err.Error(), "schema must be an object") {
			t.Fatalf("err=%v", err)
		}
	}
	{
		err := validateJSONSchema(map[string]any{"a": 1}, map[string]any{
			"type":       "object",
			"properties": map[string]any{"a": map[string]any{"type": "string"}},
		}, "$")
		if err == nil || !strings.Contains(err.Error(), "expected string") {
			t.Fatalf("err=%v", err)
		}
	}
	{
		if err := validateJSONSchema(map[string]any{}, map[string]any{"type": "object"}, "$"); err != nil {
			t.Fatalf("err=%v", err)
		}
	}
	{
		err := validateJSONSchema("x", map[string]any{"type": "array"}, "$")
		if err == nil || !strings.Contains(err.Error(), "expected array") {
			t.Fatalf("err=%v", err)
		}
	}
	{
		if err := validateJSONSchema([]any{"x"}, map[string]any{"type": "array"}, "$"); err != nil {
			t.Fatalf("err=%v", err)
		}
	}
	{
		err := validateJSONSchema([]any{"x"}, map[string]any{"type": "array", "items": 1}, "$")
		if err == nil || !strings.Contains(err.Error(), "items must be an object") {
			t.Fatalf("err=%v", err)
		}
	}
	{
		err := validateJSONSchema([]any{json.Number("1.2")}, map[string]any{
			"type":  "array",
			"items": map[string]any{"type": "integer"},
		}, "$")
		if err == nil || !strings.Contains(err.Error(), "expected integer") {
			t.Fatalf("err=%v", err)
		}
	}
	{
		if err := validateJSONSchema("x", map[string]any{"type": "string"}, "$"); err != nil {
			t.Fatalf("err=%v", err)
		}
		if err := validateJSONSchema(1, map[string]any{"type": "string"}, "$"); err == nil {
			t.Fatalf("expected error")
		}
	}
	{
		if err := validateJSONSchema(json.Number("1.2"), map[string]any{"type": "number"}, "$"); err != nil {
			t.Fatalf("err=%v", err)
		}
		if err := validateJSONSchema("x", map[string]any{"type": "number"}, "$"); err == nil {
			t.Fatalf("expected error")
		}
	}
	{
		if err := validateJSONSchema(json.Number("1"), map[string]any{"type": "integer"}, "$"); err != nil {
			t.Fatalf("err=%v", err)
		}
		if err := validateJSONSchema(json.Number("1.0"), map[string]any{"type": "integer"}, "$"); err != nil {
			t.Fatalf("err=%v", err)
		}
		if err := validateJSONSchema(json.Number("1.2"), map[string]any{"type": "integer"}, "$"); err == nil {
			t.Fatalf("expected error")
		}
	}
	{
		if err := validateJSONSchema(true, map[string]any{"type": "boolean"}, "$"); err != nil {
			t.Fatalf("err=%v", err)
		}
		if err := validateJSONSchema("x", map[string]any{"type": "boolean"}, "$"); err == nil {
			t.Fatalf("expected error")
		}
	}
	{
		if err := validateJSONSchema(nil, map[string]any{"type": "null"}, "$"); err != nil {
			t.Fatalf("err=%v", err)
		}
		if err := validateJSONSchema(1, map[string]any{"type": "null"}, "$"); err == nil {
			t.Fatalf("expected error")
		}
	}
	{
		err := validateJSONSchema(nil, map[string]any{"type": "wat"}, "$")
		if err == nil || !strings.Contains(err.Error(), "unsupported") {
			t.Fatalf("err=%v", err)
		}
	}
}

func TestJSONSchemaEvaluator_ExtraData(t *testing.T) {
	e := JSONSchemaEvaluator{}
	schema := map[string]any{"type": "object"}

	res, err := e.Evaluate(context.Background(), `{"a":1} {"b":2}`, schema)
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if res == nil || res.Passed {
		t.Fatalf("res=%#v", res)
	}
	if res.Message != "invalid json" {
		t.Fatalf("msg=%q", res.Message)
	}
}

func TestIsNumberAndIsInteger(t *testing.T) {
	if !isNumber(json.Number("1.2")) {
		t.Fatalf("expected isNumber true")
	}
	if isNumber(json.Number("x")) {
		t.Fatalf("expected isNumber false")
	}
	if !isNumber(int64(1)) {
		t.Fatalf("expected isNumber true")
	}
	if isNumber("x") {
		t.Fatalf("expected isNumber false")
	}

	if !isInteger(json.Number("1")) {
		t.Fatalf("expected isInteger true")
	}
	if !isInteger(json.Number("1.0")) {
		t.Fatalf("expected isInteger true")
	}
	if isInteger(json.Number("1.2")) {
		t.Fatalf("expected isInteger false")
	}
	if !isInteger(float32(2)) {
		t.Fatalf("expected isInteger true")
	}
	if isInteger("x") {
		t.Fatalf("expected isInteger false")
	}
}

func TestFactualityEvaluator_Evaluate(t *testing.T) {
	t.Run("NilEvaluator", func(t *testing.T) {
		var e *FactualityEvaluator
		_, err := e.Evaluate(context.Background(), "x", nil)
		if err == nil {
			t.Fatalf("expected error")
		}
	})

	t.Run("NilProvider", func(t *testing.T) {
		e := &FactualityEvaluator{}
		_, err := e.Evaluate(context.Background(), "x", "gt")
		if err == nil {
			t.Fatalf("expected error")
		}
	})

	t.Run("ExpectedTypeErrors", func(t *testing.T) {
		e := &FactualityEvaluator{Provider: &stubProvider{name: "p"}}
		if _, err := e.Evaluate(context.Background(), "x", 123); err == nil {
			t.Fatalf("expected error")
		}
		if _, err := e.Evaluate(context.Background(), "x", map[string]any{"ground_truth": 1}); err == nil {
			t.Fatalf("expected error")
		}
	})

	t.Run("MissingGroundTruth", func(t *testing.T) {
		e := &FactualityEvaluator{Provider: &stubProvider{name: "p"}}
		if _, err := e.Evaluate(context.Background(), "x", nil); err == nil {
			t.Fatalf("expected error")
		}
	})

	t.Run("ProviderErrorAndNilResp", func(t *testing.T) {
		e := &FactualityEvaluator{Provider: &stubProvider{
			name: "p",
			fn: func(ctx context.Context, req *llm.Request) (*llm.Response, error) {
				_ = ctx
				_ = req
				return nil, errors.New("boom")
			},
		}}
		_, err := e.Evaluate(context.Background(), "x", "gt")
		if err == nil || !strings.Contains(err.Error(), "llm") {
			t.Fatalf("err=%v", err)
		}

		e.Provider = &stubProvider{name: "p"}
		_, err = e.Evaluate(context.Background(), "x", "gt")
		if err == nil || !strings.Contains(err.Error(), "nil llm response") {
			t.Fatalf("err=%v", err)
		}
	})

	t.Run("InvalidJSON", func(t *testing.T) {
		e := &FactualityEvaluator{
			Provider: &stubProvider{name: "p", fn: func(ctx context.Context, req *llm.Request) (*llm.Response, error) {
				_ = ctx
				_ = req
				return textResponse("not json"), nil
			}},
		}
		res, err := e.Evaluate(context.Background(), "x", "gt")
		if err != nil {
			t.Fatalf("Evaluate: %v", err)
		}
		if res == nil || res.Passed || res.Score != 0 {
			t.Fatalf("res=%#v", res)
		}
	})

	t.Run("HasError", func(t *testing.T) {
		e := &FactualityEvaluator{
			Provider: &stubProvider{name: "p", fn: func(ctx context.Context, req *llm.Request) (*llm.Response, error) {
				_ = ctx
				_ = req
				return textResponse(`{"has_error": true, "errors": ["e1"], "reasoning": ""}`), nil
			}},
		}
		res, err := e.Evaluate(context.Background(), "x", "gt")
		if err != nil {
			t.Fatalf("Evaluate: %v", err)
		}
		if res == nil || res.Passed || res.Score != 0 || res.Message == "" {
			t.Fatalf("res=%#v", res)
		}
		if _, ok := res.Details["errors"]; !ok {
			t.Fatalf("Details=%#v", res.Details)
		}
	})

	t.Run("OK", func(t *testing.T) {
		e := &FactualityEvaluator{
			Provider: &stubProvider{name: "p", fn: func(ctx context.Context, req *llm.Request) (*llm.Response, error) {
				_ = ctx
				_ = req
				return textResponse(`{"has_error": false, "errors": [], "reasoning": "ok"}`), nil
			}},
		}
		res, err := e.Evaluate(context.Background(), "x", "gt")
		if err != nil {
			t.Fatalf("Evaluate: %v", err)
		}
		if res == nil || !res.Passed || res.Score != 1 || res.Message != "ok" {
			t.Fatalf("res=%#v", res)
		}
	})
}

func TestSimilarityEvaluator_Evaluate(t *testing.T) {
	t.Run("NilEvaluator", func(t *testing.T) {
		var e *SimilarityEvaluator
		_, err := e.Evaluate(context.Background(), "x", nil)
		if err == nil {
			t.Fatalf("expected error")
		}
	})

	t.Run("NilProvider", func(t *testing.T) {
		e := &SimilarityEvaluator{}
		_, err := e.Evaluate(context.Background(), "x", "ref")
		if err == nil {
			t.Fatalf("expected error")
		}
	})

	t.Run("ExpectedTypeErrors", func(t *testing.T) {
		e := &SimilarityEvaluator{Provider: &stubProvider{name: "p"}, Reference: "ref"}
		if _, err := e.Evaluate(context.Background(), "x", 123); err == nil {
			t.Fatalf("expected error")
		}
		if _, err := e.Evaluate(context.Background(), "x", map[string]any{"reference": 1}); err == nil {
			t.Fatalf("expected error")
		}
		if _, err := e.Evaluate(context.Background(), "x", map[string]any{"reference": "r", "min_score": "x"}); err == nil {
			t.Fatalf("expected error")
		}
	})

	t.Run("MissingReference", func(t *testing.T) {
		e := &SimilarityEvaluator{Provider: &stubProvider{name: "p"}}
		if _, err := e.Evaluate(context.Background(), "x", nil); err == nil {
			t.Fatalf("expected error")
		}
	})

	t.Run("ProviderErrorAndNilResp", func(t *testing.T) {
		e := &SimilarityEvaluator{Provider: &stubProvider{
			name: "p",
			fn: func(ctx context.Context, req *llm.Request) (*llm.Response, error) {
				_ = ctx
				_ = req
				return nil, errors.New("boom")
			},
		}, Reference: "ref"}
		_, err := e.Evaluate(context.Background(), "x", nil)
		if err == nil || !strings.Contains(err.Error(), "llm") {
			t.Fatalf("err=%v", err)
		}

		e.Provider = &stubProvider{name: "p"}
		_, err = e.Evaluate(context.Background(), "x", nil)
		if err == nil || !strings.Contains(err.Error(), "nil llm response") {
			t.Fatalf("err=%v", err)
		}
	})

	t.Run("InvalidJSON", func(t *testing.T) {
		e := &SimilarityEvaluator{
			Provider: &stubProvider{name: "p", fn: func(ctx context.Context, req *llm.Request) (*llm.Response, error) {
				_ = ctx
				_ = req
				return textResponse("not json"), nil
			}},
			Reference: "ref",
		}
		res, err := e.Evaluate(context.Background(), "x", nil)
		if err != nil {
			t.Fatalf("Evaluate: %v", err)
		}
		if res == nil || res.Passed || res.Score != 0 {
			t.Fatalf("res=%#v", res)
		}
	})

	t.Run("ScoreClampAndThreshold", func(t *testing.T) {
		e := &SimilarityEvaluator{
			Provider: &stubProvider{name: "p", fn: func(ctx context.Context, req *llm.Request) (*llm.Response, error) {
				_ = ctx
				_ = req
				return textResponse(`{"score": 2, "reasoning": ""}`), nil
			}},
			Reference: "ref",
		}
		res, err := e.Evaluate(context.Background(), "x", map[string]any{"min_score": 2})
		if err != nil {
			t.Fatalf("Evaluate: %v", err)
		}
		if res == nil || !res.Passed || res.Score != 1 || res.Details["min_score"].(float64) != 1 {
			t.Fatalf("res=%#v", res)
		}
		if res.Message != "no reasoning provided" {
			t.Fatalf("msg=%q", res.Message)
		}
	})

	t.Run("DefaultMinScore", func(t *testing.T) {
		e := &SimilarityEvaluator{
			Provider: &stubProvider{name: "p", fn: func(ctx context.Context, req *llm.Request) (*llm.Response, error) {
				_ = ctx
				_ = req
				return textResponse(`{"score": 0.5, "reasoning": "x"}`), nil
			}},
			Reference: "ref",
			MinScore:  -1,
		}
		res, err := e.Evaluate(context.Background(), "x", map[string]any{"min_score": -1})
		if err != nil {
			t.Fatalf("Evaluate: %v", err)
		}
		if res == nil || res.Passed {
			t.Fatalf("res=%#v", res)
		}
		if res.Details["min_score"].(float64) != 0.6 {
			t.Fatalf("min_score=%v", res.Details["min_score"])
		}
	})
}

func TestLLMJudgeEvaluator_EdgeCases(t *testing.T) {
	t.Run("NilEvaluator", func(t *testing.T) {
		var e *LLMJudgeEvaluator
		_, err := e.Evaluate(context.Background(), "x", nil)
		if err == nil {
			t.Fatalf("expected error")
		}
	})

	t.Run("NilProvider", func(t *testing.T) {
		e := &LLMJudgeEvaluator{}
		_, err := e.Evaluate(context.Background(), "x", "criteria")
		if err == nil {
			t.Fatalf("expected error")
		}
	})

	t.Run("ExpectedTypeErrors", func(t *testing.T) {
		e := &LLMJudgeEvaluator{Provider: &stubProvider{name: "p"}}
		if _, err := e.Evaluate(context.Background(), "x", 123); err == nil {
			t.Fatalf("expected error")
		}
		if _, err := e.Evaluate(context.Background(), "x", map[string]any{"criteria": 1}); err == nil {
			t.Fatalf("expected error")
		}
		if _, err := e.Evaluate(context.Background(), "x", map[string]any{"criteria": "c", "context": 1}); err == nil {
			t.Fatalf("expected error")
		}
		if _, err := e.Evaluate(context.Background(), "x", map[string]any{"criteria": "c", "rubric": []any{"a", 1}}); err == nil {
			t.Fatalf("expected error")
		}
		if _, err := e.Evaluate(context.Background(), "x", map[string]any{"criteria": "c", "score_scale": "x"}); err == nil {
			t.Fatalf("expected error")
		}
		if _, err := e.Evaluate(context.Background(), "x", map[string]any{"criteria": "c", "score_threshold": "x"}); err == nil {
			t.Fatalf("expected error")
		}
	})

	t.Run("MissingCriteria", func(t *testing.T) {
		e := &LLMJudgeEvaluator{Provider: &stubProvider{name: "p"}}
		if _, err := e.Evaluate(context.Background(), "x", " "); err == nil {
			t.Fatalf("expected error")
		}
	})

	t.Run("ProviderErrorAndNilResp", func(t *testing.T) {
		e := &LLMJudgeEvaluator{
			Provider: &stubProvider{name: "p", fn: func(ctx context.Context, req *llm.Request) (*llm.Response, error) {
				_ = ctx
				_ = req
				return nil, errors.New("boom")
			}},
		}
		_, err := e.Evaluate(context.Background(), "x", "criteria")
		if err == nil || !strings.Contains(err.Error(), "llm") {
			t.Fatalf("err=%v", err)
		}

		e.Provider = &stubProvider{name: "p"}
		_, err = e.Evaluate(context.Background(), "x", "criteria")
		if err == nil || !strings.Contains(err.Error(), "nil llm response") {
			t.Fatalf("err=%v", err)
		}
	})

	t.Run("RubricNilClears", func(t *testing.T) {
		e := &LLMJudgeEvaluator{
			Provider: &stubProvider{name: "p", fn: func(ctx context.Context, req *llm.Request) (*llm.Response, error) {
				_ = ctx
				_ = req
				return textResponse(`{"score": 2, "reasoning": ""}`), nil
			}},
			Rubric:     []string{"a"},
			ScoreScale: 1,
		}
		res, err := e.Evaluate(context.Background(), "x", map[string]any{
			"criteria":        "c",
			"rubric":          nil,
			"score_scale":     1,
			"score_threshold": 2,
		})
		if err != nil {
			t.Fatalf("Evaluate: %v", err)
		}
		if res == nil || !res.Passed || res.Score != 1 {
			t.Fatalf("res=%#v", res)
		}
		if res.Details["score_scale"].(int) != 2 || res.Details["score_threshold"].(float64) != 1 {
			t.Fatalf("Details=%#v", res.Details)
		}
		if res.Message != "no reasoning provided" {
			t.Fatalf("msg=%q", res.Message)
		}
	})

	t.Run("InvalidJSON", func(t *testing.T) {
		e := &LLMJudgeEvaluator{
			Provider: &stubProvider{name: "p", fn: func(ctx context.Context, req *llm.Request) (*llm.Response, error) {
				_ = ctx
				_ = req
				return textResponse("not json"), nil
			}},
		}
		res, err := e.Evaluate(context.Background(), "x", "criteria")
		if err != nil {
			t.Fatalf("Evaluate: %v", err)
		}
		if res == nil || res.Passed || res.Score != 0 {
			t.Fatalf("res=%#v", res)
		}
	})

	t.Run("ScoreOutOfRange", func(t *testing.T) {
		e := &LLMJudgeEvaluator{
			Provider: &stubProvider{name: "p", fn: func(ctx context.Context, req *llm.Request) (*llm.Response, error) {
				_ = ctx
				_ = req
				return textResponse(`{"score": 0, "reasoning": "x"}`), nil
			}},
			ScoreScale: 5,
		}
		res, err := e.Evaluate(context.Background(), "x", "criteria")
		if err != nil {
			t.Fatalf("Evaluate: %v", err)
		}
		if res == nil || res.Passed || res.Score != 0 || res.Message == "" {
			t.Fatalf("res=%#v", res)
		}
		if res.Message != "judge score out of range" {
			t.Fatalf("msg=%q", res.Message)
		}
	})
}

func TestNormalizeLikert_AsInt_AsFloat(t *testing.T) {
	if got := normalizeLikert(1, 5); got != 0 {
		t.Fatalf("got=%v", got)
	}
	if got := normalizeLikert(5, 5); got != 1 {
		t.Fatalf("got=%v", got)
	}
	if got := normalizeLikert(3, 5); got != 0.5 {
		t.Fatalf("got=%v", got)
	}
	if got := normalizeLikert(1, 1); got != 0 {
		t.Fatalf("got=%v", got)
	}

	if v, ok := asInt(int8(1)); !ok || v != 1 {
		t.Fatalf("asInt=%d ok=%v", v, ok)
	}
	if v, ok := asInt(uint64(2)); !ok || v != 2 {
		t.Fatalf("asInt=%d ok=%v", v, ok)
	}
	if v, ok := asInt(float64(3)); !ok || v != 3 {
		t.Fatalf("asInt=%d ok=%v", v, ok)
	}
	if v, ok := asInt(json.Number("4")); !ok || v != 4 {
		t.Fatalf("asInt=%d ok=%v", v, ok)
	}
	if _, ok := asInt(json.Number("1.2")); ok {
		t.Fatalf("expected asInt false")
	}
	if _, ok := asInt("x"); ok {
		t.Fatalf("expected asInt false")
	}

	if v, ok := asFloat(int16(1)); !ok || v != 1 {
		t.Fatalf("asFloat=%v ok=%v", v, ok)
	}
	if v, ok := asFloat(float32(1.5)); !ok || v != 1.5 {
		t.Fatalf("asFloat=%v ok=%v", v, ok)
	}
	if v, ok := asFloat(json.Number("2.5")); !ok || v != 2.5 {
		t.Fatalf("asFloat=%v ok=%v", v, ok)
	}
	if _, ok := asFloat(json.Number("x")); ok {
		t.Fatalf("expected asFloat false")
	}
	if _, ok := asFloat("x"); ok {
		t.Fatalf("expected asFloat false")
	}
}

func TestToolCallHelpers_More(t *testing.T) {
	{
		ok, reason := toolArgsSubsetMatch(nil, map[string]any{"a": 1})
		if ok || reason == "" {
			t.Fatalf("ok=%v reason=%q", ok, reason)
		}
	}

	{
		ok, reason := matchValue("x", nil, "$")
		if ok || reason == "" {
			t.Fatalf("ok=%v reason=%q", ok, reason)
		}
		if ok, reason := matchValue(nil, nil, "$"); !ok || reason != "" {
			t.Fatalf("ok=%v reason=%q", ok, reason)
		}
	}

	{
		ok, reason := matchValue("x", "regex:(", "$")
		if ok || !strings.Contains(reason, "invalid regex") {
			t.Fatalf("reason=%q", reason)
		}
		ok, reason = matchValue(1, "regex:^x$", "$")
		if ok || !strings.Contains(reason, "expected string") {
			t.Fatalf("reason=%q", reason)
		}
	}

	{
		ok, reason := matchValue("x", "regex:^y$", "$")
		if ok || !strings.Contains(reason, "did not match") {
			t.Fatalf("reason=%q", reason)
		}
	}

	{
		eq, comparable := numericEqual("x", 1)
		if comparable || eq {
			t.Fatalf("eq=%v comparable=%v", eq, comparable)
		}
		eq, comparable = numericEqual(json.Number("x"), 1)
		if comparable || eq {
			t.Fatalf("eq=%v comparable=%v", eq, comparable)
		}
	}

	{
		if _, ok := asStringAnyMap(map[any]any{1: "x"}); ok {
			t.Fatalf("expected false")
		}
		if m, ok := asStringAnyMap(map[string]int{"a": 1}); !ok || m["a"].(int) != 1 {
			t.Fatalf("m=%#v ok=%v", m, ok)
		}
		if _, ok := asStringAnyMap(map[int]string{1: "x"}); ok {
			t.Fatalf("expected false")
		}
		if _, ok := asStringAnyMap(123); ok {
			t.Fatalf("expected false")
		}
	}

	{
		if _, ok := asAnySlice(nil); ok {
			t.Fatalf("expected false")
		}
		if s, ok := asAnySlice([]string{"a"}); !ok || len(s) != 1 || s[0].(string) != "a" {
			t.Fatalf("s=%#v ok=%v", s, ok)
		}
		if s, ok := asAnySlice([2]int{1, 2}); !ok || len(s) != 2 || s[1].(int) != 2 {
			t.Fatalf("s=%#v ok=%v", s, ok)
		}
		if _, ok := asAnySlice(123); ok {
			t.Fatalf("expected false")
		}
	}
}
