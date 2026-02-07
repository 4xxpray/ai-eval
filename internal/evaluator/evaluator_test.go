package evaluator

import (
	"context"
	"testing"
)

func TestRegistry(t *testing.T) {
	t.Parallel()

	r := NewRegistry()
	r.Register(ExactEvaluator{})

	e, ok := r.Get("exact")
	if !ok {
		t.Fatalf("Get(exact) ok=false")
	}

	res, err := e.Evaluate(context.Background(), "ok", "ok")
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if !res.Passed || res.Score != 1.0 {
		t.Fatalf("got passed=%v score=%v", res.Passed, res.Score)
	}
}

func TestExactEvaluator(t *testing.T) {
	t.Parallel()

	e := ExactEvaluator{}

	{
		res, err := e.Evaluate(context.Background(), "a", "a")
		if err != nil {
			t.Fatalf("Evaluate: %v", err)
		}
		if !res.Passed || res.Score != 1.0 {
			t.Fatalf("match: got passed=%v score=%v", res.Passed, res.Score)
		}
	}
	{
		res, err := e.Evaluate(context.Background(), "a", "b")
		if err != nil {
			t.Fatalf("Evaluate: %v", err)
		}
		if res.Passed || res.Score != 0.0 {
			t.Fatalf("mismatch: got passed=%v score=%v", res.Passed, res.Score)
		}
	}
	{
		_, err := e.Evaluate(context.Background(), "a", []string{"a"})
		if err == nil {
			t.Fatalf("type mismatch: expected error")
		}
	}
}

func TestContainsEvaluator(t *testing.T) {
	t.Parallel()

	e := ContainsEvaluator{}

	{
		res, err := e.Evaluate(context.Background(), "hello world", []string{"hello", "world"})
		if err != nil {
			t.Fatalf("Evaluate: %v", err)
		}
		if !res.Passed || res.Score != 1.0 {
			t.Fatalf("all match: got passed=%v score=%v", res.Passed, res.Score)
		}
	}
	{
		res, err := e.Evaluate(context.Background(), "hello world", []string{"hello", "mars"})
		if err != nil {
			t.Fatalf("Evaluate: %v", err)
		}
		if res.Passed || res.Score != 0.5 {
			t.Fatalf("partial match: got passed=%v score=%v", res.Passed, res.Score)
		}
		if got := res.Details["missing"].([]string); len(got) != 1 || got[0] != "mars" {
			t.Fatalf("missing: got %#v", got)
		}
	}
}

func TestNotContainsEvaluator(t *testing.T) {
	t.Parallel()

	e := NotContainsEvaluator{}

	{
		res, err := e.Evaluate(context.Background(), "hello world", []string{"mars", "venus"})
		if err != nil {
			t.Fatalf("Evaluate: %v", err)
		}
		if !res.Passed || res.Score != 1.0 {
			t.Fatalf("none found: got passed=%v score=%v", res.Passed, res.Score)
		}
	}
	{
		res, err := e.Evaluate(context.Background(), "hello world", []string{"hello", "mars"})
		if err != nil {
			t.Fatalf("Evaluate: %v", err)
		}
		if res.Passed || res.Score != 0.5 {
			t.Fatalf("found: got passed=%v score=%v", res.Passed, res.Score)
		}
		if got := res.Details["found"].([]string); len(got) != 1 || got[0] != "hello" {
			t.Fatalf("found: got %#v", got)
		}
	}
}

func TestRegexEvaluator(t *testing.T) {
	t.Parallel()

	e := RegexEvaluator{}

	// Single pattern match
	{
		res, err := e.Evaluate(context.Background(), "hello world", `^hello`)
		if err != nil {
			t.Fatalf("Evaluate: %v", err)
		}
		if !res.Passed || res.Score != 1.0 {
			t.Fatalf("match: got passed=%v score=%v", res.Passed, res.Score)
		}
	}
	// Single pattern no match
	{
		res, err := e.Evaluate(context.Background(), "hello world", `^world`)
		if err != nil {
			t.Fatalf("Evaluate: %v", err)
		}
		if res.Passed || res.Score != 0.0 {
			t.Fatalf("no match: got passed=%v score=%v", res.Passed, res.Score)
		}
	}
	// Multiple patterns - all match
	{
		res, err := e.Evaluate(context.Background(), "hello world", []string{`hello`, `world`})
		if err != nil {
			t.Fatalf("Evaluate: %v", err)
		}
		if !res.Passed || res.Score != 1.0 {
			t.Fatalf("all match: got passed=%v score=%v", res.Passed, res.Score)
		}
	}
	// Multiple patterns - partial match
	{
		res, err := e.Evaluate(context.Background(), "hello world", []string{`hello`, `foo`})
		if err != nil {
			t.Fatalf("Evaluate: %v", err)
		}
		if res.Passed || res.Score != 0.5 {
			t.Fatalf("partial match: got passed=%v score=%v", res.Passed, res.Score)
		}
	}
	// Case insensitive
	{
		res, err := e.Evaluate(context.Background(), "Hello World", []string{`(?i)hello`, `(?i)world`})
		if err != nil {
			t.Fatalf("Evaluate: %v", err)
		}
		if !res.Passed || res.Score != 1.0 {
			t.Fatalf("case insensitive: got passed=%v score=%v", res.Passed, res.Score)
		}
	}
	// Invalid regex
	{
		_, err := e.Evaluate(context.Background(), "hello", `(`)
		if err == nil {
			t.Fatalf("invalid regex: expected error")
		}
	}
}

func TestJSONSchemaEvaluator(t *testing.T) {
	t.Parallel()

	e := JSONSchemaEvaluator{}

	schema := map[string]any{
		"type": "object",
		"required": []any{
			"foo",
		},
		"properties": map[string]any{
			"foo": map[string]any{"type": "string"},
			"n":   map[string]any{"type": "integer"},
		},
	}

	{
		res, err := e.Evaluate(context.Background(), `{"foo":"bar","n":1}`, schema)
		if err != nil {
			t.Fatalf("Evaluate: %v", err)
		}
		if !res.Passed || res.Score != 1.0 {
			t.Fatalf("valid: got passed=%v score=%v msg=%q", res.Passed, res.Score, res.Message)
		}
	}
	{
		res, err := e.Evaluate(context.Background(), `{"n":1}`, schema)
		if err != nil {
			t.Fatalf("Evaluate: %v", err)
		}
		if res.Passed || res.Score != 0.0 {
			t.Fatalf("missing required: got passed=%v score=%v", res.Passed, res.Score)
		}
	}
	{
		res, err := e.Evaluate(context.Background(), `{"foo":123,"n":1}`, schema)
		if err != nil {
			t.Fatalf("Evaluate: %v", err)
		}
		if res.Passed || res.Score != 0.0 {
			t.Fatalf("wrong type: got passed=%v score=%v", res.Passed, res.Score)
		}
	}
	{
		res, err := e.Evaluate(context.Background(), `{"foo":`, schema)
		if err != nil {
			t.Fatalf("Evaluate: %v", err)
		}
		if res.Passed || res.Score != 0.0 {
			t.Fatalf("invalid json: got passed=%v score=%v", res.Passed, res.Score)
		}
	}
	{
		_, err := e.Evaluate(context.Background(), `{"foo":"bar"}`, map[string]any{"type": 1})
		if err == nil {
			t.Fatalf("invalid schema: expected error")
		}
	}
}
