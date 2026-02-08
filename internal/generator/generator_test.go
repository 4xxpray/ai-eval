package generator

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/stellarlinkco/ai-eval/internal/llm"
)

type fakeProvider struct {
	completeFn func(ctx context.Context, req *llm.Request) (*llm.Response, error)
}

func (p fakeProvider) Name() string { return "fake" }
func (p fakeProvider) Complete(ctx context.Context, req *llm.Request) (*llm.Response, error) {
	return p.completeFn(ctx, req)
}
func (p fakeProvider) CompleteWithTools(context.Context, *llm.Request) (*llm.EvalResult, error) {
	return nil, nil
}

func TestGenerate_Errors(t *testing.T) {
	t.Parallel()

	var gnil *Generator
	if _, err := gnil.Generate(context.Background(), &GenerateRequest{PromptContent: "x"}); err == nil {
		t.Fatalf("Generate(nil generator): expected error")
	}

	g := &Generator{}
	if _, err := g.Generate(context.Background(), &GenerateRequest{PromptContent: "x"}); err == nil {
		t.Fatalf("Generate(nil provider): expected error")
	}
	g.Provider = fakeProvider{completeFn: func(context.Context, *llm.Request) (*llm.Response, error) {
		return nil, nil
	}}
	if _, err := g.Generate(context.Background(), nil); err == nil {
		t.Fatalf("Generate(nil request): expected error")
	}
	if _, err := g.Generate(context.Background(), &GenerateRequest{PromptContent: " \t "}); err == nil {
		t.Fatalf("Generate(empty prompt): expected error")
	}

	g.Provider = fakeProvider{completeFn: func(context.Context, *llm.Request) (*llm.Response, error) {
		return nil, errors.New("upstream")
	}}
	if _, err := g.Generate(context.Background(), &GenerateRequest{PromptContent: "x"}); err == nil || !strings.Contains(err.Error(), "generator: upstream") {
		t.Fatalf("Generate(upstream err): got %v", err)
	}

	g.Provider = fakeProvider{completeFn: func(context.Context, *llm.Request) (*llm.Response, error) {
		return &llm.Response{Content: []llm.ContentBlock{{Type: "text", Text: "not-json"}}}, nil
	}}
	if _, err := g.Generate(context.Background(), &GenerateRequest{PromptContent: "x"}); err == nil || !strings.Contains(err.Error(), "failed to parse response") {
		t.Fatalf("Generate(bad json): got %v", err)
	}
}

func TestGenerate_Success(t *testing.T) {
	t.Parallel()

	var captured string
	g := &Generator{
		Provider: fakeProvider{completeFn: func(ctx context.Context, req *llm.Request) (*llm.Response, error) {
			if req == nil || len(req.Messages) != 1 {
				return nil, errors.New("bad request")
			}
			captured = req.Messages[0].Content
			return &llm.Response{
				Content: []llm.ContentBlock{{
					Type: "text",
					Text: "```json\n" + strings.TrimSpace(`
{
  "analysis": "a",
  "is_system_prompt": true,
  "suggestions": ["s1"],
  "test_cases": [
    {
      "id": "c1",
      "description": "d1",
      "input": {"ENVIRONMENT_CONTEXT": "custom"},
      "expected": {"contains": ["x"], "not_contains": ["y"], "regex": ["z"]},
      "evaluators": [{"type": "llm_judge", "criteria": "crit", "score_threshold": 0.6}]
    },
    {
      "id": "c2",
      "description": "d2",
      "input": null,
      "expected": {"contains": [], "not_contains": [], "regex": []},
      "evaluators": []
    }
  ]
}
`) + "\n```",
				}},
			}, nil
		}},
	}

	res, err := g.Generate(context.Background(), &GenerateRequest{
		PromptContent: "hi",
		PromptName:    "",
		NumCases:      0,
		Variables: map[string]string{
			"ENVIRONMENT_CONTEXT": "default_ctx",
			"FOO":                 "bar",
		},
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if res == nil || res.Suite == nil {
		t.Fatalf("Generate: nil result/suite")
	}
	if res.Analysis != "a" || !res.IsSystemPrompt || len(res.Suggestions) != 1 {
		t.Fatalf("result: %#v", res)
	}
	if res.Suite.Suite != "prompt_tests" || res.Suite.Prompt != "prompt" {
		t.Fatalf("suite meta: %#v", res.Suite)
	}
	if !res.Suite.IsSystemPrompt {
		t.Fatalf("suite IsSystemPrompt: got false want true")
	}
	if len(res.Suite.Cases) != 2 {
		t.Fatalf("len(Cases): got %d want %d", len(res.Suite.Cases), 2)
	}

	c1 := res.Suite.Cases[0]
	if c1.Input["ENVIRONMENT_CONTEXT"] != "custom" {
		t.Fatalf("c1 ENVIRONMENT_CONTEXT overwritten: %#v", c1.Input)
	}
	if c1.Input["FOO"] != "bar" {
		t.Fatalf("c1 FOO missing: %#v", c1.Input)
	}
	if len(c1.Evaluators) != 1 || c1.Evaluators[0].Type != "llm_judge" {
		t.Fatalf("c1 Evaluators: %#v", c1.Evaluators)
	}

	c2 := res.Suite.Cases[1]
	if c2.Input["ENVIRONMENT_CONTEXT"] != "default_ctx" || c2.Input["FOO"] != "bar" {
		t.Fatalf("c2 variables: %#v", c2.Input)
	}

	if !strings.Contains(captured, "Generate 5 diverse test cases") {
		t.Fatalf("prompt missing default NUM_CASES: %q", captured)
	}
	if !strings.Contains(captured, "`{{ENVIRONMENT_CONTEXT}}`: default_ctx") || !strings.Contains(captured, "`{{FOO}}`: bar") {
		t.Fatalf("prompt missing variables section: %q", captured)
	}
}
