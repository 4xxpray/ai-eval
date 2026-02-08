package benchmark

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/stellarlinkco/ai-eval/internal/llm"
)

type fakeProvider struct {
	name string
	fn   func(ctx context.Context, req *llm.Request) (*llm.EvalResult, error)
}

func (p *fakeProvider) Name() string { return p.name }

func (p *fakeProvider) Complete(ctx context.Context, req *llm.Request) (*llm.Response, error) {
	_ = ctx
	_ = req
	return nil, errors.New("not implemented")
}

func (p *fakeProvider) CompleteWithTools(ctx context.Context, req *llm.Request) (*llm.EvalResult, error) {
	if p.fn == nil {
		return nil, nil
	}
	return p.fn(ctx, req)
}

type fakeDataset struct {
	name string
	load func(ctx context.Context) ([]Question, error)
	eval func(response string, expected any) (float64, error)
}

func (d *fakeDataset) Name() string        { return d.name }
func (d *fakeDataset) Description() string { return "fake" }
func (d *fakeDataset) Load(ctx context.Context) ([]Question, error) {
	if d.load == nil {
		return nil, nil
	}
	return d.load(ctx)
}
func (d *fakeDataset) Evaluate(response string, expected any) (float64, error) {
	if d.eval == nil {
		return 0, nil
	}
	return d.eval(response, expected)
}

func TestBenchmarkRunner_Run_Errors(t *testing.T) {
	ctx := context.Background()
	ds := &fakeDataset{name: "x", load: func(ctx context.Context) ([]Question, error) {
		_ = ctx
		return []Question{{ID: "1", Question: "q", Answer: "a"}}, nil
	}}

	{
		var r *BenchmarkRunner
		_, err := r.Run(ctx, ds)
		if err == nil {
			t.Fatalf("nil runner: expected error")
		}
	}
	{
		r := &BenchmarkRunner{Provider: &fakeProvider{name: "p"}}
		_, err := r.Run(nil, ds)
		if err == nil {
			t.Fatalf("nil ctx: expected error")
		}
	}
	{
		r := &BenchmarkRunner{}
		_, err := r.Run(ctx, ds)
		if err == nil {
			t.Fatalf("nil provider: expected error")
		}
	}
	{
		r := &BenchmarkRunner{Provider: &fakeProvider{name: "p"}}
		_, err := r.Run(ctx, nil)
		if err == nil {
			t.Fatalf("nil dataset: expected error")
		}
	}
}

func TestBenchmarkRunner_Run_LoadErrorAndEmptyDataset(t *testing.T) {
	ctx := context.Background()
	provider := &fakeProvider{name: "p"}

	{
		r := &BenchmarkRunner{Provider: provider}
		ds := &fakeDataset{name: "d", load: func(ctx context.Context) ([]Question, error) {
			_ = ctx
			return nil, errors.New("load failed")
		}}
		_, err := r.Run(ctx, ds)
		if err == nil || !strings.Contains(err.Error(), "load failed") {
			t.Fatalf("err=%v", err)
		}
	}

	{
		r := &BenchmarkRunner{Provider: provider}
		ds := &fakeDataset{name: "d", load: func(ctx context.Context) ([]Question, error) {
			_ = ctx
			return nil, nil
		}}
		_, err := r.Run(ctx, ds)
		if err == nil || !strings.Contains(err.Error(), "empty dataset") {
			t.Fatalf("err=%v", err)
		}
	}
}

func TestBenchmarkRunner_Run_ContextAlreadyCanceled(t *testing.T) {
	ctx := &errAfterNContext{
		Context: context.Background(),
		okCalls: 0,
		err:     context.Canceled,
	}

	r := &BenchmarkRunner{
		Provider: &fakeProvider{name: " p "},
	}

	ds := &fakeDataset{
		name: " d ",
		load: func(ctx context.Context) ([]Question, error) {
			_ = ctx
			return []Question{
				{ID: "1", Question: "q", Answer: "a"},
			}, nil
		},
	}

	res, err := r.Run(ctx, ds)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err=%v", err)
	}
	if res == nil {
		t.Fatalf("nil result")
	}
	if res.Model != "p" || res.Dataset != "d" {
		t.Fatalf("model/dataset=%q/%q", res.Model, res.Dataset)
	}
	if len(res.Results) != 0 {
		t.Fatalf("Results=%#v", res.Results)
	}
	if res.TotalTokens != 0 || res.Score != 0 || res.Accuracy != 0 {
		t.Fatalf("tokens/score/acc=%d/%v/%v", res.TotalTokens, res.Score, res.Accuracy)
	}
	if res.TotalTime < 0 {
		t.Fatalf("TotalTime=%v", res.TotalTime)
	}
}

func TestBenchmarkRunner_Run_ProviderAndEvaluateErrors(t *testing.T) {
	ctx := context.Background()

	callN := 0
	provider := &fakeProvider{
		name: "model-x",
		fn: func(ctx context.Context, req *llm.Request) (*llm.EvalResult, error) {
			_ = ctx
			_ = req
			callN++
			if callN == 1 {
				return &llm.EvalResult{
					TextContent:  "ignored",
					LatencyMs:    12,
					InputTokens:  2,
					OutputTokens: 3,
				}, errors.New("provider failed")
			}
			return &llm.EvalResult{
				TextContent:  "response",
				LatencyMs:    7,
				InputTokens:  1,
				OutputTokens: 3,
			}, nil
		},
	}

	ds := &fakeDataset{
		name: "dataset-x",
		load: func(ctx context.Context) ([]Question, error) {
			_ = ctx
			return []Question{
				{ID: "q1", Category: "c1", Question: "Q1", Answer: "E1"},
				{ID: "q2", Category: "c2", Question: "Q2", Answer: "E2"},
			}, nil
		},
		eval: func(response string, expected any) (float64, error) {
			_ = expected
			if response == "response" {
				return 0.5, errors.New("eval failed")
			}
			return 0, nil
		},
	}

	r := &BenchmarkRunner{Provider: provider}
	res, err := r.Run(ctx, ds)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if res == nil {
		t.Fatalf("nil result")
	}
	if len(res.Results) != 2 {
		t.Fatalf("Results=%#v", res.Results)
	}
	if res.TotalTokens != 9 {
		t.Fatalf("TotalTokens=%d", res.TotalTokens)
	}
	if res.Results[0].Error == "" || !strings.Contains(res.Results[0].Error, "provider failed") {
		t.Fatalf("q1 err=%q", res.Results[0].Error)
	}
	if res.Results[1].Error == "" || !strings.Contains(res.Results[1].Error, "eval failed") {
		t.Fatalf("q2 err=%q", res.Results[1].Error)
	}
	if res.Score != 0.25 || res.Accuracy != 0.25 {
		t.Fatalf("Score/Accuracy=%v/%v", res.Score, res.Accuracy)
	}
}

func TestSafeAvg(t *testing.T) {
	if got := safeAvg(1, 0); got != 0 {
		t.Fatalf("got=%v", got)
	}
	if got := safeAvg(2, 4); got != 0.5 {
		t.Fatalf("got=%v", got)
	}
}

func TestFormatPrompt(t *testing.T) {
	q := &Question{Question: " Q ", Choices: []string{"c1", "c2"}}

	{
		got := formatPrompt("mmlu", q)
		if !strings.Contains(got, "multiple-choice") || !strings.Contains(got, "A. c1") {
			t.Fatalf("mmlu=%q", got)
		}
	}
	{
		got := formatPrompt("gsm8k", q)
		if !strings.Contains(got, "Solve the following math problem") || !strings.Contains(got, "Q") {
			t.Fatalf("gsm8k=%q", got)
		}
	}
	{
		got := formatPrompt("humaneval", q)
		if !strings.Contains(got, "Write code") || !strings.Contains(got, "Q") {
			t.Fatalf("humaneval=%q", got)
		}
	}
	{
		got := formatPrompt("unknown", q)
		if !strings.Contains(got, "multiple-choice") || !strings.Contains(got, "Reply with just the letter") {
			t.Fatalf("unknown with choices=%q", got)
		}
	}
	{
		got := formatPrompt("unknown", &Question{Question: "X"})
		if got != "X\n" {
			t.Fatalf("unknown no choices=%q", got)
		}
	}
	{
		got := formatPrompt("mmlu", nil)
		if got != "" {
			t.Fatalf("nil question=%q", got)
		}
	}
}

func TestFormatMCQPrompt(t *testing.T) {
	got := formatMCQPrompt("Q", []string{"A1", "A2"})
	if !strings.Contains(got, "A. A1") || !strings.Contains(got, "B. A2") {
		t.Fatalf("got=%q", got)
	}
	if !strings.HasSuffix(got, "Reply with just the letter (e.g., A).\n") {
		t.Fatalf("suffix=%q", got)
	}
}

func TestBenchmarkRunner_Run_ContextCanceledDuringLoop(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	called := 0
	provider := &fakeProvider{
		name: "p",
		fn: func(ctx context.Context, req *llm.Request) (*llm.EvalResult, error) {
			_ = ctx
			_ = req
			called++
			if called == 1 {
				cancel()
			}
			return &llm.EvalResult{
				TextContent:  "ok",
				LatencyMs:    1,
				InputTokens:  1,
				OutputTokens: 1,
			}, nil
		},
	}

	ds := &fakeDataset{
		name: "d",
		load: func(ctx context.Context) ([]Question, error) {
			_ = ctx
			return []Question{
				{ID: "1", Question: "q1", Answer: "a1"},
				{ID: "2", Question: "q2", Answer: "a2"},
			}, nil
		},
		eval: func(response string, expected any) (float64, error) {
			_ = response
			_ = expected
			return 1, nil
		},
	}

	r := &BenchmarkRunner{Provider: provider}
	res, err := r.Run(ctx, ds)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err=%v", err)
	}
	if res == nil {
		t.Fatalf("nil result")
	}
	if len(res.Results) != 1 {
		t.Fatalf("Results=%#v", res.Results)
	}
	if res.TotalTokens != 2 {
		t.Fatalf("TotalTokens=%d", res.TotalTokens)
	}
	if res.Score != 1 || res.Accuracy != 1 {
		t.Fatalf("Score/Accuracy=%v/%v", res.Score, res.Accuracy)
	}
	if res.TotalTime < 0*time.Second {
		t.Fatalf("TotalTime=%v", res.TotalTime)
	}
}
