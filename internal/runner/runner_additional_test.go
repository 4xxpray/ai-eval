package runner

import (
	"context"
	"errors"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stellarlinkco/ai-eval/internal/evaluator"
	"github.com/stellarlinkco/ai-eval/internal/evaluator/agent"
	"github.com/stellarlinkco/ai-eval/internal/evaluator/rag"
	"github.com/stellarlinkco/ai-eval/internal/evaluator/safety"
	"github.com/stellarlinkco/ai-eval/internal/llm"
	"github.com/stellarlinkco/ai-eval/internal/prompt"
	"github.com/stellarlinkco/ai-eval/internal/testcase"
)

type stubProvider struct {
	completeWithTools func(ctx context.Context, req *llm.Request) (*llm.EvalResult, error)
}

func (p *stubProvider) Name() string { return "stub" }

func (p *stubProvider) Complete(ctx context.Context, req *llm.Request) (*llm.Response, error) {
	_ = ctx
	_ = req
	return nil, errors.New("stub: Complete not implemented")
}

func (p *stubProvider) CompleteWithTools(ctx context.Context, req *llm.Request) (*llm.EvalResult, error) {
	if p == nil || p.completeWithTools == nil {
		return nil, nil
	}
	return p.completeWithTools(ctx, req)
}

type stubToolLoopProvider struct {
	*stubProvider
	completeMultiTurn func(ctx context.Context, req *llm.Request, toolExecutor func(llm.ToolUse) (string, error), maxSteps int) (*llm.MultiTurnResult, error)
}

func (p *stubToolLoopProvider) CompleteMultiTurn(
	ctx context.Context,
	req *llm.Request,
	toolExecutor func(llm.ToolUse) (string, error),
	maxSteps int,
) (*llm.MultiTurnResult, error) {
	if p == nil || p.completeMultiTurn == nil {
		return nil, nil
	}
	return p.completeMultiTurn(ctx, req, toolExecutor, maxSteps)
}

type recordingEvaluator struct {
	name string

	gotResponse string
	gotExpected any

	res *evaluator.Result
	err error
}

func (e *recordingEvaluator) Name() string { return e.name }

func (e *recordingEvaluator) Evaluate(ctx context.Context, response string, expected any) (*evaluator.Result, error) {
	_ = ctx
	e.gotResponse = response
	e.gotExpected = expected
	return e.res, e.err
}

func TestNewRunner_DefaultsAndClamps(t *testing.T) {
	t.Parallel()

	cases := []struct {
		cfg  Config
		want Config
	}{
		{
			cfg:  Config{Trials: 0, Concurrency: 0, PassThreshold: -1},
			want: Config{Trials: 1, Concurrency: 1, PassThreshold: 0},
		},
		{
			cfg:  Config{Trials: -5, Concurrency: -1, PassThreshold: 2},
			want: Config{Trials: 1, Concurrency: 1, PassThreshold: 1},
		},
	}

	for _, tc := range cases {
		r := NewRunner(&stubProvider{}, nil, tc.cfg)
		if r.cfg.Trials != tc.want.Trials {
			t.Fatalf("Trials: got %d want %d", r.cfg.Trials, tc.want.Trials)
		}
		if r.cfg.Concurrency != tc.want.Concurrency {
			t.Fatalf("Concurrency: got %d want %d", r.cfg.Concurrency, tc.want.Concurrency)
		}
		if r.cfg.PassThreshold != tc.want.PassThreshold {
			t.Fatalf("PassThreshold: got %v want %v", r.cfg.PassThreshold, tc.want.PassThreshold)
		}
		if r.sem == nil || cap(r.sem) != tc.want.Concurrency {
			t.Fatalf("sem cap: got %d want %d", cap(r.sem), tc.want.Concurrency)
		}
	}
}

func TestRunCase_NilChecks(t *testing.T) {
	t.Parallel()

	r := &Runner{
		provider: &stubProvider{},
		registry: evaluator.NewRegistry(),
		sem:      make(chan struct{}, 1),
	}
	p := &prompt.Prompt{Name: "p", Template: "x"}
	tc := &testcase.TestCase{ID: "c1", Input: map[string]any{}}

	if _, err := (*Runner)(nil).RunCase(context.Background(), p, tc); err == nil {
		t.Fatalf("RunCase(nil runner): expected error")
	}
	if _, err := r.RunCase(nil, p, tc); err == nil {
		t.Fatalf("RunCase(nil ctx): expected error")
	}

	r2 := &Runner{registry: evaluator.NewRegistry(), sem: make(chan struct{}, 1)}
	if _, err := r2.RunCase(context.Background(), p, tc); err == nil {
		t.Fatalf("RunCase(nil provider): expected error")
	}

	r3 := &Runner{provider: &stubProvider{}, sem: make(chan struct{}, 1)}
	if _, err := r3.RunCase(context.Background(), p, tc); err == nil {
		t.Fatalf("RunCase(nil registry): expected error")
	}
	if _, err := r.RunCase(context.Background(), nil, tc); err == nil {
		t.Fatalf("RunCase(nil prompt): expected error")
	}
	if _, err := r.RunCase(context.Background(), p, nil); err == nil {
		t.Fatalf("RunCase(nil test case): expected error")
	}
}

func TestRunCase_DefaultTrialsAndSystemPromptFallback(t *testing.T) {
	t.Parallel()

	const fallback = "Please process this request according to your instructions."
	var calls int32
	provider := &stubProvider{completeWithTools: func(ctx context.Context, req *llm.Request) (*llm.EvalResult, error) {
		_ = ctx
		atomic.AddInt32(&calls, 1)

		if req.System != "SYS" {
			t.Fatalf("System: got %q want %q", req.System, "SYS")
		}
		if len(req.Messages) != 1 {
			t.Fatalf("len(Messages): got %d want %d", len(req.Messages), 1)
		}
		if req.Messages[0].Role != "user" || req.Messages[0].Content != fallback {
			t.Fatalf("Messages[0]: %#v", req.Messages[0])
		}

		return &llm.EvalResult{
			TextContent:  "ok",
			LatencyMs:    1,
			InputTokens:  1,
			OutputTokens: 1,
		}, nil
	}}

	r := NewRunner(provider, evaluator.NewRegistry(), Config{
		Trials:        0,
		PassThreshold: 0,
		Concurrency:   1,
	})

	p := &prompt.Prompt{Name: "p", Template: "SYS", IsSystemPrompt: true}
	tc := &testcase.TestCase{ID: "c1", Trials: 0, Input: map[string]any{}}

	got, err := r.RunCase(context.Background(), p, tc)
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if got == nil || len(got.Trials) != 1 {
		t.Fatalf("RunCase: %#v", got)
	}
	if gotCalls := atomic.LoadInt32(&calls); gotCalls != 1 {
		t.Fatalf("calls: got %d want %d", gotCalls, 1)
	}
}

func TestAcquire_Errors(t *testing.T) {
	t.Parallel()

	r := &Runner{}
	if err := r.acquire(context.Background()); err == nil {
		t.Fatalf("acquire(nil sem): expected error")
	}

	r.sem = make(chan struct{}, 1)
	r.sem <- struct{}{}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if err := r.acquire(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("acquire(canceled): got %v want %v", err, context.Canceled)
	}
}

func TestPromptTools_ResponseText(t *testing.T) {
	t.Parallel()

	tools := promptTools([]prompt.Tool{
		{Name: " "},
		{Name: "t", Description: " d "},
	})
	if len(tools) != 1 || tools[0].Name != "t" {
		t.Fatalf("promptTools: %#v", tools)
	}
	if tools[0].InputSchema == nil || tools[0].InputSchema["type"] != "object" {
		t.Fatalf("promptTools schema: %#v", tools[0].InputSchema)
	}

	if got := responseText(nil); got != "" {
		t.Fatalf("responseText(nil): got %q want empty", got)
	}
	got := responseText(&llm.Response{Content: []llm.ContentBlock{
		{Type: "tool_use", Name: "x"},
		{Type: "text", Text: "a"},
		{Type: "text", Text: "b"},
	}})
	if got != "ab" {
		t.Fatalf("responseText: got %q want %q", got, "ab")
	}
}

func TestToolExecutorFromMocks(t *testing.T) {
	t.Parallel()

	exec := toolExecutorFromMocks([]testcase.ToolMock{
		{
			Name:  "git",
			Error: "boom",
			Match: map[string]any{
				"cmd": "status",
				"n":   int64(1),
				"nested": map[string]any{
					"a": []any{float32(2), "x"},
				},
			},
		},
		{
			Name:     "git",
			Response: "ok",
			Match: map[string]any{
				"cmd": "log",
			},
		},
	})

	_, err := exec(llm.ToolUse{
		Name: "git",
		Input: map[string]any{
			"cmd": "status",
			"n":   1,
			"nested": map[string]any{
				"a": []any{2, "x"},
			},
		},
	})
	if err == nil || err.Error() != "boom" {
		t.Fatalf("executor error: %v", err)
	}

	out, err := exec(llm.ToolUse{Name: "git", Input: map[string]any{"cmd": "log"}})
	if err != nil || out != "ok" {
		t.Fatalf("executor: out=%q err=%v", out, err)
	}

	if _, err := exec(llm.ToolUse{Name: "other"}); err == nil {
		t.Fatalf("executor(no mock): expected error")
	}
}

func TestMatchHelpers(t *testing.T) {
	t.Parallel()

	if !matchArgs(nil, nil) {
		t.Fatalf("matchArgs(empty): expected true")
	}
	if matchArgs(map[string]any{"a": 1}, nil) {
		t.Fatalf("matchArgs(nil actual): expected false")
	}
	if matchArgs(map[string]any{"a": 1}, map[string]any{}) {
		t.Fatalf("matchArgs(missing key): expected false")
	}

	if !matchValue(nil, nil) {
		t.Fatalf("matchValue(nil,nil): expected true")
	}
	if matchValue(nil, 1) {
		t.Fatalf("matchValue(nil,non-nil): expected false")
	}
	if !matchValue(1, int8(1)) {
		t.Fatalf("matchValue(numeric equal): expected true")
	}
	if matchValue(1, "x") {
		t.Fatalf("matchValue(numeric vs string): expected false")
	}
	if !matchValue(map[string]any{"a": 1}, map[string]any{"a": int64(1)}) {
		t.Fatalf("matchValue(map): expected true")
	}
	if matchValue(map[string]any{"a": 1}, "x") {
		t.Fatalf("matchValue(map vs non-map): expected false")
	}
	if matchValue([]any{1}, []any{1, 2}) {
		t.Fatalf("matchValue(slice len): expected false")
	}
	if matchValue([]any{1, 2}, []any{1, 3}) {
		t.Fatalf("matchValue(slice mismatch): expected false")
	}
	if !matchValue([]any{1, 2}, []any{int64(1), float32(2)}) {
		t.Fatalf("matchValue(slice match): expected true")
	}
	if matchValue([]any{1}, map[string]any{}) {
		t.Fatalf("matchValue(slice vs map): expected false")
	}
	if !matchValue("x", "x") {
		t.Fatalf("matchValue(deep equal): expected true")
	}

	nums := []any{
		float64(1),
		float32(1),
		int(1),
		int8(1),
		int16(1),
		int32(1),
		int64(1),
		uint(1),
		uint8(1),
		uint16(1),
		uint32(1),
		uint64(1),
	}
	for _, v := range nums {
		if _, ok := number(v); !ok {
			t.Fatalf("number(%T): expected ok", v)
		}
	}
	if _, ok := number("x"); ok {
		t.Fatalf("number(string): expected !ok")
	}
}

func TestBuildFullResponse(t *testing.T) {
	t.Parallel()

	if got := buildFullResponse("x", nil); got != "x" {
		t.Fatalf("buildFullResponse(no tools): got %q want %q", got, "x")
	}

	out := buildFullResponse("x", []llm.ToolUse{{
		ID:   "t1",
		Name: "git",
		Input: map[string]any{
			"cmd": "status",
		},
	}})
	if !strings.Contains(out, "## Tool Calls Made") || !strings.Contains(out, "`git`") || !strings.Contains(out, "```json") {
		t.Fatalf("buildFullResponse: %q", out)
	}

	out = buildFullResponse("x", []llm.ToolUse{{
		ID:   "t1",
		Name: "git",
		Input: map[string]any{
			"bad": func() {},
		},
	}})
	if !strings.Contains(out, "## Tool Calls Made") || strings.Contains(out, "```json") {
		t.Fatalf("buildFullResponse(marshal error): %q", out)
	}
}

func TestBuildEvalTasks_CoversTypes(t *testing.T) {
	t.Parallel()

	tc := &testcase.TestCase{
		ID: "c",
		Expected: testcase.Expected{
			ExactMatch:  "ex",
			Contains:    []string{"a"},
			NotContains: []string{"b"},
			Regex:       []string{"re"},
			JSONSchema:  map[string]any{"type": "object"},
			ToolCalls:   []testcase.ToolCallExpect{{Name: "git", Order: 1, Required: true}},
		},
		MaxSteps: 7,
		Evaluators: []testcase.EvaluatorConfig{
			{Type: "llm_judge", Criteria: "c", Rubric: []string{"r"}, ScoreScale: 5, ScoreThreshold: 0.7},
			{Type: "similarity", Reference: "ref", ScoreThreshold: 0.6},
			{Type: "factuality", GroundTruth: "gt"},
			{Type: "tool_call", ScoreThreshold: 0.9},
			{Type: "faithfulness", Context: "ctx", ScoreThreshold: 0.8},
			{Type: "relevancy", Question: "q", ScoreThreshold: 0.8},
			{Type: "precision", Context: "c", Question: "q", ScoreThreshold: 0.8},
			{Type: "task_completion", Task: "t", CriteriaList: []string{"c"}, ScoreThreshold: 0.6},
			{Type: "tool_selection", ExpectedTools: []string{"git"}, ScoreThreshold: 0.5},
			{Type: "efficiency", MaxSteps: 0, MaxTokens: -1, ScoreThreshold: 0.5},
			{Type: "hallucination", GroundTruth: "gt", ScoreThreshold: 0.9},
			{Type: "toxicity", ScoreThreshold: 0.2},
			{Type: "bias", Categories: []string{"x"}, ScoreThreshold: 0.2},
			{Type: "exact", ScoreThreshold: 0.1},
			{Type: "contains", ScoreThreshold: 0.1},
			{Type: "regex", ScoreThreshold: 0.1},
			{Type: "json_schema", ScoreThreshold: 0.1},
			{Type: "unknown"},
			{Type: " "},
		},
	}

	tasks, toolCallThreshold := buildEvalTasks(tc, "promptCtx")
	if toolCallThreshold != 0.9 {
		t.Fatalf("toolCallThreshold: got %v want %v", toolCallThreshold, 0.9)
	}
	if len(tasks) == 0 {
		t.Fatalf("tasks: expected non-empty")
	}

	found := make(map[string]evalTask)
	for _, task := range tasks {
		found[task.typ] = task
	}
	if found["toxicity"].scoreThreshold != 0.8 {
		t.Fatalf("toxicity scoreThreshold: got %v want %v", found["toxicity"].scoreThreshold, 0.8)
	}
	if found["bias"].scoreThreshold != 0.8 {
		t.Fatalf("bias scoreThreshold: got %v want %v", found["bias"].scoreThreshold, 0.8)
	}
	if eff := found["efficiency"].expected.(map[string]any); eff["max_steps"] != 7 || eff["max_tokens"] != 1000 {
		t.Fatalf("efficiency expected: %#v", eff)
	}
}

func TestBuildEvalTasks_ImplicitExpectedTasks(t *testing.T) {
	t.Parallel()

	tc := &testcase.TestCase{
		ID: "c",
		Expected: testcase.Expected{
			ExactMatch:  "ex",
			Contains:    []string{"a"},
			NotContains: []string{"b"},
			Regex:       []string{"re"},
			JSONSchema:  map[string]any{"type": "object"},
		},
	}

	tasks, toolCallThreshold := buildEvalTasks(tc, "ctx")
	if toolCallThreshold != 0 {
		t.Fatalf("toolCallThreshold: got %v want 0", toolCallThreshold)
	}

	wantTypes := []string{"exact", "contains", "not_contains", "regex", "json_schema"}
	for _, typ := range wantTypes {
		found := false
		for _, task := range tasks {
			if task.typ == typ {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("missing task type %q", typ)
		}
	}
}

func TestBuildEvalTasks_EfficiencyDefaults(t *testing.T) {
	t.Parallel()

	tc := &testcase.TestCase{
		ID: "c",
		Evaluators: []testcase.EvaluatorConfig{
			{Type: "efficiency"},
		},
	}

	tasks, _ := buildEvalTasks(tc, "ctx")
	for _, task := range tasks {
		if task.typ != "efficiency" {
			continue
		}
		exp, ok := task.expected.(map[string]any)
		if !ok {
			t.Fatalf("expected type: %T", task.expected)
		}
		if exp["max_steps"] != 5 {
			t.Fatalf("max_steps: got %#v want %d", exp["max_steps"], 5)
		}
		if exp["max_tokens"] != 1000 {
			t.Fatalf("max_tokens: got %#v want %d", exp["max_tokens"], 1000)
		}
		return
	}
	t.Fatalf("missing efficiency task")
}

func TestEvaluateTrial_Branches(t *testing.T) {
	t.Parallel()

	reg := evaluator.NewRegistry()
	containsEval := &recordingEvaluator{
		name: "contains",
		res:  &evaluator.Result{Passed: true, Score: 0.4, Message: "x"},
	}
	notContainsEval := &recordingEvaluator{
		name: "not_contains",
		res:  &evaluator.Result{Passed: true, Score: 1, Message: "x"},
	}
	errorEval := &recordingEvaluator{name: "error_eval", err: errors.New("boom")}
	nilResEval := &recordingEvaluator{name: "nilres_eval"}
	effEval := &recordingEvaluator{name: "efficiency", res: &evaluator.Result{Passed: true, Score: 1}}
	toolSelEval := &recordingEvaluator{name: "tool_selection", res: &evaluator.Result{Passed: true, Score: 1}}

	reg.Register(containsEval)
	reg.Register(notContainsEval)
	reg.Register(errorEval)
	reg.Register(nilResEval)
	reg.Register(effEval)
	reg.Register(toolSelEval)

	r := &Runner{registry: reg}

	tc := &testcase.TestCase{
		ID: "c",
		Expected: testcase.Expected{
			Contains:    []string{"a"},
			NotContains: []string{"forbid"},
			ToolCalls:   []testcase.ToolCallExpect{{Name: "git", Order: 1, Required: true}},
		},
		Evaluators: []testcase.EvaluatorConfig{
			{Type: "contains", ScoreThreshold: 0.5},
			{Type: "missing_eval"},
			{Type: "error_eval"},
			{Type: "nilres_eval"},
			{Type: "tool_selection", ExpectedTools: []string{"git"}},
			{Type: "efficiency", MaxSteps: 1, MaxTokens: 10},
			{Type: "tool_call", ScoreThreshold: 0.9},
		},
	}

	toolCalls := []llm.ToolUse{{Name: "git", Input: map[string]any{"cmd": "status"}}}

	results, passed, score := r.evaluateTrial(context.Background(), tc, "promptCtx", "resp", toolCalls, 2, 20)
	if len(results) == 0 {
		t.Fatalf("results: expected non-empty")
	}
	if passed {
		t.Fatalf("passed: expected false (contains threshold + missing/error evaluators)")
	}
	if score <= 0 {
		t.Fatalf("score: got %v want > 0", score)
	}

	if containsEval.gotResponse == "resp" || !strings.Contains(containsEval.gotResponse, "## Tool Calls Made") {
		t.Fatalf("contains evalResponse: %q", containsEval.gotResponse)
	}
	if notContainsEval.gotResponse != "resp" {
		t.Fatalf("not_contains evalResponse: got %q want %q", notContainsEval.gotResponse, "resp")
	}

	if results[0].Passed {
		t.Fatalf("contains Passed: got true want false (threshold override)")
	}

	if exp, ok := toolSelEval.gotExpected.(map[string]any); !ok || exp["tool_calls"] == nil {
		t.Fatalf("tool_selection expected: %#v", toolSelEval.gotExpected)
	}
	if exp, ok := effEval.gotExpected.(map[string]any); !ok || exp["actual_steps"] != 2 || exp["actual_tokens"] != 20 {
		t.Fatalf("efficiency expected: %#v", effEval.gotExpected)
	}

	results, passed, _ = r.evaluateTrial(context.Background(), nil, "x", "y", nil, 0, 0)
	if passed || len(results) != 1 || results[0].Message != "runner: nil test case" {
		t.Fatalf("nil tc: results=%#v passed=%v", results, passed)
	}
}

func TestEvaluateTrial_ToolCallMismatchFails(t *testing.T) {
	t.Parallel()

	r := &Runner{registry: evaluator.NewRegistry()}
	tc := &testcase.TestCase{
		ID: "c",
		Expected: testcase.Expected{
			ToolCalls: []testcase.ToolCallExpect{
				{Name: "git", Order: 1, Required: true},
			},
		},
	}

	results, passed, _ := r.evaluateTrial(context.Background(), tc, "ctx", "resp", nil, 0, 0)
	if passed {
		t.Fatalf("passed: expected false")
	}
	if len(results) == 0 {
		t.Fatalf("results: expected non-empty")
	}
}

func TestRegisterLLMEvaluators_RegisterAndFill(t *testing.T) {
	t.Parallel()

	p := &stubProvider{}

	empty := evaluator.NewRegistry()
	NewRunner(p, empty, Config{Trials: 1, Concurrency: 1})
	if _, ok := empty.Get("llm_judge"); !ok {
		t.Fatalf("expected llm_judge to be registered")
	}
	if _, ok := empty.Get("efficiency"); !ok {
		t.Fatalf("expected efficiency to be registered")
	}

	reg := evaluator.NewRegistry()
	judge := &evaluator.LLMJudgeEvaluator{}
	sim := &evaluator.SimilarityEvaluator{}
	fact := &evaluator.FactualityEvaluator{}
	faith := &rag.FaithfulnessEvaluator{}
	rel := &rag.RelevancyEvaluator{}
	prec := &rag.PrecisionEvaluator{}
	task := &agent.TaskCompletionEvaluator{}
	toolSel := &agent.ToolSelectionEvaluator{}
	hall := &safety.HallucinationEvaluator{}
	tox := &safety.ToxicityEvaluator{}
	bias := &safety.BiasEvaluator{}

	reg.Register(judge)
	reg.Register(sim)
	reg.Register(fact)
	reg.Register(faith)
	reg.Register(rel)
	reg.Register(prec)
	reg.Register(task)
	reg.Register(toolSel)
	reg.Register(hall)
	reg.Register(tox)
	reg.Register(bias)

	r := &Runner{provider: p, registry: reg}
	r.registerLLMEvaluators()

	if judge.Provider == nil || sim.Provider == nil || fact.Provider == nil {
		t.Fatalf("expected Provider fields to be set")
	}
	if faith.Client == nil || rel.Client == nil || prec.Client == nil {
		t.Fatalf("expected RAG Client fields to be set")
	}
	if task.Client == nil || toolSel.Client == nil {
		t.Fatalf("expected agent Client fields to be set")
	}
	if hall.Client == nil || tox.Client == nil || bias.Client == nil {
		t.Fatalf("expected safety Client fields to be set")
	}
}

func TestRunCase_SystemPrompt_UserTask(t *testing.T) {
	t.Parallel()

	var gotReq atomic.Pointer[llm.Request]
	provider := &stubProvider{completeWithTools: func(ctx context.Context, req *llm.Request) (*llm.EvalResult, error) {
		_ = ctx
		gotReq.Store(req)
		return &llm.EvalResult{TextContent: "ok", InputTokens: 1, OutputTokens: 2, LatencyMs: 3}, nil
	}}

	reg := evaluator.NewRegistry()
	reg.Register(evaluator.ExactEvaluator{})

	r := NewRunner(provider, reg, Config{Trials: 1, PassThreshold: 1, Concurrency: 1})

	p := &prompt.Prompt{Name: "p", Template: "sys {{.x}}", IsSystemPrompt: true}
	tc := &testcase.TestCase{
		ID:    "c1",
		Input: map[string]any{"x": "y", "user_task": "do"},
		Expected: testcase.Expected{
			ExactMatch: "ok",
		},
	}

	res, err := r.RunCase(context.Background(), p, tc)
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if !res.Passed {
		t.Fatalf("Passed: got false want true")
	}

	req := gotReq.Load()
	if req == nil {
		t.Fatalf("expected request to be captured")
	}
	if req.System != "sys y" {
		t.Fatalf("System: got %q want %q", req.System, "sys y")
	}
	if len(req.Messages) != 1 || req.Messages[0].Content != "do" {
		t.Fatalf("Messages: %#v", req.Messages)
	}
}

func TestRunCase_PromptRenderError(t *testing.T) {
	t.Parallel()

	var calls int32
	provider := &stubProvider{completeWithTools: func(ctx context.Context, req *llm.Request) (*llm.EvalResult, error) {
		_ = ctx
		_ = req
		atomic.AddInt32(&calls, 1)
		return &llm.EvalResult{TextContent: "ok"}, nil
	}}

	r := NewRunner(provider, evaluator.NewRegistry(), Config{Trials: 1, PassThreshold: 1, Concurrency: 1})
	p := &prompt.Prompt{
		Name:     "p",
		Template: "{{.req}}",
		Variables: []prompt.Variable{
			{Name: "req", Required: true},
		},
	}
	tc := &testcase.TestCase{ID: "c1", Input: map[string]any{}}

	got, err := r.RunCase(context.Background(), p, tc)
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if got == nil || got.Error == nil {
		t.Fatalf("expected RunResult with Error set, got %#v", got)
	}
	if atomic.LoadInt32(&calls) != 0 {
		t.Fatalf("provider calls: got %d want 0", atomic.LoadInt32(&calls))
	}
	if len(got.Trials) != 1 || len(got.Trials[0].Evaluations) != 1 {
		t.Fatalf("Trials: %#v", got.Trials)
	}
	if got.Trials[0].Evaluations[0].Passed {
		t.Fatalf("evaluation: expected failed, got %#v", got.Trials[0].Evaluations[0])
	}
}

func TestRunCase_MultiTurn_NoToolLoopProvider(t *testing.T) {
	t.Parallel()

	provider := &stubProvider{completeWithTools: func(ctx context.Context, req *llm.Request) (*llm.EvalResult, error) {
		_ = ctx
		_ = req
		return &llm.EvalResult{TextContent: "ok"}, nil
	}}

	r := NewRunner(provider, evaluator.NewRegistry(), Config{Trials: 1, PassThreshold: 1, Concurrency: 1})
	p := &prompt.Prompt{Name: "p", Template: "x", Tools: []prompt.Tool{{Name: "git"}}}
	tc := &testcase.TestCase{
		ID:        "c1",
		Input:     map[string]any{},
		ToolMocks: []testcase.ToolMock{{Name: "git", Response: "x"}},
	}

	got, err := r.RunCase(context.Background(), p, tc)
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if got == nil || got.Error == nil {
		t.Fatalf("expected RunResult with Error set, got %#v", got)
	}
	if !strings.Contains(got.Error.Error(), "tool loops") {
		t.Fatalf("Error: %v", got.Error)
	}
}

func TestRunCase_ProviderError(t *testing.T) {
	t.Parallel()

	provider := &stubProvider{completeWithTools: func(ctx context.Context, req *llm.Request) (*llm.EvalResult, error) {
		_ = ctx
		_ = req
		return nil, errors.New("fail")
	}}

	r := NewRunner(provider, evaluator.NewRegistry(), Config{Trials: 1, PassThreshold: 1, Concurrency: 1})
	p := &prompt.Prompt{Name: "p", Template: "x"}
	tc := &testcase.TestCase{
		ID:    "c1",
		Input: map[string]any{},
	}

	got, err := r.RunCase(context.Background(), p, tc)
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if got == nil || got.Error == nil {
		t.Fatalf("expected RunResult with Error set, got %#v", got)
	}
	if got.Error.Error() != "fail" {
		t.Fatalf("Error: got %v want %q", got.Error, "fail")
	}
	if len(got.Trials) != 1 || len(got.Trials[0].Evaluations) != 1 {
		t.Fatalf("Trials: %#v", got.Trials)
	}
	if got.Trials[0].Evaluations[0].Message != "fail" {
		t.Fatalf("Evaluations[0].Message: got %q want %q", got.Trials[0].Evaluations[0].Message, "fail")
	}
}

func TestRunCase_ContextCanceledBetweenTrials(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var calls int32
	provider := &stubProvider{completeWithTools: func(ctx context.Context, req *llm.Request) (*llm.EvalResult, error) {
		_ = ctx
		_ = req
		if atomic.AddInt32(&calls, 1) == 1 {
			cancel()
		}
		return &llm.EvalResult{TextContent: "ok", InputTokens: 1, OutputTokens: 1}, nil
	}}

	reg := evaluator.NewRegistry()
	reg.Register(evaluator.ExactEvaluator{})

	r := NewRunner(provider, reg, Config{Trials: 3, PassThreshold: 1, Concurrency: 1})
	p := &prompt.Prompt{Name: "p", Template: "x"}
	tc := &testcase.TestCase{
		ID:    "c1",
		Input: map[string]any{},
		Expected: testcase.Expected{
			ExactMatch: "ok",
		},
	}

	got, err := r.RunCase(ctx, p, tc)
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if got == nil || got.Error == nil || !errors.Is(got.Error, context.Canceled) {
		t.Fatalf("expected canceled error, got %#v", got)
	}
	if len(got.Trials) != 1 {
		t.Fatalf("Trials: got %d want %d", len(got.Trials), 1)
	}
}

func TestRunCase_AcquireCanceledContext(t *testing.T) {
	t.Parallel()

	r := &Runner{
		provider: &stubProvider{},
		registry: evaluator.NewRegistry(),
		sem:      make(chan struct{}, 1),
		cfg:      Config{Trials: 1, Concurrency: 1},
	}
	r.sem <- struct{}{}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := r.RunCase(ctx, &prompt.Prompt{Name: "p", Template: "x"}, &testcase.TestCase{ID: "c"})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("RunCase(acquire canceled): %v", err)
	}
}

func TestRunSuite_ContextCanceled_FillsRemaining(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	r := &Runner{
		provider: &stubProvider{},
		registry: evaluator.NewRegistry(),
		sem:      make(chan struct{}, 1),
	}
	suite := &testcase.TestSuite{
		Suite: "s",
		Cases: []testcase.TestCase{
			{ID: "c1"},
			{ID: "c2"},
		},
	}

	res, err := r.RunSuite(ctx, &prompt.Prompt{Name: "p", Template: "x"}, suite)
	if err != nil {
		t.Fatalf("RunSuite: %v", err)
	}
	if res == nil || len(res.Results) != 2 {
		t.Fatalf("RunSuite results: %#v", res)
	}
	for _, rr := range res.Results {
		if !errors.Is(rr.Error, context.Canceled) {
			t.Fatalf("case %q error: %v", rr.CaseID, rr.Error)
		}
	}
}

func TestRunSuite_RunCaseError(t *testing.T) {
	t.Parallel()

	r := &Runner{
		registry: evaluator.NewRegistry(),
		sem:      make(chan struct{}, 1),
	}

	suite := &testcase.TestSuite{
		Suite: "s",
		Cases: []testcase.TestCase{
			{ID: "c1", Input: map[string]any{}},
		},
	}

	res, err := r.RunSuite(context.Background(), &prompt.Prompt{Name: "p", Template: "x"}, suite)
	if err != nil {
		t.Fatalf("RunSuite: %v", err)
	}
	if res == nil || len(res.Results) != 1 || res.Results[0].Error == nil {
		t.Fatalf("RunSuite results: %#v", res)
	}
	if !strings.Contains(res.Results[0].Error.Error(), "nil llm provider") {
		t.Fatalf("error: %v", res.Results[0].Error)
	}
}

type errOnlyContext struct {
	err error
}

func (c errOnlyContext) Deadline() (time.Time, bool) { return time.Time{}, false }
func (c errOnlyContext) Done() <-chan struct{}       { return nil }
func (c errOnlyContext) Err() error                  { return c.err }
func (c errOnlyContext) Value(key any) any           { return nil }

func TestRunSuite_ContextErrBranch(t *testing.T) {
	t.Parallel()

	sentinel := errors.New("sentinel")
	ctx := errOnlyContext{err: sentinel}

	r := &Runner{
		provider: &stubProvider{},
		registry: evaluator.NewRegistry(),
		sem:      make(chan struct{}, 1),
	}

	suite := &testcase.TestSuite{
		Suite: "s",
		Cases: []testcase.TestCase{
			{ID: "c1", Input: map[string]any{}},
		},
	}

	res, err := r.RunSuite(ctx, &prompt.Prompt{Name: "p", Template: "x"}, suite)
	if err != nil {
		t.Fatalf("RunSuite: %v", err)
	}
	if res == nil || len(res.Results) != 1 || !errors.Is(res.Results[0].Error, sentinel) {
		t.Fatalf("RunSuite results: %#v", res)
	}
}

type doneAfterFirstCallContext struct {
	doneCh <-chan struct{}
	err    error
	calls  int32
}

func (c *doneAfterFirstCallContext) Deadline() (time.Time, bool) { return time.Time{}, false }
func (c *doneAfterFirstCallContext) Done() <-chan struct{} {
	if atomic.AddInt32(&c.calls, 1) == 1 {
		return nil
	}
	return c.doneCh
}
func (c *doneAfterFirstCallContext) Err() error        { return c.err }
func (c *doneAfterFirstCallContext) Value(key any) any { return nil }

func TestRunSuite_GoroutineContextDoneBranch(t *testing.T) {
	t.Parallel()

	done := make(chan struct{})
	close(done)
	ctx := &doneAfterFirstCallContext{doneCh: done, err: context.Canceled}

	r := &Runner{}
	suite := &testcase.TestSuite{
		Suite: "s",
		Cases: []testcase.TestCase{
			{ID: "c1"},
		},
	}

	res, err := r.RunSuite(ctx, &prompt.Prompt{Name: "p", Template: "x"}, suite)
	if err != nil {
		t.Fatalf("RunSuite: %v", err)
	}
	if res == nil || len(res.Results) != 1 {
		t.Fatalf("RunSuite results: %#v", res)
	}
	if !errors.Is(res.Results[0].Error, context.Canceled) {
		t.Fatalf("case error: %v", res.Results[0].Error)
	}
}

func TestRunSuite_NilChecks(t *testing.T) {
	t.Parallel()

	r := &Runner{provider: &stubProvider{}, registry: evaluator.NewRegistry(), sem: make(chan struct{}, 1)}
	p := &prompt.Prompt{Name: "p", Template: "x"}
	suite := &testcase.TestSuite{Suite: "s"}

	if _, err := (*Runner)(nil).RunSuite(context.Background(), p, suite); err == nil {
		t.Fatalf("RunSuite(nil runner): expected error")
	}
	if _, err := r.RunSuite(nil, p, suite); err == nil {
		t.Fatalf("RunSuite(nil ctx): expected error")
	}
	if _, err := r.RunSuite(context.Background(), nil, suite); err == nil {
		t.Fatalf("RunSuite(nil prompt): expected error")
	}
	if _, err := r.RunSuite(context.Background(), p, nil); err == nil {
		t.Fatalf("RunSuite(nil suite): expected error")
	}
}

func TestRunCase_ProviderReturnsNilResult(t *testing.T) {
	t.Parallel()

	provider := &stubProvider{completeWithTools: func(ctx context.Context, req *llm.Request) (*llm.EvalResult, error) {
		_ = ctx
		_ = req
		return nil, nil
	}}
	reg := evaluator.NewRegistry()
	reg.Register(&recordingEvaluator{name: "exact", res: &evaluator.Result{Passed: true, Score: 1}})

	r := NewRunner(provider, reg, Config{Trials: 1, PassThreshold: 1, Concurrency: 1})
	p := &prompt.Prompt{Name: "p", Template: "x"}
	tc := &testcase.TestCase{
		ID:    "c1",
		Input: map[string]any{},
		Expected: testcase.Expected{
			ExactMatch: "ok",
		},
	}

	got, err := r.RunCase(context.Background(), p, tc)
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if got == nil || len(got.Trials) != 1 {
		t.Fatalf("RunCase: %#v", got)
	}
}

func TestRunner_release(t *testing.T) {
	t.Parallel()

	r := &Runner{sem: make(chan struct{}, 1)}
	r.sem <- struct{}{}
	r.release()
}

func TestRunCase_MultiTurn_ToolExecutorErrorPropagates(t *testing.T) {
	t.Parallel()

	provider := &stubToolLoopProvider{
		stubProvider: &stubProvider{},
		completeMultiTurn: func(ctx context.Context, req *llm.Request, toolExecutor func(llm.ToolUse) (string, error), maxSteps int) (*llm.MultiTurnResult, error) {
			_ = ctx
			_ = req
			_ = toolExecutor
			_ = maxSteps
			return &llm.MultiTurnResult{
				FinalResponse:     &llm.Response{Content: []llm.ContentBlock{{Type: "text", Text: "partial"}}},
				TotalLatencyMs:    5,
				TotalInputTokens:  2,
				TotalOutputTokens: 3,
				Steps:             1,
			}, errors.New("boom")
		},
	}

	r := NewRunner(provider, evaluator.NewRegistry(), Config{Trials: 1, PassThreshold: 1, Concurrency: 1})

	p := &prompt.Prompt{Name: "p", Template: "x", Tools: []prompt.Tool{{Name: "git"}}}
	tc := &testcase.TestCase{
		ID:        "c1",
		Input:     map[string]any{},
		ToolMocks: []testcase.ToolMock{{Name: "git", Response: "x"}},
	}

	got, err := r.RunCase(context.Background(), p, tc)
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if got == nil || got.Error == nil {
		t.Fatalf("expected RunResult with Error set, got %#v", got)
	}
	if got.Error.Error() != "boom" {
		t.Fatalf("Error: got %v want %q", got.Error, "boom")
	}
	if len(got.Trials) != 1 || got.Trials[0].Response != "partial" {
		t.Fatalf("Trials: %#v", got.Trials)
	}
	if got.LatencyMs != 5 || got.TokensUsed != 5 {
		t.Fatalf("metrics: LatencyMs=%d TokensUsed=%d", got.LatencyMs, got.TokensUsed)
	}
}
