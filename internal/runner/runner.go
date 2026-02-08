package runner

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"reflect"
	"strings"
	"sync"

	"github.com/stellarlinkco/ai-eval/internal/evaluator"
	"github.com/stellarlinkco/ai-eval/internal/evaluator/agent"
	"github.com/stellarlinkco/ai-eval/internal/evaluator/rag"
	"github.com/stellarlinkco/ai-eval/internal/evaluator/safety"
	"github.com/stellarlinkco/ai-eval/internal/llm"
	"github.com/stellarlinkco/ai-eval/internal/prompt"
	"github.com/stellarlinkco/ai-eval/internal/testcase"
)

// Runner executes test cases with an LLM provider and evaluators.
type Runner struct {
	provider llm.Provider
	registry *evaluator.Registry
	cfg      Config

	sem chan struct{}
}

// NewRunner creates a Runner with defaults and registers LLM evaluators.
func NewRunner(provider llm.Provider, registry *evaluator.Registry, cfg Config) *Runner {
	if cfg.Trials <= 0 {
		cfg.Trials = 1
	}
	if cfg.Concurrency <= 0 {
		cfg.Concurrency = 1
	}
	if cfg.PassThreshold < 0 {
		cfg.PassThreshold = 0
	}
	if cfg.PassThreshold > 1 {
		cfg.PassThreshold = 1
	}

	r := &Runner{
		provider: provider,
		registry: registry,
		cfg:      cfg,
		sem:      make(chan struct{}, cfg.Concurrency),
	}

	r.registerLLMEvaluators()
	return r
}

// RunCase executes a test case and returns aggregated results.
func (r *Runner) RunCase(ctx context.Context, p *prompt.Prompt, tc *testcase.TestCase) (*RunResult, error) {
	if r == nil {
		return nil, errors.New("runner: nil runner")
	}
	if ctx == nil {
		return nil, errors.New("runner: nil context")
	}
	if r.provider == nil {
		return nil, errors.New("runner: nil llm provider")
	}
	if r.registry == nil {
		return nil, errors.New("runner: nil evaluator registry")
	}
	if p == nil {
		return nil, errors.New("runner: nil prompt")
	}
	if tc == nil {
		return nil, errors.New("runner: nil test case")
	}

	if err := r.acquire(ctx); err != nil {
		return nil, err
	}
	defer r.release()

	trials := tc.Trials
	if trials <= 0 {
		trials = r.cfg.Trials
	}
	if trials <= 0 {
		trials = 1
	}

	out := &RunResult{
		CaseID: tc.ID,
		Trials: make([]TrialResult, 0, trials),
	}

	tools := promptTools(p.Tools)
	useMultiTurn := len(tc.ToolMocks) > 0
	maxSteps := tc.MaxSteps
	if maxSteps <= 0 {
		maxSteps = 5
	}
	var toolExecutor func(llm.ToolUse) (string, error)
	if useMultiTurn {
		toolExecutor = toolExecutorFromMocks(tc.ToolMocks)
	}

	var totalScore float64
	passedTrials := 0

trialLoop:
	for i := 0; i < trials; i++ {
		select {
		case <-ctx.Done():
			if out.Error == nil {
				out.Error = ctx.Err()
			}
			break trialLoop
		default:
		}

		trialNum := i + 1
		tr := TrialResult{TrialNum: trialNum}

		func() {
			trialCtx := ctx
			var cancel context.CancelFunc
			if r.cfg.Timeout > 0 {
				trialCtx, cancel = context.WithTimeout(ctx, r.cfg.Timeout)
				defer cancel()
			}

			rendered, err := prompt.Render(p, tc.Input)
			if err != nil {
				if out.Error == nil {
					out.Error = err
				}
				tr.Evaluations = append(tr.Evaluations, evaluator.Result{
					Passed:  false,
					Score:   0,
					Message: err.Error(),
				})
				return
			}

			// Determine user message content
			var userContent string
			var systemContent string

			if p.IsSystemPrompt {
				// For system prompts, use the rendered prompt as system message
				// and user_task from input as the user message
				systemContent = rendered
				if task, ok := tc.Input["user_task"]; ok {
					userContent = fmt.Sprintf("%v", task)
				} else {
					// Fallback: use a generic task request
					userContent = "Please process this request according to your instructions."
				}
			} else {
				// For regular prompts, use rendered as user message
				userContent = rendered
			}

			req := &llm.Request{
				Messages:  []llm.Message{{Role: "user", Content: userContent}},
				MaxTokens: 4096,
				System:    systemContent,
			}
			if len(tools) > 0 {
				req.Tools = tools
			}

			trialSteps := 0
			trialTokensUsed := 0

			if useMultiTurn {
				looper, ok := r.provider.(llm.ToolLoopProvider)
				if !ok {
					err := errors.New("runner: provider does not support tool loops")
					if out.Error == nil {
						out.Error = err
					}
					tr.Evaluations = append(tr.Evaluations, evaluator.Result{
						Passed:  false,
						Score:   0,
						Message: err.Error(),
					})
					return
				}

				res, err := looper.CompleteMultiTurn(trialCtx, req, toolExecutor, maxSteps)
				if res != nil {
					if res.FinalResponse != nil {
						tr.Response = responseText(res.FinalResponse)
					}
					tr.ToolCalls = res.AllToolCalls
					tr.LatencyMs = res.TotalLatencyMs
					trialSteps = res.Steps
					trialTokensUsed = res.TotalInputTokens + res.TotalOutputTokens

					out.LatencyMs += res.TotalLatencyMs
					out.TokensUsed += trialTokensUsed
				}
				if err != nil {
					if out.Error == nil {
						out.Error = err
					}
					tr.Evaluations = append(tr.Evaluations, evaluator.Result{
						Passed:  false,
						Score:   0,
						Message: err.Error(),
					})
					return
				}
			} else {
				res, err := r.provider.CompleteWithTools(trialCtx, req)
				if res != nil {
					tr.Response = res.TextContent
					tr.ToolCalls = res.ToolCalls
					tr.LatencyMs = res.LatencyMs
					trialSteps = 1
					trialTokensUsed = res.InputTokens + res.OutputTokens

					out.LatencyMs += res.LatencyMs
					out.TokensUsed += trialTokensUsed
				}
				if err != nil {
					if out.Error == nil {
						out.Error = err
					}
					tr.Evaluations = append(tr.Evaluations, evaluator.Result{
						Passed:  false,
						Score:   0,
						Message: err.Error(),
					})
					return
				}
			}

			tr.Evaluations, tr.Passed, tr.Score = r.evaluateTrial(trialCtx, tc, rendered, tr.Response, tr.ToolCalls, trialSteps, trialTokensUsed)
		}()

		out.Trials = append(out.Trials, tr)

		totalScore += tr.Score
		if tr.Passed {
			passedTrials++
		}
	}

	if len(out.Trials) > 0 {
		out.Score = totalScore / float64(len(out.Trials))
	}

	passRate := 0.0
	if len(out.Trials) > 0 {
		passRate = float64(passedTrials) / float64(len(out.Trials))
	}
	k := float64(len(out.Trials))
	if k > 0 {
		out.PassAtK = 1 - math.Pow(1-passRate, k)
		out.PassExpK = math.Pow(passRate, k)
	}
	out.Passed = out.PassAtK >= r.cfg.PassThreshold

	return out, nil
}

// RunSuite executes all cases in a suite and aggregates results.
func (r *Runner) RunSuite(ctx context.Context, p *prompt.Prompt, suite *testcase.TestSuite) (*SuiteResult, error) {
	if r == nil {
		return nil, errors.New("runner: nil runner")
	}
	if ctx == nil {
		return nil, errors.New("runner: nil context")
	}
	if p == nil {
		return nil, errors.New("runner: nil prompt")
	}
	if suite == nil {
		return nil, errors.New("runner: nil suite")
	}

	out := &SuiteResult{
		Suite:      suite.Suite,
		TotalCases: len(suite.Cases),
		Results:    make([]RunResult, len(suite.Cases)),
	}

	var wg sync.WaitGroup
caseLoop:
	for i := range suite.Cases {
		select {
		case <-ctx.Done():
			err := ctx.Err()
			for j := i; j < len(suite.Cases); j++ {
				tc := suite.Cases[j]
				out.Results[j] = RunResult{
					Suite:  suite.Suite,
					CaseID: tc.ID,
					Error:  err,
				}
			}
			break caseLoop
		default:
		}

		tc := suite.Cases[i]
		idx := i

		wg.Add(1)
		go func() {
			defer wg.Done()

			select {
			case <-ctx.Done():
				out.Results[idx] = RunResult{
					Suite:  suite.Suite,
					CaseID: tc.ID,
					Error:  ctx.Err(),
				}
				return
			default:
			}

			if ctx.Err() != nil {
				out.Results[idx] = RunResult{
					Suite:  suite.Suite,
					CaseID: tc.ID,
					Error:  ctx.Err(),
				}
				return
			}

			res, err := r.RunCase(ctx, p, &tc)
			if err != nil {
				out.Results[idx] = RunResult{
					Suite:  suite.Suite,
					CaseID: tc.ID,
					Error:  err,
				}
				return
			}
			res.Suite = suite.Suite
			out.Results[idx] = *res
		}()
	}
	wg.Wait()

	var scoreSum float64
	for i := range out.Results {
		rr := out.Results[i]
		if rr.Passed {
			out.PassedCases++
		} else {
			out.FailedCases++
		}
		out.TotalLatency += rr.LatencyMs
		out.TotalTokens += rr.TokensUsed
		scoreSum += rr.Score
	}

	if out.TotalCases > 0 {
		out.PassRate = float64(out.PassedCases) / float64(out.TotalCases)
		out.AvgScore = scoreSum / float64(out.TotalCases)
	}

	return out, nil
}

func (r *Runner) acquire(ctx context.Context) error {
	if r.sem == nil {
		return errors.New("runner: nil semaphore")
	}
	select {
	case r.sem <- struct{}{}:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (r *Runner) release() {
	<-r.sem
}

func promptTools(in []prompt.Tool) []llm.ToolDefinition {
	out := make([]llm.ToolDefinition, 0, len(in))
	for _, t := range in {
		name := strings.TrimSpace(t.Name)
		if name == "" {
			continue
		}
		schema := t.InputSchema
		if schema == nil {
			schema = map[string]any{"type": "object"}
		}
		out = append(out, llm.ToolDefinition{
			Name:        name,
			Description: strings.TrimSpace(t.Description),
			InputSchema: schema,
		})
	}
	return out
}

func responseText(resp *llm.Response) string {
	if resp == nil {
		return ""
	}

	var sb strings.Builder
	for _, b := range resp.Content {
		if b.Type != "text" {
			continue
		}
		sb.WriteString(b.Text)
	}
	return sb.String()
}

func toolExecutorFromMocks(mocks []testcase.ToolMock) func(llm.ToolUse) (string, error) {
	return func(toolUse llm.ToolUse) (string, error) {
		for _, m := range mocks {
			if strings.TrimSpace(m.Name) != toolUse.Name {
				continue
			}
			if len(m.Match) > 0 && !matchArgs(m.Match, toolUse.Input) {
				continue
			}
			if strings.TrimSpace(m.Error) != "" {
				return "", errors.New(m.Error)
			}
			return m.Response, nil
		}
		return "", fmt.Errorf("runner: no tool mock for %q", toolUse.Name)
	}
}

func matchArgs(match map[string]any, actual map[string]any) bool {
	if len(match) == 0 {
		return true
	}
	if actual == nil {
		return false
	}
	for k, mv := range match {
		av, ok := actual[k]
		if !ok {
			return false
		}
		if !matchValue(mv, av) {
			return false
		}
	}
	return true
}

func matchValue(match any, actual any) bool {
	if match == nil {
		return actual == nil
	}

	if mf, ok := number(match); ok {
		af, ok := number(actual)
		if !ok {
			return false
		}
		return mf == af
	}

	switch m := match.(type) {
	case map[string]any:
		a, ok := actual.(map[string]any)
		if !ok {
			return false
		}
		return matchArgs(m, a)
	case []any:
		a, ok := actual.([]any)
		if !ok || len(m) != len(a) {
			return false
		}
		for i := range m {
			if !matchValue(m[i], a[i]) {
				return false
			}
		}
		return true
	default:
		return reflect.DeepEqual(match, actual)
	}
}

func number(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case float32:
		return float64(n), true
	case int:
		return float64(n), true
	case int8:
		return float64(n), true
	case int16:
		return float64(n), true
	case int32:
		return float64(n), true
	case int64:
		return float64(n), true
	case uint:
		return float64(n), true
	case uint8:
		return float64(n), true
	case uint16:
		return float64(n), true
	case uint32:
		return float64(n), true
	case uint64:
		return float64(n), true
	default:
		return 0, false
	}
}

func (r *Runner) registerLLMEvaluators() {
	if r == nil || r.registry == nil {
		return
	}

	if existing, ok := r.registry.Get("llm_judge"); !ok {
		r.registry.Register(&evaluator.LLMJudgeEvaluator{Provider: r.provider})
	} else {
		if v, ok := existing.(*evaluator.LLMJudgeEvaluator); ok && v.Provider == nil {
			v.Provider = r.provider
		}
	}

	if existing, ok := r.registry.Get("similarity"); !ok {
		r.registry.Register(&evaluator.SimilarityEvaluator{Provider: r.provider})
	} else {
		if v, ok := existing.(*evaluator.SimilarityEvaluator); ok && v.Provider == nil {
			v.Provider = r.provider
		}
	}

	if existing, ok := r.registry.Get("factuality"); !ok {
		r.registry.Register(&evaluator.FactualityEvaluator{Provider: r.provider})
	} else {
		if v, ok := existing.(*evaluator.FactualityEvaluator); ok && v.Provider == nil {
			v.Provider = r.provider
		}
	}

	if existing, ok := r.registry.Get("faithfulness"); !ok {
		r.registry.Register(&rag.FaithfulnessEvaluator{Client: r.provider})
	} else {
		if v, ok := existing.(*rag.FaithfulnessEvaluator); ok && v.Client == nil {
			v.Client = r.provider
		}
	}
	if existing, ok := r.registry.Get("relevancy"); !ok {
		r.registry.Register(&rag.RelevancyEvaluator{Client: r.provider})
	} else {
		if v, ok := existing.(*rag.RelevancyEvaluator); ok && v.Client == nil {
			v.Client = r.provider
		}
	}
	if existing, ok := r.registry.Get("precision"); !ok {
		r.registry.Register(&rag.PrecisionEvaluator{Client: r.provider})
	} else {
		if v, ok := existing.(*rag.PrecisionEvaluator); ok && v.Client == nil {
			v.Client = r.provider
		}
	}

	if existing, ok := r.registry.Get("task_completion"); !ok {
		r.registry.Register(&agent.TaskCompletionEvaluator{Client: r.provider})
	} else {
		if v, ok := existing.(*agent.TaskCompletionEvaluator); ok && v.Client == nil {
			v.Client = r.provider
		}
	}
	if existing, ok := r.registry.Get("tool_selection"); !ok {
		r.registry.Register(&agent.ToolSelectionEvaluator{Client: r.provider})
	} else {
		if v, ok := existing.(*agent.ToolSelectionEvaluator); ok && v.Client == nil {
			v.Client = r.provider
		}
	}
	if _, ok := r.registry.Get("efficiency"); !ok {
		r.registry.Register(&agent.EfficiencyEvaluator{})
	}

	if existing, ok := r.registry.Get("hallucination"); !ok {
		r.registry.Register(&safety.HallucinationEvaluator{Client: r.provider})
	} else {
		if v, ok := existing.(*safety.HallucinationEvaluator); ok && v.Client == nil {
			v.Client = r.provider
		}
	}
	if existing, ok := r.registry.Get("toxicity"); !ok {
		r.registry.Register(&safety.ToxicityEvaluator{Client: r.provider})
	} else {
		if v, ok := existing.(*safety.ToxicityEvaluator); ok && v.Client == nil {
			v.Client = r.provider
		}
	}
	if existing, ok := r.registry.Get("bias"); !ok {
		r.registry.Register(&safety.BiasEvaluator{Client: r.provider})
	} else {
		if v, ok := existing.(*safety.BiasEvaluator); ok && v.Client == nil {
			v.Client = r.provider
		}
	}
}

type evalTask struct {
	typ            string
	expected       any
	scoreThreshold float64
}

func (r *Runner) evaluateTrial(ctx context.Context, tc *testcase.TestCase, promptContext string, response string, toolCalls []llm.ToolUse, actualSteps int, actualTokens int) ([]evaluator.Result, bool, float64) {
	if tc == nil {
		return []evaluator.Result{{
			Passed:  false,
			Score:   0,
			Message: "runner: nil test case",
		}}, false, 0
	}

	// Build full response with tool calls for LLM Judge
	fullResponse := buildFullResponse(response, toolCalls)

	tasks, toolCallThreshold := buildEvalTasks(tc, promptContext)
	var results []evaluator.Result

	allPassed := true
	scoreSum := 0.0

	for _, task := range tasks {
		e, ok := r.registry.Get(task.typ)
		if !ok {
			err := fmt.Errorf("runner: missing evaluator %q", task.typ)
			results = append(results, evaluator.Result{
				Passed:  false,
				Score:   0,
				Message: err.Error(),
			})
			allPassed = false
			continue
		}

		// Use full response (with tool calls) for llm_judge, regex, contains
		// so they can check both response text and tool call parameters
		// But not_contains should only check response text (task descriptions might mention other agents)
		evalResponse := response
		switch task.typ {
		case "llm_judge", "regex", "contains":
			evalResponse = fullResponse
		}

		expected := task.expected
		if m, ok := expected.(map[string]any); ok {
			switch task.typ {
			case "tool_selection":
				m["tool_calls"] = toolCalls
			case "efficiency":
				m["actual_steps"] = actualSteps
				m["actual_tokens"] = actualTokens
			}
			expected = m
		}

		res, err := e.Evaluate(ctx, evalResponse, expected)
		if err != nil {
			results = append(results, evaluator.Result{
				Passed:  false,
				Score:   0,
				Message: err.Error(),
			})
			allPassed = false
			continue
		}
		if res == nil {
			results = append(results, evaluator.Result{
				Passed:  false,
				Score:   0,
				Message: "runner: nil evaluator result",
			})
			allPassed = false
			continue
		}

		result := *res
		passed := result.Passed
		if task.scoreThreshold > 0 {
			passed = result.Score >= task.scoreThreshold
			result.Passed = passed
		}
		results = append(results, result)

		scoreSum += result.Score
		if !passed {
			allPassed = false
		}
	}

	if len(tc.Expected.ToolCalls) > 0 {
		tcr := evaluator.ToolCallEvaluator{Expected: tc.Expected.ToolCalls}.Evaluate(toolCalls)
		passed := tcr.Passed
		if toolCallThreshold > 0 {
			passed = tcr.Score >= toolCallThreshold
			tcr.Passed = passed
		}
		results = append(results, tcr)

		scoreSum += tcr.Score
		if !passed {
			allPassed = false
		}
	}

	avgScore := 0.0
	if len(results) > 0 {
		avgScore = scoreSum / float64(len(results))
	}
	return results, allPassed, avgScore
}

func buildEvalTasks(tc *testcase.TestCase, promptContext string) ([]evalTask, float64) {
	seen := make(map[string]struct{}, len(tc.Evaluators))
	tasks := make([]evalTask, 0, len(tc.Evaluators)+5)
	toolCallThreshold := 0.0

	for _, cfg := range tc.Evaluators {
		typ := strings.TrimSpace(cfg.Type)
		if typ == "" {
			continue
		}
		seen[typ] = struct{}{}

		switch typ {
		case "llm_judge":
			tasks = append(tasks, evalTask{
				typ: typ,
				expected: map[string]any{
					"criteria":        cfg.Criteria,
					"rubric":          cfg.Rubric,
					"score_scale":     cfg.ScoreScale,
					"score_threshold": cfg.ScoreThreshold,
					"context":         promptContext,
				},
				scoreThreshold: cfg.ScoreThreshold,
			})
		case "similarity":
			tasks = append(tasks, evalTask{
				typ: typ,
				expected: map[string]any{
					"reference": cfg.Reference,
					"min_score": cfg.ScoreThreshold,
				},
				scoreThreshold: cfg.ScoreThreshold,
			})
		case "factuality":
			tasks = append(tasks, evalTask{
				typ: typ,
				expected: map[string]any{
					"ground_truth": cfg.GroundTruth,
				},
				scoreThreshold: cfg.ScoreThreshold,
			})
		case "tool_call":
			toolCallThreshold = cfg.ScoreThreshold
		case "faithfulness":
			tasks = append(tasks, evalTask{
				typ: typ,
				expected: map[string]any{
					"context":   cfg.Context,
					"threshold": cfg.ScoreThreshold,
				},
				scoreThreshold: cfg.ScoreThreshold,
			})
		case "relevancy":
			tasks = append(tasks, evalTask{
				typ: typ,
				expected: map[string]any{
					"question":  cfg.Question,
					"threshold": cfg.ScoreThreshold,
				},
				scoreThreshold: cfg.ScoreThreshold,
			})
		case "precision":
			tasks = append(tasks, evalTask{
				typ: typ,
				expected: map[string]any{
					"context":  cfg.Context,
					"question": cfg.Question,
				},
				scoreThreshold: cfg.ScoreThreshold,
			})
		case "task_completion":
			tasks = append(tasks, evalTask{
				typ: typ,
				expected: map[string]any{
					"task":     cfg.Task,
					"criteria": cfg.CriteriaList,
				},
				scoreThreshold: cfg.ScoreThreshold,
			})
		case "tool_selection":
			tasks = append(tasks, evalTask{
				typ: typ,
				expected: map[string]any{
					"expected_tools": cfg.ExpectedTools,
				},
				scoreThreshold: cfg.ScoreThreshold,
			})
		case "efficiency":
			maxSteps := cfg.MaxSteps
			if maxSteps <= 0 && tc.MaxSteps > 0 {
				maxSteps = tc.MaxSteps
			}
			if maxSteps <= 0 {
				maxSteps = 5
			}
			maxTokens := cfg.MaxTokens
			if maxTokens < 0 {
				maxTokens = 0
			}
			if maxTokens == 0 {
				maxTokens = 1000
			}
			tasks = append(tasks, evalTask{
				typ: typ,
				expected: map[string]any{
					"max_steps":  maxSteps,
					"max_tokens": maxTokens,
				},
				scoreThreshold: cfg.ScoreThreshold,
			})
		case "hallucination":
			tasks = append(tasks, evalTask{
				typ: typ,
				expected: map[string]any{
					"ground_truth": cfg.GroundTruth,
					"threshold":    cfg.ScoreThreshold,
				},
				scoreThreshold: cfg.ScoreThreshold,
			})
		case "toxicity":
			scoreThreshold := 0.0
			if cfg.ScoreThreshold > 0 {
				scoreThreshold = 1 - cfg.ScoreThreshold
			}
			tasks = append(tasks, evalTask{
				typ: typ,
				expected: map[string]any{
					"threshold": cfg.ScoreThreshold,
				},
				scoreThreshold: scoreThreshold,
			})
		case "bias":
			scoreThreshold := 0.0
			if cfg.ScoreThreshold > 0 {
				scoreThreshold = 1 - cfg.ScoreThreshold
			}
			tasks = append(tasks, evalTask{
				typ: typ,
				expected: map[string]any{
					"categories": cfg.Categories,
					"threshold":  cfg.ScoreThreshold,
				},
				scoreThreshold: scoreThreshold,
			})
		case "exact":
			tasks = append(tasks, evalTask{typ: typ, expected: tc.Expected.ExactMatch, scoreThreshold: cfg.ScoreThreshold})
		case "contains":
			tasks = append(tasks, evalTask{typ: typ, expected: tc.Expected.Contains, scoreThreshold: cfg.ScoreThreshold})
		case "regex":
			tasks = append(tasks, evalTask{typ: typ, expected: tc.Expected.Regex, scoreThreshold: cfg.ScoreThreshold})
		case "json_schema":
			tasks = append(tasks, evalTask{typ: typ, expected: tc.Expected.JSONSchema, scoreThreshold: cfg.ScoreThreshold})
		default:
			tasks = append(tasks, evalTask{typ: typ, expected: tc.Expected, scoreThreshold: cfg.ScoreThreshold})
		}
	}

	if tc.Expected.ExactMatch != "" {
		if _, ok := seen["exact"]; !ok {
			tasks = append(tasks, evalTask{typ: "exact", expected: tc.Expected.ExactMatch})
		}
	}
	if len(tc.Expected.Contains) > 0 {
		if _, ok := seen["contains"]; !ok {
			tasks = append(tasks, evalTask{typ: "contains", expected: tc.Expected.Contains})
		}
	}
	if len(tc.Expected.NotContains) > 0 {
		tasks = append(tasks, evalTask{typ: "not_contains", expected: tc.Expected.NotContains})
	}
	if len(tc.Expected.Regex) > 0 {
		if _, ok := seen["regex"]; !ok {
			tasks = append(tasks, evalTask{typ: "regex", expected: tc.Expected.Regex})
		}
	}
	if len(tc.Expected.JSONSchema) > 0 {
		if _, ok := seen["json_schema"]; !ok {
			tasks = append(tasks, evalTask{typ: "json_schema", expected: tc.Expected.JSONSchema})
		}
	}

	return tasks, toolCallThreshold
}

// buildFullResponse creates a response string that includes both text and tool calls
// for LLM Judge evaluation.
func buildFullResponse(response string, toolCalls []llm.ToolUse) string {
	if len(toolCalls) == 0 {
		return response
	}

	var sb strings.Builder
	sb.WriteString("## Response Text\n")
	sb.WriteString(response)
	sb.WriteString("\n\n## Tool Calls Made\n")

	for i, tc := range toolCalls {
		sb.WriteString(fmt.Sprintf("\n### Tool Call %d\n", i+1))
		sb.WriteString(fmt.Sprintf("- **Tool:** `%s`\n", tc.Name))
		if tc.Input != nil {
			inputJSON, err := json.MarshalIndent(tc.Input, "", "  ")
			if err == nil {
				sb.WriteString(fmt.Sprintf("- **Input:**\n```json\n%s\n```\n", string(inputJSON)))
			}
		}
	}

	return sb.String()
}
