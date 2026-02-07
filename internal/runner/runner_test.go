package runner

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stellarlinkco/ai-eval/internal/evaluator"
	"github.com/stellarlinkco/ai-eval/internal/llm"
	"github.com/stellarlinkco/ai-eval/internal/prompt"
	"github.com/stellarlinkco/ai-eval/internal/testcase"
)

func TestRunCase_SingleTrial(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Body != nil {
			defer r.Body.Close()
		}
		b, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, "read", http.StatusBadRequest)
			return
		}

		var req map[string]any
		if err := json.Unmarshal(b, &req); err != nil {
			http.Error(w, "json", http.StatusBadRequest)
			return
		}
		msgs, _ := req["messages"].([]any)
		if len(msgs) != 1 {
			http.Error(w, "bad prompt", http.StatusBadRequest)
			return
		}
		m0, _ := msgs[0].(map[string]any)
		m0c, _ := m0["content"].([]any)
		if m0["role"] != "user" || len(m0c) != 1 {
			http.Error(w, "bad prompt", http.StatusBadRequest)
			return
		}
		b0, _ := m0c[0].(map[string]any)
		if b0["type"] != "text" || b0["text"] != "Hello Bob" {
			http.Error(w, "bad prompt", http.StatusBadRequest)
			return
		}
		if req["max_tokens"] == nil || req["max_tokens"] == float64(0) {
			http.Error(w, "missing max_tokens", http.StatusBadRequest)
			return
		}

		w.Header().Set("content-type", "application/json")
		_ = json.NewEncoder(w).Encode(messageResponse(
			"msg_1",
			req["model"].(string),
			"end_turn",
			[]map[string]any{textBlock("ok")},
			1,
			2,
		))
	}))
	t.Cleanup(srv.Close)

	provider := llm.NewClaudeProvider("k", srv.URL+"/v1", "")

	reg := evaluator.NewRegistry()
	reg.Register(evaluator.ExactEvaluator{})

	r := NewRunner(provider, reg, Config{
		Trials:        1,
		PassThreshold: 1,
		Concurrency:   1,
		Timeout:       2 * time.Second,
	})

	p := &prompt.Prompt{Name: "p", Template: "Hello {{.name}}"}
	tc := &testcase.TestCase{
		ID:    "c1",
		Input: map[string]any{"name": "Bob"},
		Expected: testcase.Expected{
			ExactMatch: "ok",
		},
	}

	got, err := r.RunCase(context.Background(), p, tc)
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if got == nil {
		t.Fatalf("RunCase: nil result")
	}
	if got.CaseID != "c1" {
		t.Fatalf("CaseID: got %q want %q", got.CaseID, "c1")
	}
	if !got.Passed {
		t.Fatalf("Passed: got false want true (PassAtK=%v)", got.PassAtK)
	}
	if got.Score != 1.0 {
		t.Fatalf("Score: got %v want %v", got.Score, 1.0)
	}
	if got.PassAtK != 1.0 || got.PassExpK != 1.0 {
		t.Fatalf("pass metrics: got pass@k=%v pass^k=%v want 1.0/1.0", got.PassAtK, got.PassExpK)
	}
	if got.TokensUsed != 3 {
		t.Fatalf("TokensUsed: got %d want %d", got.TokensUsed, 3)
	}
	if len(got.Trials) != 1 {
		t.Fatalf("len(Trials): got %d want %d", len(got.Trials), 1)
	}
	if got.Trials[0].Response != "ok" {
		t.Fatalf("Trials[0].Response: got %q want %q", got.Trials[0].Response, "ok")
	}
	if len(got.Trials[0].Evaluations) != 1 || !got.Trials[0].Evaluations[0].Passed {
		t.Fatalf("Evaluations: got %#v", got.Trials[0].Evaluations)
	}
}

func TestRunCase_MultipleTrials_PassMetrics(t *testing.T) {
	t.Parallel()

	var callNum int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := atomic.AddInt64(&callNum, 1)

		text := "bad"
		if n <= 2 {
			text = "ok"
		}

		w.Header().Set("content-type", "application/json")
		_ = json.NewEncoder(w).Encode(messageResponse(
			"msg_x",
			"m",
			"end_turn",
			[]map[string]any{textBlock(text)},
			1,
			1,
		))
	}))
	t.Cleanup(srv.Close)

	provider := llm.NewClaudeProvider("k", srv.URL+"/v1", "")

	reg := evaluator.NewRegistry()
	reg.Register(evaluator.ExactEvaluator{})

	r := NewRunner(provider, reg, Config{
		Trials:        3,
		PassThreshold: 0.8,
		Concurrency:   1,
	})

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
	if got == nil {
		t.Fatalf("RunCase: nil result")
	}
	if len(got.Trials) != 3 {
		t.Fatalf("len(Trials): got %d want %d", len(got.Trials), 3)
	}
	if !got.Passed {
		t.Fatalf("Passed: got false want true (PassAtK=%v)", got.PassAtK)
	}

	const eps = 1e-3
	wantPassAtK := 1 - mathPow(1-(2.0/3.0), 3)
	wantPassExpK := mathPow(2.0/3.0, 3)

	if diff := abs(got.PassAtK - wantPassAtK); diff > eps {
		t.Fatalf("PassAtK: got %v want %v (diff=%v)", got.PassAtK, wantPassAtK, diff)
	}
	if diff := abs(got.PassExpK - wantPassExpK); diff > eps {
		t.Fatalf("PassExpK: got %v want %v (diff=%v)", got.PassExpK, wantPassExpK, diff)
	}
	if diff := abs(got.Score - (2.0 / 3.0)); diff > eps {
		t.Fatalf("Score: got %v want %v", got.Score, 2.0/3.0)
	}
}

func TestRunSuite_ConcurrencyLimit(t *testing.T) {
	t.Parallel()

	var inFlight int64
	var maxInFlight int64
	started := make(chan struct{}, 16)
	gate := make(chan struct{})

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		cur := atomic.AddInt64(&inFlight, 1)
		for {
			prev := atomic.LoadInt64(&maxInFlight)
			if cur <= prev {
				break
			}
			if atomic.CompareAndSwapInt64(&maxInFlight, prev, cur) {
				break
			}
		}
		started <- struct{}{}
		<-gate
		atomic.AddInt64(&inFlight, -1)

		w.Header().Set("content-type", "application/json")
		_ = json.NewEncoder(w).Encode(messageResponse(
			"msg",
			"m",
			"end_turn",
			[]map[string]any{textBlock("ok")},
			1,
			1,
		))
	}))
	t.Cleanup(srv.Close)

	provider := llm.NewClaudeProvider("k", srv.URL+"/v1", "")
	reg := evaluator.NewRegistry()
	reg.Register(evaluator.ExactEvaluator{})

	r := NewRunner(provider, reg, Config{
		Trials:        1,
		PassThreshold: 1,
		Concurrency:   2,
	})

	p := &prompt.Prompt{Name: "p", Template: "x"}
	suite := &testcase.TestSuite{
		Suite:  "s",
		Prompt: "p",
		Cases: []testcase.TestCase{
			{ID: "c1", Input: map[string]any{}, Expected: testcase.Expected{ExactMatch: "ok"}},
			{ID: "c2", Input: map[string]any{}, Expected: testcase.Expected{ExactMatch: "ok"}},
			{ID: "c3", Input: map[string]any{}, Expected: testcase.Expected{ExactMatch: "ok"}},
			{ID: "c4", Input: map[string]any{}, Expected: testcase.Expected{ExactMatch: "ok"}},
		},
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	t.Cleanup(cancel)

	done := make(chan struct{})
	var res *SuiteResult
	var runErr error
	go func() {
		defer close(done)
		res, runErr = r.RunSuite(ctx, p, suite)
	}()

	select {
	case <-started:
	case <-ctx.Done():
		close(gate)
		t.Fatalf("first request did not start: %v", ctx.Err())
	}
	select {
	case <-started:
	case <-ctx.Done():
		close(gate)
		t.Fatalf("second request did not start (no concurrency?): %v", ctx.Err())
	}

	close(gate)

	select {
	case <-done:
	case <-ctx.Done():
		t.Fatalf("RunSuite timeout: %v", ctx.Err())
	}
	if runErr != nil {
		t.Fatalf("RunSuite: %v", runErr)
	}
	if res == nil {
		t.Fatalf("RunSuite: nil result")
	}
	if atomic.LoadInt64(&maxInFlight) != 2 {
		t.Fatalf("maxInFlight: got %d want %d", atomic.LoadInt64(&maxInFlight), 2)
	}
}

func TestRunCase_MultiTurnWithToolMocks(t *testing.T) {
	t.Parallel()

	var calls int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := atomic.AddInt64(&calls, 1)

		if r.Body != nil {
			defer r.Body.Close()
		}
		b, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, "read", http.StatusBadRequest)
			return
		}

		var req map[string]any
		if err := json.Unmarshal(b, &req); err != nil {
			http.Error(w, "json", http.StatusBadRequest)
			return
		}
		msgs, _ := req["messages"].([]any)

		switch n {
		case 1:
			if len(msgs) != 1 {
				http.Error(w, "messages len", http.StatusBadRequest)
				return
			}
			m0, _ := msgs[0].(map[string]any)
			m0c, _ := m0["content"].([]any)
			if m0["role"] != "user" || len(m0c) != 1 {
				http.Error(w, "bad user message", http.StatusBadRequest)
				return
			}
			b0, _ := m0c[0].(map[string]any)
			if b0["type"] != "text" || b0["text"] != "x" {
				http.Error(w, "bad user message", http.StatusBadRequest)
				return
			}

			w.Header().Set("content-type", "application/json")
			_ = json.NewEncoder(w).Encode(messageResponse(
				"msg_1",
				"m",
				"tool_use",
				[]map[string]any{toolUseBlock("toolu_1", "git", map[string]any{"cmd": "log"})},
				1,
				1,
			))
		case 2:
			if len(msgs) != 3 {
				http.Error(w, "messages len 2", http.StatusBadRequest)
				return
			}
			m2, _ := msgs[2].(map[string]any)
			m2c, _ := m2["content"].([]any)
			if m2["role"] != "user" || len(m2c) != 1 {
				http.Error(w, "tool_result message", http.StatusBadRequest)
				return
			}
			b2, _ := m2c[0].(map[string]any)
			if b2["type"] != "tool_result" || b2["tool_use_id"] != "toolu_1" {
				http.Error(w, "tool_result block", http.StatusBadRequest)
				return
			}
			b2c, _ := b2["content"].([]any)
			if len(b2c) != 1 {
				http.Error(w, "tool_result content", http.StatusBadRequest)
				return
			}
			b2t, _ := b2c[0].(map[string]any)
			if b2t["type"] != "text" || b2t["text"] != "mocked" {
				http.Error(w, "tool_result block", http.StatusBadRequest)
				return
			}

			w.Header().Set("content-type", "application/json")
			_ = json.NewEncoder(w).Encode(messageResponse(
				"msg_2",
				"m",
				"end_turn",
				[]map[string]any{textBlock("final")},
				2,
				2,
			))
		default:
			http.Error(w, "unexpected call", http.StatusBadRequest)
		}
	}))
	t.Cleanup(srv.Close)

	provider := llm.NewClaudeProvider("k", srv.URL+"/v1", "")

	reg := evaluator.NewRegistry()
	reg.Register(evaluator.ExactEvaluator{})

	r := NewRunner(provider, reg, Config{
		Trials:        1,
		PassThreshold: 1,
		Concurrency:   1,
		Timeout:       2 * time.Second,
	})

	p := &prompt.Prompt{Name: "p", Template: "x", Tools: []prompt.Tool{{Name: "git"}}}
	tc := &testcase.TestCase{
		ID:    "c1",
		Input: map[string]any{},
		Expected: testcase.Expected{
			ExactMatch: "final",
		},
		ToolMocks: []testcase.ToolMock{{Name: "git", Response: "mocked"}},
		MaxSteps:  3,
	}

	got, err := r.RunCase(context.Background(), p, tc)
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if got == nil {
		t.Fatalf("RunCase: nil result")
	}
	if !got.Passed {
		t.Fatalf("Passed: got false want true")
	}
	if got.TokensUsed != 6 {
		t.Fatalf("TokensUsed: got %d want %d", got.TokensUsed, 6)
	}
	if atomic.LoadInt64(&calls) != 2 {
		t.Fatalf("calls: got %d want %d", atomic.LoadInt64(&calls), 2)
	}
	if got.Trials[0].Response != "final" {
		t.Fatalf("Trials[0].Response: got %q want %q", got.Trials[0].Response, "final")
	}
	if len(got.Trials[0].ToolCalls) != 1 || got.Trials[0].ToolCalls[0].Name != "git" {
		t.Fatalf("ToolCalls: got %#v", got.Trials[0].ToolCalls)
	}
}

func abs(v float64) float64 {
	if v < 0 {
		return -v
	}
	return v
}

func mathPow(x float64, n int) float64 {
	out := 1.0
	for i := 0; i < n; i++ {
		out *= x
	}
	return out
}

func messageResponse(id, model, stopReason string, content []map[string]any, inputTokens, outputTokens int) map[string]any {
	return map[string]any{
		"id":            id,
		"type":          "message",
		"role":          "assistant",
		"content":       content,
		"model":         model,
		"stop_reason":   stopReason,
		"stop_sequence": "",
		"usage": map[string]any{
			"input_tokens":                inputTokens,
			"output_tokens":               outputTokens,
			"cache_creation":              map[string]any{"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 0},
			"cache_creation_input_tokens": 0,
			"cache_read_input_tokens":     0,
			"server_tool_use":             map[string]any{"web_search_requests": 0},
			"service_tier":                "standard",
		},
	}
}

func textBlock(text string) map[string]any {
	return map[string]any{
		"type": "text",
		"text": text,
	}
}

func toolUseBlock(id, name string, input map[string]any) map[string]any {
	return map[string]any{
		"type":  "tool_use",
		"id":    id,
		"name":  name,
		"input": input,
	}
}
