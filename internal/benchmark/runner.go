package benchmark

import (
	"context"
	"errors"
	"strings"
	"time"

	"github.com/stellarlinkco/ai-eval/internal/leaderboard"
	"github.com/stellarlinkco/ai-eval/internal/llm"
)

type BenchmarkRunner struct {
	Provider llm.Provider
	Store    *leaderboard.Store
}

type BenchmarkResult struct {
	Model       string
	Dataset     string
	Score       float64
	Accuracy    float64
	TotalTime   time.Duration
	TotalTokens int
	Results     []QuestionResult
}

type QuestionResult struct {
	ID       string
	Category string
	Score    float64
	Passed   bool
	Latency  time.Duration
	Tokens   int
	Error    string
}

func (r *BenchmarkRunner) Run(ctx context.Context, dataset Dataset) (*BenchmarkResult, error) {
	if r == nil {
		return nil, errors.New("benchmark: nil runner")
	}
	if ctx == nil {
		return nil, errors.New("benchmark: nil context")
	}
	if r.Provider == nil {
		return nil, errors.New("benchmark: nil provider")
	}
	if dataset == nil {
		return nil, errors.New("benchmark: nil dataset")
	}

	start := time.Now()

	qs, err := dataset.Load(ctx)
	if err != nil {
		return nil, err
	}
	if len(qs) == 0 {
		return nil, errors.New("benchmark: empty dataset")
	}

	out := &BenchmarkResult{
		Model:   strings.TrimSpace(r.Provider.Name()),
		Dataset: strings.TrimSpace(dataset.Name()),
		Results: make([]QuestionResult, 0, len(qs)),
	}

	var sumScore float64
	totalTokens := 0

	for _, q := range qs {
		if err := ctx.Err(); err != nil {
			out.TotalTime = time.Since(start)
			out.TotalTokens = totalTokens
			out.Score = safeAvg(sumScore, len(out.Results))
			out.Accuracy = out.Score
			return out, err
		}

		prompt := formatPrompt(dataset.Name(), &q)
		req := &llm.Request{
			Messages:    []llm.Message{{Role: "user", Content: prompt}},
			MaxTokens:   1024,
			Temperature: 0,
		}

		res, callErr := r.Provider.CompleteWithTools(ctx, req)

		rr := QuestionResult{
			ID:       strings.TrimSpace(q.ID),
			Category: strings.TrimSpace(q.Category),
		}

		var response string
		if res != nil {
			response = res.TextContent
			rr.Latency = time.Duration(res.LatencyMs) * time.Millisecond
			rr.Tokens = res.InputTokens + res.OutputTokens
			totalTokens += rr.Tokens
		}
		if callErr != nil {
			rr.Error = callErr.Error()
			out.Results = append(out.Results, rr)
			continue
		}

		score, evalErr := dataset.Evaluate(response, q.Answer)
		if evalErr != nil {
			rr.Error = evalErr.Error()
		}
		rr.Score = score
		rr.Passed = score >= 1.0-1e-9

		sumScore += score
		out.Results = append(out.Results, rr)
	}

	out.TotalTime = time.Since(start)
	out.TotalTokens = totalTokens
	out.Score = safeAvg(sumScore, len(out.Results))
	out.Accuracy = out.Score
	return out, nil
}

func safeAvg(sum float64, n int) float64 {
	if n <= 0 {
		return 0
	}
	return sum / float64(n)
}

func formatPrompt(datasetName string, q *Question) string {
	if q == nil {
		return ""
	}

	name := strings.ToLower(strings.TrimSpace(datasetName))
	switch name {
	case "mmlu":
		return formatMCQPrompt(q.Question, q.Choices)
	case "gsm8k":
		return "Solve the following math problem. Reply with only the final numeric answer.\n\n" + strings.TrimSpace(q.Question) + "\n"
	case "humaneval":
		return "Write code to solve the following. Reply with code only.\n\n" + strings.TrimSpace(q.Question) + "\n"
	default:
		if len(q.Choices) > 0 {
			return formatMCQPrompt(q.Question, q.Choices)
		}
		return strings.TrimSpace(q.Question) + "\n"
	}
}

func formatMCQPrompt(question string, choices []string) string {
	var sb strings.Builder
	sb.WriteString("You are taking a multiple-choice test. Choose the best answer.\n\n")
	sb.WriteString(strings.TrimSpace(question))
	sb.WriteString("\n\n")

	for i, c := range choices {
		label := string(rune('A' + i))
		sb.WriteString(label)
		sb.WriteString(". ")
		sb.WriteString(strings.TrimSpace(c))
		sb.WriteByte('\n')
	}

	sb.WriteString("\nReply with just the letter (e.g., A).\n")
	return sb.String()
}
