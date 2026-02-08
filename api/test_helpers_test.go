package api

import (
	"context"

	"github.com/stellarlinkco/ai-eval/internal/llm"
	"github.com/stellarlinkco/ai-eval/internal/store"
)

type fakeStore struct {
	SaveRunFunc              func(ctx context.Context, run *store.RunRecord) error
	SaveSuiteResultFunc      func(ctx context.Context, result *store.SuiteRecord) error
	GetRunFunc               func(ctx context.Context, id string) (*store.RunRecord, error)
	ListRunsFunc             func(ctx context.Context, filter store.RunFilter) ([]*store.RunRecord, error)
	GetSuiteResultsFunc      func(ctx context.Context, runID string) ([]*store.SuiteRecord, error)
	GetPromptHistoryFunc     func(ctx context.Context, promptName string, limit int) ([]*store.SuiteRecord, error)
	GetVersionComparisonFunc func(ctx context.Context, promptName, v1, v2 string) (*store.VersionComparison, error)
	CloseFunc                func() error
}

func (s *fakeStore) SaveRun(ctx context.Context, run *store.RunRecord) error {
	if s.SaveRunFunc != nil {
		return s.SaveRunFunc(ctx, run)
	}
	return nil
}

func (s *fakeStore) SaveSuiteResult(ctx context.Context, result *store.SuiteRecord) error {
	if s.SaveSuiteResultFunc != nil {
		return s.SaveSuiteResultFunc(ctx, result)
	}
	return nil
}

func (s *fakeStore) GetRun(ctx context.Context, id string) (*store.RunRecord, error) {
	if s.GetRunFunc != nil {
		return s.GetRunFunc(ctx, id)
	}
	return nil, nil
}

func (s *fakeStore) ListRuns(ctx context.Context, filter store.RunFilter) ([]*store.RunRecord, error) {
	if s.ListRunsFunc != nil {
		return s.ListRunsFunc(ctx, filter)
	}
	return nil, nil
}

func (s *fakeStore) GetSuiteResults(ctx context.Context, runID string) ([]*store.SuiteRecord, error) {
	if s.GetSuiteResultsFunc != nil {
		return s.GetSuiteResultsFunc(ctx, runID)
	}
	return nil, nil
}

func (s *fakeStore) GetPromptHistory(ctx context.Context, promptName string, limit int) ([]*store.SuiteRecord, error) {
	if s.GetPromptHistoryFunc != nil {
		return s.GetPromptHistoryFunc(ctx, promptName, limit)
	}
	return nil, nil
}

func (s *fakeStore) GetVersionComparison(ctx context.Context, promptName, v1, v2 string) (*store.VersionComparison, error) {
	if s.GetVersionComparisonFunc != nil {
		return s.GetVersionComparisonFunc(ctx, promptName, v1, v2)
	}
	return nil, nil
}

func (s *fakeStore) Close() error {
	if s.CloseFunc != nil {
		return s.CloseFunc()
	}
	return nil
}

type fakeProvider struct {
	CompleteFunc          func(ctx context.Context, req *llm.Request) (*llm.Response, error)
	CompleteWithToolsFunc func(ctx context.Context, req *llm.Request) (*llm.EvalResult, error)
}

func (p *fakeProvider) Name() string { return "fake" }

func (p *fakeProvider) Complete(ctx context.Context, req *llm.Request) (*llm.Response, error) {
	if p.CompleteFunc != nil {
		return p.CompleteFunc(ctx, req)
	}
	return &llm.Response{Content: []llm.ContentBlock{{Type: "text", Text: ""}}}, nil
}

func (p *fakeProvider) CompleteWithTools(ctx context.Context, req *llm.Request) (*llm.EvalResult, error) {
	if p.CompleteWithToolsFunc != nil {
		return p.CompleteWithToolsFunc(ctx, req)
	}
	return &llm.EvalResult{TextContent: ""}, nil
}
