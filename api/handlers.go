package api

import (
	"context"
	"crypto/rand"
	"database/sql"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/stellarlinkco/ai-eval/internal/evaluator"
	"github.com/stellarlinkco/ai-eval/internal/generator"
	"github.com/stellarlinkco/ai-eval/internal/optimizer"
	"github.com/stellarlinkco/ai-eval/internal/prompt"
	"github.com/stellarlinkco/ai-eval/internal/runner"
	"github.com/stellarlinkco/ai-eval/internal/store"
	"github.com/stellarlinkco/ai-eval/internal/testcase"
	"github.com/gin-gonic/gin"
	"gopkg.in/yaml.v3"
)

const (
	promptsDir = "prompts"
	testsDir   = "tests"
)

type runRequest struct {
	Prompt      string   `json:"prompt"`
	All         bool     `json:"all"`
	Trials      *int     `json:"trials,omitempty"`
	Threshold   *float64 `json:"threshold,omitempty"`
	Concurrency *int     `json:"concurrency,omitempty"`
}

type compareRequest struct {
	Prompt string `json:"prompt"`
	V1     string `json:"v1"`
	V2     string `json:"v2"`
}

type runSummary struct {
	TotalSuites  int   `json:"total_suites"`
	TotalCases   int   `json:"total_cases"`
	PassedCases  int   `json:"passed_cases"`
	FailedCases  int   `json:"failed_cases"`
	TotalLatency int64 `json:"total_latency_ms"`
	TotalTokens  int   `json:"total_tokens"`
}

type suiteRun struct {
	promptName    string
	promptVersion string
	suite         *testcase.TestSuite
	result        *runner.SuiteResult
}

func (s *Server) handleHealth(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status": "ok",
		"time":   time.Now().UTC().Format(time.RFC3339),
	})
}

func (s *Server) handleListPrompts(c *gin.Context) {
	prompts, err := prompt.LoadFromDir(promptsDir)
	if err != nil {
		respondError(c, http.StatusInternalServerError, err)
		return
	}
	prompts = compactPrompts(prompts)

	name := strings.TrimSpace(c.Query("name"))
	if name != "" {
		filtered := prompts[:0]
		for _, p := range prompts {
			if p == nil {
				continue
			}
			if strings.EqualFold(strings.TrimSpace(p.Name), name) {
				filtered = append(filtered, p)
			}
		}
		prompts = filtered
	}

	sort.Slice(prompts, func(i, j int) bool {
		return strings.ToLower(prompts[i].Name) < strings.ToLower(prompts[j].Name)
	})

	c.JSON(http.StatusOK, prompts)
}

func (s *Server) handleGetPrompt(c *gin.Context) {
	name := strings.TrimSpace(c.Param("name"))
	if name == "" {
		respondError(c, http.StatusBadRequest, errors.New("missing prompt name"))
		return
	}

	prompts, err := prompt.LoadFromDir(promptsDir)
	if err != nil {
		respondError(c, http.StatusInternalServerError, err)
		return
	}

	p, err := findPromptByName(prompts, name)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			respondError(c, http.StatusNotFound, fmt.Errorf("prompt %q not found", name))
			return
		}
		respondError(c, http.StatusConflict, err)
		return
	}

	c.JSON(http.StatusOK, p)
}

func (s *Server) handleUpsertPrompt(c *gin.Context) {
	var p prompt.Prompt
	if err := c.ShouldBindJSON(&p); err != nil {
		respondError(c, http.StatusBadRequest, err)
		return
	}

	name := strings.TrimSpace(p.Name)
	if name == "" {
		respondError(c, http.StatusBadRequest, errors.New("missing prompt name"))
		return
	}

	fileName, err := promptFileName(name)
	if err != nil {
		respondError(c, http.StatusBadRequest, err)
		return
	}

	if err := os.MkdirAll(promptsDir, 0o755); err != nil {
		respondError(c, http.StatusInternalServerError, err)
		return
	}

	payload, err := yaml.Marshal(&p)
	if err != nil {
		respondError(c, http.StatusInternalServerError, err)
		return
	}

	path := filepath.Join(promptsDir, fileName)
	if err := os.WriteFile(path, payload, 0o644); err != nil {
		respondError(c, http.StatusInternalServerError, err)
		return
	}

	c.JSON(http.StatusOK, p)
}

func (s *Server) handleDeletePrompt(c *gin.Context) {
	name := strings.TrimSpace(c.Param("name"))
	if name == "" {
		respondError(c, http.StatusBadRequest, errors.New("missing prompt name"))
		return
	}

	fileName, err := promptFileName(name)
	if err != nil {
		respondError(c, http.StatusBadRequest, err)
		return
	}

	path := filepath.Join(promptsDir, fileName)
	if err := os.Remove(path); err != nil {
		if os.IsNotExist(err) {
			respondError(c, http.StatusNotFound, fmt.Errorf("prompt %q not found", name))
			return
		}
		respondError(c, http.StatusInternalServerError, err)
		return
	}

	c.Status(http.StatusNoContent)
}

func (s *Server) handleListTests(c *gin.Context) {
	suites, err := testcase.LoadFromDir(testsDir)
	if err != nil {
		respondError(c, http.StatusInternalServerError, err)
		return
	}
	suites = compactSuites(suites)

	promptFilter := strings.TrimSpace(c.Query("prompt"))
	if promptFilter != "" {
		filtered := suites[:0]
		for _, suite := range suites {
			if suite == nil {
				continue
			}
			if strings.EqualFold(strings.TrimSpace(suite.Prompt), promptFilter) {
				filtered = append(filtered, suite)
			}
		}
		suites = filtered
	}

	sort.Slice(suites, func(i, j int) bool {
		return strings.ToLower(suites[i].Suite) < strings.ToLower(suites[j].Suite)
	})

	c.JSON(http.StatusOK, suites)
}

func (s *Server) handleGetTestSuite(c *gin.Context) {
	name := strings.TrimSpace(c.Param("suite"))
	if name == "" {
		respondError(c, http.StatusBadRequest, errors.New("missing suite name"))
		return
	}

	suites, err := testcase.LoadFromDir(testsDir)
	if err != nil {
		respondError(c, http.StatusInternalServerError, err)
		return
	}

	suite := findSuiteByName(suites, name)
	if suite == nil {
		respondError(c, http.StatusNotFound, fmt.Errorf("suite %q not found", name))
		return
	}

	c.JSON(http.StatusOK, suite)
}

func (s *Server) handleStartRun(c *gin.Context) {
	if s == nil || s.store == nil || s.provider == nil || s.config == nil {
		respondError(c, http.StatusInternalServerError, errors.New("server not initialized"))
		return
	}

	var req runRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		respondError(c, http.StatusBadRequest, err)
		return
	}

	promptName := strings.TrimSpace(req.Prompt)
	switch {
	case req.All && promptName != "":
		respondError(c, http.StatusBadRequest, errors.New("prompt and all are mutually exclusive"))
		return
	case !req.All && promptName == "":
		respondError(c, http.StatusBadRequest, errors.New("specify prompt or all"))
		return
	}

	trials := s.config.Evaluation.Trials
	if req.Trials != nil {
		trials = *req.Trials
	}
	if trials <= 0 {
		respondError(c, http.StatusBadRequest, fmt.Errorf("trials must be > 0 (got %d)", trials))
		return
	}

	threshold := s.config.Evaluation.Threshold
	if req.Threshold != nil {
		threshold = *req.Threshold
	}
	if threshold < 0 || threshold > 1 {
		respondError(c, http.StatusBadRequest, fmt.Errorf("threshold must be between 0 and 1 (got %v)", threshold))
		return
	}

	concurrency := s.config.Evaluation.Concurrency
	if req.Concurrency != nil {
		concurrency = *req.Concurrency
	}
	if concurrency <= 0 {
		concurrency = 1
	}

	prompts, err := prompt.LoadFromDir(promptsDir)
	if err != nil {
		respondError(c, http.StatusInternalServerError, err)
		return
	}
	promptByName, err := indexPrompts(prompts)
	if err != nil {
		respondError(c, http.StatusInternalServerError, err)
		return
	}

	suites, err := testcase.LoadFromDir(testsDir)
	if err != nil {
		respondError(c, http.StatusInternalServerError, err)
		return
	}
	suitesByPrompt, err := indexSuitesByPrompt(suites, promptByName)
	if err != nil {
		respondError(c, http.StatusBadRequest, err)
		return
	}

	var promptNames []string
	if req.All {
		for name := range suitesByPrompt {
			promptNames = append(promptNames, name)
		}
		sort.Strings(promptNames)
	} else {
		if _, ok := promptByName[promptName]; !ok {
			respondError(c, http.StatusNotFound, fmt.Errorf("unknown prompt %q", promptName))
			return
		}
		promptNames = []string{promptName}
	}
	if len(promptNames) == 0 {
		respondError(c, http.StatusBadRequest, errors.New("no test suites found"))
		return
	}

	reg := evaluator.NewRegistry()
	reg.Register(evaluator.ExactEvaluator{})
	reg.Register(evaluator.ContainsEvaluator{})
	reg.Register(evaluator.NotContainsEvaluator{})
	reg.Register(evaluator.RegexEvaluator{})
	reg.Register(evaluator.JSONSchemaEvaluator{})

	r := runner.NewRunner(s.provider, reg, runner.Config{
		Trials:        trials,
		PassThreshold: threshold,
		Concurrency:   concurrency,
		Timeout:       s.config.Evaluation.Timeout,
	})

	ctx := c.Request.Context()
	startedAt := time.Now().UTC()

	var runs []suiteRun
	for _, name := range promptNames {
		p := promptByName[name]
		suites := suitesByPrompt[name]
		if len(suites) == 0 {
			respondError(c, http.StatusBadRequest, fmt.Errorf("no test suites found for prompt %q", name))
			return
		}
		sort.Slice(suites, func(i, j int) bool { return suites[i].Suite < suites[j].Suite })

		for _, suite := range suites {
			res, err := r.RunSuite(ctx, p, suite)
			if err != nil {
				respondError(c, http.StatusInternalServerError, err)
				return
			}
			runs = append(runs, suiteRun{promptName: name, promptVersion: p.Version, suite: suite, result: res})
		}
	}

	finishedAt := time.Now().UTC()
	summary := summarizeRuns(runs)

	runRecord, err := s.saveRun(ctx, runs, summary, startedAt, finishedAt, promptNames, req.All, trials, threshold, concurrency)
	if err != nil {
		respondError(c, http.StatusInternalServerError, err)
		return
	}

	c.JSON(http.StatusCreated, gin.H{
		"run":     runRecord,
		"summary": summary,
	})
}

func (s *Server) handleListRuns(c *gin.Context) {
	if s == nil || s.store == nil {
		respondError(c, http.StatusInternalServerError, errors.New("server not initialized"))
		return
	}

	limit, err := parseLimitParam(c.Query("limit"), 20)
	if err != nil {
		respondError(c, http.StatusBadRequest, err)
		return
	}

	since, err := parseTimeParam(c.Query("since"))
	if err != nil {
		respondError(c, http.StatusBadRequest, err)
		return
	}

	until, err := parseTimeParam(c.Query("until"))
	if err != nil {
		respondError(c, http.StatusBadRequest, err)
		return
	}

	filter := store.RunFilter{
		PromptName:    strings.TrimSpace(c.Query("prompt")),
		PromptVersion: strings.TrimSpace(c.Query("version")),
		Since:         since,
		Until:         until,
		Limit:         limit,
	}

	runs, err := s.store.ListRuns(c.Request.Context(), filter)
	if err != nil {
		respondError(c, http.StatusInternalServerError, err)
		return
	}

	c.JSON(http.StatusOK, runs)
}

func (s *Server) handleGetRun(c *gin.Context) {
	if s == nil || s.store == nil {
		respondError(c, http.StatusInternalServerError, errors.New("server not initialized"))
		return
	}

	id := strings.TrimSpace(c.Param("id"))
	if id == "" {
		respondError(c, http.StatusBadRequest, errors.New("missing run id"))
		return
	}

	run, err := s.store.GetRun(c.Request.Context(), id)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			respondError(c, http.StatusNotFound, fmt.Errorf("run %q not found", id))
			return
		}
		respondError(c, http.StatusInternalServerError, err)
		return
	}

	c.JSON(http.StatusOK, run)
}

func (s *Server) handleGetRunResults(c *gin.Context) {
	if s == nil || s.store == nil {
		respondError(c, http.StatusInternalServerError, errors.New("server not initialized"))
		return
	}

	id := strings.TrimSpace(c.Param("id"))
	if id == "" {
		respondError(c, http.StatusBadRequest, errors.New("missing run id"))
		return
	}

	if _, err := s.store.GetRun(c.Request.Context(), id); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			respondError(c, http.StatusNotFound, fmt.Errorf("run %q not found", id))
			return
		}
		respondError(c, http.StatusInternalServerError, err)
		return
	}

	results, err := s.store.GetSuiteResults(c.Request.Context(), id)
	if err != nil {
		respondError(c, http.StatusInternalServerError, err)
		return
	}

	c.JSON(http.StatusOK, results)
}

func (s *Server) handleGetPromptHistory(c *gin.Context) {
	if s == nil || s.store == nil {
		respondError(c, http.StatusInternalServerError, errors.New("server not initialized"))
		return
	}

	promptName := strings.TrimSpace(c.Param("prompt"))
	if promptName == "" {
		respondError(c, http.StatusBadRequest, errors.New("missing prompt name"))
		return
	}

	limit, err := parseLimitParam(c.Query("limit"), 20)
	if err != nil {
		respondError(c, http.StatusBadRequest, err)
		return
	}

	results, err := s.store.GetPromptHistory(c.Request.Context(), promptName, limit)
	if err != nil {
		respondError(c, http.StatusInternalServerError, err)
		return
	}

	c.JSON(http.StatusOK, results)
}

func (s *Server) handleCompareVersions(c *gin.Context) {
	if s == nil || s.store == nil {
		respondError(c, http.StatusInternalServerError, errors.New("server not initialized"))
		return
	}

	var req compareRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		respondError(c, http.StatusBadRequest, err)
		return
	}

	if strings.TrimSpace(req.Prompt) == "" || strings.TrimSpace(req.V1) == "" || strings.TrimSpace(req.V2) == "" {
		respondError(c, http.StatusBadRequest, errors.New("prompt, v1, and v2 are required"))
		return
	}

	cmp, err := s.store.GetVersionComparison(c.Request.Context(), req.Prompt, req.V1, req.V2)
	if err != nil {
		if strings.Contains(err.Error(), "no runs") {
			respondError(c, http.StatusNotFound, err)
			return
		}
		respondError(c, http.StatusInternalServerError, err)
		return
	}

	c.JSON(http.StatusOK, cmp)
}

type diagnoseRequest struct {
	PromptContent  string `json:"prompt_content" binding:"required"`
	TestsYAML      string `json:"tests_yaml" binding:"required"`
	MaxSuggestions int    `json:"max_suggestions,omitempty"`
}

type diagnoseResponse struct {
	Suites    []evalSummaryResponse     `json:"suites"`
	Diagnosis *optimizer.DiagnoseResult `json:"diagnosis"`
}

func (s *Server) handleDiagnose(c *gin.Context) {
	if s == nil || s.provider == nil || s.config == nil {
		respondError(c, http.StatusInternalServerError, errors.New("server not initialized"))
		return
	}

	var req diagnoseRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		respondError(c, http.StatusBadRequest, err)
		return
	}

	promptContent := strings.TrimSpace(req.PromptContent)
	if promptContent == "" {
		respondError(c, http.StatusBadRequest, errors.New("prompt_content is required"))
		return
	}

	testsYAML := strings.TrimSpace(req.TestsYAML)
	if testsYAML == "" {
		respondError(c, http.StatusBadRequest, errors.New("tests_yaml is required"))
		return
	}

	suites, err := decodeTestSuitesFromYAML(testsYAML)
	if err != nil {
		respondError(c, http.StatusBadRequest, err)
		return
	}
	if len(suites) == 0 {
		respondError(c, http.StatusBadRequest, errors.New("no test suites provided"))
		return
	}

	promptName := strings.TrimSpace(suites[0].Prompt)
	if promptName == "" {
		promptName = "prompt"
	}

	isSystem := suites[0].IsSystemPrompt
	for _, suite := range suites[1:] {
		if suite == nil {
			continue
		}
		if suite.IsSystemPrompt != isSystem {
			respondError(c, http.StatusBadRequest, errors.New("mixed is_system_prompt across suites"))
			return
		}
	}

	p := &prompt.Prompt{
		Name:           promptName,
		Template:       promptContent,
		IsSystemPrompt: isSystem,
	}

	trials := s.config.Evaluation.Trials
	if trials <= 0 {
		trials = 1
	}

	threshold := s.config.Evaluation.Threshold
	if threshold < 0 || threshold > 1 {
		respondError(c, http.StatusBadRequest, fmt.Errorf("threshold must be between 0 and 1 (got %v)", threshold))
		return
	}

	concurrency := s.config.Evaluation.Concurrency
	if concurrency <= 0 {
		concurrency = 1
	}

	reg := evaluator.NewRegistry()
	reg.Register(evaluator.ExactEvaluator{})
	reg.Register(evaluator.ContainsEvaluator{})
	reg.Register(evaluator.NotContainsEvaluator{})
	reg.Register(evaluator.RegexEvaluator{})
	reg.Register(evaluator.JSONSchemaEvaluator{})

	r := runner.NewRunner(s.provider, reg, runner.Config{
		Trials:        trials,
		PassThreshold: threshold,
		Concurrency:   concurrency,
		Timeout:       s.config.Evaluation.Timeout,
	})

	ctx, cancel := context.WithTimeout(c.Request.Context(), 10*time.Minute)
	defer cancel()

	results := make([]*runner.SuiteResult, 0, len(suites))
	for _, suite := range suites {
		if suite == nil {
			continue
		}
		res, err := r.RunSuite(ctx, p, suite)
		if err != nil {
			respondError(c, http.StatusInternalServerError, err)
			return
		}
		results = append(results, res)
	}

	advisor := &optimizer.Advisor{Provider: s.provider}
	diag, err := advisor.Diagnose(ctx, &optimizer.DiagnoseRequest{
		PromptContent:  promptContent,
		EvalResults:    results,
		MaxSuggestions: req.MaxSuggestions,
	})
	if err != nil {
		respondError(c, http.StatusInternalServerError, err)
		return
	}

	resp := &diagnoseResponse{
		Suites:    make([]evalSummaryResponse, 0, len(results)),
		Diagnosis: diag,
	}
	for _, res := range results {
		if res == nil {
			continue
		}
		resp.Suites = append(resp.Suites, evalSummaryResponse{
			PassRate:   res.PassRate,
			AvgScore:   res.AvgScore,
			TotalCases: res.TotalCases,
			Passed:     res.PassedCases,
			Failed:     res.FailedCases,
		})
	}

	c.JSON(http.StatusOK, resp)
}

func decodeTestSuitesFromYAML(raw string) ([]*testcase.TestSuite, error) {
	dec := yaml.NewDecoder(strings.NewReader(raw))
	var out []*testcase.TestSuite

	for {
		s := new(testcase.TestSuite)
		err := dec.Decode(s)
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("invalid tests_yaml: %w", err)
		}
		if err := testcase.Validate(s); err != nil {
			return nil, err
		}
		out = append(out, s)
	}

	return compactSuites(out), nil
}

func respondError(c *gin.Context, status int, err error) {
	if err == nil {
		c.Status(status)
		return
	}
	c.JSON(status, gin.H{"error": err.Error()})
}

func parseLimitParam(raw string, fallback int) (int, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return fallback, nil
	}
	v, err := strconv.Atoi(raw)
	if err != nil {
		return 0, fmt.Errorf("invalid limit %q", raw)
	}
	if v <= 0 {
		return 0, fmt.Errorf("limit must be > 0")
	}
	return v, nil
}

func parseTimeParam(raw string) (time.Time, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return time.Time{}, nil
	}
	layouts := []string{time.RFC3339, "2006-01-02"}
	for _, layout := range layouts {
		if ts, err := time.Parse(layout, raw); err == nil {
			return ts, nil
		}
	}
	return time.Time{}, fmt.Errorf("invalid time %q (expected RFC3339 or YYYY-MM-DD)", raw)
}

func promptFileName(name string) (string, error) {
	name = strings.TrimSpace(name)
	if name == "" {
		return "", errors.New("missing prompt name")
	}
	if strings.HasPrefix(name, ".") {
		return "", errors.New("invalid prompt name")
	}
	if strings.ContainsAny(name, "/\\:*?\"<>|") {
		return "", errors.New("invalid prompt name")
	}
	return name + ".yaml", nil
}

func findPromptByName(prompts []*prompt.Prompt, name string) (*prompt.Prompt, error) {
	name = strings.TrimSpace(name)
	if name == "" {
		return nil, errors.New("missing prompt name")
	}

	var match *prompt.Prompt
	for _, p := range prompts {
		if p == nil {
			continue
		}
		if strings.TrimSpace(p.Name) != name {
			continue
		}
		if match != nil {
			return nil, fmt.Errorf("multiple prompts named %q", name)
		}
		match = p
	}
	if match == nil {
		return nil, sql.ErrNoRows
	}
	return match, nil
}

func findSuiteByName(suites []*testcase.TestSuite, name string) *testcase.TestSuite {
	for _, suite := range suites {
		if suite == nil {
			continue
		}
		if strings.TrimSpace(suite.Suite) == name {
			return suite
		}
	}
	return nil
}

func compactPrompts(prompts []*prompt.Prompt) []*prompt.Prompt {
	if len(prompts) == 0 {
		return prompts
	}
	out := prompts[:0]
	for _, p := range prompts {
		if p != nil {
			out = append(out, p)
		}
	}
	return out
}

func compactSuites(suites []*testcase.TestSuite) []*testcase.TestSuite {
	if len(suites) == 0 {
		return suites
	}
	out := suites[:0]
	for _, suite := range suites {
		if suite != nil {
			out = append(out, suite)
		}
	}
	return out
}

func indexPrompts(prompts []*prompt.Prompt) (map[string]*prompt.Prompt, error) {
	out := make(map[string]*prompt.Prompt, len(prompts))
	for _, p := range prompts {
		if p == nil {
			return nil, fmt.Errorf("run: nil prompt")
		}
		name := strings.TrimSpace(p.Name)
		if name == "" {
			return nil, fmt.Errorf("run: prompt with empty name")
		}
		if _, ok := out[name]; ok {
			return nil, fmt.Errorf("run: duplicate prompt name %q", name)
		}
		out[name] = p
	}
	return out, nil
}

func indexSuitesByPrompt(suites []*testcase.TestSuite, promptByName map[string]*prompt.Prompt) (map[string][]*testcase.TestSuite, error) {
	out := make(map[string][]*testcase.TestSuite)
	for _, s := range suites {
		if s == nil {
			return nil, fmt.Errorf("run: nil test suite")
		}
		promptRef := strings.TrimSpace(s.Prompt)
		if promptRef == "" {
			return nil, fmt.Errorf("run: suite %q: missing prompt reference", s.Suite)
		}
		if _, ok := promptByName[promptRef]; !ok {
			return nil, fmt.Errorf("run: suite %q references unknown prompt %q", s.Suite, promptRef)
		}
		out[promptRef] = append(out[promptRef], s)
	}
	return out, nil
}

func summarizeRuns(runs []suiteRun) runSummary {
	summary := runSummary{TotalSuites: len(runs)}
	for _, r := range runs {
		if r.result == nil {
			continue
		}
		summary.TotalCases += r.result.TotalCases
		summary.PassedCases += r.result.PassedCases
		summary.FailedCases += r.result.FailedCases
		summary.TotalLatency += r.result.TotalLatency
		summary.TotalTokens += r.result.TotalTokens
	}
	return summary
}

func (s *Server) saveRun(ctx context.Context, runs []suiteRun, summary runSummary, startedAt, finishedAt time.Time, promptNames []string, all bool, trials int, threshold float64, concurrency int) (*store.RunRecord, error) {
	if s == nil || s.store == nil {
		return nil, errors.New("server: missing store")
	}
	if ctx == nil {
		ctx = context.Background()
	}

	runID, err := newRunID()
	if err != nil {
		return nil, err
	}

	passedSuites := 0
	failedSuites := 0
	for _, r := range runs {
		if r.result != nil && r.result.FailedCases == 0 {
			passedSuites++
		} else {
			failedSuites++
		}
	}

	runRecord := &store.RunRecord{
		ID:           runID,
		StartedAt:    startedAt,
		FinishedAt:   finishedAt,
		TotalSuites:  summary.TotalSuites,
		PassedSuites: passedSuites,
		FailedSuites: failedSuites,
		Config:       s.buildRunConfig(promptNames, all, trials, threshold, concurrency),
	}

	if err := s.store.SaveRun(ctx, runRecord); err != nil {
		return nil, err
	}

	for i, r := range runs {
		if r.result == nil || r.suite == nil {
			return nil, errors.New("run: missing suite result")
		}

		caseResults := make([]store.CaseRecord, 0, len(r.result.Results))
		for _, rr := range r.result.Results {
			cr := store.CaseRecord{
				CaseID:     rr.CaseID,
				Passed:     rr.Passed,
				Score:      rr.Score,
				PassAtK:    rr.PassAtK,
				PassExpK:   rr.PassExpK,
				LatencyMs:  rr.LatencyMs,
				TokensUsed: rr.TokensUsed,
			}
			if rr.Error != nil {
				cr.Error = rr.Error.Error()
			}
			caseResults = append(caseResults, cr)
		}

		suiteRecord := &store.SuiteRecord{
			ID:            fmt.Sprintf("%s_suite_%d", runID, i+1),
			RunID:         runID,
			PromptName:    r.promptName,
			PromptVersion: r.promptVersion,
			SuiteName:     r.suite.Suite,
			TotalCases:    r.result.TotalCases,
			PassedCases:   r.result.PassedCases,
			FailedCases:   r.result.FailedCases,
			PassRate:      r.result.PassRate,
			AvgScore:      r.result.AvgScore,
			TotalLatency:  r.result.TotalLatency,
			TotalTokens:   r.result.TotalTokens,
			CreatedAt:     finishedAt,
			CaseResults:   caseResults,
		}

		if err := s.store.SaveSuiteResult(ctx, suiteRecord); err != nil {
			return nil, err
		}
	}

	return runRecord, nil
}

func (s *Server) buildRunConfig(promptNames []string, all bool, trials int, threshold float64, concurrency int) map[string]any {
	cfg := map[string]any{
		"trials":      trials,
		"threshold":   threshold,
		"concurrency": concurrency,
		"all":         all,
	}
	if len(promptNames) > 0 {
		cfg["prompts"] = append([]string(nil), promptNames...)
	}
	if s != nil && s.config != nil && s.config.Evaluation.Timeout > 0 {
		cfg["timeout_ms"] = s.config.Evaluation.Timeout.Milliseconds()
	}
	return cfg
}

func newRunID() (string, error) {
	var buf [8]byte
	if _, err := rand.Read(buf[:]); err != nil {
		return "", err
	}
	return fmt.Sprintf("run_%s_%x", time.Now().UTC().Format("20060102T150405Z"), buf), nil
}

type optimizeRequest struct {
	PromptContent string `json:"prompt_content" binding:"required"`
	PromptName    string `json:"prompt_name"`
	NumCases      int    `json:"num_cases"`
}

type optimizeResponse struct {
	Analysis        string               `json:"analysis"`
	Suggestions     []string             `json:"suggestions"`
	EvalResults     *evalSummaryResponse `json:"eval_results"`
	OptimizedPrompt string               `json:"optimized_prompt"`
	Changes         []optimizer.Change   `json:"changes"`
	Summary         string               `json:"optimization_summary"`
}

type evalSummaryResponse struct {
	PassRate   float64 `json:"pass_rate"`
	AvgScore   float64 `json:"avg_score"`
	TotalCases int     `json:"total_cases"`
	Passed     int     `json:"passed"`
	Failed     int     `json:"failed"`
}

func (s *Server) handleOptimize(c *gin.Context) {
	if s == nil || s.provider == nil {
		respondError(c, http.StatusInternalServerError, errors.New("server not initialized"))
		return
	}

	var req optimizeRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		respondError(c, http.StatusBadRequest, err)
		return
	}

	promptContent := strings.TrimSpace(req.PromptContent)
	if promptContent == "" {
		respondError(c, http.StatusBadRequest, errors.New("prompt_content is required"))
		return
	}

	promptName := strings.TrimSpace(req.PromptName)
	if promptName == "" {
		promptName = "prompt"
	}

	numCases := req.NumCases
	if numCases <= 0 {
		numCases = 5
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), 10*time.Minute)
	defer cancel()

	gen := &generator.Generator{Provider: s.provider}
	genResult, err := gen.Generate(ctx, &generator.GenerateRequest{
		PromptContent: promptContent,
		PromptName:    promptName,
		NumCases:      numCases,
	})
	if err != nil {
		respondError(c, http.StatusInternalServerError, fmt.Errorf("failed to generate test cases: %w", err))
		return
	}

	p := &prompt.Prompt{
		Name:     promptName,
		Template: promptContent,
	}

	registry := evaluator.NewRegistry()
	r := runner.NewRunner(s.provider, registry, runner.Config{
		Trials:        1,
		Concurrency:   1,
		PassThreshold: 0.6,
		Timeout:       2 * time.Minute,
	})

	suiteResult, err := r.RunSuite(ctx, p, genResult.Suite)
	if err != nil {
		respondError(c, http.StatusInternalServerError, fmt.Errorf("failed to run evaluation: %w", err))
		return
	}

	response := &optimizeResponse{
		Analysis:    genResult.Analysis,
		Suggestions: genResult.Suggestions,
		EvalResults: &evalSummaryResponse{
			PassRate:   suiteResult.PassRate,
			AvgScore:   suiteResult.AvgScore,
			TotalCases: suiteResult.TotalCases,
			Passed:     suiteResult.PassedCases,
			Failed:     suiteResult.FailedCases,
		},
	}

	if suiteResult.PassRate >= 0.9 && suiteResult.AvgScore >= 0.9 {
		response.OptimizedPrompt = promptContent
		response.Summary = "Prompt is already performing well. No optimization needed."
		c.JSON(http.StatusOK, response)
		return
	}

	opt := &optimizer.Optimizer{Provider: s.provider}
	optResult, err := opt.Optimize(ctx, &optimizer.OptimizeRequest{
		OriginalPrompt: promptContent,
		EvalResults:    suiteResult,
		MaxIterations:  1,
	})
	if err != nil {
		respondError(c, http.StatusInternalServerError, fmt.Errorf("failed to optimize prompt: %w", err))
		return
	}

	response.OptimizedPrompt = optResult.OptimizedPrompt
	response.Changes = optResult.Changes
	response.Summary = optResult.Summary

	c.JSON(http.StatusOK, response)
}
