package testcase

// TestSuite defines a suite of test cases.
type TestSuite struct {
	Suite          string     `yaml:"suite"`
	Prompt         string     `yaml:"prompt"` // Reference to prompt name
	Description    string     `yaml:"description,omitempty"`
	IsSystemPrompt bool       `yaml:"is_system_prompt,omitempty"` // If true, prompt is used as system message
	Cases          []TestCase `yaml:"cases"`
}

// TestCase defines a single evaluation case.
type TestCase struct {
	ID          string            `yaml:"id"`
	Description string            `yaml:"description,omitempty"`
	Input       map[string]any    `yaml:"input"`
	Expected    Expected          `yaml:"expected"`
	Evaluators  []EvaluatorConfig `yaml:"evaluators,omitempty"`
	Trials      int               `yaml:"trials,omitempty"` // Override default trials
	ToolMocks   []ToolMock        `yaml:"tool_mocks,omitempty"`
	MaxSteps    int               `yaml:"max_steps,omitempty"` // Max agent steps, default 5
}

// Expected defines built-in expectation assertions.
type Expected struct {
	ExactMatch  string           `yaml:"exact_match,omitempty"`
	Contains    []string         `yaml:"contains,omitempty"`
	NotContains []string         `yaml:"not_contains,omitempty"`
	Regex       []string         `yaml:"regex,omitempty"`
	JSONSchema  map[string]any   `yaml:"json_schema,omitempty"`
	ToolCalls   []ToolCallExpect `yaml:"tool_calls,omitempty"`
}

// ToolCallExpect describes an expected tool call.
type ToolCallExpect struct {
	Name      string         `yaml:"name"`
	ArgsMatch map[string]any `yaml:"args_match,omitempty"`
	Order     int            `yaml:"order,omitempty"`
	Required  bool           `yaml:"required"`
}

// EvaluatorConfig configures a custom evaluator.
type EvaluatorConfig struct {
	Type           string   `yaml:"type"`                      // exact, contains, regex, json_schema, llm_judge, similarity, factuality, tool_call, faithfulness, relevancy, precision, task_completion, tool_selection, efficiency, hallucination, toxicity, bias
	Criteria       string   `yaml:"criteria,omitempty"`        // llm_judge
	Rubric         []string `yaml:"rubric,omitempty"`          // llm_judge
	ScoreScale     int      `yaml:"score_scale,omitempty"`     // llm_judge
	Reference      string   `yaml:"reference,omitempty"`       // similarity
	GroundTruth    string   `yaml:"ground_truth,omitempty"`    // factuality, hallucination
	ScoreThreshold float64  `yaml:"score_threshold,omitempty"` // Optional override / threshold

	// RAG evaluators
	Context  string `yaml:"context,omitempty"`  // faithfulness, precision
	Question string `yaml:"question,omitempty"` // relevancy, precision

	// Agent evaluators
	Task          string   `yaml:"task,omitempty"`           // task_completion
	CriteriaList  []string `yaml:"criteria_list,omitempty"`  // task_completion
	ExpectedTools []string `yaml:"expected_tools,omitempty"` // tool_selection
	MaxSteps      int      `yaml:"max_steps,omitempty"`      // efficiency
	MaxTokens     int      `yaml:"max_tokens,omitempty"`     // efficiency

	// Safety evaluators
	Categories []string `yaml:"categories,omitempty"` // bias
}

// ToolMock defines a stubbed tool response.
type ToolMock struct {
	Name     string         `yaml:"name"`
	Response string         `yaml:"response"`
	Error    string         `yaml:"error,omitempty"`
	Match    map[string]any `yaml:"match,omitempty"` // Only apply if args match
}
