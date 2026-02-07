package generator

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/stellarlinkco/ai-eval/internal/llm"
	"github.com/stellarlinkco/ai-eval/internal/testcase"
)

// Generator generates test cases for a prompt using LLM.
type Generator struct {
	Provider llm.Provider
}

// GenerateRequest contains the prompt to generate test cases for.
type GenerateRequest struct {
	PromptContent string            // The raw prompt content
	PromptName    string            // Optional name for the prompt
	NumCases      int               // Number of test cases to generate (default: 5)
	Variables     map[string]string // Template variables with default values (e.g., {"ENVIRONMENT_CONTEXT": "Working directory: /workspace"})
}

// GenerateResult contains the generated test cases.
type GenerateResult struct {
	Suite          *testcase.TestSuite
	Analysis       string // LLM's analysis of the prompt
	Suggestions    []string
	IsSystemPrompt bool // Whether the prompt is a system prompt (AI instructions) vs user prompt
}

const generatePrompt = `You are a prompt evaluation expert. Analyze the following prompt and generate test cases to evaluate its quality.

## Prompt to Analyze
<prompt>
{{PROMPT}}
</prompt>

{{VARIABLES_SECTION}}

## Your Task
1. Analyze the prompt's purpose, constraints, and expected behavior
2. **IMPORTANT**: Determine if this is a SYSTEM PROMPT (instructions for AI behavior) or a USER PROMPT (direct task)
   - System prompts: Define AI persona, rules, workflows (e.g., "You are a code reviewer", "Task router")
   - User prompts: Direct requests (e.g., "Review this code", "Fix this bug")
3. Generate {{NUM_CASES}} diverse test cases that cover:
   - Happy path scenarios (normal usage)
   - Edge cases (boundary conditions)
   - Error handling (invalid inputs, ambiguous requests)

## Critical: System Prompt Testing
If this is a SYSTEM PROMPT, each test case MUST include a "user_task" field in the input.
The user_task simulates what a real user would ask the AI that has this system prompt.

Example for a system prompt like "You are a code review expert":
{
  "input": {
    "user_task": "Review this Python function:\ndef add(a, b):\n    return a + b"
  },
  "expected": {
    "contains": ["function", "review"]
  }
}

## Output Format
Return a JSON object with this structure:
{
  "analysis": "Brief analysis of the prompt's purpose and key characteristics",
  "is_system_prompt": true/false,
  "suggestions": ["Suggestion 1 for improvement", "Suggestion 2", ...],
  "test_cases": [
    {
      "id": "test_case_id",
      "description": "What this test case validates",
      "input": {
        "user_task": "The simulated user request (REQUIRED for system prompts)",
        "other_key": "other template variables if needed"
      },
      "expected": {
        "contains": ["keywords that should appear in response"],
        "not_contains": ["keywords that should NOT appear"],
        "regex": ["regex patterns to match"]
      },
      "evaluators": [
        {
          "type": "llm_judge",
          "criteria": "Specific criteria for LLM to evaluate the response",
          "score_threshold": 0.6
        }
      ]
    }
  ]
}

IMPORTANT: Return ONLY valid JSON, no markdown code blocks or extra text.`

// Generate creates test cases for the given prompt.
func (g *Generator) Generate(ctx context.Context, req *GenerateRequest) (*GenerateResult, error) {
	if g == nil || g.Provider == nil {
		return nil, errors.New("generator: nil provider")
	}
	if req == nil {
		return nil, errors.New("generator: nil request")
	}
	if strings.TrimSpace(req.PromptContent) == "" {
		return nil, errors.New("generator: empty prompt content")
	}

	numCases := req.NumCases
	if numCases <= 0 {
		numCases = 5
	}

	prompt := strings.ReplaceAll(generatePrompt, "{{PROMPT}}", req.PromptContent)
	prompt = strings.ReplaceAll(prompt, "{{NUM_CASES}}", fmt.Sprintf("%d", numCases))

	// Build variables section if variables are provided
	variablesSection := ""
	if len(req.Variables) > 0 {
		var sb strings.Builder
		sb.WriteString("## Template Variables\n")
		sb.WriteString("This prompt uses the following template variables. Use these default values in test cases:\n")
		for k, v := range req.Variables {
			sb.WriteString(fmt.Sprintf("- `{{%s}}`: %s\n", k, v))
		}
		variablesSection = sb.String()
	}
	prompt = strings.ReplaceAll(prompt, "{{VARIABLES_SECTION}}", variablesSection)

	resp, err := g.Provider.Complete(ctx, &llm.Request{
		Messages:  []llm.Message{{Role: "user", Content: prompt}},
		MaxTokens: 16384,
	})
	if err != nil {
		return nil, fmt.Errorf("generator: %w", err)
	}

	text := strings.TrimSpace(llm.Text(resp))
	text = strings.TrimPrefix(text, "```json")
	text = strings.TrimPrefix(text, "```")
	text = strings.TrimSuffix(text, "```")
	text = strings.TrimSpace(text)

	var parsed struct {
		Analysis       string   `json:"analysis"`
		IsSystemPrompt bool     `json:"is_system_prompt"`
		Suggestions    []string `json:"suggestions"`
		TestCases      []struct {
			ID          string         `json:"id"`
			Description string         `json:"description"`
			Input       map[string]any `json:"input"`
			Expected    struct {
				Contains    []string `json:"contains"`
				NotContains []string `json:"not_contains"`
				Regex       []string `json:"regex"`
			} `json:"expected"`
			Evaluators []struct {
				Type           string  `json:"type"`
				Criteria       string  `json:"criteria"`
				ScoreThreshold float64 `json:"score_threshold"`
			} `json:"evaluators"`
		} `json:"test_cases"`
	}

	if err := json.Unmarshal([]byte(text), &parsed); err != nil {
		return nil, fmt.Errorf("generator: failed to parse response: %w", err)
	}

	promptName := req.PromptName
	if promptName == "" {
		promptName = "prompt"
	}

	suite := &testcase.TestSuite{
		Suite:          promptName + "_tests",
		Prompt:         promptName,
		IsSystemPrompt: parsed.IsSystemPrompt,
		Cases:          make([]testcase.TestCase, 0, len(parsed.TestCases)),
	}

	for _, tc := range parsed.TestCases {
		// Ensure template variables are included in test case input
		input := tc.Input
		if input == nil {
			input = make(map[string]any)
		}
		for k, v := range req.Variables {
			if _, exists := input[k]; !exists {
				input[k] = v
			}
		}

		c := testcase.TestCase{
			ID:          tc.ID,
			Description: tc.Description,
			Input:       input,
			Expected: testcase.Expected{
				Contains:    tc.Expected.Contains,
				NotContains: tc.Expected.NotContains,
				Regex:       tc.Expected.Regex,
			},
			Evaluators: make([]testcase.EvaluatorConfig, 0, len(tc.Evaluators)),
		}
		for _, e := range tc.Evaluators {
			c.Evaluators = append(c.Evaluators, testcase.EvaluatorConfig{
				Type:           e.Type,
				Criteria:       e.Criteria,
				ScoreThreshold: e.ScoreThreshold,
			})
		}
		suite.Cases = append(suite.Cases, c)
	}

	return &GenerateResult{
		Suite:          suite,
		Analysis:       parsed.Analysis,
		Suggestions:    parsed.Suggestions,
		IsSystemPrompt: parsed.IsSystemPrompt,
	}, nil
}
