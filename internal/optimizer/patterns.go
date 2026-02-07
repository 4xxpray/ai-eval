package optimizer

import "strings"

// PatternRule describes a common prompt failure pattern we can diagnose.
type PatternRule struct {
	ID          string `json:"id"`
	Title       string `json:"title"`
	Description string `json:"description"`
}

// CommonPatterns is a curated list of common failure patterns.
var CommonPatterns = []PatternRule{
	{
		ID:          "missing_context",
		Title:       "Missing necessary context",
		Description: "The prompt does not provide required background, definitions, inputs, or constraints, so the model guesses or hallucinates.",
	},
	{
		ID:          "ambiguous_instruction",
		Title:       "Ambiguous instruction",
		Description: "The prompt leaves key decisions underspecified (scope, priority, criteria), leading to inconsistent or incorrect behavior.",
	},
	{
		ID:          "missing_examples",
		Title:       "Missing examples",
		Description: "The prompt does not include minimal examples of the desired behavior or output, making expectations hard to follow.",
	},
	{
		ID:          "conflicting_constraints",
		Title:       "Conflicting constraints",
		Description: "The prompt contains mutually exclusive requirements (e.g., 'be extremely concise' and 'include all details'), causing failure or tradeoff drift.",
	},
	{
		ID:          "output_format_unclear",
		Title:       "Unclear output format",
		Description: "The prompt does not specify a strict output schema/format, or the format is not enforceable, causing parsing/validation failures.",
	},
	{
		ID:          "edge_case_unhandled",
		Title:       "Unhandled edge cases",
		Description: "The prompt does not define behavior for edge cases (empty input, nulls, boundary values, error cases), leading to brittle outputs.",
	},
}

// PatternMatcher provides lightweight heuristic pattern hints.
// These hints are not authoritative; they only help seed LLM analysis.
type PatternMatcher struct{}

// Match returns a list of pattern IDs suggested by simple keyword heuristics.
func (PatternMatcher) Match(promptContent string, evalSummary string) []string {
	text := strings.ToLower(promptContent + "\n" + evalSummary)
	if strings.TrimSpace(text) == "" {
		return nil
	}

	var out []string
	add := func(id string) {
		for _, existing := range out {
			if existing == id {
				return
			}
		}
		out = append(out, id)
	}

	// Output format issues are the most reliably detectable.
	if containsAny(text,
		"invalid json",
		"json schema",
		"missing required field",
		"expected object",
		"expected array",
		"extra data after json",
		"parse",
		"format",
	) {
		add("output_format_unclear")
	}

	if containsAny(text,
		"missing context",
		"not enough information",
		"insufficient",
		"not provided",
		"unknown",
	) {
		add("missing_context")
	}

	if containsAny(text,
		"ambiguous",
		"unclear",
		"underspecified",
		"not specified",
	) {
		add("ambiguous_instruction")
	}

	if containsAny(text,
		"conflict",
		"contradict",
		"mutually exclusive",
	) {
		add("conflicting_constraints")
	}

	if containsAny(text,
		"edge case",
		"corner case",
		"empty",
		"nil",
		"null",
		"overflow",
		"out of range",
	) {
		add("edge_case_unhandled")
	}

	// missing_examples is intentionally not auto-detected; it's too easy to false-positive.

	return out
}

func containsAny(haystack string, needles ...string) bool {
	for _, n := range needles {
		if strings.Contains(haystack, n) {
			return true
		}
	}
	return false
}
