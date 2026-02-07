package optimizer

import (
	"testing"
)

func TestPatternMatcher_EmptyInput(t *testing.T) {
	got := PatternMatcher{}.Match("", "")
	if got != nil {
		t.Errorf("expected nil for empty input, got %v", got)
	}
	got = PatternMatcher{}.Match("   ", "   ")
	if got != nil {
		t.Errorf("expected nil for whitespace input, got %v", got)
	}
}

func TestPatternMatcher_OutputFormat(t *testing.T) {
	got := PatternMatcher{}.Match("prompt", "invalid json in response")
	assertContains(t, got, "output_format_unclear")
}

func TestPatternMatcher_MissingContext(t *testing.T) {
	got := PatternMatcher{}.Match("prompt", "missing context for the task")
	assertContains(t, got, "missing_context")
}

func TestPatternMatcher_AmbiguousInstruction(t *testing.T) {
	got := PatternMatcher{}.Match("the instruction is ambiguous", "")
	assertContains(t, got, "ambiguous_instruction")
}

func TestPatternMatcher_ConflictingConstraints(t *testing.T) {
	got := PatternMatcher{}.Match("", "requirements conflict with each other")
	assertContains(t, got, "conflicting_constraints")
}

func TestPatternMatcher_EdgeCase(t *testing.T) {
	got := PatternMatcher{}.Match("handle edge case", "null pointer")
	assertContains(t, got, "edge_case_unhandled")
}

func TestPatternMatcher_MultiplePatterns(t *testing.T) {
	got := PatternMatcher{}.Match(
		"ambiguous instruction",
		"invalid json and missing context",
	)
	assertContains(t, got, "output_format_unclear")
	assertContains(t, got, "missing_context")
	assertContains(t, got, "ambiguous_instruction")
}

func TestPatternMatcher_NoDuplicates(t *testing.T) {
	got := PatternMatcher{}.Match(
		"invalid json parse format",
		"json schema expected object",
	)
	count := 0
	for _, id := range got {
		if id == "output_format_unclear" {
			count++
		}
	}
	if count != 1 {
		t.Errorf("expected 1 output_format_unclear, got %d", count)
	}
}

func TestPatternMatcher_CaseInsensitive(t *testing.T) {
	got := PatternMatcher{}.Match("INVALID JSON", "AMBIGUOUS")
	assertContains(t, got, "output_format_unclear")
	assertContains(t, got, "ambiguous_instruction")
}

func TestPatternMatcher_NoFalsePositive(t *testing.T) {
	got := PatternMatcher{}.Match(
		"You are a helpful assistant",
		"Score: 1.0, all tests passed",
	)
	if len(got) != 0 {
		t.Errorf("expected no patterns for clean input, got %v", got)
	}
}

func TestCommonPatterns_AllHaveRequiredFields(t *testing.T) {
	for _, p := range CommonPatterns {
		if p.ID == "" {
			t.Error("pattern has empty ID")
		}
		if p.Title == "" {
			t.Errorf("pattern %s has empty Title", p.ID)
		}
		if p.Description == "" {
			t.Errorf("pattern %s has empty Description", p.ID)
		}
	}
}

func TestCommonPatterns_UniqueIDs(t *testing.T) {
	seen := make(map[string]bool)
	for _, p := range CommonPatterns {
		if seen[p.ID] {
			t.Errorf("duplicate pattern ID: %s", p.ID)
		}
		seen[p.ID] = true
	}
}

func TestContainsAny(t *testing.T) {
	if !containsAny("hello world", "world", "foo") {
		t.Error("should match 'world'")
	}
	if containsAny("hello world", "foo", "bar") {
		t.Error("should not match")
	}
	if containsAny("", "foo") {
		t.Error("empty haystack should not match")
	}
}

func assertContains(t *testing.T, got []string, want string) {
	t.Helper()
	for _, s := range got {
		if s == want {
			return
		}
	}
	t.Errorf("expected %q in %v", want, got)
}
