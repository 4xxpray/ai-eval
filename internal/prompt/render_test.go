package prompt

import (
	"strings"
	"testing"
)

func TestRender(t *testing.T) {
	t.Parallel()

	p := &Prompt{
		Name:     "t",
		Template: "Hello {{.name}} ({{.lang}})",
		Variables: []Variable{
			{Name: "name", Required: true},
			{Name: "lang", Required: false, Default: "go"},
		},
	}

	out, err := Render(p, map[string]any{"name": "Alice"})
	if err != nil {
		t.Fatalf("Render: %v", err)
	}
	if out != "Hello Alice (go)" {
		t.Fatalf("out: got %q want %q", out, "Hello Alice (go)")
	}
}

func TestRender_MissingRequired(t *testing.T) {
	t.Parallel()

	p := &Prompt{
		Template: "Hello {{.name}}",
		Variables: []Variable{
			{Name: "name", Required: true},
		},
	}

	_, err := Render(p, map[string]any{})
	if err == nil {
		t.Fatalf("Render: expected error")
	}
	if !strings.Contains(err.Error(), "missing required variable") {
		t.Fatalf("Render: got %v", err)
	}
}

func TestRender_MissingKeyInTemplate(t *testing.T) {
	t.Parallel()

	p := &Prompt{
		Template: "Hello {{.unknown}}",
	}

	_, err := Render(p, nil)
	if err == nil {
		t.Fatalf("Render: expected error")
	}
	if !strings.Contains(err.Error(), "map has no entry for key") {
		t.Fatalf("Render: got %v", err)
	}
}

func TestRender_BadTemplate(t *testing.T) {
	t.Parallel()

	p := &Prompt{
		Template: "{{",
	}

	_, err := Render(p, nil)
	if err == nil {
		t.Fatalf("Render: expected error")
	}
}

func TestRender_NilPrompt(t *testing.T) {
	t.Parallel()

	_, err := Render(nil, nil)
	if err == nil {
		t.Fatalf("Render: expected error")
	}
}

func TestRender_MustacheReplacement(t *testing.T) {
	t.Parallel()

	p := &Prompt{
		Name:     "t",
		Template: "Hello {{NAME}}",
		Variables: []Variable{
			{Name: "NAME", Required: true},
		},
	}

	out, err := Render(p, map[string]any{"NAME": "Alice"})
	if err != nil {
		t.Fatalf("Render: %v", err)
	}
	if out != "Hello Alice" {
		t.Fatalf("out: got %q want %q", out, "Hello Alice")
	}
}

func TestRender_LeavesUnknownMustachePlaceholder(t *testing.T) {
	t.Parallel()

	p := &Prompt{
		Name:     "t",
		Template: "Hello {{MISSING}}",
	}

	out, err := Render(p, nil)
	if err != nil {
		t.Fatalf("Render: %v", err)
	}
	if out != "Hello {{MISSING}}" {
		t.Fatalf("out: got %q want %q", out, "Hello {{MISSING}}")
	}
}

func TestRender_UnmatchedCloseDelimiter(t *testing.T) {
	t.Parallel()

	p := &Prompt{
		Name:     "t",
		Template: "oops }}",
	}

	_, err := Render(p, nil)
	if err == nil {
		t.Fatalf("Render: expected error")
	}
	if !strings.Contains(err.Error(), "unmatched") {
		t.Fatalf("Render: got %v", err)
	}
}

func TestRender_GoTemplateParseError(t *testing.T) {
	t.Parallel()

	p := &Prompt{
		Name:     "t",
		Template: "{{.name",
	}

	_, err := Render(p, map[string]any{"name": "x"})
	if err == nil {
		t.Fatalf("Render: expected error")
	}
	if !strings.Contains(err.Error(), "parse template") {
		t.Fatalf("Render: got %v", err)
	}
}

func TestRender_EmptyVariableNameIgnored(t *testing.T) {
	t.Parallel()

	p := &Prompt{
		Name:     "t",
		Template: "ok",
		Variables: []Variable{
			{Name: "", Required: true},
		},
	}

	out, err := Render(p, nil)
	if err != nil {
		t.Fatalf("Render: %v", err)
	}
	if out != "ok" {
		t.Fatalf("out: got %q want %q", out, "ok")
	}
}

func TestRender_DefaultNotOverrideExisting(t *testing.T) {
	t.Parallel()

	p := &Prompt{
		Name:     "t",
		Template: "Lang {{.lang}}",
		Variables: []Variable{
			{Name: "lang", Required: false, Default: "go"},
		},
	}

	out, err := Render(p, map[string]any{"lang": "python"})
	if err != nil {
		t.Fatalf("Render: %v", err)
	}
	if out != "Lang python" {
		t.Fatalf("out: got %q want %q", out, "Lang python")
	}
}
