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
