package prompt

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadFromFile(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	path := filepath.Join(dir, "example.yaml")

	const in = `
name: example
version: "1.0"
description: Example prompt
template: |
  Hello {{.name}}
variables:
  - name: name
    required: true
tools:
  - name: git
    description: "Inspect history"
metadata:
  category: test
`
	if err := os.WriteFile(path, []byte(in), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	p, err := LoadFromFile(path)
	if err != nil {
		t.Fatalf("LoadFromFile: %v", err)
	}

	if p.Name != "example" {
		t.Fatalf("Name: got %q want %q", p.Name, "example")
	}
	if p.Version != "1.0" {
		t.Fatalf("Version: got %q want %q", p.Version, "1.0")
	}
	if p.Description != "Example prompt" {
		t.Fatalf("Description: got %q want %q", p.Description, "Example prompt")
	}
	if len(p.Variables) != 1 || p.Variables[0].Name != "name" || !p.Variables[0].Required {
		t.Fatalf("Variables: got %#v", p.Variables)
	}
	if len(p.Tools) != 1 || p.Tools[0].Name != "git" || p.Tools[0].Description != "Inspect history" {
		t.Fatalf("Tools: got %#v", p.Tools)
	}
	if got := p.Metadata["category"]; got != "test" {
		t.Fatalf("Metadata.category: got %#v want %q", got, "test")
	}
}

func TestLoadFromFile_Missing(t *testing.T) {
	t.Parallel()

	_, err := LoadFromFile(filepath.Join(t.TempDir(), "missing.yaml"))
	if err == nil {
		t.Fatalf("LoadFromFile: expected error")
	}
}

func TestLoadFromDir(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()

	write := func(name, body string) {
		t.Helper()
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			t.Fatalf("WriteFile(%s): %v", name, err)
		}
	}

	write("b.yaml", "name: b\ntemplate: b\n")
	write("a.yml", "name: a\ntemplate: a\n")
	write("ignored.txt", "nope\n")

	ps, err := LoadFromDir(dir)
	if err != nil {
		t.Fatalf("LoadFromDir: %v", err)
	}
	if len(ps) != 2 {
		t.Fatalf("len: got %d want %d", len(ps), 2)
	}
	if ps[0].Name != "a" || ps[1].Name != "b" {
		t.Fatalf("order: got %q, %q", ps[0].Name, ps[1].Name)
	}
}

func TestLoadFromDir_BadYAML(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "bad.yaml"), []byte(":\n"), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	_, err := LoadFromDir(dir)
	if err == nil {
		t.Fatalf("LoadFromDir: expected error")
	}
}
