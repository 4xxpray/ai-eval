package main

import (
	"bytes"
	"os"
	"testing"
)

func TestListCommands_ErrorPaths(t *testing.T) {
	cliIntegrationMu.Lock()
	t.Cleanup(cliIntegrationMu.Unlock)

	dir := t.TempDir()
	oldwd, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd: %v", err)
	}
	if err := os.Chdir(dir); err != nil {
		t.Fatalf("Chdir: %v", err)
	}
	t.Cleanup(func() { _ = os.Chdir(oldwd) })

	var out bytes.Buffer

	prompts := newListPromptsCmd()
	prompts.SetOut(&out)
	if err := prompts.RunE(prompts, nil); err == nil {
		t.Fatalf("expected list prompts error")
	}

	out.Reset()
	tests := newListTestsCmd()
	tests.SetOut(&out)
	if err := tests.RunE(tests, nil); err == nil {
		t.Fatalf("expected list tests error")
	}
}

func TestListCmd_Wiring(t *testing.T) {
	t.Parallel()

	cmd := newListCmd()
	if cmd == nil || len(cmd.Commands()) != 2 {
		t.Fatalf("cmd=%#v", cmd)
	}
	for _, c := range cmd.Commands() {
		if c.Args == nil {
			t.Fatalf("subcmd %q: expected args validator", c.Use)
		}
	}
	if err := cmd.Args(cmd, []string{"unexpected"}); err == nil {
		t.Fatalf("expected NoArgs to reject args")
	}
	if err := cmd.Args(cmd, nil); err != nil {
		t.Fatalf("expected NoArgs to accept nil args: %v", err)
	}
}
