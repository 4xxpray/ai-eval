package ci

import (
	"io"
	"os"
	"path/filepath"
	"testing"
)

func captureStdout(t *testing.T, fn func()) string {
	t.Helper()

	old := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("Pipe: %v", err)
	}
	os.Stdout = w

	fn()
	_ = w.Close()
	os.Stdout = old

	out, err := io.ReadAll(r)
	_ = r.Close()
	if err != nil {
		t.Fatalf("ReadAll: %v", err)
	}
	return string(out)
}

func TestDetectCI(t *testing.T) {
	t.Setenv("GITHUB_ACTIONS", " true ")
	if !DetectCI() {
		t.Fatalf("DetectCI: expected true")
	}

	t.Setenv("GITHUB_ACTIONS", "false")
	if DetectCI() {
		t.Fatalf("DetectCI: expected false")
	}
}

func TestSetOutput_File(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "output.txt")
	if err := os.WriteFile(path, []byte{}, 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	t.Setenv("GITHUB_OUTPUT", path)
	SetOutput(" result ", "value")

	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}

	want := "result<<EOF\nvalue\nEOF\n"
	if string(b) != want {
		t.Fatalf("output: got %q want %q", string(b), want)
	}
}

func TestSetOutput_StdoutEscapes(t *testing.T) {
	t.Setenv("GITHUB_OUTPUT", "")

	out := captureStdout(t, func() {
		SetOutput("result", "line1\nline2%")
	})

	want := "::set-output name=result::line1%0Aline2%25\n"
	if out != want {
		t.Fatalf("stdout: got %q want %q", out, want)
	}
}

func TestSetOutput_EmptyName_NoOp(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "output.txt")
	if err := os.WriteFile(path, []byte("keep"), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	t.Setenv("GITHUB_OUTPUT", path)
	out := captureStdout(t, func() {
		SetOutput("   ", "value")
	})
	if out != "" {
		t.Fatalf("stdout: got %q want empty", out)
	}

	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	if string(b) != "keep" {
		t.Fatalf("file: got %q want %q", string(b), "keep")
	}
}

func TestSetOutput_FileOpenError_Ignored(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("GITHUB_OUTPUT", dir)
	out := captureStdout(t, func() {
		SetOutput("name", "value")
	})
	if out != "" {
		t.Fatalf("stdout: got %q want empty", out)
	}
}

func TestAddAnnotation_DefaultLevel(t *testing.T) {
	out := captureStdout(t, func() {
		AddAnnotation("bad", "", 0, "hi\n")
	})

	want := "::notice::hi%0A\n"
	if out != want {
		t.Fatalf("stdout: got %q want %q", out, want)
	}
}

func TestAddAnnotation_FileLine(t *testing.T) {
	out := captureStdout(t, func() {
		AddAnnotation("warning", "main.go", 12, "bad%")
	})

	want := "::warning file=main.go,line=12::bad%25\n"
	if out != want {
		t.Fatalf("stdout: got %q want %q", out, want)
	}
}

func TestAddAnnotation_FileNoLine(t *testing.T) {
	out := captureStdout(t, func() {
		AddAnnotation("error", "main.go", 0, "bad\r%\n")
	})

	want := "::error file=main.go::bad%0D%25%0A\n"
	if out != want {
		t.Fatalf("stdout: got %q want %q", out, want)
	}
}

func TestStartGroup_EndGroup(t *testing.T) {
	out := captureStdout(t, func() {
		StartGroup("a%\n")
		EndGroup()
	})

	want := "::group::a%25%0A\n::endgroup::\n"
	if out != want {
		t.Fatalf("stdout: got %q want %q", out, want)
	}
}

func TestSetJobSummary(t *testing.T) {
	t.Setenv("GITHUB_STEP_SUMMARY", "")
	if err := SetJobSummary("x"); err != nil {
		t.Fatalf("SetJobSummary (unset): %v", err)
	}

	dir := t.TempDir()
	path := filepath.Join(dir, "nested", "summary.md")
	t.Setenv("GITHUB_STEP_SUMMARY", path)

	if err := SetJobSummary("line1"); err != nil {
		t.Fatalf("SetJobSummary: %v", err)
	}
	if err := SetJobSummary("line2\n"); err != nil {
		t.Fatalf("SetJobSummary: %v", err)
	}

	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	want := "line1\nline2\n"
	if string(b) != want {
		t.Fatalf("summary: got %q want %q", string(b), want)
	}
}

func TestSetJobSummary_MkdirAllError(t *testing.T) {
	dir := t.TempDir()
	notADir := filepath.Join(dir, "notadir")
	if err := os.WriteFile(notADir, []byte("x"), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	t.Setenv("GITHUB_STEP_SUMMARY", filepath.Join(notADir, "summary.md"))
	if err := SetJobSummary("x"); err == nil {
		t.Fatalf("SetJobSummary: expected error")
	}
}
