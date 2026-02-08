package benchmark

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestHumanEvalDataset_NameAndDescription(t *testing.T) {
	ds := &HumanEvalDataset{}
	if ds.Name() != "humaneval" {
		t.Fatalf("Name=%q", ds.Name())
	}
	if ds.Description() == "" {
		t.Fatalf("empty description")
	}
}

func TestStripCodeFences(t *testing.T) {
	tests := []struct {
		in   string
		want string
	}{
		{in: "", want: ""},
		{in: "print('x')", want: "print('x')"},
		{in: "```python\nprint('x')\n```", want: "print('x')"},
		{in: "```py\nx\n```", want: "x"},
		{in: "```\n x \n```", want: "x"},
		{in: "```python\nx\n", want: "x"},
	}
	for _, tc := range tests {
		if got := stripCodeFences(tc.in); got != tc.want {
			t.Fatalf("stripCodeFences(%q)=%q want %q", tc.in, got, tc.want)
		}
	}
}

func TestTruncateOutput(t *testing.T) {
	if got := truncateOutput([]byte("  a  "), 0); got != "a" {
		t.Fatalf("got=%q", got)
	}
	if got := truncateOutput([]byte("a"), 1); got != "a" {
		t.Fatalf("got=%q", got)
	}
	if got := truncateOutput([]byte("abcd"), 2); got != "ab..." {
		t.Fatalf("got=%q", got)
	}
}

func fakeDockerScript() string {
	return `#!/bin/sh
set -eu

cmd="${1:-}"
shift || true

case "$cmd" in
  version)
    mode="${FAKE_DOCKER_VERSION:-ok}"
    if [ "$mode" = "sleep" ]; then
      sleep "${FAKE_DOCKER_SLEEP_SECS:-1}"
      echo "25.0.0"
      exit 0
    fi
    if [ "$mode" = "fail" ]; then
      echo "daemon down" >&2
      exit 1
    fi
    echo "25.0.0"
    exit 0
    ;;
  image)
    sub="${1:-}"
    shift || true
    if [ "$sub" != "inspect" ]; then
      echo "unsupported image subcommand" >&2
      exit 2
    fi
    mode="${FAKE_DOCKER_INSPECT:-ok}"
    if [ "$mode" = "sleep" ]; then
      sleep "${FAKE_DOCKER_SLEEP_SECS:-1}"
      echo "sha256:deadbeef"
      exit 0
    fi
    if [ "$mode" = "fail" ]; then
      echo "missing image" >&2
      exit 1
    fi
    echo "sha256:deadbeef"
    exit 0
    ;;
  run)
    mode="${FAKE_DOCKER_RUN:-ok}"
    if [ "$mode" = "sleep" ]; then
      sleep "${FAKE_DOCKER_SLEEP_SECS:-1}"
      exit 0
    fi
    if [ "$mode" = "python_fail" ]; then
      echo "python failed" >&2
      exit 1
    fi
    if [ "$mode" = "docker_fail" ]; then
      echo "run failed" >&2
      exit 125
    fi
    exit 0
    ;;
  rm)
    if [ -n "${FAKE_DOCKER_LOG:-}" ]; then
      echo "rm $*" >> "${FAKE_DOCKER_LOG}" 2>/dev/null || true
    fi
    exit 0
    ;;
  *)
    echo "unknown docker cmd $cmd" >&2
    exit 2
    ;;
esac
`
}

func fakePython3Script() string {
	return `#!/bin/sh
set -eu

last=""
for arg in "$@"; do
  last="$arg"
done
script="$last"

mode="${FAKE_PYTHON_MODE:-ok}"
case "$mode" in
  ok)
    exit 0
    ;;
  fail)
    echo "boom" >&2
    exit 1
    ;;
  sleep)
    sleep "${FAKE_PYTHON_SLEEP_SECS:-1}"
    exit 0
    ;;
  fail_if_contains)
    needle="${FAKE_PYTHON_NEEDLE:-PROMPT_MARKER}"
    if grep -q "$needle" "$script"; then
      echo "prompt present" >&2
      exit 1
    fi
    exit 0
    ;;
  *)
    echo "unknown python mode" >&2
    exit 2
    ;;
esac
`
}

func resetHumanEvalGlobals() {
	humanEvalDockerReadyOnce = sync.Once{}
	humanEvalDockerBin = ""
	humanEvalDockerReadyErr = nil
	humanEvalHostWarnOnce = sync.Once{}
}

func TestHumanEval_DockerReady(t *testing.T) {
	oldTimeout := humanEvalDockerReadyTimeout
	t.Cleanup(func() { humanEvalDockerReadyTimeout = oldTimeout })

	t.Run("DockerNotFound", func(t *testing.T) {
		resetHumanEvalGlobals()
		humanEvalDockerReadyTimeout = 50 * time.Millisecond

		tmpDir := t.TempDir()
		t.Setenv("PATH", tmpDir)

		_, err := humanEvalDockerReady()
		if err == nil {
			t.Fatalf("expected error")
		}
		if !strings.Contains(err.Error(), "docker not found") {
			t.Fatalf("err=%q", err.Error())
		}
	})

	t.Run("DaemonNotReachable", func(t *testing.T) {
		resetHumanEvalGlobals()
		humanEvalDockerReadyTimeout = 200 * time.Millisecond

		tmpDir := t.TempDir()
		_ = writeExecutable(t, tmpDir, "docker", fakeDockerScript())
		t.Setenv("PATH", tmpDir+string(os.PathListSeparator)+os.Getenv("PATH"))
		t.Setenv("FAKE_DOCKER_VERSION", "fail")

		_, err := humanEvalDockerReady()
		if err == nil {
			t.Fatalf("expected error")
		}
		if !strings.Contains(err.Error(), "daemon not reachable") {
			t.Fatalf("err=%q", err.Error())
		}
	})

	t.Run("MissingImage", func(t *testing.T) {
		resetHumanEvalGlobals()
		humanEvalDockerReadyTimeout = 200 * time.Millisecond

		tmpDir := t.TempDir()
		_ = writeExecutable(t, tmpDir, "docker", fakeDockerScript())
		t.Setenv("PATH", tmpDir+string(os.PathListSeparator)+os.Getenv("PATH"))
		t.Setenv("FAKE_DOCKER_INSPECT", "fail")

		_, err := humanEvalDockerReady()
		if err == nil {
			t.Fatalf("expected error")
		}
		if !strings.Contains(err.Error(), "missing image") {
			t.Fatalf("err=%q", err.Error())
		}
	})

	t.Run("VersionTimeout", func(t *testing.T) {
		resetHumanEvalGlobals()
		humanEvalDockerReadyTimeout = 50 * time.Millisecond

		tmpDir := t.TempDir()
		_ = writeExecutable(t, tmpDir, "docker", fakeDockerScript())
		t.Setenv("PATH", tmpDir+string(os.PathListSeparator)+os.Getenv("PATH"))
		t.Setenv("FAKE_DOCKER_VERSION", "sleep")
		t.Setenv("FAKE_DOCKER_SLEEP_SECS", "1")

		_, err := humanEvalDockerReady()
		if err == nil {
			t.Fatalf("expected error")
		}
		if !strings.Contains(err.Error(), "docker version timeout") {
			t.Fatalf("err=%q", err.Error())
		}
	})

	t.Run("InspectTimeout", func(t *testing.T) {
		resetHumanEvalGlobals()
		humanEvalDockerReadyTimeout = 50 * time.Millisecond

		tmpDir := t.TempDir()
		_ = writeExecutable(t, tmpDir, "docker", fakeDockerScript())
		t.Setenv("PATH", tmpDir+string(os.PathListSeparator)+os.Getenv("PATH"))
		t.Setenv("FAKE_DOCKER_INSPECT", "sleep")
		t.Setenv("FAKE_DOCKER_SLEEP_SECS", "1")

		_, err := humanEvalDockerReady()
		if err == nil {
			t.Fatalf("expected error")
		}
		if !strings.Contains(err.Error(), "docker image inspect timeout") {
			t.Fatalf("err=%q", err.Error())
		}
	})

	t.Run("OK", func(t *testing.T) {
		resetHumanEvalGlobals()
		humanEvalDockerReadyTimeout = 200 * time.Millisecond

		tmpDir := t.TempDir()
		_ = writeExecutable(t, tmpDir, "docker", fakeDockerScript())
		t.Setenv("PATH", tmpDir+string(os.PathListSeparator)+os.Getenv("PATH"))

		docker, err := humanEvalDockerReady()
		if err != nil {
			t.Fatalf("humanEvalDockerReady: %v", err)
		}
		if docker == "" {
			t.Fatalf("docker path empty")
		}
		// Second call uses cached result.
		docker2, err := humanEvalDockerReady()
		if err != nil || docker2 != docker {
			t.Fatalf("cached: docker=%q err=%v", docker2, err)
		}
	})
}

func TestRunHumanEvalPythonHost(t *testing.T) {
	t.Run("PythonNotFound", func(t *testing.T) {
		tmpDir := t.TempDir()
		t.Setenv("PATH", tmpDir)
		_, err := runHumanEvalPythonHost("print('x')\n", 0)
		if err == nil {
			t.Fatalf("expected error")
		}
		if !strings.Contains(err.Error(), "python3 not found") {
			t.Fatalf("err=%q", err.Error())
		}
	})

	t.Run("OK", func(t *testing.T) {
		tmpDir := t.TempDir()
		_ = writeExecutable(t, tmpDir, "python3", fakePython3Script())
		t.Setenv("PATH", tmpDir+string(os.PathListSeparator)+os.Getenv("PATH"))

		ok, err := runHumanEvalPythonHost("print('ok')\n", time.Second)
		if err != nil {
			t.Fatalf("runHumanEvalPythonHost: %v", err)
		}
		if !ok {
			t.Fatalf("ok=false")
		}
	})

	t.Run("WriteProgramError", func(t *testing.T) {
		tmpDir := t.TempDir()
		badTmp := filepath.Join(tmpDir, "notadir")
		if err := os.WriteFile(badTmp, []byte("x"), 0o644); err != nil {
			t.Fatalf("write: %v", err)
		}
		_ = writeExecutable(t, tmpDir, "python3", fakePython3Script())
		t.Setenv("PATH", tmpDir+string(os.PathListSeparator)+os.Getenv("PATH"))
		t.Setenv("TMPDIR", badTmp)

		ok, err := runHumanEvalPythonHost("print('ok')\n", time.Second)
		if err == nil || ok {
			t.Fatalf("ok=%v err=%v", ok, err)
		}
		if !strings.Contains(err.Error(), "create temp dir") {
			t.Fatalf("err=%q", err.Error())
		}
	})

	t.Run("Failed", func(t *testing.T) {
		tmpDir := t.TempDir()
		_ = writeExecutable(t, tmpDir, "python3", fakePython3Script())
		t.Setenv("PATH", tmpDir+string(os.PathListSeparator)+os.Getenv("PATH"))
		t.Setenv("FAKE_PYTHON_MODE", "fail")

		ok, err := runHumanEvalPythonHost("print('x')\n", time.Second)
		if err == nil || ok {
			t.Fatalf("ok=%v err=%v", ok, err)
		}
		if !strings.Contains(err.Error(), "python failed") {
			t.Fatalf("err=%q", err.Error())
		}
		if !strings.Contains(err.Error(), "boom") {
			t.Fatalf("err=%q", err.Error())
		}
	})

	t.Run("Timeout", func(t *testing.T) {
		tmpDir := t.TempDir()
		_ = writeExecutable(t, tmpDir, "python3", fakePython3Script())
		t.Setenv("PATH", tmpDir+string(os.PathListSeparator)+os.Getenv("PATH"))
		t.Setenv("FAKE_PYTHON_MODE", "sleep")
		t.Setenv("FAKE_PYTHON_SLEEP_SECS", "1")

		ok, err := runHumanEvalPythonHost("print('x')\n", 50*time.Millisecond)
		if err == nil || ok {
			t.Fatalf("ok=%v err=%v", ok, err)
		}
		if !strings.Contains(err.Error(), "python timeout") {
			t.Fatalf("err=%q", err.Error())
		}
	})
}

func TestRunHumanEvalPythonDocker(t *testing.T) {
	oldTimeout := humanEvalDockerReadyTimeout
	t.Cleanup(func() { humanEvalDockerReadyTimeout = oldTimeout })

	t.Run("ReadyError", func(t *testing.T) {
		resetHumanEvalGlobals()
		humanEvalDockerReadyTimeout = 50 * time.Millisecond

		tmpDir := t.TempDir()
		t.Setenv("PATH", tmpDir)

		ok, err := runHumanEvalPythonDocker("print('x')\n", time.Second)
		if err == nil || ok {
			t.Fatalf("ok=%v err=%v", ok, err)
		}
	})

	t.Run("OK", func(t *testing.T) {
		resetHumanEvalGlobals()
		humanEvalDockerReadyTimeout = 200 * time.Millisecond

		tmpDir := t.TempDir()
		_ = writeExecutable(t, tmpDir, "docker", fakeDockerScript())
		t.Setenv("PATH", tmpDir+string(os.PathListSeparator)+os.Getenv("PATH"))

		ok, err := runHumanEvalPythonDocker("print('ok')\n", time.Second)
		if err != nil {
			t.Fatalf("runHumanEvalPythonDocker: %v", err)
		}
		if !ok {
			t.Fatalf("ok=false")
		}
	})

	t.Run("WriteProgramError", func(t *testing.T) {
		resetHumanEvalGlobals()
		humanEvalDockerReadyTimeout = 200 * time.Millisecond

		tmpDir := t.TempDir()
		badTmp := filepath.Join(tmpDir, "notadir")
		if err := os.WriteFile(badTmp, []byte("x"), 0o644); err != nil {
			t.Fatalf("write: %v", err)
		}
		_ = writeExecutable(t, tmpDir, "docker", fakeDockerScript())
		t.Setenv("PATH", tmpDir+string(os.PathListSeparator)+os.Getenv("PATH"))
		t.Setenv("TMPDIR", badTmp)

		ok, err := runHumanEvalPythonDocker("print('x')\n", time.Second)
		if err == nil || ok {
			t.Fatalf("ok=%v err=%v", ok, err)
		}
		if !strings.Contains(err.Error(), "create temp dir") {
			t.Fatalf("err=%q", err.Error())
		}
	})

	t.Run("PythonFailed", func(t *testing.T) {
		resetHumanEvalGlobals()
		humanEvalDockerReadyTimeout = 200 * time.Millisecond

		tmpDir := t.TempDir()
		_ = writeExecutable(t, tmpDir, "docker", fakeDockerScript())
		t.Setenv("PATH", tmpDir+string(os.PathListSeparator)+os.Getenv("PATH"))
		t.Setenv("FAKE_DOCKER_RUN", "python_fail")

		ok, err := runHumanEvalPythonDocker("print('x')\n", time.Second)
		if err == nil || ok {
			t.Fatalf("ok=%v err=%v", ok, err)
		}
		if !strings.Contains(err.Error(), "python failed") {
			t.Fatalf("err=%q", err.Error())
		}
	})

	t.Run("DockerRunFailed", func(t *testing.T) {
		resetHumanEvalGlobals()
		humanEvalDockerReadyTimeout = 200 * time.Millisecond

		tmpDir := t.TempDir()
		_ = writeExecutable(t, tmpDir, "docker", fakeDockerScript())
		t.Setenv("PATH", tmpDir+string(os.PathListSeparator)+os.Getenv("PATH"))
		t.Setenv("FAKE_DOCKER_RUN", "docker_fail")

		ok, err := runHumanEvalPythonDocker("print('x')\n", time.Second)
		if err == nil || ok {
			t.Fatalf("ok=%v err=%v", ok, err)
		}
		if !strings.Contains(err.Error(), "docker run failed") {
			t.Fatalf("err=%q", err.Error())
		}
	})

	t.Run("Timeout", func(t *testing.T) {
		resetHumanEvalGlobals()
		humanEvalDockerReadyTimeout = 200 * time.Millisecond

		tmpDir := t.TempDir()
		logPath := filepath.Join(tmpDir, "docker.log")
		_ = writeExecutable(t, tmpDir, "docker", fakeDockerScript())
		t.Setenv("PATH", tmpDir+string(os.PathListSeparator)+os.Getenv("PATH"))
		t.Setenv("FAKE_DOCKER_RUN", "sleep")
		t.Setenv("FAKE_DOCKER_SLEEP_SECS", "1")
		t.Setenv("FAKE_DOCKER_LOG", logPath)

		ok, err := runHumanEvalPythonDocker("print('x')\n", 50*time.Millisecond)
		if err == nil || ok {
			t.Fatalf("ok=%v err=%v", ok, err)
		}
		if !strings.Contains(err.Error(), "python timeout") {
			t.Fatalf("err=%q", err.Error())
		}
		b, readErr := os.ReadFile(logPath)
		if readErr != nil {
			t.Fatalf("read log: %v", readErr)
		}
		if !strings.Contains(string(b), "rm -f") {
			t.Fatalf("log=%q", string(b))
		}
	})
}

func TestRunHumanEvalPython_Switch(t *testing.T) {
	t.Run("Disabled", func(t *testing.T) {
		t.Setenv(sandboxModeEnv, sandboxModeDisabled)
		ok, err := runHumanEvalPython("print('x')\n", time.Second)
		if err == nil || ok {
			t.Fatalf("ok=%v err=%v", ok, err)
		}
	})

	t.Run("Unknown", func(t *testing.T) {
		t.Setenv(sandboxModeEnv, "wat")
		ok, err := runHumanEvalPython("print('x')\n", time.Second)
		if err == nil || ok {
			t.Fatalf("ok=%v err=%v", ok, err)
		}
		if !strings.Contains(err.Error(), "unknown") {
			t.Fatalf("err=%q", err.Error())
		}
	})

	t.Run("Host", func(t *testing.T) {
		resetHumanEvalGlobals()
		tmpDir := t.TempDir()
		_ = writeExecutable(t, tmpDir, "python3", fakePython3Script())
		t.Setenv("PATH", tmpDir+string(os.PathListSeparator)+os.Getenv("PATH"))
		t.Setenv(sandboxModeEnv, sandboxModeHost)

		ok, err := runHumanEvalPython("print('ok')\n", time.Second)
		if err != nil || !ok {
			t.Fatalf("ok=%v err=%v", ok, err)
		}
	})

	t.Run("DockerDefault", func(t *testing.T) {
		oldTimeout := humanEvalDockerReadyTimeout
		t.Cleanup(func() { humanEvalDockerReadyTimeout = oldTimeout })

		resetHumanEvalGlobals()
		humanEvalDockerReadyTimeout = 200 * time.Millisecond

		tmpDir := t.TempDir()
		_ = writeExecutable(t, tmpDir, "docker", fakeDockerScript())
		t.Setenv("PATH", tmpDir+string(os.PathListSeparator)+os.Getenv("PATH"))
		t.Setenv(sandboxModeEnv, "")

		ok, err := runHumanEvalPython("print('ok')\n", time.Second)
		if err != nil || !ok {
			t.Fatalf("ok=%v err=%v", ok, err)
		}
	})
}

func TestHumanEvalDataset_Load(t *testing.T) {
	ds := &HumanEvalDataset{SampleSize: 1}

	{
		_, err := ds.Load(nil)
		if err == nil {
			t.Fatalf("expected error")
		}
	}

	t.Run("DefaultPathWhenEnvEmpty", func(t *testing.T) {
		cwd, err := os.Getwd()
		if err != nil {
			t.Fatalf("Getwd: %v", err)
		}
		tmpDir := t.TempDir()
		t.Cleanup(func() { _ = os.Chdir(cwd) })
		if err := os.Chdir(tmpDir); err != nil {
			t.Fatalf("Chdir: %v", err)
		}
		if err := os.MkdirAll(filepath.Dir(defaultHumanEvalPath), 0o755); err != nil {
			t.Fatalf("MkdirAll: %v", err)
		}
		writeJSONLFile(t, defaultHumanEvalPath, []any{
			humanEvalRow{Prompt: "P", Test: "T"},
		})
		t.Setenv("AI_EVAL_HUMANEVAL_PATH", "")

		out, err := ds.Load(context.Background())
		if err != nil {
			t.Fatalf("Load: %v", err)
		}
		if len(out) != 1 || out[0].Question != "P" {
			t.Fatalf("out=%#v", out)
		}
	})

	t.Run("MissingFileDefaultSample", func(t *testing.T) {
		t.Setenv("AI_EVAL_HUMANEVAL_PATH", filepath.Join(t.TempDir(), "missing.jsonl"))
		out, err := ds.Load(context.Background())
		if err != nil {
			t.Fatalf("Load: %v", err)
		}
		if len(out) != 1 || out[0].ID != "humaneval-sample-1" {
			t.Fatalf("out=%#v", out)
		}
	})

	t.Run("FromFileSkipsAndDefaults", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, "humaneval.jsonl")
		writeJSONLFile(t, path, []any{
			humanEvalRow{TaskID: "t0", Prompt: " P0 ", Test: " T0 ", EntryPoint: " f "},
			humanEvalRow{Prompt: " ", Test: "T1"},
		})
		t.Setenv("AI_EVAL_HUMANEVAL_PATH", path)

		out, err := ds.Load(context.Background())
		if err != nil {
			t.Fatalf("Load: %v", err)
		}
		if len(out) != 1 {
			t.Fatalf("out=%#v", out)
		}
		if out[0].ID != "t0" || out[0].Question != "P0" {
			t.Fatalf("q=%#v", out[0])
		}
		exp, ok := out[0].Answer.(humanEvalExpected)
		if !ok {
			t.Fatalf("answer=%T", out[0].Answer)
		}
		if exp.EntryPoint != "f" {
			t.Fatalf("exp=%#v", exp)
		}
	})

	t.Run("IDFallback", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, "humaneval.jsonl")
		writeJSONLFile(t, path, []any{
			humanEvalRow{Prompt: "P", Test: "T"},
		})
		t.Setenv("AI_EVAL_HUMANEVAL_PATH", path)

		out, err := (&HumanEvalDataset{}).Load(context.Background())
		if err != nil {
			t.Fatalf("Load: %v", err)
		}
		if len(out) != 1 || out[0].ID != "humaneval-1" {
			t.Fatalf("out=%#v", out)
		}
	})

	t.Run("AllSkippedDefaultSample", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, "humaneval.jsonl")
		writeJSONLFile(t, path, []any{
			humanEvalRow{Prompt: " ", Test: "x"},
		})
		t.Setenv("AI_EVAL_HUMANEVAL_PATH", path)

		out, err := ds.Load(context.Background())
		if err != nil {
			t.Fatalf("Load: %v", err)
		}
		if len(out) != 1 || out[0].ID != "humaneval-sample-1" {
			t.Fatalf("out=%#v", out)
		}
	})

	t.Run("ContextErrorAfterScan", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, "humaneval.jsonl")
		writeJSONLFile(t, path, []any{
			humanEvalRow{Prompt: "P", Test: "T"},
		})
		t.Setenv("AI_EVAL_HUMANEVAL_PATH", path)

		ctx := &errAfterNContext{
			Context: context.Background(),
			okCalls: 1,
			err:     context.Canceled,
		}
		_, err := ds.Load(ctx)
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("err=%v", err)
		}
	})

	t.Run("ErrorWrap", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, "humaneval.jsonl")
		if err := os.WriteFile(path, []byte("{\n"), 0o644); err != nil {
			t.Fatalf("write: %v", err)
		}
		t.Setenv("AI_EVAL_HUMANEVAL_PATH", path)

		_, err := ds.Load(context.Background())
		if err == nil {
			t.Fatalf("expected error")
		}
		if !strings.Contains(err.Error(), "humaneval: load") || !strings.Contains(err.Error(), "parse jsonl") {
			t.Fatalf("err=%q", err.Error())
		}
	})
}

func TestHumanEvalDataset_Evaluate(t *testing.T) {
	ds := &HumanEvalDataset{}

	t.Run("CodeExecDisabled", func(t *testing.T) {
		_, err := ds.Evaluate("x", humanEvalExpected{})
		if err == nil {
			t.Fatalf("expected error")
		}
		if !strings.Contains(err.Error(), "code execution disabled") {
			t.Fatalf("err=%q", err.Error())
		}
	})

	t.Run("UnsupportedExpectedType", func(t *testing.T) {
		t.Setenv(codeExecEnv, "1")
		_, err := ds.Evaluate("x", 123)
		if err == nil {
			t.Fatalf("expected error")
		}
	})

	t.Run("EmptyModelOutput", func(t *testing.T) {
		t.Setenv(codeExecEnv, "1")
		_, err := ds.Evaluate(" \n ", humanEvalExpected{Prompt: "p", Test: "t"})
		if err == nil {
			t.Fatalf("expected error")
		}
		if !strings.Contains(err.Error(), "empty model output") {
			t.Fatalf("err=%q", err.Error())
		}
	})

	t.Run("SandboxDisabled", func(t *testing.T) {
		t.Setenv(codeExecEnv, "1")
		t.Setenv(sandboxModeEnv, sandboxModeDisabled)
		_, err := ds.Evaluate("print('x')\n", humanEvalExpected{Prompt: "p", Test: "t"})
		if err == nil {
			t.Fatalf("expected error")
		}
		if !strings.Contains(err.Error(), "code execution disabled") {
			t.Fatalf("err=%q", err.Error())
		}
	})

	t.Run("OK_Host", func(t *testing.T) {
		resetHumanEvalGlobals()
		tmpDir := t.TempDir()
		_ = writeExecutable(t, tmpDir, "python3", fakePython3Script())
		t.Setenv("PATH", tmpDir+string(os.PathListSeparator)+os.Getenv("PATH"))
		t.Setenv(codeExecEnv, "1")
		t.Setenv(sandboxModeEnv, sandboxModeHost)

		score, err := ds.Evaluate("```python\nprint('ok')\n```", humanEvalExpected{
			Prompt: "PROMPT_MARKER\n",
			Test:   "print('test')\n",
		})
		if err != nil {
			t.Fatalf("Evaluate: %v", err)
		}
		if score != 1 {
			t.Fatalf("score=%v", score)
		}
	})

	t.Run("FirstFailSecondOK", func(t *testing.T) {
		resetHumanEvalGlobals()
		tmpDir := t.TempDir()
		_ = writeExecutable(t, tmpDir, "python3", fakePython3Script())
		t.Setenv("PATH", tmpDir+string(os.PathListSeparator)+os.Getenv("PATH"))
		t.Setenv(codeExecEnv, "1")
		t.Setenv(sandboxModeEnv, sandboxModeHost)
		t.Setenv("FAKE_PYTHON_MODE", "fail_if_contains")
		t.Setenv("FAKE_PYTHON_NEEDLE", "PROMPT_MARKER")

		score, err := ds.Evaluate("print('ok')\n", humanEvalExpected{
			Prompt: "PROMPT_MARKER\n",
			Test:   "print('test')\n",
		})
		if err != nil {
			t.Fatalf("Evaluate: %v", err)
		}
		if score != 1 {
			t.Fatalf("score=%v", score)
		}
	})

	t.Run("SecondAttemptErrorWins", func(t *testing.T) {
		resetHumanEvalGlobals()
		tmpDir := t.TempDir()
		_ = writeExecutable(t, tmpDir, "python3", fakePython3Script())
		t.Setenv("PATH", tmpDir+string(os.PathListSeparator)+os.Getenv("PATH"))
		t.Setenv(codeExecEnv, "1")
		t.Setenv(sandboxModeEnv, sandboxModeHost)
		t.Setenv("FAKE_PYTHON_MODE", "fail")

		_, err := ds.Evaluate("print('x')\n", &humanEvalExpected{
			Prompt: "p\n",
			Test:   "t\n",
		})
		if err == nil {
			t.Fatalf("expected error")
		}
		if !strings.Contains(err.Error(), "python failed") {
			t.Fatalf("err=%q", err.Error())
		}
	})

	t.Run("PromptEmpty_FirstAttemptErrorReturned", func(t *testing.T) {
		resetHumanEvalGlobals()
		tmpDir := t.TempDir()
		_ = writeExecutable(t, tmpDir, "python3", fakePython3Script())
		t.Setenv("PATH", tmpDir+string(os.PathListSeparator)+os.Getenv("PATH"))
		t.Setenv(codeExecEnv, "1")
		t.Setenv(sandboxModeEnv, sandboxModeHost)
		t.Setenv("FAKE_PYTHON_MODE", "fail")

		_, err := ds.Evaluate("print('x')\n", humanEvalExpected{
			Prompt: " ",
			Test:   "t\n",
		})
		if err == nil {
			t.Fatalf("expected error")
		}
		if !strings.Contains(err.Error(), "python failed") {
			t.Fatalf("err=%q", err.Error())
		}
	})
}

func TestDefaultHumanEvalSample(t *testing.T) {
	if got := defaultHumanEvalSample(); len(got) == 0 {
		t.Fatalf("empty")
	}
}

func TestWriteHumanEvalProgram_TempDirError(t *testing.T) {
	tmpDir := t.TempDir()
	badTmp := filepath.Join(tmpDir, "notadir")
	if err := os.WriteFile(badTmp, []byte("x"), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	t.Setenv("TMPDIR", badTmp)

	_, _, cleanup, err := writeHumanEvalProgram("print('x')\n")
	if err == nil {
		t.Fatalf("expected error")
	}
	if cleanup != nil {
		t.Fatalf("expected cleanup=nil")
	}
	if !strings.Contains(err.Error(), "create temp dir") {
		t.Fatalf("err=%q", err.Error())
	}
}
