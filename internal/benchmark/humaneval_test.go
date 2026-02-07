package benchmark

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestHumanEval_RunHumanEvalPython_Disabled(t *testing.T) {
	t.Setenv(sandboxModeEnv, sandboxModeDisabled)

	ok, err := runHumanEvalPython("print('ok')\n", 2*time.Second)
	if err == nil {
		t.Fatalf("expected error")
	}
	if ok {
		t.Fatalf("ok=true")
	}
	if !strings.Contains(err.Error(), "code execution disabled") {
		t.Fatalf("err=%q", err.Error())
	}
}

func TestHumanEval_RunHumanEvalPython_Host(t *testing.T) {
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skipf("python3 not available: %v", err)
	}

	t.Setenv(sandboxModeEnv, sandboxModeHost)

	ok, err := runHumanEvalPython("print('ok')\n", 5*time.Second)
	if err != nil {
		t.Fatalf("runHumanEvalPython: %v", err)
	}
	if !ok {
		t.Fatalf("ok=false")
	}
}

func TestHumanEval_RunHumanEvalPython_Host_NoDockerInPATH(t *testing.T) {
	python3, err := exec.LookPath("python3")
	if err != nil {
		t.Skipf("python3 not available: %v", err)
	}

	tmpDir := t.TempDir()
	if err := os.Symlink(python3, filepath.Join(tmpDir, "python3")); err != nil {
		t.Skipf("symlink python3: %v", err)
	}

	t.Setenv("PATH", tmpDir)
	t.Setenv(sandboxModeEnv, sandboxModeHost)

	ok, err := runHumanEvalPython("print('ok')\n", 5*time.Second)
	if err != nil {
		t.Fatalf("runHumanEvalPython: %v", err)
	}
	if !ok {
		t.Fatalf("ok=false")
	}
}

func TestHumanEval_RunHumanEvalPython_Docker(t *testing.T) {
	if _, err := humanEvalDockerReady(); err != nil {
		t.Skipf("docker sandbox not available: %v", err)
	}

	t.Setenv(sandboxModeEnv, sandboxModeDocker)

	ok, err := runHumanEvalPython("print('ok')\n", 5*time.Second)
	if err != nil {
		t.Fatalf("runHumanEvalPython: %v", err)
	}
	if !ok {
		t.Fatalf("ok=false")
	}
}
