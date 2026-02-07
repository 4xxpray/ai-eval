package benchmark

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

const (
	defaultHumanEvalPath = "data/benchmark/humaneval.jsonl"
	codeExecEnv          = "AI_EVAL_ENABLE_CODE_EXEC"
	sandboxModeEnv       = "AI_EVAL_SANDBOX_MODE"

	sandboxModeDocker   = "docker"
	sandboxModeHost     = "host"
	sandboxModeDisabled = "disabled"

	humanEvalDockerImage = "python:3.11-slim"
)

var (
	errHumanEvalCodeExecDisabled = fmt.Errorf("humaneval: code execution disabled (set %s=1)", codeExecEnv)

	humanEvalDockerReadyOnce sync.Once
	humanEvalDockerBin       string
	humanEvalDockerReadyErr  error

	humanEvalHostWarnOnce sync.Once
)

type HumanEvalDataset struct {
	SampleSize int
}

type humanEvalRow struct {
	ID         string `json:"id,omitempty"`
	TaskID     string `json:"task_id,omitempty"`
	Prompt     string `json:"prompt"`
	Test       string `json:"test"`
	EntryPoint string `json:"entry_point,omitempty"`
}

type humanEvalExpected struct {
	Prompt     string
	Test       string
	EntryPoint string
}

func (d *HumanEvalDataset) Name() string { return "humaneval" }

func (d *HumanEvalDataset) Description() string {
	return "HumanEval code generation benchmark (requires local code execution)"
}

func (d *HumanEvalDataset) Load(ctx context.Context) ([]Question, error) {
	if ctx == nil {
		return nil, errors.New("humaneval: nil context")
	}

	path := strings.TrimSpace(os.Getenv("AI_EVAL_HUMANEVAL_PATH"))
	if path == "" {
		path = defaultHumanEvalPath
	}

	rows, err := readJSONL[humanEvalRow](ctx, path)
	if err != nil {
		if os.IsNotExist(err) {
			return takeFirstN(defaultHumanEvalSample(), d.SampleSize), nil
		}
		return nil, fmt.Errorf("humaneval: load %q: %w", path, err)
	}

	out := make([]Question, 0, len(rows))
	for i, row := range rows {
		if err := ctx.Err(); err != nil {
			return out, err
		}

		prompt := strings.TrimSpace(row.Prompt)
		test := strings.TrimSpace(row.Test)
		if prompt == "" || test == "" {
			continue
		}

		id := strings.TrimSpace(row.ID)
		if id == "" {
			id = strings.TrimSpace(row.TaskID)
		}
		if id == "" {
			id = fmt.Sprintf("humaneval-%d", i+1)
		}

		out = append(out, Question{
			ID:       id,
			Question: prompt,
			Answer: humanEvalExpected{
				Prompt:     prompt,
				Test:       test,
				EntryPoint: strings.TrimSpace(row.EntryPoint),
			},
			Category: "code",
		})
	}

	out = takeFirstN(out, d.SampleSize)
	if len(out) == 0 {
		return takeFirstN(defaultHumanEvalSample(), d.SampleSize), nil
	}
	return out, nil
}

func (d *HumanEvalDataset) Evaluate(response string, expected any) (float64, error) {
	if strings.TrimSpace(os.Getenv(codeExecEnv)) != "1" {
		return 0, errHumanEvalCodeExecDisabled
	}

	exp, ok := expected.(humanEvalExpected)
	if !ok {
		if p, ok := expected.(*humanEvalExpected); ok && p != nil {
			exp = *p
		} else {
			return 0, fmt.Errorf("humaneval: unsupported expected type %T", expected)
		}
	}

	candidate := stripCodeFences(response)
	if strings.TrimSpace(candidate) == "" {
		return 0, errors.New("humaneval: empty model output")
	}

	ok, err := runHumanEvalPython(exp.Prompt+"\n"+candidate+"\n"+exp.Test, 5*time.Second)
	if ok {
		return 1, nil
	}

	if strings.TrimSpace(exp.Prompt) != "" {
		if ok2, err2 := runHumanEvalPython(candidate+"\n"+exp.Test, 5*time.Second); ok2 {
			return 1, nil
		} else if err2 != nil {
			return 0, err2
		}
	}

	if err != nil {
		return 0, err
	}
	return 0, errors.New("humaneval: python execution failed")
}

func stripCodeFences(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return ""
	}

	if strings.HasPrefix(s, "```") {
		s = strings.TrimPrefix(s, "```")
		s = strings.TrimSpace(s)

		// Optional language tag.
		if i := strings.IndexByte(s, '\n'); i >= 0 {
			first := strings.ToLower(strings.TrimSpace(s[:i]))
			if first == "python" || first == "py" {
				s = s[i+1:]
			}
		}

		if idx := strings.LastIndex(s, "```"); idx >= 0 {
			s = s[:idx]
		}
	}

	return strings.TrimSpace(s)
}

func runHumanEvalPython(program string, timeout time.Duration) (bool, error) {
	mode := strings.ToLower(strings.TrimSpace(os.Getenv(sandboxModeEnv)))
	if mode == "" {
		mode = sandboxModeDocker
	}

	switch mode {
	case sandboxModeDisabled:
		return false, errHumanEvalCodeExecDisabled
	case sandboxModeHost:
		humanEvalHostWarnOnce.Do(func() {
			log.Printf("humaneval: WARNING: executing untrusted code on host (set %s=%s for sandboxing)", sandboxModeEnv, sandboxModeDocker)
		})
		return runHumanEvalPythonHost(program, timeout)
	case sandboxModeDocker:
		return runHumanEvalPythonDocker(program, timeout)
	default:
		return false, fmt.Errorf("humaneval: unknown %s=%q (expected %s|%s|%s)", sandboxModeEnv, mode, sandboxModeDocker, sandboxModeHost, sandboxModeDisabled)
	}
}

func writeHumanEvalProgram(program string) (tmpDir string, path string, cleanup func(), _ error) {
	tmpDir, err := os.MkdirTemp("", "ai-eval-humaneval-*")
	if err != nil {
		return "", "", nil, fmt.Errorf("humaneval: create temp dir: %w", err)
	}
	cleanup = func() { _ = os.RemoveAll(tmpDir) }

	path = filepath.Join(tmpDir, "main.py")
	if err := os.WriteFile(path, []byte(program), 0o644); err != nil {
		cleanup()
		return "", "", nil, fmt.Errorf("humaneval: write program: %w", err)
	}

	return tmpDir, path, cleanup, nil
}

func runHumanEvalPythonHost(program string, timeout time.Duration) (bool, error) {
	python, err := exec.LookPath("python3")
	if err != nil {
		return false, fmt.Errorf("humaneval: python3 not found: %w", err)
	}

	tmpDir, path, cleanup, err := writeHumanEvalProgram(program)
	if err != nil {
		return false, err
	}
	defer cleanup()

	ctx := context.Background()
	if timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, timeout)
		defer cancel()
	}

	cmd := exec.CommandContext(ctx, python, "-I", "-B", path)
	cmd.Dir = tmpDir
	cmd.Env = append(os.Environ(),
		"PYTHONPATH=",
		"PYTHONSAFEPATH=1",
		"HOME="+tmpDir,
	)

	out, err := cmd.CombinedOutput()
	if ctx.Err() != nil {
		return false, fmt.Errorf("humaneval: python timeout: %w", ctx.Err())
	}
	if err != nil {
		return false, fmt.Errorf("humaneval: python failed: %s", truncateOutput(out, 4096))
	}
	return true, nil
}

func runHumanEvalPythonDocker(program string, timeout time.Duration) (bool, error) {
	docker, err := humanEvalDockerReady()
	if err != nil {
		return false, err
	}

	_, scriptPath, cleanup, err := writeHumanEvalProgram(program)
	if err != nil {
		return false, err
	}
	defer cleanup()

	ctx := context.Background()
	if timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, timeout)
		defer cancel()
	}

	containerName := fmt.Sprintf("ai-eval-humaneval-%d-%d", os.Getpid(), time.Now().UnixNano())

	args := []string{
		"run",
		"--rm",
		"--name", containerName,
		"--network=none",
		"--read-only",
		"--cap-drop=ALL",
		"--memory=128m",
		"--cpus=0.5",
		"--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=64m",
		"--security-opt", "no-new-privileges",
		"--user", "65534:65534",
		"--env", "PYTHONPATH=",
		"--env", "PYTHONSAFEPATH=1",
		"--env", "HOME=/tmp",
		"--mount", fmt.Sprintf("type=bind,source=%s,target=/tmp/main.py,readonly", scriptPath),
		humanEvalDockerImage,
		"python",
		"-I",
		"-B",
		"/tmp/main.py",
	}

	cmd := exec.CommandContext(ctx, docker, args...)
	out, runErr := cmd.CombinedOutput()
	if ctx.Err() != nil {
		killCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = exec.CommandContext(killCtx, docker, "rm", "-f", containerName).Run()
		return false, fmt.Errorf("humaneval: python timeout: %w", ctx.Err())
	}
	if runErr != nil {
		if ee, ok := runErr.(*exec.ExitError); ok {
			switch ee.ExitCode() {
			case 125, 126, 127:
				return false, fmt.Errorf("humaneval: docker run failed: %s", truncateOutput(out, 4096))
			}
		}
		return false, fmt.Errorf("humaneval: python failed: %s", truncateOutput(out, 4096))
	}
	return true, nil
}

func humanEvalDockerReady() (string, error) {
	humanEvalDockerReadyOnce.Do(func() {
		docker, err := exec.LookPath("docker")
		if err != nil {
			humanEvalDockerReadyErr = fmt.Errorf("humaneval: docker sandbox unavailable: docker not found (install Docker, or set %s=%s to run on host; UNSAFE)", sandboxModeEnv, sandboxModeHost)
			return
		}

		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		version := exec.CommandContext(ctx, docker, "version", "--format", "{{.Server.Version}}")
		out, err := version.CombinedOutput()
		if ctx.Err() != nil {
			humanEvalDockerReadyErr = fmt.Errorf("humaneval: docker sandbox unavailable: docker version timeout: %w", ctx.Err())
			return
		}
		if err != nil {
			humanEvalDockerReadyErr = fmt.Errorf("humaneval: docker sandbox unavailable: docker daemon not reachable: %s (or set %s=%s to run on host; UNSAFE)", truncateOutput(out, 4096), sandboxModeEnv, sandboxModeHost)
			return
		}

		ctx, cancel = context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		inspect := exec.CommandContext(ctx, docker, "image", "inspect", "-f", "{{.Id}}", humanEvalDockerImage)
		out, err = inspect.CombinedOutput()
		if ctx.Err() != nil {
			humanEvalDockerReadyErr = fmt.Errorf("humaneval: docker sandbox unavailable: docker image inspect timeout: %w", ctx.Err())
			return
		}
		if err != nil {
			humanEvalDockerReadyErr = fmt.Errorf("humaneval: docker sandbox unavailable: missing image %q (%s) (run: docker pull %s, or set %s=%s to run on host; UNSAFE)", humanEvalDockerImage, truncateOutput(out, 256), humanEvalDockerImage, sandboxModeEnv, sandboxModeHost)
			return
		}

		humanEvalDockerBin = docker
	})

	if humanEvalDockerReadyErr != nil {
		return "", humanEvalDockerReadyErr
	}
	return humanEvalDockerBin, nil
}

func truncateOutput(b []byte, max int) string {
	s := strings.TrimSpace(string(b))
	if max <= 0 || len(s) <= max {
		return s
	}
	return strings.TrimSpace(s[:max]) + "..."
}

func defaultHumanEvalSample() []Question {
	return []Question{
		{
			ID:       "humaneval-sample-1",
			Category: "code",
			Question: "def add(a, b):\n    \"\"\"Return the sum of a and b.\"\"\"\n",
			Answer: humanEvalExpected{
				Prompt: "def add(a, b):\n    \"\"\"Return the sum of a and b.\"\"\"\n",
				Test:   "def check(candidate):\n    assert candidate(1, 2) == 3\n    assert candidate(-1, 1) == 0\n\ncheck(add)\n",
			},
		},
		{
			ID:       "humaneval-sample-2",
			Category: "code",
			Question: "def reverse_string(s):\n    \"\"\"Return the string s reversed.\"\"\"\n",
			Answer: humanEvalExpected{
				Prompt: "def reverse_string(s):\n    \"\"\"Return the string s reversed.\"\"\"\n",
				Test:   "def check(candidate):\n    assert candidate(\"abc\") == \"cba\"\n    assert candidate(\"\") == \"\"\n\ncheck(reverse_string)\n",
			},
		},
	}
}
