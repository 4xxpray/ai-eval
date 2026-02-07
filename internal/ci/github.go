package ci

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// DetectCI returns true if running in GitHub Actions.
func DetectCI() bool {
	return strings.EqualFold(strings.TrimSpace(os.Getenv("GITHUB_ACTIONS")), "true")
}

// SetOutput sets a GitHub Actions output variable.
func SetOutput(name, value string) {
	name = strings.TrimSpace(name)
	if name == "" {
		return
	}
	if path := strings.TrimSpace(os.Getenv("GITHUB_OUTPUT")); path != "" {
		_ = appendGitHubCommandFile(path, fmt.Sprintf("%s<<EOF\n%s\nEOF\n", name, value))
		return
	}
	fmt.Printf("::set-output name=%s::%s\n", name, escapeCommandValue(value))
}

// AddAnnotation adds a GitHub Actions annotation (error, warning, notice).
func AddAnnotation(level, file string, line int, message string) {
	lvl := strings.ToLower(strings.TrimSpace(level))
	switch lvl {
	case "error", "warning", "notice":
	default:
		lvl = "notice"
	}

	msg := escapeCommandValue(message)
	file = strings.TrimSpace(file)

	if file == "" {
		fmt.Printf("::%s::%s\n", lvl, msg)
		return
	}
	if line > 0 {
		fmt.Printf("::%s file=%s,line=%d::%s\n", lvl, file, line, msg)
		return
	}
	fmt.Printf("::%s file=%s::%s\n", lvl, file, msg)
}

// StartGroup starts a collapsible group in GitHub Actions logs.
func StartGroup(name string) {
	fmt.Printf("::group::%s\n", escapeCommandValue(name))
}

// EndGroup ends a collapsible group.
func EndGroup() {
	fmt.Println("::endgroup::")
}

// SetJobSummary writes markdown to the job summary.
func SetJobSummary(markdown string) error {
	path := strings.TrimSpace(os.Getenv("GITHUB_STEP_SUMMARY"))
	if path == "" {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	if !strings.HasSuffix(markdown, "\n") {
		markdown += "\n"
	}
	return appendGitHubCommandFile(path, markdown)
}

func appendGitHubCommandFile(path, content string) error {
	f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY|os.O_CREATE, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = f.WriteString(content)
	return err
}

func escapeCommandValue(s string) string {
	s = strings.ReplaceAll(s, "%", "%25")
	s = strings.ReplaceAll(s, "\r", "%0D")
	s = strings.ReplaceAll(s, "\n", "%0A")
	return s
}
