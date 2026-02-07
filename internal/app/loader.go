package app

import (
	"fmt"
	"io/fs"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/stellarlinkco/ai-eval/internal/prompt"
	"github.com/stellarlinkco/ai-eval/internal/testcase"
)

func LoadPrompts(dir string) ([]*prompt.Prompt, error) {
	return prompt.LoadFromDir(dir)
}

func LoadPromptsRecursive(dir string) ([]*prompt.Prompt, error) {
	var paths []string
	err := filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		ext := strings.ToLower(filepath.Ext(d.Name()))
		if ext != ".yaml" && ext != ".yml" {
			return nil
		}
		paths = append(paths, path)
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("prompt: walk dir %q: %w", dir, err)
	}
	sort.Strings(paths)

	out := make([]*prompt.Prompt, 0, len(paths))
	for _, path := range paths {
		p, err := prompt.LoadFromFile(path)
		if err != nil {
			return nil, err
		}
		out = append(out, p)
	}
	return out, nil
}

func LoadTestSuites(dir string) ([]*testcase.TestSuite, error) {
	return testcase.LoadFromDir(dir)
}

func FindPromptByNameVersion(prompts []*prompt.Prompt, name string, version string) (*prompt.Prompt, error) {
	name = strings.TrimSpace(name)
	version = strings.TrimSpace(version)
	if name == "" || version == "" {
		return nil, fmt.Errorf("prompt: missing name/version")
	}

	var match *prompt.Prompt
	for _, p := range prompts {
		if p == nil {
			continue
		}
		if strings.TrimSpace(p.Name) != name {
			continue
		}
		if strings.TrimSpace(p.Version) != version {
			continue
		}
		if match != nil {
			return nil, fmt.Errorf("prompt: multiple matches for name=%q version=%q", name, version)
		}
		match = p
	}
	if match == nil {
		return nil, fmt.Errorf("prompt: no prompt found for name=%q version=%q", name, version)
	}
	return match, nil
}

func FindPromptLatestByName(prompts []*prompt.Prompt, name string) (*prompt.Prompt, error) {
	name = strings.TrimSpace(name)
	if name == "" {
		return nil, fmt.Errorf("prompt: missing name")
	}

	var best *prompt.Prompt
	for _, p := range prompts {
		if p == nil {
			continue
		}
		if strings.TrimSpace(p.Name) != name {
			continue
		}

		if best == nil {
			best = p
			continue
		}

		cmp := compareVersions(strings.TrimSpace(p.Version), strings.TrimSpace(best.Version))
		if cmp > 0 {
			best = p
			continue
		}
		if cmp == 0 && strings.TrimSpace(p.Version) != "" {
			return nil, fmt.Errorf("prompt: multiple matches for name=%q version=%q", name, strings.TrimSpace(p.Version))
		}
	}

	if best == nil {
		return nil, fmt.Errorf("prompt: unknown prompt %q", name)
	}
	return best, nil
}

func FilterSuitesByPrompt(suites []*testcase.TestSuite, promptName string) []*testcase.TestSuite {
	promptName = strings.TrimSpace(promptName)
	if promptName == "" {
		return nil
	}
	out := make([]*testcase.TestSuite, 0, len(suites))
	for _, s := range suites {
		if s == nil {
			continue
		}
		if strings.TrimSpace(s.Prompt) != promptName {
			continue
		}
		out = append(out, s)
	}
	return out
}

func compareVersions(a, b string) int {
	a = strings.TrimSpace(a)
	b = strings.TrimSpace(b)
	if a == b {
		return 0
	}

	an, aok := parseNumericVersion(a)
	bn, bok := parseNumericVersion(b)
	if aok && bok {
		max := len(an)
		if len(bn) > max {
			max = len(bn)
		}
		for i := 0; i < max; i++ {
			ai := 0
			if i < len(an) {
				ai = an[i]
			}
			bi := 0
			if i < len(bn) {
				bi = bn[i]
			}
			switch {
			case ai < bi:
				return -1
			case ai > bi:
				return 1
			}
		}
		return 0
	}

	return strings.Compare(a, b)
}

func parseNumericVersion(s string) ([]int, bool) {
	s = strings.TrimSpace(s)
	s = strings.TrimPrefix(strings.ToLower(s), "v")
	if s == "" {
		return nil, false
	}
	parts := strings.Split(s, ".")
	out := make([]int, 0, len(parts))
	for _, p := range parts {
		if p == "" {
			return nil, false
		}
		n, err := strconv.Atoi(p)
		if err != nil {
			return nil, false
		}
		out = append(out, n)
	}
	return out, true
}
