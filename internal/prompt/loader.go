package prompt

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"
)

// LoadFromFile loads a prompt definition from a YAML file.
func LoadFromFile(path string) (*Prompt, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("prompt: read %q: %w", path, err)
	}

	var p Prompt
	if err := yaml.Unmarshal(b, &p); err != nil {
		return nil, fmt.Errorf("prompt: parse %q: %w", path, err)
	}
	return &p, nil
}

// LoadFromDir loads all prompt definitions from a directory.
func LoadFromDir(dir string) ([]*Prompt, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, fmt.Errorf("prompt: read dir %q: %w", dir, err)
	}

	var paths []string
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		ext := strings.ToLower(filepath.Ext(entry.Name()))
		if ext != ".yaml" && ext != ".yml" {
			continue
		}
		paths = append(paths, filepath.Join(dir, entry.Name()))
	}
	sort.Strings(paths)

	out := make([]*Prompt, 0, len(paths))
	for _, path := range paths {
		p, err := LoadFromFile(path)
		if err != nil {
			return nil, err
		}
		out = append(out, p)
	}
	return out, nil
}
