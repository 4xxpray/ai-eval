package prompt

import (
	"bytes"
	"errors"
	"fmt"
	"regexp"
	"strings"
	"text/template"
)

// mustacheVarPattern matches {{VAR_NAME}} (without dot prefix)
var mustacheVarPattern = regexp.MustCompile(`\{\{([A-Z][A-Z0-9_]*)\}\}`)

// Render renders a prompt template with variables.
// Supports both Go template syntax ({{.VarName}}) and Mustache-style ({{VAR_NAME}}).
func Render(p *Prompt, vars map[string]any) (string, error) {
	if p == nil {
		return "", errors.New("prompt: nil prompt")
	}

	data := make(map[string]any, len(vars)+len(p.Variables))
	for k, v := range vars {
		data[k] = v
	}

	for _, v := range p.Variables {
		if v.Name == "" {
			continue
		}
		_, ok := data[v.Name]
		if ok {
			continue
		}
		if v.Required {
			return "", fmt.Errorf("prompt: missing required variable %q", v.Name)
		}
		if v.Default != "" {
			data[v.Name] = v.Default
		}
	}

	// First, do simple string replacement for Mustache-style variables {{VAR_NAME}}
	rendered := p.Template
	for k, v := range data {
		placeholder := "{{" + k + "}}"
		if strings.Contains(rendered, placeholder) {
			rendered = strings.ReplaceAll(rendered, placeholder, fmt.Sprintf("%v", v))
		}
	}

	// Check if there are any remaining Go template constructs
	if strings.Contains(rendered, "{{.") || strings.Contains(rendered, "{{range") || strings.Contains(rendered, "{{if") {
		tmpl, err := template.New(p.Name).Option("missingkey=error").Parse(rendered)
		if err != nil {
			return "", fmt.Errorf("prompt: parse template: %w", err)
		}

		var buf bytes.Buffer
		if err := tmpl.Execute(&buf, data); err != nil {
			return "", fmt.Errorf("prompt: render template: %w", err)
		}
		return buf.String(), nil
	}

	if err := validateTemplateDelimiters(rendered); err != nil {
		return "", err
	}
	return rendered, nil
}

func validateTemplateDelimiters(s string) error {
	open := 0
	for i := 0; i+1 < len(s); i++ {
		if s[i] == '{' && s[i+1] == '{' {
			open++
			i++
			continue
		}
		if s[i] == '}' && s[i+1] == '}' {
			if open == 0 {
				return errors.New("prompt: unmatched \"}}\"")
			}
			open--
			i++
		}
	}
	if open != 0 {
		return errors.New("prompt: unmatched \"{{\"")
	}
	return nil
}
