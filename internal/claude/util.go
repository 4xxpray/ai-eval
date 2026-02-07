package claude

import (
	"encoding/json"
	"errors"
	"strings"
)

// ClaudeText concatenates text blocks from a response.
func ClaudeText(resp *Response) string {
	if resp == nil {
		return ""
	}
	var sb strings.Builder
	for _, b := range resp.Content {
		if b.Type == "text" {
			sb.WriteString(b.Text)
		}
	}
	return sb.String()
}

// ParseJSONFromClaude extracts the first JSON object from Claude output into out.
func ParseJSONFromClaude(raw string, out any) error {
	s := strings.TrimSpace(raw)
	if s == "" {
		return errors.New("empty output")
	}

	if strings.HasPrefix(s, "```") {
		s = strings.TrimSpace(strings.TrimPrefix(s, "```"))
		if strings.HasPrefix(s, "json") {
			s = strings.TrimSpace(strings.TrimPrefix(s, "json"))
		}
		if idx := strings.LastIndex(s, "```"); idx >= 0 {
			s = strings.TrimSpace(s[:idx])
		}
	}

	start := strings.Index(s, "{")
	end := strings.LastIndex(s, "}")
	if start < 0 || end < 0 || start >= end {
		return errors.New("missing JSON object")
	}

	s = s[start : end+1]
	if err := json.Unmarshal([]byte(s), out); err != nil {
		return err
	}
	return nil
}
