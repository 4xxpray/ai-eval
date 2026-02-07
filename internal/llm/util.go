package llm

import (
	"encoding/json"
	"errors"
	"strings"
)

func Text(resp *Response) string {
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

// ParseJSON extracts the first JSON object from raw output into out.
func ParseJSON(raw string, out any) error {
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
	return json.Unmarshal([]byte(s), out)
}
