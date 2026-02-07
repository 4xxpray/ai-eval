package prompt

// Prompt defines a prompt template and metadata.
type Prompt struct {
	Name           string         `yaml:"name"`
	Version        string         `yaml:"version"`
	Description    string         `yaml:"description"`
	Template       string         `yaml:"template"`
	Variables      []Variable     `yaml:"variables"`
	Tools          []Tool         `yaml:"tools"`
	Metadata       map[string]any `yaml:"metadata"`
	IsSystemPrompt bool           `yaml:"is_system_prompt,omitempty"` // If true, use as system message
}

// Variable defines a prompt variable and defaults.
type Variable struct {
	Name     string `yaml:"name"`
	Required bool   `yaml:"required"`
	Default  string `yaml:"default,omitempty"`
}

// Tool describes a tool available to a prompt.
type Tool struct {
	Name        string         `yaml:"name"`
	Description string         `yaml:"description,omitempty"`
	InputSchema map[string]any `yaml:"input_schema,omitempty"`
}
