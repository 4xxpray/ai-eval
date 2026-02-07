package evaluator

import (
	"context"
	"strings"
)

// Evaluator defines a response evaluator.
type Evaluator interface {
	Name() string
	Evaluate(ctx context.Context, response string, expected any) (*Result, error)
}

// Result holds evaluation outcome data.
type Result struct {
	Passed  bool
	Score   float64 // 0.0 - 1.0
	Message string
	Details map[string]any
}

// Registry stores evaluators by name.
type Registry struct {
	evaluators map[string]Evaluator
}

// NewRegistry creates an empty evaluator registry.
func NewRegistry() *Registry {
	return &Registry{
		evaluators: make(map[string]Evaluator),
	}
}

// Register adds an evaluator to the registry.
func (r *Registry) Register(e Evaluator) {
	if r == nil {
		panic("evaluator: register on nil registry")
	}
	if e == nil {
		panic("evaluator: register nil evaluator")
	}
	name := strings.TrimSpace(e.Name())
	if name == "" {
		panic("evaluator: evaluator has empty name")
	}
	if r.evaluators == nil {
		r.evaluators = make(map[string]Evaluator)
	}
	r.evaluators[name] = e
}

// Get returns a named evaluator if present.
func (r *Registry) Get(name string) (Evaluator, bool) {
	if r == nil || r.evaluators == nil {
		return nil, false
	}
	e, ok := r.evaluators[name]
	return e, ok
}
