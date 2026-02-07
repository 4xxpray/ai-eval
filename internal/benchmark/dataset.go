package benchmark

import "context"

type Dataset interface {
	Name() string
	Description() string
	Load(ctx context.Context) ([]Question, error)
	Evaluate(response string, expected any) (float64, error)
}

type Question struct {
	ID       string
	Question string
	Choices  []string
	Answer   any
	Category string
}

