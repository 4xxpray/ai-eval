package evaluator

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"strings"
)

// JSONSchemaEvaluator validates JSON output against a schema.
type JSONSchemaEvaluator struct{}

// Name returns the evaluator identifier.
func (JSONSchemaEvaluator) Name() string {
	return "json_schema"
}

// Evaluate validates the response JSON against the expected schema.
func (JSONSchemaEvaluator) Evaluate(ctx context.Context, response string, expected any) (*Result, error) {
	schema, ok := expected.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("json_schema: expected map[string]any, got %T", expected)
	}

	var value any
	dec := json.NewDecoder(strings.NewReader(response))
	dec.UseNumber()
	if err := dec.Decode(&value); err != nil {
		return &Result{
			Passed:  false,
			Score:   0.0,
			Message: "invalid json",
			Details: map[string]any{"error": err.Error()},
		}, nil
	}
	if err := dec.Decode(&struct{}{}); err != io.EOF {
		if err == nil {
			err = fmt.Errorf("extra data after JSON value")
		}
		return &Result{
			Passed:  false,
			Score:   0.0,
			Message: "invalid json",
			Details: map[string]any{"error": err.Error()},
		}, nil
	}

	if err := validateJSONSchema(value, schema, "$"); err != nil {
		var se *schemaError
		if errors.As(err, &se) {
			return nil, err
		}
		return &Result{
			Passed:  false,
			Score:   0.0,
			Message: err.Error(),
		}, nil
	}

	return &Result{
		Passed:  true,
		Score:   1.0,
		Message: "valid json schema",
	}, nil
}

type schemaError struct {
	path string
	err  error
}

// Error returns the formatted schema error message.
func (e *schemaError) Error() string {
	return fmt.Sprintf("%s: %v", e.path, e.err)
}

// Unwrap returns the underlying schema error.
func (e *schemaError) Unwrap() error {
	return e.err
}

func validateJSONSchema(value any, schema map[string]any, path string) error {
	typ, err := schemaType(schema)
	if err != nil {
		return &schemaError{path: path, err: err}
	}

	switch typ {
	case "object":
		obj, ok := value.(map[string]any)
		if !ok {
			return fmt.Errorf("%s: expected object", path)
		}

		if raw, ok := schema["required"]; ok {
			required, err := asStringSlice(raw)
			if err != nil {
				return &schemaError{path: path, err: fmt.Errorf("required: %w", err)}
			}
			for _, key := range required {
				if _, ok := obj[key]; !ok {
					return fmt.Errorf("%s.%s: missing required field", path, key)
				}
			}
		}

		rawProps, ok := schema["properties"]
		if !ok {
			return nil
		}
		props, ok := rawProps.(map[string]any)
		if !ok {
			return &schemaError{path: path, err: fmt.Errorf("properties must be an object")}
		}

		for key, rawPropSchema := range props {
			child, ok := obj[key]
			if !ok {
				continue
			}
			propSchema, ok := rawPropSchema.(map[string]any)
			if !ok {
				return &schemaError{path: path + "." + key, err: fmt.Errorf("schema must be an object")}
			}
			if err := validateJSONSchema(child, propSchema, path+"."+key); err != nil {
				return err
			}
		}
		return nil

	case "array":
		arr, ok := value.([]any)
		if !ok {
			return fmt.Errorf("%s: expected array", path)
		}

		rawItems, ok := schema["items"]
		if !ok {
			return nil
		}
		itemsSchema, ok := rawItems.(map[string]any)
		if !ok {
			return &schemaError{path: path, err: fmt.Errorf("items must be an object")}
		}
		for i, elem := range arr {
			if err := validateJSONSchema(elem, itemsSchema, fmt.Sprintf("%s[%d]", path, i)); err != nil {
				return err
			}
		}
		return nil

	case "string":
		if _, ok := value.(string); !ok {
			return fmt.Errorf("%s: expected string", path)
		}
		return nil

	case "number":
		if !isNumber(value) {
			return fmt.Errorf("%s: expected number", path)
		}
		return nil

	case "integer":
		if !isInteger(value) {
			return fmt.Errorf("%s: expected integer", path)
		}
		return nil

	case "boolean":
		if _, ok := value.(bool); !ok {
			return fmt.Errorf("%s: expected boolean", path)
		}
		return nil

	case "null":
		if value != nil {
			return fmt.Errorf("%s: expected null", path)
		}
		return nil

	default:
		return &schemaError{path: path, err: fmt.Errorf("unsupported schema type %q", typ)}
	}
}

func schemaType(schema map[string]any) (string, error) {
	if schema == nil {
		return "", fmt.Errorf("nil schema")
	}
	if raw, ok := schema["type"]; ok {
		s, ok := raw.(string)
		if !ok {
			return "", fmt.Errorf("type must be string")
		}
		s = strings.TrimSpace(s)
		if s == "" {
			return "", fmt.Errorf("type must be non-empty")
		}
		return s, nil
	}

	if _, ok := schema["properties"]; ok {
		return "object", nil
	}
	if _, ok := schema["required"]; ok {
		return "object", nil
	}
	if _, ok := schema["items"]; ok {
		return "array", nil
	}
	return "", fmt.Errorf("missing type")
}

func isNumber(v any) bool {
	switch n := v.(type) {
	case json.Number:
		_, err := n.Float64()
		return err == nil
	case float64, float32,
		int, int64, int32, int16, int8,
		uint, uint64, uint32, uint16, uint8:
		return true
	default:
		return false
	}
}

func isInteger(v any) bool {
	switch n := v.(type) {
	case json.Number:
		if _, err := n.Int64(); err == nil {
			return true
		}
		f, err := n.Float64()
		if err != nil {
			return false
		}
		return math.Trunc(f) == f
	case float64:
		return math.Trunc(n) == n
	case float32:
		f := float64(n)
		return math.Trunc(f) == f
	case int, int64, int32, int16, int8,
		uint, uint64, uint32, uint16, uint8:
		return true
	default:
		return false
	}
}
