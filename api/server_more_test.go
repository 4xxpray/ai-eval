package api

import (
	"net"
	"testing"

	"github.com/stellarlinkco/ai-eval/internal/config"
)

func TestNewServer_RequiresAuthConfig(t *testing.T) {
	t.Setenv("AI_EVAL_API_KEY", "")
	t.Setenv("AI_EVAL_DISABLE_AUTH", "")

	if _, err := NewServer(&config.Config{}, nil, nil, nil); err == nil {
		t.Fatalf("expected error")
	}
}

func TestNewServer_SucceedsWithDisableAuth(t *testing.T) {
	t.Setenv("AI_EVAL_API_KEY", "")
	t.Setenv("AI_EVAL_DISABLE_AUTH", "true")

	s, err := NewServer(&config.Config{}, nil, nil, nil)
	if err != nil {
		t.Fatalf("NewServer: %v", err)
	}
	if s == nil || s.router == nil {
		t.Fatalf("expected server with router")
	}
}

func TestServerRun_ErrorsOnNilServer(t *testing.T) {
	var s *Server
	if err := s.Run(":0"); err == nil {
		t.Fatalf("expected error")
	}
}

func TestServerRun_ErrorsOnBadAddr(t *testing.T) {
	t.Setenv("AI_EVAL_API_KEY", "")
	t.Setenv("AI_EVAL_DISABLE_AUTH", "true")

	s, err := NewServer(&config.Config{}, nil, nil, nil)
	if err != nil {
		t.Fatalf("NewServer: %v", err)
	}

	if err := s.Run(" 127.0.0.1"); err == nil {
		t.Fatalf("expected error")
	}
}

func TestServerRun_DefaultAddrErrorsFast(t *testing.T) {
	t.Setenv("AI_EVAL_API_KEY", "")
	t.Setenv("AI_EVAL_DISABLE_AUTH", "true")

	s, err := NewServer(&config.Config{}, nil, nil, nil)
	if err != nil {
		t.Fatalf("NewServer: %v", err)
	}

	l4, _ := net.Listen("tcp4", ":8080")
	if l4 != nil {
		defer l4.Close()
	}
	l6, _ := net.Listen("tcp6", ":8080")
	if l6 != nil {
		defer l6.Close()
	}

	if err := s.Run(" "); err == nil {
		t.Fatalf("expected error")
	}
}
