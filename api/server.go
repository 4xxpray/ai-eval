package api

import (
	"errors"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/stellarlinkco/ai-eval/internal/config"
	"github.com/stellarlinkco/ai-eval/internal/leaderboard"
	"github.com/stellarlinkco/ai-eval/internal/llm"
	"github.com/stellarlinkco/ai-eval/internal/store"
)

type Server struct {
	router   *gin.Engine
	store    store.Store
	provider llm.Provider
	config   *config.Config
	lbStore  *leaderboard.Store
}

func NewServer(cfg *config.Config, st store.Store, provider llm.Provider, lbStore *leaderboard.Store) (*Server, error) {
	r := gin.New()
	s := &Server{
		router:   r,
		store:    st,
		provider: provider,
		config:   cfg,
		lbStore:  lbStore,
	}
	s.registerMiddleware()
	if err := s.registerRoutes(); err != nil {
		return nil, err
	}
	s.registerStatic()
	return s, nil
}

func (s *Server) Run(addr string) error {
	if s == nil || s.router == nil {
		return errors.New("api: nil server")
	}
	addr = strings.TrimSpace(addr)
	if addr == "" {
		addr = ":8080"
	}
	return s.router.Run(addr)
}
