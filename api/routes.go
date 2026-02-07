package api

import (
	"errors"
	"os"
	"strings"
)

func (s *Server) registerRoutes() error {
	if s == nil || s.router == nil {
		return nil
	}

	api := s.router.Group("/api")
	apiKey := strings.TrimSpace(os.Getenv("AI_EVAL_API_KEY"))
	if apiKey != "" {
		api.Use(apiKeyAuthMiddleware(apiKey))
	} else if strings.EqualFold(strings.TrimSpace(os.Getenv("AI_EVAL_DISABLE_AUTH")), "true") {
		// Explicitly allow unauthenticated access.
	} else {
		return errors.New("api: missing auth configuration: set AI_EVAL_API_KEY or set AI_EVAL_DISABLE_AUTH=true")
	}

	api.GET("/health", s.handleHealth)
	api.GET("/prompts", s.handleListPrompts)
	api.GET("/prompts/:name", s.handleGetPrompt)
	api.POST("/prompts", s.handleUpsertPrompt)
	api.DELETE("/prompts/:name", s.handleDeletePrompt)

	api.GET("/tests", s.handleListTests)
	api.GET("/tests/:suite", s.handleGetTestSuite)

	api.POST("/runs", s.handleStartRun)
	api.GET("/runs", s.handleListRuns)
	api.GET("/runs/:id", s.handleGetRun)
	api.GET("/runs/:id/results", s.handleGetRunResults)

	api.GET("/history/:prompt", s.handleGetPromptHistory)
	api.POST("/compare", s.handleCompareVersions)

	api.GET("/leaderboard", s.handleGetLeaderboard)
	api.GET("/leaderboard/history", s.handleGetModelHistory)

	// Optimize endpoint - auto evaluate and optimize prompt
	api.POST("/optimize", s.handleOptimize)

	// Diagnose endpoint - analyze failures and propose targeted fixes
	api.POST("/diagnose", s.handleDiagnose)

	return nil
}
