package api

import (
	"net/http"
	"os"
	"strings"

	"github.com/gin-gonic/gin"
)

func (s *Server) registerMiddleware() {
	if s == nil || s.router == nil {
		return
	}
	s.router.Use(requestLoggingMiddleware(), recoveryMiddleware(), corsMiddleware())
}

func corsMiddleware() gin.HandlerFunc {
	raw := strings.TrimSpace(os.Getenv("AI_EVAL_CORS_ORIGINS"))
	if raw == "" {
		return func(c *gin.Context) {
			c.Next()
		}
	}

	allowAll := false
	allowedOrigins := make(map[string]struct{})
	for _, part := range strings.Split(raw, ",") {
		origin := strings.TrimSpace(part)
		if origin == "" {
			continue
		}
		if origin == "*" {
			allowAll = true
			allowedOrigins = nil
			break
		}
		allowedOrigins[origin] = struct{}{}
	}
	if !allowAll && len(allowedOrigins) == 0 {
		return func(c *gin.Context) {
			c.Next()
		}
	}

	return func(c *gin.Context) {
		origin := strings.TrimSpace(c.GetHeader("Origin"))
		if origin != "" {
			allowed := allowAll
			if !allowed && allowedOrigins != nil {
				_, allowed = allowedOrigins[origin]
			}

			if allowed {
				if allowAll {
					c.Header("Access-Control-Allow-Origin", "*")
				} else {
					c.Header("Access-Control-Allow-Origin", origin)
					c.Header("Vary", "Origin")
				}
				c.Header("Access-Control-Allow-Methods", "GET,POST,DELETE,OPTIONS")
				c.Header("Access-Control-Allow-Headers", "Content-Type, X-API-Key")
				c.Header("Access-Control-Max-Age", "3600")
			}

			if c.Request.Method == http.MethodOptions {
				c.AbortWithStatus(http.StatusNoContent)
				return
			}
		}
		c.Next()
	}
}

func requestLoggingMiddleware() gin.HandlerFunc {
	return gin.Logger()
}

func recoveryMiddleware() gin.HandlerFunc {
	return gin.Recovery()
}

func apiKeyAuthMiddleware(expected string) gin.HandlerFunc {
	expected = strings.TrimSpace(expected)
	if expected == "" {
		return func(c *gin.Context) { c.Next() }
	}

	return func(c *gin.Context) {
		if c.Request.Method == http.MethodOptions {
			c.Next()
			return
		}
		provided := strings.TrimSpace(c.GetHeader("X-API-Key"))
		if provided == "" || provided != expected {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{"error": "unauthorized"})
			return
		}
		c.Next()
	}
}
