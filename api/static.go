package api

import (
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/gin-gonic/gin"
)

const staticRoot = "web/static"

func (s *Server) registerStatic() {
	if s == nil || s.router == nil {
		return
	}

	handler := func(c *gin.Context) {
		path := c.Request.URL.Path
		rootAbs, err := filepath.Abs(staticRoot)
		if err != nil {
			c.Status(http.StatusInternalServerError)
			return
		}
		indexPath := filepath.Join(rootAbs, "index.html")
		if path == "/" {
			c.File(indexPath)
			return
		}
		if strings.HasPrefix(path, "/api/") || path == "/api" {
			c.JSON(http.StatusNotFound, gin.H{"error": "not found"})
			return
		}

		rel := strings.TrimPrefix(path, "/")
		cleaned := filepath.Clean(rel)
		full := filepath.Join(staticRoot, cleaned)
		fullAbs, err := filepath.Abs(full)
		if err != nil {
			c.Status(http.StatusNotFound)
			return
		}
		rootPrefix := rootAbs + string(os.PathSeparator)
		if fullAbs != rootAbs && !strings.HasPrefix(fullAbs, rootPrefix) {
			c.Status(http.StatusForbidden)
			return
		}
		if info, err := os.Stat(fullAbs); err == nil && !info.IsDir() {
			c.File(fullAbs)
			return
		}
		c.File(indexPath)
	}

	s.router.GET("/*filepath", handler)
	s.router.HEAD("/*filepath", handler)
}
