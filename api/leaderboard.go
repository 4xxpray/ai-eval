package api

import (
	"errors"
	"net/http"
	"strconv"
	"strings"

	"github.com/gin-gonic/gin"
)

func (s *Server) handleGetLeaderboard(c *gin.Context) {
	if s == nil || s.lbStore == nil {
		respondError(c, http.StatusInternalServerError, errors.New("leaderboard store not configured"))
		return
	}

	dataset := strings.TrimSpace(c.Query("dataset"))
	if dataset == "" {
		respondError(c, http.StatusBadRequest, errors.New("dataset is required"))
		return
	}

	limit := 20
	if raw := strings.TrimSpace(c.Query("limit")); raw != "" {
		n, err := strconv.Atoi(raw)
		if err != nil || n <= 0 {
			respondError(c, http.StatusBadRequest, errors.New("invalid limit"))
			return
		}
		if n > 100 {
			n = 100
		}
		limit = n
	}

	entries, err := s.lbStore.GetLeaderboard(c.Request.Context(), dataset, limit)
	if err != nil {
		respondError(c, http.StatusInternalServerError, err)
		return
	}

	c.JSON(http.StatusOK, entries)
}

func (s *Server) handleGetModelHistory(c *gin.Context) {
	if s == nil || s.lbStore == nil {
		respondError(c, http.StatusInternalServerError, errors.New("leaderboard store not configured"))
		return
	}

	model := strings.TrimSpace(c.Query("model"))
	dataset := strings.TrimSpace(c.Query("dataset"))
	if model == "" || dataset == "" {
		respondError(c, http.StatusBadRequest, errors.New("model and dataset are required"))
		return
	}

	entries, err := s.lbStore.GetModelHistory(c.Request.Context(), model, dataset)
	if err != nil {
		respondError(c, http.StatusInternalServerError, err)
		return
	}

	c.JSON(http.StatusOK, entries)
}

