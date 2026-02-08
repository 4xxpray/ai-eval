package main

import (
	"bytes"
	"context"
	"errors"
	"os"
	"strings"
	"testing"

	"github.com/stellarlinkco/ai-eval/api"
	"github.com/stellarlinkco/ai-eval/internal/config"
	"github.com/stellarlinkco/ai-eval/internal/leaderboard"
	"github.com/stellarlinkco/ai-eval/internal/llm"
	"github.com/stellarlinkco/ai-eval/internal/store"
)

type stubStore struct {
	closeCalled int
	closeErr    error
}

func (s *stubStore) SaveRun(context.Context, *store.RunRecord) error { return nil }
func (s *stubStore) SaveSuiteResult(context.Context, *store.SuiteRecord) error {
	return nil
}
func (s *stubStore) GetRun(context.Context, string) (*store.RunRecord, error) { return nil, nil }
func (s *stubStore) ListRuns(context.Context, store.RunFilter) ([]*store.RunRecord, error) {
	return nil, nil
}
func (s *stubStore) GetSuiteResults(context.Context, string) ([]*store.SuiteRecord, error) {
	return nil, nil
}
func (s *stubStore) GetPromptHistory(context.Context, string, int) ([]*store.SuiteRecord, error) {
	return nil, nil
}
func (s *stubStore) GetVersionComparison(context.Context, string, string, string) (*store.VersionComparison, error) {
	return nil, nil
}
func (s *stubStore) Close() error {
	s.closeCalled++
	return s.closeErr
}

type noopProvider struct{}

func (noopProvider) Name() string { return "noop" }
func (noopProvider) Complete(context.Context, *llm.Request) (*llm.Response, error) {
	return &llm.Response{}, nil
}
func (noopProvider) CompleteWithTools(context.Context, *llm.Request) (*llm.EvalResult, error) {
	return &llm.EvalResult{}, nil
}

func saveServerGlobals(t *testing.T) func() {
	t.Helper()

	oldOsExit := osExit
	oldStderrWriter := stderrWriter
	oldLoadConfig := loadConfig
	oldOpenStore := openStore
	oldProviderFromConfig := defaultProviderFromConfig
	oldNewServer := newServer
	oldRunServer := runServer
	oldLeaderboardNewStore := leaderboardNewStore

	return func() {
		osExit = oldOsExit
		stderrWriter = oldStderrWriter
		loadConfig = oldLoadConfig
		openStore = oldOpenStore
		defaultProviderFromConfig = oldProviderFromConfig
		newServer = oldNewServer
		runServer = oldRunServer
		leaderboardNewStore = oldLeaderboardNewStore
	}
}

func TestOpenLeaderboardStore_NilConfig(t *testing.T) {
	_, err := openLeaderboardStore(nil)
	if err == nil {
		t.Fatalf("expected error")
	}
	if !strings.Contains(err.Error(), "missing config") {
		t.Fatalf("error: got %q", err)
	}
}

func TestOpenLeaderboardStore_DefaultSQLitePath(t *testing.T) {
	restore := saveServerGlobals(t)
	t.Cleanup(restore)

	oldNewStore := leaderboardNewStore
	var gotPath string
	leaderboardNewStore = func(path string) (*leaderboard.Store, error) {
		gotPath = path
		return oldNewStore(":memory:")
	}

	lb, err := openLeaderboardStore(&config.Config{})
	if err != nil {
		t.Fatalf("openLeaderboardStore: %v", err)
	}
	t.Cleanup(func() { _ = lb.Close() })

	if gotPath != store.DefaultSQLitePath {
		t.Fatalf("path: got %q want %q", gotPath, store.DefaultSQLitePath)
	}
}

func TestOpenLeaderboardStore_SQLitePathTrim(t *testing.T) {
	restore := saveServerGlobals(t)
	t.Cleanup(restore)

	oldNewStore := leaderboardNewStore
	var gotPath string
	leaderboardNewStore = func(path string) (*leaderboard.Store, error) {
		gotPath = path
		return oldNewStore(":memory:")
	}

	cfg := &config.Config{Storage: config.StorageConfig{Type: " SQlite ", Path: " \tfoo.db \n "}}
	lb, err := openLeaderboardStore(cfg)
	if err != nil {
		t.Fatalf("openLeaderboardStore: %v", err)
	}
	t.Cleanup(func() { _ = lb.Close() })

	if gotPath != "foo.db" {
		t.Fatalf("path: got %q want %q", gotPath, "foo.db")
	}
}

func TestOpenLeaderboardStore_Memory(t *testing.T) {
	restore := saveServerGlobals(t)
	t.Cleanup(restore)

	oldNewStore := leaderboardNewStore
	var gotPath string
	leaderboardNewStore = func(path string) (*leaderboard.Store, error) {
		gotPath = path
		return oldNewStore(":memory:")
	}

	cfg := &config.Config{Storage: config.StorageConfig{Type: "memory", Path: "ignored"}}
	lb, err := openLeaderboardStore(cfg)
	if err != nil {
		t.Fatalf("openLeaderboardStore: %v", err)
	}
	t.Cleanup(func() { _ = lb.Close() })

	if gotPath != ":memory:" {
		t.Fatalf("path: got %q want %q", gotPath, ":memory:")
	}
}

func TestOpenLeaderboardStore_UnsupportedType(t *testing.T) {
	_, err := openLeaderboardStore(&config.Config{Storage: config.StorageConfig{Type: "wat"}})
	if err == nil {
		t.Fatalf("expected error")
	}
	if !strings.Contains(err.Error(), "unsupported type") {
		t.Fatalf("error: got %q", err)
	}
}

func TestRunMain_Success(t *testing.T) {
	restore := saveServerGlobals(t)
	t.Cleanup(restore)

	stderrBuf := &bytes.Buffer{}
	stderrWriter = stderrBuf

	cfg := &config.Config{Storage: config.StorageConfig{Type: "memory"}}
	var gotConfigPath string
	loadConfig = func(path string) (*config.Config, error) {
		gotConfigPath = path
		return cfg, nil
	}

	st := &stubStore{}
	openStore = func(c *config.Config) (store.Store, error) {
		if c != cfg {
			t.Fatalf("openStore: cfg mismatch")
		}
		return st, nil
	}

	defaultProviderFromConfig = func(c *config.Config) (llm.Provider, error) {
		if c != cfg {
			t.Fatalf("providerFromConfig: cfg mismatch")
		}
		return noopProvider{}, nil
	}

	var gotAddr string
	runCalled := 0
	runServer = func(srv *api.Server, addr string) error {
		if srv == nil {
			t.Fatalf("runServer: nil server")
		}
		runCalled++
		gotAddr = addr
		return nil
	}

	newServer = func(c *config.Config, gotStore store.Store, provider llm.Provider, lb *leaderboard.Store) (*api.Server, error) {
		if c != cfg {
			t.Fatalf("newServer: cfg mismatch")
		}
		if gotStore != st {
			t.Fatalf("newServer: store mismatch")
		}
		if provider == nil {
			t.Fatalf("newServer: nil provider")
		}
		if lb == nil {
			t.Fatalf("newServer: nil leaderboard store")
		}
		return &api.Server{}, nil
	}

	code := runMain([]string{"-addr", "127.0.0.1:9999", "-config", "cfg.yaml"})
	if code != 0 {
		t.Fatalf("exit: got %d want %d; stderr=%q", code, 0, stderrBuf.String())
	}
	if gotConfigPath != "cfg.yaml" {
		t.Fatalf("configPath: got %q want %q", gotConfigPath, "cfg.yaml")
	}
	if runCalled != 1 || gotAddr != "127.0.0.1:9999" {
		t.Fatalf("Run: called=%d addr=%q", runCalled, gotAddr)
	}
	if st.closeCalled != 1 {
		t.Fatalf("store Close: called=%d", st.closeCalled)
	}
	if stderrBuf.Len() != 0 {
		t.Fatalf("stderr: got %q", stderrBuf.String())
	}
}

func TestRunMain_DefaultFlags(t *testing.T) {
	restore := saveServerGlobals(t)
	t.Cleanup(restore)

	cfg := &config.Config{Storage: config.StorageConfig{Type: "memory"}}
	var gotConfigPath string
	loadConfig = func(path string) (*config.Config, error) {
		gotConfigPath = path
		return cfg, nil
	}

	openStore = func(*config.Config) (store.Store, error) { return &stubStore{}, nil }
	defaultProviderFromConfig = func(*config.Config) (llm.Provider, error) { return noopProvider{}, nil }

	var gotAddr string
	runServer = func(_ *api.Server, addr string) error {
		gotAddr = addr
		return nil
	}
	newServer = func(*config.Config, store.Store, llm.Provider, *leaderboard.Store) (*api.Server, error) {
		return &api.Server{}, nil
	}

	if code := runMain(nil); code != 0 {
		t.Fatalf("exit: got %d want %d", code, 0)
	}
	if gotConfigPath != config.DefaultPath {
		t.Fatalf("configPath: got %q want %q", gotConfigPath, config.DefaultPath)
	}
	if gotAddr != ":8080" {
		t.Fatalf("addr: got %q want %q", gotAddr, ":8080")
	}
}

func TestRunMain_FlagParseError(t *testing.T) {
	restore := saveServerGlobals(t)
	t.Cleanup(restore)

	stderrBuf := &bytes.Buffer{}
	stderrWriter = stderrBuf

	loadCalled := 0
	loadConfig = func(string) (*config.Config, error) {
		loadCalled++
		return &config.Config{}, nil
	}

	if code := runMain([]string{"-nope"}); code != 2 {
		t.Fatalf("exit: got %d want %d", code, 2)
	}
	if loadCalled != 0 {
		t.Fatalf("Load: called=%d want %d", loadCalled, 0)
	}
	if stderrBuf.Len() == 0 {
		t.Fatalf("expected parse error output")
	}
}

func TestRunMain_HelpFlag(t *testing.T) {
	restore := saveServerGlobals(t)
	t.Cleanup(restore)

	stderrBuf := &bytes.Buffer{}
	stderrWriter = stderrBuf

	loadCalled := 0
	loadConfig = func(string) (*config.Config, error) {
		loadCalled++
		return &config.Config{}, nil
	}

	if code := runMain([]string{"-h"}); code != 0 {
		t.Fatalf("exit: got %d want %d", code, 0)
	}
	if loadCalled != 0 {
		t.Fatalf("Load: called=%d want %d", loadCalled, 0)
	}
}

func TestRunMain_ConfigLoadError(t *testing.T) {
	restore := saveServerGlobals(t)
	t.Cleanup(restore)

	stderrBuf := &bytes.Buffer{}
	stderrWriter = stderrBuf

	loadConfig = func(string) (*config.Config, error) {
		return nil, errors.New("boom")
	}
	openStore = func(*config.Config) (store.Store, error) {
		t.Fatalf("Open called unexpectedly")
		return nil, nil
	}

	if code := runMain([]string{"-config", "x.yaml"}); code != 1 {
		t.Fatalf("exit: got %d want %d", code, 1)
	}
	if !strings.Contains(stderrBuf.String(), "boom") {
		t.Fatalf("stderr: got %q", stderrBuf.String())
	}
}

func TestRunMain_StoreOpenError(t *testing.T) {
	restore := saveServerGlobals(t)
	t.Cleanup(restore)

	stderrBuf := &bytes.Buffer{}
	stderrWriter = stderrBuf

	loadConfig = func(string) (*config.Config, error) {
		return &config.Config{Storage: config.StorageConfig{Type: "memory"}}, nil
	}
	openStore = func(*config.Config) (store.Store, error) {
		return nil, errors.New("storefail")
	}

	if code := runMain(nil); code != 1 {
		t.Fatalf("exit: got %d want %d", code, 1)
	}
	if !strings.Contains(stderrBuf.String(), "storefail") {
		t.Fatalf("stderr: got %q", stderrBuf.String())
	}
}

func TestRunMain_LeaderboardOpenError_ClosesStore(t *testing.T) {
	restore := saveServerGlobals(t)
	t.Cleanup(restore)

	stderrBuf := &bytes.Buffer{}
	stderrWriter = stderrBuf

	loadConfig = func(string) (*config.Config, error) {
		return &config.Config{Storage: config.StorageConfig{Type: "wat"}}, nil
	}

	st := &stubStore{}
	openStore = func(*config.Config) (store.Store, error) { return st, nil }

	if code := runMain(nil); code != 1 {
		t.Fatalf("exit: got %d want %d", code, 1)
	}
	if st.closeCalled != 1 {
		t.Fatalf("store Close: called=%d", st.closeCalled)
	}
	if !strings.Contains(stderrBuf.String(), "unsupported type") {
		t.Fatalf("stderr: got %q", stderrBuf.String())
	}
}

func TestRunMain_ProviderError_ClosesStore(t *testing.T) {
	restore := saveServerGlobals(t)
	t.Cleanup(restore)

	stderrBuf := &bytes.Buffer{}
	stderrWriter = stderrBuf

	loadConfig = func(string) (*config.Config, error) {
		return &config.Config{Storage: config.StorageConfig{Type: "memory"}}, nil
	}

	st := &stubStore{}
	openStore = func(*config.Config) (store.Store, error) { return st, nil }

	defaultProviderFromConfig = func(*config.Config) (llm.Provider, error) {
		return nil, errors.New("provfail")
	}

	if code := runMain(nil); code != 1 {
		t.Fatalf("exit: got %d want %d", code, 1)
	}
	if st.closeCalled != 1 {
		t.Fatalf("store Close: called=%d", st.closeCalled)
	}
	if !strings.Contains(stderrBuf.String(), "provfail") {
		t.Fatalf("stderr: got %q", stderrBuf.String())
	}
}

func TestRunMain_NewServerError_ClosesStore(t *testing.T) {
	restore := saveServerGlobals(t)
	t.Cleanup(restore)

	stderrBuf := &bytes.Buffer{}
	stderrWriter = stderrBuf

	loadConfig = func(string) (*config.Config, error) {
		return &config.Config{Storage: config.StorageConfig{Type: "memory"}}, nil
	}

	st := &stubStore{}
	openStore = func(*config.Config) (store.Store, error) { return st, nil }
	defaultProviderFromConfig = func(*config.Config) (llm.Provider, error) { return noopProvider{}, nil }
	newServer = func(*config.Config, store.Store, llm.Provider, *leaderboard.Store) (*api.Server, error) {
		return nil, errors.New("srvfail")
	}

	if code := runMain(nil); code != 1 {
		t.Fatalf("exit: got %d want %d", code, 1)
	}
	if st.closeCalled != 1 {
		t.Fatalf("store Close: called=%d", st.closeCalled)
	}
	if !strings.Contains(stderrBuf.String(), "srvfail") {
		t.Fatalf("stderr: got %q", stderrBuf.String())
	}
}

func TestRunMain_RunError_ClosesStore(t *testing.T) {
	restore := saveServerGlobals(t)
	t.Cleanup(restore)

	stderrBuf := &bytes.Buffer{}
	stderrWriter = stderrBuf

	loadConfig = func(string) (*config.Config, error) {
		return &config.Config{Storage: config.StorageConfig{Type: "memory"}}, nil
	}

	st := &stubStore{}
	openStore = func(*config.Config) (store.Store, error) { return st, nil }
	defaultProviderFromConfig = func(*config.Config) (llm.Provider, error) { return noopProvider{}, nil }

	runServer = func(*api.Server, string) error { return errors.New("runfail") }
	newServer = func(*config.Config, store.Store, llm.Provider, *leaderboard.Store) (*api.Server, error) {
		return &api.Server{}, nil
	}

	if code := runMain(nil); code != 1 {
		t.Fatalf("exit: got %d want %d", code, 1)
	}
	if st.closeCalled != 1 {
		t.Fatalf("store Close: called=%d", st.closeCalled)
	}
	if !strings.Contains(stderrBuf.String(), "runfail") {
		t.Fatalf("stderr: got %q", stderrBuf.String())
	}
}

func TestMain_ExitCodePropagates(t *testing.T) {
	restore := saveServerGlobals(t)
	t.Cleanup(restore)

	stderrWriter = &bytes.Buffer{}

	cfg := &config.Config{Storage: config.StorageConfig{Type: "memory"}}
	loadConfig = func(string) (*config.Config, error) { return cfg, nil }
	openStore = func(*config.Config) (store.Store, error) { return &stubStore{}, nil }
	defaultProviderFromConfig = func(*config.Config) (llm.Provider, error) { return noopProvider{}, nil }
	newServer = func(*config.Config, store.Store, llm.Provider, *leaderboard.Store) (*api.Server, error) {
		return &api.Server{}, nil
	}
	runServer = func(*api.Server, string) error { return nil }

	oldArgs := osArgsForTest()
	t.Cleanup(func() { setOsArgsForTest(oldArgs) })
	setOsArgsForTest([]string{"server", "-addr", "127.0.0.1:9999"})

	exitCode := -1
	osExit = func(code int) { exitCode = code }

	main()

	if exitCode != 0 {
		t.Fatalf("exit: got %d want %d", exitCode, 0)
	}
}

func osArgsForTest() []string {
	return append([]string(nil), os.Args...)
}

func setOsArgsForTest(args []string) {
	os.Args = args
}
