package llm

import "strings"

type Registry struct {
	providers map[string]Provider
}

func NewRegistry() *Registry {
	return &Registry{
		providers: make(map[string]Provider),
	}
}

func (r *Registry) Register(p Provider) {
	if r == nil || p == nil {
		return
	}
	name := strings.ToLower(strings.TrimSpace(p.Name()))
	if name == "" {
		return
	}
	if r.providers == nil {
		r.providers = make(map[string]Provider)
	}
	r.providers[name] = p
}

func (r *Registry) Get(name string) (Provider, bool) {
	if r == nil || r.providers == nil {
		return nil, false
	}
	name = strings.ToLower(strings.TrimSpace(name))
	if name == "" {
		return nil, false
	}
	p, ok := r.providers[name]
	return p, ok
}
