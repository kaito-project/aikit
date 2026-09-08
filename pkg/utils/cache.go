package utils

import (
	"encoding/json"

	"github.com/moby/buildkit/frontend/gateway/client"
	"github.com/pkg/errors"
)

// ParseCacheImports reads the remote caches supplied by buildx to the frontend.
func ParseCacheImports(opts map[string]string) ([]client.CacheOptionsEntry, error) {
	var entries []client.CacheOptionsEntry
	if value := opts["cache-imports"]; value != "" {
		if err := json.Unmarshal([]byte(value), &entries); err != nil {
			return nil, errors.Wrap(err, "invalid cache-imports")
		}
	}
	return entries, nil
}
