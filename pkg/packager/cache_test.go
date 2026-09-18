package packager

import (
	"context"
	"reflect"
	"testing"

	"github.com/moby/buildkit/frontend/gateway/client"
)

func TestPackagersImportRemoteCache(t *testing.T) {
	for name, build := range map[string]client.BuildFunc{"modelpack": BuildModelpack, "generic": BuildGeneric} {
		t.Run(name, func(t *testing.T) {
			c := &cacheRecordingClient{opts: map[string]string{
				"build-arg:source": "model.gguf",
				"cache-imports":    "[{\"Type\":\"registry\",\"Attrs\":{\"ref\":\"example.com/models/cache:build\"}}]",
			}}
			if _, err := build(context.Background(), c); err != nil {
				t.Fatal(err)
			}
			want := []client.CacheOptionsEntry{{Type: "registry", Attrs: map[string]string{"ref": "example.com/models/cache:build"}}}
			if !reflect.DeepEqual(c.imports, want) {
				t.Fatalf("cache imports = %#v, want %#v", c.imports, want)
			}
		})
	}
}

func TestPackagerRejectsInvalidRemoteCacheOptions(t *testing.T) {
	c := &cacheRecordingClient{opts: map[string]string{"build-arg:source": "model.gguf", "cache-imports": "invalid JSON"}}
	if _, err := BuildModelpack(context.Background(), c); err == nil {
		t.Fatal("invalid cache-imports was silently ignored")
	}
	if c.solved {
		t.Fatal("packaging started with invalid cache imports")
	}
}

type cacheRecordingClient struct {
	client.Client
	opts    map[string]string
	imports []client.CacheOptionsEntry
	solved  bool
}

func (c *cacheRecordingClient) BuildOpts() client.BuildOpts {
	return client.BuildOpts{Opts: c.opts}
}

func (c *cacheRecordingClient) Solve(_ context.Context, request client.SolveRequest) (*client.Result, error) {
	c.imports = request.CacheImports
	c.solved = true
	result := client.NewResult()
	result.SetRef(&cacheRecordingReference{})
	return result, nil
}

type cacheRecordingReference struct{ client.Reference }
