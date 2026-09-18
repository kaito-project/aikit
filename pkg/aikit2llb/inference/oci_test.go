package inference

import (
	"context"
	"encoding/json"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/moby/buildkit/client/llb"
	"github.com/moby/buildkit/frontend/gateway/client"
	"github.com/moby/buildkit/solver/pb"
	digest "github.com/opencontainers/go-digest"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/pkg/errors"
	"golang.org/x/sync/errgroup"
)

const testOCIRepository = "example.com/models/test"

func TestNativeOCIModelFiles(t *testing.T) {
	weight := testOCILayer("weights/model.gguf", "weight bytes")
	config := testOCILayer("tokenizer.json", "tokenizer bytes")
	unnamed := testOCILayer("", "unnamed bytes")
	c := newOCIManifestClient()
	canonical := c.addManifest(testOCIRepository, testOCIJSON(t, specs.MediaTypeImageManifest, "layers", []specs.Descriptor{weight, config, unnamed}))
	c.tags[testOCIRepository+":latest"] = canonical
	imports := []client.CacheOptionsEntry{{Type: "registry", Attrs: map[string]string{"ref": "example.com/models/cache:build"}}}
	resolver := NewOCIResolver(c, specs.Platform{OS: "linux", Architecture: "amd64"}, imports)
	state, err := resolver.Resolve(context.Background(), "oci://"+testOCIRepository+":latest", specs.Platform{OS: "linux", Architecture: "arm64"})
	if err != nil {
		t.Fatal(err)
	}
	definition := marshalOCIState(t, state)
	assertOCIBlobSources(t, definition, testOCIRepository+"@"+weight.Digest.String(), testOCIRepository+"@"+config.Digest.String())
	for _, filename := range []string{"/weights/model.gguf", "/tokenizer.json"} {
		found := false
		for _, op := range decodeInferenceDefinition(t, definition) {
			if file := op.op.GetFile(); file != nil {
				for _, action := range file.Actions {
					if fileCopy := action.GetCopy(); fileCopy != nil && fileCopy.Dest == filename {
						found = true
					}
				}
			}
		}
		if !found {
			t.Errorf("model filename %s was not preserved", filename)
		}
	}
	requests := c.solveRequests()
	if len(requests) != 1 || !reflect.DeepEqual(requests[0].CacheImports, imports) {
		t.Fatalf("manifest solves = %#v, want one with remote cache imports", requests)
	}
	uncached := false
	for _, metadata := range requests[0].Definition.Metadata {
		uncached = uncached || metadata.IgnoreCache
	}
	if !uncached {
		t.Fatal("mutable tag lookup must be refreshed on each build")
	}
}

func TestNativeOCIDigestReferenceSkipsTagLookup(t *testing.T) {
	c := newOCIManifestClient()
	layer := testOCILayer("model.gguf", "model bytes")
	canonical := c.addManifest(testOCIRepository, testOCIJSON(t, specs.MediaTypeImageManifest, "layers", []specs.Descriptor{layer}))
	resolver := NewOCIResolver(c, specs.Platform{OS: "linux", Architecture: "amd64"}, nil)
	state, err := resolver.Resolve(context.Background(), "oci://"+canonical, specs.Platform{OS: "linux", Architecture: "arm64"})
	if err != nil {
		t.Fatal(err)
	}
	assertOCIBlobSources(t, marshalOCIState(t, state), testOCIRepository+"@"+layer.Digest.String())
	requests := c.solveRequests()
	if len(requests) != 1 {
		t.Fatalf("manifest solves = %d, want one", len(requests))
	}
	for _, data := range requests[0].Definition.Def {
		var op pb.Op
		if err := op.Unmarshal(data); err != nil {
			t.Fatal(err)
		}
		if exec := op.GetExec(); exec != nil && strings.Contains(exec.Meta.Args[2], "oras resolve") {
			t.Fatal("digest-pinned manifest resolved a tag")
		}
	}
	for _, metadata := range requests[0].Definition.Metadata {
		if metadata.IgnoreCache {
			t.Fatal("immutable manifest lookup disabled caching")
		}
	}
}

func TestNativeOCIMultiPlatformSharesManifestLookups(t *testing.T) {
	c := newOCIManifestClient()
	amd64 := specs.Platform{OS: "linux", Architecture: "amd64"}
	arm64 := specs.Platform{OS: "linux", Architecture: "arm64"}
	amdLayer := testOCILayer("model.gguf", "amd64 bytes")
	armLayer := testOCILayer("model.gguf", "arm64 bytes")
	amdManifest := testOCIJSON(t, specs.MediaTypeImageManifest, "layers", []specs.Descriptor{amdLayer})
	armManifest := testOCIJSON(t, specs.MediaTypeImageManifest, "layers", []specs.Descriptor{armLayer})
	c.addManifest(testOCIRepository, amdManifest)
	c.addManifest(testOCIRepository, armManifest)
	index := testOCIJSON(t, specs.MediaTypeImageIndex, "manifests", []specs.Descriptor{
		testOCIManifestDescriptor(amdManifest, &amd64),
		testOCIManifestDescriptor(armManifest, &arm64),
		{Digest: digest.FromString("attestation"), Platform: &specs.Platform{OS: "unknown", Architecture: "unknown"}},
	})
	c.tags[testOCIRepository+":latest"] = c.addManifest(testOCIRepository, index)
	resolver := NewOCIResolver(c, amd64, nil)
	var group errgroup.Group
	for i := 0; i < 20; i++ {
		group.Go(func() error {
			target, layer := amd64, amdLayer
			if i%2 != 0 {
				target, layer = arm64, armLayer
			}
			state, err := resolver.Resolve(context.Background(), "oci://"+testOCIRepository+":latest", target)
			if err != nil {
				return err
			}
			definition, err := state.Marshal(context.Background())
			if err != nil {
				return err
			}
			if got := ociBlobSources(definition); !reflect.DeepEqual(got, []string{testOCIRepository + "@" + layer.Digest.String()}) {
				return errors.Errorf("%s model blobs = %v", target.Architecture, got)
			}
			return nil
		})
	}
	if err := group.Wait(); err != nil {
		t.Fatal(err)
	}
	if got := len(c.solveRequests()); got != 3 {
		t.Fatalf("manifest solves = %d, want one shared index and two platform manifests", got)
	}
}

func TestNativeOCIMutableTagRefreshesAcrossBuilds(t *testing.T) {
	c := newOCIManifestClient()
	platform := specs.Platform{OS: "linux", Architecture: "amd64"}
	for _, content := range []string{"first model", "replacement model"} {
		layer := testOCILayer("model.gguf", content)
		data := testOCIJSON(t, specs.MediaTypeImageManifest, "layers", []specs.Descriptor{layer})
		c.tags[testOCIRepository+":latest"] = c.addManifest(testOCIRepository, data)
		resolver := NewOCIResolver(c, platform, nil)
		state, err := resolver.Resolve(context.Background(), "oci://"+testOCIRepository+":latest", platform)
		if err != nil {
			t.Fatal(err)
		}
		assertOCIBlobSources(t, marshalOCIState(t, state), testOCIRepository+"@"+layer.Digest.String())
	}
	if got := len(c.solveRequests()); got != 2 {
		t.Fatalf("tag lookups = %d, want one per build", got)
	}
}

func TestSelectOCIManifest(t *testing.T) {
	amd64 := specs.Platform{OS: "linux", Architecture: "amd64"}
	arm64 := specs.Platform{OS: "linux", Architecture: "arm64"}
	amd := testOCIManifestDescriptor([]byte("amd64 manifest"), &amd64)
	arm := testOCIManifestDescriptor([]byte("arm64 manifest"), &arm64)
	neutral := testOCIManifestDescriptor([]byte("neutral manifest"), nil)
	attestation := testOCIManifestDescriptor([]byte("attestation"), nil)
	attestation.Annotations = map[string]string{"vnd.docker.reference.type": "attestation-manifest"}
	tests := []struct {
		name      string
		manifests []specs.Descriptor
		target    specs.Platform
		want      digest.Digest
		wantError string
	}{
		{name: "neutral index", manifests: []specs.Descriptor{neutral}, target: arm64, want: neutral.Digest},
		{name: "single amd64 stamp stays neutral", manifests: []specs.Descriptor{amd}, target: arm64, want: amd.Digest},
		{name: "attestation does not add a platform", manifests: []specs.Descriptor{attestation, amd}, target: arm64, want: amd.Digest},
		{name: "two platforms select arm64", manifests: []specs.Descriptor{amd, attestation, arm}, target: arm64, want: arm.Digest},
		{name: "two platforms select amd64", manifests: []specs.Descriptor{arm, amd}, target: amd64, want: amd.Digest},
		{name: "missing target", manifests: []specs.Descriptor{amd, arm}, target: specs.Platform{OS: "linux", Architecture: "riscv64"}, wantError: "no manifest for"},
		{name: "empty index", target: arm64, wantError: "no model manifests"},
		{name: "only attestations", manifests: []specs.Descriptor{attestation}, target: arm64, wantError: "no model manifests"},
		{name: "invalid digest", manifests: []specs.Descriptor{{Digest: "sha256:bad"}}, target: arm64, wantError: "invalid OCI model manifest digest"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := selectOCIManifest(specs.Index{Manifests: tt.manifests}, tt.target)
			if tt.wantError != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantError) {
					t.Fatalf("error = %v, want %q", err, tt.wantError)
				}
				return
			}
			if err != nil || got.Digest != tt.want {
				t.Fatalf("selected %s, error %v, want %s", got.Digest, err, tt.want)
			}
		})
	}
}

func TestNativeOCIOllamaSelectsOnlyModelLayer(t *testing.T) {
	c := newOCIManifestClient()
	repository := "registry.ollama.ai/library/model"
	layer := testOCILayer("", "ollama model bytes")
	layer.MediaType = ollamaModelType
	other := testOCILayer("template", "template bytes")
	canonical := c.addManifest(repository, testOCIJSON(t, specs.MediaTypeImageManifest, "layers", []specs.Descriptor{other, layer}))
	resolver := NewOCIResolver(c, specs.Platform{OS: "linux", Architecture: "amd64"}, nil)
	state, err := resolver.Resolve(context.Background(), "oci://"+canonical, specs.Platform{OS: "linux", Architecture: "arm64"})
	if err != nil {
		t.Fatal(err)
	}
	assertOCIBlobSources(t, marshalOCIState(t, state), repository+"@"+layer.Digest.String())
}

func TestNativeOCIRejectsUnsafeMetadata(t *testing.T) {
	for _, name := range []string{"../escape", "/absolute", ".", "dir/../../escape", "dir\\escape", "nul\x00file"} {
		t.Run(name, func(t *testing.T) {
			c := newOCIManifestClient()
			layer := testOCILayer(name, "bytes")
			canonical := c.addManifest(testOCIRepository, testOCIJSON(t, specs.MediaTypeImageManifest, "layers", []specs.Descriptor{layer}))
			resolver := NewOCIResolver(c, specs.Platform{OS: "linux", Architecture: "amd64"}, nil)
			if _, err := resolver.Resolve(context.Background(), "oci://"+canonical, specs.Platform{}); err == nil || !strings.Contains(err.Error(), "unsafe OCI model filename") {
				t.Fatalf("error = %v, want unsafe filename", err)
			}
		})
	}
	t.Run("manifest digest mismatch", func(t *testing.T) {
		c := newOCIManifestClient()
		canonical := c.addManifest(testOCIRepository, []byte("original bytes"))
		c.manifests[canonical] = []byte("tampered bytes")
		resolver := NewOCIResolver(c, specs.Platform{OS: "linux", Architecture: "amd64"}, nil)
		if _, err := resolver.Resolve(context.Background(), "oci://"+canonical, specs.Platform{}); err == nil || !strings.Contains(err.Error(), "digest mismatch") {
			t.Fatalf("error = %v, want manifest digest mismatch", err)
		}
	})
}

func TestNativeOCIUnavailableOnOlderBuilders(t *testing.T) {
	c := newOCIManifestClient()
	c.opts = client.BuildOpts{}
	if resolver := NewOCIResolver(c, specs.Platform{OS: "linux", Architecture: "amd64"}, nil); resolver != nil {
		t.Fatal("native OCI resolver enabled without blob source support")
	}
}

type ociManifestClient struct {
	client.Client
	opts      client.BuildOpts
	manifests map[string][]byte
	tags      map[string]string
	mu        sync.Mutex
	requests  []client.SolveRequest
}

func newOCIManifestClient() *ociManifestClient {
	return &ociManifestClient{
		opts:      client.BuildOpts{LLBCaps: pb.Caps.CapSet(pb.Caps.All())},
		manifests: make(map[string][]byte),
		tags:      make(map[string]string),
	}
}

func (c *ociManifestClient) BuildOpts() client.BuildOpts { return c.opts }

func (c *ociManifestClient) addManifest(repository string, data []byte) string {
	canonical := repository + "@" + digest.FromBytes(data).String()
	c.manifests[canonical] = data
	return canonical
}

func (c *ociManifestClient) Solve(_ context.Context, request client.SolveRequest) (*client.Result, error) {
	c.mu.Lock()
	c.requests = append(c.requests, request.Clone())
	c.mu.Unlock()
	canonical := ""
	for _, data := range request.Definition.Def {
		var op pb.Op
		if err := op.Unmarshal(data); err != nil {
			return nil, err
		}
		if source := op.GetSource(); source != nil && strings.HasPrefix(source.Identifier, "docker-image+blob://") {
			canonical = strings.TrimPrefix(source.Identifier, "docker-image+blob://")
		}
		if exec := op.GetExec(); exec != nil && len(exec.Meta.Args) >= 5 && exec.Meta.Args[3] == "aikit-oci-manifest" {
			canonical = exec.Meta.Args[4]
			if resolved, ok := c.tags[canonical]; ok {
				canonical = resolved
			}
		}
	}
	data, ok := c.manifests[canonical]
	if !ok {
		return nil, errors.Errorf("unexpected manifest lookup %q", canonical)
	}
	result := client.NewResult()
	result.SetRef(&ociManifestReference{files: map[string][]byte{ociManifestFilename: data, ociResolvedRefPath: []byte(canonical)}})
	return result, nil
}

func (c *ociManifestClient) solveRequests() []client.SolveRequest {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]client.SolveRequest(nil), c.requests...)
}

type ociManifestReference struct {
	client.Reference
	files map[string][]byte
}

func (r *ociManifestReference) ReadFile(_ context.Context, request client.ReadRequest) ([]byte, error) {
	data, ok := r.files[request.Filename]
	if !ok {
		return nil, errors.Errorf("unexpected metadata file %q", request.Filename)
	}
	if request.Range != nil && len(data) > request.Range.Length {
		data = data[:request.Range.Length]
	}
	return data, nil
}

func testOCILayer(name, content string) specs.Descriptor {
	return specs.Descriptor{
		MediaType:   "application/vnd.cncf.model.weight.v1.raw",
		Digest:      digest.FromString(content),
		Size:        int64(len(content)),
		Annotations: map[string]string{specs.AnnotationTitle: name},
	}
}

func testOCIManifestDescriptor(data []byte, platform *specs.Platform) specs.Descriptor {
	return specs.Descriptor{MediaType: specs.MediaTypeImageManifest, Digest: digest.FromBytes(data), Size: int64(len(data)), Platform: platform}
}

func testOCIJSON(t *testing.T, mediaType, field string, descriptors []specs.Descriptor) []byte {
	t.Helper()
	data, err := json.Marshal(map[string]any{"schemaVersion": 2, "mediaType": mediaType, field: descriptors})
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func marshalOCIState(t *testing.T, state llb.State) *llb.Definition {
	t.Helper()
	definition, err := state.Marshal(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	return definition
}

func ociBlobSources(definition *llb.Definition) []string {
	var sources []string
	for _, data := range definition.Def {
		var op pb.Op
		if err := op.Unmarshal(data); err != nil {
			return nil
		}
		if source := op.GetSource(); source != nil && strings.HasPrefix(source.Identifier, "docker-image+blob://") {
			sources = append(sources, strings.TrimPrefix(source.Identifier, "docker-image+blob://"))
		}
	}
	return sources
}

func assertOCIBlobSources(t *testing.T, definition *llb.Definition, want ...string) {
	t.Helper()
	if got := ociBlobSources(definition); !reflect.DeepEqual(got, want) {
		t.Fatalf("native blob sources = %v, want %v", got, want)
	}
}
