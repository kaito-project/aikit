package packager

import (
	"archive/tar"
	"context"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"testing"

	v1 "github.com/modelpack/model-spec/specs-go/v1"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
)

// helper: read and unmarshal JSON file
func readJSON[T any](t *testing.T, path string) T {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	var v T
	if err := json.Unmarshal(b, &v); err != nil {
		t.Fatalf("unmarshal %s: %v", path, err)
	}
	return v
}

func Test_safeRefName(t *testing.T) {
	cases := map[string]string{
		"":                      "aikit/model",
		"My Model 1":            "my-model-1",
		"Weird!Name@With#Chars": "weird-name-with-chars",
		"Already/good_name.ok":  "already/good_name.ok",
		"UPPERCASE.and.Mixed":   "uppercase.and.mixed",
	}
	for in, want := range cases {
		if got := safeRefName(in); got != want {
			t.Errorf("safeRefName(%q)=%q want %q", in, got, want)
		}
	}
}

func Test_determineRefName(t *testing.T) {
	cases := []struct {
		opts map[string]string
		want string
	}{
		{map[string]string{"build-arg:name": "modelA"}, "modelA"},
		{map[string]string{}, "latest"},
		{nil, "latest"},
	}
	for i, c := range cases {
		if got := determineRefName(c.opts); got != c.want {
			t.Fatalf("case %d: got %q want %q", i, got, c.want)
		}
	}
}

func Test_determineName(t *testing.T) {
	cases := []struct {
		opts map[string]string
		want string
	}{
		{map[string]string{"build-arg:name": "foo"}, "foo"},
		{map[string]string{}, "aikitmodel"},
		{nil, "aikitmodel"},
	}
	for i, c := range cases {
		if got := determineName(c.opts); got != c.want {
			t.Fatalf("case %d: got %q want %q", i, got, c.want)
		}
	}
}

func Test_hasAnySuffix(t *testing.T) {
	if !hasAnySuffix("file.bin", ".bin", ".pt") {
		t.Fatal("expected true")
	}
	if hasAnySuffix("file.txt", ".bin", ".pt") {
		t.Fatal("expected false")
	}
}

func makeFile(t *testing.T, dir, name string, size int) string {
	t.Helper()
	p := filepath.Join(dir, name)
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	f, err := os.Create(p)
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	if size > 0 {
		if _, err := f.Write(make([]byte, size)); err != nil {
			t.Fatalf("write: %v", err)
		}
	}
	if err := f.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	return p
}

func Test_pack_ModelPack_DefaultsAndPartitioning(t *testing.T) {
	ctx := context.Background()
	src := t.TempDir()

	// Create files to exercise partitioning logic
	makeFile(t, src, "model.safetensors", 1024)   // weights
	makeFile(t, src, "config.json", 200)          // config
	makeFile(t, src, "README.md", 100)            // docs
	makeFile(t, src, "misc.xyz", 50)              // small unknown -> config
	makeFile(t, src, "bigfile.dat", 11*1024*1024) // large unknown -> weights

	out := t.TempDir()
	res, err := pack(ctx, Options{Source: src, OutputDir: out, Spec: SpecModelPack, Name: "My Model 1"})
	if err != nil {
		t.Fatalf("pack: %v", err)
	}
	if res.LayoutPath != out {
		t.Fatalf("unexpected layout path")
	}

	// index.json
	idx := readJSON[ocispec.Index](t, filepath.Join(out, "index.json"))
	if len(idx.Manifests) != 1 {
		t.Fatalf("expected 1 manifest, got %d", len(idx.Manifests))
	}
	if idx.Annotations[ocispec.AnnotationRefName] != safeRefName("My Model 1") {
		t.Fatalf("ref name mismatch: %s", idx.Annotations[ocispec.AnnotationRefName])
	}

	// manifest
	manifestDigest := idx.Manifests[0].Digest.Encoded()
	manifestPath := filepath.Join(out, "blobs", "sha256", manifestDigest)
	m := readJSON[ocispec.Manifest](t, manifestPath)
	if m.ArtifactType != v1.ArtifactTypeModelManifest {
		t.Fatalf("artifactType=%s", m.ArtifactType)
	}
	if len(m.Layers) != 3 { // weights, config, docs
		t.Fatalf("expected 3 layers, got %d", len(m.Layers))
	}
	// Check media types ordering (weights, config, docs) defaults
	if m.Layers[0].MediaType != v1.MediaTypeModelWeight {
		t.Fatalf("weights layer media type mismatch: %s", m.Layers[0].MediaType)
	}
	if m.Layers[1].MediaType != v1.MediaTypeModelWeightConfig {
		t.Fatalf("config layer media type mismatch: %s", m.Layers[1].MediaType)
	}
	if m.Layers[2].MediaType != v1.MediaTypeModelDoc {
		t.Fatalf("docs layer media type mismatch: %s", m.Layers[2].MediaType)
	}

	// Inspect first layer tar ordering deterministic
	firstLayer := filepath.Join(out, "blobs", "sha256", m.Layers[0].Digest.Encoded())
	f, err := os.Open(firstLayer)
	if err != nil {
		t.Fatalf("open layer: %v", err)
	}
	tr := tar.NewReader(f)
	var names []string
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("tar read: %v", err)
		}
		names = append(names, hdr.Name)
		// sanity on header normalization
		if hdr.Uid != 0 || hdr.Gid != 0 || hdr.Mode != 0o644 {
			t.Fatalf("unexpected header fields")
		}
	}
	_ = f.Close()
	// names should be sorted (weights list sorted)
	for i := 1; i < len(names); i++ {
		if names[i-1] > names[i] {
			t.Fatalf("tar entries not sorted: %v", names)
		}
	}
}

func Test_pack_ModelPack_MediaTypeOverrides(t *testing.T) {
	ctx := context.Background()
	src := t.TempDir()
	makeFile(t, src, "a.bin", 1)
	makeFile(t, src, "config.json", 1)

	out := t.TempDir()
	mts := ModelPackMediaTypes{LayerWeights: "custom/w", LayerConfig: "custom/c", LayerDocs: "custom/d", ManifestConfig: "custom/mcfg"}
	res, err := pack(ctx, Options{Source: src, OutputDir: out, Spec: SpecModelPack, MediaTypes: mts})
	if err != nil {
		t.Fatalf("pack: %v", err)
	}
	idx := readJSON[ocispec.Index](t, filepath.Join(res.LayoutPath, "index.json"))
	m := readJSON[ocispec.Manifest](t, filepath.Join(res.LayoutPath, "blobs", "sha256", idx.Manifests[0].Digest.Encoded()))
	if len(m.Layers) != 2 {
		t.Fatalf("expected 2 layers")
	}
	if m.Layers[0].MediaType != "custom/w" {
		t.Fatalf("override failed for weights")
	}
	if m.Layers[1].MediaType != "custom/c" {
		t.Fatalf("override failed for config")
	}
	if m.Config.MediaType != "custom/mcfg" {
		t.Fatalf("manifest config override failed")
	}
}

func Test_pack_Generic(t *testing.T) {
	ctx := context.Background()
	src := t.TempDir()
	makeFile(t, src, "f1.txt", 10)
	makeFile(t, src, "f2.txt", 5)

	out := t.TempDir()
	res, err := pack(ctx, Options{Source: src, OutputDir: out, Spec: SpecGeneric, Name: "Generic Artifact"})
	if err != nil {
		t.Fatalf("pack: %v", err)
	}
	idx := readJSON[ocispec.Index](t, filepath.Join(res.LayoutPath, "index.json"))
	m := readJSON[ocispec.Manifest](t, filepath.Join(res.LayoutPath, "blobs", "sha256", idx.Manifests[0].Digest.Encoded()))
	if len(m.Layers) != 1 {
		t.Fatalf("expected 1 layer for generic")
	}
	if m.Layers[0].MediaType != ocispec.MediaTypeImageLayer {
		t.Fatalf("unexpected media type: %s", m.Layers[0].MediaType)
	}
}

func Test_pack_DeterministicLayerDigest(t *testing.T) {
	ctx := context.Background()
	src := t.TempDir()
	makeFile(t, src, "a.safetensors", 10)
	makeFile(t, src, "config.json", 5)

	out1 := t.TempDir()
	out2 := t.TempDir()
	res1, err := pack(ctx, Options{Source: src, OutputDir: out1, Spec: SpecModelPack})
	if err != nil {
		t.Fatalf("pack1: %v", err)
	}
	res2, err := pack(ctx, Options{Source: src, OutputDir: out2, Spec: SpecModelPack})
	if err != nil {
		t.Fatalf("pack2: %v", err)
	}
	idx1 := readJSON[ocispec.Index](t, filepath.Join(res1.LayoutPath, "index.json"))
	idx2 := readJSON[ocispec.Index](t, filepath.Join(res2.LayoutPath, "index.json"))
	m1 := readJSON[ocispec.Manifest](t, filepath.Join(res1.LayoutPath, "blobs", "sha256", idx1.Manifests[0].Digest.Encoded()))
	m2 := readJSON[ocispec.Manifest](t, filepath.Join(res2.LayoutPath, "blobs", "sha256", idx2.Manifests[0].Digest.Encoded()))
	if len(m1.Layers) != len(m2.Layers) {
		t.Fatalf("layer count mismatch")
	}
	for i := range m1.Layers {
		if m1.Layers[i].Digest != m2.Layers[i].Digest {
			t.Fatalf("layer digest mismatch index %d: %s vs %s", i, m1.Layers[i].Digest, m2.Layers[i].Digest)
		}
	}
}
