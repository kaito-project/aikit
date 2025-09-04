package packager

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"strings"

	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
)

func TestPackModelPack_LocalDir(t *testing.T) {
	dir := t.TempDir()
	// create some fake files
	mustWrite(t, filepath.Join(dir, "config.json"), "{}")
	mustWrite(t, filepath.Join(dir, "README.md"), "hello")
	mustWrite(t, filepath.Join(dir, "weights.safetensors"), "0123456789")

	out := filepath.Join(t.TempDir(), "layout")
	res, err := Pack(context.Background(), Options{Source: dir, OutputDir: out, Spec: SpecModelPack, Name: "test/model"})
	if err != nil {
		t.Fatalf("pack: %v", err)
	}
	if res.LayoutPath != out {
		t.Fatalf("unexpected layout path: %s", res.LayoutPath)
	}
	// check index.json exists and parse
	raw, err := os.ReadFile(filepath.Join(out, "index.json"))
	if err != nil {
		t.Fatalf("read index: %v", err)
	}
	var idx ocispec.Index
	if err := json.Unmarshal(raw, &idx); err != nil {
		t.Fatalf("unmarshal index: %v", err)
	}
	if len(idx.Manifests) != 1 {
		t.Fatalf("expected 1 manifest, got %d", len(idx.Manifests))
	}
}

func TestPackGeneric_LocalDir(t *testing.T) {
	dir := t.TempDir()
	mustWrite(t, filepath.Join(dir, "a.txt"), "data")
	out := filepath.Join(t.TempDir(), "layout")
	if _, err := Pack(context.Background(), Options{Source: dir, OutputDir: out, Spec: SpecGeneric}); err != nil {
		t.Fatalf("pack: %v", err)
	}
	if _, err := os.Stat(filepath.Join(out, "index.json")); err != nil {
		t.Fatalf("index: %v", err)
	}
}

func TestPackModelPack_CustomMediaTypes(t *testing.T) {
	dir := t.TempDir()
	mustWrite(t, filepath.Join(dir, "config.json"), "{}");
	mustWrite(t, filepath.Join(dir, "README.md"), "hi");
	mustWrite(t, filepath.Join(dir, "weights.safetensors"), "0123456789");

	out := filepath.Join(t.TempDir(), "layout")
	opts := Options{
		Source: dir,
		OutputDir: out,
		Spec: SpecModelPack,
		ArtifactType: "application/x.custom.model+json",
		MediaTypes: ModelPackMediaTypes{
			ManifestConfig: "application/x.custom.config+json",
			LayerWeights:   "application/x.custom.weights.tar",
			LayerConfig:    "application/x.custom.cfg.tar",
			LayerDocs:      "application/x.custom.docs.tar",
		},
	}
	if _, err := Pack(context.Background(), opts); err != nil { t.Fatalf("pack: %v", err) }

	// Read index
	raw, err := os.ReadFile(filepath.Join(out, "index.json"))
	if err != nil { t.Fatalf("read index: %v", err) }
	var idx ocispec.Index
	if err := json.Unmarshal(raw, &idx); err != nil { t.Fatalf("unmarshal index: %v", err) }
	if len(idx.Manifests) != 1 { t.Fatalf("expected 1 manifest") }
	manDesc := idx.Manifests[0]
	if manDesc.MediaType != ocispec.MediaTypeImageManifest { t.Fatalf("unexpected manifest media type: %s", manDesc.MediaType) }

	// Load manifest blob
	mblob, err := os.ReadFile(filepath.Join(out, "blobs", "sha256", trimSha256Prefix(manDesc.Digest.String())))
	if err != nil { t.Fatalf("read manifest blob: %v", err) }
	var manifest ocispec.Manifest
	if err := json.Unmarshal(mblob, &manifest); err != nil { t.Fatalf("unmarshal manifest: %v", err) }
	if manifest.ArtifactType != opts.ArtifactType { t.Fatalf("artifactType mismatch: %s", manifest.ArtifactType) }
	if manifest.Config.MediaType != opts.MediaTypes.ManifestConfig { t.Fatalf("config media type mismatch: %s", manifest.Config.MediaType) }
	// Collect layer media types
	mts := map[string]bool{}
	for _, l := range manifest.Layers { mts[l.MediaType] = true }
	for _, expect := range []string{opts.MediaTypes.LayerWeights, opts.MediaTypes.LayerConfig, opts.MediaTypes.LayerDocs} {
		if !mts[expect] { t.Fatalf("expected layer media type %s not found", expect) }
	}
}

func trimSha256Prefix(d string) string {
	return strings.TrimPrefix(d, "sha256:")
}

func mustWrite(t *testing.T, p, s string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(s), 0o600); err != nil {
		t.Fatal(err)
	}
}
