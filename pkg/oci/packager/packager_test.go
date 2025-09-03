package packager

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

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

func mustWrite(t *testing.T, p, s string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(s), 0o600); err != nil {
		t.Fatal(err)
	}
}
