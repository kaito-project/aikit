package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/kaito-project/aikit/internal/backendcatalogimport"
	"github.com/kaito-project/aikit/pkg/backendcatalog"
)

func TestRunWritesAndChecksCatalog(t *testing.T) {
	sourcePath := filepath.Join("..", "..", "internal", "backendcatalogimport", "testdata", "index.yaml")
	snapshotPath := filepath.Join("..", "..", "internal", "backendcatalogimport", "testdata", "resolutions.json")
	source, err := os.ReadFile(sourcePath)
	if err != nil {
		t.Fatalf("read source fixture: %v", err)
	}
	digest := sha256.Sum256(source)
	config := commandConfig{
		Source: backendcatalogimport.SourcePin{
			Repository: "https://example.com/localai",
			Path:       "backend/index.yaml",
			Revision:   "1111111111111111111111111111111111111111",
			SHA256:     "sha256:" + hex.EncodeToString(digest[:]),
		},
		Version: backendcatalogimport.LocalAIVersion,
		Stdout:  &bytes.Buffer{},
	}
	outputPath := filepath.Join(t.TempDir(), "catalog.lock.json")
	arguments := []string{
		"--source", sourcePath,
		"--snapshot", snapshotPath,
		"--core-ref", "registry.example/core:v4.8.2-{architecture}",
		"--output", outputPath,
	}

	if err := run(context.Background(), arguments, config); err != nil {
		t.Fatalf("run() write error = %v", err)
	}
	generated, err := os.ReadFile(outputPath)
	if err != nil {
		t.Fatalf("read generated catalog: %v", err)
	}
	catalog, err := backendcatalog.Parse(generated)
	if err != nil {
		t.Fatalf("generated catalog is invalid: %v", err)
	}
	if catalog.SchemaVersion != "v2" {
		t.Fatalf("schemaVersion = %q, want v2", catalog.SchemaVersion)
	}
	if catalog.Defaults.Family != "llama-cpp" || len(catalog.Defaults.Selectors) != 5 {
		t.Fatalf("defaults = %#v, want llama-cpp and five runtime/platform selectors", catalog.Defaults)
	}
	if len(catalog.Entries) != 6 {
		t.Fatalf("entry count = %d, want 6", len(catalog.Entries))
	}
	var foundNVIDIA bool
	for _, entry := range catalog.Entries {
		if !strings.Contains(entry.RuntimeBase.Ref, "@sha256:") {
			t.Errorf("runtimeBase.ref = %q, want immutable digest reference", entry.RuntimeBase.Ref)
		}
		wantPackages := []string(nil)
		if entry.TargetProfile == backendcatalog.TargetProfileROCm {
			wantPackages = []string{"pciutils"}
		}
		if strings.Join(entry.SystemPackages, ",") != strings.Join(wantPackages, ",") {
			t.Errorf("systemPackages for %s/%s = %v, want %v", entry.Family, entry.Selector, entry.SystemPackages, wantPackages)
		}
		if entry.Selector == backendcatalog.SelectorNVIDIA {
			foundNVIDIA = true
			if got, want := strings.Join(entry.Environment, ","), "BUILD_TYPE=cublas,NVIDIA_DRIVER_CAPABILITIES=compute,utility,NVIDIA_REQUIRE_CUDA=cuda>=12.0,NVIDIA_VISIBLE_DEVICES=all"; got != want {
				t.Errorf("NVIDIA environment = %q, want %q", got, want)
			}
		}
	}
	if !foundNVIDIA {
		t.Fatal("generated catalog has no NVIDIA entry")
	}
	for _, legacyField := range [][]byte{[]byte(`"dependencyProfile"`), []byte(`"base"`), []byte(`"selfContained"`), []byte(`"minimumCUDA"`)} {
		if bytes.Contains(generated, legacyField) {
			t.Errorf("generated v2 catalog contains legacy field %s", legacyField)
		}
	}

	if err := run(context.Background(), append(arguments, "--check"), config); err != nil {
		t.Fatalf("run() check error = %v", err)
	}
	if err := writeFileAtomic(outputPath, append(generated, '\n')); err != nil {
		t.Fatalf("tamper with generated catalog: %v", err)
	}
	if err := run(context.Background(), append(arguments, "--check"), config); err == nil || !strings.Contains(err.Error(), "out of date") {
		t.Fatalf("run() stale check error = %v", err)
	}
	stale, err := os.ReadFile(outputPath)
	if err != nil {
		t.Fatalf("read stale catalog: %v", err)
	}
	if bytes.Equal(stale, generated) {
		t.Fatal("--check unexpectedly rewrote the stale output")
	}
}

func TestRunRequiresSource(t *testing.T) {
	err := run(context.Background(), nil, commandConfig{Stdout: &bytes.Buffer{}})
	if err == nil || !strings.Contains(err.Error(), "--source is required") {
		t.Fatalf("run() error = %v", err)
	}
}
