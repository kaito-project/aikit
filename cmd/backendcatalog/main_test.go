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
	if _, err := backendcatalog.Parse(generated); err != nil {
		t.Fatalf("generated catalog is invalid: %v", err)
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
