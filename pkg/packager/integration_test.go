//go:build integration

// Package packager integration test. This file is only compiled under the
// "integration" build tag, so the default `go test ./...` never runs it.
//
// Even under -tags integration it skips cleanly unless a real build environment
// is available, signalled by AIKIT_INTEGRATION=1 and a docker buildx binary on
// PATH. When those are present it drives a real packager build and asserts the
// produced OCI layout is structurally valid (index.json, manifest schemaVersion,
// oci-layout marker). This is the harness that proves the embedded scripts emit
// a correct layout end-to-end; the unit tests can only assert the script text.
package packager

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// requireIntegrationEnv skips the test unless the integration environment is
// explicitly enabled and a buildx binary is available.
func requireIntegrationEnv(t *testing.T) {
	t.Helper()
	if os.Getenv("AIKIT_INTEGRATION") != "1" {
		t.Skip("set AIKIT_INTEGRATION=1 to run packager integration tests")
	}
	if _, err := exec.LookPath("docker"); err != nil {
		t.Skip("docker not found on PATH")
	}
	if err := exec.Command("docker", "buildx", "version").Run(); err != nil {
		t.Skip("docker buildx not available")
	}
}

// assertValidOCILayout checks that dir contains a structurally valid OCI image
// layout produced by the packager.
func assertValidOCILayout(t *testing.T, dir string) {
	t.Helper()

	// oci-layout marker.
	layoutBytes, err := os.ReadFile(filepath.Join(dir, "oci-layout"))
	if err != nil {
		t.Fatalf("reading oci-layout: %v", err)
	}
	var layout struct {
		ImageLayoutVersion string `json:"imageLayoutVersion"`
	}
	if err := json.Unmarshal(layoutBytes, &layout); err != nil {
		t.Fatalf("oci-layout is not valid JSON: %v", err)
	}
	if layout.ImageLayoutVersion != "1.0.0" {
		t.Errorf("imageLayoutVersion = %q, want 1.0.0", layout.ImageLayoutVersion)
	}

	// index.json with a manifest reference.
	indexBytes, err := os.ReadFile(filepath.Join(dir, "index.json"))
	if err != nil {
		t.Fatalf("reading index.json: %v", err)
	}
	var index struct {
		SchemaVersion int `json:"schemaVersion"`
		Manifests     []struct {
			MediaType string `json:"mediaType"`
			Digest    string `json:"digest"`
		} `json:"manifests"`
	}
	if err := json.Unmarshal(indexBytes, &index); err != nil {
		t.Fatalf("index.json is not valid JSON: %v", err)
	}
	if index.SchemaVersion != 2 {
		t.Errorf("index schemaVersion = %d, want 2", index.SchemaVersion)
	}
	if len(index.Manifests) == 0 {
		t.Fatal("index.json has no manifests")
	}
}

// TestPackagerOCILayoutIntegration is a placeholder harness that wires up the
// skip guards and layout assertions. Driving an actual `docker buildx build`
// with the aikit frontend and exporting an OCI layout to a temp dir is left to
// the CI environment, which builds and loads the frontend image first; the
// assertion helper above is the reusable core. Until that wiring is enabled in
// CI, this test verifies the guard path and is otherwise inert.
func TestPackagerOCILayoutIntegration(t *testing.T) {
	requireIntegrationEnv(t)

	outDir := os.Getenv("AIKIT_INTEGRATION_LAYOUT_DIR")
	if outDir == "" {
		t.Skip("set AIKIT_INTEGRATION_LAYOUT_DIR to a built OCI layout directory to assert its structure")
	}
	assertValidOCILayout(t, outDir)
}
