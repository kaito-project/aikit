package packager

import (
	_ "embed"
	"strconv"

	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
)

// File categorization thresholds and patterns.
const (
	// largeFileThreshold defines the size (10 MiB) above which unknown files are categorized as weights.
	largeFileThreshold = 10485760 // 10 * 1024 * 1024.
)

// modelpackScript is the bash script used to assemble a modelpack OCI layout.
// It is parameterized entirely through environment variables (see
// modelpackScriptEnv) so the script text stays static and free of Go string
// interpolation. Source files are expected at /src and output is written to
// /layout.
//
//go:embed scripts/modelpack.sh
var modelpackScript string

// genericScript is the bash script used to assemble a generic artifact OCI
// layout. Like modelpackScript it is parameterized through environment
// variables (see genericScriptEnv).
//
//go:embed scripts/generic.sh
var genericScript string

// modelpackScriptEnv returns the environment variables the modelpack assembly
// script reads. Passing parameters as env vars (rather than interpolating them
// into the script body) keeps the script static and avoids shell-injection and
// JSON-escaping hazards in generated text.
func modelpackScriptEnv(packMode, artifactType, mtManifest, name, refName string) map[string]string {
	return map[string]string{
		"PACK_MODE":            packMode,
		"ARTIFACT_TYPE":        artifactType,
		"MT_MANIFEST":          mtManifest,
		"NAME":                 name,
		"REF_NAME":             refName,
		"LARGE_FILE_THRESHOLD": strconv.Itoa(largeFileThreshold),
	}
}

// genericScriptEnv returns the environment variables the generic assembly script
// reads. rawLayerMT and archiveLayerMT are computed in Go from packMode because
// raw mode uses a different layer media type.
func genericScriptEnv(packMode, artifactType, name, refName string, debug bool) map[string]string {
	rawLayerMT := ocispec.MediaTypeImageLayer
	archiveLayerMT := ocispec.MediaTypeImageLayer
	if packMode == packModeRaw {
		rawLayerMT = "application/octet-stream"
	}
	env := map[string]string{
		"PACK_MODE":        packMode,
		"ARTIFACT_TYPE":    artifactType,
		"RAW_LAYER_MT":     rawLayerMT,
		"ARCHIVE_LAYER_MT": archiveLayerMT,
		"NAME":             name,
		"REF_NAME":         refName,
	}
	if debug {
		env["DEBUG"] = "1"
	}
	return env
}
