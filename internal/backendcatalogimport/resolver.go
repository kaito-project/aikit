package backendcatalogimport

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os/exec"
	"sort"
	"strings"

	godigest "github.com/opencontainers/go-digest"
	"github.com/pkg/errors"
)

const snapshotSchemaVersion = "v1alpha1"

// ResolutionErrorClass identifies a stable resolver failure category.
type ResolutionErrorClass string

const (
	resolutionErrorNotFound ResolutionErrorClass = "not-found"
)

// ResolutionError is a classified OCI resolution failure.
type ResolutionError struct {
	Reference string
	Class     ResolutionErrorClass
	Err       error
}

// Error returns the classified resolution failure.
func (resolutionError *ResolutionError) Error() string {
	return fmt.Sprintf("resolve %q (%s): %v", resolutionError.Reference, resolutionError.Class, resolutionError.Err)
}

// Unwrap returns the underlying resolver failure.
func (resolutionError *ResolutionError) Unwrap() error {
	return resolutionError.Err
}

// Resolver resolves an OCI tag or index to immutable manifests.
type Resolver interface {
	Resolve(context.Context, string) ([]ResolvedManifest, error)
}

// Snapshot is the deterministic, offline resolver input format.
type Snapshot struct {
	SchemaVersion string              `json:"schemaVersion"`
	References    []SnapshotReference `json:"references"`
}

// SnapshotReference records all manifests reachable through one source reference.
type SnapshotReference struct {
	SourceRef  string               `json:"sourceRef"`
	Manifests  []ResolvedManifest   `json:"manifests,omitempty"`
	ErrorClass ResolutionErrorClass `json:"errorClass,omitempty"`
}

type snapshotResult struct {
	manifests  []ResolvedManifest
	errorClass ResolutionErrorClass
}

// SnapshotResolver resolves references without network access.
type SnapshotResolver struct {
	byReference map[string]snapshotResult
}

// NewSnapshotResolver validates and constructs an offline resolver.
func NewSnapshotResolver(snapshot Snapshot) (*SnapshotResolver, error) {
	if snapshot.SchemaVersion != snapshotSchemaVersion {
		return nil, fmt.Errorf("unsupported resolution snapshot schema version %q", snapshot.SchemaVersion)
	}

	resolver := &SnapshotResolver{byReference: make(map[string]snapshotResult, len(snapshot.References))}
	for _, reference := range snapshot.References {
		if reference.SourceRef == "" {
			return nil, errors.New("resolution snapshot contains an empty sourceRef")
		}
		if len(reference.Manifests) > 0 && reference.ErrorClass != "" {
			return nil, fmt.Errorf("resolution snapshot reference %q has both manifests and errorClass", reference.SourceRef)
		}
		if len(reference.Manifests) == 0 && reference.ErrorClass == "" {
			return nil, fmt.Errorf("resolution snapshot reference %q has neither manifests nor errorClass", reference.SourceRef)
		}
		if reference.ErrorClass != "" && reference.ErrorClass != resolutionErrorNotFound {
			return nil, fmt.Errorf("resolution snapshot reference %q has unsupported errorClass %q", reference.SourceRef, reference.ErrorClass)
		}
		if _, exists := resolver.byReference[reference.SourceRef]; exists {
			return nil, fmt.Errorf("resolution snapshot contains duplicate reference %q", reference.SourceRef)
		}

		result := snapshotResult{errorClass: reference.ErrorClass}
		if len(reference.Manifests) > 0 {
			result.manifests = append([]ResolvedManifest(nil), reference.Manifests...)
			if err := normalizeResolvedManifests(result.manifests, reference.SourceRef, true); err != nil {
				return nil, err
			}
		}
		resolver.byReference[reference.SourceRef] = result
	}

	return resolver, nil
}

// ParseSnapshot parses a strict offline resolution snapshot.
func ParseSnapshot(data []byte) (*SnapshotResolver, error) {
	var snapshot Snapshot
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&snapshot); err != nil {
		return nil, errors.Wrap(err, "parse resolution snapshot")
	}
	var trailing json.RawMessage
	if err := decoder.Decode(&trailing); err != io.EOF {
		if err == nil {
			return nil, errors.New("resolution snapshot contains multiple JSON values")
		}

		return nil, errors.Wrap(err, "parse trailing resolution snapshot data")
	}

	return NewSnapshotResolver(snapshot)
}

// Resolve resolves an exact reference from the offline snapshot.
func (resolver *SnapshotResolver) Resolve(_ context.Context, reference string) ([]ResolvedManifest, error) {
	result, ok := resolver.byReference[reference]
	if !ok {
		return nil, fmt.Errorf("reference %q is absent from the offline resolution snapshot", reference)
	}
	if result.errorClass != "" {
		return nil, &ResolutionError{Reference: reference, Class: result.errorClass, Err: errors.New("recorded offline resolution failure")}
	}

	return append([]ResolvedManifest(nil), result.manifests...), nil
}

// CraneResolver uses the crane CLI for registry transport and credential handling.
type CraneResolver struct {
	Binary string
	run    craneRunner
}

type craneRunner func(context.Context, string, ...string) ([]byte, error)

// Resolve resolves an OCI index or image using crane.
func (resolver CraneResolver) Resolve(ctx context.Context, reference string) ([]ResolvedManifest, error) {
	binary := resolver.Binary
	if binary == "" {
		binary = "crane"
	}
	runner := resolver.run
	if runner == nil {
		runner = runCrane
	}

	// Resolve a mutable source reference once, then inspect only that immutable
	// root. This prevents a moving tag from mixing manifest, config, and digest
	// data from different images during catalog promotion.
	digestBytes, err := runner(ctx, binary, "digest", reference)
	if err != nil {
		return nil, classifyCraneError(reference, err)
	}
	rootDigest, err := normalizeResolvedDigest(reference, string(digestBytes))
	if err != nil {
		return nil, err
	}
	resolvedReference := immutableReference(reference, rootDigest)

	manifestBytes, err := runner(ctx, binary, "manifest", resolvedReference)
	if err != nil {
		return nil, classifyCraneError(resolvedReference, err)
	}

	var document struct {
		Manifests []struct {
			Digest   string    `json:"digest"`
			Platform *Platform `json:"platform"`
		} `json:"manifests"`
	}
	if err := json.Unmarshal(manifestBytes, &document); err != nil {
		return nil, errors.Wrapf(err, "parse OCI manifest for %q", reference)
	}

	if len(document.Manifests) > 0 {
		manifests := make([]ResolvedManifest, 0, len(document.Manifests))
		for _, descriptor := range document.Manifests {
			if descriptor.Platform == nil || descriptor.Platform.OS == "" || descriptor.Platform.Architecture == "" {
				continue
			}
			if descriptor.Platform.OS == "unknown" || descriptor.Platform.Architecture == "unknown" {
				continue
			}
			manifests = append(manifests, ResolvedManifest{Digest: descriptor.Digest, Platform: *descriptor.Platform})
		}
		if len(manifests) == 0 {
			return nil, fmt.Errorf("OCI index %q has no platform manifests", reference)
		}
		if err := normalizeResolvedManifests(manifests, reference, false); err != nil {
			return nil, err
		}

		return manifests, nil
	}

	configBytes, err := runner(ctx, binary, "config", resolvedReference)
	if err != nil {
		return nil, classifyCraneError(resolvedReference, err)
	}
	var platform Platform
	if err := json.Unmarshal(configBytes, &platform); err != nil {
		return nil, errors.Wrapf(err, "parse OCI config for %q", reference)
	}
	manifests := []ResolvedManifest{{Digest: rootDigest, Platform: platform}}
	if err := normalizeResolvedManifests(manifests, reference, true); err != nil {
		return nil, err
	}

	return manifests, nil
}

func normalizeResolvedDigest(reference, rawDigest string) (string, error) {
	digest, err := godigest.Parse(strings.TrimSpace(rawDigest))
	if err != nil {
		return "", errors.Wrapf(err, "reference %q has invalid resolved digest %q", reference, strings.TrimSpace(rawDigest))
	}
	if digest.Algorithm() != godigest.SHA256 {
		return "", fmt.Errorf("reference %q uses unsupported resolved digest algorithm %q", reference, digest.Algorithm())
	}

	return digest.String(), nil
}

func classifyCraneError(reference string, err error) error {
	message := strings.ToLower(err.Error())
	if strings.Contains(message, "manifest_unknown") || strings.Contains(message, "manifest unknown") ||
		strings.Contains(message, "name_unknown") || strings.Contains(message, "name unknown") {
		return &ResolutionError{Reference: reference, Class: resolutionErrorNotFound, Err: err}
	}

	return err
}

func resolutionErrorClass(err error) (ResolutionErrorClass, bool) {
	var resolutionError *ResolutionError
	if !errors.As(err, &resolutionError) {
		return "", false
	}

	return resolutionError.Class, true
}

func runCrane(ctx context.Context, binary string, arguments ...string) ([]byte, error) {
	command := exec.CommandContext(ctx, binary, arguments...)
	output, err := command.CombinedOutput()
	if err != nil {
		return nil, errors.Wrapf(err, "%s %s failed: %s", binary, strings.Join(arguments, " "), strings.TrimSpace(string(output)))
	}

	return output, nil
}

func normalizeResolvedManifests(manifests []ResolvedManifest, reference string, allowEmptyPlatform bool) error {
	seen := make(map[string]string, len(manifests))
	for index := range manifests {
		digest, err := godigest.Parse(manifests[index].Digest)
		if err != nil {
			return errors.Wrapf(err, "reference %q has invalid digest %q", reference, manifests[index].Digest)
		}
		if digest.Algorithm() != godigest.SHA256 {
			return fmt.Errorf("reference %q uses unsupported digest algorithm %q", reference, digest.Algorithm())
		}
		manifests[index].Digest = digest.String()
		manifests[index].Platform = normalizePlatform(manifests[index].Platform)
		platformKey := manifests[index].Platform.key()
		emptyOS := manifests[index].Platform.OS == ""
		emptyArchitecture := manifests[index].Platform.Architecture == ""
		if emptyOS != emptyArchitecture || (emptyOS && !allowEmptyPlatform) {
			return fmt.Errorf("reference %q manifest %s has no platform", reference, digest)
		}
		if previous, exists := seen[platformKey]; exists && previous != digest.String() {
			return fmt.Errorf("reference %q has conflicting digests for platform %q", reference, platformKey)
		}
		seen[platformKey] = digest.String()
	}

	sort.Slice(manifests, func(left, right int) bool {
		leftKey := manifests[left].Platform.key()
		rightKey := manifests[right].Platform.key()
		if leftKey != rightKey {
			return leftKey < rightKey
		}

		return manifests[left].Digest < manifests[right].Digest
	})

	return nil
}

func normalizePlatform(platform Platform) Platform {
	platform.OS = strings.ToLower(strings.TrimSpace(platform.OS))
	platform.Architecture = strings.ToLower(strings.TrimSpace(platform.Architecture))
	platform.Variant = strings.ToLower(strings.TrimSpace(platform.Variant))
	switch platform.Architecture {
	case "x86_64", "x86-64":
		platform.Architecture = architectureAMD64
	case "aarch64":
		platform.Architecture = architectureARM64
	}
	if platform.Architecture == architectureARM64 && platform.Variant == "v8" {
		platform.Variant = ""
	}

	return platform
}

func (platform Platform) key() string {
	return platform.OS + "/" + platform.Architecture + "/" + platform.Variant
}
