// Package backendcatalog parses and resolves immutable LocalAI backend catalogs.
package backendcatalog

import (
	"bytes"
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"encoding/json"
	"io"
	"strings"

	"github.com/pkg/errors"
)

const schemaVersionV2 = "v2"

var (
	// ErrInvalidCatalog indicates that a catalog does not satisfy the current schema.
	ErrInvalidCatalog = errors.New("invalid backend catalog")
	// ErrInvalidRequest indicates that a resolution request is malformed.
	ErrInvalidRequest = errors.New("invalid backend catalog request")
	// ErrNotFound indicates that no exact catalog tuple matches a request.
	ErrNotFound = errors.New("backend catalog entry not found")
	// ErrUnavailable indicates that an exact match is quarantined or deprecated.
	ErrUnavailable = errors.New("backend catalog entry unavailable")
)

// Selector is an open, validated LocalAI backend selector.
type Selector string

const (
	SelectorDefault      Selector = "default"
	SelectorCPU          Selector = "cpu"
	SelectorNVIDIA       Selector = "nvidia"
	SelectorNVIDIACUDA12 Selector = "nvidia-cuda-12"
	SelectorNVIDIACUDA13 Selector = "nvidia-cuda-13"
	SelectorAMD          Selector = "amd"
	SelectorIntel        Selector = "intel"
	SelectorVulkan       Selector = "vulkan"
	SelectorMetal        Selector = "metal"
	SelectorMetalDarwin  Selector = "metal-darwin-arm64"
	SelectorNVIDIAL4T    Selector = "nvidia-l4t"
	SelectorL4TCUDA12    Selector = "nvidia-l4t-cuda-12"
	SelectorL4TCUDA13    Selector = "nvidia-l4t-cuda-13"
)

// Runtime is an AIKit runtime requested by users or recorded for a catalog entry.
type Runtime string

const (
	RuntimeCPU          Runtime = "cpu"
	RuntimeCUDA         Runtime = "cuda"
	RuntimeCUDA12       Runtime = "cuda-12"
	RuntimeCUDA13       Runtime = "cuda-13"
	RuntimeROCm         Runtime = "rocm"
	RuntimeAppleSilicon Runtime = "applesilicon"
)

// TargetProfile identifies the concrete accelerator target selected by a catalog entry.
type TargetProfile string

const (
	TargetProfileCPU       TargetProfile = "cpu"
	TargetProfileCUDA12    TargetProfile = "cuda12"
	TargetProfileCUDA13    TargetProfile = "cuda13"
	TargetProfileROCm      TargetProfile = "rocm"
	TargetProfileIntel     TargetProfile = "intel"
	TargetProfileVulkan    TargetProfile = "vulkan"
	TargetProfileMetal     TargetProfile = "metal"
	TargetProfileL4TCUDA12 TargetProfile = "l4t-cuda12"
	TargetProfileL4TCUDA13 TargetProfile = "l4t-cuda13"
)

// Status is AIKit's support status for an entry.
type Status string

const (
	StatusSupported    Status = "supported"
	StatusExperimental Status = "experimental"
	StatusQuarantined  Status = "quarantined"
	StatusDeprecated   Status = "deprecated"
)

// Channel is the upstream release channel represented by an entry.
type Channel string

const (
	// ChannelStable is the only channel supported by the current schema.
	ChannelStable Channel = "stable"
)

// RunnerProfile identifies an explicitly supported runner behavior.
type RunnerProfile string

const (
	RunnerProfileUnsupported RunnerProfile = "unsupported"
	RunnerProfileLlamaCpp    RunnerProfile = "llama-cpp"
	RunnerProfileVLLMCpp     RunnerProfile = "vllm-cpp"
	RunnerProfileHFConfig    RunnerProfile = "hf-config"
)

// Platform is an exact OCI target platform.
type Platform struct {
	OS           string `json:"os"`
	Architecture string `json:"architecture"`
	Variant      string `json:"variant,omitempty"`
}

// Artifact is an immutable OCI artifact reference.
type Artifact struct {
	Ref string `json:"ref"`
}

// BackendArtifact is an immutable installable backend artifact.
type BackendArtifact struct {
	Ref         string `json:"ref"`
	InstallName string `json:"installName"`
}

// RuntimeSymlink declares one validated compatibility link in the runtime base.
type RuntimeSymlink struct {
	Target string `json:"target"`
	Path   string `json:"path"`
}

// Source records the immutable source used to produce a catalog.
type Source struct {
	Repository string `json:"repository"`
	Revision   string `json:"revision"`
	Path       string `json:"path,omitempty"`
	SHA256     string `json:"sha256,omitempty"`
}

// Defaults defines catalog-owned selection defaults.
type Defaults struct {
	Family    string            `json:"family"`
	Selectors []DefaultSelector `json:"selectors"`
}

// DefaultSelector maps one AIKit runtime and optional platform to its default LocalAI selector.
type DefaultSelector struct {
	Runtime  Runtime   `json:"runtime"`
	Platform *Platform `json:"platform,omitempty"`
	Selector Selector  `json:"selector"`
}

// Catalog is a versioned backend catalog.
type Catalog struct {
	SchemaVersion string   `json:"schemaVersion"`
	Source        Source   `json:"source"`
	Defaults      Defaults `json:"defaults"`
	Entries       []Entry  `json:"entries"`

	digest          string
	canonicalDigest string
}

// Entry is an exact backend selection tuple and its complete install plan.
type Entry struct {
	Family            string            `json:"family"`
	Selector          Selector          `json:"selector"`
	Platform          Platform          `json:"platform"`
	Runtime           Runtime           `json:"runtime"`
	TargetProfile     TargetProfile     `json:"targetProfile"`
	Status            Status            `json:"status"`
	Channel           Channel           `json:"channel"`
	Version           string            `json:"version"`
	SourceRef         string            `json:"sourceRef"`
	RuntimeBase       Artifact          `json:"runtimeBase"`
	RunnerRuntimeBase *Artifact         `json:"runnerRuntimeBase,omitempty"`
	Core              Artifact          `json:"core"`
	Backend           BackendArtifact   `json:"backend"`
	Fallbacks         []BackendArtifact `json:"fallbacks,omitempty"`
	SystemPackages    []string          `json:"systemPackages,omitempty"`
	RuntimeSymlinks   []RuntimeSymlink  `json:"runtimeSymlinks,omitempty"`
	Environment       []string          `json:"environment,omitempty"`
	RunnerProfile     RunnerProfile     `json:"runnerProfile"`
	Workloads         []string          `json:"workloads,omitempty"`
}

// Request identifies one public runtime, family, and platform tuple. An empty
// family uses the catalog-owned default, and an empty runtime selects CPU.
type Request struct {
	Family   string
	Runtime  Runtime
	Platform Platform
}

// Resolution is a detached install plan with the digest of its source catalog.
type Resolution struct {
	Entry
	CatalogDigest string
	Source        Source
}

// Resolver resolves exact tuples from a validated, immutable snapshot.
type Resolver struct {
	entries          map[tupleKey]Entry
	defaultFamily    string
	defaults         map[Runtime]Selector
	platformDefaults map[defaultSelectorKey]Selector
	catalogDigest    string
	source           Source
}

type defaultSelectorKey struct {
	runtime      Runtime
	os           string
	architecture string
	variant      string
}

type tupleKey struct {
	family       string
	selector     Selector
	os           string
	architecture string
	variant      string
}

type runtimeSelection struct {
	catalogRuntime Runtime
	targetProfiles []TargetProfile
	selectors      []Selector
}

//go:embed catalog.lock.json
var defaultCatalogJSON []byte

// Parse strictly parses and validates a v2 catalog.
func Parse(data []byte) (*Catalog, error) {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()

	var catalog Catalog
	if err := decoder.Decode(&catalog); err != nil {
		return nil, invalidCatalog("decode catalog", err)
	}
	if err := ensureJSONEOF(decoder); err != nil {
		return nil, err
	}
	if err := validateCatalog(&catalog); err != nil {
		return nil, err
	}
	if err := setParsedCatalogDigests(&catalog, data); err != nil {
		return nil, err
	}

	return &catalog, nil
}

// Default returns a freshly parsed copy of the embedded catalog.
func Default() (*Catalog, error) {
	return Parse(defaultCatalogJSON)
}

// Digest returns the SHA-256 digest of the exact parsed catalog bytes.
func (c *Catalog) Digest() string {
	if c == nil {
		return ""
	}

	return c.digest
}

// NewResolver validates and snapshots a catalog.
func NewResolver(catalog *Catalog) (*Resolver, error) {
	if catalog == nil {
		return nil, errors.Wrap(ErrInvalidCatalog, "catalog is nil")
	}

	snapshot := cloneCatalog(*catalog)
	if err := validateCatalog(&snapshot); err != nil {
		return nil, err
	}
	if err := setResolverCatalogDigest(&snapshot, catalog); err != nil {
		return nil, err
	}

	entries := make(map[tupleKey]Entry, len(snapshot.Entries))
	for _, entry := range snapshot.Entries {
		entries[keyFor(entry.Family, entry.Selector, entry.Platform)] = cloneEntry(entry)
	}

	defaults := make(map[Runtime]Selector, len(snapshot.Defaults.Selectors))
	platformDefaults := make(map[defaultSelectorKey]Selector, len(snapshot.Defaults.Selectors))
	for _, selection := range snapshot.Defaults.Selectors {
		if selection.Platform == nil {
			defaults[selection.Runtime] = selection.Selector
			continue
		}
		platformDefaults[defaultSelectorKeyFor(selection.Runtime, *selection.Platform)] = selection.Selector
	}

	return &Resolver{
		entries:          entries,
		defaultFamily:    snapshot.Defaults.Family,
		defaults:         defaults,
		platformDefaults: platformDefaults,
		catalogDigest:    snapshot.digest,
		source:           snapshot.Source,
	}, nil
}

// Resolve returns a detached exact-match install plan without applying fallbacks.
func (r *Resolver) Resolve(request Request) (Resolution, error) {
	if r == nil {
		return Resolution{}, errors.Wrap(ErrInvalidRequest, "resolver is nil")
	}
	request.Platform = normalizeRequestPlatform(request.Platform)
	if request.Family == "" {
		request.Family = r.defaultFamily
	}
	if request.Runtime == "" {
		request.Runtime = RuntimeCPU
	}
	if err := validateRequest(request); err != nil {
		return Resolution{}, err
	}

	selection, err := r.selectRuntime(request.Runtime, request.Platform)
	if err != nil {
		return Resolution{}, err
	}

	var entry Entry
	found := false
	for _, selector := range selection.selectors {
		candidate, ok := r.entries[keyFor(request.Family, selector, request.Platform)]
		if !ok || candidate.Runtime != selection.catalogRuntime || !containsTargetProfile(selection.targetProfiles, candidate.TargetProfile) {
			continue
		}
		entry = candidate
		found = true
		break
	}
	if !found {
		return Resolution{}, errors.Wrapf(
			ErrNotFound,
			"family %q runtime %q platform %q",
			request.Family,
			request.Runtime,
			formatPlatform(request.Platform),
		)
	}
	if entry.Status == StatusQuarantined || entry.Status == StatusDeprecated {
		return Resolution{}, errors.Wrapf(ErrUnavailable, "family %q runtime %q has status %q", entry.Family, request.Runtime, entry.Status)
	}

	return Resolution{Entry: cloneEntry(entry), CatalogDigest: r.catalogDigest, Source: r.source}, nil
}

func (r *Resolver) selectRuntime(runtime Runtime, platform Platform) (runtimeSelection, error) {
	switch runtime {
	case RuntimeCPU:
		selector := r.defaults[RuntimeCPU]
		selectors := []Selector{selector}
		if selector != SelectorCPU {
			selectors = append(selectors, SelectorCPU)
		}

		return runtimeSelection{catalogRuntime: RuntimeCPU, targetProfiles: []TargetProfile{TargetProfileCPU}, selectors: selectors}, nil
	case RuntimeCUDA:
		selector, ok := r.platformDefaults[defaultSelectorKeyFor(RuntimeCUDA, platform)]
		if !ok {
			selector = r.defaults[RuntimeCUDA]
		}
		targets := []TargetProfile{TargetProfileCUDA12, TargetProfileCUDA13}
		if platform.Architecture == platformArchitectureARM64 {
			targets = []TargetProfile{TargetProfileL4TCUDA12, TargetProfileL4TCUDA13}
		}

		return runtimeSelection{catalogRuntime: RuntimeCUDA, targetProfiles: targets, selectors: []Selector{selector}}, nil
	case RuntimeCUDA12:
		selector := SelectorNVIDIACUDA12
		target := TargetProfileCUDA12
		if platform.Architecture == platformArchitectureARM64 {
			selector = SelectorL4TCUDA12
			target = TargetProfileL4TCUDA12
		}

		return runtimeSelection{catalogRuntime: RuntimeCUDA, targetProfiles: []TargetProfile{target}, selectors: []Selector{selector}}, nil
	case RuntimeCUDA13:
		selector := SelectorNVIDIACUDA13
		target := TargetProfileCUDA13
		if platform.Architecture == platformArchitectureARM64 {
			selector = SelectorL4TCUDA13
			target = TargetProfileL4TCUDA13
		}

		return runtimeSelection{catalogRuntime: RuntimeCUDA, targetProfiles: []TargetProfile{target}, selectors: []Selector{selector}}, nil
	case RuntimeROCm:
		return runtimeSelection{
			catalogRuntime: RuntimeROCm,
			targetProfiles: []TargetProfile{TargetProfileROCm},
			selectors:      []Selector{r.defaults[RuntimeROCm]},
		}, nil
	case RuntimeAppleSilicon:
		return runtimeSelection{
			catalogRuntime: RuntimeAppleSilicon,
			targetProfiles: []TargetProfile{TargetProfileVulkan},
			selectors:      []Selector{r.defaults[RuntimeAppleSilicon]},
		}, nil
	default:
		return runtimeSelection{}, errors.Wrapf(ErrInvalidRequest, "runtime %q is not supported", runtime)
	}
}

func containsTargetProfile(targets []TargetProfile, candidate TargetProfile) bool {
	for _, target := range targets {
		if candidate == target {
			return true
		}
	}

	return false
}

func ensureJSONEOF(decoder *json.Decoder) error {
	var trailing json.RawMessage
	if err := decoder.Decode(&trailing); err != io.EOF {
		if err == nil {
			return errors.Wrap(ErrInvalidCatalog, "catalog contains multiple JSON values")
		}

		return invalidCatalog("decode trailing data", err)
	}

	return nil
}

func setParsedCatalogDigests(catalog *Catalog, data []byte) error {
	canonical, err := json.Marshal(catalog)
	if err != nil {
		return invalidCatalog("marshal canonical catalog", err)
	}
	catalog.digest = digestBytes(data)
	catalog.canonicalDigest = digestBytes(canonical)

	return nil
}

func setResolverCatalogDigest(snapshot, original *Catalog) error {
	canonical, err := json.Marshal(snapshot)
	if err != nil {
		return invalidCatalog("marshal canonical catalog", err)
	}
	canonicalDigest := digestBytes(canonical)
	snapshot.canonicalDigest = canonicalDigest
	if original.digest != "" && original.canonicalDigest == canonicalDigest {
		snapshot.digest = original.digest
	} else {
		// Programmatically constructed or subsequently modified catalogs have no
		// source bytes, so their deterministic canonical JSON is the source.
		snapshot.digest = canonicalDigest
	}

	return nil
}

func digestBytes(data []byte) string {
	sum := sha256.Sum256(data)

	return "sha256:" + hex.EncodeToString(sum[:])
}

func cloneCatalog(catalog Catalog) Catalog {
	clone := catalog
	if catalog.Defaults.Selectors != nil {
		clone.Defaults.Selectors = make([]DefaultSelector, len(catalog.Defaults.Selectors))
		for i, selection := range catalog.Defaults.Selectors {
			clone.Defaults.Selectors[i] = selection
			if selection.Platform != nil {
				platform := *selection.Platform
				clone.Defaults.Selectors[i].Platform = &platform
			}
		}
	}
	if catalog.Entries != nil {
		clone.Entries = make([]Entry, len(catalog.Entries))
		for i, entry := range catalog.Entries {
			clone.Entries[i] = cloneEntry(entry)
		}
	}

	return clone
}

func cloneEntry(entry Entry) Entry {
	clone := entry
	if entry.RunnerRuntimeBase != nil {
		runnerRuntimeBase := *entry.RunnerRuntimeBase
		clone.RunnerRuntimeBase = &runnerRuntimeBase
	}
	clone.Fallbacks = append([]BackendArtifact(nil), entry.Fallbacks...)
	clone.SystemPackages = append([]string(nil), entry.SystemPackages...)
	clone.RuntimeSymlinks = append([]RuntimeSymlink(nil), entry.RuntimeSymlinks...)
	clone.Environment = append([]string(nil), entry.Environment...)
	clone.Workloads = append([]string(nil), entry.Workloads...)

	return clone
}

func keyFor(family string, selector Selector, platform Platform) tupleKey {
	return tupleKey{
		family:       family,
		selector:     selector,
		os:           platform.OS,
		architecture: platform.Architecture,
		variant:      platform.Variant,
	}
}

func defaultSelectorKeyFor(runtime Runtime, platform Platform) defaultSelectorKey {
	return defaultSelectorKey{
		runtime:      runtime,
		os:           platform.OS,
		architecture: platform.Architecture,
		variant:      platform.Variant,
	}
}

func normalizeRequestPlatform(platform Platform) Platform {
	platform.OS = strings.ToLower(strings.TrimSpace(platform.OS))
	platform.Architecture = strings.ToLower(strings.TrimSpace(platform.Architecture))
	platform.Variant = strings.ToLower(strings.TrimSpace(platform.Variant))
	switch platform.Architecture {
	case "x86_64", "x86-64":
		platform.Architecture = platformArchitectureAMD64
	case "aarch64":
		platform.Architecture = platformArchitectureARM64
	}
	if platform.OS == platformOSLinux && platform.Architecture == platformArchitectureAMD64 {
		switch platform.Variant {
		case "v2", "v3", "v4":
			platform.Variant = ""
		}
	}
	if platform.Architecture == platformArchitectureARM64 && platform.Variant == "v8" {
		platform.Variant = ""
	}

	return platform
}
