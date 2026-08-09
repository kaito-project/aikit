package backendcatalogimport

const (
	// SchemaVersion is the generated backend catalog schema version.
	SchemaVersion = "v1"

	// LocalAIVersion is the LocalAI release imported by this generator.
	LocalAIVersion = "v4.8.2"

	architectureAMD64  = "amd64"
	architectureARM64  = "arm64"
	baseAppleSilicon   = "applesilicon"
	baseDistroless     = "distroless"
	baseUbuntu         = "ubuntu"
	baseUbuntu24       = "ubuntu24"
	dependencyNone     = "none"
	familyDiffusers    = "diffusers"
	familyVLLM         = "vllm"
	familyVLLMCpp      = "vllm-cpp"
	minimumCUDA12      = "12.0"
	minimumCUDA13      = "13.0"
	platformLinux      = "linux"
	runnerHFConfig     = "hf-config"
	runnerLlamaCpp     = "llama-cpp"
	runnerUnsupported  = "unsupported"
	runtimeApple       = "applesilicon"
	runtimeCPU         = "cpu"
	runtimeCUDA        = "cuda"
	runtimeROCm        = "rocm"
	selectorAMD        = "amd"
	selectorDefault    = "default"
	selectorNVIDIA     = "nvidia"
	statusExperimental = "experimental"
	statusSupported    = "supported"
	targetCUDA12       = "cuda12"
	targetCUDA13       = "cuda13"
	targetIntel        = "intel"
	targetL4TCUDA12    = "l4t-cuda12"
	targetL4TCUDA13    = "l4t-cuda13"
	targetMetal        = "metal"
	targetROCm         = "rocm"
	targetVulkan       = "vulkan"
)

// LocalAIV482Source pins the exact upstream input accepted by the command.
var LocalAIV482Source = SourcePin{
	Repository: "https://github.com/mudler/LocalAI",
	Path:       "backend/index.yaml",
	Revision:   "5ff25d9d145e0a03a5b9a3559c620f1e1204ca6d",
	SHA256:     "sha256:bee962adf3332b9f5adea1c9ab28709d9371bee6de24c828f72b8668a694e3ca",
}

// SourcePin identifies and verifies an upstream catalog source.
type SourcePin struct {
	Repository string `json:"repository"`
	Path       string `json:"path"`
	Revision   string `json:"revision"`
	SHA256     string `json:"sha256"`
}

// Catalog is the deterministic generated lock document.
type Catalog struct {
	SchemaVersion string    `json:"schemaVersion"`
	Source        SourcePin `json:"source"`
	Entries       []Entry   `json:"entries"`
}

// Entry is one selectable family, selector, and platform tuple.
type Entry struct {
	Family            string            `json:"family"`
	Selector          string            `json:"selector"`
	Platform          Platform          `json:"platform"`
	Runtime           string            `json:"runtime"`
	TargetProfile     string            `json:"targetProfile"`
	Status            string            `json:"status"`
	Channel           string            `json:"channel"`
	Core              Artifact          `json:"core"`
	Backend           BackendArtifact   `json:"backend"`
	Fallbacks         []BackendArtifact `json:"fallbacks,omitempty"`
	Version           string            `json:"version"`
	SourceRef         string            `json:"sourceRef"`
	DependencyProfile string            `json:"dependencyProfile"`
	RunnerProfile     string            `json:"runnerProfile"`
	Base              string            `json:"base"`
	SelfContained     bool              `json:"selfContained"`
	MinimumCUDA       string            `json:"minimumCUDA,omitempty"`
	Workloads         []string          `json:"workloads,omitempty"`
}

// Platform is a normalized OCI platform.
type Platform struct {
	OS           string `json:"os"`
	Architecture string `json:"architecture"`
	Variant      string `json:"variant,omitempty"`
}

// Artifact is an immutable OCI artifact reference.
type Artifact struct {
	Ref string `json:"ref"`
}

// BackendArtifact is an immutable backend artifact and its LocalAI install name.
type BackendArtifact struct {
	Ref         string `json:"ref"`
	InstallName string `json:"installName"`
}

// ResolvedManifest identifies an immutable manifest and its platform.
type ResolvedManifest struct {
	Digest   string   `json:"digest"`
	Platform Platform `json:"platform"`
}
