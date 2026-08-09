package backendcatalogimport

const (
	// SchemaVersion is the generated backend catalog schema version.
	SchemaVersion = "v2"

	// LocalAIVersion is the LocalAI release imported by this generator.
	LocalAIVersion = "v4.8.2"

	legacyLocalAIVersion = "v3.12.1"

	architectureAMD64   = "amd64"
	architectureARM64   = "arm64"
	defaultFamily       = "llama-cpp"
	familyDiffusers     = "diffusers"
	familyVLLM          = "vllm"
	familyVLLMCpp       = "vllm-cpp"
	minimumCUDA12       = "12.0"
	minimumCUDA13       = "13.0"
	platformLinux       = "linux"
	runnerHFConfig      = "hf-config"
	runnerLlamaCpp      = "llama-cpp"
	runnerUnsupported   = "unsupported"
	runtimeApple        = "applesilicon"
	runtimeCPU          = "cpu"
	runtimeCUDA         = "cuda"
	runtimeROCm         = "rocm"
	selectorAMD         = "amd"
	selectorDefault     = "default"
	selectorNVIDIA      = "nvidia"
	selectorNVIDIAL4T   = "nvidia-l4t"
	selectorL4TCUDA12   = "nvidia-l4t-cuda-12"
	selectorL4TCUDA13   = "nvidia-l4t-cuda-13"
	statusExperimental  = "experimental"
	statusSupported     = "supported"
	targetCUDA12        = "cuda12"
	targetCUDA13        = "cuda13"
	targetIntel         = "intel"
	targetL4TCUDA12     = "l4t-cuda12"
	targetL4TCUDA13     = "l4t-cuda13"
	targetMetal         = "metal"
	targetROCm          = "rocm"
	targetVulkan        = "vulkan"
	chiseledRuntimeBase = "ghcr.io/kaito-project/aikit/base:latest"
	ubuntu22RuntimeBase = "docker.io/library/ubuntu:22.04"
	ubuntuRuntimeBase   = "docker.io/library/ubuntu:24.04"
	rocmRuntimeBase     = "docker.io/rocm/dev-ubuntu-24.04:7.2.1"
	vulkanRuntimeBase   = "ghcr.io/kaito-project/aikit/applesilicon/base:latest"
	l4tRuntimeBase      = "nvcr.io/nvidia/l4t-jetpack:r36.4.0"
	vulkanEnvironment   = "VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/virtio_icd.aarch64.json"
	cudaBuildType       = "BUILD_TYPE=cublas"
	cudaCapabilities    = "NVIDIA_DRIVER_CAPABILITIES=compute,utility"
	cudaAllCapabilities = "NVIDIA_DRIVER_CAPABILITIES=all"
	cudaHome            = "CUDA_HOME=/usr/local/cuda"
	cudaLibraryPath     = "LD_LIBRARY_PATH=/usr/local/cuda/lib64:"
	cudaPath            = "PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
	cudaVisibleDevices  = "NVIDIA_VISIBLE_DEVICES=all"
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
	Defaults      Defaults  `json:"defaults"`
	Entries       []Entry   `json:"entries"`
}

// Defaults defines the family and selector defaults owned by this snapshot.
type Defaults struct {
	Family    string            `json:"family"`
	Selectors []DefaultSelector `json:"selectors"`
}

// DefaultSelector maps one runtime and optional platform to its default LocalAI selector.
type DefaultSelector struct {
	Runtime  string    `json:"runtime"`
	Platform *Platform `json:"platform,omitempty"`
	Selector string    `json:"selector"`
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
	RuntimeBase       Artifact          `json:"runtimeBase"`
	RunnerRuntimeBase *Artifact         `json:"runnerRuntimeBase,omitempty"`
	Core              Artifact          `json:"core"`
	Backend           BackendArtifact   `json:"backend"`
	Fallbacks         []BackendArtifact `json:"fallbacks,omitempty"`
	Version           string            `json:"version"`
	SourceRef         string            `json:"sourceRef"`
	SystemPackages    []string          `json:"systemPackages,omitempty"`
	RuntimeSymlinks   []RuntimeSymlink  `json:"runtimeSymlinks,omitempty"`
	Environment       []string          `json:"environment,omitempty"`
	RunnerProfile     string            `json:"runnerProfile"`
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

// RuntimeSymlink is a compatibility link created in the selected runtime base.
type RuntimeSymlink struct {
	Target string `json:"target"`
	Path   string `json:"path"`
}

// ResolvedManifest identifies an immutable manifest and its platform.
type ResolvedManifest struct {
	Digest   string   `json:"digest"`
	Platform Platform `json:"platform"`
}
