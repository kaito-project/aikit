package utils // nolint:revive

const (
	RuntimeNVIDIA       = "cuda"
	RuntimeROCm         = "rocm"
	RuntimeAppleSilicon = "applesilicon" // experimental apple silicon runtime with vulkan arm64 support

	BackendDiffusers = "diffusers"
	BackendLlamaCpp  = "llama-cpp"
	BackendVLLM      = "vllm"

	BackendOCIRegistry = "quay.io/go-skynet/local-ai-backends"

	TargetUnsloth = "unsloth"

	DatasetAlpaca           = "alpaca"
	DatasetPromptCompletion = "prompt-completion"

	APIv1alpha1 = "v1alpha1"

	UbuntuBase       = "docker.io/library/ubuntu:22.04"
	Ubuntu24Base     = "docker.io/library/ubuntu:24.04"
	AppleSiliconBase = "ghcr.io/kaito-project/aikit/applesilicon/base:latest"
	CudaDevel        = "nvcr.io/nvidia/cuda:12.6.3-devel-ubuntu22.04@sha256:d49bb8a4ff97fb5fe477947a3f02aa8c0a53eae77e11f00ec28618a0bcaa2ad1"
	ROCmDevel        = "rocm/dev-ubuntu-22.04:7.2"

	PlatformLinux = "linux"
	PlatformAMD64 = "amd64"
	PlatformARM64 = "arm64"
)
