package utils // nolint:revive

const (
	RuntimeNVIDIA       = "cuda"
	RuntimeAppleSilicon = "applesilicon" // experimental apple silicon runtime with vulkan arm64 support

	BackendExllamaV2 = "exllama2"
	BackendDiffusers = "diffusers"
	BackendLlamaCpp  = "llama-cpp"

	BackendOCIRegistry = "quay.io/go-skynet/local-ai-backends"

	TargetUnsloth = "unsloth"

	DatasetAlpaca = "alpaca"

	APIv1alpha1 = "v1alpha1"

	UbuntuBase       = "docker.io/library/ubuntu:22.04"
	AppleSiliconBase = "ghcr.io/kaito-project/aikit/applesilicon/base:latest"
	CudaDevel        = "nvcr.io/nvidia/cuda:12.3.2-devel-ubuntu22.04"

	PlatformLinux = "linux"
	PlatformAMD64 = "amd64"
	PlatformARM64 = "arm64"
)
