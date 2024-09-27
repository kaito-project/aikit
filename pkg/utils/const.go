package utils

const (
	RuntimeNVIDIA = "cuda"

	BackendStableDiffusion = "stablediffusion"
	BackendExllamaV2       = "exllama2"
	BackendMamba           = "mamba"
	BackendDiffusers       = "diffusers"

	TargetUnsloth = "unsloth"

	DatasetAlpaca = "alpaca"

	APIv1alpha1 = "v1alpha1"

	UbuntuBase = "docker.io/library/ubuntu:22.04"
	CudaDevel  = "nvcr.io/nvidia/cuda:12.3.2-devel-ubuntu22.04"

	PlatformLinux = "linux"
	PlatformAMD64 = "amd64"
	PlatformARM64 = "arm64"
)
