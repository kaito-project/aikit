package backendcatalogimport

import (
	"slices"
	"strings"
	"testing"
)

func TestCompatibilityArtifactVersions(t *testing.T) {
	tests := []struct {
		name     string
		version  string
		family   string
		selector string
		want     string
	}{
		{name: "Diffusers default CUDA", version: LocalAIVersion, family: familyDiffusers, selector: selectorNVIDIA, want: legacyLocalAIVersion},
		{name: "Diffusers explicit CUDA 12", version: LocalAIVersion, family: familyDiffusers, selector: "nvidia-cuda-12", want: LocalAIVersion},
		{name: "Apple Silicon Vulkan", version: LocalAIVersion, family: runnerLlamaCpp, selector: targetVulkan, want: legacyLocalAIVersion},
		{name: "vLLM default CUDA", version: LocalAIVersion, family: familyVLLM, selector: selectorNVIDIA, want: LocalAIVersion},
		{name: "different imported release", version: "v5.0.0", family: familyDiffusers, selector: selectorNVIDIA, want: "v5.0.0"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := artifactVersionFor(test.version, test.family, test.selector); got != test.want {
				t.Errorf("artifactVersionFor() = %q, want %q", got, test.want)
			}
		})
	}

	const template = "registry.example/localai:" + LocalAIVersion + "-{architecture}"
	got, err := coreReferenceTemplateForVersion(template, LocalAIVersion, legacyLocalAIVersion)
	if err != nil {
		t.Fatalf("coreReferenceTemplateForVersion() error = %v", err)
	}
	if want := "registry.example/localai:" + legacyLocalAIVersion + "-{architecture}"; got != want {
		t.Errorf("coreReferenceTemplateForVersion() = %q, want %q", got, want)
	}
	if _, err := coreReferenceTemplateForVersion("registry.example/localai:stable-{architecture}", LocalAIVersion, legacyLocalAIVersion); err == nil ||
		!strings.Contains(err.Error(), "must contain imported version") {
		t.Fatalf("coreReferenceTemplateForVersion() error = %v, want missing imported version", err)
	}
}

func TestReviewedPolicyOverlay(t *testing.T) {
	cuda12Environment := []string{
		cudaBuildType,
		cudaCapabilities,
		"NVIDIA_REQUIRE_CUDA=cuda>=12.0",
		cudaVisibleDevices,
	}
	cuda13Environment := []string{
		cudaBuildType,
		cudaCapabilities,
		"NVIDIA_REQUIRE_CUDA=cuda>=13.0",
		cudaVisibleDevices,
	}
	rocmEnvironment := []string{
		"LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:/opt/rocm/llvm/lib",
		"LOCALAI_FORCE_META_BACKEND_CAPABILITY=amd",
		"PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/rocm/bin",
	}
	rocmPackages := []string{"pciutils"}
	tests := []struct {
		name              string
		family            string
		selector          string
		target            string
		architecture      string
		status            string
		runtimeBase       string
		runnerRuntimeBase string
		systemPackages    []string
		runtimeSymlinks   []RuntimeSymlink
		environment       []string
		runner            string
		installName       string
		fallbacks         int
	}{
		{
			name:              "llama CPU amd64",
			family:            runnerLlamaCpp,
			selector:          selectorDefault,
			target:            fixtureCPULlamaCpp,
			architecture:      architectureAMD64,
			status:            statusSupported,
			runtimeBase:       chiseledRuntimeBase,
			runnerRuntimeBase: ubuntu22RuntimeBase,
			runner:            runnerLlamaCpp,
		},
		{
			name:              "llama CPU arm64",
			family:            runnerLlamaCpp,
			selector:          selectorDefault,
			target:            fixtureCPULlamaCpp,
			architecture:      architectureARM64,
			status:            statusSupported,
			runtimeBase:       chiseledRuntimeBase,
			runnerRuntimeBase: ubuntu22RuntimeBase,
			runner:            runnerLlamaCpp,
		},
		{
			name:              "llama CUDA",
			family:            runnerLlamaCpp,
			selector:          selectorNVIDIA,
			target:            "cuda12-llama-cpp",
			architecture:      architectureAMD64,
			status:            statusSupported,
			runtimeBase:       chiseledRuntimeBase,
			runnerRuntimeBase: ubuntu22RuntimeBase,
			environment:       cuda12Environment,
			runner:            runnerLlamaCpp,
			fallbacks:         1,
		},
		{
			name:            "llama ROCm",
			family:          runnerLlamaCpp,
			selector:        selectorAMD,
			target:          "rocm-llama-cpp",
			architecture:    architectureAMD64,
			status:          statusExperimental,
			runtimeBase:     rocmRuntimeBase,
			systemPackages:  rocmPackages,
			runtimeSymlinks: rocmRuntimeSymlinks,
			environment:     rocmEnvironment,
			runner:          runnerLlamaCpp,
			installName:     "hipblas-llama-cpp",
			fallbacks:       1,
		},
		{
			name:         "llama Vulkan",
			family:       runnerLlamaCpp,
			selector:     targetVulkan,
			target:       "vulkan-llama-cpp",
			architecture: architectureAMD64,
			status:       statusExperimental,
			runtimeBase:  ubuntuRuntimeBase,
			runner:       runnerUnsupported,
		},
		{
			name:         "llama Apple Silicon keeps legacy install path",
			family:       runnerLlamaCpp,
			selector:     targetVulkan,
			target:       "vulkan-llama-cpp",
			architecture: architectureARM64,
			status:       statusExperimental,
			runtimeBase:  vulkanRuntimeBase,
			environment:  []string{vulkanEnvironment},
			runner:       runnerUnsupported,
			installName:  "gpu-vulkan-llama-cpp",
		},
		{
			name:            "unreviewed ROCm uses runtime base",
			family:          fixtureFamilyDemo,
			selector:        selectorAMD,
			target:          "rocm-demo",
			architecture:    architectureAMD64,
			status:          statusExperimental,
			runtimeBase:     rocmRuntimeBase,
			systemPackages:  rocmPackages,
			runtimeSymlinks: rocmRuntimeSymlinks,
			environment:     rocmEnvironment,
			runner:          runnerUnsupported,
		},
		{
			name:         "unreviewed Vulkan uses runtime base",
			family:       fixtureFamilyDemo,
			selector:     targetVulkan,
			target:       fixtureVulkanTarget,
			architecture: architectureARM64,
			status:       statusExperimental,
			runtimeBase:  vulkanRuntimeBase,
			environment:  []string{vulkanEnvironment},
			runner:       runnerUnsupported,
		},
		{
			name:         "llama L4T CUDA 12 keeps runner compatibility",
			family:       runnerLlamaCpp,
			selector:     selectorNVIDIAL4T,
			target:       "nvidia-l4t-arm64-llama-cpp",
			architecture: architectureARM64,
			status:       statusExperimental,
			runtimeBase:  l4tRuntimeBase,
			environment:  l4tEnvironment(minimumCUDA12),
			runner:       runnerLlamaCpp,
			fallbacks:    1,
		},
		{
			name:         "unreviewed L4T CUDA 13 uses runtime base",
			family:       fixtureFamilyDemo,
			selector:     selectorL4TCUDA13,
			target:       "cuda13-nvidia-l4t-arm64-demo",
			architecture: architectureARM64,
			status:       statusExperimental,
			runtimeBase:  ubuntuRuntimeBase,
			environment:  l4tEnvironment(minimumCUDA13),
			runner:       runnerUnsupported,
		},
		{
			name:         "diffusers CUDA",
			family:       familyDiffusers,
			selector:     selectorNVIDIA,
			target:       "cuda12-diffusers",
			architecture: architectureAMD64,
			status:       statusSupported,
			runtimeBase:  ubuntu22RuntimeBase,
			environment:  cuda12Environment,
			runner:       runnerHFConfig,
		},
		{
			name:           "vllm CUDA",
			family:         familyVLLM,
			selector:       selectorNVIDIA,
			target:         "cuda12-vllm",
			architecture:   architectureAMD64,
			status:         statusSupported,
			runtimeBase:    ubuntu22RuntimeBase,
			systemPackages: []string{"gcc", "libc6-dev"},
			environment:    append(cuda12Environment, vllmNativeSampler),
			runner:         runnerHFConfig,
		},
		{
			name:              "vllm-cpp CPU",
			family:            familyVLLMCpp,
			selector:          selectorDefault,
			target:            "cpu-vllm-cpp",
			architecture:      architectureAMD64,
			status:            statusSupported,
			runtimeBase:       chiseledRuntimeBase,
			runnerRuntimeBase: ubuntu22RuntimeBase,
			runner:            familyVLLMCpp,
		},
		{
			name:              "vllm-cpp CUDA",
			family:            familyVLLMCpp,
			selector:          selectorNVIDIA,
			target:            "cuda13-vllm-cpp",
			architecture:      architectureAMD64,
			status:            statusSupported,
			runtimeBase:       chiseledRuntimeBase,
			runnerRuntimeBase: ubuntu22RuntimeBase,
			environment:       cuda13Environment,
			runner:            familyVLLMCpp,
		},
		{
			name:         "NVIDIA-routed Vulkan exposes graphics",
			family:       fixtureFamilyDemo,
			selector:     selectorNVIDIA,
			target:       fixtureVulkanTarget,
			architecture: architectureAMD64,
			status:       statusExperimental,
			runtimeBase:  ubuntuRuntimeBase,
			environment:  cudaEnvironmentWithCapabilities(minimumCUDA12, cudaAllCapabilities),
			runner:       runnerUnsupported,
		},
		{
			name:         "unreviewed tuple defaults",
			family:       fixtureFamilyDemo,
			selector:     selectorNVIDIA,
			target:       "cuda12-demo",
			architecture: architectureAMD64,
			status:       statusExperimental,
			runtimeBase:  ubuntuRuntimeBase,
			environment:  cuda12Environment,
			runner:       runnerUnsupported,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			policy, err := policyFor(test.family, test.selector, test.target, "", Platform{OS: platformLinux, Architecture: test.architecture})
			if err != nil {
				t.Fatalf("policyFor() error = %v", err)
			}
			if policy.Status != test.status || policy.RuntimeBaseRef != test.runtimeBase || policy.RunnerRuntimeBaseRef != test.runnerRuntimeBase ||
				!slices.Equal(policy.SystemPackages, test.systemPackages) ||
				!slices.Equal(policy.RuntimeSymlinks, test.runtimeSymlinks) ||
				!slices.Equal(policy.Environment, test.environment) || policy.RunnerProfile != test.runner || policy.InstallName != test.installName ||
				len(policy.Fallbacks) != test.fallbacks {
				t.Fatalf("policyFor() = %#v", policy)
			}
		})
	}
}

func TestVulkanAndMetalRuntimeClassification(t *testing.T) {
	tests := []struct {
		name            string
		selector        string
		target          string
		architecture    string
		wantRuntime     string
		wantTarget      string
		wantRuntimeBase string
		wantEnvironment []string
	}{
		{
			name:            "Vulkan amd64",
			selector:        targetVulkan,
			target:          fixtureVulkanTarget,
			architecture:    architectureAMD64,
			wantRuntime:     runtimeCPU,
			wantTarget:      targetVulkan,
			wantRuntimeBase: ubuntuRuntimeBase,
		},
		{
			name:            fixtureVulkanARM64Name,
			selector:        targetVulkan,
			target:          fixtureVulkanTarget,
			architecture:    architectureARM64,
			wantRuntime:     runtimeApple,
			wantTarget:      targetVulkan,
			wantRuntimeBase: vulkanRuntimeBase,
			wantEnvironment: []string{vulkanEnvironment},
		},
		{
			name:            "Metal remains Apple Silicon",
			selector:        targetMetal,
			target:          "metal-demo",
			architecture:    architectureARM64,
			wantRuntime:     runtimeApple,
			wantTarget:      targetMetal,
			wantRuntimeBase: vulkanRuntimeBase,
			wantEnvironment: []string{vulkanEnvironment},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			policy, err := policyFor(fixtureFamilyDemo, test.selector, test.target, "", Platform{OS: platformLinux, Architecture: test.architecture})
			if err != nil {
				t.Fatalf("policyFor() error = %v", err)
			}
			if policy.Runtime != test.wantRuntime || policy.TargetProfile != test.wantTarget || policy.RuntimeBaseRef != test.wantRuntimeBase ||
				!slices.Equal(policy.Environment, test.wantEnvironment) {
				t.Fatalf("policyFor() = %#v, want runtime %q target %q runtime base %q environment %v", policy, test.wantRuntime, test.wantTarget, test.wantRuntimeBase, test.wantEnvironment)
			}
		})
	}
}

func TestGenericL4TProfileFollowsArtifactCUDA(t *testing.T) {
	policy, err := policyFor(
		familyVLLMCpp,
		selectorNVIDIAL4T,
		"nvidia-l4t-arm64-vllm-cpp",
		"quay.io/go-skynet/local-ai-backends:v4.8.2-nvidia-l4t-cuda-13-arm64-vllm-cpp",
		Platform{OS: platformLinux, Architecture: architectureARM64},
	)
	if err != nil {
		t.Fatalf("policyFor() error = %v", err)
	}
	if policy.TargetProfile != targetL4TCUDA13 || policy.RuntimeBaseRef != ubuntuRuntimeBase ||
		!slices.Equal(policy.Environment, l4tEnvironment(minimumCUDA13)) {
		t.Fatalf("policyFor() = %#v, want CUDA 13 L4T policy", policy)
	}
}
