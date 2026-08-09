package backendcatalogimport

import (
	"slices"
	"testing"
)

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
		name            string
		family          string
		selector        string
		target          string
		architecture    string
		status          string
		runtimeBase     string
		systemPackages  []string
		runtimeSymlinks []RuntimeSymlink
		environment     []string
		runner          string
		fallbacks       int
	}{
		{
			name:         "llama CPU amd64",
			family:       runnerLlamaCpp,
			selector:     selectorDefault,
			target:       fixtureCPULlamaCpp,
			architecture: architectureAMD64,
			status:       statusSupported,
			runtimeBase:  ubuntuRuntimeBase,
			runner:       runnerLlamaCpp,
		},
		{
			name:         "llama CPU arm64",
			family:       runnerLlamaCpp,
			selector:     selectorDefault,
			target:       fixtureCPULlamaCpp,
			architecture: architectureARM64,
			status:       statusSupported,
			runtimeBase:  ubuntuRuntimeBase,
			runner:       runnerLlamaCpp,
		},
		{
			name:         "llama CUDA",
			family:       runnerLlamaCpp,
			selector:     selectorNVIDIA,
			target:       "cuda12-llama-cpp",
			architecture: architectureAMD64,
			status:       statusSupported,
			runtimeBase:  ubuntuRuntimeBase,
			environment:  cuda12Environment,
			runner:       runnerLlamaCpp,
			fallbacks:    1,
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
			runner:          runnerUnsupported,
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
			name:         "unreviewed L4T CUDA 12 uses runtime base",
			family:       fixtureFamilyDemo,
			selector:     selectorNVIDIAL4T,
			target:       "nvidia-l4t-arm64-demo",
			architecture: architectureARM64,
			status:       statusExperimental,
			runtimeBase:  l4tRuntimeBase,
			environment:  l4tEnvironment(minimumCUDA12),
			runner:       runnerUnsupported,
		},
		{
			name:         "unreviewed L4T CUDA 13 uses runtime base",
			family:       fixtureFamilyDemo,
			selector:     selectorL4TCUDA13,
			target:       "cuda13-nvidia-l4t-arm64-demo",
			architecture: architectureARM64,
			status:       statusExperimental,
			runtimeBase:  l4tRuntimeBase,
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
			runtimeBase:  ubuntuRuntimeBase,
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
			runtimeBase:    ubuntuRuntimeBase,
			systemPackages: []string{"gcc", "libc6-dev"},
			environment:    cuda12Environment,
			runner:         runnerHFConfig,
		},
		{
			name:         "vllm-cpp CPU",
			family:       familyVLLMCpp,
			selector:     selectorDefault,
			target:       "cpu-vllm-cpp",
			architecture: architectureAMD64,
			status:       statusSupported,
			runtimeBase:  ubuntuRuntimeBase,
			runner:       familyVLLMCpp,
		},
		{
			name:         "vllm-cpp CUDA",
			family:       familyVLLMCpp,
			selector:     selectorNVIDIA,
			target:       "cuda13-vllm-cpp",
			architecture: architectureAMD64,
			status:       statusSupported,
			runtimeBase:  ubuntuRuntimeBase,
			environment:  cuda13Environment,
			runner:       familyVLLMCpp,
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
			policy, err := policyFor(test.family, test.selector, test.target, Platform{OS: platformLinux, Architecture: test.architecture})
			if err != nil {
				t.Fatalf("policyFor() error = %v", err)
			}
			if policy.Status != test.status || policy.RuntimeBaseRef != test.runtimeBase || !slices.Equal(policy.SystemPackages, test.systemPackages) ||
				!slices.Equal(policy.RuntimeSymlinks, test.runtimeSymlinks) ||
				!slices.Equal(policy.Environment, test.environment) || policy.RunnerProfile != test.runner || len(policy.Fallbacks) != test.fallbacks {
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
			policy, err := policyFor(fixtureFamilyDemo, test.selector, test.target, Platform{OS: platformLinux, Architecture: test.architecture})
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
