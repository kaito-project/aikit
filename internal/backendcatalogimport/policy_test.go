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
		{name: "Diffusers explicit CUDA 12", version: LocalAIVersion, family: familyDiffusers, selector: selectorNVIDIACUDA12, want: LocalAIVersion},
		{name: "Apple Silicon Vulkan", version: LocalAIVersion, family: runnerLlamaCpp, selector: targetVulkan, want: legacyLocalAIVersion},
		{name: "vLLM default CUDA", version: LocalAIVersion, family: familyVLLM, selector: selectorNVIDIA, want: LocalAIVersion},
		{name: "different imported release", version: fixtureFutureVersion, family: familyDiffusers, selector: selectorNVIDIA, want: fixtureFutureVersion},
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
		sourceRef         string
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
			sourceRef:         reviewedSourceCPULLM,
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
			sourceRef:         reviewedSourceCPULLM,
		},
		{
			name:              "llama CUDA",
			family:            runnerLlamaCpp,
			selector:          selectorNVIDIA,
			target:            backendTargetCUDALLM,
			architecture:      architectureAMD64,
			status:            statusSupported,
			runtimeBase:       chiseledRuntimeBase,
			runnerRuntimeBase: ubuntu22RuntimeBase,
			environment:       cuda12Environment,
			runner:            runnerLlamaCpp,
			fallbacks:         1,
			sourceRef:         "quay.io/go-skynet/local-ai-backends:v4.8.2-gpu-nvidia-cuda-12-llama-cpp",
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
			sourceRef:       "quay.io/go-skynet/local-ai-backends:v4.8.2-gpu-rocm-hipblas-llama-cpp",
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
			installName:  backendInstallVulkanLLM,
			sourceRef:    reviewedSourceVulkanLLM,
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
			target:       backendTargetL4TLLM,
			architecture: architectureARM64,
			status:       statusExperimental,
			runtimeBase:  l4tRuntimeBase,
			environment:  l4tEnvironment(minimumCUDA12),
			runner:       runnerLlamaCpp,
			fallbacks:    1,
			sourceRef:    reviewedSourceL4TLLM,
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
			sourceRef:    "quay.io/go-skynet/local-ai-backends:v3.12.1-gpu-nvidia-cuda-12-diffusers",
		},
		{
			name:           "vllm CUDA",
			family:         familyVLLM,
			selector:       selectorNVIDIA,
			target:         backendTargetCUDAVLLM,
			architecture:   architectureAMD64,
			status:         statusSupported,
			runtimeBase:    ubuntu22RuntimeBase,
			systemPackages: []string{systemPackageGCC, systemPackageLibcDev},
			environment:    append(cuda12Environment, vllmNativeSampler),
			runner:         runnerHFConfig,
			sourceRef:      "quay.io/go-skynet/local-ai-backends:v4.8.2-gpu-nvidia-cuda-12-vllm",
		},
		{
			name:           "vllm explicit CUDA 12 keeps native sampler",
			family:         familyVLLM,
			selector:       selectorNVIDIACUDA12,
			target:         backendTargetCUDAVLLM,
			architecture:   architectureAMD64,
			status:         statusExperimental,
			runtimeBase:    ubuntuRuntimeBase,
			systemPackages: []string{systemPackageGCC, systemPackageLibcDev},
			environment:    append(cuda12Environment, vllmNativeSampler),
			runner:         runnerUnsupported,
		},
		{
			name:           "vllm CUDA 13 does not apply CUDA 12 workaround",
			family:         familyVLLM,
			selector:       selectorNVIDIACUDA13,
			target:         "cuda13-vllm",
			architecture:   architectureAMD64,
			status:         statusExperimental,
			runtimeBase:    ubuntuRuntimeBase,
			systemPackages: []string{systemPackageGCC, systemPackageLibcDev},
			environment:    cuda13Environment,
			runner:         runnerUnsupported,
		},
		{
			name:           "vllm L4T does not apply CUDA 12 workaround",
			family:         familyVLLM,
			selector:       selectorNVIDIAL4T,
			target:         "nvidia-l4t-arm64-vllm",
			architecture:   architectureARM64,
			status:         statusExperimental,
			runtimeBase:    l4tRuntimeBase,
			systemPackages: []string{systemPackageGCC, systemPackageLibcDev},
			environment:    l4tEnvironment(minimumCUDA12),
			runner:         runnerUnsupported,
		},
		{
			name:              "vllm-cpp CPU",
			family:            familyVLLMCpp,
			selector:          selectorDefault,
			target:            backendTargetCPUVLLM,
			architecture:      architectureAMD64,
			status:            statusSupported,
			runtimeBase:       chiseledRuntimeBase,
			runnerRuntimeBase: ubuntu22RuntimeBase,
			runner:            familyVLLMCpp,
			sourceRef:         reviewedSourceCPUVLLM,
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
			sourceRef:         "quay.io/go-skynet/local-ai-backends:v4.8.2-gpu-nvidia-cuda-13-vllm-cpp",
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
		{
			name:         "Kokoro is quarantined until baked models are consumed",
			family:       familyKokoro,
			selector:     selectorDefault,
			target:       "cpu-kokoro",
			architecture: architectureAMD64,
			status:       statusQuarantined,
			runtimeBase:  ubuntuRuntimeBase,
			runner:       runnerUnsupported,
		},
		{
			name:         "SGLang CUDA 12 is quarantined for incompatible dependencies",
			family:       familySGLang,
			selector:     selectorNVIDIA,
			target:       "cuda12-sglang",
			architecture: architectureAMD64,
			status:       statusQuarantined,
			runtimeBase:  ubuntuRuntimeBase,
			environment:  cuda12Environment,
			runner:       runnerUnsupported,
		},
		{
			name:         "SGLang explicit CUDA 12 is also quarantined",
			family:       familySGLang,
			selector:     selectorNVIDIACUDA12,
			target:       "cuda12-sglang",
			architecture: architectureAMD64,
			status:       statusQuarantined,
			runtimeBase:  ubuntuRuntimeBase,
			environment:  cuda12Environment,
			runner:       runnerUnsupported,
		},
		{
			name:         "SGLang CUDA 13 remains experimental",
			family:       familySGLang,
			selector:     selectorNVIDIACUDA13,
			target:       "cuda13-sglang",
			architecture: architectureAMD64,
			status:       statusExperimental,
			runtimeBase:  ubuntuRuntimeBase,
			environment:  cuda13Environment,
			runner:       runnerUnsupported,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			policy, err := policyFor(policyInput{
				ImportedVersion: LocalAIVersion,
				Family:          test.family,
				Selector:        test.selector,
				Target:          test.target,
				SourceRef:       test.sourceRef,
				Platform:        Platform{OS: platformLinux, Architecture: test.architecture},
			})
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

func TestReviewedPolicyOverlayMatrix(t *testing.T) {
	if err := validateReviewedPolicyOverlays(reviewedPolicyOverlays); err != nil {
		t.Fatalf("validateReviewedPolicyOverlays() error = %v", err)
	}
	if got, want := len(reviewedPolicyOverlays), 13; got != want {
		t.Fatalf("reviewed overlay count = %d, want %d", got, want)
	}

	var supported, runnerEnabled, fallbacks int
	for _, overlay := range reviewedPolicyOverlays {
		name := overlay.Key.Family + "/" + overlay.Key.Selector + "/" + overlay.Key.Platform.key()
		t.Run(name, func(t *testing.T) {
			policy, err := policyFor(policyInput{
				ImportedVersion: overlay.Key.Version,
				Family:          overlay.Key.Family,
				Selector:        overlay.Key.Selector,
				Target:          overlay.Target,
				SourceRef:       overlay.SourceRef,
				Platform:        overlay.Key.Platform,
			})
			if err != nil {
				t.Fatalf("policyFor() error = %v", err)
			}
			if policy.TargetProfile != overlay.TargetProfile || policy.Status != overlay.Status || policy.RuntimeBaseRef != overlay.RuntimeBaseRef ||
				policy.RunnerRuntimeBaseRef != overlay.RunnerRuntimeBaseRef || policy.InstallName != overlay.InstallName ||
				policy.RunnerProfile != overlay.RunnerProfile || !slices.Equal(policy.Fallbacks, overlay.Fallbacks) {
				t.Fatalf("policyFor() = %#v, want overlay %#v", policy, overlay)
			}
		})
		if overlay.Status == statusSupported {
			supported++
		}
		if overlay.RunnerProfile != runnerUnsupported {
			runnerEnabled++
		}
		fallbacks += len(overlay.Fallbacks)
	}
	if supported != 8 || runnerEnabled != 12 || fallbacks != 5 {
		t.Fatalf("reviewed overlay totals = supported %d, runner-enabled %d, fallbacks %d; want 8, 12, 5", supported, runnerEnabled, fallbacks)
	}
}

func TestReviewedPolicyOverlayDriftFailsClosed(t *testing.T) {
	cpuOverlay := reviewedPolicyOverlays[0]

	t.Run("target", func(t *testing.T) {
		_, err := policyFor(policyInput{
			ImportedVersion: cpuOverlay.Key.Version,
			Family:          cpuOverlay.Key.Family,
			Selector:        cpuOverlay.Key.Selector,
			Target:          "cpu-renamed-llama-cpp",
			SourceRef:       cpuOverlay.SourceRef,
			Platform:        cpuOverlay.Key.Platform,
		})
		if err == nil || !strings.Contains(err.Error(), "reviewed policy target drift") {
			t.Fatalf("policyFor() error = %v, want target drift", err)
		}
	})

	t.Run("source reference", func(t *testing.T) {
		_, err := policyFor(policyInput{
			ImportedVersion: cpuOverlay.Key.Version,
			Family:          cpuOverlay.Key.Family,
			Selector:        cpuOverlay.Key.Selector,
			Target:          cpuOverlay.Target,
			SourceRef:       "quay.io/go-skynet/local-ai-backends:v4.8.2-cpu-repacked-llama-cpp",
			Platform:        cpuOverlay.Key.Platform,
		})
		if err == nil || !strings.Contains(err.Error(), "reviewed policy source reference drift") {
			t.Fatalf("policyFor() error = %v, want source reference drift", err)
		}
	})

	t.Run("future release", func(t *testing.T) {
		policy, err := policyFor(policyInput{
			ImportedVersion: fixtureFutureVersion,
			Family:          cpuOverlay.Key.Family,
			Selector:        cpuOverlay.Key.Selector,
			Target:          cpuOverlay.Target,
			SourceRef:       "quay.io/go-skynet/local-ai-backends:" + fixtureFutureVersion + "-cpu-llama-cpp",
			Platform:        cpuOverlay.Key.Platform,
		})
		if err != nil {
			t.Fatalf("policyFor() error = %v", err)
		}
		if policy.Status != statusExperimental || policy.RunnerProfile != runnerUnsupported || policy.RuntimeBaseRef != ubuntuRuntimeBase ||
			policy.RunnerRuntimeBaseRef != "" {
			t.Fatalf("future release policy = %#v, want unreviewed defaults", policy)
		}
	})

	t.Run("architecture", func(t *testing.T) {
		nvidiaOverlay := reviewedPolicyOverlays[2]
		policy, err := policyFor(policyInput{
			ImportedVersion: nvidiaOverlay.Key.Version,
			Family:          nvidiaOverlay.Key.Family,
			Selector:        nvidiaOverlay.Key.Selector,
			Target:          nvidiaOverlay.Target,
			SourceRef:       nvidiaOverlay.SourceRef,
			Platform:        linuxPlatform(architectureARM64),
		})
		if err != nil {
			t.Fatalf("policyFor() error = %v", err)
		}
		if policy.Status != statusExperimental || policy.RunnerProfile != runnerUnsupported || policy.RuntimeBaseRef != ubuntuRuntimeBase {
			t.Fatalf("architecture drift policy = %#v, want unreviewed defaults", policy)
		}
	})

	t.Run("variant", func(t *testing.T) {
		platform := cpuOverlay.Key.Platform
		platform.Variant = "v3"
		policy, err := policyFor(policyInput{
			ImportedVersion: cpuOverlay.Key.Version,
			Family:          cpuOverlay.Key.Family,
			Selector:        cpuOverlay.Key.Selector,
			Target:          cpuOverlay.Target,
			SourceRef:       cpuOverlay.SourceRef,
			Platform:        platform,
		})
		if err != nil {
			t.Fatalf("policyFor() error = %v", err)
		}
		if policy.Status != statusExperimental || policy.RunnerProfile != runnerUnsupported || policy.RuntimeBaseRef != ubuntuRuntimeBase {
			t.Fatalf("variant drift policy = %#v, want unreviewed defaults", policy)
		}
	})
}

func TestValidateReviewedPolicyOverlays(t *testing.T) {
	valid := reviewedPolicyOverlays[0]
	tests := []struct {
		name     string
		overlays []reviewedPolicyOverlay
		wantErr  string
	}{
		{name: "valid", overlays: []reviewedPolicyOverlay{valid}},
		{name: "empty", wantErr: "are empty"},
		{
			name: "incomplete",
			overlays: []reviewedPolicyOverlay{func() reviewedPolicyOverlay {
				incomplete := valid
				incomplete.SourceRef = ""
				return incomplete
			}()},
			wantErr: "is incomplete",
		},
		{name: "duplicate", overlays: []reviewedPolicyOverlay{valid, valid}, wantErr: "is duplicated"},
		{
			name: "target profile mismatch",
			overlays: []reviewedPolicyOverlay{func() reviewedPolicyOverlay {
				mismatch := valid
				mismatch.TargetProfile = targetCUDA12
				return mismatch
			}()},
			wantErr: "inferred",
		},
		{
			name: "noncanonical platform",
			overlays: []reviewedPolicyOverlay{func() reviewedPolicyOverlay {
				noncanonical := valid
				noncanonical.Key.Platform = Platform{OS: "LINUX", Architecture: architectureX8664}
				return noncanonical
			}()},
			wantErr: "noncanonical platform",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := validateReviewedPolicyOverlays(test.overlays)
			if test.wantErr == "" {
				if err != nil {
					t.Fatalf("validateReviewedPolicyOverlays() error = %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), test.wantErr) {
				t.Fatalf("validateReviewedPolicyOverlays() error = %v, want containing %q", err, test.wantErr)
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
			policy, err := policyFor(policyInput{
				ImportedVersion: LocalAIVersion,
				Family:          fixtureFamilyDemo,
				Selector:        test.selector,
				Target:          test.target,
				Platform:        Platform{OS: platformLinux, Architecture: test.architecture},
			})
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
	policy, err := policyFor(policyInput{
		ImportedVersion: LocalAIVersion,
		Family:          familyVLLMCpp,
		Selector:        selectorNVIDIAL4T,
		Target:          "nvidia-l4t-arm64-vllm-cpp",
		SourceRef:       "quay.io/go-skynet/local-ai-backends:v4.8.2-nvidia-l4t-cuda-13-arm64-vllm-cpp",
		Platform:        Platform{OS: platformLinux, Architecture: architectureARM64},
	})
	if err != nil {
		t.Fatalf("policyFor() error = %v", err)
	}
	if policy.TargetProfile != targetL4TCUDA13 || policy.RuntimeBaseRef != ubuntuRuntimeBase ||
		!slices.Equal(policy.Environment, l4tEnvironment(minimumCUDA13)) {
		t.Fatalf("policyFor() = %#v, want CUDA 13 L4T policy", policy)
	}
}
