package backendcatalogimport

import (
	"fmt"
	"sort"
	"strings"

	"github.com/pkg/errors"
)

type fallbackTarget struct {
	Family   string
	Selector string
}

type entryPolicy struct {
	Runtime              string
	TargetProfile        string
	Status               string
	RuntimeBaseRef       string
	RunnerRuntimeBaseRef string
	InstallName          string
	SystemPackages       []string
	RuntimeSymlinks      []RuntimeSymlink
	Environment          []string
	RunnerProfile        string
	Fallbacks            []fallbackTarget
}

type policyInput struct {
	ImportedVersion string
	Family          string
	Selector        string
	Target          string
	SourceRef       string
	Platform        Platform
}

type reviewedPolicyKey struct {
	Version  string
	Family   string
	Selector string
	Platform Platform
}

type reviewedPolicyOverlay struct {
	Key                  reviewedPolicyKey
	Target               string
	SourceRef            string
	TargetProfile        string
	Status               string
	RuntimeBaseRef       string
	RunnerRuntimeBaseRef string
	InstallName          string
	RunnerProfile        string
	Fallbacks            []fallbackTarget
}

var reviewedPolicyOverlays = []reviewedPolicyOverlay{
	{
		Key:                  reviewedPolicyKey{Version: reviewedLocalAIVersion, Family: runnerLlamaCpp, Selector: selectorDefault, Platform: linuxPlatform(architectureAMD64)},
		Target:               "cpu-llama-cpp",
		SourceRef:            reviewedSourceCPULLM,
		TargetProfile:        runtimeCPU,
		Status:               statusSupported,
		RuntimeBaseRef:       chiseledRuntimeBase,
		RunnerRuntimeBaseRef: ubuntu22RuntimeBase,
		RunnerProfile:        runnerLlamaCpp,
	},
	{
		Key:                  reviewedPolicyKey{Version: reviewedLocalAIVersion, Family: runnerLlamaCpp, Selector: selectorDefault, Platform: linuxPlatform(architectureARM64)},
		Target:               "cpu-llama-cpp",
		SourceRef:            reviewedSourceCPULLM,
		TargetProfile:        runtimeCPU,
		Status:               statusSupported,
		RuntimeBaseRef:       chiseledRuntimeBase,
		RunnerRuntimeBaseRef: ubuntu22RuntimeBase,
		RunnerProfile:        runnerLlamaCpp,
	},
	{
		Key:                  reviewedPolicyKey{Version: reviewedLocalAIVersion, Family: runnerLlamaCpp, Selector: selectorNVIDIA, Platform: linuxPlatform(architectureAMD64)},
		Target:               backendTargetCUDALLM,
		SourceRef:            "quay.io/go-skynet/local-ai-backends:v4.8.2-gpu-nvidia-cuda-12-llama-cpp",
		TargetProfile:        targetCUDA12,
		Status:               statusSupported,
		RuntimeBaseRef:       chiseledRuntimeBase,
		RunnerRuntimeBaseRef: ubuntu22RuntimeBase,
		RunnerProfile:        runnerLlamaCpp,
		Fallbacks:            []fallbackTarget{{Family: runnerLlamaCpp, Selector: selectorDefault}},
	},
	{
		Key:            reviewedPolicyKey{Version: reviewedLocalAIVersion, Family: runnerLlamaCpp, Selector: selectorAMD, Platform: linuxPlatform(architectureAMD64)},
		Target:         "rocm-llama-cpp",
		SourceRef:      "quay.io/go-skynet/local-ai-backends:v4.8.2-gpu-rocm-hipblas-llama-cpp",
		TargetProfile:  targetROCm,
		Status:         statusExperimental,
		RuntimeBaseRef: rocmRuntimeBase,
		InstallName:    "hipblas-llama-cpp",
		RunnerProfile:  runnerLlamaCpp,
		Fallbacks:      []fallbackTarget{{Family: runnerLlamaCpp, Selector: selectorDefault}},
	},
	{
		Key:            reviewedPolicyKey{Version: reviewedLocalAIVersion, Family: runnerLlamaCpp, Selector: selectorNVIDIAL4T, Platform: linuxPlatform(architectureARM64)},
		Target:         backendTargetL4TLLM,
		SourceRef:      reviewedSourceL4TLLM,
		TargetProfile:  targetL4TCUDA12,
		Status:         statusExperimental,
		RuntimeBaseRef: l4tRuntimeBase,
		RunnerProfile:  runnerLlamaCpp,
		Fallbacks:      []fallbackTarget{{Family: runnerLlamaCpp, Selector: selectorDefault}},
	},
	{
		Key:            reviewedPolicyKey{Version: reviewedLocalAIVersion, Family: runnerLlamaCpp, Selector: selectorL4TCUDA12, Platform: linuxPlatform(architectureARM64)},
		Target:         backendTargetL4TLLM,
		SourceRef:      reviewedSourceL4TLLM,
		TargetProfile:  targetL4TCUDA12,
		Status:         statusExperimental,
		RuntimeBaseRef: l4tRuntimeBase,
		RunnerProfile:  runnerLlamaCpp,
		Fallbacks:      []fallbackTarget{{Family: runnerLlamaCpp, Selector: selectorDefault}},
	},
	{
		Key:            reviewedPolicyKey{Version: reviewedLocalAIVersion, Family: runnerLlamaCpp, Selector: selectorL4TCUDA13, Platform: linuxPlatform(architectureARM64)},
		Target:         "cuda13-nvidia-l4t-arm64-llama-cpp",
		SourceRef:      "quay.io/go-skynet/local-ai-backends:v4.8.2-nvidia-l4t-cuda-13-arm64-llama-cpp",
		TargetProfile:  targetL4TCUDA13,
		Status:         statusExperimental,
		RuntimeBaseRef: ubuntuRuntimeBase,
		RunnerProfile:  runnerLlamaCpp,
		Fallbacks:      []fallbackTarget{{Family: runnerLlamaCpp, Selector: selectorDefault}},
	},
	{
		Key:            reviewedPolicyKey{Version: reviewedLocalAIVersion, Family: runnerLlamaCpp, Selector: targetVulkan, Platform: linuxPlatform(architectureARM64)},
		Target:         backendTargetVulkanLLM,
		SourceRef:      reviewedSourceVulkanLLM,
		TargetProfile:  targetVulkan,
		Status:         statusExperimental,
		RuntimeBaseRef: vulkanRuntimeBase,
		InstallName:    backendInstallVulkanLLM,
		RunnerProfile:  runnerUnsupported,
	},
	{
		Key:            reviewedPolicyKey{Version: reviewedLocalAIVersion, Family: familyDiffusers, Selector: selectorNVIDIA, Platform: linuxPlatform(architectureAMD64)},
		Target:         "cuda12-diffusers",
		SourceRef:      "quay.io/go-skynet/local-ai-backends:v3.12.1-gpu-nvidia-cuda-12-diffusers",
		TargetProfile:  targetCUDA12,
		Status:         statusSupported,
		RuntimeBaseRef: ubuntu22RuntimeBase,
		RunnerProfile:  runnerHFConfig,
	},
	{
		Key:            reviewedPolicyKey{Version: reviewedLocalAIVersion, Family: familyVLLM, Selector: selectorNVIDIA, Platform: linuxPlatform(architectureAMD64)},
		Target:         backendTargetCUDAVLLM,
		SourceRef:      "quay.io/go-skynet/local-ai-backends:v4.8.2-gpu-nvidia-cuda-12-vllm",
		TargetProfile:  targetCUDA12,
		Status:         statusSupported,
		RuntimeBaseRef: ubuntu22RuntimeBase,
		RunnerProfile:  runnerHFConfig,
	},
	{
		Key:                  reviewedPolicyKey{Version: reviewedLocalAIVersion, Family: familyVLLMCpp, Selector: selectorDefault, Platform: linuxPlatform(architectureAMD64)},
		Target:               backendTargetCPUVLLM,
		SourceRef:            reviewedSourceCPUVLLM,
		TargetProfile:        runtimeCPU,
		Status:               statusSupported,
		RuntimeBaseRef:       chiseledRuntimeBase,
		RunnerRuntimeBaseRef: ubuntu22RuntimeBase,
		RunnerProfile:        familyVLLMCpp,
	},
	{
		Key:                  reviewedPolicyKey{Version: reviewedLocalAIVersion, Family: familyVLLMCpp, Selector: selectorDefault, Platform: linuxPlatform(architectureARM64)},
		Target:               backendTargetCPUVLLM,
		SourceRef:            reviewedSourceCPUVLLM,
		TargetProfile:        runtimeCPU,
		Status:               statusSupported,
		RuntimeBaseRef:       chiseledRuntimeBase,
		RunnerRuntimeBaseRef: ubuntu22RuntimeBase,
		RunnerProfile:        familyVLLMCpp,
	},
	{
		Key:                  reviewedPolicyKey{Version: reviewedLocalAIVersion, Family: familyVLLMCpp, Selector: selectorNVIDIA, Platform: linuxPlatform(architectureAMD64)},
		Target:               "cuda13-vllm-cpp",
		SourceRef:            "quay.io/go-skynet/local-ai-backends:v4.8.2-gpu-nvidia-cuda-13-vllm-cpp",
		TargetProfile:        targetCUDA13,
		Status:               statusSupported,
		RuntimeBaseRef:       chiseledRuntimeBase,
		RunnerRuntimeBaseRef: ubuntu22RuntimeBase,
		RunnerProfile:        familyVLLMCpp,
	},
}

var reviewedSystemPackages = map[string][]string{
	familyVLLM: {systemPackageGCC, systemPackageLibcDev},
}

var rocmRuntimeSymlinks = []RuntimeSymlink{{
	Target: "libhipblaslt.so.1",
	Path:   "/opt/rocm/lib/libhipblaslt.so.0",
}}

func linuxPlatform(architecture string) Platform {
	return Platform{OS: platformLinux, Architecture: architecture}
}

func hasReviewedOverlayMapping(version, family, selector, target, sourceRef string) bool {
	for _, overlay := range reviewedPolicyOverlays {
		if overlay.Key.Version == version && overlay.Key.Family == family && overlay.Key.Selector == selector && overlay.Target == target && overlay.SourceRef == sourceRef {
			return true
		}
	}

	return false
}

func artifactVersionFor(requestedVersion, family, selector string) string {
	if requestedVersion != reviewedLocalAIVersion {
		return requestedVersion
	}

	switch family + "/" + selector {
	case familyDiffusers + "/" + selectorNVIDIA, runnerLlamaCpp + "/" + targetVulkan:
		return legacyLocalAIVersion
	default:
		return requestedVersion
	}
}

func coreReferenceTemplateForVersion(template, requestedVersion, entryVersion string) (string, error) {
	if requestedVersion == entryVersion {
		return template, nil
	}
	if strings.Count(template, requestedVersion) != 1 {
		return "", fmt.Errorf("core reference template %q must contain imported version %q exactly once for compatibility pin %q", template, requestedVersion, entryVersion)
	}

	return strings.Replace(template, requestedVersion, entryVersion, 1), nil
}

func policyFor(input policyInput) (entryPolicy, error) {
	targetProfile, err := inferTargetProfile(input.Selector, input.Target, input.SourceRef)
	if err != nil {
		return entryPolicy{}, err
	}

	policy := entryPolicy{
		Runtime:        inferRuntime(targetProfile, input.Platform),
		TargetProfile:  targetProfile,
		Status:         statusExperimental,
		RuntimeBaseRef: runtimeBaseReference(targetProfile, input.Platform),
		SystemPackages: append([]string(nil), reviewedSystemPackages[input.Family]...),
		Environment:    runtimeEnvironment(targetProfile, input.Target, input.Platform),
		RunnerProfile:  runnerUnsupported,
	}
	if targetProfile == targetROCm {
		policy.SystemPackages = append(policy.SystemPackages, "pciutils")
		policy.RuntimeSymlinks = append([]RuntimeSymlink(nil), rocmRuntimeSymlinks...)
	}
	if input.Family == familyVLLM && targetProfile == targetCUDA12 && input.Platform.OS == platformLinux && input.Platform.Architecture == architectureAMD64 {
		policy.Environment = append(policy.Environment, vllmNativeSampler)
	}
	sort.Strings(policy.SystemPackages)
	sort.Strings(policy.Environment)

	overlay, found := reviewedPolicyOverlayFor(reviewedPolicyKey{
		Version:  input.ImportedVersion,
		Family:   input.Family,
		Selector: input.Selector,
		Platform: input.Platform,
	})
	if found {
		if input.Target != overlay.Target {
			return entryPolicy{}, fmt.Errorf(
				"reviewed policy target drift for %s/%s/%s on %s: got %q, want %q",
				input.ImportedVersion, input.Family, input.Selector, input.Platform.key(), input.Target, overlay.Target,
			)
		}
		if input.SourceRef != overlay.SourceRef {
			return entryPolicy{}, fmt.Errorf(
				"reviewed policy source reference drift for %s/%s/%s on %s: got %q, want %q",
				input.ImportedVersion, input.Family, input.Selector, input.Platform.key(), input.SourceRef, overlay.SourceRef,
			)
		}
		if targetProfile != overlay.TargetProfile {
			return entryPolicy{}, fmt.Errorf(
				"reviewed policy target profile drift for %s/%s/%s on %s: got %q, want %q",
				input.ImportedVersion, input.Family, input.Selector, input.Platform.key(), targetProfile, overlay.TargetProfile,
			)
		}
		policy.Status = overlay.Status
		policy.RuntimeBaseRef = overlay.RuntimeBaseRef
		policy.RunnerRuntimeBaseRef = overlay.RunnerRuntimeBaseRef
		policy.InstallName = overlay.InstallName
		policy.RunnerProfile = overlay.RunnerProfile
		policy.Fallbacks = append([]fallbackTarget(nil), overlay.Fallbacks...)
	}

	// Kokoro ignores baked model paths and downloads an unpinned model at
	// runtime. Keep every tuple unavailable until the backend can consume the
	// immutable model materialized by AIKit.
	if input.Family == familyKokoro {
		policy.Status = statusQuarantined
	}

	// LocalAI v4.8.2's CUDA 12 SGLang bundle mixes PyTorch CUDA 13.0 with
	// TorchAudio CUDA 12.8 and exits before its gRPC service becomes ready.
	if input.Family == familySGLang && policy.TargetProfile == targetCUDA12 && input.Platform.OS == platformLinux && input.Platform.Architecture == architectureAMD64 {
		policy.Status = statusQuarantined
	}

	return policy, nil
}

func reviewedPolicyOverlayFor(key reviewedPolicyKey) (reviewedPolicyOverlay, bool) {
	for _, overlay := range reviewedPolicyOverlays {
		if overlay.Key == key {
			return overlay, true
		}
	}

	return reviewedPolicyOverlay{}, false
}

func validateReviewedPolicyOverlays(overlays []reviewedPolicyOverlay) error {
	if len(overlays) == 0 {
		return fmt.Errorf("reviewed policy overlays are empty")
	}

	seen := make(map[reviewedPolicyKey]struct{}, len(overlays))
	for _, overlay := range overlays {
		if overlay.Key.Version == "" || overlay.Key.Family == "" || overlay.Key.Selector == "" || overlay.Key.Platform.OS == "" ||
			overlay.Key.Platform.Architecture == "" || overlay.Target == "" || overlay.SourceRef == "" || overlay.TargetProfile == "" ||
			overlay.Status == "" || overlay.RuntimeBaseRef == "" || overlay.RunnerProfile == "" {
			return fmt.Errorf("reviewed policy overlay is incomplete: %#v", overlay)
		}
		if overlay.Status != statusSupported && overlay.Status != statusExperimental {
			return fmt.Errorf("reviewed policy overlay %s/%s/%s on %s has unsupported status %q", overlay.Key.Version, overlay.Key.Family, overlay.Key.Selector, overlay.Key.Platform.key(), overlay.Status)
		}
		if normalizePlatform(overlay.Key.Platform) != overlay.Key.Platform {
			return fmt.Errorf("reviewed policy overlay %s/%s/%s has noncanonical platform %q", overlay.Key.Version, overlay.Key.Family, overlay.Key.Selector, overlay.Key.Platform.key())
		}
		if overlay.RunnerProfile != runnerUnsupported && overlay.RunnerProfile != runnerLlamaCpp && overlay.RunnerProfile != familyVLLMCpp && overlay.RunnerProfile != runnerHFConfig {
			return fmt.Errorf(
				"reviewed policy overlay %s/%s/%s on %s has unsupported runner profile %q",
				overlay.Key.Version, overlay.Key.Family, overlay.Key.Selector, overlay.Key.Platform.key(), overlay.RunnerProfile,
			)
		}
		inferredTargetProfile, err := inferTargetProfile(overlay.Key.Selector, overlay.Target, overlay.SourceRef)
		if err != nil {
			return errors.Wrapf(
				err,
				"reviewed policy overlay %s/%s/%s on %s cannot infer target profile",
				overlay.Key.Version, overlay.Key.Family, overlay.Key.Selector, overlay.Key.Platform.key(),
			)
		}
		if inferredTargetProfile != overlay.TargetProfile {
			return fmt.Errorf(
				"reviewed policy overlay %s/%s/%s on %s has target profile %q, inferred %q",
				overlay.Key.Version, overlay.Key.Family, overlay.Key.Selector, overlay.Key.Platform.key(), overlay.TargetProfile, inferredTargetProfile,
			)
		}
		for _, fallback := range overlay.Fallbacks {
			if fallback.Family == "" || fallback.Selector == "" {
				return fmt.Errorf("reviewed policy overlay %s/%s/%s on %s has incomplete fallback: %#v", overlay.Key.Version, overlay.Key.Family, overlay.Key.Selector, overlay.Key.Platform.key(), fallback)
			}
		}
		if _, exists := seen[overlay.Key]; exists {
			return fmt.Errorf("reviewed policy overlay %s/%s/%s on %s is duplicated", overlay.Key.Version, overlay.Key.Family, overlay.Key.Selector, overlay.Key.Platform.key())
		}
		seen[overlay.Key] = struct{}{}
	}

	return nil
}

func runtimeBaseReference(targetProfile string, platform Platform) string {
	switch targetProfile {
	case targetROCm:
		return rocmRuntimeBase
	case targetMetal:
		return vulkanRuntimeBase
	case targetVulkan:
		if platform.Architecture == architectureARM64 {
			return vulkanRuntimeBase
		}
		return ubuntuRuntimeBase
	case targetL4TCUDA12:
		return l4tRuntimeBase
	case targetL4TCUDA13:
		return ubuntuRuntimeBase
	default:
		return ubuntuRuntimeBase
	}
}

func runtimeEnvironment(targetProfile, target string, platform Platform) []string {
	switch targetProfile {
	case targetCUDA12:
		if strings.HasPrefix(target, "vulkan-") {
			return cudaEnvironmentWithCapabilities(minimumCUDA12, cudaAllCapabilities)
		}
		return cudaEnvironment(minimumCUDA12)
	case targetCUDA13:
		if strings.HasPrefix(target, "vulkan-") {
			return cudaEnvironmentWithCapabilities(minimumCUDA13, cudaAllCapabilities)
		}
		return cudaEnvironment(minimumCUDA13)
	case targetL4TCUDA12:
		return l4tEnvironment(minimumCUDA12)
	case targetL4TCUDA13:
		return l4tEnvironment(minimumCUDA13)
	case targetROCm:
		return []string{
			"LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:/opt/rocm/llvm/lib",
			"LOCALAI_FORCE_META_BACKEND_CAPABILITY=amd",
			"PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/rocm/bin",
		}
	case targetMetal:
		return []string{vulkanEnvironment}
	case targetVulkan:
		if platform.Architecture == architectureARM64 {
			return []string{vulkanEnvironment}
		}
		return []string{}
	default:
		return []string{}
	}
}

func cudaEnvironment(minimumVersion string) []string {
	return cudaEnvironmentWithCapabilities(minimumVersion, cudaCapabilities)
}

func cudaEnvironmentWithCapabilities(minimumVersion, capabilities string) []string {
	return []string{
		cudaBuildType,
		capabilities,
		"NVIDIA_REQUIRE_CUDA=cuda>=" + minimumVersion,
		cudaVisibleDevices,
	}
}

func l4tEnvironment(minimumVersion string) []string {
	return []string{
		cudaBuildType,
		cudaHome,
		cudaLibraryPath,
		cudaAllCapabilities,
		"NVIDIA_REQUIRE_CUDA=cuda>=" + minimumVersion,
		cudaVisibleDevices,
		cudaPath,
	}
}

func inferTargetProfile(selector, target, sourceRef string) (string, error) {
	switch {
	case strings.HasPrefix(target, "cuda13-nvidia-l4t-"), selector == selectorL4TCUDA13,
		strings.Contains(sourceRef, "-nvidia-l4t-cuda-13-"):
		return targetL4TCUDA13, nil
	case strings.HasPrefix(target, "nvidia-l4t-"), selector == selectorNVIDIAL4T, selector == selectorL4TCUDA12:
		return targetL4TCUDA12, nil
	case strings.HasPrefix(target, "cuda13-"), selector == selectorNVIDIACUDA13:
		return targetCUDA13, nil
	case strings.HasPrefix(target, "cuda12-"), selector == selectorNVIDIA, selector == selectorNVIDIACUDA12:
		return targetCUDA12, nil
	case strings.HasPrefix(target, "rocm-"), selector == selectorAMD:
		return targetROCm, nil
	case strings.HasPrefix(target, "metal-"), strings.HasPrefix(selector, "metal"):
		return targetMetal, nil
	case strings.HasPrefix(target, "vulkan-"), selector == "vulkan":
		return targetVulkan, nil
	case strings.HasPrefix(target, "intel-"), selector == targetIntel:
		return targetIntel, nil
	case strings.HasPrefix(target, "cpu-"), selector == selectorDefault, selector == runtimeCPU:
		return runtimeCPU, nil
	default:
		return "", fmt.Errorf("cannot infer target profile for selector %q target %q", selector, target)
	}
}

func inferRuntime(targetProfile string, platform Platform) string {
	switch targetProfile {
	case targetCUDA12, targetCUDA13, targetL4TCUDA12, targetL4TCUDA13:
		return runtimeCUDA
	case targetROCm:
		return runtimeROCm
	case targetMetal:
		return runtimeApple
	case targetVulkan:
		if platform.Architecture == architectureARM64 {
			return runtimeApple
		}
		return runtimeCPU
	default:
		return runtimeCPU
	}
}
