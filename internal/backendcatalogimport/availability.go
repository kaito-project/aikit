package backendcatalogimport

import "fmt"

type unavailableSourcePolicy struct {
	Version    string
	Family     string
	Selector   string
	Target     string
	SourceRef  string
	ErrorClass ResolutionErrorClass
}

var reviewedUnavailableSources = []unavailableSourcePolicy{
	{
		Version:    LocalAIVersion,
		Family:     familyTurboQuant,
		Selector:   selectorAMD,
		Target:     "rocm-turboquant",
		SourceRef:  "quay.io/go-skynet/local-ai-backends:" + LocalAIVersion + "-gpu-rocm-hipblas-turboquant",
		ErrorClass: resolutionErrorNotFound,
	},
	{
		Version:    LocalAIVersion,
		Family:     "bonsai",
		Selector:   selectorNVIDIA,
		Target:     "cuda12-bonsai",
		SourceRef:  "quay.io/go-skynet/local-ai-backends:" + LocalAIVersion + "-gpu-nvidia-cuda-12-bonsai",
		ErrorClass: resolutionErrorNotFound,
	},
	{
		Version:    LocalAIVersion,
		Family:     "bonsai",
		Selector:   selectorNVIDIACUDA12,
		Target:     "cuda12-bonsai",
		SourceRef:  "quay.io/go-skynet/local-ai-backends:" + LocalAIVersion + "-gpu-nvidia-cuda-12-bonsai",
		ErrorClass: resolutionErrorNotFound,
	},
	{
		Version:    LocalAIVersion,
		Family:     "fish-speech",
		Selector:   selectorNVIDIACUDA13,
		Target:     "cuda13-fish-speech",
		SourceRef:  "quay.io/go-skynet/local-ai-backends:" + LocalAIVersion + "-gpu-nvidia-cuda-13-fish-speech",
		ErrorClass: resolutionErrorNotFound,
	},
	{
		Version:    LocalAIVersion,
		Family:     "funasr",
		Selector:   selectorNVIDIACUDA13,
		Target:     "cuda13-funasr",
		SourceRef:  "quay.io/go-skynet/local-ai-backends:" + LocalAIVersion + "-gpu-nvidia-cuda-13-funasr",
		ErrorClass: resolutionErrorNotFound,
	},
	{
		Version:    LocalAIVersion,
		Family:     "kokoros",
		Selector:   selectorDefault,
		Target:     "cpu-kokoros",
		SourceRef:  "quay.io/go-skynet/local-ai-backends:" + LocalAIVersion + "-cpu-kokoros",
		ErrorClass: resolutionErrorNotFound,
	},
	{
		Version:    LocalAIVersion,
		Family:     "llama-cpp",
		Selector:   selectorNVIDIACUDA13,
		Target:     "cuda13-llama-cpp",
		SourceRef:  "quay.io/go-skynet/local-ai-backends:" + LocalAIVersion + "-gpu-nvidia-cuda-13-llama-cpp",
		ErrorClass: resolutionErrorNotFound,
	},
	{
		Version:    LocalAIVersion,
		Family:     "neutts",
		Selector:   selectorNVIDIA,
		Target:     "cuda12-neutts",
		SourceRef:  "quay.io/go-skynet/local-ai-backends:" + LocalAIVersion + "-gpu-nvidia-cuda-12-neutts",
		ErrorClass: resolutionErrorNotFound,
	},
	{
		Version:    LocalAIVersion,
		Family:     "neutts",
		Selector:   selectorNVIDIACUDA12,
		Target:     "cuda12-neutts",
		SourceRef:  "quay.io/go-skynet/local-ai-backends:" + LocalAIVersion + "-gpu-nvidia-cuda-12-neutts",
		ErrorClass: resolutionErrorNotFound,
	},
	{
		Version:    LocalAIVersion,
		Family:     "pocket-tts",
		Selector:   selectorNVIDIACUDA13,
		Target:     "cuda13-pocket-tts",
		SourceRef:  "quay.io/go-skynet/local-ai-backends:" + LocalAIVersion + "-gpu-nvidia-cuda-13-pocket-tts",
		ErrorClass: resolutionErrorNotFound,
	},
	{
		Version:    LocalAIVersion,
		Family:     "qwen-tts",
		Selector:   selectorNVIDIACUDA13,
		Target:     "cuda13-qwen-tts",
		SourceRef:  "quay.io/go-skynet/local-ai-backends:" + LocalAIVersion + "-gpu-nvidia-cuda-13-qwen-tts",
		ErrorClass: resolutionErrorNotFound,
	},
	{
		Version:    LocalAIVersion,
		Family:     familyTurboQuant,
		Selector:   selectorNVIDIA,
		Target:     "cuda12-turboquant",
		SourceRef:  "quay.io/go-skynet/local-ai-backends:" + LocalAIVersion + "-gpu-nvidia-cuda-12-turboquant",
		ErrorClass: resolutionErrorNotFound,
	},
	{
		Version:    LocalAIVersion,
		Family:     familyTurboQuant,
		Selector:   selectorNVIDIACUDA12,
		Target:     "cuda12-turboquant",
		SourceRef:  "quay.io/go-skynet/local-ai-backends:" + LocalAIVersion + "-gpu-nvidia-cuda-12-turboquant",
		ErrorClass: resolutionErrorNotFound,
	},
	{
		Version:    LocalAIVersion,
		Family:     "voxcpm",
		Selector:   selectorNVIDIACUDA13,
		Target:     "cuda13-voxcpm",
		SourceRef:  "quay.io/go-skynet/local-ai-backends:" + LocalAIVersion + "-gpu-nvidia-cuda-13-voxcpm",
		ErrorClass: resolutionErrorNotFound,
	},
}

func validateUnavailableSourcePolicies(policies []unavailableSourcePolicy) error {
	seen := make(map[string]struct{}, len(policies))
	for _, policy := range policies {
		if policy.Version == "" || policy.Family == "" || policy.Selector == "" || policy.Target == "" || policy.SourceRef == "" {
			return fmt.Errorf("reviewed unavailable source policy is incomplete: %#v", policy)
		}
		if policy.ErrorClass != resolutionErrorNotFound {
			return fmt.Errorf("reviewed unavailable source %s/%s/%s has unsupported error class %q", policy.Version, policy.Family, policy.Selector, policy.ErrorClass)
		}
		if hasReviewedOverlayMapping(policy.Version, policy.Family, policy.Selector, policy.Target, policy.SourceRef) {
			return fmt.Errorf("reviewed unavailable source %s/%s/%s overlaps a reviewed policy tuple", policy.Version, policy.Family, policy.Selector)
		}
		key := policy.Version + "\x00" + policy.Family + "\x00" + policy.Selector + "\x00" + policy.Target + "\x00" + policy.SourceRef
		if _, exists := seen[key]; exists {
			return fmt.Errorf("reviewed unavailable source policy %s/%s/%s is duplicated", policy.Version, policy.Family, policy.Selector)
		}
		seen[key] = struct{}{}
	}

	return nil
}

func reviewedUnavailableSource(version, family, selector, target, sourceRef string) (unavailableSourcePolicy, bool) {
	for _, policy := range reviewedUnavailableSources {
		if policy.Version == version && policy.Family == family && policy.Selector == selector && policy.Target == target && policy.SourceRef == sourceRef {
			return policy, true
		}
	}

	return unavailableSourcePolicy{}, false
}

func entryEligibleForAIKit(platform Platform, runtime, targetProfile string) bool {
	if platform.OS != platformLinux {
		return false
	}
	if platform.Architecture != architectureAMD64 && platform.Architecture != architectureARM64 {
		return false
	}
	if targetProfile == targetVulkan {
		if platform.Architecture == architectureAMD64 && runtime != runtimeCPU {
			return false
		}
		if platform.Architecture == architectureARM64 && runtime != runtimeApple {
			return false
		}
	}
	if targetProfile == targetMetal && runtime != runtimeApple {
		return false
	}
	if runtime == runtimeApple && platform.Architecture != architectureARM64 {
		return false
	}
	if runtime == runtimeROCm && platform.Architecture != architectureAMD64 {
		return false
	}
	if (targetProfile == targetL4TCUDA12 || targetProfile == targetL4TCUDA13) && platform.Architecture != architectureARM64 {
		return false
	}

	return true
}
