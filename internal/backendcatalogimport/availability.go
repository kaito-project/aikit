package backendcatalogimport

import "fmt"

type unavailableSourcePolicy struct {
	Version    string
	Family     string
	Selector   string
	SourceRef  string
	ErrorClass ResolutionErrorClass
}

var reviewedUnavailableSources = []unavailableSourcePolicy{
	{
		Version:    LocalAIVersion,
		Family:     "kokoros",
		Selector:   selectorDefault,
		SourceRef:  "quay.io/go-skynet/local-ai-backends:v4.8.2-cpu-kokoros",
		ErrorClass: resolutionErrorNotFound,
	},
	{
		Version:    LocalAIVersion,
		Family:     "turboquant",
		Selector:   selectorAMD,
		SourceRef:  "quay.io/go-skynet/local-ai-backends:v4.8.2-gpu-rocm-hipblas-turboquant",
		ErrorClass: resolutionErrorNotFound,
	},
	{
		Version:    LocalAIVersion,
		Family:     familyVLLM,
		Selector:   "intel",
		SourceRef:  "quay.io/go-skynet/local-ai-backends:v4.8.2-gpu-intel-vllm",
		ErrorClass: resolutionErrorNotFound,
	},
}

func validateUnavailableSourcePolicies(policies []unavailableSourcePolicy) error {
	seen := make(map[string]struct{}, len(policies))
	for _, policy := range policies {
		if policy.Version == "" || policy.Family == "" || policy.Selector == "" || policy.SourceRef == "" {
			return fmt.Errorf("reviewed unavailable source policy is incomplete: %#v", policy)
		}
		if policy.ErrorClass != resolutionErrorNotFound {
			return fmt.Errorf("reviewed unavailable source %s/%s/%s has unsupported error class %q", policy.Version, policy.Family, policy.Selector, policy.ErrorClass)
		}
		if hasSupportedOverlay(policy.Family, policy.Selector) {
			return fmt.Errorf("reviewed unavailable source %s/%s/%s overlaps a supported policy tuple", policy.Version, policy.Family, policy.Selector)
		}
		key := policy.Version + "\x00" + policy.Family + "\x00" + policy.Selector + "\x00" + policy.SourceRef
		if _, exists := seen[key]; exists {
			return fmt.Errorf("reviewed unavailable source policy %s/%s/%s is duplicated", policy.Version, policy.Family, policy.Selector)
		}
		seen[key] = struct{}{}
	}

	return nil
}

func reviewedUnavailableSource(version, family, selector, sourceRef string) (unavailableSourcePolicy, bool) {
	for _, policy := range reviewedUnavailableSources {
		if policy.Version == version && policy.Family == family && policy.Selector == selector && policy.SourceRef == sourceRef {
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
