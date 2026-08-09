package backendcatalog

import (
	"fmt"
	"net/url"
	"path"
	"regexp"
	"strings"

	"github.com/pkg/errors"
)

const (
	maximumNameLength         = 128
	maximumReferenceLength    = 512
	maximumEnvironmentLength  = 2048
	platformArchitectureAMD64 = "amd64"
	platformArchitectureARM64 = "arm64"
	platformOSLinux           = "linux"
)

var (
	safeNamePattern        = regexp.MustCompile(`^[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?$`)
	safeTokenPattern       = regexp.MustCompile(`^[a-z0-9](?:[a-z0-9._+-]*[a-z0-9])?$`)
	digestPattern          = regexp.MustCompile(`^sha256:[0-9a-f]{64}$`)
	l4tSelectorPattern     = regexp.MustCompile(`^nvidia-l4t-[a-z0-9]+(?:[._-][a-z0-9]+)*$`)
	environmentNamePattern = regexp.MustCompile(`^[A-Z_][A-Z0-9_]*$`)
	packageNamePattern     = regexp.MustCompile(`^[a-z0-9][a-z0-9+.-]*$`)
	repositoryPartPattern  = regexp.MustCompile(`^[a-z0-9]+(?:[._-][a-z0-9]+)*$`)
	registryPattern        = regexp.MustCompile(`^[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?(?::[0-9]{1,5})?$`)
)

func validateCatalog(catalog *Catalog) error {
	if catalog.SchemaVersion != schemaVersionV2 {
		return errors.Wrapf(ErrInvalidCatalog, "schemaVersion must be %q", schemaVersionV2)
	}
	if err := validateSource(catalog.Source); err != nil {
		return err
	}
	if err := validateDefaults(catalog.Defaults); err != nil {
		return err
	}
	if catalog.Entries == nil {
		return errors.Wrap(ErrInvalidCatalog, "entries must be an array")
	}

	seen := make(map[tupleKey]int, len(catalog.Entries))
	for i, entry := range catalog.Entries {
		if err := validateEntry(entry); err != nil {
			return errors.Wrapf(err, "entry %d", i)
		}

		key := keyFor(entry.Family, entry.Selector, entry.Platform)
		if previous, ok := seen[key]; ok {
			return errors.Wrapf(
				ErrInvalidCatalog,
				"entry %d duplicates entry %d tuple family %q selector %q platform %q",
				i,
				previous,
				entry.Family,
				entry.Selector,
				formatPlatform(entry.Platform),
			)
		}
		seen[key] = i
	}

	return nil
}

func validateEntry(entry Entry) error {
	if err := validateSafeName("family", entry.Family); err != nil {
		return err
	}
	if err := validateSelector(entry.Selector); err != nil {
		return err
	}
	if err := validatePlatform(entry.Platform); err != nil {
		return err
	}
	if !validRuntime(entry.Runtime) {
		return errors.Wrapf(ErrInvalidCatalog, "runtime %q is not supported", entry.Runtime)
	}
	if !validTargetProfile(entry.TargetProfile) {
		return errors.Wrapf(ErrInvalidCatalog, "targetProfile %q is not supported", entry.TargetProfile)
	}
	if !runtimeMatchesTarget(entry.Runtime, entry.TargetProfile) {
		return errors.Wrapf(ErrInvalidCatalog, "runtime %q does not match targetProfile %q", entry.Runtime, entry.TargetProfile)
	}
	if !runtimeSupportsPlatform(entry.Runtime, entry.TargetProfile, entry.Platform) {
		return errors.Wrapf(
			ErrInvalidCatalog,
			"runtime %q targetProfile %q is not supported on platform %q",
			entry.Runtime,
			entry.TargetProfile,
			formatPlatform(entry.Platform),
		)
	}
	if !validStatus(entry.Status) {
		return errors.Wrapf(ErrInvalidCatalog, "status %q is not supported", entry.Status)
	}
	if entry.Status != StatusExperimental && !selectorMatchesTarget(entry.Selector, entry.TargetProfile) {
		return errors.Wrapf(ErrInvalidCatalog, "selector %q does not match targetProfile %q", entry.Selector, entry.TargetProfile)
	}
	if entry.Channel != ChannelStable {
		return errors.Wrapf(ErrInvalidCatalog, "channel %q is not supported", entry.Channel)
	}
	if err := validateAuditValue("version", entry.Version); err != nil {
		return err
	}
	if err := validateAuditValue("sourceRef", entry.SourceRef); err != nil {
		return err
	}
	if err := validateArtifact("runtimeBase", entry.RuntimeBase.Ref); err != nil {
		return err
	}
	if entry.RunnerRuntimeBase != nil {
		if err := validateArtifact("runnerRuntimeBase", entry.RunnerRuntimeBase.Ref); err != nil {
			return err
		}
	}
	if err := validateArtifact("core", entry.Core.Ref); err != nil {
		return err
	}
	if err := validateBackendArtifact("backend", entry.Backend); err != nil {
		return err
	}
	installNames := map[string]string{entry.Backend.InstallName: "backend"}
	for i, fallback := range entry.Fallbacks {
		field := fmt.Sprintf("fallbacks[%d]", i)
		if err := validateBackendArtifact(field, fallback); err != nil {
			return err
		}
		if previous, ok := installNames[fallback.InstallName]; ok {
			return errors.Wrapf(ErrInvalidCatalog, "%s.installName %q collides with %s", field, fallback.InstallName, previous)
		}
		installNames[fallback.InstallName] = field
	}
	if err := validateSystemPackages(entry.SystemPackages); err != nil {
		return err
	}
	if err := validateRuntimeSymlinks(entry.RuntimeSymlinks); err != nil {
		return err
	}
	if err := validateEnvironment(entry.Environment); err != nil {
		return err
	}
	if !validRunnerProfile(entry.RunnerProfile) {
		return errors.Wrapf(ErrInvalidCatalog, "runnerProfile %q is not supported", entry.RunnerProfile)
	}
	if !runnerProfileMatchesFamily(entry.RunnerProfile, entry.Family) {
		return errors.Wrapf(ErrInvalidCatalog, "runnerProfile %q does not match family %q", entry.RunnerProfile, entry.Family)
	}
	if err := validateWorkloads(entry.Workloads); err != nil {
		return err
	}

	return nil
}

func validateRequest(request Request) error {
	if err := validateSafeName("family", request.Family); err != nil {
		return errors.Wrap(ErrInvalidRequest, err.Error())
	}
	if err := validateSelector(request.Selector); err != nil {
		return errors.Wrap(ErrInvalidRequest, err.Error())
	}
	if !validRuntime(request.Runtime) {
		return errors.Wrapf(ErrInvalidRequest, "runtime %q is not supported", request.Runtime)
	}
	if err := validatePlatform(request.Platform); err != nil {
		return errors.Wrap(ErrInvalidRequest, err.Error())
	}

	return nil
}

func validateDefaults(defaults Defaults) error {
	if err := validateSafeName("defaults.family", defaults.Family); err != nil {
		return err
	}
	if defaults.Selectors == nil {
		return errors.Wrap(ErrInvalidCatalog, "defaults.selectors must be an array")
	}
	seen := make(map[defaultSelectorKey]int, len(defaults.Selectors))
	for i, selection := range defaults.Selectors {
		if !validRuntime(selection.Runtime) {
			return errors.Wrapf(ErrInvalidCatalog, "defaults.selectors[%d].runtime %q is not supported", i, selection.Runtime)
		}
		if err := validateSelector(selection.Selector); err != nil {
			return errors.Wrapf(err, "defaults.selectors[%d]", i)
		}

		platform := Platform{}
		if selection.Platform != nil {
			if err := validatePlatform(*selection.Platform); err != nil {
				return errors.Wrapf(err, "defaults.selectors[%d].platform", i)
			}
			platform = *selection.Platform
		}
		key := defaultSelectorKeyFor(selection.Runtime, platform)
		if previous, ok := seen[key]; ok {
			return errors.Wrapf(ErrInvalidCatalog, "defaults.selectors[%d] duplicates defaults.selectors[%d]", i, previous)
		}
		seen[key] = i
	}
	for _, runtime := range []Runtime{RuntimeCPU, RuntimeCUDA, RuntimeROCm, RuntimeAppleSilicon} {
		if _, ok := seen[defaultSelectorKeyFor(runtime, Platform{})]; !ok {
			return errors.Wrapf(ErrInvalidCatalog, "defaults.selectors is missing runtime %q", runtime)
		}
	}

	return nil
}

func validateSource(source Source) error {
	repository, err := url.Parse(source.Repository)
	if err != nil || repository.Scheme != "https" || repository.Host == "" || repository.User != nil {
		return errors.Wrap(ErrInvalidCatalog, "source.repository must be an absolute HTTPS URL without user information")
	}
	if err := validateAuditValue("source.revision", source.Revision); err != nil {
		return err
	}
	if source.Path != "" {
		if path.IsAbs(source.Path) || path.Clean(source.Path) != source.Path || source.Path == "." || strings.HasPrefix(source.Path, "../") {
			return errors.Wrap(ErrInvalidCatalog, "source.path must be a clean relative path")
		}
		if strings.ContainsAny(source.Path, "\x00\r\n\\") {
			return errors.Wrap(ErrInvalidCatalog, "source.path must use safe slash-separated path components")
		}
	}
	if source.SHA256 != "" && !digestPattern.MatchString(source.SHA256) {
		return errors.Wrap(ErrInvalidCatalog, "source.sha256 must use lowercase sha256:<64 hex>")
	}

	return nil
}

func validateSelector(selector Selector) error {
	switch selector {
	case SelectorDefault, SelectorCPU, SelectorNVIDIA, SelectorNVIDIACUDA12, SelectorNVIDIACUDA13, SelectorAMD, SelectorIntel, SelectorVulkan,
		SelectorMetal, SelectorMetalDarwin, SelectorNVIDIAL4T, SelectorL4TCUDA12, SelectorL4TCUDA13:
		return nil
	default:
		if l4tSelectorPattern.MatchString(string(selector)) {
			return nil
		}

		return errors.Wrapf(ErrInvalidCatalog, "selector %q is not a recognized LocalAI selector", selector)
	}
}

func validatePlatform(platform Platform) error {
	if err := validateSafeToken("platform.os", platform.OS); err != nil {
		return err
	}
	if err := validateSafeToken("platform.architecture", platform.Architecture); err != nil {
		return err
	}
	if platform.Variant != "" {
		if err := validateSafeToken("platform.variant", platform.Variant); err != nil {
			return err
		}
	}
	if normalized := normalizeRequestPlatform(platform); normalized != platform {
		return errors.Wrapf(ErrInvalidCatalog, "platform %q must use canonical OCI names", formatPlatform(platform))
	}

	return nil
}

func validateArtifact(field, ref string) error {
	if len(ref) == 0 || len(ref) > maximumReferenceLength || strings.Count(ref, "@") != 1 {
		return errors.Wrapf(ErrInvalidCatalog, "%s.ref must be a digest-qualified OCI reference", field)
	}

	name, digest, _ := strings.Cut(ref, "@")
	if !digestPattern.MatchString(digest) {
		return errors.Wrapf(ErrInvalidCatalog, "%s.ref must use lowercase sha256:<64 hex>", field)
	}
	parts := strings.Split(name, "/")
	if len(parts) < 2 || !registryPattern.MatchString(parts[0]) {
		return errors.Wrapf(ErrInvalidCatalog, "%s.ref must contain an explicit valid registry and repository", field)
	}
	for _, part := range parts[1:] {
		if !repositoryPartPattern.MatchString(part) {
			return errors.Wrapf(ErrInvalidCatalog, "%s.ref repository contains an invalid or tagged component %q", field, part)
		}
	}

	return nil
}

func validateBackendArtifact(field string, artifact BackendArtifact) error {
	if err := validateArtifact(field, artifact.Ref); err != nil {
		return err
	}
	if err := validateSafeName(field+".installName", artifact.InstallName); err != nil {
		return err
	}

	return nil
}

func validateSafeName(field, value string) error {
	if len(value) == 0 || len(value) > maximumNameLength || !safeNamePattern.MatchString(value) {
		return errors.Wrapf(ErrInvalidCatalog, "%s %q is not a safe lowercase name", field, value)
	}

	return nil
}

func validateSafeToken(field, value string) error {
	if len(value) == 0 || len(value) > maximumNameLength || !safeTokenPattern.MatchString(value) {
		return errors.Wrapf(ErrInvalidCatalog, "%s %q is not a safe lowercase token", field, value)
	}

	return nil
}

func validateAuditValue(field, value string) error {
	if value == "" || len(value) > maximumReferenceLength || strings.ContainsAny(value, "\x00\r\n") {
		return errors.Wrapf(ErrInvalidCatalog, "%s must be a non-empty single-line value", field)
	}

	return nil
}

func validateWorkloads(workloads []string) error {
	seen := make(map[string]struct{}, len(workloads))
	for i, workload := range workloads {
		if err := validateSafeToken(fmt.Sprintf("workloads[%d]", i), workload); err != nil {
			return err
		}
		if _, ok := seen[workload]; ok {
			return errors.Wrapf(ErrInvalidCatalog, "workload %q is duplicated", workload)
		}
		seen[workload] = struct{}{}
	}

	return nil
}

func validateSystemPackages(packages []string) error {
	seen := make(map[string]struct{}, len(packages))
	for i, name := range packages {
		if len(name) == 0 || len(name) > maximumNameLength || !packageNamePattern.MatchString(name) {
			return errors.Wrapf(ErrInvalidCatalog, "systemPackages[%d] %q is not a safe package name", i, name)
		}
		if _, ok := seen[name]; ok {
			return errors.Wrapf(ErrInvalidCatalog, "system package %q is duplicated", name)
		}
		seen[name] = struct{}{}
	}

	return nil
}

func validateEnvironment(environment []string) error {
	seen := make(map[string]struct{}, len(environment))
	for i, variable := range environment {
		if len(variable) == 0 || len(variable) > maximumEnvironmentLength || strings.ContainsAny(variable, "\x00\r\n") {
			return errors.Wrapf(ErrInvalidCatalog, "environment[%d] must be a non-empty single-line KEY=value", i)
		}
		name, _, found := strings.Cut(variable, "=")
		if !found || !environmentNamePattern.MatchString(name) {
			return errors.Wrapf(ErrInvalidCatalog, "environment[%d] %q has an invalid variable name", i, variable)
		}
		if _, ok := seen[name]; ok {
			return errors.Wrapf(ErrInvalidCatalog, "environment variable %q is duplicated", name)
		}
		seen[name] = struct{}{}
	}

	return nil
}

func validateRuntimeSymlinks(symlinks []RuntimeSymlink) error {
	seen := make(map[string]struct{}, len(symlinks))
	for i, symlink := range symlinks {
		if err := validateSafeName(fmt.Sprintf("runtimeSymlinks[%d].target", i), symlink.Target); err != nil {
			return err
		}
		if len(symlink.Path) == 0 || len(symlink.Path) > maximumReferenceLength || !path.IsAbs(symlink.Path) || path.Clean(symlink.Path) != symlink.Path ||
			symlink.Path == "/" || strings.ContainsAny(symlink.Path, "\x00\r\n\\") {
			return errors.Wrapf(ErrInvalidCatalog, "runtimeSymlinks[%d].path %q must be a clean absolute path", i, symlink.Path)
		}
		if _, ok := seen[symlink.Path]; ok {
			return errors.Wrapf(ErrInvalidCatalog, "runtime symlink path %q is duplicated", symlink.Path)
		}
		seen[symlink.Path] = struct{}{}
	}

	return nil
}

func validRuntime(runtime Runtime) bool {
	switch runtime {
	case RuntimeCPU, RuntimeCUDA, RuntimeROCm, RuntimeAppleSilicon:
		return true
	default:
		return false
	}
}

func validTargetProfile(profile TargetProfile) bool {
	switch profile {
	case TargetProfileCPU, TargetProfileCUDA12, TargetProfileCUDA13, TargetProfileROCm, TargetProfileIntel, TargetProfileVulkan,
		TargetProfileMetal, TargetProfileL4TCUDA12, TargetProfileL4TCUDA13:
		return true
	default:
		return false
	}
}

func runtimeMatchesTarget(runtime Runtime, target TargetProfile) bool {
	switch runtime {
	case RuntimeCPU:
		return target == TargetProfileCPU || target == TargetProfileIntel || target == TargetProfileVulkan
	case RuntimeCUDA:
		return target == TargetProfileCUDA12 || target == TargetProfileCUDA13 ||
			target == TargetProfileL4TCUDA12 || target == TargetProfileL4TCUDA13
	case RuntimeROCm:
		return target == TargetProfileROCm
	case RuntimeAppleSilicon:
		return target == TargetProfileVulkan || target == TargetProfileMetal
	default:
		return false
	}
}

func runtimeSupportsPlatform(runtime Runtime, target TargetProfile, platform Platform) bool {
	if platform.OS != platformOSLinux {
		return false
	}
	if platform.Architecture != platformArchitectureAMD64 && platform.Architecture != platformArchitectureARM64 {
		return false
	}
	if target == TargetProfileVulkan {
		return platform.Architecture == platformArchitectureAMD64 && runtime == RuntimeCPU ||
			platform.Architecture == platformArchitectureARM64 && runtime == RuntimeAppleSilicon
	}
	if runtime == RuntimeAppleSilicon && platform.Architecture != platformArchitectureARM64 {
		return false
	}
	if runtime == RuntimeROCm && platform.Architecture != platformArchitectureAMD64 {
		return false
	}
	if (target == TargetProfileL4TCUDA12 || target == TargetProfileL4TCUDA13) && platform.Architecture != platformArchitectureARM64 {
		return false
	}

	return true
}

func selectorMatchesTarget(selector Selector, target TargetProfile) bool {
	switch selector {
	case SelectorDefault, SelectorCPU:
		return target == TargetProfileCPU
	case SelectorIntel:
		return target == TargetProfileIntel
	case SelectorNVIDIA:
		return target == TargetProfileCUDA12 || target == TargetProfileCUDA13
	case SelectorNVIDIACUDA12:
		return target == TargetProfileCUDA12
	case SelectorNVIDIACUDA13:
		return target == TargetProfileCUDA13
	case SelectorAMD:
		return target == TargetProfileROCm
	case SelectorVulkan:
		return target == TargetProfileVulkan
	case SelectorMetal, SelectorMetalDarwin:
		return target == TargetProfileMetal
	case SelectorL4TCUDA12:
		return target == TargetProfileL4TCUDA12
	case SelectorL4TCUDA13:
		return target == TargetProfileL4TCUDA13
	case SelectorNVIDIAL4T:
		return target == TargetProfileL4TCUDA12 || target == TargetProfileL4TCUDA13
	default:
		if !l4tSelectorPattern.MatchString(string(selector)) {
			return false
		}
		if strings.HasPrefix(string(selector), string(SelectorL4TCUDA12)+"-") {
			return target == TargetProfileL4TCUDA12
		}
		if strings.HasPrefix(string(selector), string(SelectorL4TCUDA13)+"-") {
			return target == TargetProfileL4TCUDA13
		}

		return target == TargetProfileL4TCUDA12 || target == TargetProfileL4TCUDA13
	}
}

func validStatus(status Status) bool {
	switch status {
	case StatusSupported, StatusExperimental, StatusQuarantined, StatusDeprecated:
		return true
	default:
		return false
	}
}

func validRunnerProfile(profile RunnerProfile) bool {
	switch profile {
	case RunnerProfileUnsupported, RunnerProfileLlamaCpp, RunnerProfileVLLMCpp, RunnerProfileHFConfig:
		return true
	default:
		return false
	}
}

func runnerProfileMatchesFamily(profile RunnerProfile, family string) bool {
	switch profile {
	case RunnerProfileLlamaCpp:
		return family == string(RunnerProfileLlamaCpp)
	case RunnerProfileVLLMCpp:
		return family == string(RunnerProfileVLLMCpp)
	default:
		return true
	}
}

func formatPlatform(platform Platform) string {
	formatted := platform.OS + "/" + platform.Architecture
	if platform.Variant != "" {
		formatted += "/" + platform.Variant
	}

	return formatted
}

func invalidCatalog(action string, err error) error {
	return errors.Wrapf(ErrInvalidCatalog, "%s: %v", action, err)
}
