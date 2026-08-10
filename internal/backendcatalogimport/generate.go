package backendcatalogimport

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/pkg/errors"
)

// GenerateOptions configures one pinned catalog generation.
type GenerateOptions struct {
	Source          SourcePin
	Version         string
	CoreRefTemplate string
	Resolver        Resolver
}

type pendingEntry struct {
	Entry     Entry
	Fallbacks []fallbackTarget
}

// Generate parses, resolves, normalizes, and overlays the pinned LocalAI catalog.
func Generate(ctx context.Context, source []byte, options GenerateOptions) (Catalog, error) {
	if options.Resolver == nil {
		return Catalog{}, errors.New("OCI resolver is required")
	}
	if options.Version == "" {
		return Catalog{}, errors.New("LocalAI version is required")
	}
	if options.CoreRefTemplate == "" {
		return Catalog{}, errors.New("LocalAI core reference template is required")
	}
	if err := validateUnavailableSourcePolicies(reviewedUnavailableSources); err != nil {
		return Catalog{}, errors.Wrap(err, "validate reviewed unavailable source policy")
	}

	sourceEntries, err := parseSource(source, options.Source)
	if err != nil {
		return Catalog{}, err
	}
	concrete := make(map[string]sourceEntry, len(sourceEntries))
	for _, entry := range sourceEntries {
		if entry.URI != "" {
			concrete[entry.Name] = entry
		}
	}

	resolver := newCachedResolver(options.Resolver)
	pending := make([]pendingEntry, 0, len(sourceEntries))
	byTuple := make(map[string][]byte)
	for _, selectorEntry := range sourceEntries {
		if selectorEntry.URI != "" || len(selectorEntry.Capabilities) == 0 || strings.HasSuffix(selectorEntry.Name, "-development") {
			continue
		}
		family := selectorEntry.Name
		if selectorEntry.Alias != "" {
			family = selectorEntry.Alias
		}

		selectors := sortedMapKeys(selectorEntry.Capabilities)
		for _, selector := range selectors {
			targetName := selectorEntry.Capabilities[selector]
			if strings.HasSuffix(targetName, "-development") {
				return Catalog{}, fmt.Errorf("stable family %q selector %q targets development entry %q", family, selector, targetName)
			}
			target, ok := concrete[targetName]
			if !ok {
				return Catalog{}, fmt.Errorf("stable family %q selector %q targets missing concrete entry %q", family, selector, targetName)
			}
			entryVersion := artifactVersionFor(options.Version, family, selector)
			sourceRef, err := stableVersionReference(target.URI, entryVersion)
			if err != nil {
				return Catalog{}, errors.Wrapf(err, "normalize family %q selector %q", family, selector)
			}
			coreRefTemplate, err := coreReferenceTemplateForVersion(options.CoreRefTemplate, options.Version, entryVersion)
			if err != nil {
				return Catalog{}, errors.Wrapf(err, "select LocalAI core for family %q selector %q", family, selector)
			}
			unavailablePolicy, reviewedUnavailable := reviewedUnavailableSource(options.Version, family, selector, sourceRef)
			manifests, resolveErr := resolver.resolve(ctx, sourceRef)
			if reviewedUnavailable {
				if resolveErr == nil {
					return Catalog{}, fmt.Errorf("reviewed unavailable source %q for family %q selector %q resolved successfully; remove stale exclusion policy", sourceRef, family, selector)
				}
				actualClass, classified := resolutionErrorClass(resolveErr)
				if !classified || actualClass != unavailablePolicy.ErrorClass {
					return Catalog{}, errors.Wrapf(
						resolveErr,
						"reviewed unavailable source %q for family %q selector %q failed with unexpected class (want %q)",
						sourceRef,
						family,
						selector,
						unavailablePolicy.ErrorClass,
					)
				}

				continue
			}
			if resolveErr != nil {
				return Catalog{}, errors.Wrapf(resolveErr, "resolve family %q selector %q", family, selector)
			}

			for _, manifest := range manifests {
				if manifest.Platform.OS == "" || manifest.Platform.Architecture == "" {
					return Catalog{}, fmt.Errorf("backend reference %q manifest %s has no platform", sourceRef, manifest.Digest)
				}
				policy, err := policyFor(family, selector, targetName, sourceRef, manifest.Platform)
				if err != nil {
					return Catalog{}, errors.Wrapf(err, "apply policy for family %q selector %q", family, selector)
				}
				if !entryEligibleForAIKit(manifest.Platform, policy.Runtime, policy.TargetProfile) {
					continue
				}
				runtimeBaseRef, err := resolvePlatformReference(ctx, resolver, policy.RuntimeBaseRef, manifest.Platform, false)
				if err != nil {
					return Catalog{}, errors.Wrapf(err, "resolve runtime base for %s", manifest.Platform.key())
				}
				coreRef, err := resolveCore(ctx, resolver, coreRefTemplate, manifest.Platform)
				if err != nil {
					return Catalog{}, errors.Wrapf(err, "resolve LocalAI core for %s", manifest.Platform.key())
				}
				installName := policy.InstallName
				if installName == "" {
					installName = targetName
				}
				var runnerRuntimeBase *Artifact
				if policy.RunnerRuntimeBaseRef != "" {
					runnerRuntimeBaseRef, err := resolvePlatformReference(ctx, resolver, policy.RunnerRuntimeBaseRef, manifest.Platform, false)
					if err != nil {
						return Catalog{}, errors.Wrapf(err, "resolve runner runtime base for %s", manifest.Platform.key())
					}
					runnerRuntimeBase = &Artifact{Ref: runnerRuntimeBaseRef}
				}
				entry := Entry{
					Family:            family,
					Selector:          selector,
					Platform:          manifest.Platform,
					Runtime:           policy.Runtime,
					TargetProfile:     policy.TargetProfile,
					Status:            policy.Status,
					Channel:           "stable",
					RuntimeBase:       Artifact{Ref: runtimeBaseRef},
					RunnerRuntimeBase: runnerRuntimeBase,
					Core:              Artifact{Ref: coreRef},
					Backend:           BackendArtifact{Ref: immutableReference(sourceRef, manifest.Digest), InstallName: installName},
					Fallbacks:         []BackendArtifact{},
					Version:           entryVersion,
					SourceRef:         sourceRef,
					SystemPackages:    append([]string(nil), policy.SystemPackages...),
					RuntimeSymlinks:   append([]RuntimeSymlink(nil), policy.RuntimeSymlinks...),
					Environment:       append([]string(nil), policy.Environment...),
					RunnerProfile:     policy.RunnerProfile,
					Workloads:         normalizeWorkloads(selectorEntry.Tags),
				}
				encoded, err := json.Marshal(entry)
				if err != nil {
					return Catalog{}, errors.Wrap(err, "encode normalized entry for duplicate comparison")
				}
				tupleKey := entryTupleKey(entry.Family, entry.Selector, entry.Platform)
				if previous, exists := byTuple[tupleKey]; exists {
					if !bytes.Equal(previous, encoded) {
						return Catalog{}, fmt.Errorf("normalization produced conflicting tuple %q", tupleKey)
					}
					continue
				}
				byTuple[tupleKey] = encoded
				pending = append(pending, pendingEntry{Entry: entry, Fallbacks: policy.Fallbacks})
			}
		}
	}
	if len(pending) == 0 {
		return Catalog{}, errors.New("LocalAI backend catalog contains no stable selectable entries")
	}

	resolvedByTuple := make(map[string]BackendArtifact, len(pending))
	for _, candidate := range pending {
		resolvedByTuple[entryTupleKey(candidate.Entry.Family, candidate.Entry.Selector, candidate.Entry.Platform)] = candidate.Entry.Backend
	}
	entries := make([]Entry, 0, len(pending))
	for _, candidate := range pending {
		for _, fallback := range candidate.Fallbacks {
			key := entryTupleKey(fallback.Family, fallback.Selector, candidate.Entry.Platform)
			artifact, ok := resolvedByTuple[key]
			if !ok {
				return Catalog{}, fmt.Errorf("tuple %q fallback %q is not resolvable on the same platform", entryTupleKey(candidate.Entry.Family, candidate.Entry.Selector, candidate.Entry.Platform), key)
			}
			candidate.Entry.Fallbacks = append(candidate.Entry.Fallbacks, artifact)
		}
		sort.Slice(candidate.Entry.Fallbacks, func(left, right int) bool {
			if candidate.Entry.Fallbacks[left].InstallName != candidate.Entry.Fallbacks[right].InstallName {
				return candidate.Entry.Fallbacks[left].InstallName < candidate.Entry.Fallbacks[right].InstallName
			}

			return candidate.Entry.Fallbacks[left].Ref < candidate.Entry.Fallbacks[right].Ref
		})
		entries = append(entries, candidate.Entry)
	}
	sortEntries(entries)
	defaults := generatedDefaults()
	if err := validateGeneratedDefaultReachability(defaults, entries); err != nil {
		return Catalog{}, errors.Wrap(err, "validate generated catalog defaults")
	}

	return Catalog{
		SchemaVersion: SchemaVersion,
		Source:        options.Source,
		Defaults:      defaults,
		Entries:       entries,
	}, nil
}

func generatedDefaults() Defaults {
	return Defaults{
		Family: defaultFamily,
		Selectors: []DefaultSelector{
			{Runtime: runtimeApple, Selector: targetVulkan},
			{Runtime: runtimeCPU, Selector: selectorDefault},
			{Runtime: runtimeCUDA, Selector: selectorNVIDIA},
			{
				Runtime: runtimeCUDA,
				Platform: &Platform{
					OS:           platformLinux,
					Architecture: architectureARM64,
				},
				Selector: selectorNVIDIAL4T,
			},
			{Runtime: runtimeROCm, Selector: selectorAMD},
		},
	}
}

func validateGeneratedDefaultReachability(defaults Defaults, entries []Entry) error {
	genericSelectors := make(map[string]string, len(defaults.Selectors))
	platformSelectors := make(map[string]string, len(defaults.Selectors))
	for _, selection := range defaults.Selectors {
		if selection.Platform == nil {
			genericSelectors[selection.Runtime] = selection.Selector
			continue
		}
		platform := normalizePlatform(*selection.Platform)
		platformSelectors[defaultRuntimePlatformKey(selection.Runtime, platform)] = selection.Selector
	}

	entriesByTuple := make(map[string]Entry, len(entries))
	platformsByRuntime := make(map[string]map[string]Platform)
	for _, entry := range entries {
		platform := normalizePlatform(entry.Platform)
		if entry.Family == defaults.Family {
			entriesByTuple[entryTupleKey(entry.Family, entry.Selector, platform)] = entry
		}
		if platformsByRuntime[entry.Runtime] == nil {
			platformsByRuntime[entry.Runtime] = make(map[string]Platform)
		}
		platformsByRuntime[entry.Runtime][platform.key()] = platform
	}

	runtimes := sortedMapKeys(genericSelectors)
	for _, runtime := range runtimes {
		platforms := platformsByRuntime[runtime]
		if len(platforms) == 0 {
			return fmt.Errorf("default family %q has no entries for runtime %q", defaults.Family, runtime)
		}
		platformKeys := sortedMapKeys(platforms)
		for _, platformKey := range platformKeys {
			platform := platforms[platformKey]
			selector := genericSelectors[runtime]
			if platformSelector, ok := platformSelectors[defaultRuntimePlatformKey(runtime, platform)]; ok {
				selector = platformSelector
			}
			if err := validateGeneratedDefaultTarget(defaults.Family, runtime, selector, platform, entriesByTuple); err != nil {
				return err
			}
		}
	}

	for _, selection := range defaults.Selectors {
		if selection.Platform == nil {
			continue
		}
		if err := validateGeneratedDefaultTarget(
			defaults.Family,
			selection.Runtime,
			selection.Selector,
			normalizePlatform(*selection.Platform),
			entriesByTuple,
		); err != nil {
			return err
		}
	}

	return nil
}

func validateGeneratedDefaultTarget(family, runtime, selector string, platform Platform, entriesByTuple map[string]Entry) error {
	entry, ok := entriesByTuple[entryTupleKey(family, selector, platform)]
	if !ok {
		return fmt.Errorf("default family %q runtime %q selector %q has no entry for platform %s", family, runtime, selector, platform.key())
	}
	if entry.Runtime != runtime {
		return fmt.Errorf(
			"default family %q selector %q platform %s has runtime %q, want %q",
			family,
			selector,
			platform.key(),
			entry.Runtime,
			runtime,
		)
	}
	if entry.Status != statusSupported && entry.Status != statusExperimental {
		return fmt.Errorf(
			"default family %q runtime %q selector %q platform %s has unavailable status %q",
			family,
			runtime,
			selector,
			platform.key(),
			entry.Status,
		)
	}

	return nil
}

func defaultRuntimePlatformKey(runtime string, platform Platform) string {
	return runtime + "/" + platform.key()
}

// Marshal returns canonical, indented JSON with a final newline.
func Marshal(catalog Catalog) ([]byte, error) {
	encoded, err := json.MarshalIndent(catalog, "", "  ")
	if err != nil {
		return nil, errors.Wrap(err, "marshal backend catalog")
	}

	return append(encoded, '\n'), nil
}

type cachedResolver struct {
	resolver Resolver
	results  map[string][]ResolvedManifest
}

func newCachedResolver(resolver Resolver) *cachedResolver {
	return &cachedResolver{resolver: resolver, results: make(map[string][]ResolvedManifest)}
}

func (resolver *cachedResolver) resolve(ctx context.Context, reference string) ([]ResolvedManifest, error) {
	if result, ok := resolver.results[reference]; ok {
		return append([]ResolvedManifest(nil), result...), nil
	}
	result, err := resolver.resolver.Resolve(ctx, reference)
	if err != nil {
		return nil, err
	}
	if len(result) == 0 {
		return nil, fmt.Errorf("resolver returned no manifests for %q", reference)
	}
	result = append([]ResolvedManifest(nil), result...)
	if err := normalizeResolvedManifests(result, reference, true); err != nil {
		return nil, err
	}
	resolver.results[reference] = result

	return append([]ResolvedManifest(nil), result...), nil
}

func resolveCore(ctx context.Context, resolver *cachedResolver, template string, platform Platform) (string, error) {
	reference, specialized, err := expandCoreReference(template, platform)
	if err != nil {
		return "", err
	}
	return resolvePlatformReference(ctx, resolver, reference, platform, specialized)
}

func resolvePlatformReference(ctx context.Context, resolver *cachedResolver, reference string, platform Platform, allowPlatformless bool) (string, error) {
	manifests, err := resolver.resolve(ctx, reference)
	if err != nil {
		return "", err
	}

	for _, manifest := range manifests {
		if manifest.Platform.key() == platform.key() {
			return immutableReference(reference, manifest.Digest), nil
		}
	}
	if allowPlatformless && len(manifests) == 1 && manifests[0].Platform.key() == "//" {
		return immutableReference(reference, manifests[0].Digest), nil
	}

	return "", fmt.Errorf("reference %q has no manifest for platform %s", reference, platform.key())
}

func expandCoreReference(template string, platform Platform) (string, bool, error) {
	reference := template
	specialized := false
	for placeholder, value := range map[string]string{
		"{os}":           platform.OS,
		"{architecture}": platform.Architecture,
		"{variant}":      platform.Variant,
	} {
		if strings.Contains(reference, placeholder) {
			specialized = true
			reference = strings.ReplaceAll(reference, placeholder, value)
		}
	}
	if strings.ContainsAny(reference, "{}") {
		return "", false, fmt.Errorf("core reference template %q contains an unknown placeholder", template)
	}
	if reference == "" {
		return "", false, errors.New("expanded core reference is empty")
	}

	return reference, specialized, nil
}

func stableVersionReference(reference, version string) (string, error) {
	if strings.Contains(reference, "://") {
		return "", fmt.Errorf("OCI reference %q must not contain a URL scheme", reference)
	}
	if strings.Contains(reference, "@") {
		return "", fmt.Errorf("stable source reference %q is unexpectedly digest-pinned", reference)
	}
	lastSlash := strings.LastIndex(reference, "/")
	lastColon := strings.LastIndex(reference, ":")
	if lastColon <= lastSlash {
		return "", fmt.Errorf("stable source reference %q has no tag", reference)
	}
	tag := reference[lastColon+1:]
	if tag != "latest" && !strings.HasPrefix(tag, "latest-") {
		return "", fmt.Errorf("stable source reference %q does not use a latest tag", reference)
	}
	if !strings.HasPrefix(version, "v") {
		return "", fmt.Errorf("LocalAI version %q must begin with v", version)
	}

	return reference[:lastColon+1] + version + strings.TrimPrefix(tag, "latest"), nil
}

func immutableReference(sourceReference, digest string) string {
	repository := sourceReference
	if at := strings.LastIndex(repository, "@"); at >= 0 {
		repository = repository[:at]
	}
	lastSlash := strings.LastIndex(repository, "/")
	if lastColon := strings.LastIndex(repository, ":"); lastColon > lastSlash {
		repository = repository[:lastColon]
	}

	return repository + "@" + digest
}

func normalizeWorkloads(tags []string) []string {
	unique := make(map[string]struct{}, len(tags))
	for _, tag := range tags {
		normalized := strings.ToLower(strings.TrimSpace(tag))
		normalized = strings.Map(func(character rune) rune {
			switch {
			case character >= 'a' && character <= 'z':
				return character
			case character >= '0' && character <= '9':
				return character
			case character == '.', character == '+':
				return character
			default:
				return '-'
			}
		}, normalized)
		for strings.Contains(normalized, "--") {
			normalized = strings.ReplaceAll(normalized, "--", "-")
		}
		normalized = strings.Trim(normalized, "-")
		if normalized != "" {
			unique[normalized] = struct{}{}
		}
	}
	workloads := make([]string, 0, len(unique))
	for workload := range unique {
		workloads = append(workloads, workload)
	}
	sort.Strings(workloads)

	return workloads
}

func sortedMapKeys[T any](values map[string]T) []string {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	return keys
}

func entryTupleKey(family, selector string, platform Platform) string {
	return family + "/" + selector + "/" + platform.key()
}

func sortEntries(entries []Entry) {
	sort.Slice(entries, func(left, right int) bool {
		leftKey := entryTupleKey(entries[left].Family, entries[left].Selector, entries[left].Platform)
		rightKey := entryTupleKey(entries[right].Family, entries[right].Selector, entries[right].Platform)
		if leftKey != rightKey {
			return leftKey < rightKey
		}

		return entries[left].Backend.Ref < entries[right].Backend.Ref
	})
}
