package inference

import (
	"sync"

	"github.com/kaito-project/aikit/pkg/aikit/config"
	"github.com/kaito-project/aikit/pkg/backendcatalog"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/pkg/errors"
)

var (
	defaultResolverOnce sync.Once
	defaultResolver     *backendcatalog.Resolver
	defaultResolverErr  error
)

// ResolveBackend resolves the exact backend install plan for a configuration and platform.
func ResolveBackend(c *config.InferenceConfig, platform specs.Platform) (backendcatalog.Resolution, error) {
	resolver, err := getDefaultResolver()
	if err != nil {
		return backendcatalog.Resolution{}, err
	}

	return ResolveBackendWithResolver(c, platform, resolver)
}

// ResolveBackendWithResolver resolves an exact backend tuple without applying runtime fallbacks.
func ResolveBackendWithResolver(c *config.InferenceConfig, platform specs.Platform, resolver *backendcatalog.Resolver) (backendcatalog.Resolution, error) {
	if c == nil {
		return backendcatalog.Resolution{}, errors.New("inference config is nil")
	}
	if resolver == nil {
		return backendcatalog.Resolution{}, errors.New("backend catalog resolver is nil")
	}

	var family string
	switch len(c.Backends) {
	case 0:
		// An omitted family selects the catalog default.
	case 1:
		family = c.Backends[0]
		if family == "" {
			return backendcatalog.Resolution{}, errors.New("backend cannot be empty")
		}
	default:
		return backendcatalog.Resolution{}, errors.New("only one backend is supported at this time")
	}

	runtime := requestedRuntime(c.Runtime)

	resolution, err := resolver.Resolve(backendcatalog.Request{
		Family:  family,
		Runtime: runtime,
		Platform: backendcatalog.Platform{
			OS:           platform.OS,
			Architecture: platform.Architecture,
			Variant:      platform.Variant,
		},
	})
	if err != nil {
		return backendcatalog.Resolution{}, errors.Wrapf(
			err,
			"resolving backend %q for runtime %q on %s/%s",
			family,
			runtime,
			platform.OS,
			platform.Architecture,
		)
	}

	if isRunnerMode(c) && resolution.RunnerProfile == backendcatalog.RunnerProfileUnsupported {
		return backendcatalog.Resolution{}, errors.Errorf(
			"backend %q runtime %q does not have an audited runner profile for %s/%s",
			resolution.Family,
			runtime,
			platform.OS,
			platform.Architecture,
		)
	}

	return resolution, nil
}

func requestedRuntime(runtime string) backendcatalog.Runtime {
	if runtime == "" {
		return backendcatalog.RuntimeCPU
	}

	return backendcatalog.Runtime(runtime)
}

func getDefaultResolver() (*backendcatalog.Resolver, error) {
	defaultResolverOnce.Do(func() {
		catalog, err := backendcatalog.Default()
		if err != nil {
			defaultResolverErr = errors.Wrap(err, "loading embedded backend catalog")
			return
		}

		defaultResolver, defaultResolverErr = backendcatalog.NewResolver(catalog)
		if defaultResolverErr != nil {
			defaultResolverErr = errors.Wrap(defaultResolverErr, "indexing embedded backend catalog")
		}
	})

	return defaultResolver, defaultResolverErr
}
