package inference

import (
	"sync"

	"github.com/kaito-project/aikit/pkg/aikit/config"
	"github.com/kaito-project/aikit/pkg/backendcatalog"
	"github.com/kaito-project/aikit/pkg/utils"
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

	family := defaultBackendName
	if len(c.Backends) > 0 {
		family = c.Backends[0]
	}

	selector := backendcatalog.Selector(c.BackendCapability)
	if selector == "" {
		selector = defaultSelectorForRuntime(c.Runtime)
	}

	resolution, err := resolver.Resolve(backendcatalog.Request{
		Family:   family,
		Selector: selector,
		Platform: backendcatalog.Platform{
			OS:           platform.OS,
			Architecture: platform.Architecture,
			Variant:      platform.Variant,
		},
	})
	if err != nil {
		return backendcatalog.Resolution{}, errors.Wrapf(
			err,
			"resolving backend %q selector %q for %s/%s",
			family,
			selector,
			platform.OS,
			platform.Architecture,
		)
	}

	expectedRuntime, err := catalogRuntime(c.Runtime)
	if err != nil {
		return backendcatalog.Resolution{}, err
	}
	if resolution.Runtime != expectedRuntime {
		return backendcatalog.Resolution{}, errors.Errorf(
			"backend %q selector %q requires runtime %q, but aikitfile runtime is %q",
			family,
			selector,
			resolution.Runtime,
			c.Runtime,
		)
	}
	if isRunnerMode(c) && resolution.RunnerProfile == backendcatalog.RunnerProfileUnsupported {
		return backendcatalog.Resolution{}, errors.Errorf(
			"backend %q selector %q does not have an audited runner profile for %s/%s",
			family,
			selector,
			platform.OS,
			platform.Architecture,
		)
	}

	return resolution, nil
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

func defaultSelectorForRuntime(runtime string) backendcatalog.Selector {
	switch runtime {
	case utils.RuntimeNVIDIA:
		return backendcatalog.SelectorNVIDIA
	case utils.RuntimeROCm:
		return backendcatalog.SelectorAMD
	case utils.RuntimeAppleSilicon:
		return backendcatalog.SelectorVulkan
	default:
		return backendcatalog.SelectorDefault
	}
}

func catalogRuntime(runtime string) (backendcatalog.Runtime, error) {
	switch runtime {
	case "":
		return backendcatalog.RuntimeCPU, nil
	case utils.RuntimeNVIDIA:
		return backendcatalog.RuntimeCUDA, nil
	case utils.RuntimeROCm:
		return backendcatalog.RuntimeROCm, nil
	case utils.RuntimeAppleSilicon:
		return backendcatalog.RuntimeAppleSilicon, nil
	default:
		return "", errors.Errorf("runtime %q is not supported", runtime)
	}
}
