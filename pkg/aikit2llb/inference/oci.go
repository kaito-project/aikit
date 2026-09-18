package inference

import (
	"context"
	"encoding/json"
	"net"
	"path"
	"strconv"
	"strings"

	"github.com/containerd/containerd/v2/core/images"
	"github.com/containerd/platforms"
	"github.com/distribution/reference"
	"github.com/moby/buildkit/client/llb"
	"github.com/moby/buildkit/frontend/gateway/client"
	"github.com/moby/buildkit/solver/pb"
	"github.com/moby/buildkit/util/flightcontrol"
	"github.com/moby/buildkit/util/imageutil"
	digest "github.com/opencontainers/go-digest"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/pkg/errors"
)

const (
	ociManifestFilename = "/manifest.json"
	ociResolvedRefPath  = "/resolved-ref"
	ociBlobFilename     = "blob"
	maxOCIManifestSize  = 4 << 20
	maxOCIIndexDepth    = 8
	ollamaModelType     = "application/vnd.ollama.image.model"
	orasUnpack          = "io.deis.oras.content.unpack"
)

// OCIResolver resolves a model reference to a tree of downloaded files.
type OCIResolver interface {
	Resolve(context.Context, string, specs.Platform) (llb.State, error)
}

type ociResolver struct {
	client        client.Client
	buildPlatform specs.Platform
	cacheImports  []client.CacheOptionsEntry
	manifests     flightcontrol.CachedGroup[resolvedOCIManifest]
}

type resolvedOCIManifest struct {
	ref  reference.Canonical
	data []byte
}

// NewOCIResolver shares manifest lookups across a build's target platforms.
// Older builders retain the ORAS download path when blob sources are unavailable.
func NewOCIResolver(c client.Client, buildPlatform specs.Platform, cacheImports []client.CacheOptionsEntry) OCIResolver {
	caps := c.BuildOpts().LLBCaps
	if !caps.Contains(pb.CapSourceImageBlob) || caps.Supports(pb.CapSourceImageBlob) != nil {
		return nil
	}
	return &ociResolver{client: c, buildPlatform: buildPlatform, cacheImports: cacheImports}
}

func copyOCIModel(source string, state llb.State, target specs.Platform, resolver OCIResolver) llb.State {
	return state.Async(func(ctx context.Context, state llb.State, _ *llb.Constraints) (llb.State, error) {
		model, err := resolver.Resolve(ctx, source, target)
		if err != nil {
			return llb.State{}, errors.Wrapf(err, "resolving OCI model %s", source)
		}
		return state.File(
			llb.Copy(model, "/", "/models/", &llb.CopyInfo{CopyDirContentsOnly: true, CreateDestPath: true}),
			llb.WithCustomName("Copying OCI model "+source+" to /models"),
		), nil
	})
}

func (r *ociResolver) Resolve(ctx context.Context, source string, target specs.Platform) (llb.State, error) {
	named, err := reference.ParseNormalizedNamed(strings.TrimPrefix(source, "oci://"))
	if err != nil {
		return llb.State{}, errors.Wrap(err, "invalid OCI model reference")
	}
	named = reference.TagNameOnly(named)
	manifest, err := r.readManifest(ctx, named)
	if err != nil {
		return llb.State{}, err
	}

	for depth := 0; ; depth++ {
		mediaType, err := imageutil.DetectManifestBlobMediaType(manifest.data)
		if err != nil {
			return llb.State{}, errors.Wrap(err, "invalid OCI manifest")
		}
		switch mediaType {
		case specs.MediaTypeImageIndex, images.MediaTypeDockerSchema2ManifestList:
			if depth >= maxOCIIndexDepth {
				return llb.State{}, errors.New("OCI model index nesting exceeds limit")
			}
			var index specs.Index
			if err := json.Unmarshal(manifest.data, &index); err != nil {
				return llb.State{}, errors.Wrap(err, "decoding OCI index")
			}
			if index.SchemaVersion != 2 {
				return llb.State{}, errors.New("OCI model index must use schema version 2")
			}
			child, err := selectOCIManifest(index, target)
			if err != nil {
				return llb.State{}, err
			}
			childRef, err := reference.WithDigest(reference.TrimNamed(manifest.ref), child.Digest)
			if err != nil {
				return llb.State{}, errors.Wrap(err, "invalid OCI manifest digest")
			}
			manifest, err = r.readManifest(ctx, childRef)
			if err != nil {
				return llb.State{}, err
			}
			if int64(len(manifest.data)) != child.Size {
				return llb.State{}, errors.New("OCI model manifest size mismatch")
			}
		case specs.MediaTypeImageManifest, images.MediaTypeDockerSchema2Manifest:
			var image specs.Manifest
			if err := json.Unmarshal(manifest.data, &image); err != nil {
				return llb.State{}, errors.Wrap(err, "decoding OCI model manifest")
			}
			if image.SchemaVersion != 2 {
				return llb.State{}, errors.New("OCI model manifest must use schema version 2")
			}
			return r.modelFiles(manifest.ref, image)
		default:
			return llb.State{}, errors.Errorf("unsupported OCI model manifest type %q", mediaType)
		}
	}
}

func (r *ociResolver) readManifest(ctx context.Context, named reference.Named) (resolvedOCIManifest, error) {
	return r.manifests.Do(ctx, named.String(), func(ctx context.Context) (resolvedOCIManifest, error) {
		var result resolvedOCIManifest
		// Registry manifests use a different endpoint from native blob sources.
		// ORAS also handles artifact configs that the image metadata API cannot resolve.
		script := "ref=\"$1\"\nshift\n"
		opts := []llb.RunOption{llb.WithCustomName("Resolving OCI model manifest " + named.String())}
		if canonical, ok := named.(reference.Canonical); ok {
			result.ref = canonical
		} else {
			// Resolve mutable tags once per request, without caching the lookup across builds.
			script += "ref=$(oras resolve \"$@\" --full-reference \"$ref\")\nprintf '%s\\n' \"$ref\" > /resolved-ref\n"
			opts = append(opts, llb.IgnoreCache)
		}
		script += "oras manifest fetch \"$@\" \"$ref\" > /manifest.json\n"
		args := []string{"/bin/sh", "-ec", script, "aikit-oci-manifest", named.String()}
		if isLocalOCIRegistry(reference.Domain(named)) {
			args = append(args, "--insecure")
		}
		state := orasToolingImage(r.buildPlatform).Run(append(opts, llb.Args(args))...).Root()
		definition, err := state.Marshal(ctx)
		if err != nil {
			return result, errors.Wrap(err, "marshaling OCI manifest lookup")
		}
		solved, err := r.client.Solve(ctx, client.SolveRequest{Definition: definition.ToPB(), CacheImports: r.cacheImports})
		if err != nil {
			return result, errors.Wrap(err, "fetching OCI model manifest")
		}
		ref, err := solved.SingleRef()
		if err != nil {
			return result, err
		}
		if ref == nil {
			return result, errors.New("OCI manifest lookup returned no reference")
		}
		if result.ref == nil {
			data, err := ref.ReadFile(ctx, client.ReadRequest{Filename: ociResolvedRefPath, Range: &client.FileRange{Length: 4096}})
			if err != nil {
				return result, errors.Wrap(err, "reading resolved OCI reference")
			}
			resolved, err := reference.ParseNormalizedNamed(strings.TrimSpace(string(data)))
			if err != nil {
				return result, errors.Wrap(err, "invalid resolved OCI reference")
			}
			canonical, ok := resolved.(reference.Canonical)
			if !ok || canonical.Name() != named.Name() {
				return result, errors.New("OCI manifest lookup did not return a digest in the requested repository")
			}
			result.ref = canonical
		}
		result.data, err = ref.ReadFile(ctx, client.ReadRequest{Filename: ociManifestFilename, Range: &client.FileRange{Length: maxOCIManifestSize + 1}})
		if err != nil {
			return result, errors.Wrap(err, "reading OCI manifest")
		}
		if len(result.data) > maxOCIManifestSize {
			return result, errors.New("OCI model manifest exceeds 4 MiB")
		}
		if err := result.ref.Digest().Validate(); err != nil {
			return result, errors.Wrap(err, "invalid OCI manifest digest")
		}
		if result.ref.Digest().Algorithm().FromBytes(result.data) != result.ref.Digest() {
			return result, errors.New("OCI model manifest digest mismatch")
		}
		return result, nil
	})
}

func selectOCIManifest(index specs.Index, target specs.Platform) (specs.Descriptor, error) {
	candidates := make([]specs.Descriptor, 0, len(index.Manifests))
	uniquePlatforms := make(map[string]struct{})
	for _, descriptor := range index.Manifests {
		if descriptor.Annotations["vnd.docker.reference.type"] == "attestation-manifest" {
			continue
		}
		if p := descriptor.Platform; p != nil {
			if p.OS == "unknown" || p.Architecture == "unknown" {
				continue
			}
			if p.OS != "" && p.Architecture != "" {
				uniquePlatforms[platforms.FormatAll(platforms.Normalize(*p))] = struct{}{}
			}
		}
		candidates = append(candidates, descriptor)
	}
	if len(candidates) == 0 {
		return specs.Descriptor{}, errors.New("OCI model index has no model manifests")
	}
	// A single platform stamp is common on platform-neutral ModelPack indexes.
	// Only require a target match when the index actually offers multiple platforms.
	selected := candidates[0]
	if len(uniquePlatforms) > 1 {
		matcher := platforms.Only(target)
		found := false
		for _, candidate := range candidates {
			if candidate.Platform != nil && matcher.Match(*candidate.Platform) &&
				(!found || matcher.Less(*candidate.Platform, *selected.Platform)) {
				selected, found = candidate, true
			}
		}
		if !found {
			return specs.Descriptor{}, errors.Errorf("OCI model index has no manifest for %s", platforms.FormatAll(target))
		}
	}
	if err := selected.Digest.Validate(); err != nil {
		return specs.Descriptor{}, errors.Wrap(err, "invalid OCI model manifest digest")
	}
	if selected.Size < 0 || selected.Size > maxOCIManifestSize {
		return specs.Descriptor{}, errors.New("invalid OCI model manifest size")
	}
	return selected, nil
}

func (r *ociResolver) modelFiles(ref reference.Canonical, manifest specs.Manifest) (llb.State, error) {
	state := llb.Scratch()
	names := make(map[string]struct{})
	ollama := reference.Domain(ref) == ollamaRegistryURL
	layers := manifest.Layers
	if !ollama && manifest.Config.Annotations[specs.AnnotationTitle] != "" {
		layers = append(layers, manifest.Config)
	}
	for _, layer := range layers {
		name := layer.Annotations[specs.AnnotationTitle]
		if ollama {
			if layer.MediaType != ollamaModelType {
				continue
			}
			name = path.Base(reference.Path(ref))
		}
		if name == "" {
			continue
		}
		clean := path.Clean(name)
		if path.IsAbs(name) || clean == "." || clean == ".." || strings.HasPrefix(clean, "../") || strings.ContainsAny(name, "\\\x00") {
			return llb.State{}, errors.Errorf("unsafe OCI model filename %q", name)
		}
		if _, exists := names[clean]; exists {
			return llb.State{}, errors.Errorf("duplicate OCI model filename %q", clean)
		}
		names[clean] = struct{}{}
		if err := layer.Digest.Validate(); err != nil {
			return llb.State{}, errors.Wrap(err, "invalid OCI model blob digest")
		}
		if layer.Digest.Algorithm() != digest.SHA256 || layer.Size < 0 {
			return llb.State{}, errors.New("OCI model blobs require a sha256 digest and non-negative size")
		}
		blob := llb.ImageBlob(reference.TrimNamed(ref).Name()+"@"+layer.Digest.String(), llb.Filename(ociBlobFilename), llb.Chmod(0o644))
		// Preserve ORAS's digest and size verification, including after cache imports.
		verified := orasToolingImage(r.buildPlatform).Network(pb.NetMode_NONE).Run(
			llb.Args([]string{"/bin/sh", "-ec", `test "$(wc -c < /blob/blob)" -eq "$2"
printf '%s  /blob/blob\n' "$1" | sha256sum -c -`, "aikit-verify-blob", layer.Digest.Encoded(), strconv.FormatInt(layer.Size, 10)}),
			llb.AddMount("/blob", blob),
			llb.WithCustomName("Verifying OCI model blob "+layer.Digest.String()),
		).GetMount("/blob")
		copyInfo := &llb.CopyInfo{CreateDestPath: true}
		destination := "/" + clean
		if layer.Annotations[orasUnpack] == "true" {
			unpacked := llb.Scratch().File(llb.Copy(verified, "/"+ociBlobFilename, "/", &llb.CopyInfo{AttemptUnpack: true}))
			copyInfo.CopyDirContentsOnly = true
			state = state.File(llb.Copy(unpacked, destination, destination, copyInfo))
			continue
		}
		state = state.File(llb.Copy(verified, "/"+ociBlobFilename, destination, copyInfo))
	}
	if len(names) == 0 {
		return llb.State{}, errors.New("OCI model manifest has no downloadable model files")
	}
	return state, nil
}

func isLocalOCIRegistry(domain string) bool {
	host := domain
	if h, _, err := net.SplitHostPort(domain); err == nil {
		host = h
	}
	if host == "localhost" {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}
