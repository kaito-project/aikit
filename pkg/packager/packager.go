package packager

import (
	"archive/tar"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/kaito-project/aikit/pkg/aikit2llb/inference"
	"github.com/kaito-project/aikit/pkg/utils"
	v1 "github.com/modelpack/model-spec/specs-go/v1"
	digest "github.com/opencontainers/go-digest"
	"github.com/opencontainers/image-spec/specs-go"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
)

// SpecType identifies the target artifact spec.
type SpecType string

const (
	// SpecModelPack builds an artifact following ModelPack media types.
	SpecModelPack SpecType = "modelpack"
	// SpecGeneric builds a generic OCI artifact with a single tar layer.
	SpecGeneric SpecType = "generic"
)

// Options configures the pack operation.
type Options struct {
	// Source can be a local filesystem path or a remote reference like hf://org/repo
	Source string
	// OutputDir is the directory to write the OCI layout into. Will be created.
	OutputDir string
	// ArtifactType overrides the artifactType in the manifest. If empty, defaults per spec.
	ArtifactType string
	// Spec selects which spec to use. Defaults to SpecModelPack.
	Spec SpecType
	// Name is a human-friendly name attached as annotation on the index.
	Name string
	// MediaTypes allows overriding media types for ModelPack outputs.
	// Leave empty to use ModelPack defaults.
	MediaTypes ModelPackMediaTypes
}

// Result contains pointers to the generated files.
type Result struct {
	LayoutPath string // path to OCI layout directory (OutputDir)
}

// ModelPackMediaTypes holds optional overrides for ModelPack media types.
// Leave fields empty to use the defaults from the ModelPack spec.
type ModelPackMediaTypes struct {
	// ManifestConfig is the media type used for the manifest config descriptor.
	ManifestConfig string
	// LayerWeights overrides the media type for the weights layer(s).
	LayerWeights string
	// LayerConfig overrides the media type for the config/metadata layer(s).
	LayerConfig string
	// LayerDocs overrides the media type for the documentation layer(s).
	LayerDocs string
}

// Pack builds an OCI artifact from the given source according to the spec.
func pack(ctx context.Context, opts Options) (*Result, error) {
	if opts.Source == "" {
		return nil, fmt.Errorf("source is required")
	}
	if opts.OutputDir == "" {
		return nil, fmt.Errorf("output directory is required")
	}
	if opts.Spec == "" {
		opts.Spec = SpecModelPack
	}

	// Resolve source to a local directory path.
	srcDir, cleanup, err := resolveSource(ctx, opts.Source)
	if err != nil {
		return nil, err
	}
	defer func() {
		if cleanup != nil {
			_ = cleanup()
		}
	}()

	// Prepare OCI layout directory structure.
	if err := os.MkdirAll(filepath.Join(opts.OutputDir, "blobs", "sha256"), 0o755); err != nil {
		return nil, fmt.Errorf("create layout dirs: %w", err)
	}
	if err := os.WriteFile(filepath.Join(opts.OutputDir, "oci-layout"), []byte("{\n  \"imageLayoutVersion\": \"1.0.0\"\n}\n"), 0o600); err != nil {
		return nil, fmt.Errorf("write oci-layout: %w", err)
	}

	// Build layers according to spec.
	var layers []ocispec.Descriptor
	var configDesc ocispec.Descriptor
	switch opts.Spec {
	case SpecModelPack:
		layers, err = buildModelPackLayers(srcDir, opts.OutputDir, opts.MediaTypes)
		if err != nil {
			return nil, err
		}
		// manifest config media type (default ModelPack config)
		mcmt := opts.MediaTypes.ManifestConfig
		if mcmt == "" {
			mcmt = v1.MediaTypeModelConfig
		}
		configDesc, err = writeJSONBlob(opts.OutputDir, map[string]any{}, mcmt)
		if err != nil {
			return nil, err
		}
	case SpecGeneric:
		layer, err := buildGenericLayer(srcDir, opts.OutputDir)
		if err != nil {
			return nil, err
		}
		layers = []ocispec.Descriptor{layer}
		configDesc, err = writeJSONBlob(opts.OutputDir, map[string]any{}, ocispec.MediaTypeImageConfig)
		if err != nil {
			return nil, err
		}
	default:
		return nil, fmt.Errorf("unknown spec: %s", opts.Spec)
	}

	// Manifest
	artifactType := opts.ArtifactType
	if artifactType == "" {
		if opts.Spec == SpecModelPack {
			artifactType = v1.ArtifactTypeModelManifest
		} else {
			artifactType = "application/vnd.oci.artifact" // generic identifier
		}
	}
	manifest := ocispec.Manifest{
		Versioned: specs.Versioned{
			SchemaVersion: 2,
		},
		MediaType:    ocispec.MediaTypeImageManifest,
		ArtifactType: artifactType,
		Config:       configDesc,
		Layers:       layers,
	}
	manifestDesc, err := writeJSONBlob(opts.OutputDir, manifest, ocispec.MediaTypeImageManifest)
	if err != nil {
		return nil, err
	}

	// Index
	idx := ocispec.Index{
		MediaType: ocispec.MediaTypeImageIndex,
		Manifests: []ocispec.Descriptor{manifestDesc},
		Annotations: map[string]string{
			ocispec.AnnotationRefName:     safeRefName(opts.Name),
			ocispec.AnnotationCreated:     time.Now().UTC().Format(time.RFC3339Nano),
			ocispec.AnnotationTitle:       opts.Name,
			ocispec.AnnotationDescription: fmt.Sprintf("AI model packaged by AIKit (%s)", string(opts.Spec)),
		},
	}
	if _, err := writeJSONFile(filepath.Join(opts.OutputDir, "index.json"), idx, ocispec.MediaTypeImageIndex); err != nil {
		return nil, err
	}

	return &Result{LayoutPath: opts.OutputDir}, nil
}

// old resolveSource replaced by pluggable resolvers below

func buildModelPackLayers(srcDir, outDir string, mt ModelPackMediaTypes) ([]ocispec.Descriptor, error) {
	// Partition files
	var weights, cfg, docs []string
	err := filepath.WalkDir(srcDir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		rel, _ := filepath.Rel(srcDir, path)
		lower := strings.ToLower(filepath.Base(path))
		switch {
		case hasAnySuffix(lower, ".safetensors", ".bin", ".gguf", ".pt", ".ckpt"):
			weights = append(weights, rel)
		case lower == "config.json" || lower == "tokenizer.json" || strings.HasSuffix(lower, ".json") || strings.HasSuffix(lower, ".txt"):
			cfg = append(cfg, rel)
		case strings.HasPrefix(lower, "readme") || strings.HasPrefix(lower, "license") || strings.HasSuffix(lower, ".md"):
			docs = append(docs, rel)
		default:
			// put unknown small files into config layer
			info, _ := d.Info()
			if info != nil && info.Size() < 10*1024*1024 {
				cfg = append(cfg, rel)
			} else {
				weights = append(weights, rel)
			}
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	sort.Strings(weights)
	sort.Strings(cfg)
	sort.Strings(docs)

	// Defaults from ModelPack spec when not overridden
	if mt.LayerWeights == "" {
		mt.LayerWeights = v1.MediaTypeModelWeight
	}
	if mt.LayerConfig == "" {
		mt.LayerConfig = v1.MediaTypeModelWeightConfig
	}
	if mt.LayerDocs == "" {
		mt.LayerDocs = v1.MediaTypeModelDoc
	}

	var descs []ocispec.Descriptor
	if len(weights) > 0 {
		d, err := buildTarLayer(srcDir, outDir, weights, mt.LayerWeights)
		if err != nil {
			return nil, err
		}
		descs = append(descs, d)
	}
	if len(cfg) > 0 {
		d, err := buildTarLayer(srcDir, outDir, cfg, mt.LayerConfig)
		if err != nil {
			return nil, err
		}
		descs = append(descs, d)
	}
	if len(docs) > 0 {
		d, err := buildTarLayer(srcDir, outDir, docs, mt.LayerDocs)
		if err != nil {
			return nil, err
		}
		descs = append(descs, d)
	}
	return descs, nil
}

func buildGenericLayer(srcDir, outDir string) (ocispec.Descriptor, error) {
	// Add all files into a single tar layer
	var files []string
	err := filepath.WalkDir(srcDir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		rel, _ := filepath.Rel(srcDir, path)
		files = append(files, rel)
		return nil
	})
	if err != nil {
		return ocispec.Descriptor{}, err
	}
	sort.Strings(files)
	return buildTarLayer(srcDir, outDir, files, ocispec.MediaTypeImageLayer)
}

func buildTarLayer(srcDir, outDir string, relPaths []string, mediaType string) (ocispec.Descriptor, error) {
	// Create a deterministic tar into memory-backed buffer (acceptable for small tests). For large files, a temp file would be better.
	// We'll stream into a temp file to avoid memory issues.
	tmpf, err := os.CreateTemp("", "aikit-layer-*.tar")
	if err != nil {
		return ocispec.Descriptor{}, err
	}
	defer func() { _ = os.Remove(tmpf.Name()) }()

	tw := tar.NewWriter(tmpf)
	// Deterministic ordering
	for _, rel := range relPaths {
		full := filepath.Join(srcDir, rel)
		st, err := os.Stat(full)
		if err != nil {
			_ = tw.Close()
			return ocispec.Descriptor{}, err
		}
		hdr, err := tar.FileInfoHeader(st, "")
		if err != nil {
			_ = tw.Close()
			return ocispec.Descriptor{}, err
		}
		// Normalize header fields
		hdr.Name = filepath.ToSlash(rel)
		hdr.Mode = 0o644
		hdr.Uid = 0
		hdr.Gid = 0
		hdr.Uname = ""
		hdr.Gname = ""
		hdr.ModTime = time.Unix(0, 0)
		if err := tw.WriteHeader(hdr); err != nil {
			_ = tw.Close()
			return ocispec.Descriptor{}, err
		}
		f, err := os.Open(full)
		if err != nil {
			_ = tw.Close()
			return ocispec.Descriptor{}, err
		}
		if _, err := io.Copy(tw, f); err != nil {
			_ = f.Close()
			_ = tw.Close()
			return ocispec.Descriptor{}, err
		}
		_ = f.Close()
	}
	if err := tw.Close(); err != nil {
		return ocispec.Descriptor{}, err
	}
	// Compute digest and size
	if _, err := tmpf.Seek(0, io.SeekStart); err != nil {
		return ocispec.Descriptor{}, err
	}
	h := digest.SHA256.Digester()
	n, err := io.Copy(h.Hash(), tmpf)
	if err != nil {
		return ocispec.Descriptor{}, err
	}
	dgst := h.Digest()
	// Move to blobs location
	if _, err := tmpf.Seek(0, io.SeekStart); err != nil {
		return ocispec.Descriptor{}, err
	}
	blobPath := filepath.Join(outDir, "blobs", "sha256", strings.TrimPrefix(dgst.String(), "sha256:"))
	out, err := os.OpenFile(blobPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o600)
	if err != nil {
		return ocispec.Descriptor{}, err
	}
	if _, err := io.Copy(out, tmpf); err != nil {
		_ = out.Close()
		return ocispec.Descriptor{}, err
	}
	_ = out.Close()

	return ocispec.Descriptor{
		MediaType: mediaType,
		Digest:    dgst,
		Size:      n,
	}, nil
}

func writeJSONBlob(outDir string, v any, mediaType string) (ocispec.Descriptor, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return ocispec.Descriptor{}, err
	}
	dgst := digest.SHA256.FromBytes(b)
	blobPath := filepath.Join(outDir, "blobs", "sha256", strings.TrimPrefix(dgst.String(), "sha256:"))
	if err := os.WriteFile(blobPath, b, 0o600); err != nil {
		return ocispec.Descriptor{}, err
	}
	return ocispec.Descriptor{MediaType: mediaType, Digest: dgst, Size: int64(len(b))}, nil
}

func writeJSONFile(p string, v any, _ string) (ocispec.Descriptor, error) { // returns dummy descriptor for symmetry
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return ocispec.Descriptor{}, err
	}
	if err := os.WriteFile(p, b, 0o600); err != nil {
		return ocispec.Descriptor{}, err
	}
	return ocispec.Descriptor{}, nil
}

func hasAnySuffix(s string, suff ...string) bool {
	for _, x := range suff {
		if strings.HasSuffix(s, x) {
			return true
		}
	}
	return false
}

func safeRefName(s string) string {
	if s == "" {
		return "aikit/model"
	}
	// Convert spaces and invalid chars to dashes
	var b bytes.Buffer
	for _, r := range s {
		switch {
		case r >= 'a' && r <= 'z':
			b.WriteRune(r)
		case r >= '0' && r <= '9':
			b.WriteRune(r)
		case r == '/' || r == '.' || r == '_' || r == '-':
			b.WriteRune(r)
		case r >= 'A' && r <= 'Z':
			b.WriteRune(r + ('a' - 'A'))
		default:
			b.WriteByte('-')
		}
	}
	return b.String()
}

// SourceResolver resolves a source URI into a local directory.
type SourceResolver func(context.Context, string) (dir string, cleanup func() error, err error)

var resolvers = map[string]SourceResolver{
	"file": localResolver,
	"":     localResolver,
	"hf":   hfResolver,
	// Single-file resolvers
	"huggingface": huggingfaceFileResolver, // expecting huggingface://org/repo/[branch/]file
	"https":       httpResolver,
	"http":        httpResolver,
}

func resolveSource(ctx context.Context, source string) (string, func() error, error) {
	// Parse scheme
	scheme := ""
	if i := strings.Index(source, "://"); i != -1 {
		scheme = source[:i]
	}
	r, ok := resolvers[scheme]
	if !ok {
		return "", nil, fmt.Errorf("unsupported source scheme: %s", scheme)
	}
	return r(ctx, source)
}

// localResolver treats the source as a local filesystem path (with or without file:// prefix).
func localResolver(_ context.Context, source string) (string, func() error, error) {
	source = strings.TrimPrefix(source, "file://")
	abs, err := filepath.Abs(source)
	if err != nil {
		return "", nil, err
	}
	st, err := os.Stat(abs)
	if err != nil {
		return "", nil, err
	}
	if !st.IsDir() {
		return "", nil, fmt.Errorf("source must be a directory: %s", abs)
	}
	return abs, nil, nil
}

// hfResolver clones a Hugging Face repository (models) via git into a temp dir.
// Example: hf://TheBloke/Mistral-7B-v0.1-GGUF
func hfResolver(ctx context.Context, source string) (string, func() error, error) {
	const prefix = "hf://"
	if !strings.HasPrefix(source, prefix) {
		return "", nil, fmt.Errorf("invalid hf source: %s", source)
	}
	repo := strings.TrimPrefix(source, prefix)
	if repo == "" || strings.Contains(repo, " ") || strings.Contains(repo, "..") {
		return "", nil, fmt.Errorf("invalid hf repo: %q", repo)
	}
	tmp, err := os.MkdirTemp("", "aikit-hf-*")
	if err != nil {
		return "", nil, err
	}
	cleanup := func() error { return os.RemoveAll(tmp) }
	url := fmt.Sprintf("https://huggingface.co/%s", repo)
	cmd := exec.CommandContext(ctx, "git", "clone", "--depth", "1", url, tmp)
	if out, err := cmd.CombinedOutput(); err != nil {
		_ = cleanup()
		return "", nil, fmt.Errorf("git clone failed: %w: %s", err, string(out))
	}
	return tmp, cleanup, nil
}

// huggingfaceFileResolver downloads a single file from Hugging Face using the
// shared parser (supports optional branch). Example:
//
//	huggingface://org/repo/[branch/]path/to/file
func huggingfaceFileResolver(ctx context.Context, source string) (string, func() error, error) {
	hfURL, fileName, err := inference.ParseHuggingFaceURL(source)
	if err != nil {
		return "", nil, fmt.Errorf("invalid huggingface source: %w", err)
	}
	return downloadToTempDir(ctx, hfURL, fileName)
}

// httpResolver downloads a single file from an HTTP(S) URL into a temp dir.
func httpResolver(ctx context.Context, source string) (string, func() error, error) {
	// Deduce filename from URL
	fname := utils.FileNameFromURL(source)
	if fname == "" {
		return "", nil, fmt.Errorf("could not determine filename from URL")
	}
	return downloadToTempDir(ctx, source, fname)
}

func downloadToTempDir(ctx context.Context, url, fileName string) (string, func() error, error) {
	tmpDir, err := os.MkdirTemp("", "aikit-src-*")
	if err != nil {
		return "", nil, err
	}
	cleanup := func() error { return os.RemoveAll(tmpDir) }

	dstPath := filepath.Join(tmpDir, filepath.Base(fileName))
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		_ = cleanup()
		return "", nil, err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		_ = cleanup()
		return "", nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		_ = cleanup()
		return "", nil, fmt.Errorf("download failed: %s", resp.Status)
	}
	f, err := os.OpenFile(dstPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		_ = cleanup()
		return "", nil, err
	}
	if _, err := io.Copy(f, resp.Body); err != nil {
		_ = f.Close()
		_ = cleanup()
		return "", nil, err
	}
	if err := f.Close(); err != nil {
		_ = cleanup()
		return "", nil, err
	}
	return tmpDir, cleanup, nil
}
