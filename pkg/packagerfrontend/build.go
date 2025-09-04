// Package packagerfrontend implements the BuildKit gateway frontend used to
// fetch model sources (local, HTTP, Hugging Face) and produce a minimal image
// containing those artifacts for further export (image/oci layout).
package packagerfrontend

import (
	"context"
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"strings"

	"github.com/containerd/platforms"
	"github.com/kaito-project/aikit/pkg/aikit2llb/inference"
	"github.com/kaito-project/aikit/pkg/oci/packager"
	v1 "github.com/modelpack/model-spec/specs-go/v1"
	"github.com/moby/buildkit/client/llb"
	"github.com/moby/buildkit/frontend/gateway/client"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/moby/buildkit/exporter/containerimage/exptypes"
	digest "github.com/opencontainers/go-digest"
)

const (
	localNameContext = "context"
)

// Build is the BuildKit frontend entrypoint for the packager syntax.
// It takes build-arg: source, and optionally spec, then creates an image
// containing the referenced model/artifacts.
func Build(ctx context.Context, c client.Client) (*client.Result, error) {
	opts := c.BuildOpts().Opts
	sessionID := c.BuildOpts().SessionID

	source := getBuildArg(opts, "source")
	emit := getBuildArg(opts, "emit")

	// Scalable LLB-based modelpack packaging (shell implementation; no extra binary build)
	if emit == "modelpack-llb" {
		if source == "" { return nil, fmt.Errorf("source is required for emit=modelpack-llb") }
		// Only modelpack spec
		name := getBuildArg(opts, "name")
		artifactType := v1.ArtifactTypeModelManifest
		mtManifest := v1.MediaTypeModelConfig
		mtWeights := v1.MediaTypeModelWeight
		mtConfig := v1.MediaTypeModelWeightConfig
		mtDocs := v1.MediaTypeModelDoc

		// Build model source state
		var modelState llb.State
		switch {
		case source == "." || source == "context":
			modelState = llb.Local(localNameContext, llb.SessionID(sessionID), llb.SharedKeyHint(localNameContext))
		case strings.HasPrefix(source, "https://") || strings.HasPrefix(source, "http://"):
			// Ensure remote single file keeps its basename (avoids anonymous tmp name issues)
			base := path.Base(source)
			modelState = llb.HTTP(source, llb.Filename(base))
		case strings.HasPrefix(source, "huggingface://"):
			hfURL, modelFile, err := inference.ParseHuggingFaceURL(source)
			if err != nil { return nil, fmt.Errorf("invalid huggingface source: %w", err) }
			modelState = llb.HTTP(hfURL, llb.Filename(modelFile))
		default:
			include := source
			if strings.HasSuffix(include, "/") { include += "**" }
			modelState = llb.Local(localNameContext,
				llb.IncludePatterns([]string{include}),
				llb.SessionID(sessionID),
				llb.SharedKeyHint(localNameContext+":"+include),
			)
		}

		// Shell script (bash) to classify files and build OCI layout deterministically.
		script := fmt.Sprintf(`set -e
mkdir -p /layout/blobs/sha256
src=/src
if [ -f /src ]; then mkdir -p /worksrc && cp /src /worksrc/; src=/worksrc; fi
cd "$src"
# Prepare list files
> /tmp/weights.list
> /tmp/config.list
> /tmp/docs.list

while IFS= read -r f; do
	base=$(basename "$f" | tr A-Z a-z)
	case "$base" in
		*.safetensors|*.bin|*.gguf|*.pt|*.ckpt)
			echo "$f" >> /tmp/weights.list ;;
		config.json|tokenizer.json|*.json|*.txt)
			echo "$f" >> /tmp/config.list ;;
		readme*|license*|*.md)
			echo "$f" >> /tmp/docs.list ;;
		*)
			sz=$(stat -c%%s "$f")
			if [ "$sz" -lt 10485760 ]; then echo "$f" >> /tmp/config.list; else echo "$f" >> /tmp/weights.list; fi ;;
	esac
done < <(find . -type f -print | sed 's|^./||' | LC_ALL=C sort)

make_tar() {
	listfile="$1"; out="$2"
	[ ! -s "$listfile" ] && return 0
	tar --sort=name --mtime='@0' --owner=0 --group=0 --numeric-owner -cf "$out" -T "$listfile"
}

layers_json=""
add_layer() {
	file="$1"; mt="$2"
	[ ! -f "$file" ] && return 0
	dgst=$(sha256sum "$file" | cut -d' ' -f1)
	size=$(stat -c%%s "$file")
	mv "$file" /layout/blobs/sha256/$dgst
	if [ -n "$layers_json" ]; then layers_json="$layers_json , "; fi
	layers_json="${layers_json}{ \"mediaType\": \"$mt\", \"digest\": \"sha256:$dgst\", \"size\": $size }"
}

make_tar /tmp/weights.list /tmp/weights.tar || true
make_tar /tmp/config.list /tmp/config.tar || true
make_tar /tmp/docs.list /tmp/docs.tar || true

add_layer /tmp/weights.tar %s
add_layer /tmp/config.tar %s
add_layer /tmp/docs.tar %s

printf '{}' > /tmp/manifest-config.json
mc_dgst=$(sha256sum /tmp/manifest-config.json | cut -d' ' -f1)
mc_size=$(stat -c%%s /tmp/manifest-config.json)
cp /tmp/manifest-config.json /layout/blobs/sha256/$mc_dgst

manifest="{ \"schemaVersion\": 2, \"mediaType\": \"application/vnd.oci.image.manifest.v1+json\", \"artifactType\": \"%s\", \"config\": {\"mediaType\": \"%s\", \"digest\": \"sha256:$mc_dgst\", \"size\": $mc_size}, \"layers\": [ $layers_json ] }"
printf %%s "$manifest" > /tmp/manifest.json
m_dgst=$(sha256sum /tmp/manifest.json | cut -d' ' -f1)
m_size=$(stat -c%%s /tmp/manifest.json)
cp /tmp/manifest.json /layout/blobs/sha256/$m_dgst

cat > /layout/index.json <<IDX
{ "schemaVersion": 2, "mediaType": "application/vnd.oci.image.index.v1+json", "manifests": [ { "mediaType": "application/vnd.oci.image.manifest.v1+json", "digest": "sha256:$m_dgst", "size": $m_size, "annotations": { "org.opencontainers.image.title": "%s" } } ] }
IDX
printf '{ "imageLayoutVersion": "1.0.0" }' > /layout/oci-layout
`, mtWeights, mtConfig, mtDocs, artifactType, mtManifest, name)

		run := llb.Image("golang:1.23").
			Run(llb.Args([]string{"bash", "-c", script}),
				llb.AddMount("/src", modelState, llb.Readonly),
			)
		// Copy contents of /layout into root (no nested layout directory)
		final := llb.Scratch().File(llb.Copy(run.Root(), "/layout/", "/"))
		def, err := final.Marshal(ctx, llb.WithCustomName("packager:modelpack-llb"))
		if err != nil { return nil, err }
		resSolve, err := c.Solve(ctx, client.SolveRequest{Definition: def.ToPB()})
		if err != nil { return nil, err }
		ref, err := resSolve.SingleRef(); if err != nil { return nil, err }
		cfg := ocispec.Image{}; cfg.OS = "linux"; cfg.Architecture = "amd64"; cfg.RootFS = ocispec.RootFS{Type: "layers", DiffIDs: []digest.Digest{}}
		bCfg, _ := json.Marshal(cfg)
		out := client.NewResult(); out.AddMeta(exptypes.ExporterImageConfigKey, bCfg); out.SetRef(ref); out.AddMeta("aikit.emit", []byte("modelpack-llb"))
		return out, nil
	}

	// Generic OCI artifact mode (single tar layer) using LLB shell without overrides
	if emit == "generic-llb" {
		if source == "" { return nil, fmt.Errorf("source is required for emit=generic-llb") }
		name := getBuildArg(opts, "name")
		artifactType := "application/vnd.oci.artifact" // simple generic artifact type

		var srcState llb.State
		switch {
		case source == "." || source == "context":
			srcState = llb.Local(localNameContext, llb.SessionID(sessionID), llb.SharedKeyHint(localNameContext))
		case strings.HasPrefix(source, "https://") || strings.HasPrefix(source, "http://"):
			srcState = llb.HTTP(source)
		case strings.HasPrefix(source, "huggingface://"):
			hfURL, modelFile, err := inference.ParseHuggingFaceURL(source)
			if err != nil { return nil, fmt.Errorf("invalid huggingface source: %w", err) }
			srcState = llb.HTTP(hfURL, llb.Filename(modelFile))
		default:
			include := source
			if strings.HasSuffix(include, "/") { include += "**" }
			srcState = llb.Local(localNameContext,
				llb.IncludePatterns([]string{include}),
				llb.SessionID(sessionID),
				llb.SharedKeyHint(localNameContext+":"+include),
			)
		}

		genericScript := fmt.Sprintf(`set -euo pipefail
mkdir -p /layout/blobs/sha256
work=/src
if [ -f /src ]; then mkdir -p /worksrc && cp /src /worksrc/; work=/worksrc; fi
cd "$work"
find . -type f -print | sed 's|^./||' | LC_ALL=C sort > /tmp/files.list
tar --sort=name --mtime='@0' --owner=0 --group=0 --numeric-owner -cf /tmp/layer.tar -T /tmp/files.list || true
if [ -f /tmp/layer.tar ]; then
  dgst=$(sha256sum /tmp/layer.tar | awk '{print $1}')
  size=$(stat -c%%s /tmp/layer.tar)
  mv /tmp/layer.tar /layout/blobs/sha256/$dgst
  layerDesc="{ \"mediaType\": \"%s\", \"digest\": \"sha256:$dgst\", \"size\": $size }"
else
  layerDesc=""
fi
printf '{}' > /tmp/config.json
cfg_dgst=$(sha256sum /tmp/config.json | awk '{print $1}')
cfg_size=$(stat -c%%s /tmp/config.json)
cp /tmp/config.json /layout/blobs/sha256/$cfg_dgst
manifest=$(cat <<JSON
{ "schemaVersion": 2, "mediaType": "application/vnd.oci.image.manifest.v1+json", "artifactType": "%s", "config": {"mediaType": "application/vnd.oci.image.config.v1+json", "digest": "sha256:$cfg_dgst", "size": $cfg_size}, "layers": [ $layerDesc ] }
JSON
)
printf '%%s' "$manifest" > /tmp/manifest.json
m_dgst=$(sha256sum /tmp/manifest.json | awk '{print $1}')
m_size=$(stat -c%%s /tmp/manifest.json)
cp /tmp/manifest.json /layout/blobs/sha256/$m_dgst
cat > /layout/index.json <<EOF
{ "schemaVersion": 2, "mediaType": "application/vnd.oci.image.index.v1+json", "manifests": [ { "mediaType": "application/vnd.oci.image.manifest.v1+json", "digest": "sha256:$m_dgst", "size": $m_size, "annotations": { "org.opencontainers.image.title": "%s" } } ] }
EOF
cat > /layout/oci-layout <<EOF
{ "imageLayoutVersion": "1.0.0" }
EOF
`, ocispec.MediaTypeImageLayer, artifactType, name)

		run := llb.Image("golang:1.23").
			Run(llb.Shlex("sh -c '"+genericScript+"'"),
				llb.AddMount("/src", srcState, llb.Readonly),
			)
		final := llb.Scratch().File(llb.Copy(run.Root(), "/layout/", "/"))
		def, err := final.Marshal(ctx, llb.WithCustomName("packager:generic-llb"))
		if err != nil { return nil, err }
		resSolve, err := c.Solve(ctx, client.SolveRequest{Definition: def.ToPB()})
		if err != nil { return nil, err }
		ref, err := resSolve.SingleRef(); if err != nil { return nil, err }
		cfg := ocispec.Image{}; cfg.OS = "linux"; cfg.Architecture = "amd64"; cfg.RootFS = ocispec.RootFS{Type: "layers", DiffIDs: []digest.Digest{}}
		bCfg, _ := json.Marshal(cfg)
		out := client.NewResult(); out.AddMeta(exptypes.ExporterImageConfigKey, bCfg); out.SetRef(ref); out.AddMeta("aikit.emit", []byte("generic-llb"))
		return out, nil
	}

	// Special emit path: modelpack layout embedded directly into rootfs (MVP)
	if emit == "modelpack" || emit == "modelpack-layout" {
		if source == "" {
			return nil, fmt.Errorf("source is required for emit=modelpack")
		}
		// Only default modelpack behavior (no overrides)
		specStr := string(packager.SpecModelPack)
		name := getBuildArg(opts, "name")
		artifactType := "" // let packager default

		// Run packager (writes an OCI layout to temp dir)
		layoutDir, err := os.MkdirTemp("", "aikit-front-pack-*")
		if err != nil { return nil, err }
		resPack, err := packager.Pack(ctx, packager.Options{Source: source, OutputDir: layoutDir, Spec: packager.SpecType(specStr), Name: name, ArtifactType: artifactType})
		if err != nil { return nil, fmt.Errorf("packager: %w", err) }

		// Embed layout files into rootfs via Mkfile (size guard to avoid huge inline blobs)
		const maxInline = 25 * 1024 * 1024 // 25MB per file safeguard
		st := llb.Scratch()
		err = filepath.WalkDir(resPack.LayoutPath, func(p string, d fs.DirEntry, err error) error {
			if err != nil { return err }
			if d.IsDir() { return nil }
			rel, _ := filepath.Rel(resPack.LayoutPath, p)
			info, err := d.Info(); if err != nil { return err }
			if info.Size() > maxInline {
				return fmt.Errorf("file %s larger than inline limit (%d bytes > %d); use CLI pack instead", rel, info.Size(), maxInline)
			}
			b, err := os.ReadFile(p)
			if err != nil { return err }
			st = st.File(llb.Mkfile(filepath.ToSlash(rel), 0o644, b))
			return nil
		})
		if err != nil { return nil, err }

		// Build result definition
		def, err := st.Marshal(ctx, llb.WithCustomName("packager:modelpack"))
		if err != nil { return nil, err }
		res, err := c.Solve(ctx, client.SolveRequest{Definition: def.ToPB()})
		if err != nil { return nil, err }
		ref, err := res.SingleRef(); if err != nil { return nil, err }

		// Minimal image config
		cfg := ocispec.Image{}
		cfg.OS = "linux"
		cfg.Architecture = "amd64"
		cfg.RootFS = ocispec.RootFS{Type: "layers", DiffIDs: []digest.Digest{}}
		bCfg, _ := json.Marshal(cfg)
		out := client.NewResult()
		out.AddMeta(exptypes.ExporterImageConfigKey, bCfg)
		out.SetRef(ref)
		// Annotation to signal this rootfs is an OCI layout, not a runtime filesystem
		out.AddMeta("aikit.emit", []byte("modelpack-layout"))
		return out, nil
	}

	var st llb.State
	switch {
	case source == "" || source == "." || source == "context":
		// Use entire local context
		s := llb.Local(localNameContext,
			llb.SessionID(sessionID),
			llb.SharedKeyHint(localNameContext),
		)
		st = llb.Scratch().File(llb.Copy(s, "/", "/"))

	case strings.HasPrefix(source, "https://") || strings.HasPrefix(source, "http://"):
		http := llb.HTTP(source)
		// Copy the downloaded file(s) into the image root
		st = llb.Scratch().File(llb.Copy(http, "/", "/"))

	case strings.HasPrefix(source, "huggingface://"):
		// Reuse shared parser to support optional branch: huggingface://{namespace}/{repo}/{[branch/]file}
		hfURL, modelFile, err := inference.ParseHuggingFaceURL(source)
		if err != nil {
			return nil, fmt.Errorf("invalid huggingface source: %w", err)
		}
		// Download with a fixed filename and copy into image root preserving the name.
		m := llb.HTTP(hfURL, llb.Filename(modelFile))
		st = llb.Scratch().File(llb.Copy(m, modelFile, "/"+filepath.ToSlash(path.Base(modelFile))))

	default:
		// Treat as a path inside local build context (relative glob)
		include := source
		if strings.HasSuffix(include, "/") {
			include += "**"
		}
		s := llb.Local(localNameContext,
			llb.IncludePatterns([]string{include}),
			llb.SessionID(sessionID),
			llb.SharedKeyHint(localNameContext+":"+include),
		)
		st = llb.Scratch().File(llb.Copy(s, "/", "/"))
	}

	// Choose platform: prefer requested, otherwise worker default
	// Default to first worker platform or the host default
	targetPlatform := platforms.DefaultSpec()
	if workers := c.BuildOpts().Workers; len(workers) > 0 && len(workers[0].Platforms) > 0 {
		targetPlatform = workers[0].Platforms[0]
	}
	if pStr, ok := opts["platform"]; ok && pStr != "" {
		// pick first requested platform if multiple
		if parts := strings.Split(pStr, ","); len(parts) > 0 {
			if tp, err := platforms.Parse(parts[0]); err == nil {
				targetPlatform = tp
			}
		}
	}

	// Marshal definition (platform selection is conveyed via export metadata below)
	def, err := st.Marshal(ctx, llb.WithCustomName("packager"))
	if err != nil {
		return nil, err
	}

	res, err := c.Solve(ctx, client.SolveRequest{Definition: def.ToPB()})
	if err != nil {
		return nil, err
	}
	// Attach a minimal valid image config (with required OS/Architecture & rootfs)
	ref, err := res.SingleRef()
	if err != nil {
		return nil, err
	}
	cfg := ocispec.Image{}
	// Populate mandatory fields
	cfg.OS = targetPlatform.OS
	cfg.Architecture = targetPlatform.Architecture
	cfg.RootFS = ocispec.RootFS{Type: "layers", DiffIDs: []digest.Digest{}}
	b, err := json.Marshal(cfg)
	if err != nil {
		return nil, err
	}
	out := client.NewResult()
	out.AddMeta(exptypes.ExporterImageConfigKey, b)
	out.SetRef(ref)
	return out, nil
}

func getBuildArg(opts map[string]string, k string) string {
	if opts != nil {
		if v, ok := opts["build-arg:"+k]; ok {
			return v
		}
	}
	return ""
}
