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
	"github.com/moby/buildkit/client/llb"
	"github.com/moby/buildkit/exporter/containerimage/exptypes"
	"github.com/moby/buildkit/frontend/gateway/client"
	v1 "github.com/modelpack/model-spec/specs-go/v1"
	digest "github.com/opencontainers/go-digest"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
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
	format := getBuildArg(opts, "format")
	hfToken := getBuildArg(opts, "hf-token") // optional Hugging Face token

	// Scalable LLB-based modelpack packaging (shell implementation; no extra binary build)
	if format == "modelpack" {
		if source == "" { return nil, fmt.Errorf("source is required for format=modelpack") }
		packMode := getBuildArg(opts, "layer_packaging") // raw|tar|tar+gzip|tar+zstd
		if packMode == "" { packMode = "tar" }
		name := getBuildArg(opts, "name")
		artifactType := v1.ArtifactTypeModelManifest
		mtManifest := v1.MediaTypeModelConfig

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
			spec, err := inference.ParseHuggingFaceSpec(source)
			if err != nil { return nil, fmt.Errorf("invalid huggingface source: %w", err) }
			// Use huggingface-cli to download entire repo snapshot deterministically
			var tokenExport string
			if hfToken != "" { tokenExport = "export HUGGING_FACE_HUB_TOKEN=\"" + hfToken + "\"\n" }
			dlScript := fmt.Sprintf(`set -euo pipefail
			%s
			mkdir -p /out
			huggingface-cli download %s/%s --revision %s --local-dir /out --local-dir-use-symlinks False
			# remove transient cache / lock artifacts
			rm -rf /out/.cache || true
			find /out -type f -name '*.lock' -delete || true
			`, tokenExport, spec.Namespace, spec.Model, spec.Revision)
			run := llb.Image("docker.io/sozercan/hf-cli:latest").Run(llb.Args([]string{"bash", "-c", dlScript}))
			modelState = llb.Scratch().File(llb.Copy(run.Root(), "/out/", "/"))
		default:
			include := source
			if strings.HasSuffix(include, "/") { include += "**" }
			modelState = llb.Local(localNameContext,
				llb.IncludePatterns([]string{include}),
				llb.SessionID(sessionID),
				llb.SharedKeyHint(localNameContext+":"+include),
			)
		}

		// Shell script (bash) rewritten with numbered fmt verbs & escaped %% to prevent Go formatting of shell printf/stat.
		scriptTemplate := `set -euo pipefail
PACK_MODE=%[1]s
mkdir -p /layout/blobs/sha256
src=/src
if [ -f /src ]; then mkdir -p /worksrc && cp /src /worksrc/; src=/worksrc; fi
cd "$src"
> /tmp/weights.list
> /tmp/config.list
> /tmp/docs.list
> /tmp/code.list
> /tmp/dataset.list

while IFS= read -r f; do
	base=$(basename "$f" | tr A-Z a-z)
	case "$base" in
		*.safetensors|*.bin|*.gguf|*.pt|*.ckpt) echo "$f" >> /tmp/weights.list ;;
		readme*|license*|license|*.md) echo "$f" >> /tmp/docs.list ;;
		config.json|tokenizer.json|*tokenizer*.json|generation_config.json|*.json|*.txt) echo "$f" >> /tmp/config.list ;;
		*.py|*.sh|*.ipynb|*.go|*.js|*.ts) echo "$f" >> /tmp/code.list ;;
		*.csv|*.tsv|*.jsonl|*.parquet|*.arrow|*.h5|*.npz) echo "$f" >> /tmp/dataset.list ;;
		*) sz=$(stat -c%%s "$f"); if [ "$sz" -gt 10485760 ]; then echo "$f" >> /tmp/weights.list; else echo "$f" >> /tmp/config.list; fi ;;
	esac
done < <(find . -type f ! -name '*.lock' ! -path './.cache/*' -print | sed 's|^./||' | LC_ALL=C sort)

layers_json=""
append_layer() { file="$1"; mt="$2"; fpath="$3"; metaJson="$4"; untested="$5"; [ ! -f "$file" ] && return 0; dgst=$(sha256sum "$file" | cut -d' ' -f1); size=$(stat -c%%s "$file"); mv "$file" /layout/blobs/sha256/$dgst; [ -n "$layers_json" ] && layers_json="$layers_json , "; metaEsc=$(printf '%%s' "$metaJson" | sed 's/"/\\\"/g'); ann="{ \"org.cncf.model.filepath\": \"$fpath\", \"org.cncf.model.file.metadata+json\": \"$metaEsc\", \"org.cncf.model.file.mediatype.untested\": \"$untested\" }"; layers_json="${layers_json}{ \"mediaType\": \"$mt\", \"digest\": \"sha256:$dgst\", \"size\": $size, \"annotations\": $ann }"; }

det_tar() { list="$1"; out="$2"; [ ! -s "$list" ] && return 1; tar --sort=name --mtime='@0' --owner=0 --group=0 --numeric-owner -cf "$out" -T "$list"; }

add_category() {
	list="$1"; cat="$2"; mtRaw="$3"; mtTar="$4"; mtTarGz="$5"; mtTarZst="$6"; [ ! -s "$list" ] && return 0
	case "$PACK_MODE" in
		raw)
			while IFS= read -r f; do fsize=$(stat -c%%s "$f"); meta=$(printf '{"name":"%%s","mode":420,"uid":0,"gid":0,"size":%%s,"mtime":"1970-01-01T00:00:00Z","typeflag":0}' "$f" "$fsize"); tmpCp=/tmp/raw-$(basename "$f"); cp "$f" "$tmpCp"; append_layer "$tmpCp" "$mtRaw" "$f" "$meta" "true"; done < "$list" ;;
		tar|tar+gzip|tar+zstd)
			if [ "$cat" = "weights" ]; then
				while IFS= read -r f; do b=$(basename "$f"); tmpTar=/tmp/${cat}-$b.tar; tar --sort=name --mtime='@0' --owner=0 --group=0 --numeric-owner -cf "$tmpTar" -C "$(dirname "$f")" "$b"; case "$PACK_MODE" in tar) mt=$mtTar ;; tar+gzip) gzip -n "$tmpTar"; tmpTar="$tmpTar.gz"; mt=$mtTarGz ;; tar+zstd) zstd -q --no-progress "$tmpTar"; tmpTar="$tmpTar.zst"; mt=$mtTarZst ;; esac; fsize=$(stat -c%%s "$f"); meta=$(printf '{"name":"%%s","mode":420,"uid":0,"gid":0,"size":%%s,"mtime":"1970-01-01T00:00:00Z","typeflag":0}' "$f" "$fsize"); append_layer "$tmpTar" "$mt" "$f" "$meta" "true"; done < "$list"
			else
				tmpTar=/tmp/${cat}.tar; det_tar "$list" "$tmpTar" || return 0; case "$PACK_MODE" in tar) outFile="$tmpTar"; mt=$mtTar ;; tar+gzip) gzip -n "$tmpTar"; outFile="$tmpTar.gz"; mt=$mtTarGz ;; tar+zstd) zstd -q --no-progress "$tmpTar"; outFile="$tmpTar.zst"; mt=$mtTarZst ;; esac; count=$(wc -l < "$list" | tr -d ' '); totalSize=0; while IFS= read -r f2; do sz=$(stat -c%%s "$f2"); totalSize=$((totalSize + sz)); done < "$list"; meta=$(printf '{"name":"%%s","mode":420,"uid":0,"gid":0,"size":%%s,"mtime":"1970-01-01T00:00:00Z","typeflag":0,"files":%%d}' "$cat" "$totalSize" "$count"); append_layer "$outFile" "$mt" "$cat" "$meta" "true"
			fi ;;
		*) echo "unknown PACK_MODE $PACK_MODE" >&2; exit 1 ;;
	esac
}

add_category /tmp/weights.list weights \
	application/vnd.cncf.model.weight.v1.raw \
	application/vnd.cncf.model.weight.v1.tar \
	application/vnd.cncf.model.weight.v1.tar+gzip \
	application/vnd.cncf.model.weight.v1.tar+zstd
add_category /tmp/config.list config \
	application/vnd.cncf.model.weight.config.v1.raw \
	application/vnd.cncf.model.weight.config.v1.tar \
	application/vnd.cncf.model.weight.config.v1.tar+gzip \
	application/vnd.cncf.model.weight.config.v1.tar+zstd
add_category /tmp/docs.list docs \
	application/vnd.cncf.model.doc.v1.raw \
	application/vnd.cncf.model.doc.v1.tar \
	application/vnd.cncf.model.doc.v1.tar+gzip \
	application/vnd.cncf.model.doc.v1.tar+zstd
add_category /tmp/code.list code \
	application/vnd.cncf.model.code.v1.raw \
	application/vnd.cncf.model.code.v1.tar \
	application/vnd.cncf.model.code.v1.tar+gzip \
	application/vnd.cncf.model.code.v1.tar+zstd
add_category /tmp/dataset.list dataset \
	application/vnd.cncf.model.dataset.v1.raw \
	application/vnd.cncf.model.dataset.v1.tar \
	application/vnd.cncf.model.dataset.v1.tar+gzip \
	application/vnd.cncf.model.dataset.v1.tar+zstd

printf '{}' > /tmp/manifest-config.json
mc_dgst=$(sha256sum /tmp/manifest-config.json | cut -d' ' -f1)
mc_size=$(stat -c%%s /tmp/manifest-config.json)
cp /tmp/manifest-config.json /layout/blobs/sha256/$mc_dgst

cat > /tmp/manifest.json <<EOF_MANIFEST
{ "schemaVersion": 2, "mediaType": "application/vnd.oci.image.manifest.v1+json", "artifactType": "%[2]s", "config": {"mediaType": "%[3]s", "digest": "sha256:$mc_dgst", "size": $mc_size}, "layers": [ $layers_json ] }
EOF_MANIFEST
# Basic validation to catch accidental plain-text or unexpanded vars
if [ "$(head -c1 /tmp/manifest.json)" != "{" ] || \
	 ! grep -q '"schemaVersion": 2' /tmp/manifest.json || \
	 ! grep -q '"mediaType": "application/vnd.oci.image.manifest.v1+json"' /tmp/manifest.json; then
	echo "manifest validation failed" >&2; echo "---- manifest contents ----" >&2; cat /tmp/manifest.json >&2; exit 1
fi
m_dgst=$(sha256sum /tmp/manifest.json | cut -d' ' -f1)
m_size=$(stat -c%%s /tmp/manifest.json)
cp /tmp/manifest.json /layout/blobs/sha256/$m_dgst

cat > /layout/index.json <<IDX
{ "schemaVersion": 2, "mediaType": "application/vnd.oci.image.index.v1+json", "manifests": [ { "mediaType": "application/vnd.oci.image.manifest.v1+json", "digest": "sha256:$m_dgst", "size": $m_size, "annotations": { "org.opencontainers.image.title": "%[4]s", "org.opencontainers.image.ref.name": "latest" } } ] }
IDX
printf '{ "imageLayoutVersion": "1.0.0" }' > /layout/oci-layout
`
		script := fmt.Sprintf(scriptTemplate, packMode, artifactType, mtManifest, name)

		run := llb.Image("cgr.dev/chainguard/bash:latest").
			Run(llb.Args([]string{"bash", "-c", script}),
				llb.AddMount("/src", modelState, llb.Readonly),
			)
		// Copy contents of /layout into root (no nested layout directory)
		final := llb.Scratch().File(llb.Copy(run.Root(), "/layout/", "/"))
		def, err := final.Marshal(ctx, llb.WithCustomName("packager:modelpack"))
		if err != nil { return nil, err }
		resSolve, err := c.Solve(ctx, client.SolveRequest{Definition: def.ToPB()})
		if err != nil { return nil, err }
		ref, err := resSolve.SingleRef(); if err != nil { return nil, err }
		cfg := ocispec.Image{}; cfg.OS = "linux"; cfg.Architecture = "amd64"; cfg.RootFS = ocispec.RootFS{Type: "layers", DiffIDs: []digest.Digest{}}
		bCfg, _ := json.Marshal(cfg)
		out := client.NewResult(); out.AddMeta(exptypes.ExporterImageConfigKey, bCfg); out.SetRef(ref); out.AddMeta("aikit.format", []byte("modelpack"))
		return out, nil
	}

	// Generic OCI artifact mode (single tar layer) using LLB shell without overrides
	if format == "generic" {
		if source == "" { return nil, fmt.Errorf("source is required for format=generic") }
		name := getBuildArg(opts, "name")
		artifactType := "application/vnd.oci.artifact" // simple generic artifact type

		var srcState llb.State
		switch {
		case source == "." || source == "context":
			srcState = llb.Local(localNameContext, llb.SessionID(sessionID), llb.SharedKeyHint(localNameContext))
		case strings.HasPrefix(source, "https://") || strings.HasPrefix(source, "http://"):
			srcState = llb.HTTP(source)
		case strings.HasPrefix(source, "huggingface://"):
			spec, err := inference.ParseHuggingFaceSpec(source)
			if err != nil { return nil, fmt.Errorf("invalid huggingface source: %w", err) }
			var tokenExport string
			if hfToken != "" { tokenExport = "export HUGGING_FACE_HUB_TOKEN=\"" + hfToken + "\"\n" }
			dlScript := fmt.Sprintf(`set -euo pipefail
			%s
			mkdir -p /out
			huggingface-cli download %s/%s --revision %s --local-dir /out --local-dir-use-symlinks False
			# remove transient cache / lock artifacts
			rm -rf /out/.cache || true
			find /out -type f -name '*.lock' -delete || true
			`, tokenExport, spec.Namespace, spec.Model, spec.Revision)
			run := llb.Image("docker.io/sozercan/hf-cli:latest").Run(llb.Args([]string{"bash", "-c", dlScript}))
			srcState = llb.Scratch().File(llb.Copy(run.Root(), "/out/", "/"))
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
find . -type f ! -name '*.lock' ! -path './.cache/*' -print | sed 's|^./||' | LC_ALL=C sort > /tmp/files.list
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
{ "schemaVersion": 2, "mediaType": "application/vnd.oci.image.index.v1+json", "manifests": [ { "mediaType": "application/vnd.oci.image.manifest.v1+json", "digest": "sha256:$m_dgst", "size": $m_size, "annotations": { "org.opencontainers.image.title": "%s", "org.opencontainers.image.ref.name": "latest" } } ] }
EOF
cat > /layout/oci-layout <<EOF
{ "imageLayoutVersion": "1.0.0" }
EOF
`, ocispec.MediaTypeImageLayer, artifactType, name)

		run := llb.Image("cgr.dev/chainguard/bash:latest").
			Run(llb.Shlex("sh -c '"+genericScript+"'"),
				llb.AddMount("/src", srcState, llb.Readonly),
			)
		final := llb.Scratch().File(llb.Copy(run.Root(), "/layout/", "/"))
		def, err := final.Marshal(ctx, llb.WithCustomName("packager:generic"))
		if err != nil { return nil, err }
		resSolve, err := c.Solve(ctx, client.SolveRequest{Definition: def.ToPB()})
		if err != nil { return nil, err }
		ref, err := resSolve.SingleRef(); if err != nil { return nil, err }
		cfg := ocispec.Image{}; cfg.OS = "linux"; cfg.Architecture = "amd64"; cfg.RootFS = ocispec.RootFS{Type: "layers", DiffIDs: []digest.Digest{}}
		bCfg, _ := json.Marshal(cfg)
		out := client.NewResult(); out.AddMeta(exptypes.ExporterImageConfigKey, bCfg); out.SetRef(ref); out.AddMeta("aikit.format", []byte("generic"))
		return out, nil
	}

	// Special format path: modelpack layout embedded directly into rootfs (MVP)
	if format == "modelpack" {
		if source == "" {
			return nil, fmt.Errorf("source is required for format=modelpack")
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
		out.AddMeta("aikit.format", []byte("modelpack-layout"))
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
		spec, err := inference.ParseHuggingFaceSpec(source)
		if err != nil { return nil, fmt.Errorf("invalid huggingface source: %w", err) }
		var tokenExport string
		if hfToken != "" { tokenExport = "export HUGGING_FACE_HUB_TOKEN=\"" + hfToken + "\"\n" }
		dlScript := fmt.Sprintf(`set -euo pipefail
	%s
	mkdir -p /out
	huggingface-cli download %s/%s --revision %s --local-dir /out --local-dir-use-symlinks False
	# remove transient cache / lock artifacts
	rm -rf /out/.cache || true
	find /out -type f -name '*.lock' -delete || true
	`, tokenExport, spec.Namespace, spec.Model, spec.Revision)
		run := llb.Image("docker.io/sozercan/hf-cli:latest").Run(llb.Args([]string{"bash", "-c", dlScript}))
		st = llb.Scratch().File(llb.Copy(run.Root(), "/out/", "/"))

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
