package packager

import (
	"fmt"

	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
)

// File categorization thresholds and patterns.
const (
	// largeFileThreshold defines the size (10 MiB) above which unknown files are categorized as weights.
	largeFileThreshold = 10485760 // 10 * 1024 * 1024
)

// generateModelpackScript returns the bash script used to assemble a modelpack OCI layout.
//
// This script performs the following operations:
//  1. Categorizes files into weights, config, docs, code, and dataset based on extensions and size
//  2. Packages each category according to packMode (raw, tar, tar+gzip, tar+zstd)
//  3. Computes SHA256 digests and creates OCI layout with proper annotations
//  4. Validates the generated manifest structure
//
// The script runs in a bash container and expects:
//   - Source files mounted at /src (read-only)
//   - Output directory at /layout/ (writable)
//   - Standard unix tools: find, tar, gzip, zstd, sha256sum
//
// Arguments:
//
//	packMode: raw|tar|tar+gzip|tar+zstd - how to package layer content
//	artifactType: model artifact type (e.g. v1.ArtifactTypeModelManifest)
//	mtManifest: manifest config media type (e.g. v1.MediaTypeModelConfig)
//	name: annotation org.opencontainers.image.title
//	refName: annotation org.opencontainers.image.ref.name
func generateModelpackScript(packMode, artifactType, mtManifest, name, refName string) string { //nolint:lll
	tmpl := `set -euo pipefail
PACK_MODE=%[1]s

# Initialize OCI layout directory structure
mkdir -p /layout/blobs/sha256

# Handle single file input (copy to temporary directory)
src=/src
if [ -f /src ]; then mkdir -p /worksrc && cp /src /worksrc/; src=/worksrc; fi
cd "$src"

# Initialize category lists for file classification
> /tmp/weights.list
> /tmp/config.list
> /tmp/docs.list
> /tmp/code.list
> /tmp/dataset.list

# Find all files, excluding lock files and cache, and sort deterministically
# Batch stat calls so file discovery does not launch one shell per file
find . -type f ! -name '*.lock' ! -path './.cache/*' -exec stat -c '%%n|%%s' {} + | \
	LC_ALL=C sort > /tmp/allfiles_with_size.list

# Categorize files by extension and size into appropriate lists
# File size is already computed and cached
declare -A file_sizes=()
while IFS='|' read -r f sz; do
	f=${f#./}
	file_sizes["$f"]=$sz
	base=${f##*/}
	base=${base,,}
	case "$base" in
		# Model weight files
		*.safetensors|*.bin|*.gguf|*.pt|*.ckpt) echo "$f" >> /tmp/weights.list ;;
		# Documentation files
		readme*|license*|license|*.md) echo "$f" >> /tmp/docs.list ;;
		# Configuration and tokenizer files
		config.json|tokenizer.json|*tokenizer*.json|generation_config.json|*.json|*.txt) echo "$f" >> /tmp/config.list ;;
		# Code files
		*.py|*.sh|*.ipynb|*.go|*.js|*.ts) echo "$f" >> /tmp/code.list ;;
		# Dataset files
		*.csv|*.tsv|*.jsonl|*.parquet|*.arrow|*.h5|*.npz) echo "$f" >> /tmp/dataset.list ;;
		# Unknown files: large ones (>10MB) go to weights, small ones to config
		*) if [ "$sz" -gt %[6]d ]; then echo "$f" >> /tmp/weights.list; else echo "$f" >> /tmp/config.list; fi ;;
	esac
done < /tmp/allfiles_with_size.list

# Stream manifest layers to a file to avoid quadratic shell string concatenation
layers_file=/tmp/layers.json
: > "$layers_file"
first_layer=1

# append_layer: Add a file as a layer blob with annotations
# Args: file path, media type, filepath annotation, metadata JSON, untested flag, optional size, optional blob action
append_layer() {
	file="$1"; mt="$2"; fpath="$3"; metaJson="$4"; untested="$5"; size="${6-}"; blob_action="${7:-move}"
	[ ! -f "$file" ] && return 0
	read -r dgst _ < <(sha256sum "$file")
	[ -z "$size" ] && size=$(stat -c%%s "$file")
	if [ "$blob_action" = "copy" ]; then
		cp "$file" "/layout/blobs/sha256/$dgst"
	else
		mv "$file" "/layout/blobs/sha256/$dgst"
	fi
	if [ "$first_layer" -eq 0 ]; then printf ' , ' >> "$layers_file"; fi
	first_layer=0
	metaEsc=${metaJson//\"/\\\"}
	printf '{ "mediaType": "%%s", "digest": "sha256:%%s", "size": %%s, "annotations": { "org.opencontainers.image.title": "%%s", "org.cncf.model.filepath": "%%s", "org.cncf.model.file.metadata+json": "%%s", "org.cncf.model.file.mediatype.untested": "%%s" } }' \
		"$mt" "$dgst" "$size" "$fpath" "$fpath" "$metaEsc" "$untested" >> "$layers_file"
}

# det_tar: Create deterministic tar archive from file list
det_tar() { list="$1"; out="$2"; [ ! -s "$list" ] && return 1; tar -cf "$out" -T "$list"; }

# add_category: Process a file category and add layers according to pack mode
# Args: list file, category name, raw media type, tar media type, tar+gzip media type, tar+zstd media type
add_category() {
	list="$1"; cat="$2"; mtRaw="$3"; mtTar="$4"; mtTarGz="$5"; mtTarZst="$6"
	[ ! -s "$list" ] && return 0
	case "$PACK_MODE" in
		raw)
			# Raw mode: each file becomes its own layer
			while IFS= read -r f; do
				fsize=${file_sizes["$f"]}
				meta=$(printf '{"name":"%%s","mode":420,"uid":0,"gid":0,"size":%%s,"mtime":"1970-01-01T00:00:00Z","typeflag":0}' "$f" "$fsize")
				append_layer "$f" "$mtRaw" "$f" "$meta" "true" "$fsize" copy
			done < "$list" ;;
		tar|tar+gzip|tar+zstd)
			if [ "$cat" = "weights" ]; then
				# Weights: tar each file individually (can be large)
				while IFS= read -r f; do
					b=${f##*/}
					dir=${f%%/*}
					[ "$dir" = "$f" ] && dir=.
					tmpTar=/tmp/${cat}-$b.tar
					tar -cf "$tmpTar" -C "$dir" "$b"
					case "$PACK_MODE" in
						tar) mt=$mtTar ;;
						tar+gzip) gzip -n "$tmpTar"; tmpTar="$tmpTar.gz"; mt=$mtTarGz ;;
						tar+zstd) zstd -q --no-progress "$tmpTar"; tmpTar="$tmpTar.zst"; mt=$mtTarZst ;;
					esac
					fsize=${file_sizes["$f"]}
					meta=$(printf '{"name":"%%s","mode":420,"uid":0,"gid":0,"size":%%s,"mtime":"1970-01-01T00:00:00Z","typeflag":0}' "$f" "$fsize")
					append_layer "$tmpTar" "$mt" "$f" "$meta" "true"
				done < "$list"
			else
				# Non-weights: bundle all category files into single tar
				tmpTar=/tmp/${cat}.tar
				det_tar "$list" "$tmpTar" || return 0
				case "$PACK_MODE" in
					tar) outFile="$tmpTar"; mt=$mtTar ;;
						tar+gzip) gzip -n "$tmpTar"; outFile="$tmpTar.gz"; mt=$mtTarGz ;;
						tar+zstd) zstd -q --no-progress "$tmpTar"; outFile="$tmpTar.zst"; mt=$mtTarZst ;;
					esac
					count=0
					totalSize=0
					while IFS= read -r f2; do
						sz=${file_sizes["$f2"]}
						totalSize=$((totalSize + sz))
						count=$((count + 1))
					done < "$list"
				meta=$(printf '{"name":"%%s","mode":420,"uid":0,"gid":0,"size":%%s,"mtime":"1970-01-01T00:00:00Z","typeflag":0,"files":%%d}' "$cat" "$totalSize" "$count")
				append_layer "$outFile" "$mt" "$cat" "$meta" "true"
			fi ;;
		*) echo "unknown PACK_MODE $PACK_MODE" >&2; exit 1 ;;
	esac
}

# Process each file category with appropriate ModelPack media types
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

# Create empty manifest config and add as blob
printf '{}' > /tmp/manifest-config.json
read -r mc_dgst _ < <(sha256sum /tmp/manifest-config.json)
mc_size=$(stat -c%%s /tmp/manifest-config.json)
cp /tmp/manifest-config.json /layout/blobs/sha256/$mc_dgst

# Generate OCI manifest with all layers
{
	printf '%%s' '{ "schemaVersion": 2, "mediaType": "application/vnd.oci.image.manifest.v1+json", "artifactType": "%[2]s", "config": {"mediaType": "%[3]s", "digest": "sha256:'"$mc_dgst"'", "size": '"$mc_size"'}, "layers": [ '
	cat "$layers_file"
	printf '%%s\n' ' ] }'
} > /tmp/manifest.json

# Validate manifest structure
if [ "$(head -c1 /tmp/manifest.json)" != "{" ] || \
	 ! grep -q '"schemaVersion": 2' /tmp/manifest.json || \
	 ! grep -q '"mediaType": "application/vnd.oci.image.manifest.v1+json"' /tmp/manifest.json; then
	echo "manifest validation failed" >&2; cat /tmp/manifest.json >&2; exit 1
fi

# Add manifest as blob
read -r m_dgst _ < <(sha256sum /tmp/manifest.json)
m_size=$(stat -c%%s /tmp/manifest.json)
cp /tmp/manifest.json /layout/blobs/sha256/$m_dgst

# Create OCI index pointing to manifest
cat > /layout/index.json <<IDX
{ "schemaVersion": 2, "mediaType": "application/vnd.oci.image.index.v1+json", "manifests": [ { "mediaType": "application/vnd.oci.image.manifest.v1+json", "digest": "sha256:$m_dgst", "size": $m_size, "annotations": { "org.opencontainers.image.title": "%[4]s", "org.opencontainers.image.ref.name": "%[5]s" } } ] }
IDX

# Create OCI layout version marker
printf '{ "imageLayoutVersion": "1.0.0" }' > /layout/oci-layout
`
	return fmt.Sprintf(tmpl, packMode, artifactType, mtManifest, name, refName, largeFileThreshold)
}

// generateGenericScript builds the generic artifact OCI layout assembly script.
//
// This script performs simpler packaging than modelpack:
//  1. Finds all files in source
//  2. Packages them according to packMode (raw, tar, tar+gzip, tar+zstd)
//  3. Creates OCI layout with single layer or multiple raw layers
//
// Arguments:
//
//	packMode: raw|tar|tar+gzip|tar+zstd - packaging method
//	artifactType: artifact type for manifest (default: application/vnd.unknown.artifact.v1)
//	name: annotation org.opencontainers.image.title
//	refName: annotation org.opencontainers.image.ref.name
//	debug: if true, enables bash debug mode (set -x)
func generateGenericScript(packMode, artifactType, name, refName string, debug bool) string { //nolint:lll
	debugLine := ""
	if debug {
		debugLine = "set -x"
	}
	rawLayerMT := ocispec.MediaTypeImageLayer
	archiveLayerMT := ocispec.MediaTypeImageLayer
	if packMode == packModeRaw {
		rawLayerMT = "application/octet-stream"
	}
	tmpl := `set -euo pipefail
%s
PACK_MODE=%s

# Initialize OCI layout directory structure
mkdir -p /layout/blobs/sha256

# Handle single file input (copy to temporary directory)
work=/src
if [ -f /src ]; then mkdir -p /worksrc && cp /src /worksrc/; work=/worksrc; fi
cd "$work"

# Find all files, excluding lock files and cache, sorted deterministically
# Batch stat calls so file discovery does not launch one shell per file
find . -type f ! -name '*.lock' ! -path './.cache/*' -exec stat -c '%%n|%%s' {} + | \
	LC_ALL=C sort > /tmp/files_with_size.list

# Cache file sizes by path and extract the sorted file list
declare -A file_sizes=()
> /tmp/files.list
while IFS='|' read -r f sz; do
	f=${f#./}
	file_sizes["$f"]=$sz
	printf '%%s\n' "$f" >> /tmp/files.list
done < /tmp/files_with_size.list

# Stream manifest layers to a file to avoid quadratic shell string concatenation
layers_file=/tmp/layers.json
: > "$layers_file"
first_layer=1

# append_layer: Add a file as a layer blob with annotations
# Args: file path, media type, title (original filename), optional size, optional blob action
append_layer() {
	file="$1"; mt="$2"; title="$3"; size="${4-}"; blob_action="${5:-move}"
	[ ! -f "$file" ] && return 0
	read -r dgst _ < <(sha256sum "$file")
	[ -z "$size" ] && size=$(stat -c%%s "$file")
	if [ "$blob_action" = "copy" ]; then
		cp "$file" "/layout/blobs/sha256/$dgst"
	else
		mv "$file" "/layout/blobs/sha256/$dgst"
	fi
	if [ "$first_layer" -eq 0 ]; then printf ' , ' >> "$layers_file"; fi
	first_layer=0
	printf '{ "mediaType": "%%s", "digest": "sha256:%%s", "size": %%s, "annotations": { "org.opencontainers.image.title": "%%s" } }' \
		"$mt" "$dgst" "$size" "$title" >> "$layers_file"
}

# Process files according to pack mode
case "$PACK_MODE" in
	raw)
		# Raw mode: each file becomes its own layer
		while IFS= read -r f; do
			append_layer "$f" "%s" "$f" "${file_sizes["$f"]}" copy
		done < /tmp/files.list ;;
	tar|tar+gzip|tar+zstd)
		# Archive mode: bundle all files into single tar
		tarFile=/tmp/allfiles.tar
		tar -cf "$tarFile" -T /tmp/files.list || true
		mt="%s"
		layerName="allfiles.tar"
		case "$PACK_MODE" in
			tar) outFile="$tarFile" ;;
			tar+gzip) gzip -n "$tarFile"; outFile="$tarFile.gz"; layerName="allfiles.tar.gz" ;;
			tar+zstd) zstd -q --no-progress "$tarFile"; outFile="$tarFile.zst"; layerName="allfiles.tar.zst" ;;
		esac
		append_layer "$outFile" "$mt" "$layerName" ;;
	*) echo "unknown PACK_MODE $PACK_MODE" >&2; exit 1 ;;
esac

# Create empty config blob
printf '{}' > /tmp/config.json
read -r cfg_dgst _ < <(sha256sum /tmp/config.json)
cfg_size=$(stat -c%%s /tmp/config.json)
cp /tmp/config.json /layout/blobs/sha256/$cfg_dgst

# Generate OCI manifest
{
	printf '%%s' '{ "schemaVersion": 2, "mediaType": "application/vnd.oci.image.manifest.v1+json", "artifactType": "%s", "config": {"mediaType": "application/vnd.oci.empty.v1+json", "digest": "sha256:'"$cfg_dgst"'", "size": '"$cfg_size"'}, "layers": [ '
	cat "$layers_file"
	printf '%%s' ' ] }'
} > /tmp/manifest.json

# Add manifest as blob
read -r m_dgst _ < <(sha256sum /tmp/manifest.json)
m_size=$(stat -c%%s /tmp/manifest.json)
cp /tmp/manifest.json /layout/blobs/sha256/$m_dgst

# Create OCI index pointing to manifest
cat > /layout/index.json <<EOF
{ "schemaVersion": 2, "mediaType": "application/vnd.oci.image.index.v1+json", "manifests": [ { "mediaType": "application/vnd.oci.image.manifest.v1+json", "digest": "sha256:$m_dgst", "size": $m_size, "annotations": { "org.opencontainers.image.title": "%s", "org.opencontainers.image.ref.name": "%s" } } ] }
EOF

# Create OCI layout version marker
cat > /layout/oci-layout <<EOF
{ "imageLayoutVersion": "1.0.0" }
EOF
`
	return fmt.Sprintf(tmpl, debugLine, packMode, rawLayerMT, archiveLayerMT, artifactType, name, refName)
}
