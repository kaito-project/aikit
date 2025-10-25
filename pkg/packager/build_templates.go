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
# Also cache file sizes in parallel to avoid repeated stat calls
find . -type f ! -name '*.lock' ! -path './.cache/*' -print0 | \
	xargs -0 -P $(nproc) -I {} sh -c 'echo "{}|$(stat -c%%s "{}")"' | \
	LC_ALL=C sort > /tmp/allfiles_with_size.list

# Categorize files by extension and size into appropriate lists
# File size is already computed and cached
while IFS='|' read -r f sz; do
	f=${f#./}
	base=$(basename "$f" | tr A-Z a-z)
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
	# Cache size for later use
	echo "$f|$sz" >> /tmp/file_sizes.cache
done < /tmp/allfiles_with_size.list

# Initialize JSON array for manifest layers
layers_json=""

# get_cached_size: Retrieve cached file size to avoid repeated stat calls
get_cached_size() {
	local file="$1"
	grep -F "$file|" /tmp/file_sizes.cache 2>/dev/null | cut -d'|' -f2 | head -n1
}

# append_layer: Add a file as a layer blob with annotations
# Args: file path, media type, filepath annotation, metadata JSON, untested flag
append_layer() {
	file="$1"; mt="$2"; fpath="$3"; metaJson="$4"; untested="$5"
	[ ! -f "$file" ] && return 0
	dgst=$(sha256sum "$file" | cut -d' ' -f1)
	size=$(stat -c%%s "$file")
	mv "$file" /layout/blobs/sha256/$dgst
	[ -n "$layers_json" ] && layers_json="$layers_json , "
	metaEsc=$(printf '%%s' "$metaJson" | sed 's/"/\\"/g')
	ann="{ \"org.cncf.model.filepath\": \"$fpath\", \"org.cncf.model.file.metadata+json\": \"$metaEsc\", \"org.cncf.model.file.mediatype.untested\": \"$untested\" }"
	layers_json="${layers_json}{ \"mediaType\": \"$mt\", \"digest\": \"sha256:$dgst\", \"size\": $size, \"annotations\": $ann }"
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
				fsize=$(get_cached_size "$f")
				[ -z "$fsize" ] && fsize=$(stat -c%%s "$f")  # Fallback to stat if cache miss
				meta=$(printf '{"name":"%%s","mode":420,"uid":0,"gid":0,"size":%%s,"mtime":"1970-01-01T00:00:00Z","typeflag":0}' "$f" "$fsize")
				tmpCp=/tmp/raw-$(basename "$f")
				cp "$f" "$tmpCp"
				append_layer "$tmpCp" "$mtRaw" "$f" "$meta" "true"
			done < "$list" ;;
		tar|tar+gzip|tar+zstd)
			if [ "$cat" = "weights" ]; then
				# Weights: tar each file individually (can be large)
				while IFS= read -r f; do
					b=$(basename "$f")
					tmpTar=/tmp/${cat}-$b.tar
					tar -cf "$tmpTar" -C "$(dirname "$f")" "$b"
					case "$PACK_MODE" in
						tar) mt=$mtTar ;;
						tar+gzip) gzip -n "$tmpTar"; tmpTar="$tmpTar.gz"; mt=$mtTarGz ;;
						tar+zstd) zstd -q --no-progress "$tmpTar"; tmpTar="$tmpTar.zst"; mt=$mtTarZst ;;
					esac
					fsize=$(get_cached_size "$f")
					[ -z "$fsize" ] && fsize=$(stat -c%%s "$f")
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
				count=$(wc -l < "$list" | tr -d ' ')
				totalSize=0
				while IFS= read -r f2; do
					sz=$(get_cached_size "$f2")
					[ -z "$sz" ] && sz=$(stat -c%%s "$f2")
					totalSize=$((totalSize + sz))
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
mc_dgst=$(sha256sum /tmp/manifest-config.json | cut -d' ' -f1)
mc_size=$(stat -c%%s /tmp/manifest-config.json)
cp /tmp/manifest-config.json /layout/blobs/sha256/$mc_dgst

# Generate OCI manifest with all layers
cat > /tmp/manifest.json <<EOF_MANIFEST
{ "schemaVersion": 2, "mediaType": "application/vnd.oci.image.manifest.v1+json", "artifactType": "%[2]s", "config": {"mediaType": "%[3]s", "digest": "sha256:$mc_dgst", "size": $mc_size}, "layers": [ $layers_json ] }
EOF_MANIFEST

# Validate manifest structure
if [ "$(head -c1 /tmp/manifest.json)" != "{" ] || \
	 ! grep -q '"schemaVersion": 2' /tmp/manifest.json || \
	 ! grep -q '"mediaType": "application/vnd.oci.image.manifest.v1+json"' /tmp/manifest.json; then
	echo "manifest validation failed" >&2; cat /tmp/manifest.json >&2; exit 1
fi

# Add manifest as blob
m_dgst=$(sha256sum /tmp/manifest.json | cut -d' ' -f1)
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
# Cache file sizes for later use
find . -type f ! -name '*.lock' ! -path './.cache/*' -print0 | \
	xargs -0 -P $(nproc) -I {} sh -c 'f="{}"; echo "$f|$(stat -c%%s "$f")"' | \
	sed 's|^\./||' | LC_ALL=C sort > /tmp/files_with_size.list

# Extract just the file paths for processing
cut -d'|' -f1 < /tmp/files_with_size.list > /tmp/files.list

# Initialize JSON array for manifest layers
layers_json=""

# get_file_size: Retrieve cached file size
get_file_size() {
	grep -F "$1|" /tmp/files_with_size.list 2>/dev/null | cut -d'|' -f2 | head -n1
}

# append_layer: Add a file as a layer blob
# Args: file path, media type
append_layer() {
	file="$1"; mt="$2"
	[ ! -f "$file" ] && return 0
	dgst=$(sha256sum "$file" | cut -d' ' -f1)
	size=$(stat -c%%s "$file")
	mv "$file" /layout/blobs/sha256/$dgst
	[ -n "$layers_json" ] && layers_json="$layers_json , "
	layers_json="${layers_json}{ \"mediaType\": \"$mt\", \"digest\": \"sha256:$dgst\", \"size\": $size }"
}

# Process files according to pack mode
case "$PACK_MODE" in
	raw)
		# Raw mode: each file becomes its own layer
		while IFS= read -r f; do
			cp "$f" "/tmp/$(basename "$f")"
			append_layer "/tmp/$(basename "$f")" "%s"
		done < /tmp/files.list ;;
	tar|tar+gzip|tar+zstd)
		# Archive mode: bundle all files into single tar
		tarFile=/tmp/allfiles.tar
		tar -cf "$tarFile" -T /tmp/files.list || true
		mt="%s"
		case "$PACK_MODE" in
			tar) outFile="$tarFile" ;;
			tar+gzip) gzip -n "$tarFile"; outFile="$tarFile.gz" ;;
			tar+zstd) zstd -q --no-progress "$tarFile"; outFile="$tarFile.zst" ;;
		esac
		append_layer "$outFile" "$mt" ;;
	*) echo "unknown PACK_MODE $PACK_MODE" >&2; exit 1 ;;
esac

# Create empty config blob
printf '{}' > /tmp/config.json
cfg_dgst=$(sha256sum /tmp/config.json | awk '{print $1}')
cfg_size=$(stat -c%%s /tmp/config.json)
cp /tmp/config.json /layout/blobs/sha256/$cfg_dgst

# Generate OCI manifest
manifest="{ \"schemaVersion\": 2, \"mediaType\": \"application/vnd.oci.image.manifest.v1+json\", \"artifactType\": \"%s\", \"config\": {\"mediaType\": \"application/vnd.oci.empty.v1+json\", \"digest\": \"sha256:$cfg_dgst\", \"size\": $cfg_size}, \"layers\": [ $layers_json ] }"
printf '%%s' "$manifest" > /tmp/manifest.json

# Add manifest as blob
m_dgst=$(sha256sum /tmp/manifest.json | awk '{print $1}')
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
