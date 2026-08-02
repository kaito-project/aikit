#!/usr/bin/env bash

set -euo pipefail
# Optional debug tracing, enabled when DEBUG is set to a non-empty value.
[ -n "${DEBUG:-}" ] && set -x

# Assemble a generic artifact OCI layout from files mounted at /src into /layout.
# AIKIT_SOURCE_DIR, AIKIT_LAYOUT_DIR, and AIKIT_TMP_DIR override those locations
# for executable tests; production builds use the defaults.
#
# Parameters are passed via environment variables:
#   PACK_MODE         raw|tar|tar+gzip|tar+zstd - packaging method
#   ARTIFACT_TYPE     manifest artifactType
#   RAW_LAYER_MT      media type for raw-mode layers
#   ARCHIVE_LAYER_MT  media type for archive-mode layers
#   NAME              org.opencontainers.image.title annotation
#   REF_NAME          org.opencontainers.image.ref.name annotation

source_dir=${AIKIT_SOURCE_DIR:-/src}
layout_dir=${AIKIT_LAYOUT_DIR:-/layout}
tmp_root=${AIKIT_TMP_DIR:-${TMPDIR:-/tmp}}
mkdir -p "$layout_dir/blobs/sha256" "$tmp_root"
tmp_dir=$(mktemp -d "$tmp_root/aikit-generic.XXXXXX")
cleanup() { rm -rf -- "$tmp_dir"; }
trap cleanup EXIT

# Handle single file input by copying it to an isolated working directory.
work=$source_dir
if [ -f "$source_dir" ]; then
	work="$tmp_dir/worksrc"
	mkdir -p "$work"
	cp -- "$source_dir" "$work/"
fi
cd "$work"

# Discover paths in deterministic byte order. Paths stay NUL-delimited, while
# stat emits one numeric size per line in the same order. xargs batches stat
# calls without ever parsing a filename as text.
paths_file="$tmp_dir/files.paths"
sizes_file="$tmp_dir/files.sizes"
records_file="$tmp_dir/files.records"
find . -type f ! -name '*.lock' ! -path './.cache/*' -print0 | LC_ALL=C sort -z > "$paths_file"
xargs -0 -r stat -c '%s' -- < "$paths_file" > "$sizes_file"
: > "$records_file"

exec 3<"$paths_file"
exec 4<"$sizes_file"
while IFS= read -r -d '' f <&3; do
	if ! IFS= read -r sz <&4; then
		printf 'missing size for discovered path %q\n' "$f" >&2
		exit 1
	fi
	f=${f#./}
	printf '%s\0%s\0' "$f" "$sz" >> "$records_file"
done
if IFS= read -r extra_size <&4; then
	printf 'unexpected size without a discovered path: %s\n' "$extra_size" >&2
	exit 1
fi
exec 3<&-
exec 4<&-

# Stream manifest layers to a file to avoid quadratic shell string concatenation.
layers_file="$tmp_dir/layers.json"
: > "$layers_file"
first_layer=1

# escape_json escapes a string for safe inclusion as a JSON string value.
# Backslashes must be escaped before double-quotes. Tabs and newlines are also
# escaped so paths containing them do not produce invalid JSON.
escape_json() {
	local LC_ALL=C
	local value=$1 char code
	while [ -n "$value" ]; do
		char=${value:0:1}
		value=${value:1}
		case "$char" in
			'"') printf '%s' "\\\"" ;;
			$'\\') printf '%s' "\\\\" ;;
			$'\b') printf '%s' "\\b" ;;
			$'\f') printf '%s' "\\f" ;;
			$'\n') printf '%s' "\\n" ;;
			$'\r') printf '%s' "\\r" ;;
			$'\t') printf '%s' "\\t" ;;
			*)
				printf -v code '%d' "'$char"
				if [ "$code" -lt 32 ]; then
					printf '\\u%04x' "$code"
				else
					printf '%s' "$char"
				fi
				;;
		esac
	done
}

# append_layer adds a file as a layer blob with annotations.
# Args: file path, media type, title (original filename), optional size, and
# optional blob action.
append_layer() {
	local file=$1 mt=$2 title=$3 size=${4-} blob_action=${5:-move}
	local dgst titleEsc
	[ ! -f "$file" ] && return 0
	read -r dgst _ < <(sha256sum < "$file")
	[ -z "$size" ] && size=$(stat -c%s -- "$file")
	if [ "$blob_action" = "copy" ]; then
		cp -- "$file" "$layout_dir/blobs/sha256/$dgst"
	else
		mv -- "$file" "$layout_dir/blobs/sha256/$dgst"
	fi
	if [ "$first_layer" -eq 0 ]; then printf ' , ' >> "$layers_file"; fi
	first_layer=0
	titleEsc=$(escape_json "$title")
	printf '{ "mediaType": "%s", "digest": "sha256:%s", "size": %s, "annotations": { "org.opencontainers.image.title": "%s" } }' \
		"$mt" "$dgst" "$size" "$titleEsc" >> "$layers_file"
}

# Process files according to pack mode.
case "$PACK_MODE" in
	raw)
		# Raw mode copies source files directly to digest-addressed blobs, so
		# basename collisions cannot occur and staging is unnecessary.
		while IFS= read -r -d '' f && IFS= read -r -d '' fsize; do
			append_layer "$f" "$RAW_LAYER_MT" "$f" "$fsize" copy
		done < "$records_file" ;;
	tar|tar+gzip|tar+zstd)
		# BusyBox tar does not support NUL-delimited -T input. Load the NUL-safe
		# records into a Bash array so every argument remains exact.
		files=()
		while IFS= read -r -d '' f && IFS= read -r -d '' _; do
			files+=("$f")
		done < "$records_file"
		if [ "${#files[@]}" -eq 0 ]; then
			printf 'no files to package\n' >&2
			exit 1
		fi
		tarFile=$(mktemp "$tmp_dir/archive.XXXXXX")
		tar -cf "$tarFile" -- "${files[@]}"
		mt=$ARCHIVE_LAYER_MT
		layerName=allfiles.tar
		case "$PACK_MODE" in
			tar) outFile=$tarFile ;;
			tar+gzip) gzip -n "$tarFile"; outFile="$tarFile.gz"; layerName=allfiles.tar.gz ;;
			tar+zstd) zstd -q --no-progress "$tarFile"; outFile="$tarFile.zst"; layerName=allfiles.tar.zst ;;
		esac
		append_layer "$outFile" "$mt" "$layerName" ;;
	*) printf 'unknown PACK_MODE %s\n' "$PACK_MODE" >&2; exit 1 ;;
esac

# Create empty config blob.
config_file="$tmp_dir/config.json"
printf '{}' > "$config_file"
read -r cfg_dgst _ < <(sha256sum < "$config_file")
cfg_size=$(stat -c%s -- "$config_file")
cp -- "$config_file" "$layout_dir/blobs/sha256/$cfg_dgst"

# Generate OCI manifest.
artifactTypeEsc=$(escape_json "$ARTIFACT_TYPE")
manifest_file="$tmp_dir/manifest.json"
{
	printf '%s' "{ \"schemaVersion\": 2, \"mediaType\": \"application/vnd.oci.image.manifest.v1+json\", \"artifactType\": \"$artifactTypeEsc\", \"config\": {\"mediaType\": \"application/vnd.oci.empty.v1+json\", \"digest\": \"sha256:$cfg_dgst\", \"size\": $cfg_size}, \"layers\": [ "
	cat "$layers_file"
	printf '%s' ' ] }'
} > "$manifest_file"

# Add manifest as blob.
read -r m_dgst _ < <(sha256sum < "$manifest_file")
m_size=$(stat -c%s -- "$manifest_file")
cp -- "$manifest_file" "$layout_dir/blobs/sha256/$m_dgst"

# Create OCI index pointing to manifest.
nameEsc=$(escape_json "$NAME")
refNameEsc=$(escape_json "$REF_NAME")
cat > "$layout_dir/index.json" <<EOF_INDEX
{ "schemaVersion": 2, "mediaType": "application/vnd.oci.image.index.v1+json", "manifests": [ { "mediaType": "application/vnd.oci.image.manifest.v1+json", "digest": "sha256:$m_dgst", "size": $m_size, "annotations": { "org.opencontainers.image.title": "$nameEsc", "org.opencontainers.image.ref.name": "$refNameEsc" } } ] }
EOF_INDEX

# Create OCI layout version marker.
printf '{ "imageLayoutVersion": "1.0.0" }\n' > "$layout_dir/oci-layout"
