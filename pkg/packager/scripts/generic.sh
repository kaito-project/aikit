set -euo pipefail
# Optional debug tracing, enabled when DEBUG is set to a non-empty value.
[ -n "${DEBUG:-}" ] && set -x

# Assemble a generic artifact OCI layout from files mounted at /src into /layout.
#
# Parameters are passed via environment variables:
#   PACK_MODE         raw|tar|tar+gzip|tar+zstd - packaging method
#   ARTIFACT_TYPE     manifest artifactType
#   RAW_LAYER_MT      media type for raw-mode layers
#   ARCHIVE_LAYER_MT  media type for archive-mode layers
#   NAME              org.opencontainers.image.title annotation
#   REF_NAME          org.opencontainers.image.ref.name annotation

# Initialize OCI layout directory structure.
mkdir -p /layout/blobs/sha256

# Handle single file input (copy to temporary directory).
work=/src
if [ -f /src ]; then mkdir -p /worksrc && cp /src /worksrc/; work=/worksrc; fi
cd "$work"

# Find all files, excluding lock files and cache, sorted deterministically.
# Cache file sizes for later use.
find . -type f ! -name '*.lock' ! -path './.cache/*' -print0 | \
	xargs -0 -P $(nproc) -I {} sh -c 'f="{}"; echo "$f|$(stat -c%s "$f")"' | \
	sed 's|^\./||' | LC_ALL=C sort > /tmp/files_with_size.list

# Extract just the file paths for processing.
cut -d'|' -f1 < /tmp/files_with_size.list > /tmp/files.list

# Initialize JSON array for manifest layers.
layers_json=""

# escape_json escapes a string for safe inclusion as a JSON string value.
# Backslashes must be escaped before double-quotes. Tabs and newlines are also
# escaped so paths containing them do not produce invalid JSON.
escape_json() {
	printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' -e 's/\t/\\t/g' | awk 'BEGIN{ORS=""} {if(NR>1)printf "\\n"; print}'
}

# get_file_size retrieves a cached file size.
get_file_size() {
	grep -F "$1|" /tmp/files_with_size.list 2>/dev/null | cut -d'|' -f2 | head -n1
}

# append_layer adds a file as a layer blob with annotations.
# Args: file path, media type, title (original filename).
append_layer() {
	file="$1"; mt="$2"; title="$3"
	[ ! -f "$file" ] && return 0
	dgst=$(sha256sum "$file" | cut -d' ' -f1)
	size=$(stat -c%s "$file")
	mv "$file" /layout/blobs/sha256/$dgst
	[ -n "$layers_json" ] && layers_json="$layers_json , "
	titleEsc=$(escape_json "$title")
	ann="{ \"org.opencontainers.image.title\": \"$titleEsc\" }"
	layers_json="${layers_json}{ \"mediaType\": \"$mt\", \"digest\": \"sha256:$dgst\", \"size\": $size, \"annotations\": $ann }"
}

# Process files according to pack mode.
case "$PACK_MODE" in
	raw)
		# Raw mode: each file becomes its own layer.
		while IFS= read -r f; do
			cp "$f" "/tmp/$(basename "$f")"
			append_layer "/tmp/$(basename "$f")" "$RAW_LAYER_MT" "$f"
		done < /tmp/files.list ;;
	tar|tar+gzip|tar+zstd)
		# Archive mode: bundle all files into a single tar.
		tarFile=/tmp/allfiles.tar
		tar -cf "$tarFile" -T /tmp/files.list || true
		mt="$ARCHIVE_LAYER_MT"
		layerName="allfiles.tar"
		case "$PACK_MODE" in
			tar) outFile="$tarFile" ;;
			tar+gzip) gzip -n "$tarFile"; outFile="$tarFile.gz"; layerName="allfiles.tar.gz" ;;
			tar+zstd) zstd -q --no-progress "$tarFile"; outFile="$tarFile.zst"; layerName="allfiles.tar.zst" ;;
		esac
		append_layer "$outFile" "$mt" "$layerName" ;;
	*) echo "unknown PACK_MODE $PACK_MODE" >&2; exit 1 ;;
esac

# Create empty config blob.
printf '{}' > /tmp/config.json
cfg_dgst=$(sha256sum /tmp/config.json | awk '{print $1}')
cfg_size=$(stat -c%s /tmp/config.json)
cp /tmp/config.json /layout/blobs/sha256/$cfg_dgst

# Generate OCI manifest.
manifest="{ \"schemaVersion\": 2, \"mediaType\": \"application/vnd.oci.image.manifest.v1+json\", \"artifactType\": \"$ARTIFACT_TYPE\", \"config\": {\"mediaType\": \"application/vnd.oci.empty.v1+json\", \"digest\": \"sha256:$cfg_dgst\", \"size\": $cfg_size}, \"layers\": [ $layers_json ] }"
printf '%s' "$manifest" > /tmp/manifest.json

# Add manifest as blob.
m_dgst=$(sha256sum /tmp/manifest.json | awk '{print $1}')
m_size=$(stat -c%s /tmp/manifest.json)
cp /tmp/manifest.json /layout/blobs/sha256/$m_dgst

# Create OCI index pointing to manifest.
nameEsc=$(escape_json "$NAME")
refNameEsc=$(escape_json "$REF_NAME")
cat > /layout/index.json <<EOF
{ "schemaVersion": 2, "mediaType": "application/vnd.oci.image.index.v1+json", "manifests": [ { "mediaType": "application/vnd.oci.image.manifest.v1+json", "digest": "sha256:$m_dgst", "size": $m_size, "annotations": { "org.opencontainers.image.title": "$nameEsc", "org.opencontainers.image.ref.name": "$refNameEsc" } } ] }
EOF

# Create OCI layout version marker.
cat > /layout/oci-layout <<EOF
{ "imageLayoutVersion": "1.0.0" }
EOF
