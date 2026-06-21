set -euo pipefail

# Assemble a modelpack OCI layout from files mounted at /src into /layout.
#
# Parameters are passed via environment variables:
#   PACK_MODE             raw|tar|tar+gzip|tar+zstd - how to package layer content
#   ARTIFACT_TYPE         manifest artifactType (e.g. v1.ArtifactTypeModelManifest)
#   MT_MANIFEST           manifest config media type (e.g. v1.MediaTypeModelConfig)
#   NAME                  org.opencontainers.image.title annotation
#   REF_NAME              org.opencontainers.image.ref.name annotation
#   LARGE_FILE_THRESHOLD  size in bytes above which unknown files are treated as weights

# Initialize OCI layout directory structure.
mkdir -p /layout/blobs/sha256

# Handle single file input (copy to temporary directory).
src=/src
if [ -f /src ]; then mkdir -p /worksrc && cp /src /worksrc/; src=/worksrc; fi
cd "$src"

# Initialize category lists for file classification.
> /tmp/weights.list
> /tmp/config.list
> /tmp/docs.list
> /tmp/code.list
> /tmp/dataset.list

# Find all files, excluding lock files and cache, and sort deterministically.
# Also cache file sizes in parallel to avoid repeated stat calls.
find . -type f ! -name '*.lock' ! -path './.cache/*' -print0 | \
	xargs -0 -P $(nproc) -I {} sh -c 'echo "{}|$(stat -c%s "{}")"' | \
	LC_ALL=C sort > /tmp/allfiles_with_size.list

# Categorize files by extension and size into appropriate lists.
# File size is already computed and cached.
while IFS='|' read -r f sz; do
	f=${f#./}
	base=$(basename "$f" | tr A-Z a-z)
	case "$base" in
		# Model weight files.
		*.safetensors|*.bin|*.gguf|*.pt|*.ckpt) echo "$f" >> /tmp/weights.list ;;
		# Documentation files.
		readme*|license*|license|*.md) echo "$f" >> /tmp/docs.list ;;
		# Configuration and tokenizer files.
		config.json|tokenizer.json|*tokenizer*.json|generation_config.json|*.json|*.txt) echo "$f" >> /tmp/config.list ;;
		# Code files.
		*.py|*.sh|*.ipynb|*.go|*.js|*.ts) echo "$f" >> /tmp/code.list ;;
		# Dataset files.
		*.csv|*.tsv|*.jsonl|*.parquet|*.arrow|*.h5|*.npz) echo "$f" >> /tmp/dataset.list ;;
		# Unknown files: large ones go to weights, small ones to config.
		*) if [ "$sz" -gt "$LARGE_FILE_THRESHOLD" ]; then echo "$f" >> /tmp/weights.list; else echo "$f" >> /tmp/config.list; fi ;;
	esac
	# Cache size for later use.
	echo "$f|$sz" >> /tmp/file_sizes.cache
done < /tmp/allfiles_with_size.list

# Initialize JSON array for manifest layers.
layers_json=""

# escape_json escapes a string for safe inclusion as a JSON string value.
# Backslashes must be escaped before double-quotes so the two passes do not
# interfere. Tabs and newlines are also escaped so paths containing them do not
# produce invalid JSON (literal control characters are not allowed in JSON
# strings). This prevents file paths or metadata containing " \ tab or newline
# from producing invalid JSON.
escape_json() {
	printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' -e 's/\t/\\t/g' | awk 'BEGIN{ORS=""} {if(NR>1)printf "\\n"; print}'
}

# get_cached_size retrieves a cached file size to avoid repeated stat calls.
get_cached_size() {
	file="$1"
	grep -F "$file|" /tmp/file_sizes.cache 2>/dev/null | cut -d'|' -f2 | head -n1
}

# append_layer adds a file as a layer blob with annotations.
# Args: file path, media type, filepath annotation, metadata JSON, untested flag.
append_layer() {
	file="$1"; mt="$2"; fpath="$3"; metaJson="$4"; untested="$5"
	[ ! -f "$file" ] && return 0
	dgst=$(sha256sum "$file" | cut -d' ' -f1)
	size=$(stat -c%s "$file")
	mv "$file" /layout/blobs/sha256/$dgst
	[ -n "$layers_json" ] && layers_json="$layers_json , "
	fpathEsc=$(escape_json "$fpath")
	metaEsc=$(escape_json "$metaJson")
	ann="{ \"org.opencontainers.image.title\": \"$fpathEsc\", \"org.cncf.model.filepath\": \"$fpathEsc\", \"org.cncf.model.file.metadata+json\": \"$metaEsc\", \"org.cncf.model.file.mediatype.untested\": \"$untested\" }"
	layers_json="${layers_json}{ \"mediaType\": \"$mt\", \"digest\": \"sha256:$dgst\", \"size\": $size, \"annotations\": $ann }"
}

# det_tar creates a deterministic tar archive from a file list.
det_tar() { list="$1"; out="$2"; [ ! -s "$list" ] && return 1; tar -cf "$out" -T "$list"; }

# add_category processes a file category and adds layers according to pack mode.
# Args: list file, category name, raw media type, tar media type, tar+gzip media type, tar+zstd media type.
add_category() {
	list="$1"; cat="$2"; mtRaw="$3"; mtTar="$4"; mtTarGz="$5"; mtTarZst="$6"
	[ ! -s "$list" ] && return 0
	case "$PACK_MODE" in
		raw)
			# Raw mode: each file becomes its own layer.
			while IFS= read -r f; do
				fsize=$(get_cached_size "$f")
				[ -z "$fsize" ] && fsize=$(stat -c%s "$f")  # Fallback to stat if cache miss.
				nameEsc=$(escape_json "$f")
				meta=$(printf '{"name":"%s","mode":420,"uid":0,"gid":0,"size":%s,"mtime":"1970-01-01T00:00:00Z","typeflag":0}' "$nameEsc" "$fsize")
				tmpCp=/tmp/raw-$(basename "$f")
				cp "$f" "$tmpCp"
				append_layer "$tmpCp" "$mtRaw" "$f" "$meta" "true"
			done < "$list" ;;
		tar|tar+gzip|tar+zstd)
			if [ "$cat" = "weights" ]; then
				# Weights: tar each file individually (can be large).
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
					[ -z "$fsize" ] && fsize=$(stat -c%s "$f")
					nameEsc=$(escape_json "$f")
					meta=$(printf '{"name":"%s","mode":420,"uid":0,"gid":0,"size":%s,"mtime":"1970-01-01T00:00:00Z","typeflag":0}' "$nameEsc" "$fsize")
					append_layer "$tmpTar" "$mt" "$f" "$meta" "true"
				done < "$list"
			else
				# Non-weights: bundle all category files into a single tar.
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
					[ -z "$sz" ] && sz=$(stat -c%s "$f2")
					totalSize=$((totalSize + sz))
				done < "$list"
				nameEsc=$(escape_json "$cat")
				meta=$(printf '{"name":"%s","mode":420,"uid":0,"gid":0,"size":%s,"mtime":"1970-01-01T00:00:00Z","typeflag":0,"files":%d}' "$nameEsc" "$totalSize" "$count")
				append_layer "$outFile" "$mt" "$cat" "$meta" "true"
			fi ;;
		*) echo "unknown PACK_MODE $PACK_MODE" >&2; exit 1 ;;
	esac
}

# Process each file category with appropriate ModelPack media types.
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

# Create empty manifest config and add as blob.
printf '{}' > /tmp/manifest-config.json
mc_dgst=$(sha256sum /tmp/manifest-config.json | cut -d' ' -f1)
mc_size=$(stat -c%s /tmp/manifest-config.json)
cp /tmp/manifest-config.json /layout/blobs/sha256/$mc_dgst

# Generate OCI manifest with all layers.
cat > /tmp/manifest.json <<EOF_MANIFEST
{ "schemaVersion": 2, "mediaType": "application/vnd.oci.image.manifest.v1+json", "artifactType": "$ARTIFACT_TYPE", "config": {"mediaType": "$MT_MANIFEST", "digest": "sha256:$mc_dgst", "size": $mc_size}, "layers": [ $layers_json ] }
EOF_MANIFEST

# Validate manifest structure.
if [ "$(head -c1 /tmp/manifest.json)" != "{" ] || \
	 ! grep -q '"schemaVersion": 2' /tmp/manifest.json || \
	 ! grep -q '"mediaType": "application/vnd.oci.image.manifest.v1+json"' /tmp/manifest.json; then
	echo "manifest validation failed" >&2; cat /tmp/manifest.json >&2; exit 1
fi

# Add manifest as blob.
m_dgst=$(sha256sum /tmp/manifest.json | cut -d' ' -f1)
m_size=$(stat -c%s /tmp/manifest.json)
cp /tmp/manifest.json /layout/blobs/sha256/$m_dgst

# Create OCI index pointing to manifest.
nameEsc=$(escape_json "$NAME")
refNameEsc=$(escape_json "$REF_NAME")
cat > /layout/index.json <<IDX
{ "schemaVersion": 2, "mediaType": "application/vnd.oci.image.index.v1+json", "manifests": [ { "mediaType": "application/vnd.oci.image.manifest.v1+json", "digest": "sha256:$m_dgst", "size": $m_size, "annotations": { "org.opencontainers.image.title": "$nameEsc", "org.opencontainers.image.ref.name": "$refNameEsc" } } ] }
IDX

# Create OCI layout version marker.
printf '{ "imageLayoutVersion": "1.0.0" }' > /layout/oci-layout
