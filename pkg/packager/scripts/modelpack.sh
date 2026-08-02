#!/usr/bin/env bash

set -euo pipefail

# Assemble a modelpack OCI layout from files mounted at /src into /layout.
# AIKIT_SOURCE_DIR, AIKIT_LAYOUT_DIR, and AIKIT_TMP_DIR override those locations
# for executable tests; production builds use the defaults.
#
# Parameters are passed via environment variables:
#   PACK_MODE             raw|tar|tar+gzip|tar+zstd - how to package layer content
#   ARTIFACT_TYPE         manifest artifactType (e.g. v1.ArtifactTypeModelManifest)
#   MT_MANIFEST           manifest config media type (e.g. v1.MediaTypeModelConfig)
#   NAME                  org.opencontainers.image.title annotation
#   REF_NAME              org.opencontainers.image.ref.name annotation
#   LARGE_FILE_THRESHOLD  size in bytes above which unknown files are treated as weights

source_dir=${AIKIT_SOURCE_DIR:-/src}
layout_dir=${AIKIT_LAYOUT_DIR:-/layout}
tmp_root=${AIKIT_TMP_DIR:-${TMPDIR:-/tmp}}
mkdir -p "$layout_dir/blobs/sha256" "$tmp_root"
tmp_dir=$(mktemp -d "$tmp_root/aikit-modelpack.XXXXXX")
cleanup() { rm -rf -- "$tmp_dir"; }
trap cleanup EXIT

# Handle single file input by copying it to an isolated working directory.
src=$source_dir
if [ -f "$source_dir" ]; then
	src="$tmp_dir/worksrc"
	mkdir -p "$src"
	cp -- "$source_dir" "$src/"
fi
cd "$src"

# Every category record is a NUL-delimited path followed by a NUL-delimited
# decimal size. Filenames are never serialized with newline or pipe delimiters.
weights_list="$tmp_dir/weights.records"
config_list="$tmp_dir/config.records"
docs_list="$tmp_dir/docs.records"
code_list="$tmp_dir/code.records"
dataset_list="$tmp_dir/dataset.records"
: > "$weights_list"
: > "$config_list"
: > "$docs_list"
: > "$code_list"
: > "$dataset_list"

# Discover paths in deterministic byte order. Paths stay NUL-delimited, while
# stat emits one numeric size per line in the same order. xargs batches stat
# calls without ever parsing a filename as text.
paths_file="$tmp_dir/allfiles.paths"
sizes_file="$tmp_dir/allfiles.sizes"
find . -type f ! -name '*.lock' ! -path './.cache/*' -print0 | LC_ALL=C sort -z > "$paths_file"
xargs -0 -r stat -c '%s' -- < "$paths_file" > "$sizes_file"

# Categorize files by extension and size into NUL-safe records.
exec 3<"$paths_file"
exec 4<"$sizes_file"
while IFS= read -r -d '' f <&3; do
	if ! IFS= read -r sz <&4; then
		printf 'missing size for discovered path %q\n' "$f" >&2
		exit 1
	fi
	f=${f#./}
	base=${f##*/}
	base=${base,,}
	case "$base" in
		# Model weight files.
		*.safetensors|*.bin|*.gguf|*.pt|*.ckpt) list=$weights_list ;;
		# Documentation files.
		readme*|license*|*.md) list=$docs_list ;;
		# Configuration and tokenizer files.
		config.json|tokenizer.json|*tokenizer*.json|generation_config.json|*.json|*.txt) list=$config_list ;;
		# Code files.
		*.py|*.sh|*.ipynb|*.go|*.js|*.ts) list=$code_list ;;
		# Dataset files.
		*.csv|*.tsv|*.jsonl|*.parquet|*.arrow|*.h5|*.npz) list=$dataset_list ;;
		# Unknown files: large ones go to weights, small ones to config.
		*) if [ "$sz" -gt "$LARGE_FILE_THRESHOLD" ]; then list=$weights_list; else list=$config_list; fi ;;
	esac
	printf '%s\0%s\0' "$f" "$sz" >> "$list"
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
# Backslashes must be escaped before double-quotes so the two passes do not
# interfere. Tabs and newlines are also escaped so paths containing them do not
# produce invalid JSON.
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

# BusyBox tar supports newline-delimited -T input but not a NUL-delimited list,
# so arbitrary filenames cannot be streamed to one invocation through -T. Build
# bounded tar chunks from NUL-delimited xargs batches, strip each chunk's end
# marker, and concatenate the member records into one valid archive. The final
# directory marker makes trailer detection unambiguous even when file data ends
# in one or more all-zero 512-byte blocks.
archive_batch_bytes=32768
zero_block="$tmp_dir/zero.block"
tar_batch_script="$tmp_dir/archive-batch.sh"
dd if=/dev/zero of="$zero_block" bs=512 count=1 2>/dev/null
cat > "$tar_batch_script" <<'EOF_ARCHIVE_BATCH'
#!/usr/bin/env bash
set -euo pipefail

archive=$1
tmp_dir=$2
zero_block=$3
shift 3

chunk=$(mktemp "$tmp_dir/archive-chunk.XXXXXX")
cleanup_batch() { rm -f -- "$chunk"; }
trap cleanup_batch EXIT

tar -cf "$chunk" --no-recursion -- "$@" .
size=$(stat -c%s -- "$chunk")
if [ "$size" -eq 0 ] || [ $((size % 512)) -ne 0 ]; then
	printf 'tar chunk has invalid size %s\n' "$size" >&2
	exit 1
fi

blocks=$((size / 512))
end=$blocks
while [ "$end" -gt 0 ] && cmp -s -n 512 "$zero_block" "$chunk" 0 $(((end - 1) * 512)); do
	end=$((end - 1))
done
if [ "$end" -lt 1 ] || [ $((blocks - end)) -lt 2 ]; then
	printf 'tar chunk is missing its end-of-archive marker\n' >&2
	exit 1
fi

marker_name=$(dd if="$chunk" bs=512 skip=$((end - 1)) count=1 2>/dev/null | dd bs=1 count=100 2>/dev/null | tr -d '\000')
case "$marker_name" in
	.|./) ;;
	*)
		printf 'unexpected tar chunk marker %q\n' "$marker_name" >&2
		exit 1
		;;
esac

payload_blocks=$((end - 1))
if [ "$payload_blocks" -gt 0 ]; then
	dd if="$chunk" bs=512 count="$payload_blocks" 2>/dev/null >> "$archive"
fi
EOF_ARCHIVE_BATCH

create_tar_archive() {
	local members=$1 archive=$2
	: > "$archive"
	xargs -0 -r -s "$archive_batch_bytes" bash "$tar_batch_script" "$archive" "$tmp_dir" "$zero_block" < "$members"
	cat "$zero_block" "$zero_block" >> "$archive"
}

# append_layer adds a file as a layer blob with annotations.
# Args: file path, media type, filepath annotation, metadata JSON, untested flag,
# optional size, and optional blob action.
append_layer() {
	local file=$1 mt=$2 fpath=$3 metaJson=$4 untested=$5 size=${6-} blob_action=${7:-move}
	local dgst fpathEsc metaEsc
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
	fpathEsc=$(escape_json "$fpath")
	metaEsc=$(escape_json "$metaJson")
	printf '{ "mediaType": "%s", "digest": "sha256:%s", "size": %s, "annotations": { "org.opencontainers.image.title": "%s", "org.cncf.model.filepath": "%s", "org.cncf.model.file.metadata+json": "%s", "org.cncf.model.file.mediatype.untested": "%s" } }' \
		"$mt" "$dgst" "$size" "$fpathEsc" "$fpathEsc" "$metaEsc" "$untested" >> "$layers_file"
}

# add_category processes a file category and adds layers according to pack mode.
# Args: record file, category name, raw media type, tar media type, tar+gzip media type, tar+zstd media type.
add_category() {
	local list=$1 cat=$2 mtRaw=$3 mtTar=$4 mtTarGz=$5 mtTarZst=$6
	local f fsize nameEsc meta tmpTar mt outFile count totalSize members_file
	[ ! -s "$list" ] && return 0
	case "$PACK_MODE" in
		raw)
			# Raw mode copies source files directly to digest-addressed blobs, so
			# basename collisions cannot occur and staging is unnecessary.
			while IFS= read -r -d '' f && IFS= read -r -d '' fsize; do
				nameEsc=$(escape_json "$f")
				meta=$(printf '{"name":"%s","mode":420,"uid":0,"gid":0,"size":%s,"mtime":"1970-01-01T00:00:00Z","typeflag":0}' "$nameEsc" "$fsize")
				append_layer "$f" "$mtRaw" "$f" "$meta" true "$fsize" copy
			done < "$list" ;;
		tar|tar+gzip|tar+zstd)
			if [ "$cat" = weights ]; then
				# Weights are archived individually and retain their exact basename.
				while IFS= read -r -d '' f && IFS= read -r -d '' fsize; do
					b=${f##*/}
					dir=${f%/*}
					[ "$dir" = "$f" ] && dir=.
					tmpTar=$(mktemp "$tmp_dir/aikit-${cat}.XXXXXX")
					tar -cf "$tmpTar" -C "$dir" -- "$b"
					case "$PACK_MODE" in
						tar) mt=$mtTar ;;
						tar+gzip) gzip -n "$tmpTar"; tmpTar="$tmpTar.gz"; mt=$mtTarGz ;;
						tar+zstd) zstd -q --no-progress "$tmpTar"; tmpTar="$tmpTar.zst"; mt=$mtTarZst ;;
					esac
					nameEsc=$(escape_json "$f")
					meta=$(printf '{"name":"%s","mode":420,"uid":0,"gid":0,"size":%s,"mtime":"1970-01-01T00:00:00Z","typeflag":0}' "$nameEsc" "$fsize")
					append_layer "$tmpTar" "$mt" "$f" "$meta" true
				done < "$list"
			else
				members_file="$tmp_dir/${cat}.members"
				: > "$members_file"
				count=0
				totalSize=0
				while IFS= read -r -d '' f && IFS= read -r -d '' fsize; do
					printf '%s\0' "$f" >> "$members_file"
					totalSize=$((totalSize + fsize))
					count=$((count + 1))
				done < "$list"
				[ "$count" -eq 0 ] && return 0
				tmpTar=$(mktemp "$tmp_dir/aikit-${cat}.XXXXXX")
				create_tar_archive "$members_file" "$tmpTar"
				case "$PACK_MODE" in
					tar) outFile=$tmpTar; mt=$mtTar ;;
					tar+gzip) gzip -n "$tmpTar"; outFile="$tmpTar.gz"; mt=$mtTarGz ;;
					tar+zstd) zstd -q --no-progress "$tmpTar"; outFile="$tmpTar.zst"; mt=$mtTarZst ;;
				esac
				nameEsc=$(escape_json "$cat")
				meta=$(printf '{"name":"%s","mode":420,"uid":0,"gid":0,"size":%s,"mtime":"1970-01-01T00:00:00Z","typeflag":0,"files":%d}' "$nameEsc" "$totalSize" "$count")
				append_layer "$outFile" "$mt" "$cat" "$meta" true
			fi ;;
		*) printf 'unknown PACK_MODE %s\n' "$PACK_MODE" >&2; exit 1 ;;
	esac
}

# Process categories in deterministic ModelPack order. The first call preserves
# the historical embedded-script marker: add_category /tmp/weights.list weights.
add_category "$weights_list" weights \
	application/vnd.cncf.model.weight.v1.raw \
	application/vnd.cncf.model.weight.v1.tar \
	application/vnd.cncf.model.weight.v1.tar+gzip \
	application/vnd.cncf.model.weight.v1.tar+zstd
add_category "$config_list" config \
	application/vnd.cncf.model.weight.config.v1.raw \
	application/vnd.cncf.model.weight.config.v1.tar \
	application/vnd.cncf.model.weight.config.v1.tar+gzip \
	application/vnd.cncf.model.weight.config.v1.tar+zstd
add_category "$docs_list" docs \
	application/vnd.cncf.model.doc.v1.raw \
	application/vnd.cncf.model.doc.v1.tar \
	application/vnd.cncf.model.doc.v1.tar+gzip \
	application/vnd.cncf.model.doc.v1.tar+zstd
add_category "$code_list" code \
	application/vnd.cncf.model.code.v1.raw \
	application/vnd.cncf.model.code.v1.tar \
	application/vnd.cncf.model.code.v1.tar+gzip \
	application/vnd.cncf.model.code.v1.tar+zstd
add_category "$dataset_list" dataset \
	application/vnd.cncf.model.dataset.v1.raw \
	application/vnd.cncf.model.dataset.v1.tar \
	application/vnd.cncf.model.dataset.v1.tar+gzip \
	application/vnd.cncf.model.dataset.v1.tar+zstd

# Create empty manifest config and add as blob.
manifest_config_file="$tmp_dir/manifest-config.json"
printf '{}' > "$manifest_config_file"
read -r mc_dgst _ < <(sha256sum < "$manifest_config_file")
mc_size=$(stat -c%s -- "$manifest_config_file")
cp -- "$manifest_config_file" "$layout_dir/blobs/sha256/$mc_dgst"

# Generate OCI manifest with all layers.
artifactTypeEsc=$(escape_json "$ARTIFACT_TYPE")
mtManifestEsc=$(escape_json "$MT_MANIFEST")
manifest_file="$tmp_dir/manifest.json"
{
	printf '%s' "{ \"schemaVersion\": 2, \"mediaType\": \"application/vnd.oci.image.manifest.v1+json\", \"artifactType\": \"$artifactTypeEsc\", \"config\": {\"mediaType\": \"$mtManifestEsc\", \"digest\": \"sha256:$mc_dgst\", \"size\": $mc_size}, \"layers\": [ "
	cat "$layers_file"
	printf '%s\n' ' ] }'
} > "$manifest_file"

# Validate manifest structure.
if [ "$(head -c1 "$manifest_file")" != "{" ] || \
	 ! grep -q '"schemaVersion": 2' "$manifest_file" || \
	 ! grep -q '"mediaType": "application/vnd.oci.image.manifest.v1+json"' "$manifest_file"; then
	echo "manifest validation failed" >&2
	cat "$manifest_file" >&2
	exit 1
fi

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
printf '{ "imageLayoutVersion": "1.0.0" }' > "$layout_dir/oci-layout"
