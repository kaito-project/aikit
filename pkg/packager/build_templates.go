package packager

import (
	"fmt"

	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
)

// generateModelpackScript returns the bash script used to assemble a modelpack OCI layout.
// Arguments:
//
//	packMode: raw|tar|tar+gzip|tar+zstd
//	artifactType: model artifact type (e.g. v1.ArtifactTypeModelManifest)
//	mtManifest: manifest config media type (e.g. v1.MediaTypeModelConfig)
//	name: annotation org.opencontainers.image.title
//	refName: annotation org.opencontainers.image.ref.name
func generateModelpackScript(packMode, artifactType, mtManifest, name, refName string) string { //nolint:lll
	tmpl := `set -euo pipefail
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

find . -type f ! -name '*.lock' ! -path './.cache/*' -print | sed 's|^./||' | LC_ALL=C sort > /tmp/allfiles.list
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
done < /tmp/allfiles.list

layers_json=""
append_layer() { file="$1"; mt="$2"; fpath="$3"; metaJson="$4"; untested="$5"; [ ! -f "$file" ] && return 0; dgst=$(sha256sum "$file" | cut -d' ' -f1); size=$(stat -c%%s "$file"); mv "$file" /layout/blobs/sha256/$dgst; [ -n "$layers_json" ] && layers_json="$layers_json , "; metaEsc=$(printf '%%s' "$metaJson" | sed 's/"/\\"/g'); ann="{ \"org.cncf.model.filepath\": \"$fpath\", \"org.cncf.model.file.metadata+json\": \"$metaEsc\", \"org.cncf.model.file.mediatype.untested\": \"$untested\" }"; layers_json="${layers_json}{ \"mediaType\": \"$mt\", \"digest\": \"sha256:$dgst\", \"size\": $size, \"annotations\": $ann }"; }

det_tar() { list="$1"; out="$2"; [ ! -s "$list" ] && return 1; tar -cf "$out" -T "$list"; }

add_category() {
	list="$1"; cat="$2"; mtRaw="$3"; mtTar="$4"; mtTarGz="$5"; mtTarZst="$6"; [ ! -s "$list" ] && return 0
	case "$PACK_MODE" in
		raw)
			while IFS= read -r f; do fsize=$(stat -c%%s "$f"); meta=$(printf '{"name":"%%s","mode":420,"uid":0,"gid":0,"size":%%s,"mtime":"1970-01-01T00:00:00Z","typeflag":0}' "$f" "$fsize"); tmpCp=/tmp/raw-$(basename "$f"); cp "$f" "$tmpCp"; append_layer "$tmpCp" "$mtRaw" "$f" "$meta" "true"; done < "$list" ;;
		tar|tar+gzip|tar+zstd)
			if [ "$cat" = "weights" ]; then
				while IFS= read -r f; do b=$(basename "$f"); tmpTar=/tmp/${cat}-$b.tar; tar -cf "$tmpTar" -C "$(dirname "$f")" "$b"; case "$PACK_MODE" in tar) mt=$mtTar ;; tar+gzip) gzip -n "$tmpTar"; tmpTar="$tmpTar.gz"; mt=$mtTarGz ;; tar+zstd) zstd -q --no-progress "$tmpTar"; tmpTar="$tmpTar.zst"; mt=$mtTarZst ;; esac; fsize=$(stat -c%%s "$f"); meta=$(printf '{"name":"%%s","mode":420,"uid":0,"gid":0,"size":%%s,"mtime":"1970-01-01T00:00:00Z","typeflag":0}' "$f" "$fsize"); append_layer "$tmpTar" "$mt" "$f" "$meta" "true"; done < "$list"
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
if [ "$(head -c1 /tmp/manifest.json)" != "{" ] || \
	 ! grep -q '"schemaVersion": 2' /tmp/manifest.json || \
	 ! grep -q '"mediaType": "application/vnd.oci.image.manifest.v1+json"' /tmp/manifest.json; then
	echo "manifest validation failed" >&2; cat /tmp/manifest.json >&2; exit 1
fi
m_dgst=$(sha256sum /tmp/manifest.json | cut -d' ' -f1)
m_size=$(stat -c%%s /tmp/manifest.json)
cp /tmp/manifest.json /layout/blobs/sha256/$m_dgst

cat > /layout/index.json <<IDX
{ "schemaVersion": 2, "mediaType": "application/vnd.oci.image.index.v1+json", "manifests": [ { "mediaType": "application/vnd.oci.image.manifest.v1+json", "digest": "sha256:$m_dgst", "size": $m_size, "annotations": { "org.opencontainers.image.title": "%[4]s", "org.opencontainers.image.ref.name": "%[5]s" } } ] }
IDX
printf '{ "imageLayoutVersion": "1.0.0" }' > /layout/oci-layout
`
	return fmt.Sprintf(tmpl, packMode, artifactType, mtManifest, name, refName)
}

// generateGenericScript builds the generic artifact OCI layout assembly script.
func generateGenericScript(packMode, artifactType, name, refName string, debug bool) string { //nolint:lll
	debugLine := ""
	if debug {
		debugLine = "set -x"
	}
	rawLayerMT := ocispec.MediaTypeImageLayer
	archiveLayerMT := ocispec.MediaTypeImageLayer
	if packMode == packModeRaw { //
		rawLayerMT = "application/octet-stream"
	}
	tmpl := `set -euo pipefail
%s
PACK_MODE=%s
mkdir -p /layout/blobs/sha256
work=/src
if [ -f /src ]; then mkdir -p /worksrc && cp /src /worksrc/; work=/worksrc; fi
cd "$work"
find . -type f ! -name '*.lock' ! -path './.cache/*' -print | sed 's|^./||' | LC_ALL=C sort > /tmp/files.list
layers_json=""
append_layer() { file="$1"; mt="$2"; [ ! -f "$file" ] && return 0; dgst=$(sha256sum "$file" | cut -d' ' -f1); size=$(stat -c%%s "$file"); mv "$file" /layout/blobs/sha256/$dgst; [ -n "$layers_json" ] && layers_json="$layers_json , "; layers_json="${layers_json}{ \"mediaType\": \"$mt\", \"digest\": \"sha256:$dgst\", \"size\": $size }"; }
case "$PACK_MODE" in
	raw)
		while IFS= read -r f; do cp "$f" "/tmp/$(basename "$f")"; append_layer "/tmp/$(basename "$f")" "%s"; done < /tmp/files.list ;;
	tar|tar+gzip|tar+zstd)
		tarFile=/tmp/allfiles.tar; tar -cf "$tarFile" -T /tmp/files.list || true
		mt="%s"
		case "$PACK_MODE" in
			tar) outFile="$tarFile" ;;
			tar+gzip) gzip -n "$tarFile"; outFile="$tarFile.gz" ;;
			tar+zstd) zstd -q --no-progress "$tarFile"; outFile="$tarFile.zst" ;;
		esac
		append_layer "$outFile" "$mt" ;;
	*) echo "unknown PACK_MODE $PACK_MODE" >&2; exit 1 ;;
esac
printf '{}' > /tmp/config.json
cfg_dgst=$(sha256sum /tmp/config.json | awk '{print $1}')
cfg_size=$(stat -c%%s /tmp/config.json)
cp /tmp/config.json /layout/blobs/sha256/$cfg_dgst
manifest="{ \"schemaVersion\": 2, \"mediaType\": \"application/vnd.oci.image.manifest.v1+json\", \"artifactType\": \"%s\", \"config\": {\"mediaType\": \"application/vnd.oci.empty.v1+json\", \"digest\": \"sha256:$cfg_dgst\", \"size\": $cfg_size}, \"layers\": [ $layers_json ] }"
printf '%%s' "$manifest" > /tmp/manifest.json
m_dgst=$(sha256sum /tmp/manifest.json | awk '{print $1}')
m_size=$(stat -c%%s /tmp/manifest.json)
cp /tmp/manifest.json /layout/blobs/sha256/$m_dgst
cat > /layout/index.json <<EOF
{ "schemaVersion": 2, "mediaType": "application/vnd.oci.image.index.v1+json", "manifests": [ { "mediaType": "application/vnd.oci.image.manifest.v1+json", "digest": "sha256:$m_dgst", "size": $m_size, "annotations": { "org.opencontainers.image.title": "%s", "org.opencontainers.image.ref.name": "%s" } } ] }
EOF
cat > /layout/oci-layout <<EOF
{ "imageLayoutVersion": "1.0.0" }
EOF
`
	return fmt.Sprintf(tmpl, debugLine, packMode, rawLayerMT, archiveLayerMT, artifactType, name, refName)
}
