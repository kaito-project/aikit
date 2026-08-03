#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <image-reference> <os/architecture> <max-compressed-mib>" >&2
  exit 2
fi

image=$1
platform=$2
max_mib=$3

if ! command -v jq >/dev/null 2>&1; then
  echo "Required command not found: jq" >&2
  exit 2
fi

if command -v docker >/dev/null 2>&1 && docker buildx version >/dev/null 2>&1; then
  manifest_inspector=(docker buildx imagetools inspect --raw)
elif command -v crane >/dev/null 2>&1; then
  manifest_inspector=(crane manifest)
else
  echo "Either Docker Buildx or crane is required to inspect image manifests" >&2
  exit 2
fi

inspect_manifest() {
  "${manifest_inspector[@]}" "$1"
}

if ! [[ $max_mib =~ ^[0-9]+$ ]]; then
  echo "Maximum compressed size must be an integer MiB value: $max_mib" >&2
  exit 2
fi

os=${platform%%/*}
architecture=${platform#*/}
if [[ -z $os || -z $architecture || $architecture == "$platform" ]]; then
  echo "Platform must use os/architecture form: $platform" >&2
  exit 2
fi

manifest=$(inspect_manifest "$image")
if jq -e '.manifests != null' >/dev/null <<<"$manifest"; then
  digest=$(jq -r --arg os "$os" --arg arch "$architecture" '
    .manifests[]
    | select(.platform.os == $os and .platform.architecture == $arch)
    | .digest
  ' <<<"$manifest" | head -n 1)
  if [[ -z $digest ]]; then
    echo "No manifest found for $platform in $image" >&2
    exit 1
  fi

  repository=${image%@*}
  manifest=$(inspect_manifest "${repository}@${digest}")
fi

compressed_bytes=$(jq -r '[.layers[]?.size] | add // 0' <<<"$manifest")
if ! [[ $compressed_bytes =~ ^[0-9]+$ ]]; then
  echo "Could not determine compressed layer size for $image" >&2
  exit 1
fi

compressed_mib=$(( (compressed_bytes + 1048575) / 1048576 ))
max_bytes=$(( max_mib * 1048576 ))
printf '%s (%s): %d bytes (%d MiB compressed; budget %d MiB)\n' \
  "$image" "$platform" "$compressed_bytes" "$compressed_mib" "$max_mib"

if [[ -n ${GITHUB_STEP_SUMMARY:-} ]]; then
  printf -- '- %s (%s): **%d MiB** compressed (budget: %d MiB)\n' \
    "$image" "$platform" "$compressed_mib" "$max_mib" >>"$GITHUB_STEP_SUMMARY"
fi

if (( compressed_bytes > max_bytes )); then
  echo "Compressed image size exceeds budget by $((compressed_mib - max_mib)) MiB" >&2
  exit 1
fi
