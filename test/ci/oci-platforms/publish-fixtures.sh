#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <registry/repository>" >&2
  exit 2
fi

repository=${1%/}
fixture_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
work_dir=$(mktemp -d)
trap 'rm -rf "$work_dir"' EXIT

oras_args=(--plain-http)
model_artifact_type=application/vnd.cncf.model.manifest.v1+json
model_layer_type=application/vnd.cncf.model.weight.v1.raw
attestation_artifact_type=application/vnd.in-toto+json
attestation_layer_type=application/vnd.in-toto+json

push_artifact() {
  local tag=$1
  local payload=$2
  local artifact_type=$3
  local layer_type=$4
  local payload_dir
  local payload_name

  payload_dir=$(dirname "$payload")
  payload_name=$(basename "$payload")

  (
    cd "$payload_dir"
    oras push "${oras_args[@]}" \
      --artifact-type "$artifact_type" \
      --annotation "org.opencontainers.image.description=$tag" \
      "$repository:$tag" \
      "$payload_name:$layer_type"
  )

  oras manifest fetch "${oras_args[@]}" --descriptor "$repository:$tag" > "$work_dir/$tag.descriptor.json"
  jq -e '
    .mediaType == "application/vnd.oci.image.manifest.v1+json" and
    (.digest | startswith("sha256:")) and
    (.size > 0)
  ' "$work_dir/$tag.descriptor.json" >/dev/null
}

push_artifact \
  single-amd64 \
  "$fixture_dir/payloads/single/model.gguf" \
  "$model_artifact_type" \
  "$model_layer_type"
push_artifact \
  multi-amd64 \
  "$fixture_dir/payloads/multi-amd64/model.gguf" \
  "$model_artifact_type" \
  "$model_layer_type"
push_artifact \
  multi-arm64 \
  "$fixture_dir/payloads/multi-arm64/model.gguf" \
  "$model_artifact_type" \
  "$model_layer_type"
push_artifact \
  multi-attestation \
  "$fixture_dir/payloads/attestation/predicate.json" \
  "$attestation_artifact_type" \
  "$attestation_layer_type"

jq -n \
  --slurpfile child "$work_dir/single-amd64.descriptor.json" \
  '{
    schemaVersion: 2,
    mediaType: "application/vnd.oci.image.index.v1+json",
    artifactType: "application/vnd.cncf.model.index.v1+json",
    manifests: [
      ($child[0] + {platform: {os: "linux", architecture: "amd64"}})
    ]
  }' > "$work_dir/single-stamped.index.json"

jq -n \
  --slurpfile amd64 "$work_dir/multi-amd64.descriptor.json" \
  --slurpfile arm64 "$work_dir/multi-arm64.descriptor.json" \
  --slurpfile attestation "$work_dir/multi-attestation.descriptor.json" \
  '{
    schemaVersion: 2,
    mediaType: "application/vnd.oci.image.index.v1+json",
    artifactType: "application/vnd.cncf.model.index.v1+json",
    manifests: [
      ($amd64[0] + {platform: {os: "linux", architecture: "amd64"}}),
      ($arm64[0] + {platform: {os: "linux", architecture: "arm64"}}),
      ($attestation[0] + {
        platform: {os: "unknown", architecture: "unknown"},
        annotations: {
          "vnd.docker.reference.type": "attestation-manifest",
          "vnd.docker.reference.digest": $arm64[0].digest
        }
      })
    ]
  }' > "$work_dir/multi-platform.index.json"

oras manifest push "${oras_args[@]}" \
  "$repository:single-stamped" \
  "$work_dir/single-stamped.index.json"
oras manifest push "${oras_args[@]}" \
  "$repository:multi-platform" \
  "$work_dir/multi-platform.index.json"

oras manifest fetch "${oras_args[@]}" "$repository:single-stamped" |
  jq -e '
    .mediaType == "application/vnd.oci.image.index.v1+json" and
    (.manifests | length == 1) and
    (.manifests[0].platform == {os: "linux", architecture: "amd64"})
  ' >/dev/null

oras manifest fetch "${oras_args[@]}" "$repository:multi-platform" |
  jq -e '
    .mediaType == "application/vnd.oci.image.index.v1+json" and
    (.manifests | length == 3) and
    ([.manifests[].platform | select(.os == "linux") | .architecture] | sort == ["amd64", "arm64"]) and
    ([.manifests[] | select(.annotations["vnd.docker.reference.type"] == "attestation-manifest")] | length == 1) and
    ([.manifests[] | select(.platform.os == "unknown" and .platform.architecture == "unknown")] | length == 1)
  ' >/dev/null

printf 'Published OCI platform fixtures:\n'
printf '  %s:single-stamped\n' "$repository"
printf '  %s:multi-platform\n' "$repository"
