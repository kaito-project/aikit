#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 7 || $# -gt 8 ]]; then
  echo "Usage: $0 <ghcr-repository> <latest|stable-version> <artifact> <source-repository> <workflow-path> <oidc-issuer> <remote-tags> [target-version]" >&2
  exit 2
fi

ghcr_repository=$1
reference=$2
artifact=$3
source_repository=$4
workflow_path=$5
oidc_issuer=$6
remote_tags=$7
target_version=${8:-}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
digest_resolver="$script_dir/resolve-ghcr-tag-digest.sh"
export LC_ALL=C

fail() {
  echo "Release alias floor check failed: $*" >&2
  exit 1
}

parse_version() {
  local version=$1

  if ! [[ $version =~ ^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$ ]]; then
    return 1
  fi

  parsed_major=${BASH_REMATCH[1]}
  parsed_minor=${BASH_REMATCH[2]}
  parsed_patch=${BASH_REMATCH[3]}
}

compare_component() {
  local left=$1
  local right=$2

  component_comparison=0
  if ((${#left} > ${#right})); then
    component_comparison=1
  elif ((${#left} < ${#right})); then
    component_comparison=-1
  elif [[ $left > $right ]]; then
    component_comparison=1
  elif [[ $left < $right ]]; then
    component_comparison=-1
  fi
}

compare_versions() {
  local left=$1
  local right=$2
  local left_major
  local left_minor
  local left_patch
  local right_major
  local right_minor
  local right_patch

  parse_version "$left" || return 2
  left_major=$parsed_major
  left_minor=$parsed_minor
  left_patch=$parsed_patch
  parse_version "$right" || return 2
  right_major=$parsed_major
  right_minor=$parsed_minor
  right_patch=$parsed_patch

  version_comparison=0
  compare_component "$left_major" "$right_major"
  if ((component_comparison != 0)); then
    version_comparison=$component_comparison
    return
  fi
  compare_component "$left_minor" "$right_minor"
  if ((component_comparison != 0)); then
    version_comparison=$component_comparison
    return
  fi
  compare_component "$left_patch" "$right_patch"
  version_comparison=$component_comparison
}

emit_floor() {
  local found=$1
  local version=$2
  local digest=$3
  local promote=

  if [[ -n $target_version ]]; then
    promote=true
    if [[ $found == true ]]; then
      compare_versions "$target_version" "$version"
      if ((version_comparison < 0)); then
        promote=false
      fi
    fi
  fi

  printf '%s\n' "$version"
  if [[ -n ${GITHUB_OUTPUT:-} ]]; then
    {
      echo "found=$found"
      echo "version=$version"
      echo "digest=$digest"
      if [[ -n $promote ]]; then
        echo "promote=$promote"
      fi
    } >>"$GITHUB_OUTPUT"
  fi
  if [[ $promote == false ]]; then
    echo "Target $target_version is below existing release floor $version; mutable promotion must be skipped." >&2
  fi
}

if ! [[ $ghcr_repository =~ ^[a-z0-9._-]+(/[a-z0-9._-]+)+$ ]]; then
  fail "invalid GHCR repository path: $ghcr_repository"
fi
explicit_version=false
if [[ $reference == latest ]]; then
  :
elif parse_version "$reference"; then
  explicit_version=true
else
  fail "invalid release reference: $reference"
fi
if [[ -n $target_version ]] && ! parse_version "$target_version"; then
  fail "invalid target version: $target_version"
fi
if ! [[ $artifact =~ ^[a-z0-9][a-z0-9._-]*$ ]]; then
  fail "invalid release artifact: $artifact"
fi
if ! [[ $source_repository =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]]; then
  fail "invalid source repository: $source_repository"
fi
if ! [[ $workflow_path =~ ^\.github/workflows/[A-Za-z0-9_.-]+\.ya?ml$ ]]; then
  fail "invalid workflow path: $workflow_path"
fi
workflow_name=${workflow_path##*/}
workflow_name=${workflow_name%.yaml}
workflow_name=${workflow_name%.yml}
if ! [[ $oidc_issuer =~ ^https://[A-Za-z0-9._~:/?@!\$\&\(\)\*+,\;=%-]+$ ]]; then
  fail "invalid OIDC issuer: $oidc_issuer"
fi
if [[ ! -f $remote_tags ]]; then
  fail "remote tags input must be a regular file"
fi
if [[ ! -x $digest_resolver ]]; then
  fail "GHCR digest resolver is unavailable"
fi
if ! command -v cosign >/dev/null 2>&1 || ! command -v jq >/dev/null 2>&1; then
  fail "cosign and jq are required"
fi

reference_digest=$(
  "$digest_resolver" "$ghcr_repository" "$reference"
)
if [[ $reference_digest == absent ]]; then
  if [[ $explicit_version == true ]]; then
    fail "immutable tag ghcr.io/${ghcr_repository}:${reference} is absent"
  fi
  echo "Mutable alias ghcr.io/${ghcr_repository}:${reference} is absent; no release floor exists." >&2
  emit_floor false none absent
  exit 0
fi
if ! [[ $reference_digest =~ ^sha256:[0-9a-f]{64}$ ]]; then
  fail "digest resolver returned an invalid digest"
fi

tag_records=$(mktemp)
trap 'rm -f "$tag_records"' EXIT
: >"$tag_records"
versions=()
while read -r object_id ref extra; do
  if [[ -z ${object_id:-} && -z ${ref:-} ]]; then
    continue
  fi
  if [[ -n ${extra:-} || ! $object_id =~ ^[0-9a-f]{40}$ || ! $ref =~ ^refs/tags/(v[^\^]+)(\^\{\})?$ ]]; then
    fail "invalid remote tag record"
  fi

  version=${BASH_REMATCH[1]}
  peeled=${BASH_REMATCH[2]:-}
  if ! parse_version "$version"; then
    continue
  fi

  record_type=object
  if [[ -n $peeled ]]; then
    record_type=peeled
  fi
  printf '%s\t%s\t%s\n' "$version" "$record_type" "$object_id" >>"$tag_records"

  seen=false
  if ((${#versions[@]} > 0)); then
    for existing_version in "${versions[@]}"; do
      if [[ $existing_version == "$version" ]]; then
        seen=true
        break
      fi
    done
  fi
  if [[ $seen == false ]]; then
    versions[${#versions[@]}]=$version
  fi
done <"$remote_tags"

resolve_tag_commit() {
  local version=$1
  local object_ids
  local peeled_ids
  local object_count
  local peeled_count

  object_ids=$(awk -F '\t' -v version="$version" \
    '$1 == version && $2 == "object" { print $3 }' "$tag_records" | sort -u)
  peeled_ids=$(awk -F '\t' -v version="$version" \
    '$1 == version && $2 == "peeled" { print $3 }' "$tag_records" | sort -u)
  object_count=$(awk 'NF { count++ } END { print count + 0 }' <<<"$object_ids")
  peeled_count=$(awk 'NF { count++ } END { print count + 0 }' <<<"$peeled_ids")

  if ((object_count > 1 || peeled_count > 1)); then
    fail "conflicting remote tag records for $version"
  fi
  if ((object_count == 0)); then
    return 1
  fi
  if ((peeled_count == 1)); then
    resolved_tag_commit=$peeled_ids
  else
    resolved_tag_commit=$object_ids
  fi
}

verify_version() {
  local version=$1
  local digest=$2
  local release_commit=$3
  local certificate_identity
  local legacy_verification_json
  local verification_json

  certificate_identity="https://github.com/${source_repository}/${workflow_path}@refs/tags/${version}"
  if verification_json=$(cosign verify "ghcr.io/${ghcr_repository}@${digest}" \
    -a "release-artifact=${artifact}" \
    -a "release-commit=${release_commit}" \
    --certificate-oidc-issuer "$oidc_issuer" \
    --certificate-identity "$certificate_identity" \
    --output json 2>/dev/null) && \
    jq -e \
      --arg artifact "$artifact" \
      --arg commit "$release_commit" \
      --arg digest "$digest" '
        type == "array" and
        any(.[];
          (.critical.image["docker-manifest-digest"] == $digest) and
          (.optional | type == "object") and
          (.optional["release-artifact"] == $artifact) and
          (.optional["release-commit"] == $commit) and
          (.optional["release-run-id"] | type == "string" and test("^[1-9][0-9]*$")) and
          (.optional["release-run-attempt"] |
            type == "string" and test("^([1-9][0-9]{0,2}|1000)$"))
        )
      ' <<<"$verification_json" >/dev/null; then
    return 0
  fi

  # Older releases predate signed annotations. Their Fulcio certificates still
  # bind the exact image, tag commit, repository, and GitHub workflow.
  if ! legacy_verification_json=$(cosign verify "ghcr.io/${ghcr_repository}@${digest}" \
    --certificate-github-workflow-name "$workflow_name" \
    --certificate-github-workflow-ref "refs/tags/${version}" \
    --certificate-github-workflow-repository "$source_repository" \
    --certificate-github-workflow-sha "$release_commit" \
    --certificate-github-workflow-trigger push \
    --certificate-oidc-issuer "$oidc_issuer" \
    --certificate-identity "$certificate_identity" \
    --output json 2>/dev/null); then
    return 1
  fi
  jq -e \
    --arg digest "$digest" \
    --arg image "ghcr.io/${ghcr_repository}@${digest}" '
      type == "array" and
      any(.[];
        .critical.type == "https://sigstore.dev/cosign/sign/v1" and
        .critical.image["docker-manifest-digest"] == $digest and
        .critical.identity["docker-reference"] == $image and
        .optional == {}
      )
    ' <<<"$legacy_verification_json" >/dev/null
}

if [[ $explicit_version == true ]]; then
  if ! resolve_tag_commit "$reference"; then
    fail "source tag $reference is absent"
  fi
  if ! verify_version "$reference" "$reference_digest" "$resolved_tag_commit"; then
    fail "immutable tag ghcr.io/${ghcr_repository}:${reference} lacks trusted release provenance"
  fi
  echo "Validated immutable release floor ${reference} at ${reference_digest}." >&2
  emit_floor true "$reference" "$reference_digest"
  exit 0
fi

version_count=${#versions[@]}
if ((version_count == 0)); then
  fail "remote tags contain no stable release versions"
fi
for ((left_index = 0; left_index < version_count; left_index += 1)); do
  highest_index=$left_index
  for ((right_index = left_index + 1; right_index < version_count; right_index += 1)); do
    compare_versions "${versions[$right_index]}" "${versions[$highest_index]}"
    if ((version_comparison > 0)); then
      highest_index=$right_index
    fi
  done
  if ((highest_index != left_index)); then
    swap=${versions[$left_index]}
    versions[left_index]=${versions[highest_index]}
    versions[highest_index]=$swap
  fi
done

for version in "${versions[@]}"; do
  if ! resolve_tag_commit "$version"; then
    fail "source tag $version has no tag object"
  fi
  version_digest=$(
    "$digest_resolver" "$ghcr_repository" "$version"
  )
  if [[ $version_digest != "$reference_digest" ]]; then
    continue
  fi
  if verify_version "$version" "$reference_digest" "$resolved_tag_commit"; then
    echo "Validated mutable alias floor ${version} at ${reference_digest}." >&2
    emit_floor true "$version" "$reference_digest"
    exit 0
  fi
done

fail "mutable alias ghcr.io/${ghcr_repository}:${reference} has no trusted immutable release mapping"
