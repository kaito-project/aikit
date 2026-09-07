#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "Usage: $0 <owner/repository> <workflow-path> <runs-json> <remote-tags> [target-version]" >&2
  exit 2
fi

repository=$1
workflow_path=$2
runs_json=$3
remote_tags=$4
target_version=${5:-}
export LC_ALL=C

fail() {
  echo "Latest release selection failed: $*" >&2
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

if ! [[ $repository =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]]; then
  fail "invalid repository: $repository"
fi
if ! [[ $workflow_path =~ ^\.github/workflows/[A-Za-z0-9_.-]+\.ya?ml$ ]]; then
  fail "invalid workflow path: $workflow_path"
fi
if [[ -n $target_version ]] && ! parse_version "$target_version"; then
  fail "invalid target version: $target_version"
fi
if [[ ! -f $runs_json || ! -f $remote_tags ]]; then
  fail "run metadata and remote tag inputs must be regular files"
fi
if ! jq -e '
  type == "array" and
  all(.[]; type == "object" and (.workflow_runs | type == "array"))
' "$runs_json" >/dev/null; then
  fail "run metadata is not a paginated Actions workflow-runs response"
fi

tag_records=$(mktemp)
trap 'rm -f "$tag_records"' EXIT
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

  if [[ -n $peeled ]]; then
    printf '%s\tpeeled\t%s\n' "$version" "$object_id" >>"$tag_records"
  else
    printf '%s\tobject\t%s\n' "$version" "$object_id" >>"$tag_records"
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

best_version=
best_commit=
best_run_id=
best_run_attempt=
while IFS=$'\t' read -r version head_sha run_id run_attempt; do
  if ! parse_version "$version"; then
    continue
  fi
  if ! [[ $head_sha =~ ^[0-9a-f]{40}$ && $run_id =~ ^[1-9][0-9]*$ && $run_attempt =~ ^[1-9][0-9]*$ ]]; then
    fail "Actions returned malformed successful release metadata"
  fi
  if ! resolve_tag_commit "$version" || [[ $resolved_tag_commit != "$head_sha" ]]; then
    continue
  fi

  if [[ -z $best_version ]]; then
    best_version=$version
    best_commit=$head_sha
    best_run_id=$run_id
    best_run_attempt=$run_attempt
    continue
  fi
  compare_versions "$version" "$best_version"
  replace=false
  if ((version_comparison > 0)); then
    replace=true
  elif ((version_comparison == 0)); then
    if [[ $run_id != "$best_run_id" ]]; then
      fail "multiple successful workflow runs exist for $version"
    fi
    compare_component "$run_attempt" "$best_run_attempt"
    if ((component_comparison < 0)); then
      replace=true
    fi
  fi
  if [[ $replace == true ]]; then
    best_version=$version
    best_commit=$head_sha
    best_run_id=$run_id
    best_run_attempt=$run_attempt
  fi
done < <(
  jq -r \
    --arg repository "$repository" \
    --arg workflow_path "$workflow_path" \
    --arg target_version "$target_version" '
      .[] | .workflow_runs[] |
      select(
        .path == $workflow_path and
        .event == "push" and
        .status == "completed" and
        .conclusion == "success" and
        .head_repository.full_name == $repository and
        ($target_version == "" or .head_branch == $target_version) and
        (.head_branch | type == "string") and
        (.head_sha | type == "string") and
        (.id | type == "number") and
        (.run_attempt | type == "number")
      ) |
      [.head_branch, .head_sha, .id, .run_attempt] | @tsv
    ' "$runs_json"
)

found=false
if [[ -n $best_version ]]; then
  found=true
fi

printf 'Selected latest successful release: %s.\n' "${best_version:-none}"
if [[ -n ${GITHUB_OUTPUT:-} ]]; then
  {
    echo "found=$found"
    echo "version=${best_version:-none}"
    echo "commit=${best_commit:-none}"
    echo "run_id=${best_run_id:-none}"
    echo "run_attempt=${best_run_attempt:-none}"
  } >>"$GITHUB_OUTPUT"
fi
