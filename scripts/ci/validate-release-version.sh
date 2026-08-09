#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <release-version> <new|existing|either> [<release-branch-commit|missing>]" >&2
  exit 2
fi

release_version=$1
expected_state=$2
release_branch_commit=${3:-}
release_remote=${RELEASE_REMOTE:-origin}
export LC_ALL=C

fail() {
  echo "Release version validation failed: $*" >&2
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

if ! parse_version "$release_version"; then
  fail "version must use stable semantic version form vX.Y.Z: $release_version"
fi
if ((${#release_version} > 128)); then
  fail "version exceeds the 128-character OCI tag limit: $release_version"
fi
release_major=$parsed_major
release_minor=$parsed_minor

case $expected_state in
  new | existing | either) ;;
  *)
    echo "Expected state must be new, existing, or either: $expected_state" >&2
    exit 2
    ;;
esac

if ! remote_tags=$(git ls-remote --refs --tags "$release_remote" 'refs/tags/v*'); then
  fail "could not list release tags from $release_remote"
fi

tag_exists=false
latest_stable=
latest_release_line=
while read -r _ tag_ref; do
  if [[ -z ${tag_ref:-} ]]; then
    continue
  fi
  tag=${tag_ref#refs/tags/}
  if ! parse_version "$tag"; then
    continue
  fi
  tag_major=$parsed_major
  tag_minor=$parsed_minor

  if [[ $tag == "$release_version" ]]; then
    tag_exists=true
  fi
  if [[ -z $latest_stable ]]; then
    latest_stable=$tag
  else
    compare_versions "$tag" "$latest_stable"
    if ((version_comparison > 0)); then
      latest_stable=$tag
    fi
  fi
  if [[ $tag_major == "$release_major" && $tag_minor == "$release_minor" ]]; then
    if [[ -z $latest_release_line ]]; then
      latest_release_line=$tag
    else
      compare_versions "$tag" "$latest_release_line"
      if ((version_comparison > 0)); then
        latest_release_line=$tag
      fi
    fi
  fi
done <<<"$remote_tags"

if [[ $expected_state == new && $tag_exists == true ]]; then
  fail "$release_version already exists"
fi
if [[ $expected_state == existing && $tag_exists != true ]]; then
  fail "$release_version does not exist"
fi
if [[ $tag_exists != true && -n $latest_release_line ]]; then
  compare_versions "$release_version" "$latest_release_line"
  if ((version_comparison <= 0)); then
    fail "$release_version must be newer than $latest_release_line on its release line"
  fi
fi

resolved_release_branch_commit=
if [[ -n $release_branch_commit && $release_branch_commit != missing ]]; then
  if ! resolved_release_branch_commit=$(git rev-parse "${release_branch_commit}^{commit}" 2>/dev/null); then
    fail "release branch commit does not resolve to a commit: $release_branch_commit"
  fi
fi

if [[ -n $release_branch_commit && -n $latest_release_line ]]; then
  if [[ $release_branch_commit == missing ]]; then
    fail "release branch is missing for previously released line $latest_release_line; restore it from historical ancestry"
  fi
  if ! git fetch --quiet --force --no-tags "$release_remote" \
    "refs/tags/${latest_release_line}"; then
    fail "could not fetch latest release-line tag $latest_release_line from $release_remote"
  fi
  if ! latest_release_commit=$(git rev-parse 'FETCH_HEAD^{commit}' 2>/dev/null); then
    fail "latest release-line tag $latest_release_line does not resolve to a commit"
  fi
  if ! git merge-base --is-ancestor "$latest_release_commit" "$resolved_release_branch_commit"; then
    fail "release branch commit $resolved_release_branch_commit does not descend from $latest_release_line at $latest_release_commit"
  fi
fi

publish_latest=true
if [[ -n $latest_stable ]]; then
  compare_versions "$release_version" "$latest_stable"
  if ((version_comparison < 0)); then
    publish_latest=false
  fi
fi

printf 'Validated %s (%s); latest stable tag is %s; publish latest: %s.\n' \
  "$release_version" "$expected_state" "${latest_stable:-none}" "$publish_latest"

if [[ -n ${GITHUB_OUTPUT:-} ]]; then
  {
    echo "tag_exists=$tag_exists"
    echo "latest_stable=${latest_stable:-none}"
    echo "latest_release_line=${latest_release_line:-none}"
    echo "publish_latest=$publish_latest"
  } >>"$GITHUB_OUTPUT"
fi
