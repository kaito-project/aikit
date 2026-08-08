#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <release-version> <release-commit>" >&2
  exit 2
fi

release_version=$1
release_commit_input=$2
release_remote=${RELEASE_REMOTE:-origin}

fail() {
  echo "Release validation failed: $*" >&2
  exit 1
}

strip_matching_quotes() {
  local value=$1
  local first_character
  local last_character

  if [[ ${#value} -ge 2 ]]; then
    first_character=${value:0:1}
    last_character=${value: -1}
    if [[ $first_character == '"' && $last_character == '"' ]] || [[ $first_character == "'" && $last_character == "'" ]]; then
      value=${value:1:${#value}-2}
    fi
  fi

  printf '%s' "$value"
}

for required_command in git gh; do
  if ! command -v "$required_command" >/dev/null 2>&1; then
    echo "Required command not found: $required_command" >&2
    exit 2
  fi
done

if [[ -z ${GITHUB_REPOSITORY:-} ]]; then
  echo "GITHUB_REPOSITORY must be set to owner/repository" >&2
  exit 2
fi

if [[ -z ${GH_TOKEN:-} && -z ${GITHUB_TOKEN:-} ]]; then
  echo "GH_TOKEN or GITHUB_TOKEN must be set for release pull request validation" >&2
  exit 2
fi
if [[ -z ${GH_TOKEN:-} ]]; then
  export GH_TOKEN=$GITHUB_TOKEN
fi

if ! [[ $release_version =~ ^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$ ]]; then
  fail "version must use stable semantic version form vX.Y.Z: $release_version"
fi

release_branch="release-${BASH_REMATCH[1]}.${BASH_REMATCH[2]}"
expected_chart_version=${release_version#v}

if ! release_commit=$(git rev-parse --verify "${release_commit_input}^{commit}" 2>/dev/null); then
  fail "commit does not resolve to a Git commit: $release_commit_input"
fi

if ! makefile_content=$(git show "${release_commit}:Makefile"); then
  fail "Makefile is missing at $release_commit"
fi
makefile_version_count=$(awk '/^[[:space:]]*VERSION[[:space:]]*:=/ { count++ } END { print count + 0 }' <<<"$makefile_content")
if [[ $makefile_version_count -ne 1 ]]; then
  fail "Makefile must contain exactly one VERSION assignment; found $makefile_version_count"
fi
makefile_version=$(awk '
  /^[[:space:]]*VERSION[[:space:]]*:=/ {
    value = $0
    sub(/^[[:space:]]*VERSION[[:space:]]*:=[[:space:]]*/, "", value)
    sub(/[[:space:]]*$/, "", value)
    print value
  }
' <<<"$makefile_content")
if [[ $makefile_version != "$release_version" ]]; then
  fail "Makefile VERSION is ${makefile_version:-missing}; expected $release_version"
fi

if ! chart_content=$(git show "${release_commit}:charts/aikit/Chart.yaml"); then
  fail "charts/aikit/Chart.yaml is missing at $release_commit"
fi
chart_version_count=$(awk '/^version:[[:space:]]*/ { count++ } END { print count + 0 }' <<<"$chart_content")
chart_app_version_count=$(awk '/^appVersion:[[:space:]]*/ { count++ } END { print count + 0 }' <<<"$chart_content")
if [[ $chart_version_count -ne 1 ]]; then
  fail "Helm chart must contain exactly one top-level version; found $chart_version_count"
fi
if [[ $chart_app_version_count -ne 1 ]]; then
  fail "Helm chart must contain exactly one top-level appVersion; found $chart_app_version_count"
fi
chart_version=$(awk '
  /^version:[[:space:]]*/ {
    value = $0
    sub(/^version:[[:space:]]*/, "", value)
    sub(/[[:space:]]*$/, "", value)
    print value
  }
' <<<"$chart_content")
chart_app_version=$(awk '
  /^appVersion:[[:space:]]*/ {
    value = $0
    sub(/^appVersion:[[:space:]]*/, "", value)
    sub(/[[:space:]]*$/, "", value)
    print value
  }
' <<<"$chart_content")
chart_version=$(strip_matching_quotes "$chart_version")
chart_app_version=$(strip_matching_quotes "$chart_app_version")
if [[ $chart_version != "$expected_chart_version" ]]; then
  fail "Helm chart version is ${chart_version:-missing}; expected $expected_chart_version"
fi
if [[ $chart_app_version != "$release_version" ]]; then
  fail "Helm chart appVersion is ${chart_app_version:-missing}; expected $release_version"
fi

if ! git fetch --quiet --no-tags "$release_remote" "refs/heads/${release_branch}"; then
  fail "release branch does not exist or cannot be fetched: $release_branch"
fi
release_branch_commit=$(git rev-parse --verify 'FETCH_HEAD^{commit}')
if ! git merge-base --is-ancestor "$release_commit" "$release_branch_commit"; then
  fail "$release_commit is not reachable from $release_branch"
fi

if ! release_pr_numbers=$(gh api \
  --method GET \
  --paginate \
  "repos/${GITHUB_REPOSITORY}/issues" \
  -f state=closed \
  -f "labels=release-pr,${release_version}" \
  -f per_page=100 \
  --jq '.[] | select(.pull_request != null) | .number'); then
  fail "could not query release pull request candidates"
fi

if [[ -z $release_pr_numbers ]]; then
  fail "no closed pull request has labels release-pr and $release_version"
fi

matching_pr=
while read -r pr_number; do
  if [[ -z $pr_number ]]; then
    continue
  fi
  if ! [[ $pr_number =~ ^[0-9]+$ ]]; then
    fail "GitHub returned an invalid pull request number: $pr_number"
  fi
  if ! pr_details=$(gh api \
    "repos/${GITHUB_REPOSITORY}/pulls/${pr_number}" \
    --jq '[.merged, .base.ref, (.merge_commit_sha // "-")] | @tsv'); then
    fail "could not inspect release pull request #$pr_number"
  fi
  IFS=$'\t' read -r merged base_branch merge_commit <<<"$pr_details"
  if [[ $merged != true || $base_branch != "$release_branch" || $merge_commit == "-" ]]; then
    continue
  fi
  if git cat-file -e "${merge_commit}^{commit}" 2>/dev/null && git merge-base --is-ancestor "$merge_commit" "$release_commit"; then
    matching_pr=$pr_number
    break
  fi
done <<<"$release_pr_numbers"

if [[ -z $matching_pr ]]; then
  fail "no matching release pull request merge is an ancestor of $release_commit"
fi

echo "Validated $release_version at $release_commit on $release_branch using release pull request #$matching_pr."

if [[ -n ${GITHUB_OUTPUT:-} ]]; then
  {
    echo "release_branch=$release_branch"
    echo "release_commit=$release_commit"
    echo "release_pr=$matching_pr"
  } >>"$GITHUB_OUTPUT"
fi
