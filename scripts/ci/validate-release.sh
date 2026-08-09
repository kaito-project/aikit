#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <release-version> <release-commit> [trusted-guardrail-commit]" >&2
  exit 2
fi

release_version=$1
release_commit_input=$2
trusted_guardrail_commit_input=${3:-}
release_remote=${RELEASE_REMOTE:-origin}

fail() {
  echo "Release validation failed: $*" >&2
  exit 1
}

workflow_runs_are_valid() {
  local subject=$1
  local action_runs=$2
  local checks_ok=true
  local lint_succeeded=false
  local unit_test_succeeded=false
  local seen_workflow_paths=$'\n'
  local created_at
  local run_id
  local workflow_path
  local status
  local conclusion

  while IFS=$'\t' read -r created_at run_id workflow_path status conclusion; do
    if [[ -z $workflow_path ]]; then
      continue
    fi
    if [[ -z $created_at || ! $run_id =~ ^[0-9]+$ || $workflow_path != .github/workflows/* || -z $status || -z $conclusion ]]; then
      fail "GitHub returned invalid workflow run data for $subject"
    fi
    if [[ $seen_workflow_paths == *$'\n'"$workflow_path"$'\n'* ]]; then
      continue
    fi
    seen_workflow_paths+="${workflow_path}"$'\n'

    if [[ $status != completed || ! $conclusion =~ ^(success|neutral|skipped)$ ]]; then
      echo "$subject workflow $workflow_path is ${status}/${conclusion}; all latest workflow runs must complete successfully." >&2
      checks_ok=false
    fi
    if [[ $workflow_path == .github/workflows/lint.yaml && $status == completed && $conclusion == success ]]; then
      lint_succeeded=true
    fi
    if [[ $workflow_path == .github/workflows/unit-test.yaml && $status == completed && $conclusion == success ]]; then
      unit_test_succeeded=true
    fi
  done < <(LC_ALL=C sort -t $'\t' -k1,1r -k2,2nr <<<"$action_runs")

  if [[ $lint_succeeded != true || $unit_test_succeeded != true ]]; then
    echo "$subject must have successful lint and unit-test workflow runs." >&2
    checks_ok=false
  fi
  [[ $checks_ok == true ]]
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

manifest_validation_error=
manifest_makefile_version=
manifest_chart_version=
manifest_chart_app_version=
read_release_manifests() {
  local commit=$1
  local makefile_content
  local makefile_version_count
  local chart_content
  local chart_version_count
  local chart_app_version_count

  manifest_validation_error=
  manifest_makefile_version=
  manifest_chart_version=
  manifest_chart_app_version=
  if ! makefile_content=$(git show "${commit}:Makefile" 2>/dev/null); then
    manifest_validation_error="Makefile is missing at $commit"
    return 1
  fi
  makefile_version_count=$(awk '/^[[:space:]]*VERSION[[:space:]]*:=/ { count++ } END { print count + 0 }' <<<"$makefile_content")
  if [[ $makefile_version_count -ne 1 ]]; then
    manifest_validation_error="Makefile at $commit must contain exactly one VERSION assignment; found $makefile_version_count"
    return 1
  fi
  manifest_makefile_version=$(awk '
    /^[[:space:]]*VERSION[[:space:]]*:=/ {
      value = $0
      sub(/^[[:space:]]*VERSION[[:space:]]*:=[[:space:]]*/, "", value)
      sub(/[[:space:]]*$/, "", value)
      print value
    }
  ' <<<"$makefile_content")

  if ! chart_content=$(git show "${commit}:charts/aikit/Chart.yaml" 2>/dev/null); then
    manifest_validation_error="charts/aikit/Chart.yaml is missing at $commit"
    return 1
  fi
  chart_version_count=$(awk '/^version:[[:space:]]*/ { count++ } END { print count + 0 }' <<<"$chart_content")
  chart_app_version_count=$(awk '/^appVersion:[[:space:]]*/ { count++ } END { print count + 0 }' <<<"$chart_content")
  if [[ $chart_version_count -ne 1 ]]; then
    manifest_validation_error="Helm chart at $commit must contain exactly one top-level version; found $chart_version_count"
    return 1
  fi
  if [[ $chart_app_version_count -ne 1 ]]; then
    manifest_validation_error="Helm chart at $commit must contain exactly one top-level appVersion; found $chart_app_version_count"
    return 1
  fi
  manifest_chart_version=$(awk '
    /^version:[[:space:]]*/ {
      value = $0
      sub(/^version:[[:space:]]*/, "", value)
      sub(/[[:space:]]*$/, "", value)
      print value
    }
  ' <<<"$chart_content")
  manifest_chart_app_version=$(awk '
    /^appVersion:[[:space:]]*/ {
      value = $0
      sub(/^appVersion:[[:space:]]*/, "", value)
      sub(/[[:space:]]*$/, "", value)
      print value
    }
  ' <<<"$chart_content")
  manifest_chart_version=$(strip_matching_quotes "$manifest_chart_version")
  manifest_chart_app_version=$(strip_matching_quotes "$manifest_chart_app_version")
}

release_manifests_match() {
  local commit=$1

  if ! read_release_manifests "$commit"; then
    return 1
  fi
  if [[ $manifest_makefile_version != "$release_version" ]]; then
    manifest_validation_error="Makefile VERSION at $commit is ${manifest_makefile_version:-missing}; expected $release_version"
    return 1
  fi
  if [[ $manifest_chart_version != "$expected_chart_version" ]]; then
    manifest_validation_error="Helm chart version at $commit is ${manifest_chart_version:-missing}; expected $expected_chart_version"
    return 1
  fi
  if [[ $manifest_chart_app_version != "$release_version" ]]; then
    manifest_validation_error="Helm chart appVersion at $commit is ${manifest_chart_app_version:-missing}; expected $release_version"
    return 1
  fi
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

if [[ -n $trusted_guardrail_commit_input ]]; then
  if ! trusted_guardrail_commit=$(git rev-parse --verify "${trusted_guardrail_commit_input}^{commit}" 2>/dev/null); then
    fail "trusted guardrail commit does not resolve to a Git commit: $trusted_guardrail_commit_input"
  fi
  release_guardrail_paths=(
    .github/workflows
    scripts/ci/check-image-size.sh
    scripts/ci/validate-release-version.sh
    scripts/ci/validate-release.sh
  )
  for guardrail_path in "${release_guardrail_paths[@]}"; do
    if ! git cat-file -e "${trusted_guardrail_commit}:${guardrail_path}" 2>/dev/null; then
      fail "trusted release guardrail is missing at $trusted_guardrail_commit: $guardrail_path"
    fi
    if ! git cat-file -e "${release_commit}:${guardrail_path}" 2>/dev/null; then
      fail "release commit is missing trusted release guardrail: $guardrail_path"
    fi
  done
  if ! git diff --quiet --no-ext-diff --no-textconv \
    "$trusted_guardrail_commit" "$release_commit" -- "${release_guardrail_paths[@]}"; then
    changed_guardrails=$(git diff --name-only --no-ext-diff --no-textconv \
      "$trusted_guardrail_commit" "$release_commit" -- "${release_guardrail_paths[@]}")
    fail "release commit does not contain the trusted publisher guardrails from $trusted_guardrail_commit: ${changed_guardrails//$'\n'/, }"
  fi
fi

if ! release_manifests_match "$release_commit"; then
  fail "$manifest_validation_error"
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
  -f labels=release-pr \
  -f per_page=100 \
  --jq '.[] | select(.pull_request != null) | .number'); then
  fail "could not query release pull request candidates"
fi

if [[ -z $release_pr_numbers ]]; then
  fail "no closed pull request has the release-pr label"
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
    --jq '[.merged, .base.ref, (.base.sha // "-"), (.merge_commit_sha // "-"), (.head.sha // "-"), (.head.ref // "-"), (.head.repo.full_name // "-")] | @tsv'); then
    fail "could not inspect release pull request #$pr_number"
  fi
  IFS=$'\t' read -r merged base_branch base_commit merge_commit head_commit head_branch head_repository <<<"$pr_details"
  if [[ $merged != true ||
    $base_branch != "$release_branch" ||
    ( $head_branch != "prepare-${release_version}" && $head_branch != "release-${release_version}" ) ||
    $head_repository != "$GITHUB_REPOSITORY" ||
    $base_commit == "-" ||
    $merge_commit == "-" ||
    $head_commit == "-" ]]; then
    continue
  fi
  if ! [[ $base_commit =~ ^[0-9a-f]{40}$ && $merge_commit =~ ^[0-9a-f]{40}$ && $head_commit =~ ^[0-9a-f]{40}$ ]]; then
    fail "GitHub returned an invalid base, merge, or head commit for release pull request #$pr_number"
  fi
  if ! git cat-file -e "${base_commit}^{commit}" 2>/dev/null || \
    ! git cat-file -e "${merge_commit}^{commit}" 2>/dev/null || \
    ! git merge-base --is-ancestor "$base_commit" "$merge_commit" || \
    ! git merge-base --is-ancestor "$merge_commit" "$release_commit"; then
    continue
  fi

  if ! changed_files=$(gh api \
    --method GET \
    --paginate \
    "repos/${GITHUB_REPOSITORY}/pulls/${pr_number}/files" \
    -f per_page=100 \
    --jq '.[].filename'); then
    fail "could not inspect changed files for release pull request #$pr_number"
  fi
  makefile_changed=false
  chart_changed=false
  while IFS= read -r changed_file; do
    case $changed_file in
      Makefile) makefile_changed=true ;;
      charts/aikit/Chart.yaml) chart_changed=true ;;
    esac
  done <<<"$changed_files"
  if [[ $makefile_changed != true || $chart_changed != true ]]; then
    echo "Release pull request #$pr_number did not change both release manifests." >&2
    continue
  fi
  if ! read_release_manifests "$base_commit"; then
    echo "Release pull request #$pr_number has invalid base manifests: $manifest_validation_error." >&2
    continue
  fi
  if [[ $manifest_makefile_version == "$release_version" || \
    $manifest_chart_version == "$expected_chart_version" || \
    $manifest_chart_app_version == "$release_version" ]]; then
    echo "Release pull request #$pr_number did not introduce every version field for $release_version." >&2
    continue
  fi
  if ! release_manifests_match "$merge_commit"; then
    echo "Release pull request #$pr_number does not establish $release_version: $manifest_validation_error." >&2
    continue
  fi

  if ! action_runs=$(gh api \
    --method GET \
    --paginate \
    "repos/${GITHUB_REPOSITORY}/actions/runs" \
    -f event=pull_request \
    -f "head_sha=${head_commit}" \
    -f per_page=100 \
    --jq ".workflow_runs[] | select(.event == \"pull_request\" and .head_sha == \"${head_commit}\" and any(.pull_requests[]?; .number == ${pr_number})) | [.created_at, .id, .path, .status, (.conclusion // \"-\")] | @tsv"); then
    fail "could not inspect workflow runs for release pull request #$pr_number"
  fi

  if workflow_runs_are_valid "Release pull request #$pr_number" "$action_runs"; then
    matching_pr=$pr_number
    break
  fi
done <<<"$release_pr_numbers"

if [[ -z $matching_pr ]]; then
  fail "no checked, matching release pull request merge is an ancestor of $release_commit"
fi

if ! release_action_runs=$(gh api \
  --method GET \
  --paginate \
  "repos/${GITHUB_REPOSITORY}/actions/runs" \
  -f event=push \
  -f "branch=${release_branch}" \
  -f "head_sha=${release_commit}" \
  -f per_page=100 \
  --jq ".workflow_runs[] | select(.event == \"push\" and .head_branch == \"${release_branch}\" and .head_sha == \"${release_commit}\") | [.created_at, .id, .path, .status, (.conclusion // \"-\")] | @tsv"); then
  fail "could not inspect branch workflow runs for release commit $release_commit"
fi
if ! workflow_runs_are_valid "Release commit $release_commit on $release_branch" "$release_action_runs"; then
  fail "release commit $release_commit does not have successful branch CI"
fi

echo "Validated $release_version at $release_commit on $release_branch using release pull request #$matching_pr."

if [[ -n ${GITHUB_OUTPUT:-} ]]; then
  {
    echo "release_branch=$release_branch"
    echo "release_commit=$release_commit"
    echo "release_pr=$matching_pr"
  } >>"$GITHUB_OUTPUT"
fi
