#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
validator="$script_dir/validate-release.sh"
work_dir=$(mktemp -d)
trap 'rm -rf "$work_dir"' EXIT
mkdir -p "$work_dir/bin"

cat >"$work_dir/bin/gh" <<'GH'
#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-} == api ]]; then
  case " $* " in
    *"/issues "*)
      if [[ " $* " != *" -f labels=release-pr "* ]]; then
        echo "release pull request lookup must use only the stable release-pr label" >&2
        exit 1
      fi
      printf '%s' "${FAKE_GH_ISSUES:-}"
      exit 0
      ;;
    *"/pulls/"*"/files "*)
      printf '%s' "${FAKE_GH_PR_FILES:-}"
      exit 0
      ;;
    *"/pulls/"*)
      printf '%s' "${FAKE_GH_PR_DETAILS:-}"
      exit 0
      ;;
    *"/actions/runs "*)
      if [[ " $* " == *" -f event=push "* ]]; then
        if [[ " $* " != *" -f branch=${FAKE_EXPECTED_RELEASE_BRANCH} "* ||
          " $* " != *" -f head_sha=${FAKE_EXPECTED_RELEASE_COMMIT} "* ||
          " $* " != *".event == \"push\""* ||
          " $* " != *".head_branch == \"${FAKE_EXPECTED_RELEASE_BRANCH}\""* ||
          " $* " != *".head_sha == \"${FAKE_EXPECTED_RELEASE_COMMIT}\""* ]]; then
          echo "release commit lookup must bind push runs to the exact branch and commit" >&2
          exit 1
        fi
        printf '%s' "${FAKE_GH_RELEASE_ACTION_RUNS:-}"
      else
        if [[ " $* " != *" -f event=pull_request "* ||
          " $* " != *" -f head_sha=${FAKE_EXPECTED_PR_HEAD} "* ||
          " $* " != *".event == \"pull_request\""* ||
          " $* " != *".head_sha == \"${FAKE_EXPECTED_PR_HEAD}\""* ]]; then
          echo "release pull request lookup must bind runs to the exact head commit" >&2
          exit 1
        fi
        printf '%s' "${FAKE_GH_PR_ACTION_RUNS:-}"
      fi
      exit 0
      ;;
  esac
fi
echo "unexpected fake gh invocation: $*" >&2
exit 1
GH
chmod +x "$work_dir/bin/gh"

git init --quiet --bare "$work_dir/origin.git"
git init --quiet -b main "$work_dir/repository"
git -C "$work_dir/repository" config user.email test@example.com
git -C "$work_dir/repository" config user.name "Release Validator Test"
git -C "$work_dir/repository" remote add origin "$work_dir/origin.git"

guardrail_files=(
  .github/workflows/lint.yaml
  .github/workflows/release-runners.yaml
  .github/workflows/release.yaml
  .github/workflows/unit-test.yaml
  scripts/ci/check-image-size.sh
  scripts/ci/validate-release-version.sh
  scripts/ci/validate-release.sh
)
mkdir -p \
  "$work_dir/repository/.github/workflows" \
  "$work_dir/repository/charts/aikit" \
  "$work_dir/repository/scripts/ci"
printf '%s\n' 'VERSION := v1.2.2' >"$work_dir/repository/Makefile"
printf '%s\n' 'version: 1.2.2' 'appVersion: v1.2.2' >"$work_dir/repository/charts/aikit/Chart.yaml"
for guardrail_path in "${guardrail_files[@]}"; do
  printf 'trusted guardrail fixture: %s\n' "$guardrail_path" >"$work_dir/repository/$guardrail_path"
done
git -C "$work_dir/repository" add Makefile charts/aikit/Chart.yaml "${guardrail_files[@]}"
git -C "$work_dir/repository" commit --quiet -m "initial version"
initial_commit=$(git -C "$work_dir/repository" rev-parse HEAD)
git -C "$work_dir/repository" push --quiet --set-upstream origin main

git -C "$work_dir/repository" checkout --quiet -b release-1.2
printf '%s\n' 'unrelated release branch change' >"$work_dir/repository/notes.txt"
git -C "$work_dir/repository" add notes.txt
git -C "$work_dir/repository" commit --quiet -m "update release notes"
unrelated_ancestor_commit=$(git -C "$work_dir/repository" rev-parse HEAD)

printf '%s\n' 'VERSION := v1.2.3' >"$work_dir/repository/Makefile"
printf '%s\n' 'version: 1.2.3' 'appVersion: v1.2.3' >"$work_dir/repository/charts/aikit/Chart.yaml"
git -C "$work_dir/repository" add Makefile charts/aikit/Chart.yaml
git -C "$work_dir/repository" commit --quiet -m "prepare v1.2.3"
release_pr_commit=$(git -C "$work_dir/repository" rev-parse HEAD)

printf '%s\n' '# release metadata' >>"$work_dir/repository/Makefile"
printf '%s\n' '# release metadata' >>"$work_dir/repository/charts/aikit/Chart.yaml"
git -C "$work_dir/repository" add Makefile charts/aikit/Chart.yaml
git -C "$work_dir/repository" commit --quiet -m "document release metadata"
harmless_manifest_commit=$(git -C "$work_dir/repository" rev-parse HEAD)

printf '%s\n' 'post-preparation fix' >"$work_dir/repository/fix.txt"
git -C "$work_dir/repository" add fix.txt
git -C "$work_dir/repository" commit --quiet -m "fix release candidate"
release_commit=$(git -C "$work_dir/repository" rev-parse HEAD)
git -C "$work_dir/repository" push --quiet --set-upstream origin release-1.2

valid_action_runs=$(printf '%s\n%s\n%s\n%s' \
  $'2026-08-08T20:00:02Z\t102\t.github/workflows/lint.yaml\tcompleted\tsuccess' \
  $'2026-08-08T20:00:01Z\t101\t.github/workflows/unit-test.yaml\tcompleted\tsuccess' \
  $'2026-08-08T19:00:00Z\t100\t.github/workflows/lint.yaml\tcompleted\tfailure' \
  $'2026-08-08T18:00:00Z\t99\t.github/workflows/dependabot.yaml\tcompleted\tskipped')
valid_pr_files=$(printf '%s\n%s' Makefile charts/aikit/Chart.yaml)

run_validator() {
  local version=$1
  local commit=$2
  local release_pr_issues=$3
  local release_pr_details=$4
  local release_pr_action_runs=${5-$valid_action_runs}
  local release_pr_files=${6-$valid_pr_files}
  local release_commit_action_runs=${7-$valid_action_runs}
  local trusted_guardrail_commit=${8-}
  local -a validator_args=("$version" "$commit")
  if [[ -n $trusted_guardrail_commit ]]; then
    validator_args+=("$trusted_guardrail_commit")
  fi
  (
    cd "$work_dir/repository"
    PATH="$work_dir/bin:$PATH" \
      FAKE_GH_ISSUES="$release_pr_issues" \
      FAKE_GH_PR_DETAILS="$release_pr_details" \
      FAKE_GH_PR_FILES="$release_pr_files" \
      FAKE_GH_PR_ACTION_RUNS="$release_pr_action_runs" \
      FAKE_GH_RELEASE_ACTION_RUNS="$release_commit_action_runs" \
      FAKE_EXPECTED_PR_HEAD="$release_pr_commit" \
      FAKE_EXPECTED_RELEASE_BRANCH=release-1.2 \
      FAKE_EXPECTED_RELEASE_COMMIT="$commit" \
      GH_TOKEN=test-token \
      GITHUB_REPOSITORY=example/aikit \
      "$validator" "${validator_args[@]}"
  )
}

format_pr_details() {
  local base_commit=$1
  local merge_commit=$2
  local head_commit=$3
  local head_branch=${4:-prepare-v1.2.3}
  local head_repository=${5:-example/aikit}

  printf 'true\trelease-1.2\t%s\t%s\t%s\t%s\t%s' \
    "$base_commit" "$merge_commit" "$head_commit" "$head_branch" "$head_repository"
}

valid_pr_details=$(format_pr_details \
  "$unrelated_ancestor_commit" "$release_pr_commit" "$release_pr_commit")
run_validator v1.2.3 "$release_commit" 42 "$valid_pr_details" >/dev/null
run_validator \
  v1.2.3 "$release_commit" 42 "$valid_pr_details" "$valid_action_runs" "$valid_pr_files" "$valid_action_runs" "$release_commit" \
  >/dev/null

git -C "$work_dir/repository" checkout --quiet -b trusted-guardrails "$release_commit"
printf '%s\n' 'trusted publisher update' >>"$work_dir/repository/.github/workflows/release.yaml"
git -C "$work_dir/repository" add .github/workflows/release.yaml
git -C "$work_dir/repository" commit --quiet -m "update trusted publisher guardrail"
trusted_guardrail_commit=$(git -C "$work_dir/repository" rev-parse HEAD)
if run_validator \
  v1.2.3 "$release_commit" 42 "$valid_pr_details" "$valid_action_runs" "$valid_pr_files" "$valid_action_runs" "$trusted_guardrail_commit" \
  >/dev/null 2>&1; then
  echo "release commit with stale publisher guardrails unexpectedly passed" >&2
  exit 1
fi

git -C "$work_dir/repository" checkout --quiet -b missing-guardrail "$release_commit"
git -C "$work_dir/repository" rm --quiet scripts/ci/validate-release-version.sh
git -C "$work_dir/repository" commit --quiet -m "remove release guardrail"
missing_guardrail_commit=$(git -C "$work_dir/repository" rev-parse HEAD)
if run_validator \
  v1.2.3 "$missing_guardrail_commit" 42 "$valid_pr_details" "$valid_action_runs" "$valid_pr_files" "$valid_action_runs" "$release_commit" \
  >/dev/null 2>&1; then
  echo "release commit missing a trusted publisher guardrail unexpectedly passed" >&2
  exit 1
fi

git -C "$work_dir/repository" checkout --quiet -b added-tag-workflow "$release_commit"
printf '%s\n' 'on: push' >"$work_dir/repository/.github/workflows/untrusted.yaml"
git -C "$work_dir/repository" add .github/workflows/untrusted.yaml
git -C "$work_dir/repository" commit --quiet -m "add untrusted tag workflow"
added_tag_workflow_commit=$(git -C "$work_dir/repository" rev-parse HEAD)
if run_validator \
  v1.2.3 "$added_tag_workflow_commit" 42 "$valid_pr_details" "$valid_action_runs" "$valid_pr_files" "$valid_action_runs" "$release_commit" \
  >/dev/null 2>&1; then
  echo "release commit with an untrusted workflow unexpectedly passed" >&2
  exit 1
fi

legacy_pr_details=$(format_pr_details \
  "$unrelated_ancestor_commit" "$release_pr_commit" "$release_pr_commit" release-v1.2.3)
run_validator v1.2.3 "$release_commit" 42 "$legacy_pr_details" >/dev/null

wrong_head_branch_pr_details=$(format_pr_details \
  "$unrelated_ancestor_commit" "$release_pr_commit" "$release_pr_commit" prepare-v1.2.4)
if run_validator v1.2.3 "$release_commit" 42 "$wrong_head_branch_pr_details" >/dev/null 2>&1; then
  echo "release pull request from an unexpected head branch unexpectedly passed" >&2
  exit 1
fi

wrong_head_repository_pr_details=$(format_pr_details \
  "$unrelated_ancestor_commit" "$release_pr_commit" "$release_pr_commit" prepare-v1.2.3 fork/aikit)
if run_validator v1.2.3 "$release_commit" 42 "$wrong_head_repository_pr_details" >/dev/null 2>&1; then
  echo "release pull request from another repository unexpectedly passed" >&2
  exit 1
fi

if run_validator v1.2.3 "$release_commit" 42 "$valid_pr_details" "" >/dev/null 2>&1; then
  echo "release pull request without checks unexpectedly passed" >&2
  exit 1
fi

failed_action_runs=$(printf '%s\n%s\n%s' \
  $'2026-08-08T20:00:02Z\t102\t.github/workflows/lint.yaml\tcompleted\tsuccess' \
  $'2026-08-08T20:00:01Z\t101\t.github/workflows/unit-test.yaml\tcompleted\tfailure' \
  $'2026-08-08T19:00:00Z\t100\t.github/workflows/unit-test.yaml\tcompleted\tsuccess')
if run_validator v1.2.3 "$release_commit" 42 "$valid_pr_details" "$failed_action_runs" >/dev/null 2>&1; then
  echo "release pull request with a failed required workflow unexpectedly passed" >&2
  exit 1
fi
if run_validator \
  v1.2.3 "$release_commit" 42 "$valid_pr_details" "$valid_action_runs" "$valid_pr_files" "$failed_action_runs" \
  >/dev/null 2>&1; then
  echo "release commit with a failed required workflow unexpectedly passed" >&2
  exit 1
fi
if run_validator \
  v1.2.3 "$release_commit" 42 "$valid_pr_details" "$valid_action_runs" "$valid_pr_files" "" \
  >/dev/null 2>&1; then
  echo "release commit without checks unexpectedly passed" >&2
  exit 1
fi

pending_action_runs=$(printf '%s\n%s' \
  $'2026-08-08T20:00:02Z\t102\t.github/workflows/lint.yaml\tin_progress\t-' \
  $'2026-08-08T20:00:01Z\t101\t.github/workflows/unit-test.yaml\tcompleted\tsuccess')
if run_validator v1.2.3 "$release_commit" 42 "$valid_pr_details" "$pending_action_runs" >/dev/null 2>&1; then
  echo "release pull request with an unfinished workflow unexpectedly passed" >&2
  exit 1
fi
if run_validator \
  v1.2.3 "$release_commit" 42 "$valid_pr_details" "$valid_action_runs" "$valid_pr_files" "$pending_action_runs" \
  >/dev/null 2>&1; then
  echo "release commit with an unfinished workflow unexpectedly passed" >&2
  exit 1
fi

failed_optional_action_runs=$(printf '%s\n%s\n%s' \
  $'2026-08-08T20:00:03Z\t103\t.github/workflows/docker-test.yaml\tcompleted\tfailure' \
  $'2026-08-08T20:00:02Z\t102\t.github/workflows/lint.yaml\tcompleted\tsuccess' \
  $'2026-08-08T20:00:01Z\t101\t.github/workflows/unit-test.yaml\tcompleted\tsuccess')
if run_validator v1.2.3 "$release_commit" 42 "$valid_pr_details" "$failed_optional_action_runs" >/dev/null 2>&1; then
  echo "release pull request with another failed workflow unexpectedly passed" >&2
  exit 1
fi
if run_validator \
  v1.2.3 "$release_commit" 42 "$valid_pr_details" "$valid_action_runs" "$valid_pr_files" "$failed_optional_action_runs" \
  >/dev/null 2>&1; then
  echo "release commit with another failed workflow unexpectedly passed" >&2
  exit 1
fi

if run_validator v1.2.4 "$release_commit" 42 "$valid_pr_details" >/dev/null 2>&1; then
  echo "mismatched version unexpectedly passed" >&2
  exit 1
fi

git -C "$work_dir/repository" checkout --quiet -b side-release "$release_pr_commit"
printf '%s\n' 'not on the release branch' >"$work_dir/repository/side.txt"
git -C "$work_dir/repository" add side.txt
git -C "$work_dir/repository" commit --quiet -m "side release commit"
side_commit=$(git -C "$work_dir/repository" rev-parse HEAD)

if run_validator v1.2.3 "$side_commit" 42 "$valid_pr_details" >/dev/null 2>&1; then
  echo "commit outside the release branch unexpectedly passed" >&2
  exit 1
fi

unrelated_pr_details=$(format_pr_details "$release_pr_commit" "$side_commit" "$side_commit")
if run_validator v1.2.3 "$release_commit" 43 "$unrelated_pr_details" >/dev/null 2>&1; then
  echo "unrelated release pull request unexpectedly passed" >&2
  exit 1
fi

missing_manifest_pr_details=$(format_pr_details \
  "$unrelated_ancestor_commit" "$release_pr_commit" "$release_pr_commit")
if run_validator v1.2.3 "$release_commit" 44 "$missing_manifest_pr_details" "$valid_action_runs" Makefile >/dev/null 2>&1; then
  echo "release pull request missing a release manifest unexpectedly passed" >&2
  exit 1
fi

relabeled_ancestor_pr_details=$(format_pr_details \
  "$initial_commit" "$unrelated_ancestor_commit" "$unrelated_ancestor_commit")
if run_validator v1.2.3 "$release_commit" 45 "$relabeled_ancestor_pr_details" "$valid_action_runs" "$valid_pr_files" >/dev/null 2>&1; then
  echo "relabeled ancestor pull request with old manifest values unexpectedly passed" >&2
  exit 1
fi

harmless_manifest_pr_details=$(format_pr_details \
  "$release_pr_commit" "$harmless_manifest_commit" "$harmless_manifest_commit")
if run_validator v1.2.3 "$release_commit" 46 "$harmless_manifest_pr_details" "$valid_action_runs" "$valid_pr_files" >/dev/null 2>&1; then
  echo "pull request that preserved existing release values unexpectedly passed" >&2
  exit 1
fi

if run_validator v1.2.3 "$release_commit" "" "$valid_pr_details" >/dev/null 2>&1; then
  echo "missing release pull request unexpectedly passed" >&2
  exit 1
fi

if run_validator 1.2.3 "$release_commit" 42 "$valid_pr_details" >/dev/null 2>&1; then
  echo "invalid release version unexpectedly passed" >&2
  exit 1
fi

if run_validator v01.2.3 "$release_commit" 42 "$valid_pr_details" >/dev/null 2>&1; then
  echo "release version with a leading zero unexpectedly passed" >&2
  exit 1
fi

git -C "$work_dir/repository" checkout --quiet -b duplicate-makefile "$release_commit"
printf '%s\n' 'VERSION := v9.9.9' >>"$work_dir/repository/Makefile"
git -C "$work_dir/repository" add Makefile
git -C "$work_dir/repository" commit --quiet -m "duplicate Makefile version"
duplicate_makefile_commit=$(git -C "$work_dir/repository" rev-parse HEAD)
if run_validator v1.2.3 "$duplicate_makefile_commit" 42 "$valid_pr_details" >/dev/null 2>&1; then
  echo "duplicate Makefile VERSION unexpectedly passed" >&2
  exit 1
fi

git -C "$work_dir/repository" checkout --quiet -b duplicate-chart "$release_commit"
printf '%s\n' 'appVersion: v9.9.9' >>"$work_dir/repository/charts/aikit/Chart.yaml"
git -C "$work_dir/repository" add charts/aikit/Chart.yaml
git -C "$work_dir/repository" commit --quiet -m "duplicate chart appVersion"
duplicate_chart_commit=$(git -C "$work_dir/repository" rev-parse HEAD)
if run_validator v1.2.3 "$duplicate_chart_commit" 42 "$valid_pr_details" >/dev/null 2>&1; then
  echo "duplicate chart appVersion unexpectedly passed" >&2
  exit 1
fi

git -C "$work_dir/repository" checkout --quiet -b trailing-makefile-value "$release_commit"
printf '%s\n' 'VERSION := v1.2.3 unexpected' >"$work_dir/repository/Makefile"
git -C "$work_dir/repository" add Makefile
git -C "$work_dir/repository" commit --quiet -m "add trailing Makefile value"
trailing_makefile_commit=$(git -C "$work_dir/repository" rev-parse HEAD)
if run_validator v1.2.3 "$trailing_makefile_commit" 42 "$valid_pr_details" >/dev/null 2>&1; then
  echo "trailing Makefile VERSION value unexpectedly passed" >&2
  exit 1
fi

git -C "$work_dir/repository" checkout --quiet -b trailing-chart-value "$release_commit"
printf '%s\n' 'version: 1.2.3 unexpected' 'appVersion: v1.2.3' >"$work_dir/repository/charts/aikit/Chart.yaml"
git -C "$work_dir/repository" add charts/aikit/Chart.yaml
git -C "$work_dir/repository" commit --quiet -m "add trailing chart value"
trailing_chart_commit=$(git -C "$work_dir/repository" rev-parse HEAD)
if run_validator v1.2.3 "$trailing_chart_commit" 42 "$valid_pr_details" >/dev/null 2>&1; then
  echo "trailing chart version value unexpectedly passed" >&2
  exit 1
fi

echo "release validator tests passed"
