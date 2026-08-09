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
      printf '%s' "${FAKE_GH_ISSUES:-}"
      exit 0
      ;;
    *"/pulls/"*)
      printf '%s' "${FAKE_GH_PR_DETAILS:-}"
      exit 0
      ;;
    *"/actions/runs "*)
      printf '%s' "${FAKE_GH_ACTION_RUNS:-}"
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

mkdir -p "$work_dir/repository/charts/aikit"
printf '%s\n' 'VERSION := v1.2.2' >"$work_dir/repository/Makefile"
printf '%s\n' 'version: 1.2.2' 'appVersion: v1.2.2' >"$work_dir/repository/charts/aikit/Chart.yaml"
git -C "$work_dir/repository" add Makefile charts/aikit/Chart.yaml
git -C "$work_dir/repository" commit --quiet -m "initial version"
git -C "$work_dir/repository" push --quiet --set-upstream origin main

git -C "$work_dir/repository" checkout --quiet -b release-1.2
printf '%s\n' 'VERSION := v1.2.3' >"$work_dir/repository/Makefile"
printf '%s\n' 'version: 1.2.3' 'appVersion: v1.2.3' >"$work_dir/repository/charts/aikit/Chart.yaml"
git -C "$work_dir/repository" add Makefile charts/aikit/Chart.yaml
git -C "$work_dir/repository" commit --quiet -m "prepare v1.2.3"
release_pr_commit=$(git -C "$work_dir/repository" rev-parse HEAD)

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

run_validator() {
  local version=$1
  local commit=$2
  local release_pr_issues=$3
  local release_pr_details=$4
  local release_action_runs=${5-$valid_action_runs}
  (
    cd "$work_dir/repository"
    PATH="$work_dir/bin:$PATH" \
      FAKE_GH_ISSUES="$release_pr_issues" \
      FAKE_GH_PR_DETAILS="$release_pr_details" \
      FAKE_GH_ACTION_RUNS="$release_action_runs" \
      GH_TOKEN=test-token \
      GITHUB_REPOSITORY=example/aikit \
      "$validator" "$version" "$commit"
  )
}

valid_pr_details=$(printf 'true\trelease-1.2\t%s\t%s' "$release_pr_commit" "$release_pr_commit")
run_validator v1.2.3 "$release_commit" 42 "$valid_pr_details" >/dev/null

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

pending_action_runs=$(printf '%s\n%s' \
  $'2026-08-08T20:00:02Z\t102\t.github/workflows/lint.yaml\tin_progress\t-' \
  $'2026-08-08T20:00:01Z\t101\t.github/workflows/unit-test.yaml\tcompleted\tsuccess')
if run_validator v1.2.3 "$release_commit" 42 "$valid_pr_details" "$pending_action_runs" >/dev/null 2>&1; then
  echo "release pull request with an unfinished workflow unexpectedly passed" >&2
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

unrelated_pr_details=$(printf 'true\trelease-1.2\t%s' "$side_commit")
if run_validator v1.2.3 "$release_commit" 43 "$unrelated_pr_details" >/dev/null 2>&1; then
  echo "unrelated release pull request unexpectedly passed" >&2
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
