#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
planner="$script_dir/plan-release-sync.sh"
work_dir=$(mktemp -d)
trap 'rm -rf "$work_dir"' EXIT
main_makefile="$work_dir/main-Makefile"
main_chart="$work_dir/main-Chart.yaml"
pending_makefile="$work_dir/pending-Makefile"
pending_chart="$work_dir/pending-Chart.yaml"
pending_second_makefile="$work_dir/pending-second-Makefile"
pending_second_chart="$work_dir/pending-second-Chart.yaml"

write_manifests() {
  local version=$1
  local makefile=$2
  local chart=$3

  printf 'VERSION := %s\n' "$version" >"$makefile"
  printf 'version: %s\nappVersion: %s\n' "${version#v}" "$version" >"$chart"
}

assert_plan() {
  local expected_action=$1
  local release_version=$2
  local main_version=$3
  local pending_version=${4:-}
  local actual_action
  local arguments

  write_manifests "$main_version" "$main_makefile" "$main_chart"
  arguments=("$release_version" "$main_makefile" "$main_chart")
  if [[ -n $pending_version ]]; then
    write_manifests "$pending_version" "$pending_makefile" "$pending_chart"
    arguments+=("$pending_makefile" "$pending_chart")
  fi

  actual_action=$("$planner" "${arguments[@]}")
  if [[ $actual_action != "$expected_action" ]]; then
    echo "expected $expected_action for release $release_version from main $main_version and pending ${pending_version:-none}; got $actual_action" >&2
    exit 1
  fi
}

assert_plan update v0.22.1 v0.21.0
assert_plan none v0.22.1 v0.22.0
assert_plan none v0.22.9 v0.23.0
assert_plan update v1.0.0 v0.99.0
assert_plan update v18446744073709551616.0.0 v1.0.0
assert_plan none v1.0.0 v18446744073709551616.0.0

# A canonical pending target is monotonic: older and same-line releases only
# ensure its pull request exists, while a newer release replaces it.
assert_plan ensure v0.22.1 v0.21.0 v0.23.0
assert_plan ensure v0.22.2 v0.21.0 v0.22.1
assert_plan update v0.23.0 v0.21.0 v0.22.1
assert_plan update v1.0.0 v0.23.0 v0.22.1
assert_plan none v0.22.1 v0.23.0 v0.22.1

write_manifests v0.21.0 "$main_makefile" "$main_chart"
printf '%s\n' 'version: 0.20.0' 'appVersion: v0.21.0' >"$main_chart"
if "$planner" v0.22.1 "$main_makefile" "$main_chart" >/dev/null 2>&1; then
  echo "inconsistent manifests unexpectedly produced a sync plan" >&2
  exit 1
fi

write_manifests v0.21.0 "$main_makefile" "$main_chart"
printf '%s\n' 'version:0.21.0' 'appVersion: v0.21.0' >"$main_chart"
if "$planner" v0.22.1 "$main_makefile" "$main_chart" >/dev/null 2>&1; then
  echo "chart version without YAML separator unexpectedly produced a sync plan" >&2
  exit 1
fi

write_manifests v0.21.0 "$main_makefile" "$main_chart"
printf '%s\n' 'VERSION := v0.20.0' >>"$main_makefile"
if "$planner" v0.22.1 "$main_makefile" "$main_chart" >/dev/null 2>&1; then
  echo "duplicate Makefile VERSION unexpectedly produced a sync plan" >&2
  exit 1
fi

write_manifests v0.21.0 "$main_makefile" "$main_chart"
if "$planner" v00.22.1 "$main_makefile" "$main_chart" >/dev/null 2>&1; then
  echo "invalid release version unexpectedly produced a sync plan" >&2
  exit 1
fi

printf '%s\n' 'VERSION := v0.21.0' >"$main_makefile"
printf '%s\n' 'version: "0.21.0"' "appVersion: 'v0.21.0'" >"$main_chart"
if [[ $("$planner" v0.22.1 "$main_makefile" "$main_chart") != update ]]; then
  echo "quoted chart versions unexpectedly failed sync planning" >&2
  exit 1
fi

write_manifests v0.21.0 "$main_makefile" "$main_chart"
write_manifests v0.22.1 "$pending_makefile" "$pending_chart"
output_file="$work_dir/github-output"
GITHUB_OUTPUT="$output_file" "$planner" \
  v0.22.2 "$main_makefile" "$main_chart" "$pending_makefile" "$pending_chart" >/dev/null
expected_output=$(printf '%s\n' 'action=ensure' 'needed=true' 'target_version=v0.22.1')
if [[ $(<"$output_file") != "$expected_output" ]]; then
  echo "planner did not write the expected GitHub Actions outputs" >&2
  exit 1
fi

# Pending targets are selected with arbitrary-precision SemVer ordering.
write_manifests v1.0.0 "$main_makefile" "$main_chart"
write_manifests v18446744073709551616.0.0 "$pending_makefile" "$pending_chart"
write_manifests v2.0.0 "$pending_second_makefile" "$pending_second_chart"
: >"$output_file"
GITHUB_OUTPUT="$output_file" "$planner" \
  v3.0.0 \
  "$main_makefile" "$main_chart" \
  "$pending_makefile" "$pending_chart" \
  "$pending_second_makefile" "$pending_second_chart" >/dev/null
expected_output=$(printf '%s\n' 'action=ensure' 'needed=true' 'target_version=v18446744073709551616.0.0')
if [[ $(<"$output_file") != "$expected_output" ]]; then
  echo "planner did not retain the highest oversized pending release line" >&2
  exit 1
fi

write_manifests v1.0.0 "$main_makefile" "$main_chart"
write_manifests v2.0.18446744073709551616 "$pending_makefile" "$pending_chart"
write_manifests v2.0.1 "$pending_second_makefile" "$pending_second_chart"
: >"$output_file"
GITHUB_OUTPUT="$output_file" "$planner" \
  v1.5.0 \
  "$main_makefile" "$main_chart" \
  "$pending_makefile" "$pending_chart" \
  "$pending_second_makefile" "$pending_second_chart" >/dev/null
expected_output=$(printf '%s\n' 'action=ensure' 'needed=true' 'target_version=v2.0.18446744073709551616')
if [[ $(<"$output_file") != "$expected_output" ]]; then
  echo "planner did not retain the highest oversized pending patch" >&2
  exit 1
fi

echo "release sync planner tests passed"
