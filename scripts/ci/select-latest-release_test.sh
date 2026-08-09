#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
selector="$script_dir/select-latest-release.sh"
work_dir=$(mktemp -d)
trap 'rm -rf "$work_dir"' EXIT

sha_021=1111111111111111111111111111111111111111
sha_022=2222222222222222222222222222222222222222
sha_023=3333333333333333333333333333333333333333
sha_big=4444444444444444444444444444444444444444

write_tags() {
  cat >"$work_dir/tags" <<EOF
$sha_021	refs/tags/v0.21.0
$sha_022	refs/tags/v0.22.0
aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa	refs/tags/v0.23.0
$sha_023	refs/tags/v0.23.0^{}
$sha_big	refs/tags/v9223372036854775808.0.0
eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee	refs/tags/v0.24.0-rc.1
EOF
}

write_runs() {
  local runs=$1
  printf '[{"workflow_runs":%s}]\n' "$runs" >"$work_dir/runs.json"
}

run_record() {
  local version=$1
  local sha=$2
  local conclusion=${3:-success}
  local event=${4:-push}
  local repository=${5:-kaito-project/aikit}
  local path=${6:-.github/workflows/release.yaml}

  jq -cn \
    --arg version "$version" \
    --arg sha "$sha" \
    --arg conclusion "$conclusion" \
    --arg event "$event" \
    --arg repository "$repository" \
    --arg path "$path" \
    '{
      path: $path,
      event: $event,
      status: "completed",
      conclusion: $conclusion,
      head_branch: $version,
      head_sha: $sha,
      head_repository: {full_name: $repository}
    }'
}

run_selector() {
  local output=$1

  GITHUB_OUTPUT="$output" "$selector" \
    kaito-project/aikit .github/workflows/release.yaml \
    "$work_dir/runs.json" "$work_dir/tags" >/dev/null
}

assert_output() {
  local output=$1
  local expected=$2

  if ! grep -qxF "$expected" "$output"; then
    echo "missing output '$expected' in $output" >&2
    exit 1
  fi
}

write_tags

runs=$(jq -cn \
  --argjson older "$(run_record v0.21.0 "$sha_021")" \
  --argjson newer "$(run_record v0.22.0 "$sha_022")" \
  --argjson failed "$(run_record v0.23.0 "$sha_023" failure)" \
  '[$newer, $failed, $older]')
write_runs "$runs"
output="$work_dir/failed-newer.out"
run_selector "$output"
assert_output "$output" 'found=true'
assert_output "$output" 'version=v0.22.0'
assert_output "$output" "commit=$sha_022"

runs=$(jq -cn \
  --argjson newer "$(run_record v0.23.0 "$sha_023")" \
  --argjson older "$(run_record v0.21.0 "$sha_021")" \
  '[$newer, $older]')
write_runs "$runs"
output="$work_dir/out-of-order.out"
run_selector "$output"
assert_output "$output" 'version=v0.23.0'

runs=$(jq -cn \
  --argjson big "$(run_record v9223372036854775808.0.0 "$sha_big")" \
  --argjson normal "$(run_record v0.23.0 "$sha_023")" \
  '[$normal, $big]')
write_runs "$runs"
output="$work_dir/large-version.out"
run_selector "$output"
assert_output "$output" 'version=v9223372036854775808.0.0'

runs=$(jq -cn \
  --argjson wrong_sha "$(run_record v0.23.0 ffffffffffffffffffffffffffffffffffffffff)" \
  --argjson dispatch "$(run_record v0.22.0 "$sha_022" success workflow_dispatch)" \
  --argjson fork "$(run_record v0.22.0 "$sha_022" success push someone/aikit)" \
  --argjson wrong_path "$(run_record v0.22.0 "$sha_022" success push kaito-project/aikit .github/workflows/other.yaml)" \
  --argjson malformed "$(run_record v0.22.0-rc.1 "$sha_022")" \
  --argjson valid "$(run_record v0.21.0 "$sha_021")" \
  '[$wrong_sha, $dispatch, $fork, $wrong_path, $malformed, $valid]')
write_runs "$runs"
output="$work_dir/untrusted-metadata.out"
run_selector "$output"
assert_output "$output" 'version=v0.21.0'

runs=$(jq -cn \
  --argjson failed "$(run_record v0.23.0 "$sha_023" failure)" \
  --argjson cancelled "$(run_record v0.22.0 "$sha_022" cancelled)" \
  '[$failed, $cancelled]')
write_runs "$runs"
output="$work_dir/no-success.out"
run_selector "$output"
assert_output "$output" 'found=false'
assert_output "$output" 'version=none'
assert_output "$output" 'commit=none'

printf '[{"workflow_runs":"invalid"}]\n' >"$work_dir/runs.json"
if run_selector "$work_dir/invalid-json.out" >/dev/null 2>&1; then
  echo "invalid workflow-runs response unexpectedly passed" >&2
  exit 1
fi

echo "latest release selector tests passed"
