#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
selector="$script_dir/select-runner-candidates.sh"
work_dir=$(mktemp -d)
trap 'rm -rf "$work_dir"' EXIT

selected_run_id=424242
successful_attempt=3
selected_commit=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
wrong_commit=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
base_jobs="$work_dir/jobs.json"

cat >"$base_jobs" <<JSON
[
  {"name":"preflight","run_id":424242,"run_attempt":3,"head_sha":"$selected_commit","status":"completed","conclusion":"success"},
  {"name":"release-runners (llama-cpp-cpu)","run_id":424242,"run_attempt":1,"head_sha":"$selected_commit","status":"completed","conclusion":"failure"},
  {"name":"release-runners (llama-cpp-cpu)","run_id":424242,"run_attempt":2,"head_sha":"$selected_commit","status":"completed","conclusion":"success"},
  {"name":"release-runners (llama-cpp-cuda)","run_id":424242,"run_attempt":1,"head_sha":"$selected_commit","status":"completed","conclusion":"success"},
  {"name":"release-runners (llama-cpp-cuda)","run_id":424242,"run_attempt":4,"head_sha":"$wrong_commit","status":"completed","conclusion":"failure"},
  {"name":"release-runners (diffusers-cuda)","run_id":424242,"run_attempt":1,"head_sha":"$selected_commit","status":"completed","conclusion":"failure"},
  {"name":"release-runners (diffusers-cuda)","run_id":424242,"run_attempt":3,"head_sha":"$selected_commit","status":"completed","conclusion":"success"},
  {"name":"release-runners (vllm-cuda)","run_id":424242,"run_attempt":2,"head_sha":"$selected_commit","status":"completed","conclusion":"success"},
  {"name":"release-runners (vllm-cpp-cpu)","run_id":424242,"run_attempt":3,"head_sha":"$selected_commit","status":"completed","conclusion":"success"},
  {"name":"release-runners (vllm-cpp-cuda)","run_id":424242,"run_attempt":1,"head_sha":"$selected_commit","status":"completed","conclusion":"success"},
  {"name":"release-runners (vllm-cpp-cuda)","run_id":424242,"run_attempt":4,"head_sha":"$wrong_commit","status":"in_progress","conclusion":null}
]
JSON

run_selector() {
  "$selector" "$selected_run_id" "$successful_attempt" "$selected_commit" "$1"
}

expect_failure() {
  local description=$1
  local jobs_file=$2

  if run_selector "$jobs_file" >/dev/null 2>&1; then
    echo "$description unexpectedly passed" >&2
    exit 1
  fi
}

expected_plan=$(printf '%s\t%s\n' \
  llama-cpp-cpu 2 \
  llama-cpp-cuda 1 \
  diffusers-cuda 3 \
  vllm-cuda 2 \
  vllm-cpp-cpu 3 \
  vllm-cpp-cuda 1)
actual_plan=$(run_selector "$base_jobs")
if [[ $actual_plan != "$expected_plan" ]]; then
  echo "mixed-attempt selection returned an unexpected plan:" >&2
  printf '%s\n' "$actual_plan" >&2
  exit 1
fi

enveloped_jobs="$work_dir/enveloped.json"
jq '{jobs: .}' "$base_jobs" >"$enveloped_jobs"
if [[ $(run_selector "$enveloped_jobs") != "$expected_plan" ]]; then
  echo "GitHub jobs envelope returned an unexpected plan" >&2
  exit 1
fi

if [[ $(run_selector - <"$base_jobs") != "$expected_plan" ]]; then
  echo "standard-input selection returned an unexpected plan" >&2
  exit 1
fi

missing_jobs="$work_dir/missing.json"
jq 'map(select(.name != "release-runners (vllm-cpp-cuda)" or .run_attempt != 1))' \
  "$base_jobs" >"$missing_jobs"
expect_failure "missing eligible runner job" "$missing_jobs"

failed_jobs="$work_dir/failed.json"
jq 'map(if .name == "release-runners (diffusers-cuda)" and .run_attempt == 3 then .conclusion = "failure" else . end)' \
  "$base_jobs" >"$failed_jobs"
expect_failure "failed latest runner job" "$failed_jobs"

incomplete_jobs="$work_dir/incomplete.json"
jq 'map(if .name == "release-runners (vllm-cuda)" and .run_attempt == 2 then .status = "in_progress" | .conclusion = null else . end)' \
  "$base_jobs" >"$incomplete_jobs"
expect_failure "incomplete latest runner job" "$incomplete_jobs"

duplicate_jobs="$work_dir/duplicate.json"
jq '. + [(.[] | select(.name == "release-runners (vllm-cuda)" and .run_attempt == 2))]' \
  "$base_jobs" >"$duplicate_jobs"
expect_failure "duplicate runner job" "$duplicate_jobs"

wrong_sha_jobs="$work_dir/wrong-sha.json"
jq --arg wrong_commit "$wrong_commit" \
  'map(if .name == "release-runners (vllm-cpp-cpu)" and .run_attempt == 3 then .head_sha = $wrong_commit else . end)' \
  "$base_jobs" >"$wrong_sha_jobs"
expect_failure "runner job for the wrong commit" "$wrong_sha_jobs"

malformed_jobs="$work_dir/malformed.json"
jq 'map(if .name == "release-runners (llama-cpp-cpu)" and .run_attempt == 2 then .run_attempt = "2" else . end)' \
  "$base_jobs" >"$malformed_jobs"
expect_failure "malformed runner job" "$malformed_jobs"

missing_field_jobs="$work_dir/missing-field.json"
jq 'map(if .name == "release-runners (vllm-cpp-cuda)" and .run_attempt == 4 then del(.conclusion) else . end)' \
  "$base_jobs" >"$missing_field_jobs"
expect_failure "runner job with a missing field" "$missing_field_jobs"

wrong_run_jobs="$work_dir/wrong-run.json"
jq 'map(if .name == "release-runners (llama-cpp-cpu)" and .run_attempt == 2 then .run_id = 999999 else . end)' \
  "$base_jobs" >"$wrong_run_jobs"
expect_failure "runner job from another run" "$wrong_run_jobs"

unexpected_jobs="$work_dir/unexpected.json"
jq '. + [{
  name: "release-runners (unexpected)",
  run_id: 424242,
  run_attempt: 1,
  head_sha: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  status: "completed",
  conclusion: "success"
}]' "$base_jobs" >"$unexpected_jobs"
expect_failure "unexpected release-runners job" "$unexpected_jobs"

echo "runner candidate selector tests passed"
