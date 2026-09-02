#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 <selected-run-id> <successful-workflow-attempt> <selected-commit> <jobs-json-file|->" >&2
  exit 2
fi

selected_run_id=$1
successful_workflow_attempt=$2
selected_commit=$3
jobs_json_file=$4

if ! [[ $selected_run_id =~ ^[1-9][0-9]*$ ]]; then
  echo "Selected run ID must be a positive integer: $selected_run_id" >&2
  exit 2
fi
if ! [[ $successful_workflow_attempt =~ ^[1-9][0-9]*$ ]]; then
  echo "Successful workflow attempt must be a positive integer: $successful_workflow_attempt" >&2
  exit 2
fi
if ! [[ $selected_commit =~ ^[0-9a-f]{40}$ ]]; then
  echo "Selected commit must be a full lowercase Git SHA: $selected_commit" >&2
  exit 2
fi
if [[ $jobs_json_file != - && ! -f $jobs_json_file ]]; then
  echo "Jobs JSON file does not exist or is not a regular file: $jobs_json_file" >&2
  exit 2
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "Required command not found: jq" >&2
  exit 2
fi

if ! plan=$(jq -s -er \
  --argjson selected_run_id "$selected_run_id" \
  --argjson successful_attempt "$successful_workflow_attempt" \
  --arg selected_commit "$selected_commit" '
  def fail($message): error("Runner candidate selection failed: " + $message);
  def expected_jobs:
    [
      {runner: "llama-cpp-cpu", name: "release-runners (llama-cpp-cpu)"},
      {runner: "llama-cpp-cuda", name: "release-runners (llama-cpp-cuda)"},
      {runner: "diffusers-cuda", name: "release-runners (diffusers-cuda)"},
      {runner: "vllm-cuda", name: "release-runners (vllm-cuda)"},
      {runner: "vllm-cpp-cpu", name: "release-runners (vllm-cpp-cpu)"},
      {runner: "vllm-cpp-cuda", name: "release-runners (vllm-cpp-cuda)"}
    ];
  def valid_job_record:
    if type != "object" then
      false
    else
      (has("name") and has("run_id") and has("run_attempt") and has("head_sha") and
       has("status") and has("conclusion")) and
      ((.name | type) == "string" and (.name | length) > 0) and
      ((.run_id | type) == "number" and .run_id > 0 and (.run_id | floor) == .run_id) and
      ((.run_attempt | type) == "number" and .run_attempt > 0 and (.run_attempt | floor) == .run_attempt) and
      ((.head_sha | type) == "string" and (.head_sha | test("^[0-9a-f]{40}$"))) and
      ((.status | type) == "string" and (.status | length) > 0) and
      (((.conclusion | type) == "string" and (.conclusion | length) > 0) or .conclusion == null)
    end;

  (if length != 1 then
     fail("input must contain exactly one JSON document")
   else
     .[0]
   end
   | if type == "array" then
       .
     elif type == "object" and ((.jobs | type) == "array") then
       .jobs
     else
       fail("input must be a job array or an object with a jobs array")
     end) as $jobs
  | expected_jobs as $expected
  | ($jobs | to_entries | map(select(.value | valid_job_record | not)) | first) as $malformed
  | if $malformed != null then
      fail("job record at index \($malformed.key) is malformed")
    elif any($jobs[]; .run_id != $selected_run_id) then
      fail("job records must all belong to selected run \($selected_run_id)")
    elif ([
      $jobs[] as $job
      | select(
          ($job.name | startswith("release-runners")) and
          (($expected | map(.name) | index($job.name)) == null)
        )
    ] | length) != 0 then
      fail("job records contain an unexpected release-runners job name")
    elif ([
      $expected[] as $expected_job
      | $jobs
      | map(select(.name == $expected_job.name))
      | group_by(.run_attempt)[]
      | select(length != 1)
    ] | length) != 0 then
      fail("job records contain duplicate release-runners jobs for one attempt")
    else
      [
        $expected[] as $expected_job
        | ($jobs | map(select(
            .name == $expected_job.name and
            .run_attempt <= $successful_attempt
          ))) as $eligible
        | if ($eligible | length) == 0 then
            fail("missing \($expected_job.name) at or before attempt \($successful_attempt)")
          else
            ($eligible | max_by(.run_attempt)) as $candidate
            | if $candidate.status != "completed" then
                fail("\($expected_job.name) attempt \($candidate.run_attempt) is not completed")
              elif $candidate.conclusion != "success" then
                fail("\($expected_job.name) attempt \($candidate.run_attempt) did not succeed")
              elif $candidate.head_sha != $selected_commit then
                fail("\($expected_job.name) attempt \($candidate.run_attempt) does not match selected commit \($selected_commit)")
              else
                "\($expected_job.runner)\t\($candidate.run_attempt)"
              end
          end
      ]
      | join("\n")
    end
' "$jobs_json_file"); then
  exit 1
fi

printf '%s\n' "$plan"
