#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 && $# -ne 5 ]]; then
  echo "Usage: $0 <owner/repository> <workflow-path> <output-json> [required-run-id required-run-attempt]" >&2
  exit 2
fi

repository=$1
workflow_path=$2
output_json=$3
required_run_id=${4:-}
required_run_attempt=${5:-}

fail() {
  echo "Release attempt collection failed: $*" >&2
  exit 1
}

if ! [[ $repository =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]]; then
  fail "invalid repository: $repository"
fi
if ! [[ $workflow_path =~ ^\.github/workflows/([A-Za-z0-9_.-]+\.ya?ml)$ ]]; then
  fail "invalid workflow path: $workflow_path"
fi
workflow_file=${BASH_REMATCH[1]}
if [[ -n $required_run_id ]] && \
  { ! [[ $required_run_id =~ ^[1-9][0-9]*$ ]] || \
    ! [[ $required_run_attempt =~ ^[1-9][0-9]{0,3}$ ]]; }; then
  fail "required run coordinates must be positive integers"
fi

run_index=$(mktemp)
attempt_records=$(mktemp)
attempt_record=$(mktemp)
trap 'rm -f "$run_index" "$attempt_records" "$attempt_record"' EXIT

validate_attempt_record() {
  local record=$1
  local run_id=$2
  local run_attempt=$3

  jq -e \
    --arg repository "$repository" \
    --arg workflow_path "$workflow_path" \
    --argjson run_id "$run_id" \
    --argjson run_attempt "$run_attempt" '
      type == "object" and
      .id == $run_id and
      .run_attempt == $run_attempt and
      .path == $workflow_path and
      .event == "push" and
      .head_repository.full_name == $repository and
      (.head_branch | type == "string") and
      (.head_branch | test("^v(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)$")) and
      (.head_sha | type == "string") and
      (.head_sha | test("^[0-9a-f]{40}$")) and
      (.status | type == "string") and
      (.status | length > 0) and
      (((.conclusion | type) == "string" and (.conclusion | length > 0)) or .conclusion == null)
    ' "$record" >/dev/null
}

gh api --paginate --slurp \
  "repos/${repository}/actions/workflows/${workflow_file}/runs?event=push&per_page=100" \
  >"$run_index"

if ! jq -e \
  --arg repository "$repository" \
  --arg workflow_path "$workflow_path" '
    type == "array" and
    all(.[]; type == "object" and (.workflow_runs | type == "array")) and
    ([
      .[] | .workflow_runs[] |
      select(
        .path == $workflow_path and
        .event == "push" and
        .head_repository.full_name == $repository and
        (.head_branch | type == "string") and
        (.head_branch | test("^v(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)$"))
      )
    ] | all(.[];
      (.id | type == "number") and
      (.id | floor == .) and
      .id > 0 and
      (.run_attempt | type == "number") and
      (.run_attempt | floor == .) and
      .run_attempt > 0 and
      .run_attempt <= 1000
    ))
  ' "$run_index" >/dev/null; then
  fail "workflow run index is malformed"
fi

: >"$attempt_records"
while IFS=$'\t' read -r run_id run_attempt; do
  if ! [[ $run_id =~ ^[1-9][0-9]*$ && $run_attempt =~ ^[1-9][0-9]{0,3}$ ]]; then
    fail "workflow run index returned invalid attempt coordinates"
  fi

  attempt=1
  while ((attempt <= run_attempt)); do
    gh api \
      "repos/${repository}/actions/runs/${run_id}/attempts/${attempt}" \
      >"$attempt_record"
    if ! validate_attempt_record "$attempt_record" "$run_id" "$attempt"; then
      fail "workflow attempt ${run_id}/${attempt} is malformed"
    fi
    jq -c '.' "$attempt_record" >>"$attempt_records"
    ((attempt += 1))
  done
done < <(
  jq -r \
    --arg repository "$repository" \
    --arg workflow_path "$workflow_path" '
      .[] | .workflow_runs[] |
      select(
        .path == $workflow_path and
        .event == "push" and
        .head_repository.full_name == $repository and
        (.head_branch | type == "string") and
        (.head_branch | test("^v(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)$"))
      ) |
      [.id, .run_attempt] | @tsv
    ' "$run_index"
)

if [[ -n $required_run_id ]]; then
  attempt=1
  while ((attempt <= required_run_attempt)); do
    gh api \
      "repos/${repository}/actions/runs/${required_run_id}/attempts/${attempt}" \
      >"$attempt_record"
    if ! validate_attempt_record \
      "$attempt_record" "$required_run_id" "$attempt"; then
      fail "required workflow attempt ${required_run_id}/${attempt} is malformed"
    fi
    jq -c '.' "$attempt_record" >>"$attempt_records"
    ((attempt += 1))
  done
fi

if ! jq -s '[{workflow_runs: unique_by([.id, .run_attempt])}]' \
  "$attempt_records" >"$output_json"; then
  fail "could not assemble workflow attempt metadata"
fi
