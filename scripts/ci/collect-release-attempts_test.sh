#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
collector="$script_dir/collect-release-attempts.sh"
work_dir=$(mktemp -d)
trap 'rm -rf "$work_dir"' EXIT
mkdir -p "$work_dir/bin"

cat >"$work_dir/bin/gh" <<'GH'
#!/usr/bin/env bash
set -euo pipefail

printf '%s\n' "$*" >>"$FAKE_GH_CALLS"

if [[ $* == 'api --paginate --slurp repos/kaito-project/aikit/actions/workflows/release.yaml/runs?event=push&per_page=100' ]]; then
  if [[ ${FAKE_GH_MODE:-valid} == malformed ]]; then
    printf '%s\n' '[{"workflow_runs":[{"id":42,"run_attempt":"3","path":".github/workflows/release.yaml","event":"push","head_branch":"v1.2.3","head_repository":{"full_name":"kaito-project/aikit"}}]}]'
  else
    cat <<'JSON'
[{"workflow_runs":[
  {"id":42,"run_attempt":3,"path":".github/workflows/release.yaml","event":"push","head_branch":"v1.2.3","head_repository":{"full_name":"kaito-project/aikit"}},
  {"id":43,"run_attempt":1,"path":".github/workflows/release.yaml","event":"push","head_branch":"v1.3.0-rc.1","head_repository":{"full_name":"kaito-project/aikit"}},
  {"id":44,"run_attempt":1,"path":".github/workflows/other.yaml","event":"push","head_branch":"v9.9.9","head_repository":{"full_name":"kaito-project/aikit"}},
  {"id":45,"run_attempt":1,"path":".github/workflows/release.yaml","event":"push","head_branch":"v9.9.9","head_repository":{"full_name":"someone/aikit"}}
]}]
JSON
  fi
  exit 0
fi

case "$*" in
  'api repos/kaito-project/aikit/actions/runs/42/attempts/1')
    printf '%s\n' '{"id":42,"run_attempt":1,"path":".github/workflows/release.yaml","event":"push","status":"completed","conclusion":"success","head_branch":"v1.2.3","head_sha":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","head_repository":{"full_name":"kaito-project/aikit"}}'
    ;;
  'api repos/kaito-project/aikit/actions/runs/42/attempts/2')
    if [[ ${FAKE_GH_MODE:-valid} == malformed-attempt ]]; then
      printf '%s\n' '{"id":42,"run_attempt":99,"path":".github/workflows/release.yaml","event":"push","status":"completed","conclusion":"failure","head_branch":"v1.2.3","head_sha":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","head_repository":{"full_name":"kaito-project/aikit"}}'
    else
      printf '%s\n' '{"id":42,"run_attempt":2,"path":".github/workflows/release.yaml","event":"push","status":"completed","conclusion":"failure","head_branch":"v1.2.3","head_sha":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","head_repository":{"full_name":"kaito-project/aikit"}}'
    fi
    ;;
  'api repos/kaito-project/aikit/actions/runs/42/attempts/3')
    printf '%s\n' '{"id":42,"run_attempt":3,"path":".github/workflows/release.yaml","event":"push","status":"completed","conclusion":"success","head_branch":"v1.2.3","head_sha":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","head_repository":{"full_name":"kaito-project/aikit"}}'
    ;;
  'api repos/kaito-project/aikit/actions/runs/99/attempts/1')
    printf '%s\n' '{"id":99,"run_attempt":1,"path":".github/workflows/release.yaml","event":"push","status":"completed","conclusion":"success","head_branch":"v1.2.4","head_sha":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","head_repository":{"full_name":"kaito-project/aikit"}}'
    ;;
  'api repos/kaito-project/aikit/actions/runs/99/attempts/2')
    printf '%s\n' '{"id":99,"run_attempt":2,"path":".github/workflows/release.yaml","event":"push","status":"completed","conclusion":"failure","head_branch":"v1.2.4","head_sha":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","head_repository":{"full_name":"kaito-project/aikit"}}'
    ;;
  'api repos/kaito-project/aikit/actions/runs/99/attempts/3')
    printf '%s\n' '{"id":99,"run_attempt":3,"path":".github/workflows/release.yaml","event":"push","status":"completed","conclusion":"success","head_branch":"v1.2.4","head_sha":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","head_repository":{"full_name":"kaito-project/aikit"}}'
    ;;
  *)
    echo "unexpected fake gh invocation: $*" >&2
    exit 1
    ;;
esac
GH
chmod +x "$work_dir/bin/gh"

calls="$work_dir/calls"
output="$work_dir/attempts.json"
: >"$calls"
PATH="$work_dir/bin:$PATH" FAKE_GH_CALLS="$calls" \
  "$collector" kaito-project/aikit .github/workflows/release.yaml "$output"

if ! jq -e '
  type == "array" and length == 1 and
  (.[0].workflow_runs | length == 3) and
  (.[0].workflow_runs | map(.run_attempt) == [1, 2, 3]) and
  (.[0].workflow_runs | map(.id) | unique == [42])
' "$output" >/dev/null; then
  echo "collector returned unexpected attempt metadata" >&2
  exit 1
fi
if [[ $(wc -l <"$calls" | tr -d ' ') -ne 4 ]] || \
  grep -Eq '/runs/(43|44|45)/' "$calls"; then
  echo "collector fetched attempts outside the trusted stable workflow run" >&2
  exit 1
fi

: >"$calls"
PATH="$work_dir/bin:$PATH" FAKE_GH_CALLS="$calls" \
  "$collector" kaito-project/aikit .github/workflows/release.yaml "$output" 99 3
if ! jq -e '
  (.[0].workflow_runs | length == 6) and
  (.[0].workflow_runs | map(.id) | unique == [42, 99]) and
  (.[0].workflow_runs | map(select(.id == 99)) | map(.run_attempt) == [1, 2, 3]) and
  (.[0].workflow_runs | any(.id == 99 and .run_attempt == 1 and .conclusion == "success"))
' "$output" >/dev/null; then
  echo "collector did not retain every required attempt missing from the run index" >&2
  exit 1
fi
if [[ $(grep -c '/runs/99/attempts/' "$calls") -ne 3 ]]; then
  echo "collector did not fetch every required workflow attempt exactly once" >&2
  exit 1
fi

: >"$calls"
PATH="$work_dir/bin:$PATH" FAKE_GH_CALLS="$calls" \
  "$collector" kaito-project/aikit .github/workflows/release.yaml "$output" 42 3
if [[ $(jq '.[0].workflow_runs | map(select(.id == 42 and .run_attempt == 3)) | length' "$output") -ne 1 ]]; then
  echo "collector did not deduplicate a required indexed attempt" >&2
  exit 1
fi

: >"$calls"
if PATH="$work_dir/bin:$PATH" FAKE_GH_CALLS="$calls" FAKE_GH_MODE=malformed \
  "$collector" kaito-project/aikit .github/workflows/release.yaml "$output" \
  >/dev/null 2>&1; then
  echo "collector accepted malformed workflow run metadata" >&2
  exit 1
fi
if [[ $(wc -l <"$calls" | tr -d ' ') -ne 1 ]]; then
  echo "collector fetched attempts after rejecting the workflow run index" >&2
  exit 1
fi

: >"$calls"
if PATH="$work_dir/bin:$PATH" FAKE_GH_CALLS="$calls" FAKE_GH_MODE=malformed-attempt \
  "$collector" kaito-project/aikit .github/workflows/release.yaml "$output" \
  >/dev/null 2>&1; then
  echo "collector accepted malformed workflow attempt metadata" >&2
  exit 1
fi

echo "release attempt collector tests passed"
