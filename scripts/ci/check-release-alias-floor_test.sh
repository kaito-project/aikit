#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
checker="$script_dir/check-release-alias-floor.sh"
work_dir=$(mktemp -d)
trap 'rm -rf "$work_dir"' EXIT
mkdir -p "$work_dir/bin"

sha_021=1111111111111111111111111111111111111111
sha_022=2222222222222222222222222222222222222222
sha_0221=2222222222222222222222222222222222222221
sha_023=3333333333333333333333333333333333333333
sha_big=4444444444444444444444444444444444444444
tag_object_022=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
digest_021=sha256:1111111111111111111111111111111111111111111111111111111111111111
digest_shared=sha256:2222222222222222222222222222222222222222222222222222222222222222
digest_big=sha256:3333333333333333333333333333333333333333333333333333333333333333
digest_unmapped=sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff
issuer=https://token.actions.githubusercontent.com
mapping_file="$work_dir/digests"
signatures_file="$work_dir/signatures"
tags_file="$work_dir/tags"

cat >"$tags_file" <<EOF
$sha_021	refs/tags/v0.21.0
$tag_object_022	refs/tags/v0.22.0
$sha_022	refs/tags/v0.22.0^{}
$sha_0221	refs/tags/v0.22.1
$sha_023	refs/tags/v0.23.0
$sha_big	refs/tags/v9223372036854775808.0.0
eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee	refs/tags/v0.24.0-rc.1
EOF

cat >"$work_dir/bin/curl" <<'CURL'
#!/usr/bin/env bash
set -euo pipefail

dump_header=
url=
while (($# > 0)); do
  case $1 in
    --dump-header)
      dump_header=$2
      shift 2
      ;;
    http://*|https://*)
      url=$1
      shift
      ;;
    *)
      shift
      ;;
  esac
done

if [[ $url == https://ghcr.io/token ]]; then
  printf '%s\n' '{"token":"scoped-registry-token"}'
  exit 0
fi
if [[ $url != https://ghcr.io/v2/kaito-project/aikit/aikit/manifests/* ]]; then
  echo "unexpected fake curl URL: $url" >&2
  exit 1
fi

tag=${url##*/}
value=$(awk -F '\t' -v tag="$tag" '$1 == tag { print $2; exit }' "$FAKE_DIGESTS_FILE")
case $value in
  sha256:*)
    printf 'HTTP/2 200\r\nDocker-Content-Digest: %s\r\n\r\n' "$value" >"$dump_header"
    printf '200'
    ;;
  forbidden)
    printf 'HTTP/2 403\r\n\r\n' >"$dump_header"
    printf '403'
    ;;
  invalid)
    printf 'HTTP/2 200\r\nDocker-Content-Digest: invalid\r\n\r\n' >"$dump_header"
    printf '200'
    ;;
  absent|'')
    printf 'HTTP/2 404\r\n\r\n' >"$dump_header"
    printf '404'
    ;;
  *)
    echo "unexpected fake digest value: $value" >&2
    exit 1
    ;;
esac
CURL
chmod +x "$work_dir/bin/curl"

cat >"$work_dir/bin/cosign" <<'COSIGN'
#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} != verify || -z ${2:-} ]]; then
  echo "unexpected fake cosign invocation" >&2
  exit 1
fi
image=$2
shift 2
artifact=
commit=
identity=
issuer=
output=
workflow_name=
workflow_ref=
workflow_repository=
workflow_sha=
workflow_trigger=
while (($# > 0)); do
  case $1 in
    -a)
      case $2 in
        release-artifact=*) artifact=${2#release-artifact=} ;;
        release-commit=*) commit=${2#release-commit=} ;;
      esac
      shift 2
      ;;
    --certificate-identity)
      identity=$2
      shift 2
      ;;
    --certificate-oidc-issuer)
      issuer=$2
      shift 2
      ;;
    --certificate-github-workflow-name)
      workflow_name=$2
      shift 2
      ;;
    --certificate-github-workflow-ref)
      workflow_ref=$2
      shift 2
      ;;
    --certificate-github-workflow-repository)
      workflow_repository=$2
      shift 2
      ;;
    --certificate-github-workflow-sha)
      workflow_sha=$2
      shift 2
      ;;
    --certificate-github-workflow-trigger)
      workflow_trigger=$2
      shift 2
      ;;
    --output)
      output=$2
      shift 2
      ;;
    *)
      echo "unexpected fake cosign argument: $1" >&2
      exit 1
      ;;
  esac
done

identity_prefix="https://github.com/kaito-project/aikit/.github/workflows/release.yaml@refs/tags/"
if [[ $issuer != https://token.actions.githubusercontent.com || \
  $identity != "$identity_prefix"* || $output != json ]]; then
  exit 1
fi
version=${identity#"$identity_prefix"}
record=$(awk -F '\t' -v version="$version" '$1 == version { print; exit }' "$FAKE_SIGNATURES_FILE")
if [[ -z $record ]]; then
  exit 1
fi
IFS=$'\t' read -r _ expected_artifact expected_commit run_id run_attempt state <<<"$record"
if [[ $state == invalid ]]; then
  exit 1
fi

digest=${image##*@}
if [[ $state == legacy* ]]; then
  if [[ -n $workflow_name || -n $workflow_ref || -n $workflow_repository || \
    -n $workflow_sha || -n $workflow_trigger ]]; then
    if [[ $workflow_name != release || \
      $workflow_ref != "refs/tags/${version}" || \
      $workflow_repository != kaito-project/aikit || \
      $workflow_sha != "$expected_commit" || \
      $workflow_trigger != push || $state == legacy-wrong-claims ]]; then
      exit 1
    fi
  fi

  legacy_type=https://sigstore.dev/cosign/sign/v1
  legacy_reference=$image
  legacy_optional='{}'
  case $state in
    legacy-wrong-reference)
      legacy_reference=ghcr.io/kaito-project/aikit/aikit@sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
      ;;
    legacy-wrong-type)
      legacy_type=invalid
      ;;
    legacy-nonempty)
      legacy_optional='{"release-artifact":"aikit"}'
      ;;
  esac
  jq -cn \
    --arg digest "$digest" \
    --arg image "$legacy_reference" \
    --arg type "$legacy_type" \
    --argjson optional "$legacy_optional" \
    '[{
      critical: {
        type: $type,
        image: {"docker-manifest-digest": $digest},
        identity: {"docker-reference": $image}
      },
      optional: $optional
    }]'
  exit 0
fi
if [[ $artifact != "$expected_artifact" || $commit != "$expected_commit" ]]; then
  exit 1
fi
if [[ $state == wrong-digest ]]; then
  digest=sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
fi
if [[ $state == missing-run ]]; then
  jq -cn \
    --arg artifact "$artifact" \
    --arg commit "$commit" \
    --arg digest "$digest" \
    '[{
      critical: {image: {"docker-manifest-digest": $digest}},
      optional: {
        "release-artifact": $artifact,
        "release-commit": $commit
      }
    }]'
else
  jq -cn \
    --arg artifact "$artifact" \
    --arg commit "$commit" \
    --arg digest "$digest" \
    --arg run_attempt "$run_attempt" \
    --arg run_id "$run_id" \
    '[{
      critical: {image: {"docker-manifest-digest": $digest}},
      optional: {
        "release-artifact": $artifact,
        "release-commit": $commit,
        "release-run-attempt": $run_attempt,
        "release-run-id": $run_id
      }
    }]'
fi
COSIGN
chmod +x "$work_dir/bin/cosign"

run_checker() {
  local reference=$1
  local target_version=${2:-}
  local checker_args=(
    kaito-project/aikit/aikit "$reference" aikit
    kaito-project/aikit .github/workflows/release.yaml "$issuer" "$tags_file"
  )

  if [[ -n $target_version ]]; then
    checker_args+=("$target_version")
  fi

  PATH="$work_dir/bin:$PATH" \
    FAKE_DIGESTS_FILE="$mapping_file" \
    FAKE_SIGNATURES_FILE="$signatures_file" \
    GHCR_USERNAME=test-user \
    GHCR_TOKEN=test-token \
    "$checker" "${checker_args[@]}"
}

expect_failure() {
  local description=$1
  local reference=$2

  if run_checker "$reference" >/dev/null 2>&1; then
    echo "$description unexpectedly passed" >&2
    exit 1
  fi
}

cat >"$mapping_file" <<EOF
latest	absent
EOF
: >"$signatures_file"
output_file="$work_dir/absent-output"
result=$(GITHUB_OUTPUT="$output_file" run_checker latest v0.22.0 2>/dev/null)
if [[ $result != none ]] || \
  ! grep -qxF 'found=false' "$output_file" || \
  ! grep -qxF 'version=none' "$output_file" || \
  ! grep -qxF 'digest=absent' "$output_file" || \
  ! grep -qxF 'promote=true' "$output_file"; then
  echo "absent latest alias did not return an explicit no-floor result" >&2
  exit 1
fi

cat >"$mapping_file" <<EOF
latest	$digest_shared
v0.21.0	$digest_021
v0.22.0	$digest_shared
v0.23.0	$digest_shared
v9223372036854775808.0.0	$digest_big
EOF
cat >"$signatures_file" <<EOF
v0.22.0	aikit	$sha_022	220	2	valid
v0.23.0	aikit	$sha_023	230	1	valid
EOF
if [[ $(run_checker latest 2>/dev/null) != v0.23.0 ]]; then
  echo "latest alias did not return its highest signed immutable version" >&2
  exit 1
fi

lower_output="$work_dir/lower-target-output"
if [[ $(GITHUB_OUTPUT="$lower_output" run_checker latest v0.22.0 2>/dev/null) != v0.23.0 ]] || \
  ! grep -qxF 'promote=false' "$lower_output"; then
  echo "lower target did not return a successful no-op result" >&2
  exit 1
fi
equal_output="$work_dir/equal-target-output"
if [[ $(GITHUB_OUTPUT="$equal_output" run_checker latest v0.23.0 2>/dev/null) != v0.23.0 ]] || \
  ! grep -qxF 'promote=true' "$equal_output"; then
  echo "target equal to the floor was not allowed" >&2
  exit 1
fi
forward_output="$work_dir/forward-target-output"
if [[ $(GITHUB_OUTPUT="$forward_output" run_checker latest v1.0.0 2>/dev/null) != v0.23.0 ]] || \
  ! grep -qxF 'promote=true' "$forward_output"; then
  echo "target above the floor was not allowed" >&2
  exit 1
fi
explicit_output="$work_dir/explicit-target-output"
if [[ $(GITHUB_OUTPUT="$explicit_output" run_checker v0.22.0 v0.21.0 2>/dev/null) != v0.22.0 ]] || \
  ! grep -qxF 'promote=false' "$explicit_output"; then
  echo "explicit GitHub Latest version did not validate exactly that version" >&2
  exit 1
fi

cat >"$mapping_file" <<EOF
latest	$digest_shared
v0.22.1	$digest_shared
v0.23.0	$digest_021
EOF
cat >"$signatures_file" <<EOF
v0.22.1	aikit	$sha_0221	31276791536	1	legacy
EOF
if [[ $(run_checker latest 2>/dev/null) != v0.22.1 ]]; then
  echo "legacy certificate claims did not establish a durable alias floor" >&2
  exit 1
fi
if [[ $(run_checker v0.22.1 2>/dev/null) != v0.22.1 ]]; then
  echo "legacy GitHub Latest floor did not validate" >&2
  exit 1
fi

for invalid_legacy_state in \
  legacy-wrong-claims \
  legacy-wrong-reference \
  legacy-wrong-type \
  legacy-nonempty; do
  cat >"$signatures_file" <<EOF
v0.22.1	aikit	$sha_0221	31276791536	1	$invalid_legacy_state
EOF
  expect_failure "$invalid_legacy_state legacy signature" latest
done

cat >"$mapping_file" <<EOF
latest	$digest_shared
v0.21.0	$digest_021
v0.22.0	$digest_shared
v0.23.0	$digest_shared
v9223372036854775808.0.0	$digest_big
EOF
cat >"$signatures_file" <<EOF
v0.22.0	aikit	$sha_022	220	2	valid
v0.23.0	aikit	$sha_023	230	1	invalid
EOF
if [[ $(run_checker latest 2>/dev/null) != v0.22.0 ]]; then
  echo "alias floor did not fall back to the highest verified matching version" >&2
  exit 1
fi

cat >"$mapping_file" <<EOF
latest	$digest_big
v0.21.0	$digest_021
v0.22.0	$digest_shared
v0.23.0	$digest_big
v9223372036854775808.0.0	$digest_big
EOF
cat >"$signatures_file" <<EOF
v0.23.0	aikit	$sha_023	230	1	valid
v9223372036854775808.0.0	aikit	$sha_big	999	1	valid
EOF
if [[ $(run_checker latest 2>/dev/null) != v9223372036854775808.0.0 ]]; then
  echo "alias floor did not compare arbitrarily large semantic versions correctly" >&2
  exit 1
fi

cat >"$mapping_file" <<EOF
latest	$digest_unmapped
v0.21.0	$digest_021
v0.22.0	$digest_shared
v0.23.0	$digest_shared
v9223372036854775808.0.0	$digest_big
EOF
: >"$signatures_file"
expect_failure "unmapped mutable alias" latest

cat >"$mapping_file" <<EOF
latest	$digest_shared
v0.21.0	$digest_021
v0.22.0	$digest_021
v0.23.0	$digest_shared
v9223372036854775808.0.0	$digest_big
EOF
cat >"$signatures_file" <<EOF
v0.23.0	aikit	$sha_023	230	1	missing-run
EOF
expect_failure "signature without attempt-bound run provenance" latest

cat >"$signatures_file" <<EOF
v0.23.0	aikit	$sha_023	230	1001	valid
EOF
expect_failure "signature with an invalid run attempt" latest

cat >"$signatures_file" <<EOF
v0.23.0	aikit	$sha_023	230	1	wrong-digest
EOF
expect_failure "signature for a different manifest digest" latest

cat >"$signatures_file" <<EOF
v0.23.0	aikit	$sha_022	230	1	valid
EOF
expect_failure "signature bound to the wrong release commit" latest

cat >"$mapping_file" <<EOF
v0.23.0	absent
EOF
: >"$signatures_file"
expect_failure "absent explicit immutable version" v0.23.0

cat >"$mapping_file" <<EOF
latest	forbidden
EOF
expect_failure "registry authorization failure" latest
if run_checker latest v0.23 >/dev/null 2>&1; then
  echo "invalid target version unexpectedly passed" >&2
  exit 1
fi

malformed_tags="$work_dir/malformed-tags"
printf '%s\t%s\n' "$sha_023" refs/heads/v0.23.0 >"$malformed_tags"
cat >"$mapping_file" <<EOF
latest	$digest_shared
EOF
if PATH="$work_dir/bin:$PATH" \
  FAKE_DIGESTS_FILE="$mapping_file" \
  FAKE_SIGNATURES_FILE="$signatures_file" \
  GHCR_USERNAME=test-user \
  GHCR_TOKEN=test-token \
  "$checker" kaito-project/aikit/aikit latest aikit \
  kaito-project/aikit .github/workflows/release.yaml "$issuer" "$malformed_tags" \
  >/dev/null 2>&1; then
  echo "malformed remote tag input unexpectedly passed" >&2
  exit 1
fi

echo "release alias floor tests passed"
