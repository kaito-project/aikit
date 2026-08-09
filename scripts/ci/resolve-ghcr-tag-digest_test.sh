#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
resolver="$script_dir/resolve-ghcr-tag-digest.sh"
work_dir=$(mktemp -d)
trap 'rm -rf "$work_dir"' EXIT
mkdir -p "$work_dir/bin"

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

case $url in
  https://ghcr.io/token)
    if [[ ${FAKE_GHCR_MODE:-present} == malformed-token ]]; then
      printf '%s\n' '{}'
    else
      printf '%s\n' '{"token":"scoped-registry-token"}'
    fi
    ;;
  https://ghcr.io/v2/kaito-project/aikit/aikit/manifests/v1.2.3|\
  https://ghcr.io/v2/kaito-project/aikit/aikit/manifests/latest)
    case ${FAKE_GHCR_MODE:-present} in
      present)
        printf 'HTTP/2 200\r\nDocker-Content-Digest: sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\r\n\r\n' \
          >"$dump_header"
        printf '200'
        ;;
      absent)
        printf 'HTTP/2 404\r\n\r\n' >"$dump_header"
        printf '404'
        ;;
      invalid-digest)
        printf 'HTTP/2 200\r\nDocker-Content-Digest: sha256:not-a-digest\r\n\r\n' \
          >"$dump_header"
        printf '200'
        ;;
      forbidden)
        printf 'HTTP/2 403\r\n\r\n' >"$dump_header"
        printf '403'
        ;;
      *)
        echo "unexpected fake GHCR mode: ${FAKE_GHCR_MODE}" >&2
        exit 1
        ;;
    esac
    ;;
  *)
    echo "unexpected fake curl URL: $url" >&2
    exit 1
    ;;
esac
CURL
chmod +x "$work_dir/bin/curl"

run_resolver() {
  local mode=$1
  local tag=${2:-v1.2.3}

  PATH="$work_dir/bin:$PATH" \
    FAKE_GHCR_MODE="$mode" \
    GHCR_USERNAME=test-user \
    GHCR_TOKEN=test-token \
    "$resolver" kaito-project/aikit/aikit "$tag"
}

if [[ $(run_resolver present) != sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ]]; then
  echo "existing GHCR tag returned an unexpected digest" >&2
  exit 1
fi
if [[ $(run_resolver absent) != absent ]]; then
  echo "missing GHCR tag was not reported as absent" >&2
  exit 1
fi
if [[ $(run_resolver present latest) != sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ]]; then
  echo "existing GHCR latest alias returned an unexpected digest" >&2
  exit 1
fi
for invalid_mode in invalid-digest forbidden malformed-token; do
  if run_resolver "$invalid_mode" >/dev/null 2>&1; then
    echo "$invalid_mode GHCR response unexpectedly passed" >&2
    exit 1
  fi
done
if GHCR_USERNAME=test-user GHCR_TOKEN=test-token \
  "$resolver" kaito-project/aikit/aikit v1.2.3-rc.1 >/dev/null 2>&1; then
  echo "prerelease GHCR tag unexpectedly passed" >&2
  exit 1
fi

echo "GHCR tag resolver tests passed"
