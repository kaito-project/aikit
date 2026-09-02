#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <ghcr-repository> <latest-or-stable-version-tag>" >&2
  exit 2
fi

repository=$1
tag=$2

fail() {
  echo "GHCR tag resolution failed: $*" >&2
  exit 1
}

if ! [[ $repository =~ ^[a-z0-9._-]+(/[a-z0-9._-]+)+$ ]]; then
  fail "invalid repository path: $repository"
fi
if [[ $tag != latest ]] && \
  ! [[ $tag =~ ^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$ ]]; then
  fail "invalid release tag: $tag"
fi
if [[ -z ${GHCR_USERNAME:-} || -z ${GHCR_TOKEN:-} ]]; then
  fail "GHCR_USERNAME and GHCR_TOKEN are required"
fi
if ! command -v curl >/dev/null 2>&1 || ! command -v jq >/dev/null 2>&1; then
  fail "curl and jq are required"
fi

if ! registry_token=$(curl --fail --silent --show-error --get \
  --user "${GHCR_USERNAME}:${GHCR_TOKEN}" \
  --data-urlencode service=ghcr.io \
  --data-urlencode "scope=repository:${repository}:pull" \
  https://ghcr.io/token | \
  jq -er '(.token // .access_token) | select(type == "string" and length > 0)'); then
  fail "could not obtain a scoped registry token"
fi

response_headers=$(mktemp)
trap 'rm -f "$response_headers"' EXIT
if ! http_status=$(curl --silent --show-error --head \
  --output /dev/null \
  --dump-header "$response_headers" \
  --write-out '%{http_code}' \
  --header "Authorization: Bearer ${registry_token}" \
  --header 'Accept: application/vnd.oci.image.index.v1+json, application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.docker.distribution.manifest.v2+json' \
  "https://ghcr.io/v2/${repository}/manifests/${tag}"); then
  fail "registry manifest request failed"
fi

case $http_status in
  200)
    digest=$(tr -d '\r' <"$response_headers" | awk '
      tolower($0) ~ /^docker-content-digest:[[:space:]]*/ {
        line = $0
        sub(/^[^:]*:[[:space:]]*/, "", line)
        value = line
      }
      END { print value }
    ')
    if ! [[ $digest =~ ^sha256:[0-9a-f]{64}$ ]]; then
      fail "registry returned an invalid manifest digest"
    fi
    printf '%s\n' "$digest"
    ;;
  404)
    printf '%s\n' absent
    ;;
  *)
    fail "registry returned HTTP $http_status"
    ;;
esac
