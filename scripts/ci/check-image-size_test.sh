#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
checker="$script_dir/check-image-size.sh"
work_dir=$(mktemp -d)
trap 'rm -rf "$work_dir"' EXIT
mkdir -p "$work_dir/bin"

cat > "$work_dir/bin/docker" <<'DOCKER'
#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-} == buildx && ${2:-} == version ]]; then
  exit 0
fi
if [[ ${1:-} == buildx && ${2:-} == imagetools && ${3:-} == inspect && ${4:-} == --raw ]]; then
  cat "$FAKE_MANIFEST"
  exit 0
fi
echo "unexpected fake docker invocation: $*" >&2
exit 1
DOCKER
chmod +x "$work_dir/bin/docker"

run_checker() {
  local manifest=$1
  PATH="$work_dir/bin:$PATH" FAKE_MANIFEST="$manifest" \
    "$checker" example.invalid/image:latest linux/amd64 1
}

valid_manifest="$work_dir/valid.json"
printf '%s\n' '{"schemaVersion":2,"layers":[{"size":100},{"size":200}]}' > "$valid_manifest"
valid_output=$(run_checker "$valid_manifest")
if [[ $valid_output != *"300 bytes"* ]]; then
  echo "valid manifest returned unexpected output: $valid_output" >&2
  exit 1
fi

invalid_manifests=(
  '{}'
  '{"layers":null}'
  '{"layers":[]}'
  '{"layers":[{}]}'
  '{"layers":[{"size":"100"}]}'
  '{"layers":[{"size":1.5}]}'
  '{"layers":[{"size":-1}]}'
)

for index in "${!invalid_manifests[@]}"; do
  manifest="$work_dir/invalid-$index.json"
  printf '%s\n' "${invalid_manifests[$index]}" > "$manifest"
  if run_checker "$manifest" >/dev/null 2>&1; then
    echo "invalid manifest unexpectedly passed: ${invalid_manifests[$index]}" >&2
    exit 1
  fi
done

echo "image size checker tests passed"
