#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
readonly ROOT_DIR
readonly FINETUNE_DIR="$ROOT_DIR/pkg/finetune"
readonly REQUIRED_UV_VERSION="0.12.1"
readonly RESOLUTION_CUTOFF="2026-08-05T00:00:00Z"
readonly LOCK_FILE="pylock.toml"

if ! command -v uv > /dev/null 2>&1; then
  echo "uv $REQUIRED_UV_VERSION is required but was not found" >&2
  exit 1
fi

installed_uv_version=$(uv --version | awk '{print $2}')
if [[ "$installed_uv_version" != "$REQUIRED_UV_VERSION" ]]; then
  echo "uv $REQUIRED_UV_VERSION is required; found $installed_uv_version" >&2
  exit 1
fi

compile_lock() {
  local directory=$1

  (
    cd "$directory"
    uv pip compile requirements.in \
      --upgrade \
      --torch-backend=cu126 \
      --python-version 3.10 \
      --python-platform x86_64-manylinux_2_28 \
      --exclude-newer "$RESOLUTION_CUTOFF" \
      --only-binary=:all: \
      --format pylock.toml \
      -o "$LOCK_FILE"
  )
}

case "${1:-update}" in
  update)
    if (( $# > 1 )); then
      echo "usage: $0 [--check]" >&2
      exit 2
    fi
    compile_lock "$FINETUNE_DIR"
    ;;
  --check)
    if (( $# != 1 )); then
      echo "usage: $0 [--check]" >&2
      exit 2
    fi

    temporary_directory=$(mktemp -d)
    readonly temporary_directory
    trap 'rm -rf -- "$temporary_directory"' EXIT

    cp "$FINETUNE_DIR/requirements.in" "$temporary_directory/requirements.in"
    compile_lock "$temporary_directory"

    if ! cmp -s "$FINETUNE_DIR/$LOCK_FILE" "$temporary_directory/$LOCK_FILE"; then
      echo "$FINETUNE_DIR/$LOCK_FILE is out of date; run make update-unsloth-lock" >&2
      exit 1
    fi
    ;;
  *)
    echo "usage: $0 [--check]" >&2
    exit 2
    ;;
esac
