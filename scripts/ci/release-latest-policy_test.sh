#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(cd "$script_dir/../.." && pwd)
artifact_publisher="$repository_root/.github/workflows/release.yaml"
runner_publisher="$repository_root/.github/workflows/release-runners.yaml"
reconciler="$repository_root/.github/workflows/reconcile-release-latest.yaml"

for publisher in "$artifact_publisher" "$runner_publisher"; do
  if grep -q ':latest' "$publisher"; then
    echo "$publisher must not write mutable latest aliases" >&2
    exit 1
  fi
  if grep -q 'make_latest=true' "$publisher"; then
    echo "$publisher must not assign GitHub Latest" >&2
    exit 1
  fi
done

if ! grep -q -- '--latest=false' "$artifact_publisher"; then
  echo "artifact publisher must create GitHub releases with Latest disabled" >&2
  exit 1
fi
if ! grep -q 'workflow_run:' "$reconciler" || \
  ! grep -q 'select-latest-release.sh' "$reconciler"; then
  echo "trusted latest reconciliation trigger or selector is missing" >&2
  exit 1
fi

echo "release latest policy tests passed"
