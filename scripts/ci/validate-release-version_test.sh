#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
validator="$script_dir/validate-release-version.sh"
work_dir=$(mktemp -d)
trap 'rm -rf "$work_dir"' EXIT

git init --quiet --bare "$work_dir/origin.git"
git init --quiet -b main "$work_dir/repository"
git -C "$work_dir/repository" config user.email test@example.com
git -C "$work_dir/repository" config user.name "Release Version Test"
printf '%s\n' test >"$work_dir/repository/file"
git -C "$work_dir/repository" add file
git -C "$work_dir/repository" commit --quiet -m initial
git -C "$work_dir/repository" remote add origin "$work_dir/origin.git"
for tag in v0.21.2 v0.22.0 v9.0.0-rc.1; do
  git -C "$work_dir/repository" tag "$tag"
done
git -C "$work_dir/repository" push --quiet origin main --tags

run_validator() {
  local version=$1
  local state=$2
  local output_file=$3
  local release_branch_commit=${4:-}
  local arguments=("$version" "$state")

  if [[ -n $release_branch_commit ]]; then
    arguments+=("$release_branch_commit")
  fi

  (
    cd "$work_dir/repository"
    RELEASE_REMOTE=origin GITHUB_OUTPUT="$output_file" "$validator" "${arguments[@]}"
  )
}

assert_output() {
  local output_file=$1
  local expected=$2

  if ! grep -qxF "$expected" "$output_file"; then
    echo "missing output '$expected' in $output_file" >&2
    exit 1
  fi
}

output_file="$work_dir/new-latest-output"
run_validator v0.23.0 new "$output_file" >/dev/null
assert_output "$output_file" 'tag_exists=false'
assert_output "$output_file" 'latest_stable=v0.22.0'
assert_output "$output_file" 'latest_release_line=none'
assert_output "$output_file" 'publish_latest=true'

output_file="$work_dir/maintenance-output"
run_validator v0.21.3 new "$output_file" >/dev/null
assert_output "$output_file" 'tag_exists=false'
assert_output "$output_file" 'latest_release_line=v0.21.2'
assert_output "$output_file" 'publish_latest=false'

output_file="$work_dir/existing-latest-output"
run_validator v0.22.0 existing "$output_file" >/dev/null
assert_output "$output_file" 'tag_exists=true'
assert_output "$output_file" 'latest_release_line=v0.22.0'
assert_output "$output_file" 'publish_latest=true'

output_file="$work_dir/existing-older-output"
run_validator v0.21.2 either "$output_file" >/dev/null
assert_output "$output_file" 'tag_exists=true'
assert_output "$output_file" 'latest_release_line=v0.21.2'
assert_output "$output_file" 'publish_latest=false'

release_branch_commit=$(git -C "$work_dir/repository" rev-parse HEAD)
run_validator v0.21.3 new "$work_dir/restored-branch-output" "$release_branch_commit" >/dev/null
run_validator v0.23.0 new "$work_dir/new-line-output" missing >/dev/null
run_validator v9.0.0 new "$work_dir/prerelease-only-line-output" missing >/dev/null

if run_validator v0.21.3 new "$work_dir/missing-release-line-output" missing >/dev/null 2>&1; then
  echo "missing previously released branch unexpectedly passed" >&2
  exit 1
fi

release_tree=$(git -C "$work_dir/repository" rev-parse 'HEAD^{tree}')
unrelated_commit=$(printf 'unrelated release history\n' | git -C "$work_dir/repository" commit-tree "$release_tree")
if run_validator v0.21.3 new "$work_dir/unrelated-branch-output" "$unrelated_commit" >/dev/null 2>&1; then
  echo "release branch unrelated to the latest line tag unexpectedly passed" >&2
  exit 1
fi

git -C "$work_dir/repository" tag -a v0.21.4 -m v0.21.4
git -C "$work_dir/repository" push --quiet origin v0.21.4
output_file="$work_dir/annotated-line-output"
run_validator v0.21.5 new "$output_file" "$release_branch_commit" >/dev/null
assert_output "$output_file" 'latest_release_line=v0.21.4'

run_validator v0.24.0 new "$work_dir/pre-race-output" missing >/dev/null
git -C "$work_dir/repository" tag v0.24.0
git -C "$work_dir/repository" push --quiet origin v0.24.0
if run_validator v0.24.1 new "$work_dir/post-race-output" missing >/dev/null 2>&1; then
  echo "new same-line tag appearing before branch creation unexpectedly passed" >&2
  exit 1
fi

if run_validator v0.21.1 new "$work_dir/lower-patch-output" >/dev/null 2>&1; then
  echo "lower unused patch version unexpectedly passed" >&2
  exit 1
fi
if run_validator v0.21.2 new "$work_dir/duplicate-output" >/dev/null 2>&1; then
  echo "existing version unexpectedly passed as new" >&2
  exit 1
fi
if run_validator v0.22.1 existing "$work_dir/missing-output" >/dev/null 2>&1; then
  echo "missing version unexpectedly passed as existing" >&2
  exit 1
fi
if run_validator v00.22.1 new "$work_dir/invalid-output" >/dev/null 2>&1; then
  echo "invalid version unexpectedly passed" >&2
  exit 1
fi

oci_tag_digits=
for _ in {1..123}; do
  oci_tag_digits+="1"
done
run_validator "v${oci_tag_digits}.0.0" new "$work_dir/max-oci-tag-output" >/dev/null
if run_validator "v${oci_tag_digits}1.0.0" new "$work_dir/long-oci-tag-output" >/dev/null 2>&1; then
  echo "version exceeding the OCI tag limit unexpectedly passed" >&2
  exit 1
fi

echo "release version validator tests passed"
