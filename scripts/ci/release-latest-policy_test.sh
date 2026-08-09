#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(cd "$script_dir/../.." && pwd)
artifact_publisher="$repository_root/.github/workflows/release.yaml"
runner_publisher="$repository_root/.github/workflows/release-runners.yaml"
reconciler="$repository_root/.github/workflows/reconcile-release-latest.yaml"
release_publisher="$repository_root/.github/workflows/publish-release.yaml"
literal_dollar='$'
publisher_candidate="candidate-${literal_dollar}{{ github.run_id }}-${literal_dollar}{{ github.run_attempt }}"
publisher_chart_artifact="helm-chart-${literal_dollar}{{ github.run_id }}-${literal_dollar}{{ github.run_attempt }}"
app_reconciler_candidate="candidate-${literal_dollar}{RUN_ID}-${literal_dollar}{RUN_ATTEMPT}"
reconciler_chart_artifact="artifact_name=\"helm-chart-${literal_dollar}{RUN_ID}-${literal_dollar}{RUN_ATTEMPT}\""
runner_reconciler_candidate="candidate-${literal_dollar}{RUN_ID}-${literal_dollar}{candidate_attempt}"
reconciler_version_tag="--tag \"${literal_dollar}{image}:${literal_dollar}{VERSION}\""
reconciler_latest_tag="--tag \"${literal_dollar}{image}:latest\""
release_commit_validator_arg="validator_args+=(\"${literal_dollar}{RELEASE_COMMIT}\")"
trusted_main_environment="TRUSTED_MAIN_COMMIT: ${literal_dollar}{{ github.sha }}"
trusted_main_check="if [[ \"${literal_dollar}{main_commit}\" != \"${literal_dollar}{TRUSTED_MAIN_COMMIT}\" ]]"

extract_job() {
  local job=$1
  awk -v job="  ${job}:" '
    $0 == job {
      printing = 1
    }
    printing && $0 != job && $0 ~ /^  [-_a-zA-Z0-9]+:$/ {
      exit
    }
    printing {
      print
    }
  ' "$reconciler"
}

promote_app_job=$(extract_job promote-app-version)
publish_app_job=$(extract_job publish-app-release)
reconcile_app_job=$(extract_job reconcile-app)

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

if grep -qF -- '- name: Publish Helm chart' "$artifact_publisher" || \
  grep -qF -- '- name: Create GitHub release' "$artifact_publisher"; then
  echo "tag workflows must not publish public Helm charts or GitHub Releases" >&2
  exit 1
fi
if grep -q 'stefanprodan/helm-gh-pages' "$artifact_publisher" "$reconciler"; then
  echo "release workflows must not use the mutable, overwrite-on-retry Helm publisher" >&2
  exit 1
fi
if ! grep -qF -- '- name: Package Helm chart candidate' "$artifact_publisher" || \
  ! grep -qF -- '- name: Upload Helm chart candidate' "$artifact_publisher" || \
  ! grep -qF -- "$publisher_chart_artifact" "$artifact_publisher" || \
  ! grep -qF -- "$reconciler_chart_artifact" "$reconciler"; then
  echo "Helm chart candidates must be bound to the exact successful workflow attempt" >&2
  exit 1
fi
if ! grep -q -- '--latest=false' "$reconciler"; then
  echo "app promotion must create GitHub Releases with Latest disabled" >&2
  exit 1
fi
image_publish_line=$(grep -nF -- '- name: Publish the immutable app version' "$reconciler" | cut -d: -f1)
helm_publish_line=$(grep -nF -- '- name: Publish Helm chart' "$reconciler" | cut -d: -f1)
github_release_line=$(grep -nF -- '- name: Publish stable GitHub release' "$reconciler" | cut -d: -f1)
if [[ -z $image_publish_line || -z $helm_publish_line || -z $github_release_line ]] || \
  ((helm_publish_line <= image_publish_line || github_release_line <= helm_publish_line)); then
  echo "trusted promotion must publish immutable image, Helm chart, then GitHub Release" >&2
  exit 1
fi
if [[ -z $promote_app_job || -z $publish_app_job || -z $reconcile_app_job ]] || \
  ! grep -q 'packages: write' <<<"$promote_app_job" || \
  grep -q 'contents: write' <<<"$promote_app_job" || \
  ! grep -q 'contents: write' <<<"$publish_app_job" || \
  grep -q 'packages: write' <<<"$publish_app_job"; then
  echo "immutable image and Helm/GitHub publication must use separate write tokens" >&2
  exit 1
fi
if ! grep -qF -- 'git/refs/heads/gh-pages' <<<"$publish_app_job" || \
  ! grep -qF -- '-F force=false' <<<"$publish_app_job" || \
  ! grep -qF -- 'already exists with different bytes or type' <<<"$publish_app_job"; then
  echo "Helm publication must atomically update gh-pages and reject version overwrites" >&2
  exit 1
fi
if [[ $(grep -cF '.draft == false and' "$reconciler") -lt 3 ]]; then
  echo "existing, newly created, and latest-selected GitHub Releases must be stable" >&2
  exit 1
fi
if ! grep -q 'workflow_run:' "$reconciler" || \
  ! grep -q 'select-latest-release.sh' "$reconciler"; then
  echo "trusted latest reconciliation trigger or selector is missing" >&2
  exit 1
fi
if ! grep -qF "$publisher_candidate" "$artifact_publisher" || \
  ! grep -qF "$publisher_candidate" "$runner_publisher" || \
  ! grep -qF "$app_reconciler_candidate" "$reconciler" || \
  ! grep -qF "$runner_reconciler_candidate" "$reconciler"; then
  echo "publisher candidates must be bound to exact workflow attempts" >&2
  exit 1
fi
if [[ $(grep -c 'select-runner-candidates.sh' "$reconciler") -ne 2 ]]; then
  echo "exact and latest runner promotion must bind every image to successful matrix attempts" >&2
  exit 1
fi
for annotation in release-artifact release-commit release-run-attempt release-run-id; do
  if ! grep -q -- "-a .*${annotation}" "$artifact_publisher" || \
    ! grep -q -- "-a .*${annotation}" "$runner_publisher" || \
    ! grep -q -- "-a .*${annotation}" "$reconciler"; then
    echo "candidate signatures must bind ${annotation}" >&2
    exit 1
  fi
done
if grep -q 'resolve-ghcr-tag-digest.sh' "$artifact_publisher" || \
  grep -q 'resolve-ghcr-tag-digest.sh' "$runner_publisher"; then
  echo "tag workflows must stage candidates without publishing public version tags" >&2
  exit 1
fi
if ! grep -q '^  promote-app-version:' "$reconciler" || \
  ! grep -q '^  publish-app-release:' "$reconciler" || \
  ! grep -q '^  promote-runner-version:' "$reconciler" || \
  [[ $(grep -c "github.event.workflow_run.conclusion == 'success'" "$reconciler") -ne 3 ]]; then
  echo "immutable versions must be promoted only after the exact publisher run succeeds" >&2
  exit 1
fi
if [[ $(grep -cF -- "$reconciler_version_tag" "$reconciler") -ne 2 ]]; then
  echo "app and runner exact-trigger jobs must publish immutable version tags" >&2
  exit 1
fi
if [[ $(grep -cF -- "$reconciler_latest_tag" "$reconciler") -ne 2 ]]; then
  echo "app and runner global reconciliation must publish their latest aliases" >&2
  exit 1
fi
if ! grep -q -- '- promote-app-version' <<<"$reconcile_app_job" || \
  ! grep -q -- '- publish-app-release' <<<"$reconcile_app_job" || \
  ! grep -q 'needs: promote-runner-version' "$reconciler" || \
  [[ $(grep -c '^[[:space:]]*always() &&' "$reconciler") -ne 2 ]]; then
  echo "global latest reconciliation must run independently after exact promotion" >&2
  exit 1
fi
if [[ $(grep -c 'EXPECTED_RUN_ID:.*github.event.workflow_run.id' "$reconciler") -ne 2 ]] || \
  [[ $(grep -c 'EXPECTED_RUN_ATTEMPT:.*github.event.workflow_run.run_attempt' "$reconciler") -ne 2 ]]; then
  echo "exact promotion must use the triggering workflow run and attempt" >&2
  exit 1
fi
if [[ $(grep -c 'group: release-artifacts' "$artifact_publisher") -ne 1 ]] || \
  [[ $(grep -c 'group: release-artifacts' "$reconciler") -ne 3 ]] || \
  [[ $(grep -c 'group: release-runner-images' "$runner_publisher") -ne 1 ]] || \
  [[ $(grep -c 'group: release-runner-images' "$reconciler") -ne 2 ]]; then
  echo "publishers and reconcilers must share serialization groups" >&2
  exit 1
fi
if [[ $(grep -cF "$release_commit_validator_arg" "$release_publisher") -ne 2 ]]; then
  echo "new release validation must bind release-line ancestry to the selected commit before and after approval" >&2
  exit 1
fi
if [[ $(grep -cF "$trusted_main_environment" "$release_publisher") -ne 1 ]] || \
  ! grep -qF "$trusted_main_check" "$release_publisher"; then
  echo "privileged version sync must reject a stale trusted main snapshot before minting its App token" >&2
  exit 1
fi

echo "release latest policy tests passed"
