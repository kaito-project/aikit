#!/usr/bin/env bash

set -euo pipefail

if (( $# > 1 )); then
  echo "usage: $0 [benchmark-output]" >&2
  exit 2
fi

readonly benchmark_package='./pkg/aikit2llb/inference'
readonly benchmark_name='^BenchmarkAikit2LLBMarshal$'
readonly benchmark_samples=3

output_file=${1:-}
temporary_output=''
report_file=$(mktemp)

trap 'rm -f -- "$report_file"; if [[ -n "$temporary_output" ]]; then rm -f -- "$temporary_output"; fi' EXIT

if [[ -z "$output_file" ]]; then
  output_file=$(mktemp)
  temporary_output=$output_file
else
  mkdir -p -- "$(dirname -- "$output_file")"
fi

# Run one iteration per sample; wall-clock and allocation metrics are diagnostic only.
# Repeat samples to detect accidental graph-metric instability.
go test "$benchmark_package" \
  -run '^$' \
  -bench "$benchmark_name" \
  -benchmem \
  -benchtime=1x \
  -count="$benchmark_samples" \
  2>&1 | tee "$output_file"

set +e
awk -v expected_samples="$benchmark_samples" -v report_file="$report_file" '
  BEGIN {
    names[1] = "LocalModel"
    names[2] = "LlamaFixture"
    names[3] = "ManyPromptTemplates"
    benchmark_count = 3

    # Leave deliberate headroom over the compact graph. Digest-qualified catalog
    # source identifiers add a constant payload without adding graph operations.
    # Reject the pre-compaction shape: 21 and 117 ops, with 76,639 bytes for 100 templates.
    min_ops["LocalModel"] = 15
    min_ops["LlamaFixture"] = 15
    min_ops["ManyPromptTemplates"] = 15
    max_ops["LocalModel"] = 20
    max_ops["LlamaFixture"] = 20
    max_ops["ManyPromptTemplates"] = 20
    min_bytes["LocalModel"] = 2400
    min_bytes["LlamaFixture"] = 4800
    min_bytes["ManyPromptTemplates"] = 60000
    max_bytes["LocalModel"] = 3600
    max_bytes["LlamaFixture"] = 6400
    max_bytes["ManyPromptTemplates"] = 75000
  }

  $1 ~ /^BenchmarkAikit2LLBMarshal\// {
    name = $1
    sub(/^BenchmarkAikit2LLBMarshal\//, "", name)
    sub(/-[0-9]+$/, "", name)

    found_ops = 0
    found_bytes = 0
    for (field = 2; field < NF; field++) {
      if ($(field + 1) == "ops/graph") {
        current_ops = $field + 0
        found_ops = 1
      } else if ($(field + 1) == "opbytes/graph") {
        current_bytes = $field + 0
        found_bytes = 1
      }
    }

    if (!found_ops || !found_bytes) {
      malformed[name]++
      next
    }

    samples[name]++
    if (samples[name] == 1) {
      observed_ops[name] = current_ops
      observed_bytes[name] = current_bytes
    } else if (observed_ops[name] != current_ops || observed_bytes[name] != current_bytes) {
      unstable[name] = 1
    }
  }

  END {
    print "LLB graph benchmark structural budgets:"
    print ""
    printf "| Benchmark | Samples | ops/graph | Allowed | opbytes/graph | Allowed | Result |\n" > report_file
    printf "| --- | ---: | ---: | ---: | ---: | ---: | --- |\n" >> report_file

    failed = 0
    for (benchmark_index = 1; benchmark_index <= benchmark_count; benchmark_index++) {
      name = names[benchmark_index]
      result = "pass"

      if (samples[name] != expected_samples) {
        result = "fail: missing samples"
        printf "ERROR: %s produced %d complete samples; expected %d.\n", name, samples[name], expected_samples > "/dev/stderr"
        failed = 1
      }
      if (malformed[name] > 0) {
        result = "fail: malformed metrics"
        printf "ERROR: %s produced %d samples without both graph metrics.\n", name, malformed[name] > "/dev/stderr"
        failed = 1
      }
      if (unstable[name]) {
        result = "fail: unstable metrics"
        printf "ERROR: %s graph metrics changed between benchmark samples.\n", name > "/dev/stderr"
        failed = 1
      }
      if (samples[name] > 0 && observed_ops[name] < min_ops[name]) {
        result = "fail: operation floor"
        printf "ERROR: %s uses %.0f ops/graph; minimum is %.0f.\n", name, observed_ops[name], min_ops[name] > "/dev/stderr"
        failed = 1
      }
      if (samples[name] > 0 && observed_ops[name] > max_ops[name]) {
        result = "fail: operation ceiling"
        printf "ERROR: %s uses %.0f ops/graph; maximum is %.0f.\n", name, observed_ops[name], max_ops[name] > "/dev/stderr"
        failed = 1
      }
      if (samples[name] > 0 && observed_bytes[name] < min_bytes[name]) {
        result = "fail: byte floor"
        printf "ERROR: %s uses %.0f opbytes/graph; minimum is %.0f.\n", name, observed_bytes[name], min_bytes[name] > "/dev/stderr"
        failed = 1
      }
      if (samples[name] > 0 && observed_bytes[name] > max_bytes[name]) {
        result = "fail: byte ceiling"
        printf "ERROR: %s uses %.0f opbytes/graph; maximum is %.0f.\n", name, observed_bytes[name], max_bytes[name] > "/dev/stderr"
        failed = 1
      }

      printf "  %-24s samples=%d ops=%g[%g,%g] bytes=%g[%g,%g] %s\n", name, samples[name], observed_ops[name], min_ops[name], max_ops[name], observed_bytes[name], min_bytes[name], max_bytes[name], result
      printf "| `%s` | %d | %.0f | %.0f-%.0f | %.0f | %.0f-%.0f | %s |\n", name, samples[name], observed_ops[name], min_ops[name], max_ops[name], observed_bytes[name], min_bytes[name], max_bytes[name], result >> report_file
    }

    # Allow at most one constant graph operation for the large prompt-template fixture.
    # Catch per-template graph growth without timing noise.
    if (samples["LlamaFixture"] > 0 && samples["ManyPromptTemplates"] > 0) {
      prompt_op_delta = observed_ops["ManyPromptTemplates"] - observed_ops["LlamaFixture"]
      prompt_delta_result = "pass"
      if (prompt_op_delta > 1) {
        prompt_delta_result = "fail"
        printf "ERROR: ManyPromptTemplates adds %.0f ops over LlamaFixture; at most 1 is allowed.\n", prompt_op_delta > "/dev/stderr"
        failed = 1
      }
      printf "  %-24s delta=%g/1 %s\n", "PromptTemplateOpGrowth", prompt_op_delta, prompt_delta_result
      printf "\nPrompt-template operation delta: **%.0f** (budget: at most 1) - %s.\n", prompt_op_delta, prompt_delta_result >> report_file
    }

    exit failed
  }
' "$output_file"
validation_status=$?
set -e

if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
  {
    echo '### LLB graph benchmark'
    echo
    echo 'Wall-clock and allocation values are recorded for diagnostics but are not gated.'
    echo
    cat "$report_file"
    echo
    echo '<details><summary>Raw benchmark output</summary>'
    echo
    echo '```text'
    cat "$output_file"
    echo '```'
    echo '</details>'
  } >> "$GITHUB_STEP_SUMMARY"
fi

exit "$validation_status"
