#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 1 ]]; then
  manifest_paths=(Makefile charts/aikit/Chart.yaml)
elif [[ $# -ge 3 && $((($# - 1) % 2)) -eq 0 ]]; then
  manifest_paths=("${@:2}")
else
  echo "Usage: $0 <release-version> [<Makefile> <Chart.yaml> ...]" >&2
  exit 2
fi

release_version=$1

fail() {
  echo "Release sync planning failed: $*" >&2
  exit 1
}

strip_matching_quotes() {
  local value=$1
  local first_character
  local last_character

  if [[ ${#value} -ge 2 ]]; then
    first_character=${value:0:1}
    last_character=${value: -1}
    if [[ $first_character == '"' && $last_character == '"' ]] || [[ $first_character == "'" && $last_character == "'" ]]; then
      value=${value:1:${#value}-2}
    fi
  fi

  printf '%s' "$value"
}

parse_version() {
  local value=$1
  local description=$2

  if ! [[ $value =~ ^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$ ]]; then
    fail "$description must use stable semantic version form vX.Y.Z: ${value:-missing}"
  fi

  parsed_major=${BASH_REMATCH[1]}
  parsed_minor=${BASH_REMATCH[2]}
  parsed_patch=${BASH_REMATCH[3]}
}

read_manifest_version() {
  local makefile_path=$1
  local chart_path=$2
  local description=$3
  local makefile_version_count
  local chart_version_count
  local chart_app_version_count
  local chart_version
  local chart_app_version

  if [[ ! -f $makefile_path ]]; then
    fail "$description Makefile is missing: $makefile_path"
  fi
  makefile_version_count=$(awk '/^[[:space:]]*VERSION[[:space:]]*:=/ { count++ } END { print count + 0 }' "$makefile_path")
  if [[ $makefile_version_count -ne 1 ]]; then
    fail "$description Makefile must contain exactly one VERSION assignment; found $makefile_version_count"
  fi
  manifest_version=$(awk '
    /^[[:space:]]*VERSION[[:space:]]*:=/ {
      value = $0
      sub(/^[[:space:]]*VERSION[[:space:]]*:=[[:space:]]*/, "", value)
      sub(/[[:space:]]*$/, "", value)
      print value
    }
  ' "$makefile_path")
  parse_version "$manifest_version" "$description VERSION"

  if [[ ! -f $chart_path ]]; then
    fail "$description Helm chart is missing: $chart_path"
  fi
  chart_version_count=$(awk '/^version:[[:space:]]+/ { count++ } END { print count + 0 }' "$chart_path")
  chart_app_version_count=$(awk '/^appVersion:[[:space:]]+/ { count++ } END { print count + 0 }' "$chart_path")
  if [[ $chart_version_count -ne 1 ]]; then
    fail "$description Helm chart must contain exactly one top-level version; found $chart_version_count"
  fi
  if [[ $chart_app_version_count -ne 1 ]]; then
    fail "$description Helm chart must contain exactly one top-level appVersion; found $chart_app_version_count"
  fi
  chart_version=$(awk '
    /^version:[[:space:]]+/ {
      value = $0
      sub(/^version:[[:space:]]+/, "", value)
      sub(/[[:space:]]*$/, "", value)
      print value
    }
  ' "$chart_path")
  chart_app_version=$(awk '
    /^appVersion:[[:space:]]+/ {
      value = $0
      sub(/^appVersion:[[:space:]]+/, "", value)
      sub(/[[:space:]]*$/, "", value)
      print value
    }
  ' "$chart_path")
  chart_version=$(strip_matching_quotes "$chart_version")
  chart_app_version=$(strip_matching_quotes "$chart_app_version")
  if [[ $chart_version != "${manifest_version#v}" || $chart_app_version != "$manifest_version" ]]; then
    fail "$description version manifests are inconsistent"
  fi
}

compare_decimal_components() {
  local left=$1
  local right=$2
  local LC_ALL=C

  comparison=0
  if ((${#left} > ${#right})); then
    comparison=1
  elif ((${#left} < ${#right})); then
    comparison=-1
  elif [[ $left > $right ]]; then
    comparison=1
  elif [[ $left < $right ]]; then
    comparison=-1
  fi
}

compare_release_lines() {
  compare_decimal_components "$1" "$3"
  if ((comparison != 0)); then
    return
  fi
  compare_decimal_components "$2" "$4"
}

compare_versions() {
  compare_decimal_components "$1" "$4"
  if ((comparison != 0)); then
    return
  fi
  compare_decimal_components "$2" "$5"
  if ((comparison != 0)); then
    return
  fi
  compare_decimal_components "$3" "$6"
}

parse_version "$release_version" "release version"
release_major=$parsed_major
release_minor=$parsed_minor

manifest_versions=()
manifest_majors=()
manifest_minors=()
manifest_patches=()
for ((index = 0; index < ${#manifest_paths[@]}; index += 2)); do
  target_number=$((index / 2 + 1))
  read_manifest_version "${manifest_paths[index]}" "${manifest_paths[index + 1]}" "target $target_number"
  manifest_versions+=("$manifest_version")
  manifest_majors+=("$parsed_major")
  manifest_minors+=("$parsed_minor")
  manifest_patches+=("$parsed_patch")
done

main_version=${manifest_versions[0]}
main_major=${manifest_majors[0]}
main_minor=${manifest_minors[0]}

pending_index=-1
for ((index = 1; index < ${#manifest_versions[@]}; index++)); do
  if ((pending_index == -1)); then
    pending_index=$index
  else
    compare_versions \
      "${manifest_majors[index]}" "${manifest_minors[index]}" "${manifest_patches[index]}" \
      "${manifest_majors[pending_index]}" "${manifest_minors[pending_index]}" "${manifest_patches[pending_index]}"
    if ((comparison > 0)); then
      pending_index=$index
    fi
  fi
done

action=none
target_version=$main_version
if ((pending_index >= 0)); then
  compare_release_lines \
    "${manifest_majors[pending_index]}" "${manifest_minors[pending_index]}" \
    "$main_major" "$main_minor"
  pending_vs_main=$comparison
else
  pending_vs_main=-1
fi

if ((pending_vs_main > 0)); then
  compare_release_lines \
    "$release_major" "$release_minor" \
    "${manifest_majors[pending_index]}" "${manifest_minors[pending_index]}"
  if ((comparison > 0)); then
    action=update
    target_version=$release_version
  else
    action=ensure
    target_version=${manifest_versions[pending_index]}
  fi
else
  compare_release_lines "$release_major" "$release_minor" "$main_major" "$main_minor"
  if ((comparison > 0)); then
    action=update
    target_version=$release_version
  fi
fi

needed=false
if [[ $action != none ]]; then
  needed=true
fi

printf '%s\n' "$action"
if [[ -n ${GITHUB_OUTPUT:-} ]]; then
  {
    echo "action=$action"
    echo "needed=$needed"
    echo "target_version=$target_version"
  } >>"$GITHUB_OUTPUT"
fi
