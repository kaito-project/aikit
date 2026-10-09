#!/bin/bash
set -euo pipefail

catalog="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/models/catalog.json"

case "${1:-}" in
    --models)
        jq -ce --argjson requested "${2:-[]}" '
            if ($requested | type) != "array" then
                error("models must be a JSON array")
            elif any($requested[]; type != "string") then
                error("model IDs must be strings")
            elif ($requested | length) == 0 then
                keys_unsorted
            else
                ($requested - keys) as $unknown
                | if ($unknown | length) > 0 then
                    error("unknown or retired model IDs: \($unknown | join(", "))")
                  else
                    $requested | unique
                  end
            end
        ' "$catalog"
        ;;
    --images)
        jq -ce '[.[] | "ghcr.io/kaito-project/aikit/\(.image):\(.tag)"]' "$catalog"
        ;;
    "")
        echo "Usage: $0 --models [JSON_ARRAY] | --images | MODEL_ID" >&2
        exit 1
        ;;
    *)
        jq -er --arg model "$1" '
            .[$model] // error("unknown or retired model ID: \($model)")
            | "MODEL_NAME=\(.image)",
              "MODEL_SIZE=\(.tag)",
              "MODEL_ALIAS=\(.alias // "")",
              "MODEL_PLATFORMS=\(.platforms // "linux/amd64,linux/arm64")"
        ' "$catalog"
        ;;
esac
