#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <image> <artifact-directory>" >&2
  exit 2
fi

image=$1
mkdir -p "$2"
artifact_dir=$(cd "$2" && pwd)
container_id=
cleanup() {
  if [[ -n $container_id ]]; then
    docker logs "$container_id" > "$artifact_dir/server.log" 2>&1 || true
    docker rm -f "$container_id" >/dev/null || true
  fi
}
trap cleanup EXIT

# An empty cache and no network require the preset to supply the whole pipeline.
container_id=$(docker run -d --gpus all --network none \
  --tmpfs /tmp/flux2-hf-cache \
  -e HF_HOME=/tmp/flux2-hf-cache \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  --mount "type=bind,source=$artifact_dir,target=/tmp/flux2-smoke" \
  "$image")

# Use the backend's Python to call the API over the container's loopback device.
docker exec -i "$container_id" /backends/cuda12-diffusers/venv/bin/python - <<'PY'
import base64
import io
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image

artifact_dir = Path("/tmp/flux2-smoke")
endpoint = "http://127.0.0.1:8080/v1"
deadline = time.monotonic() + 120
while True:
    try:
        with urllib.request.urlopen(f"{endpoint}/models", timeout=5) as response:
            models = json.load(response)
        break
    except (urllib.error.URLError, TimeoutError):
        if time.monotonic() >= deadline:
            raise SystemExit("LocalAI did not become ready within 120 seconds")
        time.sleep(2)

artifact_dir.joinpath("models.json").write_text(json.dumps(models, indent=2) + "\n")
if not any(model.get("id") == "flux-2-klein-4b" for model in models.get("data", [])):
    raise SystemExit("FLUX.2 Klein preset is missing from /v1/models")

request = urllib.request.Request(
    f"{endpoint}/images/generations",
    data=json.dumps({
        "model": "flux-2-klein-4b",
        "prompt": "A red apple on a wooden table in daylight",
        "size": "512x512",
        "n": 1,
        "response_format": "b64_json",
        "seed": 42,
    }).encode(),
    headers={"Content-Type": "application/json"},
)
started = time.monotonic()
try:
    with urllib.request.urlopen(request, timeout=900) as response:
        result = json.load(response)
except urllib.error.HTTPError as error:
    artifact_dir.joinpath("error.txt").write_bytes(error.read())
    raise

artifact_dir.joinpath("response.json").write_text(json.dumps(result, indent=2) + "\n")
if result.get("error") or not isinstance(result.get("data"), list) or len(result["data"]) != 1:
    raise SystemExit("Expected exactly one generated image; see response.json")

image_bytes = base64.b64decode(result["data"][0]["b64_json"], validate=True)
with Image.open(io.BytesIO(image_bytes)) as generated:
    generated.load()
    if generated.format != "PNG" or generated.size != (512, 512):
        raise SystemExit(f"Unexpected output: {generated.format}, {generated.size}")
    if not any(low < high for low, high in generated.convert("RGB").getextrema()):
        raise SystemExit("Generated image is a solid color")

artifact_dir.joinpath("generated.png").write_bytes(image_bytes)
print(f"Generated a 512x512 PNG offline in {time.monotonic() - started:.1f}s")
PY
