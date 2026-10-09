#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <docker|podman> <image> <artifact-directory>" >&2
  exit 2
fi

engine=$1
image=$2
case "$engine" in
  docker)
    gpu_args=(--gpus all)
    device_pattern='using device CUDA[0-9]+'
    ;;
  podman)
    gpu_args=(--device /dev/dri)
    device_pattern='using device Vulkan[0-9]+ \(Virtio-GPU Venus.*Apple'
    ;;
  *)
    echo "Unsupported container engine: $engine" >&2
    exit 2
    ;;
esac

mkdir -p "$3"
artifact_dir=$(cd "$3" && pwd)
container_id=
cleanup() {
  if [[ -n $container_id ]]; then
    "$engine" logs "$container_id" > "$artifact_dir/server.log" 2>&1 || true
    "$engine" rm -f "$container_id" >/dev/null || true
  fi
}
trap cleanup EXIT

curl --fail --location --silent --show-error --max-time 60 \
  https://raw.githubusercontent.com/mudler/parakeet.cpp/1bfbebfaaf493866f49597cd3b7901959d395c60/tests/fixtures/speech.wav \
  --output "$artifact_dir/reference.wav"
python3 - "$artifact_dir/reference.wav" <<'PY'
import hashlib
import pathlib
import sys

digest = hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest()
if digest != "5fceacff0315d49cb59fcc505bcecf1ed5f2f35c2897b1e65a59f30e5d922150":
    raise SystemExit("Reference WAV checksum mismatch")
PY

# Enable llama.cpp device and layer logs for the GPU assertions.
container_id=$("$engine" create "$image")
"$engine" cp "$container_id:/config.yaml" "$artifact_dir/config.yaml"
"$engine" rm "$container_id" >/dev/null
container_id=
printf '\n  options:\n    - verbosity:4\n' >> "$artifact_dir/config.yaml"

container_id=$("$engine" run -d "${gpu_args[@]}" \
  -p 127.0.0.1::8080 \
  --mount "type=bind,source=$artifact_dir/reference.wav,target=/models/reference.wav,readonly" \
  --mount "type=bind,source=$artifact_dir/config.yaml,target=/config.yaml,readonly" \
  "$image" --debug --config-file=/config.yaml)
address=$("$engine" port "$container_id" 8080/tcp)
deadline=$((SECONDS + 120))
until curl --silent --fail --max-time 5 "http://$address/v1/models" > "$artifact_dir/models.json"; do
  if [[ $SECONDS -ge $deadline || $("$engine" inspect -f '{{.State.Running}}' "$container_id") != true ]]; then
    echo "Qwen3-TTS did not become ready; see $artifact_dir/server.log" >&2
    exit 1
  fi
  sleep 2
done

curl --fail --silent --show-error --max-time 600 \
  "http://$address/v1/audio/speech" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-tts","input":"Hello from AIKit.","voice":"/models/reference.wav","response_format":"wav"}' \
  --output "$artifact_dir/speech.wav"
python3 - "$artifact_dir/speech.wav" <<'PY'
import sys
import wave

with wave.open(sys.argv[1], "rb") as audio:
    frames = audio.getnframes()
    samples = audio.readframes(frames)
    if frames == 0 or len(samples) != frames * audio.getnchannels() * audio.getsampwidth() or not any(samples):
        raise SystemExit("Generated WAV is empty, truncated, or silent")
    print(f"Generated {frames} frames at {audio.getframerate()} Hz with {audio.getnchannels()} channel(s)")
PY

"$engine" logs "$container_id" > "$artifact_dir/server.log" 2>&1
grep -E "$device_pattern" "$artifact_dir/server.log"
grep -E 'offloaded [1-9][0-9]*/[1-9][0-9]* layers to GPU' "$artifact_dir/server.log"
