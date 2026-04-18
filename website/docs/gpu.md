---
title: GPU Acceleration
---

:::note
AIKit supports NVIDIA GPU acceleration, AMD GPU acceleration via ROCm, and experimental support for Apple Silicon. Please open an issue if you'd like to see support for other GPU vendors.
:::

## NVIDIA

AIKit supports GPU accelerated inferencing with [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit). You must also have [NVIDIA Drivers](https://www.nvidia.com/en-us/drivers/unix/) installed on your host machine.

For Kubernetes, [NVIDIA GPU Operator](https://github.com/NVIDIA/gpu-operator) provides a streamlined way to install the NVIDIA drivers and container toolkit to configure your cluster to use GPUs.

To get started with GPU-accelerated inferencing, make sure to set the following in your `aikitfile` and build your model.

```yaml
runtime: cuda         # use NVIDIA CUDA runtime
```

For `llama` backend, set the following in your `config`:

```yaml
f16: true             # use float16 precision
gpu_layers: 35        # number of layers to offload to GPU
low_vram: true        # for devices with low VRAM
```

:::tip
Make sure to customize these values based on your model and GPU specs.
:::

After building the model, you can run it with [`--gpus all`](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html#gpu-enumeration) flag to enable GPU support:

```bash
# for pre-made models, replace "my-model" with the image name
docker run --rm --gpus all -p 8080:8080 my-model
```

If GPU acceleration is working, you'll see output that is similar to following in the debug logs:

```bash
5:32AM DBG GRPC(llama-2-7b-chat.Q4_K_M.gguf-127.0.0.1:43735): stderr ggml_init_cublas: found 1 CUDA devices:
5:32AM DBG GRPC(llama-2-7b-chat.Q4_K_M.gguf-127.0.0.1:43735): stderr   Device 0: Tesla T4, compute capability 7.5
...
5:32AM DBG GRPC(llama-2-7b-chat.Q4_K_M.gguf-127.0.0.1:43735): stderr llm_load_tensors: using CUDA for GPU acceleration
5:32AM DBG GRPC(llama-2-7b-chat.Q4_K_M.gguf-127.0.0.1:43735): stderr llm_load_tensors: mem required  =   70.41 MB (+ 2048.00 MB per state)
5:32AM DBG GRPC(llama-2-7b-chat.Q4_K_M.gguf-127.0.0.1:43735): stderr llm_load_tensors: offloading 32 repeating layers to GPU
5:32AM DBG GRPC(llama-2-7b-chat.Q4_K_M.gguf-127.0.0.1:43735): stderr llm_load_tensors: offloading non-repeating layers to GPU
5:32AM DBG GRPC(llama-2-7b-chat.Q4_K_M.gguf-127.0.0.1:43735): stderr llm_load_tensors: offloading v cache to GPU
5:32AM DBG GRPC(llama-2-7b-chat.Q4_K_M.gguf-127.0.0.1:43735): stderr llm_load_tensors: offloading k cache to GPU
5:32AM DBG GRPC(llama-2-7b-chat.Q4_K_M.gguf-127.0.0.1:43735): stderr llm_load_tensors: offloaded 35/35 layers to GPU
5:32AM DBG GRPC(llama-2-7b-chat.Q4_K_M.gguf-127.0.0.1:43735): stderr llm_load_tensors: VRAM used: 5869 MB
```

### Demo

https://www.youtube.com/watch?v=yFh_Zfk34PE

### vLLM Backend

AIKit supports the [vLLM](https://docs.vllm.ai/) backend for high-throughput GPU inference with HuggingFace safetensors models. vLLM requires NVIDIA CUDA runtime and only supports amd64 architecture.

Example aikitfile:

```yaml
#syntax=ghcr.io/kaito-project/aikit/aikit:latest
apiVersion: v1alpha1
debug: true
runtime: cuda
backends:
  - vllm
config: |
  - name: Qwen2.5-0.5B-Instruct
    backend: vllm
    parameters:
      model: Qwen/Qwen2.5-0.5B-Instruct
    use_tokenizer_template: true
```

vLLM will download the model from HuggingFace at container startup. You can also embed models at build time using the `models` section with a `huggingface://` source.

After building, run with GPU support:

```bash
docker run --rm --gpus all -p 8080:8080 my-model
```

Test with:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen2.5-0.5B-Instruct","messages":[{"role":"user","content":"Hello"}]}'
```

## AMD GPU (ROCm - Experimental)

AIKit supports AMD GPU acceleration using [ROCm 7.2](https://rocm.docs.amd.com/en/latest/) with primary focus on **Strix Halo APUs (gfx1151)** and broader support for RDNA3, RDNA2, and RDNA1 architectures. This implementation leverages LocalAI's `hipblas` backend for ROCm acceleration with automatic architecture detection and build optimization.

**Supported AMD GPUs:**
- **AMD Ryzen AI Max+ (Strix Halo)** - gfx1151 (primary target)
- **RDNA3/RDNA2/RDNA1 GPUs** - gfx900, gfx906, gfx908, gfx940, gfx941, gfx942, gfx90a, gfx1030, gfx1031, gfx1100, gfx1101

Currently, only the `llama-cpp` backend supports ROCm acceleration.

### Prerequisites

1. Install ROCm 7.2+ on your host system following the [official ROCm installation guide](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html)
2. Ensure your GPU target architecture is supported. AIKit specifically targets **gfx1151** for Strix Halo APUs, with fallback support for other architectures (gfx900,gfx906,gfx908,gfx940,gfx941,gfx942,gfx90a,gfx1030,gfx1031,gfx1100,gfx1101)
3. Ensure your user is in the `render` and `video` groups:
   ```bash
   sudo usermod -a -G render,video $USER
   ```
4. Verify ROCm installation:
   ```bash
   rocm-smi
   ```

### Configuration

To enable ROCm GPU acceleration, set the following in your `aikitfile`:

```yaml
runtime: rocm         # use AMD ROCm runtime
backends:
  - llama-cpp        # only llama-cpp backend supports ROCm
```

For the `llama-cpp` backend, configure GPU acceleration in your `config`:

```yaml
f16: true             # use float16 precision
mmap: true            # memory-mapped I/O for efficient model loading
gpu_layers: 99        # offload all layers to GPU (use lower values for large models)
context_size: 4096    # adjust based on model and available memory
threads: 1            # single thread is sufficient for GPU-accelerated inference
```

:::tip Strix Halo Optimization
For Strix Halo APUs with sufficient BIOS VRAM (8 GB+), `gpu_layers: 99` with `mmap: true` works well for small models (1-3B). For larger models, reduce `gpu_layers` and `context_size` based on available memory.
:::

:::note
AIKit automatically configures Strix Halo-specific runtime optimizations including `HSA_OVERRIDE_GFX_VERSION=11.5.1` during the image build. The ROCm backend is pre-compiled with gfx1151 support, so no runtime recompilation is needed.
:::

### Building and Testing ROCm Models

Build your ROCm-accelerated model:

```bash
# Build with ROCm runtime
make build-test-model TEST_FILE=test/aikitfile-llama-rocm.yaml PLATFORMS=linux/amd64

# Or build a custom model
docker buildx build -f my-aikitfile-rocm.yaml -t my-rocm-model --build-arg runtime=rocm .
```

### Running ROCm-accelerated Models

After building your model with `runtime: rocm`, run it with the ROCm device access flags:

```bash
# for pre-made models, replace "my-model" with the image name
docker run --rm --device /dev/kfd --device /dev/dri \
  --group-add video --group-add $(stat -c '%g' /dev/dri/renderD128) \
  -p 8080:8080 my-model

# or use the make target for testing
make run-test-model-rocm
```

:::note
`--group-add` must use the numeric GID of the render device (not the name `render`) because the group name may not exist inside the container. `stat -c '%g' /dev/dri/renderD128` reads the GID from the host device node.
:::

### Example aikitfile for ROCm

https://github.com/kaito-project/aikit/blob/main/test/aikitfile-llama-rocm.yaml

If ROCm acceleration is working correctly, you'll see output similar to:

**For Strix Halo APUs (gfx1151):**
```bash
ggml_cuda_init: found 1 ROCm devices (Total VRAM: 131072 MiB):
  Device 0: AMD Radeon Graphics, gfx1151 (0x1151), VMM: no, Wave Size: 32, VRAM: 131072 MiB
llama_model_load_from_file_impl: using device ROCm0 (AMD Radeon Graphics) - 91589 MiB free
load_tensors: offloading output layer to GPU
load_tensors: offloading 25 repeating layers to GPU
load_tensors: offloaded 27/27 layers to GPU
prompt eval time =      64.75 ms /    21 tokens (    3.08 ms per token,   324.31 tokens per second)
       eval time =      13.15 ms /     3 tokens (    4.38 ms per token,   228.12 tokens per second)
```

### Minimum VRAM Requirements for APUs

AMD APUs (like Strix Halo) share system RAM with the GPU but expose a small portion as "dedicated VRAM" controlled by a BIOS setting (typically called "UMA Frame Buffer Size"). The ROCm HIP runtime allocates pinned host buffers from this dedicated VRAM pool, so it **must** be large enough to hold the model's `ROCm_Host` buffer plus display overhead.

| Model Size | Example | ROCm_Host Buffer | Display Overhead | **Minimum VRAM (BIOS)** |
|---|---|---|---|---|
| Tiny (≤1B, Q2) | GLM-4.7-distill-1b Q2_K | ~560 MiB | ~160 MiB | **1 GB** |
| Small (≤1B, Q4) | GLM-4.7-distill-1b Q4_K_M | ~900 MiB | ~160 MiB | **2 GB** |
| Medium (7B, Q4) | Llama-3.2-7B Q4_K_M | ~4 GiB | ~160 MiB | **6 GB** |
| Large (20B, Q4) | GPT-OSS-20B mxfp4 | ~12 GiB | ~160 MiB | **16 GB** |

:::warning
The default BIOS setting on most APU systems is **512 MiB**, which is too small for any model. If you see `ROCm error: out of memory` at `hipStreamCreateWithFlags`, increase VRAM in BIOS.
:::

#### How to increase dedicated VRAM on AMD APUs

1. Reboot and enter BIOS/UEFI setup (typically **F2** or **Del** during POST)
2. Navigate to **Advanced** → **AMD CBS** → **NBIO Common Options** → **GFX Configuration**
   (path varies by vendor — look for "UMA Frame Buffer Size", "VRAM Size", or "iGPU Memory")
3. Change the value from **512M** to at least **8G** (or **16G** for larger models)
4. Save and exit (F10)
5. After reboot, verify:
   ```bash
   cat /sys/class/drm/card*/device/mem_info_vram_total
   # 8589934592 = 8 GiB, 17179869184 = 16 GiB
   ```

:::tip
AMD recommends setting VRAM to the minimum needed and using the `amd-ttm` tool to configure shared memory (GTT) for the remaining system RAM. See the [ROCm on Ryzen install guide](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/native_linux/install-ryzen.html) for details.
:::

### Troubleshooting

#### Common Issues and Solutions:

- **Permission denied errors**: Ensure user is in `render` and `video` groups and reboot after adding
- **Device not found**: Verify ROCm installation with `rocm-smi` and check `/dev/kfd` and `/dev/dri` exist
- **Strix Halo APU not detected**:
  - Verify you're running ROCm 7.2+ which supports gfx1151
  - AIKit automatically sets `HSA_OVERRIDE_GFX_VERSION=11.5.1` and `GPU_TARGETS=gfx1151`
  - Check `dmesg | grep amdgpu` for hardware detection issues
- **Performance issues**:
  - For Strix Halo APUs: Use `gpu_layers: 99` with `mmap: true` for small models (1-3B); reduce for larger models
  - For discrete GPUs: Gradually increase `gpu_layers` based on VRAM capacity
  - Monitor memory usage with `rocm-smi`
- **Memory errors**:
  - Reduce `gpu_layers` and `context_size` if you see out-of-memory errors
  - For Strix Halo: Ensure adequate BIOS VRAM setting (8 GB+) and system RAM
  - Enable `low_vram: true` if running close to memory limits
- **Build issues**:
  - Backends are pre-compiled with gfx1151 support — no runtime recompilation is needed
  - If you see `hip_fatbin.cpp: No compatible code objects found`, the backend was not compiled for your GPU architecture
  - Ensure Docker has sufficient disk space for the ROCm image layers (~15 GB)

#### Strix Halo Specific Notes:

- **Memory Architecture**: Strix Halo uses unified memory architecture (UMA) - models consume system RAM, not dedicated VRAM
- **Expected Performance**: ~320 tok/s prompt eval, ~230 tok/s generation for 1B Q4 models; expect 15-30 tok/s generation for 7B models depending on RAM speed and GPU clocks
- **Startup**: Backends are pre-compiled — container starts inference-ready in seconds, no rebuild needed

## Apple Silicon (experimental)

:::note
Apple Silicon is an experimental runtime and it may change in the future. This runtime is specific to Apple Silicon only, and it will not work as expected on other architectures, including Intel Macs.
:::

AIKit supports Apple Silicon GPU acceleration with Podman Desktop for Mac with [`libkrun`](https://github.com/containers/libkrun). Please see [Podman Desktop documentation](https://podman-desktop.io/docs/podman/gpu) on how to enable GPU support.

To get started with Apple Silicon GPU-accelerated inferencing, make sure to set the following in your `aikitfile` and build your model.

```yaml
runtime: applesilicon         # use Apple Silicon runtime
```

Please note that only the default `llama.cpp` backend with `gguf` models are supported for Apple Silicon.

After building the model, you can run it with:

```bash
# for pre-made models, replace "my-model" with the image name
podman run --rm --device /dev/dri -p 8080:8080 my-model
```

If GPU acceleration is working, you'll see output that is similar to following in the debug logs:

```bash
6:16AM DBG GRPC(phi-3.5-3.8b-instruct-127.0.0.1:39883): stderr ggml_vulkan: Found 1 Vulkan devices:
6:16AM DBG GRPC(phi-3.5-3.8b-instruct-127.0.0.1:39883): stderr Vulkan0: Virtio-GPU Venus (Apple M1 Max) (venus) | uma: 1 | fp16: 1 | warp size: 32
6:16AM DBG GRPC(phi-3.5-3.8b-instruct-127.0.0.1:39883): stderr llama_load_model_from_file: using device Vulkan0 (Virtio-GPU Venus (Apple M1 Max)) - 65536 MiB free
...
6:16AM DBG GRPC(phi-3.5-3.8b-instruct-127.0.0.1:39883): stderr llm_load_tensors: offloading 32 repeating layers to GPU
6:16AM DBG GRPC(phi-3.5-3.8b-instruct-127.0.0.1:39883): stderr llm_load_tensors: offloading output layer to GPU
6:16AM DBG GRPC(phi-3.5-3.8b-instruct-127.0.0.1:39883): stderr llm_load_tensors: offloaded 33/33 layers to GPU
6:16AM DBG GRPC(phi-3.5-3.8b-instruct-127.0.0.1:39883): stderr llm_load_tensors:   CPU_Mapped model buffer size =    52.84 MiB
6:16AM DBG GRPC(phi-3.5-3.8b-instruct-127.0.0.1:39883): stderr llm_load_tensors:      Vulkan0 model buffer size =  2228.82 MiB
```
