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
gpu_layers: 35        # number of layers to offload to GPU
low_vram: true        # recommended for APUs with shared memory
```

:::tip Strix Halo Optimization
For Strix Halo APUs, start with `gpu_layers: 20-30` and `low_vram: true` since these APUs use shared system memory rather than dedicated VRAM.
:::

:::note
AIKit automatically configures Strix Halo-specific optimizations including `HSA_OVERRIDE_GFX_VERSION=11.5.1` and `GPU_TARGETS=gfx1151` during build.
:::

### Building and Testing ROCm Models

Build your ROCm-accelerated model:

```bash
# Build with ROCm runtime
make build-test-model TEST_FILE=test/aikitfile-llama-rocm.yaml RUNTIME=rocm

# Or build a custom model
docker buildx build -f my-aikitfile-rocm.yaml -t my-rocm-model --build-arg runtime=rocm .
```

### Running ROCm-accelerated Models

After building your model with `runtime: rocm`, run it with the ROCm device access flags:

```bash
# for pre-made models, replace "my-model" with the image name
docker run --rm --device /dev/kfd --device /dev/dri -p 8080:8080 my-model

# or use the make target for testing
make run-test-model-rocm
```

### Example aikitfile for ROCm

```yaml
apiVersion: v1alpha1
runtime: rocm
backends:
  - llama-cpp
models:
  - name: llama-3.2-1b-instruct
    source: https://huggingface.co/MaziyarPanahi/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct.Q4_K_M.gguf
    sha256: "e4650dd6b45ef456066b11e4927f775eef4dd1e0e8473c3c0f27dd19ee13cc4e"
config: |
  backend: llama
  parameters:
    model: llama-3.2-1b-instruct
  name: llama-3.2-1b-instruct
  f16: true
  context_size: 8192
  gpu_layers: 35
```

If ROCm acceleration is working correctly, you'll see output similar to:

**For Strix Halo APUs:**
```bash
5:32AM DBG GRPC(llama-3.2-1b-instruct): stderr ggml_init_rocm: found 1 ROCm devices:
5:32AM DBG GRPC(llama-3.2-1b-instruct): stderr   Device 0: AMD Ryzen AI Max+ Graphics, compute capability gfx1151
5:32AM DBG GRPC(llama-3.2-1b-instruct): stderr llm_load_tensors: using ROCm for GPU acceleration
5:32AM DBG GRPC(llama-3.2-1b-instruct): stderr llm_load_tensors: offloading 25 repeating layers to GPU
5:32AM DBG GRPC(llama-3.2-1b-instruct): stderr llm_load_tensors: offloaded 25/33 layers to GPU
5:32AM DBG GRPC(llama-3.2-1b-instruct): stderr llm_load_tensors: VRAM used: 1536 MB (shared system memory)
```

**For discrete AMD GPUs:**
```bash
5:32AM DBG GRPC(llama-3.2-1b-instruct): stderr ggml_init_rocm: found 1 ROCm devices:
5:32AM DBG GRPC(llama-3.2-1b-instruct): stderr   Device 0: AMD Radeon RX 7900 XT, compute capability gfx1100
5:32AM DBG GRPC(llama-3.2-1b-instruct): stderr llm_load_tensors: using ROCm for GPU acceleration
5:32AM DBG GRPC(llama-3.2-1b-instruct): stderr llm_load_tensors: offloading 35 repeating layers to GPU
5:32AM DBG GRPC(llama-3.2-1b-instruct): stderr llm_load_tensors: offloaded 35/35 layers to GPU
5:32AM DBG GRPC(llama-3.2-1b-instruct): stderr llm_load_tensors: VRAM used: 4096 MB
```

### Troubleshooting

#### Common Issues and Solutions:

- **Permission denied errors**: Ensure user is in `render` and `video` groups and reboot after adding
- **Device not found**: Verify ROCm installation with `rocm-smi` and check `/dev/kfd` and `/dev/dri` exist
- **Strix Halo APU not detected**:
  - Verify you're running ROCm 7.2+ which supports gfx1151
  - AIKit automatically sets `HSA_OVERRIDE_GFX_VERSION=11.5.1` and `GPU_TARGETS=gfx1151`
  - Check `dmesg | grep amdgpu` for hardware detection issues
- **Performance issues**:
  - For Strix Halo APUs: Start with `gpu_layers: 20-30` and enable `low_vram: true`
  - For discrete GPUs: Gradually increase `gpu_layers` based on VRAM capacity
  - Monitor memory usage with `rocm-smi`
- **Memory errors**:
  - Enable `low_vram: true` especially for APUs
  - Reduce `gpu_layers` or `context_size`
  - For Strix Halo: Ensure adequate system RAM (models use shared memory)
- **Build issues**:
  - LocalAI rebuilds backends for gfx1151 during first container startup (may take 10-15 minutes)
  - Check logs for compilation errors related to ROCm/HIP
  - Ensure Docker has sufficient disk space for backend compilation

#### Strix Halo Specific Notes:

- **Memory Architecture**: Strix Halo uses unified memory architecture (UMA) - models consume system RAM, not dedicated VRAM
- **Expected Performance**: Expect 15-30 tokens/sec for 7B models depending on RAM speed and GPU clocks
- **First Run**: Initial container startup will rebuild llama.cpp backend for gfx1151 (one-time ~10-15 min process)

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
