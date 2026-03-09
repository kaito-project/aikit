---
title: GPU Acceleration
---

:::note
At this time, only NVIDIA GPU acceleration is supported, with experimental support for Apple Silicon. Please open an issue if you'd like to see support for other GPU vendors.
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
