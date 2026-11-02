# SenseNova-Loop

Text-only support for [SenseNova-Loop-9B-Preview](https://huggingface.co/sensenova/SenseNova-Loop-9B-Preview): a Qwen3.5 hybrid trunk whose layers 4-27 run twice (56 layer visits from 32 stored layers), with two mHC residual streams and a router on each sublayer of the looped layers. Every visit has its own KV cache and recurrent state. The vision tower and the MTP head are not supported.

The base implementation comes from [GuideboardLabs/llama.cpp-sensenova-loop](https://github.com/GuideboardLabs/llama.cpp-sensenova-loop). This branch adds CUDA kernels for the router ops and loads every GGUF layout that is published.

## Supported GGUF files

| Repository | Layout |
|---|---|
| [AtomicChat/SenseNova-Loop-9B-Preview-GGUF](https://huggingface.co/AtomicChat/SenseNova-Loop-9B-Preview-GGUF) | arch `qwen35` with loop keys, one block per layer |
| [guideboardlabs/SenseNova-Loop-9B-Preview-GGUF](https://huggingface.co/guideboardlabs/SenseNova-Loop-9B-Preview-GGUF), [webmp3/Sakura-SenseNova-Loop-9B-Preview-GGUF](https://huggingface.co/webmp3/Sakura-SenseNova-Loop-9B-Preview-GGUF) | arch `sensenova_loop`, one block per visit, looped layers stored once |
| [notfakerv/SenseNova-Loop-9B-Preview-GGUF](https://huggingface.co/notfakerv/SenseNova-Loop-9B-Preview-GGUF), [annelo/SenseNova-Loop-9B-Preview-Q4_K_M-Vision-GGUF](https://huggingface.co/annelo/SenseNova-Loop-9B-Preview-Q4_K_M-Vision-GGUF) | arch `sensenova_loop`, one block per visit, looped layers stored twice (more memory) |

To convert the original checkpoint yourself:

```bash
python convert_hf_to_gguf.py /path/to/SenseNova-Loop-9B-Preview --outtype bf16 --outfile SenseNova-Loop-9B-Preview-BF16.gguf
./build/bin/llama-quantize SenseNova-Loop-9B-Preview-BF16.gguf SenseNova-Loop-9B-Preview-Q4_K_M.gguf Q4_K_M
```

## Windows CUDA 12 build

The workflow [build-windows-cuda12.yml](../.github/workflows/build-windows-cuda12.yml) runs on every push to this branch and can be started by hand from the Actions tab. Each run uploads one artifact per CUDA version, with `llama-server.exe`, the other tools, all the DLLs they load (`llama.dll`, `ggml*.dll`, `ggml-cuda.dll`, ...) and the CUDA runtime (`cudart64_12.dll`, `cublas64_12.dll`, `cublasLt64_12.dll`):

- `llama-<commit>-bin-win-cuda-12.4-x64`: same CUDA version as the upstream release, needs driver R550 or newer. An RTX 50 series GPU runs it through PTX JIT, which makes the first start slow.
- `llama-<commit>-bin-win-cuda-12.8-x64`: adds native code for the RTX 50 series, needs driver R570 or newer.

Unzip the artifact into one folder and run:

```bash
llama-server.exe -m SenseNova-Loop-9B-Preview-Q4_K_M.gguf -ngl 99 -c 16384 --port 8080
```

The model thinks before it answers by default. Add `--reasoning off` to the server to turn that off, or send `"chat_template_kwargs": {"enable_thinking": false}` with a request.

To check the CUDA kernels of the router ops on your GPU:

```bash
test-backend-ops.exe -o MHC_ROUTE
test-backend-ops.exe -o MHC_MERGE
```

To build on your own Windows machine instead, install Visual Studio 2022 with the C++ workload and the CUDA 12 toolkit, then in a Developer PowerShell:

```bash
cmake -S . -B build -DGGML_CUDA=ON
cmake --build build --config Release -j
```

The executables are in `build\bin\Release`. Copy `cudart64_12.dll`, `cublas64_12.dll` and `cublasLt64_12.dll` from `%CUDA_PATH%\bin` next to them if the CUDA `bin` folder is not on your `PATH`.
