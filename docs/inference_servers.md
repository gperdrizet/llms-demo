# Inference servers

## Ollama

Local inference server - runs models on your machine behind a REST API.

- [Model library](https://ollama.com/library)
- [Documentation](https://github.com/ollama/ollama/blob/main/docs/README.md)
- [API reference](https://github.com/ollama/ollama/blob/main/docs/api.md)

```bash
# Install
curl -fsSL https://ollama.com/install.sh | sh

# Start the server (runs on localhost:11434)
ollama serve

# Pull a model
ollama pull qwen2.5:3b

# List downloaded models
ollama list

# Run a model interactively
ollama run qwen2.5:3b

# Remove a model
ollama rm qwen2.5:3b
```

>**Note**: if you are running the demos in this repo via a devcontainer as intended, you do not need to install Ollama. The container environment includes it.

### Environment variables

| Variable | Purpose |
|----------|---------|
| `OLLAMA_MODELS` | Directory where models are stored |
| `OLLAMA_HOST` | Server address (default `127.0.0.1:11434`) |

`OLLAMA_MODELS` is included in `.env.example`. `OLLAMA_HOST` is an optional
Ollama server setting, not a template entry. Export these in the shell that
starts `ollama serve`; the server does not automatically read the demos' `.env`.

---

## llama.cpp

High-performance C/C++ inference engine. Runs GGUF-quantized models and can split MoE layers across CPU and GPU, making it possible to serve large models (100B+) on consumer hardware.

- [GitHub](https://github.com/ggml-org/llama.cpp)
- [GGUF model format](https://huggingface.co/docs/hub/gguf)
- [Server documentation](https://github.com/ggml-org/llama.cpp/blob/master/examples/server/README.md)

```bash
# Build from source with CUDA support (compiles for your GPU automatically)
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j$(nproc)
```

> **Note**: The build compiles CUDA kernels for the GPU(s) detected on your machine.
> This takes several minutes but only needs to be done once. If you change GPUs, rebuild.

```bash
# Start the server with CPU/GPU MoE split
# Replace <model.gguf> with the path to your GGUF file
llama.cpp/build/bin/llama-server \
    -m <model.gguf> \
    --n-gpu-layers 999 \
    --n-cpu-moe <N> \
    -c 0 --flash-attn on \
    --jinja \
    --host 0.0.0.0 --port 8502 --api-key "dummy"
```

See the [Models](models.md) section for complete, copy-paste run commands.

The server exposes an **OpenAI-compatible API**, so any OpenAI client library can connect to it.

```python
from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:8502/v1',
    api_key='your-api-key',
)

response = client.chat.completions.create(
    model='model-name',
    messages=[
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'Hello!'},
    ],
    temperature=0.7,
)

print(response.choices[0].message.content)
```

### Key server flags

| Flag | Purpose |
|------|----------|
| `-m` | Path to the GGUF model file |
| `--n-gpu-layers` | Number of layers to offload to GPU (`999` = all non-MoE layers) |
| `--n-cpu-moe` | Number of MoE blocks to keep on CPU (e.g. `36` = all MoE on CPU) |
| `-c` | Context length (`0` = model maximum) |
| `--flash-attn` | Enable flash attention |
| `--host` / `--port` | Server bind address and port |
| `--jinja` | Enable Jinja chat templates (required for harmony and similar formats) |
| `--api-key` | API key for authenticating requests |

### CPU/GPU MoE split explained

Mixture-of-Experts models have two types of layers: **attention layers** (small, benefit from GPU) and **MoE/expert layers** (large, run well on CPU). The `--n-cpu-moe` flag controls how many MoE blocks stay on CPU:

| Config | VRAM usage | Generation speed |
|--------|-----------|------------------|
| `--n-cpu-moe 36` (all MoE on CPU) | ~5-8 GB | ~18-25 tok/s |
| `--n-cpu-moe 28` (8 MoE on GPU) | ~22 GB | ~25-31 tok/s |

This makes it possible to run a 120B parameter model with as little as 8 GB of VRAM and 64 GB of system RAM.
