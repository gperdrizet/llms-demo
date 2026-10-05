# LLM chatbots demo

[![Build and Deploy Documentation](https://github.com/gperdrizet/llms-demo/actions/workflows/docs.yml/badge.svg)](https://github.com/gperdrizet/llms-demo/actions/workflows/docs.yml)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1.2-1C3C3C?logo=langchain&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FF6B00?logo=huggingface&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-UI-FF7C00?logo=gradio&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-black?logo=ollama&logoColor=white)
![llama.cpp](https://img.shields.io/badge/llama.cpp-GGUF-green)
![PostgreSQL](https://img.shields.io/badge/pgvector-PostgreSQL-4169E1?logo=postgresql&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-vector%20store-E8572A)

## Introduction

This repository provides hands-on examples and learning resources for working with large language models (LLMs) in local development environments.

### Topics covered

- Local inference with Ollama and llama.cpp
- Direct model loading with HuggingFace Transformers
- LangChain: prompt templates, output parsers, chains, and agents
- RAG (Retrieval-Augmented Generation) with pgvector
- Gradio web interfaces
- Prompting techniques: zero-shot, few-shot, chain-of-thought, ReAct

### Resources included

- **9 demos**: chatbots, LangChain patterns, agents, RAG knowledge systems, fine-tuning & evaluation
- **8 slide decks**: covering deployment, prompting, LangChain, fine-tuning, and evaluation
- **7 activities**: hands-on exercises building on each demo

---

## Documentation

**Complete documentation:** [https://gperdrizet.github.io/llms-demo](https://gperdrizet.github.io/llms-demo/)

The documentation covers:
- Setup and installation
- Demo usage and concepts
- Inference server configuration
- Library reference with code examples
- Model specifications and serving commands
- Systemd deployment for production use
- Slide and activity guides

---

## Quickstart

### Option 1: Local VS Code dev container

#### 1. Fork and clone

1. Click **Fork** in the top-right corner of this repo on GitHub to create your own copy.
2. Clone your fork:

   ```bash
   git clone https://github.com/<your-username>/llms-demo.git
   ```

#### 2. Open in a dev container

1. Open the cloned folder in VS Code.
2. When prompted **"Reopen in Container"**, click it - or run the command **Dev Containers: Reopen in Container** from the Command Palette (`Ctrl+Shift+P`).
3. VS Code will build and start the container. This takes a few minutes the first time.

The default container is CPU-only. To run GPU-dependent demos locally, you need
an NVIDIA GPU and NVIDIA Container Toolkit. Before opening the container, change
the image in `.devcontainer/devcontainer.json` to `gperdrizet/llms-gpu` and
uncomment the `"--gpus", "all"` arguments.

Before running a demo, complete [Configure your environment](#configure-your-environment-both-options) below.

### Option 2: GitHub Codespaces (no local setup)

1. Click **Fork** in the top-right corner of this repo on GitHub to create your own copy.
2. In your fork, click **Code** → **Codespaces** → **Create codespace on main**.
3. Wait for the dev container to finish setting up. VS Code will open in your browser; no local Docker or Python installation is needed.
4. Follow [Configure your environment](#configure-your-environment-both-options) below, using the remote OpenAI-compatible endpoint provided by your instructor.
5. Run a demo that uses the remote endpoint, for example:

   ```bash
   python demos/chatbots/gradio_chatbot.py
   ```

   Select **OpenAI compatible** as the model backend. In the **Ports** panel,
   open the forwarded URL for port **7860** to access the Gradio interface.
   Keep the port private.

> **GPU limitation:** GitHub Codespaces does not provide an NVIDIA GPU.
> The GPU-dependent demos will **not run** there: running a downloaded Hugging
> Face language model locally, hosting a local inference server (Ollama or
> llama.cpp), and fine-tuning. Use a remote OpenAI-compatible server for the
> instead.

Never commit `.env` or API keys. Codespaces usage is subject to your GitHub
account's quota; stop your Codespace when you are finished.

### Configure your environment (both options)

1. In the VS Code terminal, from the repository root, copy the example configuration:

   ```bash
   cp .env.example .env
   ```

   Do this only when first setting up the repository; do not overwrite an existing
   `.env` containing your configuration.
2. Open `.env` in VS Code and replace the placeholder values for the demos you plan to run:

   | Settings | What to enter |
   |----------|---------------|
   | `OPENAI_API_URL`, `OPENAI_API_KEY`, `OPENAI_API_MODEL` | Your OpenAI-compatible server's full API base URL (including `https://` and `/v1` if required), API key, and exact model name. In Codespaces, use the remote endpoint supplied by your instructor. |
   | `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT`, `DB_NAME` | Connection details for a PostgreSQL database with pgvector, required for the RAG demo. |
   | `HF_TOKEN` | Your Hugging Face access token if your demo needs authenticated model downloads. |
   | `ANTHROPIC_API_KEY` | Your Anthropic API key if you use a Claude-based demo. |
   | `HF_HOME`, `HF_HUB_CACHE`, `TRANSFORMERS_CACHE`, `OLLAMA_MODELS` | Keep the supplied model-storage paths for the standard dev container, or adjust them if your workspace is in a different location. |

3. Save `.env` before starting the demo. Restart any already-running demo after changing these settings.

Configure only the services you need; comment out unused placeholder credentials.
Never commit `.env` or share its contents. The repository's `.gitignore` already
excludes it.

### What happens during container startup

The dev container uses the `gperdrizet/llms-cpu` image by default. On first creation, the `postCreateCommand` runs automatically and does the following:

| Step | What it does |
|------|-------------|
| `mkdir -p models/hugging_face && mkdir -p models/ollama` | Creates local directories for model storage |
| `pip install "numpy<2.0" "contourpy<1.3" --force-reinstall` | Installs compatible versions of NumPy and contourpy |
| `pip install -r requirements.txt` | Installs additional Python dependencies; the base image supplies the demo runtime |

The container also pre-configures the following:

| Setting | Detail |
|---------|--------|
| **GPU access** | Disabled by default; available only on a configured local GPU host, not in Codespaces |
| **Python interpreter** | `/usr/bin/python` is set as the default |
| **Model-storage settings** | `.env.example` supplies `HF_HOME`, `HF_HUB_CACHE`, `TRANSFORMERS_CACHE`, and `OLLAMA_MODELS`; copy them into `.env`. For a standalone `ollama serve` process, export `OLLAMA_MODELS` in its shell as well. |
| **Port 7860** | Forwarded automatically for Gradio web UIs |
| **VS Code extensions** | Python, Jupyter, Code Spell Checker, and Marp (slide viewer) are installed |

Once the container is ready and the required backend and `.env` settings are
configured, you can start running the supported demos.

---

## Running the demos

See the [Demos documentation](https://gperdrizet.github.io/llms-demo/demos.html) for detailed instructions on running each chatbot, including:
- Concepts covered in each demo
- Tools and libraries used
- Step-by-step setup and execution

**Quick example** - Ollama chatbot:
```bash
# 1. Start the Ollama server
ollama serve

# 2. Pull a model (in another terminal)
ollama pull qwen2.5:3b

# 3. Run the chatbot
python demos/chatbots/ollama_chatbot.py
```

For complete instructions on all four demos, visit the [documentation](https://gperdrizet.github.io/llms-demo/).
