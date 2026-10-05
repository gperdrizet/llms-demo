# Quickstart

## 1. Fork and clone

1. Click **Fork** in the top-right corner of this repo on GitHub to create your own copy.
2. Clone your fork:

   ```bash
   git clone https://github.com/<your-username>/llms-demo.git
   ```

## 2. Open in a dev container

1. Open the cloned folder in VS Code.
2. When prompted **"Reopen in Container"**, click it (or run the command **Dev Containers: Reopen in Container** from the Command Palette `Ctrl+Shift+P`).
3. VS Code will build and start the container. This takes a few minutes the first time.

## 3. Configure your environment

The configuration template is [`.env.example`](https://github.com/gperdrizet/llms-demo/blob/main/.env.example).
From the repository root, copy it once:

```bash
cp .env.example .env
```

Do not overwrite an existing `.env`. Open it in VS Code and replace the
placeholders for the services your demos use:

| Variables | What to configure |
|-----------|-------------------|
| `OPENAI_API_URL`, `OPENAI_API_KEY`, `OPENAI_API_MODEL` | Full OpenAI-compatible API base URL (including `http://` or `https://` and `/v1` if required), API key, and exact model name. These apply to local llama.cpp and remote servers alike. |
| `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT`, `DB_NAME` | PostgreSQL connection details for the RAG demo; the database must support pgvector. `DB_PORT` defaults to `5432`. |
| `HF_TOKEN` | Hugging Face access token for authenticated or gated model downloads. |
| `ANTHROPIC_API_KEY` | Anthropic API key if using Claude. |
| `HF_HOME`, `HF_HUB_CACHE`, `TRANSFORMERS_CACHE`, `OLLAMA_MODELS` | Model-storage paths; keep the template values for the standard workspace or adjust them for your location. |

Comment out unused placeholder credentials, save the file, and restart any
running demos after changing it. Never commit or share `.env`; it is excluded
by `.gitignore`.

The Python demos load `.env` themselves. Standalone server commands do not:
export settings such as `OLLAMA_MODELS` in the server's shell if needed.

## 4. What happens during container startup

The dev container uses the CPU-only `gperdrizet/llms-cpu` image by default. On first creation, the `postCreateCommand` runs automatically and does the following:

| Step | What it does |
|------|-------------|
| `mkdir -p models/hugging_face && mkdir -p models/ollama` | Creates local directories for model storage |
| `pip install "numpy<2.0" "contourpy<1.3" --force-reinstall` | Installs compatible versions of NumPy and contourpy |
| `pip install -r requirements.txt` | Installs additional Python dependencies; the base image supplies the demo runtime |

The container also pre-configures the following:

| Setting | Detail |
|---------|--------|
| **GPU access** | Disabled by default. GPU-dependent demos require a local NVIDIA GPU, NVIDIA Container Toolkit, the `gperdrizet/llms-gpu` image, and enabled `--gpus all` arguments. |
| **Python interpreter** | `/usr/bin/python` is set as the default |
| **Model-storage settings** | Supplied in `.env.example`; configure them as described above |
| **Port 7860** | Forwarded automatically for Gradio web UIs |
| **VS Code extensions** | Python, Jupyter, Code Spell Checker, and Marp (slide viewer) are installed |

Once the container is ready and your backend and `.env` settings are configured,
you can run the supported demos. For the browser-based, CPU-only GitHub Codespaces
option, see the [README quickstart](https://github.com/gperdrizet/llms-demo#quickstart).
