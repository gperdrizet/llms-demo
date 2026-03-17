# Demos

This repository includes five chatbot demos that demonstrate different approaches to local LLM inference. Each demo covers specific concepts and tools:

## Demo overview

### 1. Ollama chatbot (`src/ollama_chatbot.py`)

**Concepts covered:**
- Using a local inference server (Ollama)
- Structured message types (SystemMessage, HumanMessage, AIMessage)
- Conversation history management
- Terminal-based interaction

**Tools used:**
- [Ollama](inference_servers.md) - Local inference server
- [LangChain](libraries.md) - LLM application framework

### 2. llama.cpp chatbot (`src/llamacpp_chatbot.py`)

**Concepts covered:**
- Running large MoE models (120B+ parameters) on consumer hardware
- CPU/GPU memory split for expert layers
- OpenAI-compatible API usage
- Remote vs. local inference servers

**Tools used:**
- [llama.cpp](inference_servers.md) - High-performance C++ inference engine
- OpenAI Python client - Standard API interface

### 3. Gradio chatbot (`src/gradio_chatbot.py`)

**Concepts covered:**
- Web-based chat interfaces
- Multi-backend architecture (switching between Ollama/llama.cpp)
- System prompt customization
- Error handling and user feedback

**Tools used:**
- [Gradio](libraries.md) - Rapid UI prototyping
- [LangChain](libraries.md) - LLM orchestration
- [Ollama](inference_servers.md) - Default backend

### 4. HuggingFace chatbot (`src/huggingface_chatbot.py`)

**Concepts covered:**
- Direct model loading (no inference server)
- Chat templates and tokenization
- Generation parameters (temperature, max tokens)
- Decoding and response formatting

**Tools used:**
- [HuggingFace Transformers](libraries.md) - Model loading and inference
- PyTorch - Underlying tensor operations

### 5. ReAct agent chatbot (`src/react_agent_chatbot.py`)

**Concepts covered:**
- ReAct (Reasoning + Acting) agent pattern
- Multi-step reasoning with tool use
- Tool selection and execution
- Agent iteration loops and error handling

**Tools used:**
- [LangChain](libraries.md) - Agent framework and tool integration
- [Ollama](inference_servers.md) or [llama.cpp](inference_servers.md) - Backend LLM
- [Gradio](libraries.md) - Web interface with reasoning visualization

---

## Running the demos

### Ollama chatbot

```bash
# 1. Start the Ollama server in a terminal
ollama serve

# 2. Pull a model (in another terminal)
ollama pull qwen2.5:3b

# 3. Run the chatbot
python src/ollama_chatbot.py
```

### llama.cpp chatbot

You have two choices: use the hosted model at `gpt.perdrizet.org`, or run llama.cpp locally.

**Option 1: Using the remote server (recommended for quick start)**

```bash
# 1. Create a .env file with your API credentials
cp .env.example .env

# 2. Edit .env and set:
#    PERDRIZET_URL=gpt.perdrizet.org
#    PERDRIZET_API_KEY=your-api-key-here

# 3. Run the chatbot
python src/llamacpp_chatbot.py
```

**Option 2: Running llama.cpp locally**

```bash
# 1. Download a GGUF model (e.g., GPT-OSS-120B)
python utils/download_gpt_oss_120b.py

# 2. Build llama.cpp with CUDA support
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j$(nproc)
cd ..

# 3. Start the llama-server (see model-specific commands in the Models section)
llama.cpp/build/bin/llama-server -m <model.gguf> <flags...>

# 4. Run the chatbot (in another terminal)
python src/llamacpp_chatbot.py
```

> **Note**: For localhost, the defaults work automatically (localhost:8502 with "dummy" API key). For remote servers, configure `PERDRIZET_URL` and `PERDRIZET_API_KEY` in your `.env` file.

### Gradio chatbot

```bash
# 1. Start the Ollama server in a terminal
ollama serve

# 2. Pull a model (in another terminal)
ollama pull qwen2.5:3b

# 3. Run the Gradio chatbot
python src/gradio_chatbot.py

# 4. Open the URL shown in the terminal (usually http://127.0.0.1:7860)
```

### HuggingFace chatbot

```bash
# 1. Run the chatbot (downloads model on first run)
python src/huggingface_chatbot.py

# Note: This loads the model directly into memory (no inference server needed).
# First run will download approximately 6GB of model files to models/hugging_face/
```

### ReAct agent chatbot

```bash
# 1. Start the Ollama server in a terminal
ollama serve

# 2. Pull a model (in another terminal)
ollama pull qwen2.5:3b

# 3. Run the ReAct agent chatbot
python src/react_agent_chatbot.py

# 4. Open the URL shown in the terminal (usually http://127.0.0.1:7860)
```

**Try these example questions:**
- "How many days until Christmas from today?"
- "Calculate 15% tip on a $47.50 bill"
- "I was born on March 15, 1990. How old am I in days?"
- "What's 25% of 360, divided by 3?"
- "How many weeks between today and New Year's Day 2027?"

**What to observe:**
- Watch the **Reasoning Process** panel (right side) to see how the agent thinks
- Notice when it decides to use tools vs. when it can answer directly
- See the Thought → Action → Observation loop in action
- Try asking multi-step questions that require multiple tool calls

