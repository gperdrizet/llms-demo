---
marp: true
theme: default
paginate: true
style: |
  section {
    background-color: #1a1a2e;
    color: #e0e0e0;
  }
  section h1, section h2, section h3 {
    color: #e2b55a;
  }
  section a {
    color: #93c5fd;
  }
  section strong {
    color: #f0f0f0;
  }
  section table th {
    background-color: #2a2a4a;
    color: #e2b55a;
  }
  section table td {
    background-color: #1e1e36;
    color: #e0e0e0;
  }
  section code {
    background-color: #2a2a4a;
    color: #e0e0e0;
  }
  section pre {
    background-color: #12122a;
  }
  section::after {
    color: #888;
  }
---

# Lesson 45: Practical LLM deployment

**Working with large language models**

---

## Recap: what we learned

Last lesson covered the 2026 state of the art:
- The generative-AI landscape: image, video, audio, and LLMs
- Frontier models as tiered product families (flagship / near-flagship / cost-and-volume)
- Open-weight frontier and architecture themes (MoE, reasoning tiers, ~1M context)
- Per-task economics and the shift to long-horizon agentic benchmarks

**Today:** How do we actually **deploy and use** these models in practice?

---

## Anatomy of an LLM application

A minimal LLM application has layers:

```
┌─────────────────────────────────┐
│  Your application (Python, JS)  │  <- Your code
├─────────────────────────────────┤
│  Framework (LangChain, etc.)    │  <- Orchestration, prompts, chains
├─────────────────────────────────┤
│  Inference server (Ollama, API) │  <- Hosts the model, handles requests
├─────────────────────────────────┤
│  Model (Llama, GPT, etc.)       │  <- The weights that generate text
└─────────────────────────────────┘
```

---

## Today's outline

These slides cover key concepts - demos and activities will apply them after.

We'll move through the stack from bottom to top:

1. **Model layer**: decoding strategies and hyperparameters
2. **Inference layer**: deployment options, quantization, cost
3. **Framework layer**: LangChain, LlamaIndex, and others
4. **Application layer**: how your code 'talks' to the model

---

# Model layer

How text generation works (configured via the inference server)

- Decoding strategies
- Key hyperparameters
- Performance optimizations

---

## Decoding strategies

After an LLM computes probabilities for the next token, a **decoding strategy** decides which token to pick:

- **Greedy**: always pick the highest-probability token. Fast but repetitive.
- **Beam search**: track top-N sequences in parallel, pick the best overall.
- **Top-k sampling**: sample randomly from the top K most likely tokens.
- **Top-p (nucleus) sampling**: sample from the smallest set of tokens whose probabilities sum to P.
- **Mirostat**: adaptively adjusts sampling to maintain a target surprise level. Rarely used in practice.

**Key takeaway:** Default is probably fine for most things

---

## Key hyperparameters

| Parameter | What it controls | Range |
|-----------|-----------------|-------|
| *temperature* | Randomness (low = focused, high = creative) | 0.0 - 1.0 |
| *max_tokens* | Maximum response length | 1 - model limit |
| *top_p* | Nucleus sampling threshold | 0.0 - 1.0 |
| *top_k* | Top tokens to sample from | 1 - 100 |
| *frequency_penalty* | Penalizes repeated tokens | 0.0 - 2.0 |

**Key takeaway:** Default is probably fine for most things

---

## Performance optimizations

Common techniques to make models faster and more efficient:

| Optimization | What it does |
|--------------|--------------|
| **Quantization** | Reduce precision (16-bit → 8-bit → 4-bit) for less memory |
| **Flash Attention** | Faster, memory-efficient attention computation |
| **KV Cache** | Store past key-value pairs to avoid recomputation |
| **Model pruning** | Remove less important weights to reduce size |

**Tools:** bitsandbytes (quantization), FlashAttention-2, vLLM (KV cache + batching)

**Key takeaway:** These optimizations happen under the hood - you benefit automatically

---

## Where we are

```
┌─────────────────────────────────┐
│  Your application (Python, JS)  │
├─────────────────────────────────┤
│  Framework (LangChain, etc.)    │
├─────────────────────────────────┤
│  Inference server (Ollama, API) │
├─────────────────────────────────┤
│  Model (Llama, GPT, etc.)       │  <- Done
└─────────────────────────────────┘
```

**Next:** Inference layer - where and how models are hosted

---

# Inference layer

Where and how models are hosted

- Deployment options (local, cloud, API)
- Quantization
- Tokens and cost

---

## Self-hosting locally

- Complete privacy and control, offline capability
- No recurring API costs,
- Open weight models only
- Requires GPU/RAM, maintenance burden

**Two approaches:**

1. **Direct loading** - Load model in your Python code (Transformers, PyTorch)
2. **Inference server** - Run a separate server process (Ollama, llama.cpp)

---

## Direct loading

Load models directly in your application code:

| Library | Strengths |
|---------|-----------|
| [HuggingFace transformers](https://huggingface.co/docs/transformers) | Largest model ecosystem, easy to use |
| [PyTorch](https://pytorch.org) | Lower-level control, custom architectures |
| [TensorFlow](https://www.tensorflow.org) | Production deployment, TensorFlow ecosystem |
| [ONNX Runtime](https://onnxruntime.ai) | Optimized inference, cross-platform |
| [vLLM](https://github.com/vllm-project/vllm) | High-performance, batching (also works as server) |

**Pros:** Full control, no separate server, direct integration
**Cons:** Model stays in memory while app runs, harder to share across processes

**Best for:** Research, experimentation, custom pipelines, batch processing

---

## Inference servers

Run a separate service that hosts the model:

| Server | Strengths |
|--------|-----------|
| [Ollama](https://ollama.com) | Easy setup, model management, REST API |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | Fast, efficient, CPU-friendly |
| [vLLM](https://github.com/vllm-project/vllm) | High-performance, batching (also works as library) |
| GPT4All, LM Studio | User-friendly desktop apps |

**Pros:** Modular, model & app independent, share model across processes
**Cons:** Additional setup, API calls (local but HTTP)

**Best for:** App deployment, streaming apps/online processing 

---

## Self-hosting in the cloud

| Type | Examples |
|------|---------|
| Fully managed | AWS SageMaker, Google Vertex AI |
| Serverless | Modal, [HF Inference](https://huggingface.co/inference-endpoints) |
| GPU rental | RunPod, Lambda Labs |
| Inference servers | vLLM, TGI |

**Pros:** Scalable hardware on-demand, good for production and team access
**Cons:** Ongoing costs, setup complexity

**Best for:** Production deployment, compute resource access

---

## API access

**Providers:** OpenAI, Anthropic, Google, Cohere
**Pros:** No infrastructure to manage, access to frontier models
**Cons:** Pay per token, rate limits, privacy considerations

**Best for:** Simplicity, you need the latest & greatest models

---

## Tokens and cost

- API providers charge **per token** (roughly 1 token = 0.75 words)
- Input and output tokens are priced separately; output costs 2-4x more

| | Input (per 1M tokens) | Output (per 1M tokens) |
|-|----------------------|------------------------|
| Mid-tier API | ~$0.50 | ~$1.50 |
| Frontier API | ~$10 | ~$30 |

**Key takeaway:** prompt length matters - shorter prompts = lower cost

---

## Where we are

```
┌─────────────────────────────────┐
│  Your application (Python, JS)  │
├─────────────────────────────────┤
│  Framework (LangChain, etc.)    │
├─────────────────────────────────┤
│  Inference server (Ollama, API) │  <- Done
├─────────────────────────────────┤
│  Model (Llama, GPT, etc.)       │  <- Done
└─────────────────────────────────┘
```

**Next:** Framework layer - tools that connect your code to models

---

# Framework layer

Tools that connect your code to models

- Why use a framework?
- LangChain, LlamaIndex, Haystack, and others

---

## Why use a framework?

You can call an API directly - so why add a framework?

- **Provider abstraction**: swap between OpenAI, Ollama, Anthropic without rewriting code
- **Prompt templating**: reusable prompts with variables instead of string concatenation
- **Chaining**: connect multiple LLM calls into a pipeline (e.g., summarize then translate)
- **Memory**: maintain conversation state across multiple turns
- **Tool use**: let the model call functions, search the web, query databases

Frameworks are optional for simple apps, but essential as complexity grows.

---

## Frameworks for LLM applications

| Framework | Focus |
|-----------|-------|
| [LangChain](https://python.langchain.com) | Unified interface, chains, agents, RAG |
| [LlamaIndex](https://www.llamaindex.ai) | Data framework for RAG and knowledge bases |
| [Haystack](https://haystack.deepset.ai) | NLP pipelines and document search |
| [smolagents](https://huggingface.co/docs/smolagents) | Lightweight agents with code-based tool calling |
| Semantic Kernel | Microsoft's orchestration framework |
| AutoGen | Multi-agent conversations and workflows |
| DSPy | Optimizing prompts programmatically |

---

## Where we are

```
┌─────────────────────────────────┐
│  Your application (Python, JS)  │
├─────────────────────────────────┤
│  Framework (LangChain, etc.)    │  <- Done
├─────────────────────────────────┤
│  Inference server (Ollama, API) │  <- Done
├─────────────────────────────────┤
│  Model (Llama, GPT, etc.)       │  <- Done
└─────────────────────────────────┘
```

**Next:** Application layer - how your code talks to the model

---

# Application layer

How your code talks to the model

- System prompts
- Message format
- Streaming vs batch responses

---

## System prompts

The **system prompt** defines your application's behavior:

```
system: "You are a customer support agent for Acme Corp.
         Only answer questions about our products.
         If unsure, say you don't know."
```

- Sets tone, role, and constraints
- Controls what the model should and should not do
- Persists across all turns in a conversation
- This is the primary way you "program" an LLM

We'll cover prompt engineering in depth in lessons 46-47.

---

## Message format

Regardless of use case, LLMs receive a list of messages with **roles**:

```
system:    "You are a helpful coding assistant."
user:      "How do I reverse a list in Python?"
assistant: "You can use list.reverse() or slicing: my_list[::-1]"
```

- **system**: sets behavior and constraints
- **user**: the input (human message, document to summarize, code to review, etc.)
- **assistant**: the model's previous responses
- For multi-turn apps, the full history is sent with every request
- For single-shot tasks (summarization, classification), just system + one user message

---

## Streaming vs batch responses

**Batch (default):**
- Wait for the entire response, then return it
- Simpler to implement
- Fine for background tasks (summarization, classification, data extraction)

**Streaming:**
- Return tokens as they are generated, one at a time
- Better for interactive applications (chatbots, code assistants)
- Slightly more complex (handle partial responses)

Choose based on your use case - not every app needs streaming.

---

## Where we are

```
┌─────────────────────────────────┐
│  Your application (Python, JS)  │  <- Done
├─────────────────────────────────┤
│  Framework (LangChain, etc.)    │  <- Done
├─────────────────────────────────┤
│  Inference server (Ollama, API) │  <- Done
├─────────────────────────────────┤
│  Model (Llama, GPT, etc.)       │  <- Done
└─────────────────────────────────┘
```

We've covered the full stack!

---

## Summary

- LLM applications have layers: **model → inference → framework → application**
- **Decoding strategies** and **hyperparameters** control how models generate text
- **Inference**: choose local, cloud, or API based on privacy, cost, and scale
- **Frameworks** like LangChain orchestrate prompts, chains, and agents
- **Chat APIs** with system/user/assistant roles are the modern standard

---

## Next lesson

**Lessons 46-47: Prompt engineering**
- Zero-shot and few-shot prompting
- Chain of thought reasoning
- Tree of thoughts
- Getting the most out of your chosen LLM

---

# Questions?

**Resources:**
- [Ollama](https://ollama.com)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [LangChain](https://python.langchain.com)
- [Hugging Face](https://huggingface.co)
