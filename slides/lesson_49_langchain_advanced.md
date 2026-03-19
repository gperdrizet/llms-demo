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

# Lesson 49: LangChain advanced features

**Memory, document pipelines, and agents**

---

## Recap: LangChain basics

Last lesson we covered:
- **Chat models** — unified interface to different LLM providers
- **Chat prompt templates** — structured, reusable prompts with variables
- **Output parsers** — extract structured data from text responses
- **Basic chains** — compose components into workflows with LCEL

**Today:** Three more powerful LangChain features that make real applications possible

---

## Today's outline

1. **Memory** — giving LLMs conversation history
2. **Document loading, splitting, and embedding** — feeding your own data to an LLM
3. **Agents** — letting the LLM decide what to do next

After the slides: **Demo 6** (ReAct agent) and **Demo 7** (RAG system) bring all of this together.

---

# Memory

Giving LLMs conversation history

---

## The stateless problem

LLMs have **no memory between calls**.

Each call is independent — the model has no idea what was said before.

**Without memory:**
- User: "My name is Alex."
- User: "What's my name?"
- LLM: "I don't know your name."

**With memory:**
- The conversation history is included in every prompt
- The LLM "remembers" because it can re-read what was said

---

## How memory works in LangChain

Memory components **store and retrieve conversation history**.

The pattern:
1. **Load** past messages from memory before calling the LLM
2. **Inject** those messages into the prompt via `MessagesPlaceholder`
3. **Save** the new human message and AI response back to memory after each turn

The LLM itself doesn't change — the context window just grows with each turn.

---

## Memory classes

| Class | What it stores |
|-------|---------------|
| `ChatMessageHistory` | Raw list of messages — the foundation for all other memory |
| `ConversationBufferMemory` | Full transcript — every message, no truncation |
| `ConversationBufferWindowMemory` | A sliding window of the last *k* exchanges |
| `ConversationSummaryMemory` | Compresses older history into a running summary |
| `ConversationSummaryBufferMemory` | Summary for old history + full buffer for recent messages |

**Trade-off:** More history = better context but higher cost and latency.

---

## Choosing a memory strategy

**Short conversations** (chatbots, Q&A sessions):
- `ConversationBufferMemory` — simple, no information loss

**Long conversations** (tutors, assistants):
- `ConversationBufferWindowMemory` — keep recent context, drop the old
- `ConversationSummaryMemory` — preserve meaning, reduce token cost

**Very long or open-ended sessions:**
- `ConversationSummaryBufferMemory` — best of both: summary + recent window

**Production applications** store memory externally (Redis, PostgreSQL) using `RedisChatMessageHistory` or `PostgresChatMessageHistory`.

---

## RunnableWithMessageHistory

The modern LCEL-compatible wrapper for adding memory to any chain.

**Key idea:** Wrap an existing chain with `RunnableWithMessageHistory` and provide:
- A function that returns a `ChatMessageHistory` for a given session ID
- The name of the input key that carries the human message
- The name of the key where history should be injected

**Session IDs** allow multiple independent conversations to coexist — each user or session gets their own history.

---

# Document loading, splitting, and embedding

Feeding your own data to an LLM

---

## Why RAG?

LLMs are trained on public data up to a cutoff date.

They know nothing about:
- Your internal documents
- Recent events
- Private data

**Retrieval-Augmented Generation (RAG)** solves this by retrieving relevant content at query time and injecting it into the prompt as context.

---

## The RAG pipeline

Four stages:

1. **Load** — read documents from their source
2. **Split** — break documents into small, overlapping chunks
3. **Embed** — convert chunks into vectors that capture semantic meaning
4. **Store** — save vectors in a vector database for fast retrieval

At query time: embed the question → find nearest chunks → pass as context to LLM.

---

## Document loaders

**Document loaders** read data from a source and return a list of `Document` objects.

Each `Document` has:
- `page_content` — the raw text
- `metadata` — source information (file path, page number, URL, etc.)

| Loader | Source |
|--------|--------|
| `TextLoader` | Plain text and Markdown files |
| `PyPDFLoader` | PDF files (page-by-page) |
| `Docx2txtLoader` | Word documents |
| `WikipediaLoader` | Wikipedia articles by search query |
| `WebBaseLoader` | Web pages via URL |
| `DirectoryLoader` | All files in a directory (with a sub-loader) |

---

## Text splitters

LLMs have a **finite context window** — you can't feed in an entire document.

Text splitters divide documents into **chunks** small enough to embed and retrieve individually.

**Key parameters:**
- `chunk_size` — maximum characters per chunk
- `chunk_overlap` — characters shared between adjacent chunks (preserves context at boundaries)

| Splitter | Strategy |
|----------|----------|
| `RecursiveCharacterTextSplitter` | Splits on `\n\n`, `\n`, ` `, `""` in order — preserves structure |
| `CharacterTextSplitter` | Splits on a single separator |
| `TokenTextSplitter` | Splits on token count, not characters |
| `MarkdownHeaderTextSplitter` | Splits at heading boundaries, preserving section context |

**`RecursiveCharacterTextSplitter` is the recommended default** for most use cases.

---

## Embeddings

**Embeddings** convert text into dense numerical vectors that capture semantic meaning.

Similar meaning → similar vectors → close together in vector space.

**This is what makes semantic search possible** — you aren't matching keywords, you're matching meaning.

| Class | Model | Notes |
|-------|-------|-------|
| `HuggingFaceEmbeddings` | Any sentence-transformers model | Local, no API key |
| `OpenAIEmbeddings` | text-embedding-ada-002 | API call, high quality |
| `OllamaEmbeddings` | Any Ollama model | Local server required |

`all-MiniLM-L6-v2` is a fast, compact model (384 dimensions) that works well for general-purpose RAG.

---

## Vector stores

A **vector store** indexes embeddings for fast nearest-neighbour search.

**At ingest:** embed chunks → store vectors + metadata + original text.

**At query time:** embed question → find the *k* most similar chunks → return them.

| Class | Backend |
|-------|---------|
| `PGVector` | PostgreSQL with pgvector extension |
| `Chroma` | Local in-process or persistent on disk |
| `FAISS` | Facebook AI Similarity Search — in-memory, very fast |
| `Pinecone` | Managed cloud vector database |

**`PGVector`** is a good production choice if you already use PostgreSQL — no extra infrastructure.
**`Chroma`** is the easiest to get started with locally.

---

## Retrievers

A **retriever** wraps a vector store and exposes a simple `get_relevant_documents(query)` interface.

LangChain provides several retrieval strategies built on top of vector stores:

| Retriever | Strategy |
|-----------|----------|
| `VectorStoreRetriever` | Plain similarity search — return top *k* chunks |
| `MultiQueryRetriever` | Generate multiple query variants, merge results |
| `ContextualCompressionRetriever` | Post-process chunks to extract only relevant sentences |
| `EnsembleRetriever` | Combine dense (vector) and sparse (BM25) search results |

For most use cases, `VectorStoreRetriever` with a well-tuned *k* is sufficient.

---

# Agents

Letting the LLM decide what to do next

---

## What is an agent?

In a standard chain, the **developer** decides the sequence of steps at design time.

In an **agent**, the **LLM** decides which actions to take at runtime, based on the result of previous steps.

**The LLM acts as a reasoning engine:**
1. Observe the current situation
2. Decide which tool to call (if any)
3. Execute the tool
4. Observe the result
5. Repeat until the task is complete

---

## Tools

**Tools** are functions the agent can call.

Each tool has:
- A **name** — what the LLM uses to refer to it
- A **description** — natural language explanation of what it does and when to use it
- An **input schema** — what arguments it expects

**The description is critical** — the LLM reads it to decide whether to use the tool. Vague descriptions lead to wrong tool selection.

LangChain provides built-in tools (`DuckDuckGoSearchRun`, `WikipediaQueryRun`, `PythonREPLTool`) and a `@tool` decorator for defining custom Python functions as tools.

---

## The ReAct pattern

**ReAct** (Reasoning + Acting) is the most common agent pattern.

Each step follows a strict cycle:

| Step | What happens |
|------|-------------|
| **Thought** | The LLM reasons about what to do next |
| **Action** | The LLM selects a tool and provides arguments |
| **Observation** | The tool runs and returns a result |

The cycle repeats until the LLM produces a **Final Answer** instead of an Action.

This structure keeps the LLM grounded — it must justify each action with explicit reasoning.

---

## Agent components in LangChain

| Component | Role |
|-----------|------|
| `Tool` / `@tool` | A callable function the agent can invoke |
| `create_react_agent` | Builds a ReAct agent from a prompt, LLM, and tool list |
| `AgentExecutor` | Runs the agent loop — calls LLM, executes tools, collects observations |
| `AgentFinish` | Signals the agent has produced a final answer |
| `AgentAction` | Represents a single tool call decision |

**`AgentExecutor`** handles the observation loop so you don't have to write it manually. It also enforces a `max_iterations` limit to prevent infinite loops.

---

## Agent vs. chain: when to use each

**Use a chain when:**
- The steps are known at design time
- The workflow is deterministic
- You need predictable latency and cost

**Use an agent when:**
- The number of steps is unknown in advance
- The task requires choosing between multiple tools
- You need the model to adapt based on intermediate results

**Agents are more powerful but less predictable.** Start with chains and reach for agents only when the task genuinely requires dynamic decision-making.

---

## What we covered today

1. **Memory** — store and inject conversation history; choose a strategy based on length and cost
2. **Document pipelines** — load → split → embed → store; retrieve at query time
3. **Agents** — LLM as decision-maker; ReAct pattern; tools with clear descriptions

**Next steps:**
1. **Demo 6:** ReAct agent with custom tools
2. **Demo 7:** RAG pipeline with pgvector
3. **Activity 4:** Add tools to a ReAct agent
4. **Activity 5:** Extend the RAG system with a new document source

---

## Additional resources

**LangChain documentation:**
- [Memory](https://python.langchain.com/docs/modules/memory/)
- [Document loaders](https://python.langchain.com/docs/modules/data_connection/document_loaders/)
- [Text splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [Vector stores](https://python.langchain.com/docs/modules/data_connection/vectorstores/)
- [Agents](https://python.langchain.com/docs/modules/agents/)

**Papers:**
- [ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2022)](https://arxiv.org/abs/2210.03629)
- [RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)

---

# Questions?

**Next:** Demo 6 — ReAct agent in action, Demo 7 — RAG knowledge system
