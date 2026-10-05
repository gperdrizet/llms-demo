# Activity 5: Extending the RAG knowledge system

**Objective:** Add a new document source to the RAG system by implementing the `BaseIngestor` interface.

**Duration:** 45-60 minutes

---

## Overview

The RAG demo uses a modular ingestor system. Each ingestor knows how to:
1. **Load** documents from some source (Wikipedia, a URL, …)
2. **Split** them into chunks suitable for embedding

All ingestors share the same interface so the rest of the system doesn't need to care where the documents came from.

```
Source                  Ingestor             Vector store
------                  --------             ------------
Wikipedia topic    →    WikipediaIngestor →  PGVector (PostgreSQL)
URL                →    URLIngestor       ↗
```

Only `WikipediaIngestor` is implemented in the demo. Your task is to add a `URLIngestor` - and once registered, it will automatically appear as an option in the demo UI.

---

## Setup

### 1. Start the RAG demo

Make sure your LLM backend is running (llama.cpp or Ollama), then:

```bash
python demos/rag_system/rag_demo.py
```

Open the Gradio interface and confirm the **Ingest** tab shows "Wikipedia" as the only source.

### 2. Create your activity file

Create `activities/my_ingestors.py` for testing outside the demo:

```python
"""Activity 5: Testing new ingestors."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "demos/rag_system"))
```

---

## Part 1: Explore the existing system

### Step 1: Read the base interface

Open `demos/rag_system/ingestors/base.py`. Note:
- `source_type` — a property returning the display name shown in the UI radio button
- `load(source)` — the only method you need to implement; returns a list of `Document` objects

### Step 2: Read the Wikipedia ingestor

Open `demos/rag_system/ingestors/wikipedia.py`:
- It queries Wikipedia's HTTPS API to search for articles and fetch their text
- It checks HTTP and API errors before parsing results, so rate limits are reported clearly
- It passes the raw documents through `RecursiveCharacterTextSplitter` so they become small chunks
- It preserves article title, source URL, and summary metadata

If ingestion reports **HTTP 429**, Wikipedia is rate limiting requests. Wait before
retrying (follow `Retry-After` if shown), and try fewer articles with
`WikipediaIngestor(load_max_docs=1)`. Non-JSON responses can also indicate a proxy
or network filter; these errors occur before embedding or database storage.

### Step 3: Test the Wikipedia ingestor

Add this to `activities/my_ingestors.py` and run it:

```python
from ingestors import WikipediaIngestor

ingestor = WikipediaIngestor(load_max_docs=1)  # just 1 article to keep it quick
docs = ingestor.load("Large language model")

print(f"Loaded {len(docs)} chunks")
print(f"\nFirst chunk:\n{docs[0].page_content[:300]}")
print(f"\nMetadata: {docs[0].metadata}")
```

**Questions to consider:**
- How large is each chunk in characters?
- What metadata fields does `WikipediaIngestor` attach?
- What happens if you search for a topic that doesn't exist on Wikipedia?

---

## Part 2: Implement URLIngestor

**Goal:** Create an ingestor that fetches and indexes the content of any web page.

### Step 1: Install the dependency

`WebBaseLoader` requires `beautifulsoup4`:

```bash
pip install beautifulsoup4
```

### Step 2: Create the ingestor file

Create `demos/rag_system/ingestors/url.py`:

```python
"""URL ingestor - loads a web page from a given URL."""

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from .base import BaseIngestor


class URLIngestor(BaseIngestor):
    """Ingestor that fetches and chunks content from a web page URL."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        # TODO: Store the splitter
        pass

    @property
    def source_type(self) -> str:
        # TODO: Return a human-readable name (e.g. "URL")
        pass

    def load(self, source: str) -> list[Document]:
        """Fetch and chunk the page at *source* URL.

        Args:
            source: A full URL, e.g. "https://docs.python.org/3/tutorial/"

        Returns:
            List of split Document chunks.
        """
        # TODO: Use WebBaseLoader(web_paths=[source]) to fetch the page
        # TODO: Split the loaded documents and return the chunks
        pass
```

### Step 3: Test your ingestor

Add to `activities/my_ingestors.py`:

```python
from ingestors.url import URLIngestor

url_ingestor = URLIngestor()
docs = url_ingestor.load("https://en.wikipedia.org/wiki/Retrieval-augmented_generation")

print(f"Loaded {len(docs)} chunks")
for doc in docs[:3]:
    print(f"  source: {doc.metadata.get('source', 'unknown')}")
    print(f"  preview: {doc.page_content[:120]}\n")
```

### Success criteria

- [ ] `URLIngestor` extends `BaseIngestor`
- [ ] `source_type` returns a meaningful string
- [ ] `load()` returns a non-empty list of chunks for a valid URL
- [ ] `doc.metadata` includes the source URL

### Hints

<details>
<summary>Click to reveal hints</summary>

1. `WebBaseLoader` signature: `WebBaseLoader(web_paths=["https://..."])`
2. `loader.load()` returns a list of `Document` objects - split them with your splitter
3. The metadata will contain `"source"` (the URL) automatically - useful for citations
4. If a page returns no useful text, try a different URL; some sites block scrapers

</details>

---

## Part 3: Register your ingestor in the demo

### Step 1: Export your classes from the package

Open `demos/rag_system/ingestors/__init__.py` and add your new class:

```python
from .base import BaseIngestor
from .wikipedia import WikipediaIngestor
from .url import URLIngestor               # add

__all__ = ["BaseIngestor", "WikipediaIngestor", "URLIngestor"]
```

### Step 2: Add it to the INGESTORS registry

Open `demos/rag_system/rag_demo.py`. Find the `INGESTORS` dictionary and uncomment / add your entry:

```python
from ingestors import WikipediaIngestor, URLIngestor

INGESTORS = {
    "Wikipedia": WikipediaIngestor(),
    "URL": URLIngestor(),
}
```

### Step 3: Verify in the UI

Restart the demo. You should see **"URL"** appear alongside "Wikipedia" in the ingestor radio. Test it and confirm chunks appear in the Sources panel after querying.

---

## Key takeaways

1. **Interface > implementation** — the Gradio UI only knows about `BaseIngestor`; adding a new source is just one new file + one dict entry
2. **Chunking strategy matters** — chunk size affects retrieval quality; too large and you retrieve irrelevant noise, too small and you lose context
3. **Metadata is your citation system** — `Document.metadata` carries source information that surfaces in the "Sources" panel after querying
4. **The same embeddings model processes all sources** — as long as text is in the model's language, the source type is irrelevant to the retriever

**Next step:** Try ingesting multiple sources on the same topic (e.g. Wikipedia + a URL) and see how the answers change!

**Duration:** 45-60 minutes

---
