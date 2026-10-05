"""Wikipedia ingestor.

Fetches Wikipedia articles by topic and splits them into chunks
suitable for embedding and vector storage.
"""

from typing import Any

import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from .base import BaseIngestor


class WikipediaIngestor(BaseIngestor):
    """Ingestor that fetches Wikipedia articles by topic name."""

    _API_URL = "https://en.wikipedia.org/w/api.php"
    _USER_AGENT = "llms-demo/1.0 (https://github.com/gperdrizet/llms-demo)"

    def __init__(
        self,
        load_max_docs: int = 3,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        
        self.load_max_docs = load_max_docs
        
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    @property
    def source_type(self) -> str:
        return "Wikipedia"

    def _query(self, params: dict[str, str | int]) -> dict[str, Any]:
        try:
            response = requests.get(
                self._API_URL,
                params={"action": "query", "format": "json", **params},
                headers={"User-Agent": self._USER_AGENT},
                timeout=30,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"Could not reach the Wikipedia API: {exc}") from exc

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                retry_after = f"{retry_after} seconds"
            retry_hint = (
                f"Retry after {retry_after}."
                if retry_after
                else "Wait before retrying and reduce the number of articles requested."
            )
            raise RuntimeError(
                f"Wikipedia API rate limit reached (HTTP 429). {retry_hint}"
            )

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(
                f"Wikipedia API returned HTTP {response.status_code}. "
                "Check network access or try again later."
            ) from exc

        try:
            data = response.json()
        except requests.exceptions.JSONDecodeError as exc:
            content_type = response.headers.get("Content-Type", "unknown")
            raise RuntimeError(
                f"Wikipedia API returned a non-JSON response "
                f"(HTTP {response.status_code}, Content-Type: {content_type}). "
                "Check for a proxy or network filter blocking Wikipedia."
            ) from exc

        if not isinstance(data, dict):
            raise RuntimeError("Wikipedia API returned an unexpected response format.")
        if "error" in data:
            raise RuntimeError(f"Wikipedia API error: {data['error']}")
        if not isinstance(data.get("query"), dict):
            raise RuntimeError("Wikipedia API response is missing query results.")

        return data["query"]

    def load(self, source: str) -> list[Document]:
        """Fetch Wikipedia articles matching *source* and return chunks.

        Args:
            source: Wikipedia search query / topic name,
                    e.g. "Python programming language".

        Returns:
            List of text chunks as LangChain Documents.
        """

        results = self._query({
            "list": "search",
            "srsearch": source[:300],
            "srlimit": self.load_max_docs,
            "srprop": "",
        })
        docs = []

        for result in results["search"][:self.load_max_docs]:
            title = result["title"]
            page_results = self._query({
                "titles": title,
                "prop": "extracts|info|pageprops",
                "explaintext": 1,
                "inprop": "url",
                "ppprop": "disambiguation",
                "redirects": 1,
            })
            page = next(iter(page_results["pages"].values()))

            if "missing" in page or "disambiguation" in page.get("pageprops", {}):
                continue

            summary_results = self._query({
                "pageids": page["pageid"],
                "prop": "extracts",
                "explaintext": 1,
                "exintro": 1,
            })
            summary_page = next(iter(summary_results["pages"].values()))
            docs.append(Document(
                page_content=page["extract"][:4000],
                metadata={
                    "title": title,
                    "source": page["fullurl"],
                    "summary": summary_page["extract"],
                },
            ))

        return self._splitter.split_documents(docs)
