"""ChromaDB-backed vector store for semantic document search.

Wraps ChromaDB behind a simple interface: add documents, search by
query. The ``from_directory`` factory handles the full pipeline of
reading markdown files, chunking them, and indexing into an
ephemeral (in-memory) collection.
"""

import hashlib
from pathlib import Path

import chromadb

from .chunker import chunk_markdown


class VectorStore:
    """Semantic search over document chunks using ChromaDB.

    Uses ChromaDB's default embedding function (Sentence Transformers)
    and an ephemeral in-memory client. The knowledge base is small
    enough that re-indexing on startup is fast and avoids stale cache
    issues.

    Attributes:
        _collection: The underlying ChromaDB collection.
    """

    def __init__(self, collection: chromadb.Collection) -> None:
        """Initialize with a ChromaDB collection.

        Args:
            collection: A ChromaDB collection to store and query
                documents against.
        """
        self._collection = collection

    def add(
        self,
        documents: list[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
    ) -> None:
        """Add documents to the vector store.

        Args:
            documents: List of text strings to index.
            metadatas: Optional metadata dicts, one per document.
            ids: Optional document IDs. If not provided, deterministic
                IDs are generated from content hashes.
        """
        if not documents:
            return

        if ids is None:
            ids = [
                hashlib.sha256(doc.encode()).hexdigest()[:16]
                for doc in documents
            ]

        kwargs: dict = {"documents": documents, "ids": ids}
        if metadatas is not None:
            kwargs["metadatas"] = metadatas

        self._collection.add(**kwargs)

    def search(self, query: str, n_results: int = 3) -> list[str]:
        """Search for documents similar to the query.

        Args:
            query: The search query string.
            n_results: Maximum number of results to return.

        Returns:
            A list of document text strings, ordered by relevance.
        """
        if self._collection.count() == 0:
            return []

        results = self._collection.query(
            query_texts=[query],
            n_results=min(n_results, self._collection.count()),
        )

        documents = results.get("documents")
        if not documents or not documents[0]:
            return []

        return documents[0]

    @classmethod
    def from_directory(
        cls,
        directory: Path,
        collection_name: str = "knowledge_base",
    ) -> "VectorStore":
        """Create a vector store from a directory of markdown files.

        Reads all ``.md`` files from the directory, chunks them using
        ``chunk_markdown``, and indexes the chunks into an ephemeral
        ChromaDB collection.

        Args:
            directory: Path to a directory containing ``.md`` files.
            collection_name: Name for the ChromaDB collection.

        Returns:
            A populated VectorStore ready for searching.
        """
        client = chromadb.Client()
        collection = client.get_or_create_collection(name=collection_name)
        store = cls(collection)

        md_files = sorted(directory.glob("*.md"))
        all_documents: list[str] = []
        all_metadatas: list[dict] = []

        for md_file in md_files:
            text = md_file.read_text()
            chunks = chunk_markdown(text, source=md_file.name)
            for chunk in chunks:
                all_documents.append(chunk["text"])
                all_metadatas.append({"source": chunk["source"]})

        if all_documents:
            store.add(all_documents, metadatas=all_metadatas)

        return store
