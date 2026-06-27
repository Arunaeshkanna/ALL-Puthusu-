from dataclasses import dataclass

from langchain_core.documents import Document

from app.rag.vectordb import get_vector_db


@dataclass
class RetrievalResult:
    query: str
    documents: list[Document]

    @property
    def context(self) -> str:
        blocks = []
        for index, doc in enumerate(self.documents, start=1):
            source = doc.metadata.get("source", "Unknown")
            blocks.append(f"[{index}] Source: {source}\n{doc.page_content}")
        return "\n\n".join(blocks)

    @property
    def sources(self) -> list[str]:
        seen = []
        for doc in self.documents:
            source = doc.metadata.get("source", "Unknown")
            if source not in seen:
                seen.append(source)
        return seen


def retrieve_documents(
    query: str,
    k: int = 6,
    source_filter: str | None = None,
) -> RetrievalResult:
    db = get_vector_db()
    search_filter = {"source": source_filter} if source_filter else None
    docs = db.similarity_search(query, k=k, filter=search_filter)
    return RetrievalResult(query=query, documents=docs)


def list_document_sources() -> list[str]:
    db = get_vector_db()
    data = db.get(include=["metadatas"])
    sources = []
    for metadata in data.get("metadatas", []):
        source = metadata.get("source", "Unknown") if metadata else "Unknown"
        if source not in sources:
            sources.append(source)
    return sorted(sources)
