from app.rag.retriever import retrieve_documents


def retrieve_context(query: str, k: int = 6, source_filter: str | None = None) -> dict:
    result = retrieve_documents(query, k=k, source_filter=source_filter)
    return {
        "context": result.context,
        "sources": result.sources,
        "retrieved_count": len(result.documents),
    }
