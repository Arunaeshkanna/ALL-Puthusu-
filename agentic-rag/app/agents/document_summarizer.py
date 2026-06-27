from langchain_core.documents import Document

from app.agents.common import invoke_prompt, make_agent


document_summarizer_agent = make_agent(
    role="Document Summary Analyst",
    goal="Summarize newly ingested documents and suggest useful questions.",
    backstory="You help users understand what their uploaded documents contain before they ask questions.",
)


def summarize_documents(chunks: list[Document]) -> str:
    if not chunks:
        return "No document content was available to summarize."

    sources = []
    excerpts = []
    total_chars = 0

    for chunk in chunks:
        source = chunk.metadata.get("source", "Unknown")
        if source not in sources:
            sources.append(source)

        text = chunk.page_content.strip()
        if not text:
            continue

        excerpt = f"Source: {source}\n{text}"
        if total_chars + len(excerpt) > 9000:
            break
        excerpts.append(excerpt)
        total_chars += len(excerpt)

    excerpt_text = "\n\n".join(excerpts)

    prompt = f"""
You are analyzing newly uploaded documents for a RAG system.

Use only the content below.

Create a useful document overview with these sections:
1. Document Summary
2. Key Topics
3. Important Entities
4. Possible Questions To Ask

Rules:
- Be specific to the uploaded content.
- Do not invent details not present in the excerpts.
- If the excerpts are limited, say that the summary is based on available chunks.
- Keep it clear and practical.

Sources:
{", ".join(sources)}

Document excerpts:
{excerpt_text}
"""
    return invoke_prompt(prompt)
