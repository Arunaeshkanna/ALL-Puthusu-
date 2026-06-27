from langchain_community.document_loaders import WebBaseLoader


def load_url_text(url: str) -> str:
    docs = WebBaseLoader(url).load()
    return "\n\n".join(doc.page_content for doc in docs)

