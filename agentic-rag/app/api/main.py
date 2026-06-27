from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, Form

from app.graph.workflow import run_agentic_rag
from app.rag.ingestion import ingest_documents


app = FastAPI(title="Agentic RAG API")


class QueryRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query")
def query(request: QueryRequest):
    return run_agentic_rag(request.question)


@app.post("/ingest")
async def ingest(
    files: list[UploadFile] = File(default=[]),
    urls: str = Form(default=""),
):
    prepared_files = []
    for uploaded in files:
        content = await uploaded.read()
        prepared_files.append(_ApiUpload(uploaded.filename, content))

    url_list = [url.strip() for url in urls.splitlines() if url.strip()]
    count = ingest_documents(files=prepared_files, urls=url_list)
    return {"indexed_chunks": count}


class _ApiUpload:
    def __init__(self, name: str, content: bytes):
        self.name = name
        self._content = content

    def read(self) -> bytes:
        return self._content
