# Agentic RAG

A document-grounded Agentic RAG project using LangGraph, LangChain, Groq, Chroma, CrewAI role agents, Streamlit, and FastAPI.

## Run

```powershell
cd agentic-rag
copy .env.example .env
streamlit run streamlit_app.py
```

For the API:

```powershell
uvicorn app.api.main:app --reload
```

## Flow

1. Planner decides the retrieval strategy.
2. Query rewriter creates a better semantic query.
3. Retriever pulls chunks from Chroma.
4. Verifier checks if the context can answer the question.
5. Reflector retries retrieval when evidence is weak.
6. Report writer creates the final grounded answer.

