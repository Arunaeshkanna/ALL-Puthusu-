from functools import lru_cache

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from app.config import EMBEDDING_MODEL, VECTOR_STORE_DIR


@lru_cache(maxsize=1)
def get_vector_db() -> Chroma:
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    return Chroma(
        persist_directory=str(VECTOR_STORE_DIR),
        embedding_function=embeddings,
    )


def reset_vector_db_cache() -> None:
    get_vector_db.cache_clear()

