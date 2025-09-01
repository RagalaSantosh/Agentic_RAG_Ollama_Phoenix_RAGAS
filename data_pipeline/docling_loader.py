import re
import os
import logging
from pathlib import Path
from typing import List, Dict, Any
from functools import lru_cache
from llama_index.core import Document
from dotenv import load_dotenv


from llama_index.core import (
    Settings,
    Document,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

try:
    from docling.document_converter import DocumentConverter
    _HAS_DOCLING = True
except Exception:
    _HAS_DOCLING = False

load_dotenv()
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


PG_PARAMS = {
    "host": os.getenv("PG_HOST"),
    "port": int(os.getenv("PG_PORT")),
    "database": os.getenv("PG_DB"),
    "user": os.getenv("PG_USER"),
    "password": os.getenv("PG_PASS"),
    "table_name": os.getenv("RAG_TABLE"),
}
GEN_MODEL   = os.getenv("GEN_MODEL")
EMBED_MODEL = os.getenv("EMBED_MODEL")
OLLAMA_BASE = os.getenv("OLLAMA_BASE")
EMBED_DIM   = int(os.getenv("EMBED_DIM", "1024"))


EMBED = OllamaEmbedding(model_name=EMBED_MODEL, base_url=OLLAMA_BASE, request_timeout=600.0)
LLM   = Ollama(model=GEN_MODEL, base_url=OLLAMA_BASE, request_timeout=300.0, keep_alive="10m")
Settings.llm = LLM
Settings.embed_model = EMBED



def _split_variable_chunks(text: str, max_tokens: int = 400) -> List[str]:

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    chunks, cur = [], []

    def flush():
        if cur:
            chunks.append(" ".join(cur).strip())
            cur.clear()

    for ln in lines:
        if re.search(r"\b\d{5,}\b", ln) or "@" in ln:
            flush()
            chunks.append(ln)  
            continue


        if re.match(r"^#+\s", ln):
            flush()
            chunks.append(ln)
            continue


        if ln.startswith(("-", "*", "•")):
            flush()
            chunks.append(ln)
            continue

        cur.append(ln)
        if sum(len(x.split()) for x in cur) > max_tokens:
            flush()

    flush()
    return chunks


def convert_file_to_text(path: Path) -> str:

    try:
        if _HAS_DOCLING:
            converter = DocumentConverter()
            result = converter.convert(str(path))
            return result.document.export_to_markdown()
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        logger.warning(f"[docling] conversion failed for {path.name}: {e}; using raw text fallback")
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""


def _wrap_documents(docs: List[Dict[str, Any]]) -> List[Document]:

    out: List[Document] = []
    for d in docs:
        doc_id = str(d.get("id") or "")
        text = d.get("text") or ""
        source = d.get("source") or "unknown"
        if not text:
            continue

        meta_rest = {k: v for k, v in d.items() if k != "text"}
        chunks = _split_variable_chunks(text, max_tokens=400)

        for i, chunk in enumerate(chunks):
            cid = f"{doc_id}::chunk{i}"
            out.append(Document(
                text=chunk,
                doc_id=cid,
                metadata={"source": source, "doc_id": doc_id, "chunk": i, **meta_rest},
            ))

    if not out:
        logger.warning("[ingest] no non-empty documents to index")
    else:
        logger.info(f"[ingest] prepared {len(out)} variable chunks")
    return out


@lru_cache(maxsize=1)
def _get_vector_store_cached() -> PGVectorStore:
    return PGVectorStore.from_params(
        database=PG_PARAMS["database"],
        host=PG_PARAMS["host"],
        port=PG_PARAMS["port"],
        user=PG_PARAMS["user"],
        password=PG_PARAMS["password"],
        table_name=PG_PARAMS["table_name"],
        embed_dim=EMBED_DIM,
    )

def get_vector_store() -> PGVectorStore:
    return _get_vector_store_cached()

@lru_cache(maxsize=1)
def _get_index_cached() -> VectorStoreIndex:
    vstore = get_vector_store()
    storage = StorageContext.from_defaults(vector_store=vstore)
    return VectorStoreIndex.from_vector_store(vstore, storage_context=storage, llm=LLM)

def get_index() -> VectorStoreIndex:
    return _get_index_cached()


def upsert_documents(docs: List[Dict[str, Any]], batch_size: int = 50) -> None:

    vstore = get_vector_store()
    li_docs = _wrap_documents(docs)
    if not li_docs:
        return

    storage = StorageContext.from_defaults(vector_store=vstore)
    total = len(li_docs)
    for i in range(0, total, batch_size):
        batch = li_docs[i : i + batch_size]
        logger.info(f"[upsert] indexing batch {i}..{i+len(batch)-1} / {total}")
        VectorStoreIndex.from_documents(batch, storage_context=storage, llm=LLM)

    refresh_global_index()


def refresh_global_index() -> None:

    _get_index_cached.cache_clear() 
    idx = get_index()

    try:
        from rag_core.retriever import set_index 
        set_index(idx)
    except Exception as e:
        
        try:
            import importlib
            retriever_mod = importlib.import_module("rag_core.retriever")
            setattr(retriever_mod, "INDEX", idx)
        except Exception:
            logger.warning(f"[ingest] could not assign INDEX via fallback: {e}")
            raise

    logger.info("[ingest] global retriever INDEX refreshed")


