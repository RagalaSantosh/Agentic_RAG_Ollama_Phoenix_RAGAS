
import os, logging, time
from typing import List, Dict, Any, Optional
from llama_index.llms.ollama import Ollama
from httpx import ReadTimeout
from .retriever import vector_retrieve
from dotenv import load_dotenv
from api.li_memory import get_memory
load_dotenv(r"C:/Users/user/Projects/NorthBay8/.env")
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

OLLAMA_BASE = os.getenv("OLLAMA_HOST")

GEN_MODEL        = os.getenv("GEN_MODEL")
GEN_MAX_TOKENS   = int(os.getenv("GEN_MAX_TOKENS"))
GEN_NUM_CTX      = int(os.getenv("GEN_NUM_CTX"))
GEN_TEMPERATURE  = float(os.getenv("GEN_TEMPERATURE", "0.2"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "6000"))
MAX_CHUNK_CHARS   = int(os.getenv("MAX_CHUNK_CHARS", "1500"))
GEN_TIMEOUT_S     = float(os.getenv("GEN_TIMEOUT_S", "300.0"))
MAX_MEMORY_CHARS  = int(os.getenv("MAX_MEMORY_CHARS", "1200"))

LLM = Ollama(
    model=GEN_MODEL,
    base_url=OLLAMA_BASE,
    request_timeout=GEN_TIMEOUT_S,
    keep_alive="10m",
    additional_kwargs={
        "num_predict": GEN_MAX_TOKENS,
        "num_ctx": GEN_NUM_CTX,
        "temperature": GEN_TEMPERATURE,
    },
)


def _format_sources(nodes, top_k: int) -> List[Dict[str, Any]]:
    out = []
    for n in nodes[:top_k]:
        meta = getattr(n, "metadata", {}) or {}
        raw  = getattr(n, "_safe_text", None) or getattr(n, "text", None) or n.get_text() or ""
        snippet = raw[:500]
        score = getattr(n, "score", None)
        try:
            score = float(score) if score is not None else None
        except Exception:
            score = None
        out.append({
            "id": getattr(n, "node_id", None) or meta.get("doc_id"),
            "source": meta.get("source", "unknown"),
            "page": meta.get("page"),
            "score": score,
            "text": snippet,
            "content": snippet,
            "snippet": snippet,
        })
    return out

def _build_context(nodes: list, max_total: int, max_per_chunk: int) -> str:
    parts: List[str] = []
    total = 0
    for n in nodes:
        txt = getattr(n, "_safe_text", None) or getattr(n, "text", None) or n.get_text() or ""
        if not txt:
            continue
        piece = txt[:max_per_chunk].strip()
        if not piece:
            continue
        add_len = len(piece) + 2
        if total + add_len > max_total:
            remaining = max_total - total - 2
            if remaining > 0:
                parts.append(piece[:remaining].rstrip())
            break
        parts.append(piece)
        total += add_len
    return "\n\n".join(parts)


def _build_history_text(session_id: str, max_chars: int = MAX_MEMORY_CHARS) -> str:
    """
    Build clean chat history for the prompt.
    Only include USER/ASSISTANT turns and drop system/meta instructions.
    """
    try:
        mem = get_memory(session_id)
        turns = mem.get_all() if hasattr(mem, "get_all") else []

        parts = []
        for m in turns:

            if isinstance(m, dict): 
                role = str(m.get("role", "")).upper()
                content = str(m.get("content", "")).strip()
            else: 
                role = getattr(m, "role", "").upper()
                content = getattr(m, "content", "").strip()


            if not content:
                continue
            if role not in {"USER", "ASSISTANT"}:
                continue
            if "output must be" in content.lower() or "json" in content.lower():

                continue

            parts.append(f"{role}: {content}")

        text = "\n".join(parts)

        return text if len(text) <= max_chars else text[-max_chars:]
    except Exception as e:
        logger.warning(f"[memory] failed to build history: {e}")
        return ""



def answer(query: str, top_k: int = 4, session_id: str = "anon"):

    t0 = time.perf_counter()

    nodes = vector_retrieve(query=query, top_k=top_k)
    if not nodes:
        return "I don't have that in the documents.", []

    context = _build_context(nodes, MAX_CONTEXT_CHARS, MAX_CHUNK_CHARS)
    if not context.strip():
        return "I don't have that in the documents.", []

    history_text = _build_history_text(session_id)
    history_block = f"\n\nCHAT HISTORY (recent):\n{history_text.strip()}" if history_text.strip() else ""

    print("history", history_block)
    prompt = (
        "You are a strict RAG assistant. Ground your answer in the retrieved CONTEXT. "
        "You may also use the recent CHAT HISTORY for pronoun/coreference resolution or facts explicitly stated there. "
        "If neither provides enough information, say so briefly. Be concise.\n\n"
        f"{history_block}\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {query}\nANSWER:"
    )

    try:
        resp = LLM.complete(prompt)
        text = (resp.text or "").strip()
    except ReadTimeout:
        logger.warning("LLM ReadTimeout for query=%r", query)
        raise

    sources = _format_sources(nodes, top_k=top_k)
    logger.info("[answer] nodes=%d len(answer)=%d time=%.3fs",
                len(nodes), len(text), time.perf_counter() - t0)
    return text if text else "I don't have that in the documents.", sources
