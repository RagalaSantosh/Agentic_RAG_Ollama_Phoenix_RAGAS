# crewai_agents/crew.py
import os, logging
from typing import Callable

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# Align with the rest of your app
OLLAMA_BASE = os.getenv("OLLAMA_BASE") or os.getenv("OLLAMA_HOST") or "http://127.0.0.1:11434"
GEN_MODEL   = os.getenv("GEN_MODEL", "llama3.1:8b")

try:
    from llama_index.llms.ollama import Ollama as LI_Ollama
except Exception:
    LI_Ollama = None


def _coerce_text(resp) -> str:
    if isinstance(resp, str):
        return resp.strip()
    try:
        # llama_index.Ollama.complete returns an object with .text
        return (getattr(resp, "text", None) or str(resp or "")).strip()
    except Exception:
        return ""


def _valid_refinement(orig: str, refined: str) -> bool:
    if not refined:
        return False
    lo = refined.lower()
    # Filter out “meta” responses
    bad = ("key elements", "generate a concise", "json", "guidelines",
           "your entire response", "output must be", "examples:", "chat history")
    if any(b in lo for b in bad):
        return False
    import re
    def toks(s: str):
        return set(t for t in re.findall(r"[a-z0-9@._]+", s.lower()) if len(t) > 2)
    o = toks(orig); r = toks(refined)
    alpha_o = {t for t in o if any(c.isalpha() for c in t)}
    return (len(alpha_o & r) >= max(1, len(alpha_o)//2))


class _RefinerCrew:
    def __init__(self, infer_fn: Callable[[str], str]):
        self._infer = infer_fn

    def kickoff(self, inputs=None):
        q = (inputs or {}).get("query") if isinstance(inputs, dict) else None
        if not q:
            return ""

        meta_markers = ("### task", "### guidelines", "json format", "chat history", "output must be")
        if any(m in q.strip().lower() for m in meta_markers):
            logger.info("[crew] meta/system-like prompt detected; returning original")
            return q

        prompt = (
            "You rewrite user questions to be concise and retrieval-friendly.\n"
            "Rules:\n"
            "1) Keep all key entities/terms from the original (names, IDs, emails).\n"
            "2) One short sentence; no commentary, no examples, no JSON.\n"
            "3) If no rewrite is needed, return the original unchanged.\n"
            "4) Output ONLY the refined question.\n\n"
            f"Original: {q}\n"
            "Refined:"
        )
        try:
            refined = _coerce_text(self._infer(prompt))
            if any(m in refined.lower() for m in meta_markers):
                logger.info("[crew] model returned meta/instructions; using original")
                return q
            return refined if _valid_refinement(q, refined) else q
        except Exception as e:
            logger.warning("[crew] refinement failed: %s; using original", e)
            return q


def build_crew():
    """
    Only Ollama via llama_index. If unavailable, returns an identity-like crew
    that effectively keeps the original question.
    """
    if LI_Ollama is None:
        logger.info("[crew] llama_index Ollama not available; using identity fallback")
        return _RefinerCrew(lambda _: "")

    # Create a single client instance for reuse
    client = LI_Ollama(model=GEN_MODEL, base_url=OLLAMA_BASE, request_timeout=20.0, keep_alive="5m")
    logger.info("[crew] Ollama refiner ready: model=%s base=%s", GEN_MODEL, OLLAMA_BASE)

    def _infer(prompt: str) -> str:
        return client.complete(prompt)

    return _RefinerCrew(_infer)
