# evaluation/ragas_helper.py
import os
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

ENABLE_RAGAS = os.getenv("ENABLE_RAGAS", "false").lower() == "true"

def _prepare_env():
    """
    Point Ragas' OpenAI wrapper to YOUR FastAPI (which proxies to Ollama).
    The dummy key is enough for most OpenAI-compatible clients.
    """
    os.environ.setdefault("OPENAI_API_KEY", "local-eval")  # dummy
    # IMPORTANT: base URL WITHOUT trailing slash; legacy clients will hit /completions
    os.environ.setdefault("OPENAI_BASE_URL", os.getenv("OPENAI_BASE_URL", "http://localhost:8000"))

def _get_openai_llm():
    """
    Ragas changed module paths across versions; try them all.
    """
    try:
        from ragas.llms import OpenAI  # older
        return OpenAI
    except Exception:
        pass
    try:
        from ragas.llms.openai import OpenAI  # mid
        return OpenAI
    except Exception:
        pass
    try:
        from ragas.llms.base import OpenAI  # newer
        return OpenAI
    except Exception as e:
        raise ImportError("No compatible ragas.llms.OpenAI class found") from e

def maybe_eval(question: str, answer: str, contexts: List[str]) -> Dict[str, Any]:
    """
    Minimal, robust Ragas run using ONLY LLM-based metrics (faithfulness, answer_relevancy).
    Returns {} on any failure.
    """
    if not ENABLE_RAGAS:
        return {}

    q = (question or "").strip()
    a = (answer or "").strip()
    ctxs = [c.strip() for c in (contexts or []) if isinstance(c, str) and c.strip()]
    if not q or not a or not ctxs:
        return {}

    try:
        _prepare_env()

        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        from datasets import Dataset

        ds = Dataset.from_dict({
            "question": [q],
            "answer": [a],
            "contexts": [ctxs],
        })

        OpenAICls = _get_openai_llm()
        # Use your gen model name so you evaluate with the same LLM
        model_name = os.getenv("RAGAS_LLM_MODEL", os.getenv("GEN_MODEL", "llama3.1:8b"))
        llm = OpenAICls(model=model_name)

        res = evaluate(
            ds,
            metrics=[faithfulness, answer_relevancy],
            llm=llm,
        )

        # Be defensive: pick only known metric columns and numeric-like values
        row = res.to_pandas().iloc[0].to_dict()
        out: Dict[str, Any] = {}
        for key, val in row.items():
            if key in {"faithfulness", "answer_relevancy"}:
                try:
                    out[key] = float(val) if val is not None else None
                except Exception:
                    pass
        return out

    except Exception as e:
        logger.warning(f"[ragas] evaluation skipped: {e}")
        return {}
