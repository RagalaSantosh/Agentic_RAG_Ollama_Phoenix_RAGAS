import os
import json
import time
import requests
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv

load_dotenv(r"C:/Users/user/Projects/NorthBay8/.env")

RAG_URL   = os.getenv("RAG_URL", "http://localhost:8000/v1/chat/completions")
OUT_FILE  = os.getenv("OUT_FILE", "evaluation/results.jsonl")
RAG_TO    = float(os.getenv("RAG_TIMEOUT", "60"))

ragas_available = False
evaluate_fn = None
metric_context_recall = None

try:
    from ragas import evaluate as _evaluate
    from ragas.metrics import context_recall as _context_recall 
    evaluate_fn = _evaluate
    metric_context_recall = _context_recall
    ragas_available = True
except Exception:
    ragas_available = False
    evaluate_fn = None
    metric_context_recall = None

DEFAULT_TESTS = [
    {"question": "What are the key steps in onboarding?", "reference": "registration, KYC, approval, activation"},
    {"question": "Where is eligibility criteria stated?", "reference": "section 3"},
]

def load_tests(path: str = None):
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    return DEFAULT_TESTS

def _is_openai_chat(url: str) -> bool:
    return "/v1/chat/completions" in url

def predictor(q: str) -> Dict[str, Any]:
    """Calls your RAG API and returns {'answer', 'sources', 'raw'}."""
    try:
        if _is_openai_chat(RAG_URL):
            payload = {
                "model": os.getenv("GEN_MODEL", "llama3.1:8b"),
                "messages": [{"role": "user", "content": q}],
                "stream": False,
            }
            r = requests.post(RAG_URL, json=payload, timeout=RAG_TO)
            r.raise_for_status()
            data = r.json()
            answer = (
                data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
            )
            sources = data.get("references", [])
            return {"answer": answer, "sources": sources, "raw": data}
        else:
            r = requests.post(RAG_URL, json={"message": q}, timeout=RAG_TO)
            r.raise_for_status()
            data = r.json()
            answer = data.get("answer", "")
            sources = data.get("references") or data.get("sources") or []
            return {"answer": answer, "sources": sources, "raw": data}
    except Exception as e:
        return {"answer": f"⚠️ Error calling RAG API: {e}", "sources": [], "raw": {"error": str(e)}}


_embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def _semantic_cosine(a: str, b: str) -> float:
    a, b = a or "", b or ""
    if not a.strip() or not b.strip():
        return 0.0
    try:
        emb = _embed_model.encode([a, b], convert_to_tensor=True, normalize_embeddings=True)
        sim = float(util.cos_sim(emb[0], emb[1]))
        return float(np.clip(sim, 0.0, 1.0)) 
    except Exception as e:
        print(f"[warn] embedding similarity failed: {e}")
        return 0.0


def evaluate_case(question: str, reference: str, answer: str, contexts: List[str]) -> Dict[str, Any]:
    q = (question or "").strip()
    a = (answer or "").strip()
    ctxs = contexts or []
    joined_ctx = " ".join([c for c in ctxs if c])[:1500]

    gt = (reference or "").strip() or joined_ctx


    sim_ans_ctx = _semantic_cosine(a, joined_ctx)  
    sim_ans_gt = _semantic_cosine(a, gt)           
    sim_q_ans = _semantic_cosine(q, a)              
    sim_q_ctx = _semantic_cosine(q, joined_ctx)   

    out = {
        "faithfulness": round(sim_ans_ctx, 3),       
        "correctness": round(sim_ans_gt, 3),        
        "relevance": round(sim_q_ctx, 3),            
    }
    return out


def _json_clean(obj):
    import math
    if isinstance(obj, dict):
        return {k: _json_clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_clean(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj

def run_eval(test_file: str = None):
    tests = load_tests(test_file)
    os.makedirs(os.path.dirname(OUT_FILE) or ".", exist_ok=True)
    results = []
    t0 = time.time()

    for rec in tests:
        q = rec["question"]
        ref = rec.get("reference", "")

        pred = predictor(q)
        ans = pred["answer"]

        ctxs = []
        for s in (pred["sources"] or []):
            ctxs.append(s.get("content") or s.get("text") or s.get("snippet") or "")

        scores = evaluate_case(q, ref, ans, ctxs)

        row = {
            "question": q,
            "reference": ref,
            "answer": ans,
            "ragas": scores,  
        }
        results.append(row)

        with open(OUT_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(_json_clean(row), ensure_ascii=False) + "\n")

    dur = round(time.time() - t0, 2)
    print(f"Saved {len(results)} evaluation results to {OUT_FILE} in {dur}s")
    return results

