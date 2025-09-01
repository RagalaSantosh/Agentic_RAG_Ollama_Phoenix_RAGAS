
import os, json, uuid, time, logging, tempfile, asyncio, math
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager, contextmanager, nullcontext

from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from logging.handlers import RotatingFileHandler

from dotenv import load_dotenv
from rag_core.pipeline import answer
from data_pipeline.docling_loader import convert_file_to_text, upsert_documents
from data_pipeline.docling_loader import get_vector_store, LLM
from llama_index.core import StorageContext, VectorStoreIndex
from rag_core.retriever import set_index, index_info

from api.li_memory import get_memory

load_dotenv(r"C:/Users/user/Projects/NorthBay8/.env")

ENABLE_PHOENIX   = os.getenv("ENABLE_PHOENIX", "false").lower() == "true"
ENABLE_CREW      = os.getenv("ENABLE_CREW", "false").lower() == "true"
ENABLE_RAGAS     = os.getenv("ENABLE_RAGAS", "false").lower() == "true"
USE_CREW_DEFAULT = os.getenv("USE_CREW_DEFAULT", "false").lower() == "true"

span = log_query_trace = init_phoenix = shutdown_phoenix = instrument_fastapi_app = None

logger = logging.getLogger("RAG_API")
LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(), RotatingFileHandler(LOG_DIR / "app.log", maxBytes=2_000_000, backupCount=3)],
)
logger.info("BOOT: ENABLE_CREW=%s USE_CREW_DEFAULT=%s", ENABLE_CREW, os.getenv("USE_CREW_DEFAULT"))
TX_LOG_PATH = Path(os.getenv("TX_LOG_PATH", LOG_DIR / "transactions.jsonl"))

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: list
    stream: bool = False
    scope: Optional[str] = None
    k: Optional[int] = None
    use_crew: Optional[bool] = None
    class Config: extra = "allow"

if ENABLE_PHOENIX:
    try:
        from evaluation.phoenix_tracing import (
            span as _span,
            log_query_trace as _log_query_trace,
            init_phoenix as _init_phoenix,
            shutdown_phoenix as _shutdown_phoenix,
            instrument_fastapi_app as _instrument_app,
        )
        span = _span; log_query_trace = _log_query_trace
        init_phoenix = _init_phoenix; shutdown_phoenix = _shutdown_phoenix
        instrument_fastapi_app = _instrument_app
    except Exception:
        pass

def _session_id(req: Request) -> str:
    sid = (req.headers.get("x-session-id")
           or req.cookies.get("sid")
           or req.headers.get("x-client-id"))
    if sid:
        return sid.strip()

    try:
        ip = req.client.host if req.client else "anon"
    except Exception:
        ip = "anon"
    return f"ip:{ip}"


def _memory_snippet(mem, max_chars: int = 400) -> str:
    """
    Best-effort extraction of recent chat history from your memory object.
    Tries multiple common shapes; falls back silently if not available.
    Returns a short single-string snippet like:
      "User: Who is Santosh?\nAssistant: ...\nUser: What is his phone number?"
    """
    msgs: List[Dict[str, str]] = []
    try:
        # common shapes this function can handle without knowing exact class
        if hasattr(mem, "get_recent"):
            items = mem.get_recent(10)  # type: ignore[attr-defined]
        elif hasattr(mem, "history"):
            items = getattr(mem, "history")  # list-like
        elif hasattr(mem, "messages"):
            items = getattr(mem, "messages")
        elif hasattr(mem, "dump"):
            items = mem.dump()  # type: ignore[attr-defined]
        elif hasattr(mem, "get"):
            items = mem.get()   # type: ignore[attr-defined]
        else:
            items = []
        # normalize to [{'role': ..., 'content': ...}]
        for it in items or []:
            if isinstance(it, dict) and "role" in it and "content" in it:
                msgs.append({"role": str(it["role"]).lower(), "content": str(it["content"])})
            elif isinstance(it, (list, tuple)) and len(it) >= 2:
                msgs.append({"role": str(it[0]).lower(), "content": str(it[1])})
            elif isinstance(it, str):
                # assume user text if we cannot infer role
                msgs.append({"role": "user", "content": it})
    except Exception:
        pass

    if not msgs:
        return ""

    out_lines: List[str] = []
    total = 0
    for m in msgs[-8:]:
        line = f"{'User' if 'user' in m['role'] else 'Assistant'}: {m['content'].strip()}"
        if not line:
            continue
        add = len(line) + 1
        if total + add > max_chars:
            break
        out_lines.append(line)
        total += add
    return "\n".join(out_lines)

def _maybe_refine_query(q: str) -> str:
    if not q:
        return q
    if not ENABLE_CREW:
        logger.info("[crew] ENABLE_CREW=false -> skipping refinement")
        return q
    try:
        from crewai_agents.crew import build_crew
        crew = build_crew()
        logger.info("[crew] kickoff start, original=%r", q)
        out = crew.kickoff(inputs={"query": q})
        refined = None
        try:
            refined = str(out).strip()
            if hasattr(out, "output_text"):
                refined = (out.output_text or refined).strip()
            elif hasattr(out, "raw"):
                refined = (out.raw or refined).strip()
        except Exception:
            refined = None
        final_q = refined or q
        logger.info("[crew] kickoff done, refined=%r, final=%r", refined, final_q)
        return final_q
    except Exception as e:
        logger.exception("[crew] refinement failed: %s", e)
        return q

def _maybe_ragas(question: str, answer_text: str, contexts: List[str]) -> dict:
    if not ENABLE_RAGAS:
        return {}
    try:
        from evaluation.ragas_small import evaluate_case
    except Exception as e:
        logger.warning(f"[ragas] evaluator not importable: {e}")
        return {}
    try:
        raw = evaluate_case(
            question=question or "",
            reference="",
            answer=answer_text or "",
            contexts=contexts or [],
        ) or {}
        wanted = {"faithfulness", "relevance", "correctness", "answer_relevancy", "answer_correctness", "context_recall"}
        out = {}
        for k in wanted:
            v = raw.get(k)
            if isinstance(v, (int, float)) and not (math.isnan(v) or math.isinf(v)):
                out[k] = float(v)
        return out
    except Exception as e:
        logger.warning(f"[ragas] evaluation failed: {e}")
        return {}

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

def _tx(kind: str, rid: str, **fields):
    rec = {"ts": time.time(), "kind": kind, "request_id": rid, **fields}
    try:
        with open(TX_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    if ENABLE_PHOENIX and init_phoenix:
        try:
            init_phoenix()
        except Exception as e:
            logger.warning(f"Phoenix init failed: {e}")

    # --- SET INDEX ONCE HERE ---
    try:
        vstore = get_vector_store()
        storage = StorageContext.from_defaults(vector_store=vstore)
        idx = VectorStoreIndex.from_vector_store(vstore, storage_context=storage, llm=LLM)
        set_index(idx)
        logger.info("[startup] Global INDEX set from PGVector")
    except Exception as e:
        logger.error(f"[startup] Failed to set INDEX: {e}")
    # ---------------------------

    yield

    if ENABLE_PHOENIX and shutdown_phoenix:
        try:
            shutdown_phoenix()
        except Exception:
            pass

app = FastAPI(title="OpenAI-Compatible RAG API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

def _normalize_sources(sources: list) -> list:
    out = []
    for s in (sources or []):
        src = s.get("source") or s.get("metadata", {}).get("source") or ""
        page = s.get("page") or s.get("metadata", {}).get("page")
        sid  = s.get("id") or s.get("node_id")
        text = s.get("content") or s.get("text") or s.get("snippet") or "[no-snippet]"
        out.append({"id": sid, "source": src, "page": page, "text": text, "snippet": s.get("snippet") or text})
    return out

def _sse(d: dict) -> str:
    return f"data: {json.dumps(d, ensure_ascii=False)}\n\n"

def rag_span(k: int):
    if ENABLE_PHOENIX and span:
        @contextmanager
        def _cm():
            with span("chat.completions", k=k) as sp:
                try:
                    sp.set_attribute("openinference.span.kind", "CHAIN")
                    sp.set_attribute("span.kind", "chain")
                except Exception:
                    pass
                yield sp
        return _cm()
    return nullcontext()

@app.get("/v1/models")
async def get_models():
    model_id = os.getenv("GEN_MODEL", "llama3.2:3b")
    return {"object": "list", "data": [{"id": model_id, "object": "model", "owned_by": "user"}]}

@app.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest, request: Request):
    rid = getattr(request.state, "request_id", uuid.uuid4().hex)
    created = int(time.time())

    try:
        user_msg = body.messages[-1]["content"]
    except Exception:
        return JSONResponse({"error": {"message": "Invalid messages array"}}, status_code=400)

    sid = _session_id(request)
    mem = get_memory(sid)
    if hasattr(mem, "put_user"):
        mem.put_user(user_msg)
    elif hasattr(mem, "append_user"):
        mem.append_user(user_msg)
    elif hasattr(mem, "put"):
        mem.put({"role": "user", "content": user_msg})

    hdr = request.headers.get("x-use-crew")
    if hdr is not None:
        use_crew = hdr.strip().lower() in {"1", "true", "yes", "y", "on"}
    else:
        use_crew_default = os.getenv("USE_CREW_DEFAULT", "false").lower() == "true"
        use_crew = body.use_crew if getattr(body, "use_crew", None) is not None else use_crew_default

    logger.info(
        "[crew] effective=%s (ENABLE_CREW=%s, hdr=%r, body=%r, default_env=%s)",
        use_crew, ENABLE_CREW, hdr, getattr(body, "use_crew", None),
        os.getenv("USE_CREW_DEFAULT", "false")
    )

    crew_meta = {"enabled": bool(use_crew and ENABLE_CREW), "used": False, "original": user_msg, "refined": None}
    k = int(max(1, (body.k or 4)))
    answer_timeout_s = float(os.getenv("ANSWER_TIMEOUT_S", "100.0"))
    model_id = os.getenv("GEN_MODEL", "llama3.2:3b")

    want_stream = bool(getattr(body, "stream", False))
    accept_hdr = (request.headers.get("accept") or "")
    if "text/event-stream" in accept_hdr.lower():
        want_stream = True

    q = user_msg
    if crew_meta["enabled"]:
        refined = _maybe_refine_query(user_msg) or user_msg
        crew_meta["refined"] = refined
        crew_meta["used"] = (refined != user_msg)
        q = refined

    convo = _memory_snippet(mem, max_chars=400)
    q_for_answer = f"{q}\n\n(Conversation context: {convo})" if convo else q

    if want_stream:

        async def agen_realstream():
            yield _sse({
                "id": f"chatcmpl-{created}",
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            })

            if crew_meta.get("enabled"):
                yield _sse({
                    "id": f"chatcmpl-{created}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_id,
                    "choices": [{"index": 0, "delta": {"content": "↻ refining query...\n"}, "finish_reason": None}],
                })
                if crew_meta["refined"]:
                    yield _sse({
                        "id": f"chatcmpl-{created}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model_id,
                        "choices": [{"index": 0, "delta": {"content": f"refined: {q}\n"}, "finish_reason": None}],
                    })

            yield _sse({
                "id": f"chatcmpl-{created}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_id,
                "choices": [{"index": 0, "delta": {"content": "↻ retrieving...\n"}, "finish_reason": None}],
            })

            t0 = time.perf_counter()
            try:
                history_text = ""
                try:

                    history_text = mem.render(max_turns=6, max_chars=1200)
                    if history_text:
                        logger.info("RAG_API:MEMORY FOR CONTEXT >>>\n%s", history_text)
                    else:
                        logger.info("RAG_API:MEMORY FOR CONTEXT >>> (empty)")
                except Exception as e:
                    logger.info(f"RAG_API:MEMORY FOR CONTEXT >>> {e}")
                    history_text = ""

                with rag_span(k):
                    text, sources = await asyncio.wait_for(
                        run_in_threadpool(answer, q, k, sid),  
                        timeout=answer_timeout_s
                    )
            except asyncio.TimeoutError:
                logger.error("answer() timed out after %.1fs", answer_timeout_s)
                yield _sse({
                    "id": f"chatcmpl-{created}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_id,
                    "choices": [{"index": 0, "delta": {"content": "\n[Upstream timed out]\n"}, "finish_reason": None}],
                })
                yield _sse({
                    "id": f"chatcmpl-{created}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_id,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                })
                yield "data: [DONE]\n\n"
                return

            if hasattr(mem, "put_assistant"):
                mem.put_assistant(text or "")
            elif hasattr(mem, "append_assistant"):
                mem.append_assistant(text or "")
            elif hasattr(mem, "put"):
                mem.put({"role": "assistant", "content": text or ""})

            srcs = _normalize_sources(sources)
            latency = round(time.perf_counter() - t0, 3)

            ragas_scores = {}
            try:
                contexts = [s.get("text", "") for s in srcs]
                ragas_scores = _maybe_ragas(q_for_answer, text or "", contexts) or {}
                if ENABLE_PHOENIX and log_query_trace:
                    log_query_trace(
                        query=q_for_answer, answer=text or "", contexts=contexts, sources=srcs,
                        metadata={"latency_s": latency, "request_id": rid},
                    )
            except Exception:
                pass

            try:
                _tx("chat_completion", rid, query=q_for_answer, k=k,
                    latency_s=latency, answer_len=len(text or ""),
                    sources=srcs, ragas=ragas_scores, enable_ragas=ENABLE_RAGAS,
                    crew=crew_meta)
            except Exception:
                pass

            yield _sse({
                "id": f"chatcmpl-{created}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_id,
                "choices": [{"index": 0, "delta": {"content": f"↻ generating (retrieval {latency:.1f}s)\n\n"}, "finish_reason": None}],
            })

            for tok in (text or "").split():
                yield _sse({
                    "id": f"chatcmpl-{created}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_id,
                    "choices": [{"index": 0, "delta": {"content": tok + " "}, "finish_reason": None}],
                })
                await asyncio.sleep(0.001)

            yield _sse({
                "id": f"chatcmpl-{created}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_id,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            })
            yield "data: [DONE]\n\n"

        return StreamingResponse(agen_realstream(), media_type="text/event-stream; charset=utf-8")


    t0 = time.perf_counter()

    try:
        history_text = ""
        try:
            history_text = mem.render(max_turns=6, max_chars=1200)
            if history_text:
                logger.info("RAG_API:MEMORY FOR CONTEXT >>>\n%s", history_text)
            else:
                logger.info("RAG_API:MEMORY FOR CONTEXT >>> (empty)")
        except Exception:
            logger.info("RAG_API:MEMORY FOR CONTEXT >>> (error while rendering)")
            history_text = ""
        with rag_span(k):
            text, sources = await asyncio.wait_for(
                run_in_threadpool(answer, q, k, sid), 
                timeout=answer_timeout_s
            )
    except asyncio.TimeoutError:
        logger.error("answer() timed out after %.1fs", answer_timeout_s)
        return JSONResponse(content={
            "id": f"chatcmpl-{created}",
            "object": "chat.completion",
            "created": created,
            "model": model_id,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "[Upstream timed out]"}, "finish_reason": "timeout"}],
            "references": [],
            "metrics": {"ragas": {}},
            "crew": crew_meta,
        }, status_code=504)


    if hasattr(mem, "put_assistant"):
        mem.put_assistant(text or "")
    elif hasattr(mem, "append_assistant"):
        mem.append_assistant(text or "")
    elif hasattr(mem, "put"):
        mem.put({"role": "assistant", "content": text or ""})

    srcs = _normalize_sources(sources)
    latency = round(time.perf_counter() - t0, 3)
    ragas_scores = {}
    try:
        contexts = [s.get("text", "") for s in srcs]
        ragas_scores = _maybe_ragas(q_for_answer, text or "", contexts) or {}
        if ENABLE_PHOENIX and log_query_trace:
            log_query_trace(
                query=q_for_answer, answer=text or "", contexts=contexts, sources=srcs,
                metadata={"latency_s": latency, "request_id": rid},
            )
    except Exception:
        pass

    try:
        _tx("chat_completion", rid, query=q_for_answer, k=k,
            latency_s=latency, answer_len=len(text or ""), sources=srcs,
            ragas=ragas_scores, enable_ragas=ENABLE_RAGAS, crew=crew_meta)
    except Exception:
        pass

    resp = {
        "id": f"chatcmpl-{created}",
        "object": "chat.completion",
        "created": created,
        "model": model_id,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text or ""},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "references": srcs,
        "metrics": {"ragas": ragas_scores},
        "crew": crew_meta,
    }
    return JSONResponse(content=_json_clean(resp))

@app.post("/debug/crew_refine")
async def debug_crew_refine(body: ChatCompletionRequest):
    original = body.messages[-1]["content"] if body.messages else ""
    refined = original
    used = False
    if body.use_crew:
        try:
            refined_try = _maybe_refine_query(original)
            used = (refined_try.strip() != "" and refined_try != original)
            refined = refined_try or original
        except Exception:
            pass
    return {"crew": {"enabled": bool(body.use_crew and ENABLE_CREW), "used": used, "original": original, "refined": refined}}

@app.post("/tools/upload-docs/files")
@app.post("/tools/upload-docs/files/")
async def upload_docs(files: Optional[List[UploadFile]] = File(None), file: Optional[UploadFile] = File(None),
                      also_session: bool = Form(True), request: Request = None):
    rid = getattr(request.state, "request_id", uuid.uuid4().hex) if request else uuid.uuid4().hex
    tmp_paths: List[Path] = []
    try:
        uploads: List[UploadFile] = []
        if files: uploads.extend(files)
        if file: uploads.append(file)
        if not uploads:
            raise HTTPException(status_code=400, detail="No files uploaded")

        to_upsert: List[Dict[str, Any]] = []
        for uf in uploads:
            suffix = Path(uf.filename).suffix or ".bin"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await uf.read())
                tmp_paths.append(Path(tmp.name))
            text = convert_file_to_text(tmp_paths[-1])
            to_upsert.append({"id": uf.filename, "text": text, "source": uf.filename})

        await run_in_threadpool(upsert_documents, to_upsert) 

        _tx("upload_files", rid, count=len(to_upsert))
        return {"ok": True, "count": len(to_upsert)}
    finally:
        for p in tmp_paths: p.unlink(missing_ok=True)

@app.get("/debug/index")
async def debug_index():
    return index_info()

@app.post("/api/v1/files")
@app.post("/api/v1/files/")
async def api_v1_files(file: UploadFile = File(...), also_session: bool = Form(True), request: Request = None):
    return await upload_docs(file=file, also_session=also_session, request=request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
