# evaluation/phoenix_tracing.py
import os
import json
import time
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


ENABLE_PHOENIX       = os.getenv("ENABLE_PHOENIX", "true").lower() == "true"
PHOENIX_USE_OTLP     = os.getenv("PHOENIX_USE_OTLP", "false").lower() == "true"
PHOENIX_OTLP_ENDPOINT= os.getenv("PHOENIX_OTLP_ENDPOINT", "http://localhost:6006/v1/traces")
SERVICE_NAME         = os.getenv("OTEL_SERVICE_NAME", "rag-service")
TRACE_OUT            = os.getenv("PHOENIX_TRACE_OUT", "evaluation/phoenix_traces.jsonl")

PHOENIX_SAMPLER      = os.getenv("PHOENIX_SAMPLER", "always_on").lower()
PHOENIX_SAMPLE_RATIO = float(os.getenv("PHOENIX_SAMPLE_RATIO", "1.0"))


trace = None
TracerProvider = None
Resource = None
BatchSpanProcessor = None
OTLPSpanExporter = None
Sampler = None
AlwaysOnSampler = None
AlwaysOffSampler = None
TraceIdRatioBased = None

try:
    from opentelemetry import trace as _trace
    from opentelemetry.sdk.trace import TracerProvider as _TracerProvider
    from opentelemetry.sdk.resources import Resource as _Resource
    from opentelemetry.sdk.trace.export import BatchSpanProcessor as _BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as _OTLPSpanExporter
    from opentelemetry.sdk.trace.sampling import (
        Sampler as _Sampler,
        ALWAYS_ON as _AlwaysOnSampler,
        ALWAYS_OFF as _AlwaysOffSampler,
        TraceIdRatioBased as _TraceIdRatioBased,
    )
    trace = _trace
    TracerProvider = _TracerProvider
    Resource = _Resource
    BatchSpanProcessor = _BatchSpanProcessor
    OTLPSpanExporter = _OTLPSpanExporter
    Sampler = _Sampler
    AlwaysOnSampler = _AlwaysOnSampler
    AlwaysOffSampler = _AlwaysOffSampler
    TraceIdRatioBased = _TraceIdRatioBased
except Exception:
    pass

try:
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
except Exception:
    LlamaIndexInstrumentor = None

try:
    from openinference.instrumentation.requests import RequestsInstrumentor
except Exception:
    RequestsInstrumentor = None

try:
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
except Exception:
    HTTPXClientInstrumentor = None

_INITIALIZED = False
_PROVIDER = None 


def _build_sampler():
    if Sampler is None:
        return None
    if PHOENIX_SAMPLER == "always_off":
        return AlwaysOffSampler
    if PHOENIX_SAMPLER == "traceidratio":

        r = max(0.0, min(1.0, PHOENIX_SAMPLE_RATIO))
        return TraceIdRatioBased(r)

    return AlwaysOnSampler


def init_phoenix():
    """Initialize OTel + (optionally) OTLP exporter. Safe to call multiple times."""
    global _INITIALIZED, _PROVIDER
    if _INITIALIZED:
        return
    if not ENABLE_PHOENIX:
        logger.info("[phoenix] disabled via env ENABLE_PHOENIX=false")
        return
    if trace is None:
        logger.warning("[phoenix] opentelemetry not installed; skipping OTel setup")
        return

    try:
        sampler = _build_sampler()
        if sampler is None:
            _PROVIDER = TracerProvider(resource=Resource.create({"service.name": SERVICE_NAME}))
        else:
            _PROVIDER = TracerProvider(resource=Resource.create({"service.name": SERVICE_NAME}), sampler=sampler)

        if PHOENIX_USE_OTLP:
            try:
                exporter = OTLPSpanExporter(endpoint=PHOENIX_OTLP_ENDPOINT)
                _PROVIDER.add_span_processor(BatchSpanProcessor(exporter))
                logger.info(f"[phoenix] OTLP exporter attached -> {PHOENIX_OTLP_ENDPOINT}")
            except Exception as e:
                logger.warning(f"[phoenix] failed to attach OTLP exporter: {e}")
        else:
            logger.info("[phoenix] OTLP exporter disabled (PHOENIX_USE_OTLP=false); JSONL only")

        trace.set_tracer_provider(_PROVIDER)

        # Instrumentations
        if RequestsInstrumentor:
            RequestsInstrumentor().instrument()
        if LlamaIndexInstrumentor:
            LlamaIndexInstrumentor().instrument()
        if HTTPXClientInstrumentor:
            try:
                HTTPXClientInstrumentor().instrument()
            except Exception as e:
                logger.info(f"[phoenix] httpx instrumentation not available: {e}")

        logger.info("[phoenix] OpenTelemetry + OpenInference instrumented (service=%s)", SERVICE_NAME)
        _INITIALIZED = True
    except Exception as e:
        logger.warning(f"[phoenix] init failed: {e}")


def shutdown_phoenix():

    try:
        if _PROVIDER and hasattr(_PROVIDER, "shutdown"):
            _PROVIDER.shutdown()
    except Exception:
        pass


def instrument_fastapi_app(app):
    return app


@contextmanager
def span(name: str, **attrs):
    if trace is None:
        yield
        return
    tracer = trace.get_tracer(SERVICE_NAME)
    with tracer.start_as_current_span(name) as s:
        for k, v in (attrs or {}).items():
            try:
                s.set_attribute(f"rag.{k}", v)
            except Exception:
                pass
        yield


def log_query_trace(query: str, answer: str, *, contexts=None, sources=None, metadata=None, started_at=None):

    rec = {
        "ts": time.time(),
        "query": query,
        "answer_len": len(answer or ""),
        "contexts_len": len(contexts or []),
        "sources_len": len(sources or []),
        "metadata": metadata or {},
    }
    try:
        os.makedirs(os.path.dirname(TRACE_OUT) or ".", exist_ok=True)
        with open(TRACE_OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info(f"[phoenix] wrote trace JSONL -> {TRACE_OUT}")
    except Exception as e:
        logger.warning(f"[phoenix] failed to write JSONL trace: {e}")

    if trace is not None:
        try:
            tracer = trace.get_tracer(SERVICE_NAME)
            with tracer.start_as_current_span("rag.query_log") as s:
                s.set_attribute("rag.answer.len", len(answer or ""))
                s.set_attribute("rag.context.count", len(contexts or []))
                s.set_attribute("rag.sources.count", len(sources or []))
                # mirror metadata (flat)
                for k, v in (metadata or {}).items():
                    try:
                        s.set_attribute(f"rag.meta.{k}", v)
                    except Exception:
                        pass
        except Exception:
            pass
