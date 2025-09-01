RAG Service – Retrieval Augmented Generation Pipeline

This project implements a Retrieval Augmented Generation (RAG) system powered by Ollama, LlamaIndex, Postgres/pgvector, and OpenTelemetry/Phoenix tracing.
It provides:

🔍 Hybrid Retrieval: Vector similarity (pgvector) + BM25 + Reciprocal Rank Fusion (RRF).

🤖 LLM Integration: Query answering using llama3.1:8b served via Ollama.

📑 Document Ingestion: Convert PDFs/Docs into structured chunks with Docling.

📊 Evaluation: RAGAS-inspired evaluation of faithfulness, correctness, and relevance.

📡 Tracing & Observability: Local JSONL logs + optional OpenTelemetry (OTLP).



📦 Installation
1. Clone repository
git clone https://github.com/yourusername/rag-service.git
cd rag-service

2. Create virtual environment
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows

3. Install dependencies
pip install -r requirements.txt

🔧 Configuration

All runtime settings are managed via .env.

Example .env
# --- General ---
LOG_DIR=logs
TX_LOG_PATH=logs/transactions.jsonl
OPENAI_COMPAT_MODEL_ID=rag-ollama

# --- Phoenix tracing ---
ENABLE_PHOENIX=true
PHOENIX_OTLP_ENDPOINT=http://localhost:6006/v1/traces
OTEL_SERVICE_NAME=rag-service
PHOENIX_TRACE_OUT=evaluation/phoenix_traces.jsonl
PHOENIX_USE_OTLP=true

# --- Ollama / LLM ---
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_BASE=http://127.0.0.1:11434
GEN_MODEL=llama3.1:8b
GEN_MAX_TOKENS=800
GEN_NUM_CTX=8192
GEN_TEMPERATURE=0.2
GEN_TIMEOUT_S=300.0
EMBED_MODEL=mxbai-embed-large:latest
EMBED_BATCH=64

# --- Retriever ---
RETRIEVER_SIM_TOP_K=32
RETRIEVER_FINAL_K=4

# --- Postgres (pgvector) ---
PG_HOST=localhost
PG_PORT=5432
PG_DB=ragdb
PG_USER=raguser
PG_PASS=ragpass
RAG_TABLE=rag_docs

🚀 Usage
1. Start Postgres with pgvector
docker run -d \
  --name rag-postgres \
  -e POSTGRES_USER=raguser \
  -e POSTGRES_PASSWORD=ragpass \
  -e POSTGRES_DB=ragdb \
  -p 5432:5432 \
  ankane/pgvector

2. Run Ollama server
ollama serve
ollama run llama3.1:8b
ollama run mxbai-embed-large:latest

3. Ingest documents
from data_pipeline.docling_loader import upsert_documents

docs = [
    {"id": "doc1", "text": "This is a test document", "source": "test.pdf"}
]
upsert_documents(docs)

4. Query pipeline
from api.pipeline import answer

resp, sources = answer("What is this document about?", top_k=4, session_id="demo")
print("Answer:", resp)
print("Sources:", sources)

📊 Evaluation

Run the evaluation script against your RAG API:

python evaluation/ragas.py


Output → evaluation/results.jsonl, e.g.:

{
  "question": "What are the key steps in onboarding?",
  "reference": "registration, KYC, approval, activation",
  "answer": "The onboarding involves registration, KYC, approval, and account activation.",
  "ragas": {"faithfulness": 0.92, "correctness": 0.95, "relevance": 0.88}
}

🔎 Observability (Phoenix)

Every query generates:

JSONL trace → evaluation/phoenix_traces.jsonl

Optional OpenTelemetry spans → exported to http://localhost:6006/v1/traces

Example record:

{
  "ts": 1725177291.2,
  "query": "What is his phone number?",
  "answer_len": 15,
  "contexts_len": 3,
  "sources_len": 3,
  "metadata": {}
}

🧩 Key Features

Docling-based ingestion → clean Markdown extraction from PDFs.

Chunking heuristics → splits by headers, bullets, contact info.

Hybrid retriever → vector + BM25 + Reciprocal Rank Fusion + CrossEncoder reranking.

Strict RAG prompting → prevents hallucination, grounds answers in context.

Evaluation suite → semantic similarity metrics for correctness, faithfulness, relevance.

Tracing hooks → JSONL logs + OTLP exporter for observability.

🛠️ Tech Stack

LLM: Ollama + llama3.1:8b

Embeddings: mxbai-embed-large:latest

Vector DB: Postgres + pgvector

Framework: LlamaIndex

Evaluation: SentenceTransformers (MiniLM-L6-v2) + ragas (optional)

Observability: OpenTelemetry + Phoenix + JSONL logging

📌 Future Improvements

Add more ragas metrics (e.g., answer relevancy, hallucination rate).

Add CI/CD integration for automated evaluations.

Support multi-modal ingestion (images, tables).

Deploy with Docker Compose (Postgres + Ollama + API).

