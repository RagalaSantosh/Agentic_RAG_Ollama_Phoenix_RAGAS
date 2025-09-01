
import logging
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import NodeWithScore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import LongContextReorder, SimilarityPostprocessor
from sentence_transformers import CrossEncoder

from data_pipeline.docling_loader import get_vector_store, LLM

load_dotenv(r"C:/Users/user/Projects/NorthBay8/.env")
logger = logging.getLogger(__name__)

SIMILARITY_TOP_K  = 10
FINAL_TOP_K       = 4
SIMILARITY_CUTOFF = 0.0
USE_LONG_REORDER  = True

RRF_K             = 60

RERANK_MODEL      = "cross-encoder/ms-marco-MiniLM-L-2-v2"
cross_encoder = CrossEncoder(RERANK_MODEL)

INDEX: Optional[VectorStoreIndex] = None 


def set_index(idx: VectorStoreIndex):
    global INDEX
    INDEX = idx
    try:
        logger.info("[retriever] Global INDEX set: %s", type(INDEX).__name__)
    except Exception:
        pass

def _lazy_init_index() -> VectorStoreIndex:
    global INDEX
    if INDEX is None:
        vstore = get_vector_store()
        storage = StorageContext.from_defaults(vector_store=vstore)
        INDEX = VectorStoreIndex.from_vector_store(vstore, storage_context=storage, llm=LLM)
        logger.info("[retriever] Lazy-initialized INDEX from PGVector")
    return INDEX

def get_index() -> VectorStoreIndex:
    return _lazy_init_index()

def index_info() -> Dict[str, Any]:
    return {
        "ready": INDEX is not None,
        "type": type(INDEX).__name__ if INDEX is not None else None,
    }


def _node_id(n: NodeWithScore) -> str:
    meta = getattr(n, "metadata", None) or {}
    return getattr(n, "node_id", None) or meta.get("id") or meta.get("doc_id") or str(id(n))

def _dedup(nodes: List[NodeWithScore]) -> List[NodeWithScore]:
    seen, out = set(), []
    for n in nodes:
        nid = _node_id(n)
        if nid not in seen:
            seen.add(nid)
            out.append(n)
    return out

def _rerank(nodes: List[NodeWithScore], query: str) -> List[NodeWithScore]:
    if len(nodes) <= 1:
        return nodes
    try:
        pairs = [(query, n.get_content()) for n in nodes]
        scores = cross_encoder.predict(pairs)
        for n, s in zip(nodes, scores):
            n.score = float(s)
        nodes = sorted(nodes, key=lambda x: x.score, reverse=True)
    except Exception as e:
        logger.warning("Rerank failed: %s", e)
    return nodes[:FINAL_TOP_K]

def _postprocess(nodes: List[NodeWithScore], query: str) -> List[NodeWithScore]:
    if SIMILARITY_CUTOFF > 0:
        nodes = SimilarityPostprocessor(similarity_cutoff=SIMILARITY_CUTOFF).postprocess_nodes(nodes)
    nodes = _dedup(nodes)
    nodes = _rerank(nodes, query)
    if USE_LONG_REORDER and len(nodes) > 1:
        nodes = LongContextReorder().postprocess_nodes(nodes)
    return nodes[:FINAL_TOP_K]


def _try_bm25(index: VectorStoreIndex):
    try:
        return index.as_retriever(retriever_mode="bm25", similarity_top_k=SIMILARITY_TOP_K)
    except Exception as e:
        logger.debug("BM25 unavailable: %s (vector-only fallback)", e)
        return None

def _rrf_fuse(*ranked_lists: List[NodeWithScore]) -> List[NodeWithScore]:

    scores: Dict[str, float] = {}
    pick: Dict[str, NodeWithScore] = {}
    for lst in ranked_lists:
        for rank, node in enumerate(lst, start=1):
            nid = _node_id(node)
            scores[nid] = scores.get(nid, 0.0) + 1.0 / (RRF_K + rank)
            if nid not in pick:
                pick[nid] = node
    fused = list(pick.values())
    fused.sort(key=lambda n: scores.get(_node_id(n), 0.0), reverse=True)
    return fused

def _fusion(index: VectorStoreIndex, query: str) -> List[NodeWithScore]:
    logger.info("[retriever] vector.retrieve start")
    vec = index.as_retriever(similarity_top_k=SIMILARITY_TOP_K)
    vec_nodes = vec.retrieve(query)
    logger.info("[retriever] vector.retrieve done -> %d nodes", len(vec_nodes))

    bm25_ret = _try_bm25(index)
    if bm25_ret is None:
        return _postprocess(vec_nodes, query)

    try:
        logger.info("[retriever] bm25.retrieve start")
        bm25_nodes = bm25_ret.retrieve(query) 
        logger.info("[retriever] bm25.retrieve done -> %d nodes", len(bm25_nodes))
        fused = _rrf_fuse(bm25_nodes, vec_nodes)
        return _postprocess(fused, query)
    except Exception as e:
        logger.warning("Fusion failed (%s). Using vector-only results.", e)
        return _postprocess(vec_nodes, query)


def vector_retrieve(query: str, top_k: int) -> List[NodeWithScore]:

    global FINAL_TOP_K
    prev_final = FINAL_TOP_K
    try:
        FINAL_TOP_K = max(1, top_k or 1)
        index = get_index()            
        nodes = _fusion(index, query)
        return nodes[:FINAL_TOP_K]
    finally:
        FINAL_TOP_K = prev_final
