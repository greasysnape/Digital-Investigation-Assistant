"""
rag_core.py
===========
RAG core: embedding, FAISS search, reranking.
"""

import glob as _glob
import json
import os

import numpy as np

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
FAISS_PATH_TECH     = "./vector_db/Tech.faiss"
META_PATH_TECH      = "./vector_db/Tech_metadata.json"
FAISS_PATH_PROTOCOL = "./vector_db/Protocol.faiss"
META_PATH_PROTOCOL  = "./vector_db/Protocol_metadata.json"
CASES_DIR           = "./cases"
EMBEDDER_PATH       = "./models/all-mpnet-base-v2"
RERANKER_MODEL      = "./models/ms-marco-MiniLM-L-6-v2"

# ─────────────────────────────────────────────────────────────
# Singletons
# ─────────────────────────────────────────────────────────────
_embedder = None
_reranker = None
_tech_index = None;  _tech_docs = {}
_proto_index = None; _proto_docs = {}
_case_index = None;  _case_docs = {}
_active_case = None


def _load_faiss(index_path, meta_path):
    """Load a FAISS index + metadata JSON. Returns (index, doc_store)."""
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        return None, {}
    idx = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return idx, raw["metadata"]


def get_embedder():
    """Returns the cached SentenceTransformer embedder."""
    global _embedder
    if _embedder is None:
        if not RAG_AVAILABLE:
            return None
        print("  [rag_core] Loading embedding model...")
        _embedder = SentenceTransformer(EMBEDDER_PATH, device="cpu")
        print("  [rag_core] Embedding model loaded.")
    return _embedder


def get_reranker():
    """Returns the cached CrossEncoder reranker."""
    global _reranker
    if _reranker is None:
        if not RAG_AVAILABLE:
            return None
        try:
            import warnings
            import logging
            from sentence_transformers import CrossEncoder
            print("  [rag_core] Loading reranker...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                logging.disable(logging.WARNING)
                _reranker = CrossEncoder(
                    RERANKER_MODEL,
                    device="cpu",
                    automodel_args={"low_cpu_mem_usage": False}
                )
                logging.disable(logging.NOTSET)
            print("  [rag_core] Reranker loaded.")
        except Exception as e:
            print(f"  [rag_core] Reranker not available: {e}")
            return None
    return _reranker


def get_shared_dbs():
    """Load and cache the shared Tech + Protocol FAISS indexes."""
    global _tech_index, _tech_docs, _proto_index, _proto_docs
    if _tech_index is None:
        print("  [rag_core] Loading shared FAISS indexes...")
        _tech_index, _tech_docs = _load_faiss(FAISS_PATH_TECH, META_PATH_TECH)
        _proto_index, _proto_docs = _load_faiss(FAISS_PATH_PROTOCOL, META_PATH_PROTOCOL)
        t = _tech_index.ntotal if _tech_index else 0
        p = _proto_index.ntotal if _proto_index else 0
        print(f"  [rag_core] Tech: {t} vectors | Protocol: {p} vectors")
    return _proto_index, _proto_docs, _tech_index, _tech_docs


def load_case_db(case_name: str):
    """Load a case-specific FAISS index. Returns (index, doc_store)."""
    global _case_index, _case_docs, _active_case
    if _active_case == case_name and _case_index is not None:
        return _case_index, _case_docs

    case_db_dir = os.path.join(CASES_DIR, case_name, "case_db")
    if not os.path.isdir(case_db_dir):
        _case_index, _case_docs, _active_case = None, {}, None
        return None, {}

    faiss_files = _glob.glob(os.path.join(case_db_dir, "*.faiss"))
    json_files = _glob.glob(os.path.join(case_db_dir, "*.json"))
    if not faiss_files or not json_files:
        _case_index, _case_docs, _active_case = None, {}, None
        return None, {}

    _case_index, _case_docs = _load_faiss(faiss_files[0], json_files[0])
    _active_case = case_name
    n = _case_index.ntotal if _case_index else 0
    print(f"  [rag_core] Case DB loaded: {case_name} ({n} vectors)")
    return _case_index, _case_docs


# Alias for Streamlit cached wrapper compatibility
get_case_db = load_case_db


def unload_case_db():
    """Clear the active case DB."""
    global _case_index, _case_docs, _active_case
    _case_index, _case_docs, _active_case = None, {}, None


def get_active_case():
    """Returns the currently loaded case name."""
    return _active_case


def faiss_search(index, doc_store: dict, query: str,
                 k_retrieve: int = 20, k_rerank: int = 8) -> tuple:
    """
    FAISS retrieval → cross-encoder rerank → top results.

    Returns:
        (result_text, list_of_chunks)
    """
    embedder = get_embedder()
    if embedder is None or index is None:
        return "", []

    vec = embedder.encode([query]).astype("float32")
    vec /= np.linalg.norm(vec)

    actual_k = min(k_retrieve, index.ntotal)
    _, indices = index.search(vec, actual_k)

    candidates = []
    for i in indices[0]:
        if i != -1 and str(i) in doc_store:
            candidates.append(doc_store[str(i)]["document"])

    if not candidates:
        return "", []

    # Rerank
    reranker = get_reranker()
    if reranker is not None:
        pairs = [[query, doc] for doc in candidates]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        top_chunks = [doc for _, doc in ranked[:k_rerank]]
    else:
        top_chunks = candidates[:k_rerank]

    result_text = "".join(f"---\n{c}\n" for c in top_chunks)
    return result_text, top_chunks


# ─────────────────────────────────────────────────────────────
# Convenience search wrappers
# ─────────────────────────────────────────────────────────────
def search_tech_db(query: str) -> tuple:
    """Search the Tech DB. Returns (text, chunks)."""
    _, _, t_idx, t_docs = get_shared_dbs()
    if t_idx is None:
        return "Tech DB not loaded.", []
    return faiss_search(t_idx, t_docs, query)


def search_protocol_db(query: str) -> tuple:
    """Search the Protocol DB. Returns (text, chunks)."""
    p_idx, p_docs, _, _ = get_shared_dbs()
    if p_idx is None:
        return "Protocol DB not loaded.", []
    return faiss_search(p_idx, p_docs, query)


def search_case_db(query: str) -> tuple:
    """Search the active case DB. Returns (text, chunks)."""
    if _case_index is None:
        return "Case DB not loaded.", []
    return faiss_search(_case_index, _case_docs, query)
