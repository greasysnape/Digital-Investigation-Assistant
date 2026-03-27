"""
tools.py
========
Streamlit-aware wrappers around rag_core.
Provides cached resource loading and LangGraph-compatible tool functions.
"""

import streamlit as st

from rag_core import (
    RAG_AVAILABLE as RAG_ENABLED,
    get_embedder as _core_get_embedder,
    get_reranker as _core_get_reranker,
    get_shared_dbs as _core_get_shared_dbs,
    get_case_db as _core_get_case_db,
    load_case_db as _core_load_case_db,
    unload_case_db as _core_unload_case_db,
    get_active_case,
    faiss_search,
    search_tech_db as _core_search_tech,
    search_protocol_db as _core_search_protocol,
    search_case_db as _core_search_case,
)

try:
    from eval.eval_logger import log_rag_retrieval as _log_rag
except Exception:
    _log_rag = None

# Module-level state
ACTIVE_SESSION_ID = ""


# ─────────────────────────────────────────────────────────────
# Cached wrappers for Streamlit (show_spinner on first load)
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Initialising Embedding Model...")
def get_embedder():
    return _core_get_embedder()


@st.cache_resource(show_spinner=False)
def get_reranker():
    return _core_get_reranker()


@st.cache_resource(show_spinner="Loading Shared Databases...")
def get_shared_dbs():
    return _core_get_shared_dbs()


@st.cache_resource(show_spinner=False)
def get_case_db(case_name: str):
    return _core_get_case_db(case_name)


# ─────────────────────────────────────────────────────────────
# State management
# ─────────────────────────────────────────────────────────────
def load_case_db(case_name: str) -> bool:
    _core_load_case_db(case_name)
    idx, _ = _core_get_case_db(case_name) if case_name else (None, {})
    return idx is not None


def unload_case_db():
    _core_unload_case_db()


def set_active_session(session_id: str):
    global ACTIVE_SESSION_ID
    ACTIVE_SESSION_ID = session_id


# ─────────────────────────────────────────────────────────────
# Search tools (LangGraph-compatible: return str)
# ─────────────────────────────────────────────────────────────
def search_protocol_db(query: str) -> str:
    """
    Searches the FAISS Protocol Database.
    Useful for compliance and regulation enquiries.
    """
    if not RAG_ENABLED:
        return "System Notice: RAG libraries are missing."
    # Ensure shared DBs are loaded
    get_shared_dbs()
    result_text, chunks = _core_search_protocol(query)
    if _log_rag and chunks:
        try:
            _log_rag(agent="Protocol_Retriever", query=query,
                     contexts=chunks, session_id=ACTIVE_SESSION_ID)
        except Exception:
            pass
    return result_text if result_text else "No relevant protocols found."


def search_tech_db(query: str) -> str:
    """
    Searches the FAISS Technical Database for tool usage and manuals.
    Useful for 'how-to' enquiries and tool recommendations.
    """
    if not RAG_ENABLED:
        return "System Notice: RAG libraries are missing."
    get_shared_dbs()
    result_text, chunks = _core_search_tech(query)
    if _log_rag and chunks:
        try:
            _log_rag(agent="Tech_Retriever", query=query,
                     contexts=chunks, session_id=ACTIVE_SESSION_ID)
        except Exception:
            pass
    return result_text if result_text else "No technical documentation found."


def search_case_db(query: str) -> str:
    """
    Searches the case-specific FAISS database for evidence and documents
    related to the currently loaded investigation case.
    """
    if not RAG_ENABLED:
        return "System Notice: RAG libraries are missing."
    result_text, chunks = _core_search_case(query)
    if _log_rag and chunks:
        try:
            _log_rag(agent="Case_Retriever", query=query,
                     contexts=chunks, session_id=ACTIVE_SESSION_ID)
        except Exception:
            pass
    return result_text if result_text else "No relevant case documents found."
