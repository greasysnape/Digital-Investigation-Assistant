"""
eval_batch_run.py
=================
Batch evaluation runner using the same ReACT loop as main.py.

Features:
  - Two-Pass ReACT: Phase 1 gathers info, Phase 2 synthesises answer
  - Reranker (cross-encoder) for improved retrieval quality
  - Empty responses are collected and retried at the end

Output:
  eval/results/ragas_eval_dataset.json  — RAGAS-ready dataset (incremental)
  eval/results/batch_run_summary.json   — per-question status + elapsed time
  history/eval_batch/<session>.json     — full session history

Usage:
    python eval/eval_batch_run.py                          # full curated testset
    python eval/eval_batch_run.py --limit 5                # first 5 (test)
    python eval/eval_batch_run.py --testset eval/results/ragas_testset.json
"""

import argparse
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TESTSET_PATH = "eval/results/ragas_testset.json"
DATASET_PATH = "eval/results/ragas_eval_dataset.json"
SUMMARY_PATH = "eval/results/batch_run_summary.json"
HISTORY_DIR  = "history/No_Case"

REACT_MODEL  = "qwen3.5:9b"
MAX_STEPS    = 5

# ── RAG (shared via rag_core) ──
from rag_core import (
    RAG_AVAILABLE as _RAG_OK,
    get_shared_dbs as _init_shared_dbs,
    search_tech_db as _search_tech,
    search_protocol_db as _search_proto,
)


def _load_resources():
    _init_shared_dbs()


def search_tech_db(query):
    return _search_tech(query)

def search_proto_db(query):
    return _search_proto(query)

# ── ReACT agent (shared via react_engine) ──
from react_engine import run_agent


def react_loop(question):
    """Run the LangGraph ReACT agent. Returns (final_answer, context_chunks)."""
    answer, elapsed, steps = run_agent(
        question=question,
        has_case=False,
        model=REACT_MODEL,
        max_steps=MAX_STEPS,
        verbose=True,
    )
    # Collect full context from tool observations for RAGAS
    contexts = []
    for s in steps:
        if s.get("node") == "observation" and s.get("content"):
            contexts.append(s["content"])
    return answer, contexts


# ─────────────────────────────────────────────────────────────
# Session History (compatible with main_react.py)
# ─────────────────────────────────────────────────────────────
def _create_session():
    """Create a new session dict."""
    return {
        "id":       str(uuid.uuid4()),
        "title":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "case":     None,
        "messages": [],
    }

def _save_session(session):
    os.makedirs(HISTORY_DIR, exist_ok=True)
    safe_title = re.sub(r'[\\/*?:"<>|]', "", session["title"])
    fp = os.path.join(HISTORY_DIR, f"{safe_title}_{session['id'][:8]}.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(session, f, ensure_ascii=False, indent=4)
    return fp


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _load_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError):
        return default

def _save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def _run_question(question, ground_truth, idx, total, session, eval_dataset, summary):
    """Run a single question. Returns True if successful (non-empty answer)."""
    print(f"\n[{idx}/{total}] {question[:80]}...")
    start = time.time()

    try:
        answer, contexts = react_loop(question)
        elapsed = round(time.time() - start, 1)

        # Check for empty, malformed, or no-RAG response
        stripped = (answer or "").strip()
        is_bad = (
            not stripped
            or len(stripped) < 10
            or stripped.startswith("Thought:")
            or "Action:" in stripped
            or "<tool_call>" in stripped
            or not contexts
        )
        if is_bad:
            if not stripped or len(stripped) < 10:
                reason = "empty or too short"
            elif not contexts:
                reason = "no RAG contexts retrieved"
            else:
                reason = "malformed (raw Thought/Action in response)"
            print(f"  [!] Bad response ({reason}, {len(stripped)} chars) — will retry later")
            return False

        print(f"  [{elapsed}s] {answer[:100].replace(chr(10), ' ')}...")

        # Save to session history
        session["messages"].append({"role": "user",      "content": question})
        session["messages"].append({"role": "assistant", "content": answer})
        _save_session(session)

        # Save to eval dataset
        eval_dataset.append({
            "user_input":         question,
            "reference":          ground_truth,
            "retrieved_contexts": contexts,
            "response":           answer,
            "elapsed_seconds":    elapsed,
        })
        summary.append({
            "idx":             idx,
            "question":        question,
            "status":          "ok",
            "elapsed_seconds": elapsed,
            "contexts_found":  len(contexts),
        })
        _save_json(DATASET_PATH, eval_dataset)
        _save_json(SUMMARY_PATH, summary)
        return True

    except Exception as e:
        elapsed = round(time.time() - start, 1)
        print(f"  [!] Error: {e}")
        summary.append({"idx": idx, "question": question, "status": f"error: {e}"})
        _save_json(SUMMARY_PATH, summary)
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", default=TESTSET_PATH)
    parser.add_argument("--limit",   type=int, default=0)
    parser.add_argument("--max-retries", type=int, default=2,
                        help="Max retry rounds for failed questions")
    args = parser.parse_args()

    if not _RAG_OK:
        print("[!] faiss / sentence_transformers not installed.")
        sys.exit(1)

    if not os.path.exists(args.testset):
        print(f"[!] Testset not found: {args.testset}")
        sys.exit(1)

    with open(args.testset, "r", encoding="utf-8") as f:
        testset = json.load(f)
    if args.limit:
        testset = testset[:args.limit]

    os.makedirs("eval/results", exist_ok=True)
    _load_resources()

    # Load existing progress
    eval_dataset = _load_json(DATASET_PATH, [])
    summary      = _load_json(SUMMARY_PATH, [])
    done_qs      = {e["user_input"] for e in eval_dataset}

    # Create session for history
    session = _create_session()

    total = len(testset)
    print("=" * 60)
    print(f"  DIA Batch Evaluation")
    print(f"  Testset  : {args.testset}  ({total} questions)")
    print(f"  Done     : {len(done_qs)} already completed")
    print(f"  Model    : {REACT_MODEL}")
    print(f"  Session  : {session['id'][:8]}")
    print("=" * 60)

    # ── First pass ──
    failed = []  # (index, item) tuples for retry

    for i, item in enumerate(testset):
        question     = item["question"]
        ground_truth = item.get("ground_truth", "")

        if question in done_qs:
            print(f"  [{i+1}/{total}] Already done.")
            continue

        ok = _run_question(question, ground_truth, i+1, total,
                           session, eval_dataset, summary)
        if not ok:
            failed.append((i+1, item))

    # ── Retry failed questions ──
    for retry_round in range(1, args.max_retries + 1):
        if not failed:
            break

        print(f"\n{'='*60}")
        print(f"  RETRY ROUND {retry_round}: {len(failed)} questions")
        print(f"{'='*60}")

        still_failed = []
        for idx, item in failed:
            question     = item["question"]
            ground_truth = item.get("ground_truth", "")

            # Skip if somehow completed in the meantime
            if question in {e["user_input"] for e in eval_dataset}:
                continue

            ok = _run_question(question, ground_truth, idx, total,
                               session, eval_dataset, summary)
            if not ok:
                still_failed.append((idx, item))

        failed = still_failed

    # ── Final summary ──
    _save_session(session)  # final save

    ok_count  = sum(1 for r in summary if r.get("status") == "ok")
    err_count = len(summary) - ok_count
    print(f"\n{'='*60}")
    print(f"  Done: {ok_count}/{len(summary)} succeeded, {err_count} errors")
    if failed:
        print(f"  Still failed after retries: {len(failed)} questions")
        for idx, item in failed:
            print(f"    [{idx}] {item['question'][:60]}...")
    print(f"  Dataset  -> {DATASET_PATH}")
    print(f"  History  -> {HISTORY_DIR}/")
    print(f"  Next: python eval/eval_ragas.py --eval-only")
    print("=" * 60)


if __name__ == "__main__":
    main()
