"""
eval_ragas.py
=============
RAG Retrieval Quality Evaluation using RAGAS

Evaluates the DIA system's RAG pipeline quality using four RAGAS metrics:
  - Faithfulness      : response is grounded in retrieved contexts
  - Answer Relevance  : response actually addresses the question
  - Context Precision : retrieved chunks are relevant to the query
  - Context Recall    : all necessary info was retrieved

Workflow:
  1. Generate testset (questions + ground truth) from DB docs  →  ragas_testset.json
  2. Run eval_batch_run.py to generate DIA responses           →  ragas_eval_dataset.json
  3. Evaluate with RAGAS (this script, --eval-only)             →  ragas_results.json

Usage:
    # Generate testset only
    python eval/eval_ragas.py --gen-only

    # Evaluate pre-generated responses (from eval_batch_run.py)
    python eval/eval_ragas.py --eval-only
"""

import argparse
import json
import os
import sys
import random
from dotenv import load_dotenv
load_dotenv()

try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TESTSET_PATH      = "eval/results/ragas_testset.json"
RESULT_PATH       = "eval/results/ragas_results.json"
EVAL_DATASET_PATH = "eval/results/ragas_eval_dataset.json"

TECH_FAISS     = "./vector_db/Tech.faiss"
TECH_META      = "./vector_db/Tech_metadata.json"
PROTO_FAISS    = "./vector_db/Protocol.faiss"
PROTO_META     = "./vector_db/Protocol_metadata.json"


# ─────────────────────────────────────────────────────────────
# FAISS loader (shared via rag_core)
# ─────────────────────────────────────────────────────────────
from rag_core import (
    RAG_AVAILABLE as RAG_OK,
    _load_faiss as load_faiss,
)

def _sanitize(text: str) -> str:
    import re
    text = text.encode("utf-8", errors="ignore").decode("utf-8")
    text = text.encode("utf-16", "surrogatepass").decode("utf-16", "ignore")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    text = re.sub(r" {3,}", "  ", text)
    return text[:3000]

def metadata_to_langchain_docs(doc_store: dict):
    from langchain_core.documents import Document
    docs = []
    for k, v in doc_store.items():
        text = _sanitize(v.get("document", "").strip())
        if len(text) >= 50:
            # RAGAS TestsetGenerator requires 'filename' in metadata to group chunks
            docs.append(Document(page_content=text, metadata={"filename": k}))
    return docs


# ─────────────────────────────────────────────────────────────
# Phase 1 — Ground-truth generation
# ─────────────────────────────────────────────────────────────
def generate_testset(tech_docs: dict, proto_docs: dict,
                     size_per_db: int, api_key: str,
                     save_path: str = None) -> list:
    from ragas.testset import TestsetGenerator
    from ragas.run_config import RunConfig
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    from ragas.testset.synthesizers import (
        SingleHopSpecificQuerySynthesizer,
        MultiHopAbstractQuerySynthesizer,
        MultiHopSpecificQuerySynthesizer,
    )

    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", api_key=api_key,
                                         temperature=0))
    emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
              model="text-embedding-3-small", api_key=api_key))
    gen = TestsetGenerator(llm=llm, embedding_model=emb)

    # Distributions for different query synthesisers
    query_distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=llm), 0.5),
        (MultiHopAbstractQuerySynthesizer(llm=llm),  0.3),
        (MultiHopSpecificQuerySynthesizer(llm=llm),  0.2),
    ]

    # max_retries: JSON 파싱 실패 시 재시도 횟수
    run_config = RunConfig(max_retries=5, max_wait=90, timeout=120)

    testset = []
    for db_name, doc_store in [("Tech_DB", tech_docs), ("Protocol_DB", proto_docs)]:
        if not doc_store:
            continue
        print(f"\n  Generating {size_per_db} ground-truth pairs from {db_name}...")
        all_docs = metadata_to_langchain_docs(doc_store)
        sample_size = min(len(all_docs), 500)
        docs = random.sample(all_docs, sample_size)

        ts = gen.generate_with_langchain_docs(docs, testset_size=size_per_db,
                                              query_distribution=query_distribution,
                                              run_config=run_config)
        df = ts.to_pandas()
        for row in df.itertuples():
            synthesizer = getattr(row, "synthesizer_name", "unknown")
            testset.append({
                "db":             db_name,
                "question":       str(row.user_input),
                "ground_truth":   str(row.reference),
                "question_type":  synthesizer,
            })
            print(f"  ✔ [{synthesizer}] {row.user_input[:55]}...")

        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(testset, f, ensure_ascii=False, indent=2)
            print(f"  💾 중간 저장 ({len(testset)}개) → {save_path}")

    return testset


# ─────────────────────────────────────────────────────────────
# Phase 2 — RAGAS evaluation
# ─────────────────────────────────────────────────────────────
def run_ragas(eval_dataset: list, api_key: str) -> tuple:
    from ragas import evaluate
    from ragas.metrics import (
        Faithfulness,
        ResponseRelevancy,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from datasets import Dataset

    class _FixedTempChatOpenAI(ChatOpenAI):
        """gpt-5-mini only supports temperature=1; intercept payload to force it."""
        def _get_request_payload(self, input_: any, *, stop=None, **kwargs) -> dict:
            payload = super()._get_request_payload(input_, stop=stop, **kwargs)
            payload["temperature"] = 1
            return payload

    judge_llm = LangchainLLMWrapper(_FixedTempChatOpenAI(model="gpt-5.4-nano-2026-03-17", api_key=api_key, temperature=1))
    judge_emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(api_key=api_key))

    # Strip non-RAGAS fields (e.g. elapsed_seconds) and filter incomplete entries
    ragas_keys = {"user_input", "reference", "retrieved_contexts", "response"}
    data = {"user_input": [], "reference": [], "retrieved_contexts": [], "response": []}

    skipped_entries = []
    for entry in eval_dataset:
        if not entry.get("retrieved_contexts") or not entry.get("response"):
            skipped_entries.append(entry)
            continue
        for k in ragas_keys:
            data[k].append(entry.get(k, ""))

    if skipped_entries:
        print(f"  ⚠️  Skipped {len(skipped_entries)} entries with missing contexts or response.")

    if not data["user_input"]:
        print("❌ No usable entries in eval dataset.")
        return {}

    import math

    # Always run all 4 metrics — ground_truth needed for precision & recall
    has_gt = any(r for r in data["reference"])
    if not has_gt:
        print("  ⚠️  No ground_truth found — context_precision & context_recall will be NaN.")

    metrics = [
        Faithfulness(llm=judge_llm),
        ResponseRelevancy(llm=judge_llm, embeddings=judge_emb),
        LLMContextPrecisionWithReference(llm=judge_llm),
        LLMContextRecall(llm=judge_llm),
    ]

    METRIC_KEYS = [
        ("faithfulness",                        "Faithfulness"),
        ("response_relevancy",                  "Answer Relevance"),
        ("llm_context_precision_with_reference","Context Precision"),
        ("llm_context_recall",                  "Context Recall"),
    ]

    print(f"  Evaluating {len(data['user_input'])} entries...")

    dataset = Dataset.from_dict(data)
    result  = evaluate(dataset=dataset, metrics=metrics)

    # RAGAS renames columns across versions — list fallbacks for each metric
    METRIC_KEYS = [
        ("faithfulness",                        "Faithfulness",     ["faithfulness"]),
        ("response_relevancy",                  "Answer Relevance", ["response_relevancy", "answer_relevancy"]),
        ("llm_context_precision_with_reference","Context Precision",["llm_context_precision_with_reference", "context_precision"]),
        ("llm_context_recall",                  "Context Recall",   ["llm_context_recall", "context_recall"]),
    ]

    scores = {}
    df = None
    try:
        df = result.to_pandas()
        print(f"  Available columns: {list(df.columns)}")
        for canonical, label, aliases in METRIC_KEYS:
            col = next((a for a in aliases if a in df.columns), None)
            if col:
                val = df[col].mean()
                if math.isnan(val):
                    scores[canonical] = None
                    print(f"  ⚠️  {label} ({col}): NaN")
                else:
                    scores[canonical] = round(float(val), 4)
            else:
                scores[canonical] = None
                print(f"  ⚠️  {label}: column not found (tried {aliases})")
    except Exception as e:
        print(f"  ⚠️  to_pandas() failed ({e}), falling back to direct access")
        df = None
        for key, label, _ in METRIC_KEYS:
            try:
                val = float(result[key])
                scores[key] = None if math.isnan(val) else round(val, 4)
            except Exception:
                scores[key] = None

    # Save per-question scores to CSV
    if df is not None:
        detail_path = "eval/results/ragas_per_question.csv"
        try:
            df.to_csv(detail_path, index=False, encoding="utf-8-sig")
            print(f"  ✅ Per-question scores saved → {detail_path}")
        except Exception as e:
            print(f"  ⚠️  Failed to save per-question CSV: {e}")

    return scores, df, skipped_entries


# ─────────────────────────────────────────────────────────────
# Breakdown helpers
# ─────────────────────────────────────────────────────────────
METRIC_KEYS_FLAT = [
    ("faithfulness",                         "Faithfulness",      ["faithfulness"]),
    ("response_relevancy",                   "Answer Relevance",  ["response_relevancy", "answer_relevancy"]),
    ("llm_context_precision_with_reference", "Context Precision", ["llm_context_precision_with_reference", "context_precision"]),
    ("llm_context_recall",                   "Context Recall",    ["llm_context_recall", "context_recall"]),
]

def _simplify_type(qt: str) -> str:
    q = qt.lower()
    if "single" in q:
        return "Easy"
    elif "abstract" in q:
        return "Intermediate"
    elif "multi" in q and "specific" in q:
        return "Difficult"
    return qt

def _print_breakdown(df, eval_dataset: list, testset: list) -> dict:
    import math

    # Resolve actual column names in df
    metric_cols = {}
    for canonical, _, aliases in METRIC_KEYS_FLAT:
        col = next((a for a in aliases if a in df.columns), None)
        if col:
            metric_cols[canonical] = col

    if not metric_cols:
        print("  ⚠️  No metric columns found for breakdown.")
        return {}

    ts_lookup = {t["question"]: t for t in testset}
    el_lookup = {e["user_input"]: e.get("elapsed_seconds", None) for e in eval_dataset}

    df = df.copy()
    df["db"]            = df["user_input"].map(lambda q: ts_lookup.get(q, {}).get("db", "unknown"))
    df["question_type"] = df["user_input"].map(lambda q: _simplify_type(ts_lookup.get(q, {}).get("question_type", "unknown")))
    df["elapsed"]       = df["user_input"].map(lambda q: el_lookup.get(q, None))

    def _fmt(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "  N/A "
        return f"{v:.4f}"

    def _group_stats(group_df) -> dict:
        scores = {}
        for canonical, _, _ in METRIC_KEYS_FLAT:
            col = metric_cols.get(canonical)
            val = float(group_df[col].mean()) if col else float("nan")
            scores[canonical] = None if math.isnan(val) else round(val, 4)
        elapsed_vals = group_df["elapsed"].dropna()
        return {
            "n": len(group_df),
            "elapsed_avg": round(float(elapsed_vals.mean()), 1) if len(elapsed_vals) else None,
            "scores": scores,
        }

    def _print_group(group_df, label):
        stats = _group_stats(group_df)
        parts = []
        for canonical, disp_label, _ in METRIC_KEYS_FLAT:
            val = stats["scores"].get(canonical)
            parts.append(f"{disp_label}: {_fmt(val)}")
        elapsed_str = f"{stats['elapsed_avg']:.1f}s (avg)" if stats["elapsed_avg"] is not None else "N/A"
        print(f"  {label:<35} n={stats['n']:>3}  |  {' | '.join(parts)}  |  Time: {elapsed_str}")
        return stats

    breakdown = {"by_db": {}, "by_difficulty": {}, "by_db_x_difficulty": {}}

    print("\n" + "=" * 60)
    print("  BREAKDOWN BY DB")
    print("=" * 60)
    for db_val, grp in df.groupby("db"):
        breakdown["by_db"][db_val] = _print_group(grp, db_val)

    print("\n" + "=" * 60)
    print("  BREAKDOWN BY DIFFICULTY")
    print("=" * 60)
    for qt_val, grp in df.groupby("question_type"):
        breakdown["by_difficulty"][qt_val] = _print_group(grp, qt_val)

    print("\n" + "=" * 60)
    print("  BREAKDOWN BY DB × DIFFICULTY")
    print("=" * 60)
    for (db_val, qt_val), grp in df.groupby(["db", "question_type"]):
        key = f"{db_val} / {qt_val}"
        breakdown["by_db_x_difficulty"][key] = _print_group(grp, key)

    return breakdown


# ─────────────────────────────────────────────────────────────
# Token usage helpers
# ─────────────────────────────────────────────────────────────
from contextlib import contextmanager

@contextmanager
def _null_ctx():
    """Fallback context manager when get_openai_callback is unavailable."""
    yield None

def _print_usage(cb) -> None:
    print("\n" + "=" * 60)
    print("  API USAGE")
    print("=" * 60)
    if cb is None:
        print("  (langchain_community not installed — usage tracking unavailable)")
        return
    print(f"  Prompt tokens    : {cb.prompt_tokens:>10,}")
    print(f"  Completion tokens: {cb.completion_tokens:>10,}")
    print(f"  Total tokens     : {cb.total_tokens:>10,}")
    print(f"  Total cost (USD) : ${cb.total_cost:>10.4f}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-gen",  action="store_true",
                        help="Skip ground-truth generation; load from saved testset.")
    parser.add_argument("--gen-only",  action="store_true",
                        help="Generate testset only and save to file, skip evaluation.")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip generation; load ragas_eval_dataset.json and evaluate.")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not set.")
        sys.exit(1)

    if not RAG_OK:
        sys.exit(1)

    os.makedirs("eval/results", exist_ok=True)

    print("=" * 60)
    print("  RAGAS RAG Quality Evaluation")
    print("=" * 60)

    try:
        from langchain_community.callbacks import get_openai_callback
        _has_cb = True
    except ImportError:
        _has_cb = False

    with (get_openai_callback() if _has_cb else _null_ctx()) as cb:

        # ── eval-only: load pre-built dataset from eval_batch_run.py ──
        if args.eval_only:
            if not os.path.exists(EVAL_DATASET_PATH):
                print(f"❌ Eval dataset not found: {EVAL_DATASET_PATH}")
                print("   Run eval_batch_run.py first.")
                sys.exit(1)
            with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
                eval_dataset = json.load(f)

            testset = []
            if os.path.exists(TESTSET_PATH):
                with open(TESTSET_PATH, "r", encoding="utf-8") as f:
                    testset = json.load(f)

            print(f"\n⏭️  Loaded eval dataset: {len(eval_dataset)} entries")
            print("\n── RAGAS Evaluation ─────────────────────────────────")
            scores, df, skipped = run_ragas(eval_dataset, api_key)

            # ── RAG Coverage ──
            total_q = len(eval_dataset) + len(skipped)
            rag_q = len(eval_dataset)
            non_rag_q = len(skipped)

            # Also count entries in eval_dataset that have empty contexts
            no_ctx_in_dataset = [e for e in eval_dataset if not e.get("retrieved_contexts")]
            rag_q -= len(no_ctx_in_dataset)
            non_rag_q += len(no_ctx_in_dataset)
            all_skipped = skipped + no_ctx_in_dataset

            # Build testset lookup for db/difficulty info
            gt_lookup = {}
            if testset:
                for t in testset:
                    gt_lookup[t.get("question", "")] = {
                        "db": t.get("db", "Unknown"),
                        "difficulty": t.get("question_type", "Unknown"),
                    }

            print("\n" + "=" * 60)
            print("  RAG COVERAGE")
            print("=" * 60)
            print(f"  Total questions     : {total_q}")
            print(f"  RAG-answered        : {rag_q} ({rag_q/total_q*100:.1f}%)")
            print(f"  Non-RAG (skipped)   : {non_rag_q} ({non_rag_q/total_q*100:.1f}%)")

            if all_skipped:
                print(f"\n  Non-RAG questions:")
                for entry in all_skipped:
                    q = entry.get("user_input", "")
                    info = gt_lookup.get(q, {"db": "Unknown", "difficulty": "Unknown"})
                    print(f"    - [{info['db']} / {info['difficulty']}] {q[:70]}...")

            print("\n" + "=" * 60)
            print("  OVERALL RESULTS")
            print("=" * 60)
            for canonical, label, _ in METRIC_KEYS_FLAT:
                val = scores.get(canonical)
                val_str = f"{val:.4f}" if val is not None else "N/A"
                print(f"  {label:<30}: {val_str}")

            breakdown = {}
            if df is not None and testset:
                breakdown = _print_breakdown(df, eval_dataset, testset)
            elif df is not None:
                print("  ⚠️  Testset not found — skipping breakdown.")

            output = {
                "eval_dataset_entries": len(eval_dataset),
                "rag_coverage": {
                    "total": total_q,
                    "rag_answered": rag_q,
                    "non_rag": non_rag_q,
                    "non_rag_questions": [
                        {
                            "question": e.get("user_input", ""),
                            "db": gt_lookup.get(e.get("user_input", ""), {}).get("db", "Unknown"),
                            "difficulty": gt_lookup.get(e.get("user_input", ""), {}).get("difficulty", "Unknown"),
                        }
                        for e in all_skipped
                    ],
                },
                "scores":               scores,
                "breakdown":            breakdown,
            }
            with open(RESULT_PATH, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)

            print(f"\n  ✅ Saved → {RESULT_PATH}")
            _print_usage(cb if _has_cb else None)
            return

        # ── Phase 1: Ground-truth generation ──────────────────────────
        _, tech_docs  = load_faiss(TECH_FAISS,  TECH_META)
        _, proto_docs = load_faiss(PROTO_FAISS, PROTO_META)

        if args.skip_gen and os.path.exists(TESTSET_PATH):
            print(f"\n⏭️  Loading saved testset: {TESTSET_PATH}")
            with open(TESTSET_PATH, "r", encoding="utf-8") as f:
                testset = json.load(f)
            print(f"   {len(testset)} ground-truth pairs loaded.")
        else:
            print("\n── Phase 1: Ground-Truth Generation ────────────────")
            testset = generate_testset(tech_docs, proto_docs,
                                       size_per_db=150, api_key=api_key,
                                       save_path=TESTSET_PATH)
            with open(TESTSET_PATH, "w", encoding="utf-8") as f:
                json.dump(testset, f, ensure_ascii=False, indent=2)
            csv_path = TESTSET_PATH.replace(".json", ".csv")
            import csv
            with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["db", "question_type", "question", "ground_truth"])
                writer.writeheader()
                writer.writerows(testset)
            print(f"\n  ✅ Testset saved ({len(testset)} pairs) → {TESTSET_PATH}")
            print(f"  ✅ CSV saved → {csv_path}")

        if args.gen_only:
            print("\n  --gen-only Option: Skipping evaluation phase.")
            _print_usage(cb if _has_cb else None)
            return

        # ── Direct evaluation requires --eval-only with pre-generated dataset ──
        print("\n❌ Direct evaluation is no longer supported.")
        print("   Please run eval_batch_run.py first, then use --eval-only.")
        print("   Example:")
        print("     python eval/eval_batch_run.py")
        print("     python eval/eval_ragas.py --eval-only")
        _print_usage(cb if _has_cb else None)


if __name__ == "__main__":
    main()
