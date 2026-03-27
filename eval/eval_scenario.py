"""
eval_scenario.py
================
Scenario-based evaluation: LLM-as-a-Judge

For each scenario:
  1. Run the query through DIA (ReACT loop with FAISS) to get a response
  2. Optionally generate a commercial LLM baseline response
  3. LLM-as-a-Judge: structured 5-criteria scoring

Usage:
    # Need OpenAI API key in .env for commercial LLM calls and judge evaluation
    
    cd /path/to/DIA

    # Run all (DIA + commercial + both evaluations)
    python eval/eval_scenario.py

    # Evaluate only (skip DIA/commercial generation, use cached responses)
    python eval/eval_scenario.py --eval-only

    # Skip commercial LLM baseline
    python eval/eval_scenario.py --no-commercial
"""

import argparse
import json
import os
import re
import sys
import time

import requests
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
JUDGE_MODEL       = "o3"
COMMERCIAL_MODELS = ["gpt-4o-mini"]
RESULT_DIR        = "eval/results"
RESPONSE_DIR      = "eval/scenarios/responses"
VANILLA_DIR       = "eval/scenarios/vanilla"
RAGONLY_DIR       = "eval/scenarios/ragonly"
COMMERCIAL_DIR    = "eval/scenarios/commercial"

# Global API usage tracker (total + per-model)
_api_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost_usd": 0.0}
_api_usage_by_model = {}  # model_name -> {"prompt_tokens", "completion_tokens", "total_tokens", "cost_usd"}

OLLAMA_URL   = "http://localhost:11434/api/chat"
ReACT_MODEL  = "qwen3.5:9b"
MAX_STEPS    = 3


# ─────────────────────────────────────────────────────────────
# Scenarios
# ─────────────────────────────────────────────────────────────
CASES_DIR = "./cases"

SCENARIOS = [
    {
        "id": "S01",
        "case_name": "#001_Scenario",
        "title": "Insider Trade Secret Leakage Investigation",
        "query": (
            "An employee at a semiconductor company is suspected of copying proprietary circuit "
            "designs onto a USB drive and emailing them to a personal account before resigning. "
            "We have seized the suspect's company-issued laptop and a USB flash drive.\n\n"
            "Please help me with the following:\n"
            "1. How can I trace which USB devices were connected to the suspect's Windows PC, "
            "and where are these connection records stored in the system?\n"
            "2. What Windows artefacts can tell me which files the suspect opened, when, and "
            "from which location?\n"
            "3. According to the case evidence, what specific USB device was connected to the "
            "suspect's laptop, when was it first connected, and what files were accessed from it?\n"
            "4. For the evidence to hold up under the Unfair Competition Prevention and Trade Secret "
            "Protection Act, what legal criteria must the leaked information satisfy to qualify "
            "as a trade secret?\n"
            "5. Based on the case timeline, reconstruct the sequence of the suspect's data exfiltration "
            "activities — from accessing the confidential server to emailing the files out."
        ),
        "ground_truth_points": [
            "Extraction of USB connection history from Windows Registry (USBSTOR) and setupapi.dev.log",
            "Verification of file access history through Prefetch, LNK files, and Jump Lists",
            "Case-specific: USB device E-02 (SanDisk Ultra, Serial SD-128-A992) was connected to E-01 on 2026-02-18 at 14:28:12 KST, and AEGIS_Circuit_Diagram_V3.gds was accessed from it",
            "Criteria for determining whether the three requirements of trade secrets (non-public nature, economic utility, confidentiality management) are satisfied",
            "Case-specific: Timeline reconstruction showing server access (Feb 16 22:14), USB mount (Feb 18 14:28), file copy (14:35), Gmail login (15:02), email attachment (15:08), USB removal (16:45)",
        ],
    },
    {
        "id": "S02",
        "case_name": "#002_Scenario",
        "title": "Tracking Dark Web Child Sexual Exploitation Material Distributors",
        "query": (
            "An international joint investigation has identified a suspect who used a Korean VPN "
            "service to access a dark web CSAM distribution site via Tor. We have seized the "
            "suspect's laptop and an external SSD.\n\n"
            "I need guidance on the investigative approach:\n"
            "1. What forensic traces does the Tor Browser leave on a Windows system, and where "
            "should I look for them?\n"
            "2. The suspect used a domestic VPN provider called SafeWay VPN. What is the legal "
            "procedure under Korean law to compel the provider to hand over connection logs?\n"
            "3. According to the seized evidence, what did the forensic examiner find on the "
            "external SSD, and what multi-step technique was used to uncover hidden content?\n"
            "4. Are there any special legal provisions in Korean law regarding digital evidence "
            "in child sexual exploitation cases?\n"
            "5. The original intelligence came from Germany's BKA via an international cooperation "
            "request. Based on the case documents, describe how this international cooperation "
            "was initiated and how it led to the domestic investigation."
        ),
        "ground_truth_points": [
            "Tor Browser artefact locations (profiles.ini, cert9.db, places.sqlite, browser cache) and Prefetch evidence of Tor execution",
            "Procedure for requesting communication confirmation data to obtain VPN logs under Article 13 of the Protection of Communications Secrets Act",
            "Case-specific: E-02 contained a VeraCrypt hidden container (project_archive.dat, 400GB) decrypted using a passphrase recovered from RAM, revealing 1,422 CSAM files in /ShadowNet/Uploads/Completed/",
            "Special provisions regarding digital evidence under the Act on the Protection of Children and Juveniles against Sexual Abuse (Article 11)",
            "Case-specific: BKA issued MLAT request on 2024-02-10, Seoul Central District Court issued warrant for SafeWay VPN logs (2024-SW-9981), logs revealed suspect IP 211.xxx.xxx.45 matching upload timestamps",
        ],
    },
    {
        "id": "S03",
        "case_name": "#003_Scenario",
        "title": "Compulsory Collection of Cloud-Stored Evidence",
        "query": (
            "A former CFO is suspected of embezzling company funds and storing evidence of the "
            "transactions on Google Drive and KakaoTalk's cloud storage. The suspect refuses to "
            "provide access credentials. We have seized a Windows 11 laptop.\n\n"
            "I need to recover cloud-stored evidence and build the case:\n"
            "1. Is there any locally cached Google Drive data on the seized laptop, and where "
            "would I find it?\n"
            "2. The suspect communicated via KakaoTalk on PC. How can I locate and examine the "
            "local message database?\n"
            "3. According to the case evidence, what specific actions did the suspect take to "
            "conceal or destroy evidence before the seizure, and what forensic artefacts revealed this?\n"
            "4. What is the legal basis under Korean criminal procedure for compelling cloud "
            "service providers to produce data when the suspect refuses to cooperate?\n"
            "5. Based on the case documents, what communications between the suspect and potential "
            "accomplices were found, and how do they indicate coordinated evidence destruction?"
        ),
        "ground_truth_points": [
            "Google Drive for Desktop local cache path and artefacts (LNK files confirming access to specific financial documents like 2025_Q4_Overseas_Transfer_List.xlsx)",
            "KakaoTalk PC message database location (AppData\\Local\\Kakao\\KakaoTalk\\users\\[User_Hash]\\db\\KakaoTalk.db) and SQLite chat_logs table analysis",
            "Case-specific: Suspect connected E-02 at 23:45 on March 9, ran Eraser.exe at 23:50 to shred local files, then synced deletions to Google Drive at 00:15 on March 10 — hours before the seizure",
            "Legal basis for specifying cloud accounts in warrants and ordering service providers to submit data (Article 106(3) of the Criminal Procedure Act)",
            "Case-specific: KakaoTalk messages on March 2 — suspect asked 'Did you clear the drive?' with reply 'Working on it. Moving files to Talk Drawer now.' indicating coordinated evidence destruction",
        ],
    },
]

# ─────────────────────────────────────────────────────────────
# RAG (shared with main app via rag_core)
# ─────────────────────────────────────────────────────────────
from rag_core import (
    get_shared_dbs as _init_shared_dbs,
    load_case_db,
    search_tech_db as _search_tech,
    search_protocol_db as _search_proto,
    search_case_db as _search_case,
)


def _load_resources():
    """Load embedding model and shared FAISS indexes via rag_core."""
    _init_shared_dbs()


def search_tech_db(query):
    return _search_tech(query)

def search_proto_db(query):
    return _search_proto(query)

def search_case_db(query):
    return _search_case(query)

TOOLS = {
    "search_case_db":     search_case_db,
    "search_tech_db":     search_tech_db,
    "search_protocol_db": search_proto_db,
}


# ─────────────────────────────────────────────────────────────
# ReACT agent (shared via react_engine)
# ─────────────────────────────────────────────────────────────
from react_engine import run_agent


def ReACT_loop(question):
    """Run the LangGraph ReACT agent. Returns (final_answer, elapsed_seconds)."""
    answer, elapsed, steps = run_agent(
        question=question,
        has_case=True,
        model=ReACT_MODEL,
        max_steps=MAX_STEPS,
        verbose=True,
    )
    return answer, elapsed


# ─────────────────────────────────────────────────────────────
# Commercial LLM baseline
# ─────────────────────────────────────────────────────────────
# Pricing per 1M tokens (USD) — update as needed
MODEL_PRICING = {
    "o3":          {"input": 2.00,  "output": 8.00},
    "gpt-4o-mini":       {"input": 0.15,  "output": 0.60},
}


def call_commercial_llm(query, model, api_key):
    """Returns (response_text, elapsed_seconds, cost_usd, usage_dict)."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an expert digital forensics investigator. Provide a thorough and accurate response to the user's query."},
            {"role": "user",   "content": query},
        ],
    }
    # Some models only support default temperature — skip for those
    _no_temp = ("o3")
    if model not in _no_temp:
        payload["temperature"] = 0.3
    start = time.time()
    try:
        r = requests.post("https://api.openai.com/v1/chat/completions",
                          json=payload, headers=headers, timeout=180)
        r.raise_for_status()
        elapsed = round(time.time() - start, 1)
        data = r.json()
        text  = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage", {})

        # Calculate cost
        pricing = MODEL_PRICING.get(model, {"input": 0, "output": 0})
        in_tok  = usage.get("prompt_tokens", 0)
        out_tok = usage.get("completion_tokens", 0)
        cost = (in_tok * pricing["input"] + out_tok * pricing["output"]) / 1_000_000

        # Accumulate global usage
        _api_usage["prompt_tokens"]     += in_tok
        _api_usage["completion_tokens"] += out_tok
        _api_usage["total_tokens"]      += in_tok + out_tok
        _api_usage["cost_usd"]          += cost

        # Accumulate per-model usage
        if model not in _api_usage_by_model:
            _api_usage_by_model[model] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost_usd": 0.0}
        _api_usage_by_model[model]["prompt_tokens"]     += in_tok
        _api_usage_by_model[model]["completion_tokens"] += out_tok
        _api_usage_by_model[model]["total_tokens"]      += in_tok + out_tok
        _api_usage_by_model[model]["cost_usd"]          += cost

        return text, elapsed, round(cost, 6), {
            "prompt_tokens":     in_tok,
            "completion_tokens": out_tok,
            "total_tokens":      in_tok + out_tok,
        }
    except Exception as e:
        elapsed = round(time.time() - start, 1)
        return f"ERROR: {e}", elapsed, 0.0, {}


# ─────────────────────────────────────────────────────────────
# Vanilla LLM baseline (same model, no RAG/ReACT)
# ─────────────────────────────────────────────────────────────
def call_vanilla_llm(query):
    """Call the same Ollama model without RAG or ReACT. Returns (text, elapsed)."""
    payload = {
        "model": ReACT_MODEL,
        "messages": [
            {"role": "system", "content": "You are an expert digital forensics investigator. Provide a thorough and accurate response to the user's query."},
            {"role": "user",   "content": query},
        ],
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 4096},
    }
    start = time.time()
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=300)
        r.raise_for_status()
        text = r.json().get("message", {}).get("content", "")
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        elapsed = round(time.time() - start, 1)
        return text, elapsed
    except Exception as e:
        elapsed = round(time.time() - start, 1)
        return f"ERROR: {e}", elapsed


# ─────────────────────────────────────────────────────────────
# RAG-only baseline (same model + RAG, no ReACT)
# ─────────────────────────────────────────────────────────────
def call_rag_only(query):
    """Search all 3 DBs, stuff context into prompt, single Ollama call. Returns (text, elapsed)."""
    start = time.time()

    tech_result, _  = search_tech_db(query)
    proto_result, _ = search_proto_db(query)
    case_result, _  = search_case_db(query)

    context = (
        "=== Technical Forensics Knowledge ===\n" + tech_result[:2500] + "\n\n"
        "=== Legal / Procedural Knowledge ===\n" + proto_result[:2500] + "\n\n"
        "=== Case-Specific Evidence ===\n" + case_result[:2500]
    )

    payload = {
        "model": ReACT_MODEL,
        "messages": [
            {"role": "system", "content": (
                "You are an expert digital forensics investigator. Use the provided context to answer the question thoroughly and accurately."
            )},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"},
        ],
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 16384},
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=300)
        r.raise_for_status()
        text = r.json().get("message", {}).get("content", "")
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        elapsed = round(time.time() - start, 1)
        return text, elapsed
    except Exception as e:
        elapsed = round(time.time() - start, 1)
        return f"ERROR: {e}", elapsed


# ─────────────────────────────────────────────────────────────
# LLM-as-a-Judge evaluation
# ─────────────────────────────────────────────────────────────
JUDGE_SYSTEM = """You are an expert evaluator for digital forensics investigation reports.

You will be presented with a forensic investigation scenario and a response generated by an AI system.
The scenario contains two types of questions:
- **Type A (General Knowledge)**: Questions about general forensic techniques, tools, or legal procedures.
- **Type B (Case-Specific)**: Questions that require access to specific case evidence (seized device details, forensic findings, timelines, communications). Only a system with access to the case database can answer these with concrete details.

Evaluate the response on five criteria, scoring 1-5 each:

1. factual_accuracy     - Are all technical and legal claims factually correct? Are forensic artefact names, file paths, legal article numbers, and tool references accurate? Penalise fabricated or incorrect details.
2. case_specificity     - For Type B questions: does the response cite concrete case data (device serial numbers, exact timestamps, file names, suspect actions) from the investigation? Score 1 if only generic advice is given for case-specific questions. For Type A questions: score 3 (neutral) as case data is not expected.
3. completeness         - Does the response address all 5 sub-questions in the query? Each unanswered or superficially answered sub-question reduces the score.
4. practical_usefulness - Could a real investigator act on this response? Are the procedures, commands, file paths, or legal steps specific enough to follow without additional research?
5. clarity              - Is the response logically structured, clearly written, and easy to follow? Is the information presented in a coherent order without unnecessary repetition?

Scoring Guide:
  5 = Excellent - Fully correct, cites specific case data where required, addresses all sub-questions.
  4 = Good - Mostly correct with minor gaps; case data is present but incomplete.
  3 = Adequate - Partially addresses the query; case-specific questions answered generically.
  2 = Weak - Significant errors or omissions; case-specific questions largely unanswered.
  1 = Poor - Incorrect, irrelevant, or only provides vague generalities.

Ground Truth Coverage:
For each ground truth point, determine whether the response adequately addresses it (true/false).
- Points prefixed with "Case-specific:" REQUIRE concrete case data (e.g., "E-02, SanDisk Ultra, Serial SD-128-A992, connected at 14:28:12 KST"). A generic explanation of methodology without actual case data = false.

**CRITICAL INSTRUCTIONS:**
- Do NOT let response length influence scores. Penalise padding.
- A response giving only generic forensic advice for a case-specific question MUST score low on case_specificity (1-2) and completeness.
- If the response fabricates case details that do not match the ground truth, penalise factual_accuracy severely.

Reply ONLY with a JSON object (no markdown, no explanation):
{
  "factual_accuracy": <1-5>,
  "case_specificity": <1-5>,
  "completeness": <1-5>,
  "practical_usefulness": <1-5>,
  "clarity": <1-5>,
  "gt_coverage": [true/false, true/false, true/false, true/false, true/false],
  "rationale": "<2-3 sentences: main strengths/weaknesses, whether case-specific questions were answered with real case data or generic advice.>"
}"""


def call_judge(scenario, response_text, api_key):
    gt_list = "\n".join(f"  {i+1}. {g}" for i, g in enumerate(scenario["ground_truth_points"]))
    user_msg = (
        f"Scenario: {scenario['title']}\n\n"
        f"Query: {scenario['query']}\n\n"
        f"Ground Truth Points (check coverage for each):\n{gt_list}\n\n"
        f"Response to evaluate:\n{response_text}"
    )
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
    }
    _no_temp = ("o3")
    if JUDGE_MODEL not in _no_temp:
        payload["temperature"] = 0.0
    try:
        r = requests.post("https://api.openai.com/v1/chat/completions",
                          json=payload, headers=headers, timeout=90)
        r.raise_for_status()
        data = r.json()
        content = data["choices"][0]["message"]["content"].strip()
        content = re.sub(r"^```json\s*", "", content)
        content = re.sub(r"\s*```$",    "", content)

        # Accumulate global usage
        usage = data.get("usage", {})
        in_tok  = usage.get("prompt_tokens", 0)
        out_tok = usage.get("completion_tokens", 0)
        pricing = MODEL_PRICING.get(JUDGE_MODEL, {"input": 0, "output": 0})
        cost = (in_tok * pricing["input"] + out_tok * pricing["output"]) / 1_000_000
        _api_usage["prompt_tokens"]     += in_tok
        _api_usage["completion_tokens"] += out_tok
        _api_usage["total_tokens"]      += in_tok + out_tok
        _api_usage["cost_usd"]          += cost

        # Per-model (judge)
        judge_key = f"judge ({JUDGE_MODEL})"
        if judge_key not in _api_usage_by_model:
            _api_usage_by_model[judge_key] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost_usd": 0.0}
        _api_usage_by_model[judge_key]["prompt_tokens"]     += in_tok
        _api_usage_by_model[judge_key]["completion_tokens"] += out_tok
        _api_usage_by_model[judge_key]["total_tokens"]      += in_tok + out_tok
        _api_usage_by_model[judge_key]["cost_usd"]          += cost

        return json.loads(content)
    except Exception as e:
        print(f"    [!] Judge error: {e}")
        return {}


# ─────────────────────────────────────────────────────────────
# Response I/O
# ─────────────────────────────────────────────────────────────
def save_response(sid, directory, text, meta=None):
    """Save response text + optional metadata (elapsed, cost, usage)."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{sid}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    if meta:
        meta_path = os.path.join(directory, f"{sid}_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    return path


def load_response(sid, directory):
    """Returns (text, meta_dict) or (None, None)."""
    # Try .md first, fallback to .txt for backward compatibility
    path = os.path.join(directory, f"{sid}.md")
    if not os.path.exists(path):
        path = os.path.join(directory, f"{sid}.txt")
    if not os.path.exists(path):
        return None, None
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    meta = None
    meta_path = os.path.join(directory, f"{sid}_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    return text, meta


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
CRITERIA = ["factual_accuracy", "case_specificity", "completeness",
            "practical_usefulness", "clarity"]


def avg_scores(scores_list):
    if not scores_list:
        return {}
    return {k: round(sum(s.get(k, 0) for s in scores_list) / len(scores_list), 2)
            for k in CRITERIA}


def print_judge_table(dia_scores_list, van_scores_list, rag_scores_list, comm_scores_map):
    """Print comparison table: DIA vs Vanilla vs RAG-only vs Commercial."""
    dia_avg = avg_scores(dia_scores_list)
    van_avg = avg_scores(van_scores_list)
    rag_avg = avg_scores(rag_scores_list)

    header = f"  {'Criterion':<25}  {'DIA':>6}  {'Vanilla':>8}  {'RAG-only':>9}"
    for m in COMMERCIAL_MODELS:
        header += f"  {m[:12]:>12}"
    print(header)
    print(f"  {'─'*25}" + "  ──────" + "  ────────" + "  ─────────" + "  ──────────" * len(COMMERCIAL_MODELS))

    comm_avgs = {m: avg_scores(v) for m, v in comm_scores_map.items()}

    for c in CRITERIA:
        row = f"  {c:<25}  {dia_avg.get(c, 'N/A'):>6}  {van_avg.get(c, 'N/A'):>8}  {rag_avg.get(c, 'N/A'):>9}"
        for m in COMMERCIAL_MODELS:
            v = comm_avgs.get(m, {}).get(c, "N/A")
            row += f"  {v:>12}"
        print(row)

    def _overall(avg_dict):
        if not avg_dict:
            return 0
        return round(sum(avg_dict.get(c, 0) for c in CRITERIA) / len(CRITERIA), 2)

    dia_overall = _overall(dia_avg)
    print(f"\n  {'Overall (DIA)':<25}  {dia_overall:>6}")
    if van_avg:
        print(f"  {'Overall (Vanilla)':<25}  {_overall(van_avg):>6}")
    if rag_avg:
        print(f"  {'Overall (RAG-only)':<25}  {_overall(rag_avg):>6}")
    for m in COMMERCIAL_MODELS:
        co = _overall(comm_avgs.get(m, {}))
        print(f"  {'Overall (' + m + ')':<25}  {co:>6}")

    return dia_avg, van_avg, rag_avg, comm_avgs


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only",     action="store_true",
                        help="Skip response generation; use cached responses only.")
    parser.add_argument("--no-vanilla",    action="store_true",
                        help="Skip vanilla LLM (same model, no RAG) baseline.")
    parser.add_argument("--no-ragonly",   action="store_true",
                        help="Skip RAG-only LLM (same model + RAG, no ReACT) baseline.")
    parser.add_argument("--no-commercial", action="store_true",
                        help="Skip commercial LLM baseline generation.")
    parser.add_argument("--no-judge",      action="store_true",
                        help="Skip LLM-as-a-Judge evaluation.")
    parser.add_argument("--scenarios",     type=str, default=None,
                        help="Comma-separated scenario IDs to run (e.g., S01 or S01,S02). Default: all.")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        if not args.no_judge:
            print("[!] OPENAI_API_KEY not set. LLM-as-a-Judge will be skipped.")
            args.no_judge = True
        if not args.no_commercial:
            print("[!] OPENAI_API_KEY not set. Commercial LLM baseline will be skipped.")
            args.no_commercial = True

    os.makedirs(RESULT_DIR,      exist_ok=True)
    os.makedirs(RESPONSE_DIR,   exist_ok=True)
    os.makedirs(VANILLA_DIR,    exist_ok=True)
    os.makedirs(RAGONLY_DIR,    exist_ok=True)
    os.makedirs(COMMERCIAL_DIR, exist_ok=True)

    print("=" * 65)
    print("  Scenario Evaluation: LLM-as-a-Judge")
    print(f"  DIA Model       : {ReACT_MODEL}")
    print(f"  Judge Model     : {JUDGE_MODEL}")
    print(f"  Scenarios       : {len(SCENARIOS)}")
    print(f"  LLM-as-a-Judge  : {'ON' if not args.no_judge else 'OFF'}")
    print("=" * 65)

    # ── Load FAISS if we need to generate DIA responses ──
    if not args.eval_only:
        from rag_core import RAG_AVAILABLE
        if not RAG_AVAILABLE:
            print("[!] faiss/sentence_transformers not installed. Cannot generate DIA responses.")
            print("    Use --eval-only with pre-cached responses.")
            sys.exit(1)
        _load_resources()

    results = {}
    dia_judge_list  = []
    van_judge_list  = []
    rag_judge_list  = []
    comm_judge_map  = {m: [] for m in COMMERCIAL_MODELS}

    # Filter scenarios if --scenarios is specified
    selected_ids = None
    if args.scenarios:
        selected_ids = set(s.strip() for s in args.scenarios.split(","))

    for scenario in SCENARIOS:
        sid   = scenario["id"]
        title = scenario["title"]
        if selected_ids and sid not in selected_ids:
            continue
        print(f"\n{'─'*65}")
        print(f"  {sid}: {title}")
        print(f"{'─'*65}")

        # ── Load Case DB ───────────────
        case_name = scenario.get("case_name")
        if case_name and not args.eval_only:
            load_case_db(case_name)

        # ── DIA response ──────────────────────────────────────
        dia_response, dia_meta = load_response(sid, RESPONSE_DIR)
        dia_elapsed = None

        if dia_response and args.eval_only:
            dia_elapsed = dia_meta.get("elapsed") if dia_meta else None
            print(f"  DIA response loaded from cache ({len(dia_response)} chars, {dia_elapsed}s)")
        elif dia_response and not args.eval_only:
            dia_elapsed = dia_meta.get("elapsed") if dia_meta else None
            print(f"  DIA response found in cache ({len(dia_response)} chars, {dia_elapsed}s) — using cached")
        else:
            if args.eval_only:
                print(f"  [!] No cached DIA response for {sid}. Run without --eval-only first.")
                results[sid] = {"title": title, "dia": None, "commercial": {}}
                continue
            print(f" 🔷 Running DIA ReACT loop...")
            dia_response, dia_elapsed = ReACT_loop(scenario["query"])
            save_response(sid, RESPONSE_DIR, dia_response,
                          meta={"elapsed": dia_elapsed, "cost_usd": 0.0})
            print(f" 🔶 DIA response generated ({len(dia_response)} chars, {dia_elapsed}s)\n")

        # ── Commercial LLM responses ──────────────────────────
        comm_responses = {}   # model -> text
        comm_elapsed   = {}   # model -> seconds
        comm_cost      = {}   # model -> USD
        comm_usage     = {}   # model -> token dict
        if not args.no_commercial:
            for model in COMMERCIAL_MODELS:
                safe = model.replace("/", "_").replace(":", "_")
                cached, cached_meta = load_response(sid, os.path.join(COMMERCIAL_DIR, safe))
                if cached:
                    comm_responses[model] = cached
                    if cached_meta:
                        comm_elapsed[model] = cached_meta.get("elapsed")
                        comm_cost[model]    = cached_meta.get("cost_usd", 0.0)
                        comm_usage[model]   = cached_meta.get("usage", {})
                    print(f"  {model} response loaded from cache ({len(cached)} chars, {comm_elapsed.get(model)}s, ${comm_cost.get(model, 0):.4f})")
                elif not args.eval_only:
                    print(f" 🔷 Generating {model} response...")
                    resp, elapsed, cost, usage = call_commercial_llm(scenario["query"], model, api_key)
                    save_response(sid, os.path.join(COMMERCIAL_DIR, safe), resp,
                                  meta={"elapsed": elapsed, "cost_usd": cost, "usage": usage})
                    comm_responses[model] = resp
                    comm_elapsed[model]   = elapsed
                    comm_cost[model]      = cost
                    comm_usage[model]     = usage
                    print(f" 🔶 {model} response generated ({len(resp)} chars, {elapsed}s, ${cost:.4f})\n")
                    time.sleep(1)

        # ── Vanilla LLM response (same model, no RAG) ─────────
        van_response = None
        van_elapsed  = None
        if not args.no_vanilla:
            van_response, van_meta = load_response(sid, VANILLA_DIR)
            if van_response:
                van_elapsed = van_meta.get("elapsed") if van_meta else None
                print(f"  Vanilla response loaded from cache ({len(van_response)} chars, {van_elapsed}s)")
            elif not args.eval_only:
                print(f" 🔷 Running Vanilla {ReACT_MODEL} (no RAG)...")
                van_response, van_elapsed = call_vanilla_llm(scenario["query"])
                save_response(sid, VANILLA_DIR, van_response,
                              meta={"elapsed": van_elapsed, "cost_usd": 0.0})
                print(f" 🔶 Vanilla response generated ({len(van_response)} chars, {van_elapsed}s)\n")

        # ── RAG-only response (same model + RAG, no ReACT) ────
        rag_response = None
        rag_elapsed  = None
        if not args.no_ragonly:
            rag_response, rag_meta = load_response(sid, RAGONLY_DIR)
            if rag_response:
                rag_elapsed = rag_meta.get("elapsed") if rag_meta else None
                print(f"  RAG-only response loaded from cache ({len(rag_response)} chars, {rag_elapsed}s)")
            elif not args.eval_only:
                print(f" 🔷 Running RAG-only {ReACT_MODEL} (RAG, no ReACT)...")
                rag_response, rag_elapsed = call_rag_only(scenario["query"])
                save_response(sid, RAGONLY_DIR, rag_response,
                              meta={"elapsed": rag_elapsed, "cost_usd": 0.0})
                print(f" 🔶 RAG-only response generated ({len(rag_response)} chars, {rag_elapsed}s)\n")

        # ── LLM-as-a-Judge ────────────────────────────────────
        judge_dia  = {}
        judge_van  = {}
        judge_rag  = {}
        judge_comm = {}

        if not args.no_judge:
            print(f"  Judging DIA response...")
            judge_dia = call_judge(scenario, dia_response, api_key)
            if judge_dia:
                scores_only = {k: v for k, v in judge_dia.items() if k in CRITERIA}
                gt_cov = judge_dia.get("gt_coverage", [])
                covered = sum(1 for x in gt_cov if x)
                print(f"  Judge DIA: {scores_only}")
                print(f"  GT coverage: {covered}/{len(gt_cov)}")
                print(f"  Rationale: {judge_dia.get('rationale', '')}")
                dia_judge_list.append(judge_dia)

            if van_response:
                print(f"  Judging Vanilla response...")
                judge_van = call_judge(scenario, van_response, api_key)
                if judge_van:
                    scores_only = {k: v for k, v in judge_van.items() if k in CRITERIA}
                    print(f"  Judge Vanilla: {scores_only}")
                    van_judge_list.append(judge_van)

            if rag_response:
                print(f"  Judging RAG-only response...")
                judge_rag = call_judge(scenario, rag_response, api_key)
                if judge_rag:
                    scores_only = {k: v for k, v in judge_rag.items() if k in CRITERIA}
                    print(f"  Judge RAG-only: {scores_only}")
                    rag_judge_list.append(judge_rag)

            for model, resp in comm_responses.items():
                print(f"  Judging {model} response...")
                js = call_judge(scenario, resp, api_key)
                judge_comm[model] = js
                if js:
                    scores_only = {k: v for k, v in js.items() if k in CRITERIA}
                    print(f"  Judge {model}: {scores_only}")
                    comm_judge_map[model].append(js)
                time.sleep(1)

        # ── Store results ─────────────────────────────────────
        results[sid] = {
            "title":    title,
            "query":    scenario["query"],
            "ground_truth_points": scenario["ground_truth_points"],
            "dia": {
                "response":    dia_response,
                "elapsed":     dia_elapsed,
                "cost_usd":    0.0,
                "judge":       judge_dia,
            },
            "vanilla": {
                "response":  van_response,
                "elapsed":   van_elapsed,
                "cost_usd":  0.0,
                "judge":     judge_van,
            } if van_response else None,
            "ragonly": {
                "response":  rag_response,
                "elapsed":   rag_elapsed,
                "cost_usd":  0.0,
                "judge":     judge_rag,
            } if rag_response else None,
            "commercial": {
                model: {
                    "response":  comm_responses.get(model),
                    "elapsed":   comm_elapsed.get(model),
                    "cost_usd":  comm_cost.get(model, 0.0),
                    "usage":     comm_usage.get(model),
                    "judge":     judge_comm.get(model),
                }
                for model in COMMERCIAL_MODELS
                if model in comm_responses
            },
        }

    # ── Aggregate Summary ─────────────────────────────────────
    print("\n" + "=" * 65)
    print("  AGGREGATE SUMMARY")
    print("=" * 65)

    # Latency & Cost summary
    print("\n  --- Latency & Cost ---")
    dia_times = [r["dia"]["elapsed"] for r in results.values()
                 if isinstance(r.get("dia"), dict) and r["dia"].get("elapsed") is not None]
    if dia_times:
        print(f"  {'DIA':<20} avg: {sum(dia_times)/len(dia_times):>7.1f}s  |  cost: $0.0000 (local)")

    van_times = [r["vanilla"]["elapsed"] for r in results.values()
                 if r.get("vanilla") and r["vanilla"].get("elapsed") is not None]
    if van_times:
        print(f"  {'Vanilla':<20} avg: {sum(van_times)/len(van_times):>7.1f}s  |  cost: $0.0000 (local)")

    rag_times = [r["ragonly"]["elapsed"] for r in results.values()
                 if r.get("ragonly") and r["ragonly"].get("elapsed") is not None]
    if rag_times:
        print(f"  {'RAG-only':<20} avg: {sum(rag_times)/len(rag_times):>7.1f}s  |  cost: $0.0000 (local)")

    for model in COMMERCIAL_MODELS:
        ctimes = [r["commercial"][model]["elapsed"] for r in results.values()
                  if model in r.get("commercial", {}) and r["commercial"][model].get("elapsed") is not None]
        ccosts = [r["commercial"][model]["cost_usd"] for r in results.values()
                  if model in r.get("commercial", {}) and r["commercial"][model].get("cost_usd") is not None]
        if ctimes:
            avg_t = sum(ctimes) / len(ctimes)
            tot_c = sum(ccosts)
            print(f"  {model:<20} avg: {avg_t:>7.1f}s  |  cost: ${tot_c:.4f} (total for {len(ccosts)} scenarios)")

    # LLM-as-a-Judge summary
    if not args.no_judge:
        print("\n  --- LLM-as-a-Judge ---")
        print_judge_table(dia_judge_list, van_judge_list, rag_judge_list, comm_judge_map)

        # GT coverage summary
        print("\n  --- Ground Truth Coverage (Judge) ---")
        print(f"  {'Scenario':<8}  {'DIA':>8}  {'Vanilla':>8}  {'RAG-only':>9}", end="")
        for m in COMMERCIAL_MODELS:
            print(f"  {m[:12]:>12}", end="")
        print()
        for sid, res in results.items():
            if not isinstance(res.get("dia"), dict):
                continue
            def _gt_str(source):
                if not source:
                    return "N/A"
                cov = source.get("judge", {}).get("gt_coverage", [])
                return f"{sum(1 for x in cov if x)}/{len(cov)}" if cov else "N/A"

            row = f"  {sid:<8}  {_gt_str(res['dia']):>8}  {_gt_str(res.get('vanilla')):>8}  {_gt_str(res.get('ragonly')):>9}"
            for m in COMMERCIAL_MODELS:
                row += f"  {_gt_str(res.get('commercial', {}).get(m)):>12}"
            print(row)

    # ── Per-scenario Judge breakdown ──────────────────────────
    if not args.no_judge:
        print("\n  --- LLM-as-a-Judge (per scenario) ---")
        all_systems = ["DIA", "Vanilla", "RAG-only"] + [m[:12] for m in COMMERCIAL_MODELS]
        for sid, res in results.items():
            if not isinstance(res.get("dia"), dict):
                continue
            print(f"\n  [{sid}] {res.get('title', '')}")
            header = f"    {'Criterion':<25}"
            for sys_name in all_systems:
                header += f"  {sys_name:>12}"
            print(header)
            print(f"    {'─'*25}" + "  ────────────" * len(all_systems))

            def _get_judge(source):
                if not source:
                    return {}
                return {k: v for k, v in source.get("judge", {}).items() if k in CRITERIA}

            judges = [
                _get_judge(res.get("dia")),
                _get_judge(res.get("vanilla")),
                _get_judge(res.get("ragonly")),
            ]
            for m in COMMERCIAL_MODELS:
                judges.append(_get_judge(res.get("commercial", {}).get(m)))

            for c in CRITERIA:
                row = f"    {c:<25}"
                for j in judges:
                    v = j.get(c, "N/A")
                    row += f"  {v:>12}"
                print(row)

            # Overall per scenario
            row_overall = f"    {'Overall':<25}"
            for j in judges:
                if j:
                    ov = round(sum(j.get(c, 0) for c in CRITERIA) / len(CRITERIA), 2)
                    row_overall += f"  {ov:>12}"
                else:
                    row_overall += f"  {'N/A':>12}"
            print(row_overall)

    # ── Save results ──────────────────────────────────────────
    save_data = {}
    for sid, res in results.items():
        if not isinstance(res.get("dia"), dict):
            save_data[sid] = res
            continue
        van_save = None
        if res.get("vanilla"):
            van_save = {
                "elapsed":   res["vanilla"].get("elapsed"),
                "cost_usd":  0.0,
                "judge":     res["vanilla"].get("judge"),
            }
        rag_save = None
        if res.get("ragonly"):
            rag_save = {
                "elapsed":   res["ragonly"].get("elapsed"),
                "cost_usd":  0.0,
                "judge":     res["ragonly"].get("judge"),
            }
        save_data[sid] = {
            "title":               res["title"],
            "ground_truth_points": res.get("ground_truth_points", []),
            "dia": {
                "elapsed":   res["dia"].get("elapsed"),
                "cost_usd":  0.0,
                "judge":     res["dia"].get("judge"),
            },
            "vanilla": van_save,
            "ragonly":  rag_save,
            "commercial": {
                m: {
                    "elapsed":   v.get("elapsed"),
                    "cost_usd":  v.get("cost_usd", 0.0),
                    "usage":     v.get("usage"),
                    "judge":     v.get("judge"),
                }
                for m, v in res.get("commercial", {}).items()
            },
        }

    # Summary stats
    comm_total_costs = {}
    for model in COMMERCIAL_MODELS:
        costs = [r["commercial"][model]["cost_usd"] for r in results.values()
                 if model in r.get("commercial", {}) and r["commercial"][model].get("cost_usd") is not None]
        comm_total_costs[model] = round(sum(costs), 6) if costs else 0.0

    save_data["_summary"] = {
        "dia_judge_avg":         avg_scores(dia_judge_list),
        "dia_avg_elapsed":       round(sum(dia_times)/len(dia_times), 1) if dia_times else None,
        "dia_total_cost":        0.0,
        "vanilla_judge_avg":     avg_scores(van_judge_list),
        "vanilla_avg_elapsed":   round(sum(van_times)/len(van_times), 1) if van_times else None,
        "vanilla_total_cost":    0.0,
        "ragonly_judge_avg":     avg_scores(rag_judge_list),
        "ragonly_avg_elapsed":   round(sum(rag_times)/len(rag_times), 1) if rag_times else None,
        "ragonly_total_cost":    0.0,
        "commercial_judge_avg":  {m: avg_scores(v) for m, v in comm_judge_map.items()},
        "commercial_total_cost": comm_total_costs,
    }

    out_path = os.path.join(RESULT_DIR, "scenario_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved -> {out_path}")

    # ── API Usage Summary ─────────────────────────────────────
    print("\n" + "=" * 65)
    print("  API USAGE")
    print("=" * 65)

    if _api_usage_by_model:
        print(f"  {'Model':<35} {'Prompt':>10} {'Completion':>12} {'Total':>10} {'Cost (USD)':>12}")
        print(f"  {'─'*35} {'─'*10} {'─'*12} {'─'*10} {'─'*12}")
        for model_name, usage in sorted(_api_usage_by_model.items()):
            print(f"  {model_name:<35} {usage['prompt_tokens']:>10,} {usage['completion_tokens']:>12,} {usage['total_tokens']:>10,} ${usage['cost_usd']:>11.4f}")
        print(f"  {'─'*35} {'─'*10} {'─'*12} {'─'*10} {'─'*12}")

    print(f"  {'TOTAL':<35} {_api_usage['prompt_tokens']:>10,} {_api_usage['completion_tokens']:>12,} {_api_usage['total_tokens']:>10,} ${_api_usage['cost_usd']:>11.4f}")
    print("=" * 65)
    print("\n")


if __name__ == "__main__":
    main()
