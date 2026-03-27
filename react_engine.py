import os
import time
import operator
import re as _re
from typing import Annotated, TypedDict, List, Union, Dict

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver

from rag_core import (
    search_tech_db as _rag_search_tech,
    search_protocol_db as _rag_search_protocol,
    search_case_db as _rag_search_case,
    get_shared_dbs,
)

# ─────────────────────────────────────────────────────────────
# Persistent memory (SQLite checkpointer)
# ─────────────────────────────────────────────────────────────
_CHECKPOINT_DB = os.path.join(os.path.dirname(__file__), "checkpoints.db")
_checkpointer = None

def get_checkpointer():
    """Returns a singleton SqliteSaver for conversation memory."""
    global _checkpointer
    if _checkpointer is None:
        import sqlite3
        conn = sqlite3.connect(_CHECKPOINT_DB, check_same_thread=False)
        _checkpointer = SqliteSaver(conn)
    return _checkpointer

# ─────────────────────────────────────────────────────────────
# Config & State Definition
# ─────────────────────────────────────────────────────────────
DEFAULT_MODEL = "qwen3.5:9b"
MAX_STEPS     = 3

class AgentState(TypedDict):
    """The state of the investigation agent."""
    # Annotated with operator.add to append new messages to the history
    messages: Annotated[List[BaseMessage], operator.add]
    has_case: bool

# ─────────────────────────────────────────────────────────────
# Tool wrappers (Standardised for LangGraph)
# ─────────────────────────────────────────────────────────────
def search_tech_db(query: str) -> str:
    """Searches the FAISS Technical Database for tool usage and manuals."""
    text, _ = _rag_search_tech(query)
    return text if text else "No technical documentation found."

def search_protocol_db(query: str) -> str:
    """Searches the FAISS Protocol Database for laws, regulations, and standards."""
    text, _ = _rag_search_protocol(query)
    return text if text else "No relevant protocols found."

def search_case_db(query: str) -> str:
    """Searches the case-specific FAISS database for evidence and documents."""
    text, _ = _rag_search_case(query)
    return text if text else "No relevant case documents found."

# ─────────────────────────────────────────────────────────────
# System Prompts
# ─────────────────────────────────────────────────────────────

SYSTEM_IDENTITY = """You are DIA (Digital Investigation Assistant), an expert AI system for digital forensics investigations.
You combine three domains of expertise:
- Technical Forensics: artefact analysis, tool usage, evidence recovery methods
- Legal & Compliance: laws, regulations, protocols, chain-of-custody, ISO standards
- Case Analysis: interpreting seized evidence, timelines, and suspect activities"""

def _build_reasoning_prompt(has_case: bool, has_observations: bool) -> str:
    """Build the system prompt for the reasoning (Thought) node."""

    db_descriptions = (
        "## Available Databases\n"
        "- **search_tech_db(query)**: Forensic tools, techniques, artefact locations, scripts, NIST guidelines, "
        "tool manuals (Autopsy, EnCase, Volatility, Wireshark, etc.)\n"
        "- **search_protocol_db(query)**: Laws and regulations (Korean Criminal Act, Criminal Procedure Act, "
        "Protection of Communications Secrets Act, etc.), ISO/IEC standards, chain-of-custody protocols\n"
    )
    if has_case:
        db_descriptions += (
            "- **search_case_db(query)**: Case-specific evidence — seized device inventories, forensic analysis "
            "reports, event timelines, chain-of-custody records, suspect profiles, communications. "
            "**You MUST search this database when the question relates to the current investigation.**\n"
        )

    if not has_observations:
        task = """## Your Task
1. Analyse the user's question. Does it require forensic knowledge, legal information, or case evidence?
2. If YES — write a search plan to gather the needed information.
3. If NO (e.g. greetings, simple conversation, questions you can answer directly) — use finish() to respond immediately.

## Output Format — if search is needed:
PLAN:
- search_tech_db(concise query here)
- search_case_db(concise query here)
Reason: [explain why these searches are needed]

## Output Format — if no search needed:
PLAN:
- finish()
Reason: [your direct response to the user]

## Rules
- Start with PLAN first, then explain your reasoning after.
- Keep search queries short (1-4 words). Multiple searches run in parallel.
- STOP after your reasoning. Do NOT write a final answer."""
    else:
        task = """## Your Task
1. Read the Observations you have gathered so far.
2. Compare against the user's original question — which parts are answered, which still need information?
3. Decide: do you need more searches, or is the information sufficient?

## Output Format — if more searches needed:
PLAN:
- search_protocol_db(concise query here)
Reason: [explain what information is still missing and why this search will help]

## Output Format — if sufficient and ready to conclude:
PLAN:
- finish()
Reason: [briefly summarise what you gathered]

## Rules
- Start with PLAN first, then explain.
- STOP after your reasoning. Do NOT write a final answer or summary.
- Do NOT repeat searches you have already performed."""

    return f"""{SYSTEM_IDENTITY}

You are in the **REASONING** phase.

{db_descriptions}
{task}"""


# ─────────────────────────────────────────────────────────────
# Node Functions
# ─────────────────────────────────────────────────────────────

REASONING_REMINDER_SEARCH = """[REMINDER] Write PLAN with specific DB name and query. Example:
PLAN:
- search_tech_db(USB registry artefacts)
- search_protocol_db(digital evidence law Korea)
Reason: [why]"""

REASONING_REMINDER_CONCLUDE = """[REMINDER] Need more info? Write PLAN. Have enough? Write finish(). Example:
PLAN:
- search_protocol_db(chain of custody requirements)
Reason: [what is missing]
OR
PLAN:
- finish()
Reason: [summary of what was gathered]"""


def reasoning_node(state: AgentState):
    """Phase 1-A: Analyse the situation, plan the next action."""
    has_case = state.get("has_case", False)
    has_observations = any(isinstance(msg, ToolMessage) for msg in state.get("messages", []))
    system_prompt = _build_reasoning_prompt(has_case, has_observations)

    llm = ChatOllama(model=DEFAULT_MODEL, temperature=0.2, num_predict=4096, think=False)
    messages = [{"role": "system", "content": system_prompt}] + state['messages']

    # Extract original user question for the reminder
    user_question = ""
    for msg in state.get("messages", []):
        if isinstance(msg, HumanMessage):
            user_question = str(msg.content)[:200]
            break

    # Append a reminder at the end so the model doesn't forget the format or the question
    reminder = REASONING_REMINDER_CONCLUDE if has_observations else REASONING_REMINDER_SEARCH
    reminder += f"\n\n[Original Question] {user_question}"
    messages.append({"role": "user", "content": reminder})

    response = llm.invoke(messages)
    return {"messages": [response]}


def _parse_plan(thought_text: str, has_case: bool) -> list:
    """
    Parse PLAN lines from the Thought output.
    Returns list of (tool_name, query) tuples.

    Expected format in Thought:
        PLAN:
        - search_tech_db(USB connection artefacts Windows registry)
        - search_case_db(seized USB device details)
    """
    valid_tools = {"search_tech_db", "search_protocol_db"}
    if has_case:
        valid_tools.add("search_case_db")

    calls = []
    # Match patterns like: search_tech_db(query text here)
    for match in _re.finditer(r"(search_\w+_db)\(([^)]+)\)", thought_text):
        tool_name = match.group(1).strip()
        query = match.group(2).strip()
        if tool_name in valid_tools and query:
            calls.append((tool_name, query))

    return calls


def acting_node(state: AgentState):
    """Phase 1-B: Parse the Thought's PLAN and create tool calls, or end the loop."""
    has_case = state.get("has_case", False)

    # Get the last Thought
    last_thought = ""
    for msg in reversed(state.get("messages", [])):
        if hasattr(msg, "content") and msg.content:
            last_thought = str(msg.content)
            break

    # Check for finish() signal
    if _re.search(r"finish\(\)", last_thought, _re.IGNORECASE):
        # If observations exist → proceed to Phase 2 synthesis (not direct answer)
        has_observations = any(isinstance(msg, ToolMessage) for msg in state.get("messages", []))
        if has_observations:
            return {"messages": [AIMessage(content="Investigation complete. Proceeding to synthesis.")]}
        else:
            # No observations → use Reason as direct answer
            reason_match = _re.search(r"Reason:\s*(.+)", last_thought, _re.DOTALL)
            direct_answer = reason_match.group(1).strip() if reason_match else "Investigation complete."
            return {"messages": [AIMessage(content=f"DIRECT_ANSWER:{direct_answer}")]}

    # Parse PLAN from Thought
    planned_calls = _parse_plan(last_thought, has_case)

    if planned_calls:
        tool_calls = []
        for i, (tool_name, query) in enumerate(planned_calls):
            tool_calls.append({
                "name": tool_name,
                "args": {"query": query},
                "id": f"call_{tool_name}_{i}",
            })
        return {"messages": [AIMessage(content="", tool_calls=tool_calls)]}

    # Fallback: no PLAN and no CONCLUSION — treat as conclusion
    return {"messages": [AIMessage(content="No further searches planned. Proceeding to synthesis.")]}

def should_continue(state: AgentState):
    """Determine whether to proceed with tool execution or conclude."""
    last_message = state['messages'][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "observation"
    return END

# ─────────────────────────────────────────────────────────────
# Agent Creation
# ─────────────────────────────────────────────────────────────
def create_investigation_agent(has_case: bool = False, use_memory: bool = False):
    workflow = StateGraph(AgentState)
    
    workflow.add_node("reason", reasoning_node)
    workflow.add_node("act", acting_node)
    
    tools = [search_tech_db, search_protocol_db]
    if has_case:
        tools.append(search_case_db)
    workflow.add_node("observation", ToolNode(tools))
    
    workflow.set_entry_point("reason")
    workflow.add_edge("reason", "act")
    workflow.add_conditional_edges(
        "act",
        should_continue,
        {"observation": "observation", END: END}
    )
    workflow.add_edge("observation", "reason")
    
    checkpointer = get_checkpointer() if use_memory else None
    return workflow.compile(checkpointer=checkpointer)

# ─────────────────────────────────────────────────────────────
# Synthesis Phase
# ─────────────────────────────────────────────────────────────
SYNTHESIS_SYSTEM = f"""{SYSTEM_IDENTITY}

You are now in the **SYNTHESIS** phase.

## Core Principles
2. **Specificity**: When citing case data from case_db, use exact values (timestamps, serial numbers, file paths, article numbers, etc). If you cannot find specific data or value, do NOT make assumptions or fabricate details. State that it appears to be missing from the case documents.
3. **Cross-Domain Synthesis**: When user's enquiry involves multiple domains, integrate your findings where relevant. If the user did not ask for an integrated answer, no need to integrate. For example, if the user asks about a suspect's activities (case_db) AND relevant laws (protocol_db), you can mention how certain actions may violate specific regulations, even if not explicitly asked for.
4. **Match the user's intent**: Read the original enquiry carefully. If the user asks for a brief summary, be concise. If they ask for a detailed report, be thorough. If they ask a simple question, give a direct answer. Do NOT over-produce content beyond what was requested.
5. **Language**: Default language should be English. However, you must respond in the SAME LANGUAGE as the original enquiry.

## CRITICAL OUTPUT RULES
- Output ONLY the final response to the user. Do NOT write your internal reasoning, planning, drafting process, or self-critique.
- Start your response directly with the answer content."""

def _synthesise(question: str, observations: list, model: str, verbose: bool = True) -> str:
    obs_text = "\n\n---\n\n".join(observations)
    llm = ChatOllama(model=model, temperature=0.2, num_predict=16384, think=False)

    prompt = f"## Gathered Observations & Artefacts\n{obs_text}\n\n## Original Enquiry\n{question}\n\nUsing the gathered observations, respond to the original enquiry. Match the scope and detail level to what the user asked for."

    if verbose:
        sys_len = len(SYNTHESIS_SYSTEM)
        prompt_len = len(prompt)
        obs_len = len(obs_text)
        print(f"  [Synthesis Input] system: {sys_len} chars, prompt: {prompt_len} chars (observations: {obs_len} chars, {len(observations)} items)")

    response = llm.invoke([
        {"role": "system", "content": SYNTHESIS_SYSTEM},
        {"role": "user", "content": prompt},
    ])

    content = response.content or ""

    # 1. Cut at leaked special tokens (model hallucinating new conversations)
    for stop_token in ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]:
        if stop_token in content:
            content = content[:content.index(stop_token)]

    # 2. Remove <think>...</think> blocks
    content = _re.sub(r"<think>.*?</think>", "", content, flags=_re.DOTALL)

    # 3. Handle malformed think tags
    if "</think>" in content:
        content = content.split("</think>")[-1]
    if "<think>" in content:
        content = content.split("<think>")[0]

    return content.strip()

# ─────────────────────────────────────────────────────────────
# Execution Engine
# ─────────────────────────────────────────────────────────────
def run_agent(question: str, has_case: bool = False, model: str = DEFAULT_MODEL, 
              max_steps: int = MAX_STEPS, verbose: bool = True) -> tuple:
    get_shared_dbs()
    
    agent = create_investigation_agent(has_case=has_case)
    inputs = {"messages": [HumanMessage(content=question)], "has_case": has_case}
    
    start_time = time.time()
    steps_log = []
    observations = []
    final_raw_output = ""

    if verbose: print(f"\n[*] Starting Investigation: {question[:50]}...")

    try:
      for step in agent.stream(inputs, config={"recursion_limit": max_steps * 3}):
        for node_name, output in step.items():
            last_msg = output["messages"][-1]
            
            if node_name == "reason":
                if verbose: print(f"\n[Thought] {last_msg.content[:300]}...")
                steps_log.append({"node": "reason", "content": last_msg.content})
                
            elif node_name == "act":
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    for tc in last_msg.tool_calls:
                        if verbose: print(f"[Action]  Invoking {tc['name']}")
                        steps_log.append({"node": "act", "tool": tc['name'], "args": tc['args']})
                else:
                    final_raw_output = last_msg.content
                    
            elif node_name == "observation":
                for tool_msg in output["messages"]:
                    obs = str(tool_msg.content)
                    observations.append(obs)
                    if verbose:
                        print(f"[Observation] ({len(obs)} chars)")
                        print(f"  {obs[:300].replace(chr(10), ' ')}")
                        if len(obs) > 300:
                            print(f"  ... ({len(obs) - 300} more chars)")
                    steps_log.append({"node": "observation", "content": obs})
    except Exception as e:
        if verbose:
            print(f"\n[!] ReACT loop ended: {str(e)[:100]}")
            print(f"    Proceeding to synthesis with {len(observations)} observations collected so far.")

    # ── Check for direct answer (no RAG needed) ──
    if final_raw_output and "DIRECT_ANSWER:" in final_raw_output:
        final_answer = final_raw_output.split("DIRECT_ANSWER:", 1)[1].strip()
        if verbose:
            print(f"\n[Direct Answer] {len(final_answer)} chars (no Phase 2 needed)")
        elapsed = round(time.time() - start_time, 1)
        if verbose:
            print(f"[Done] {len(final_answer)} chars, {elapsed}s")
        return final_answer, elapsed, steps_log

    # ── Final Answer Synthesis & Filtration ──
    if verbose:
        print(f"\n{'='*50}")
        print(f"[Phase 2] Synthesis starting... ({len(observations)} observations collected)")
    if observations:
        final_answer = _synthesise(question, observations, model, verbose)
        if verbose:
            print(f"[Phase 2] Synthesis complete. ({len(final_answer)} chars)")
    else:
        final_answer = final_raw_output or ""
        if verbose:
            print(f"[Phase 2] No observations — using raw output. ({len(final_answer)} chars)")

    # SAFETY NET: If stripping <think> tags leaves an empty string, recovery is needed
    stripped_answer = _re.sub(r"<think>.*?</think>", "", final_answer, flags=_re.DOTALL).strip()
    
    if not stripped_answer and "<think>" in final_answer:
        # Extract content inside tags if outside is empty
        match = _re.search(r"<think>(.*?)</think>", final_answer, flags=_re.DOTALL)
        final_answer = match.group(1).strip() if match else final_answer
    else:
        final_answer = stripped_answer

    # Extract content after Final Answer prefix
    if "Final Answer:" in final_answer:
        final_answer = final_answer.split("Final Answer:", 1)[1]

    # Clean up residual tokens
    for token_pattern in [r"<\|endoftext\|>.*", r"<\|im_start\|>.*", r"<\|im_end\|>.*"]:
        final_answer = _re.sub(token_pattern, "", final_answer, flags=_re.DOTALL)
    
    final_answer = final_answer.strip()
    if not final_answer and verbose: print("    [Warning] Final answer is empty after filtration.")

    elapsed = round(time.time() - start_time, 1)
    return final_answer, elapsed, steps_log