# DIA - Digital Investigation Assistant

A Digital Investigation Assistant Chatbot, run by ReAct + Doman-Specific RAG system and built with LangGraph and local LLMs. DIA searches domain-specific knowledge bases — forensic techniques, legal procedures, and case evidence — to provide grounded, actionable answers without transmitting sensitive data to external services.

---

## Architecture & Workflow

DIA employs a **ReACT (Reasoning + Acting)** agent implemented as a custom LangGraph StateGraph with three nodes:

1. **Reason**: The agent analyses the query, reviews any observations gathered so far, and decides whether to search a database or conclude the investigation.
2. **Act**: The agent's plan is parsed and converted into tool calls for parallel database retrieval.
3. **Observation**: FAISS retrieves candidate documents; a cross-encoder reranker selects the most relevant chunks.

After the ReACT loop concludes, a **Synthesis** phase consolidates all retrieved observations into a final response via a separate LLM call.

### Architecture
![Framework](./figures/DIA%20Architecture.png)

### Workflow
![Workflow](./figures/DIA%20Workflow.png)

## User Interface

The web interface is built with Streamlit and provides the following features:

- **Model Selection**: Switch between locally available Ollama models from the sidebar.
- **Case Management**: Load case-specific evidence databases and switch between investigation cases.
- **File Attachments**: Attach documents (PDF, DOCX, XLSX, TXT, CSV) and images directly in the chat input for analysis.
- **Session History**: Conversations are persisted across sessions, with full chat history accessible from the sidebar.
- **ReACT Transparency**: The agent's reasoning process (Thought, Action, Observation) is displayed in a collapsible panel, with only the final answer shown outside.

![UI](./figures/DIA%20UI.png)

### Knowledge Bases

| Database | Contents |
|----------|----------|
| **Tech DB** | NIST guidelines, forensic tool manuals (Autopsy, EnCase, Volatility, Wireshark, Eric Zimmerman tools, etc.), Forensic Wiki |
| **Protocol DB** | South Korean criminal law (translated to English), ISO/IEC standards, US federal procedure, digital evidence collection regulations |
| **Case DB** | Per-case seized evidence inventories, forensic analysis reports, chain-of-custody records, event timelines, suspect profiles |

### Key Components

| Component | Role |
|-----------|------|
| `main.py` | Streamlit web UI with session management, file attachment support, and model selection |
| `react_engine.py` | ReACT agent engine (LangGraph StateGraph with Reason/Act/Observation nodes + Synthesis phase) |
| `rag_core.py` | FAISS vector search, cross-encoder reranking, embedding model management |
| `tools.py` | Streamlit-aware wrappers around `rag_core` with caching and logging |
| `config.py` | Directory paths and helper functions |

---

## Requirements

- Python 3.10+
- A virtual environment recommended (e.g. Miniconda, venv)
- [Ollama](https://ollama.com) with a multimodal or ReACT-capable model (default: `qwen3.5:9b`)
- NVIDIA GPU with 16 GB+ VRAM recommended
- OpenAI API key (for evaluation scripts only)

---

## Installation

```bash
git clone https://github.com/your-repo/DIA.git
cd DIA
pip install -r requirements.txt
```

Pull the default model:

```bash
ollama pull qwen3.5:9b
```

Create a `.env` file if running evaluations that require the OpenAI API:

```
OPENAI_API_KEY=sk-...
```

---

## Usage

### Running the Application

```bash
streamlit run main.py
```

The application opens at `http://localhost:8501`.

1. Select an LLM model from the sidebar dropdown (lists all locally available Ollama models).
2. Select an investigation case (or proceed without one for general queries).
3. Ask forensic, legal, or case-specific questions. Attach files or images if needed.
4. The agent searches relevant databases via the ReACT loop and returns a structured answer.
5. Session history is persisted to `history/` and conversation memory is maintained via SQLite.

### Running Evaluations

**RAGAS (RAG pipeline quality):**

```bash
# Step 1: Generate testset (questions + ground truth) from DB documents
python eval/eval_ragas.py --gen-only

# Step 2: Batch-run all questions through the ReACT agent
python eval/eval_batch_run.py

# Step 3: Evaluate with RAGAS metrics
python eval/eval_ragas.py --eval-only
```

**Scenario evaluation (LLM-as-a-Judge):**

```bash
# Run all systems and evaluate
python eval/eval_scenario.py

# Run a specific scenario only
python eval/eval_scenario.py --scenarios S01

# Evaluate using cached responses only
python eval/eval_scenario.py --eval-only
```

---

## Evaluation Metrics

### RAGAS (RAG Pipeline Quality)

Evaluated on 100 automatically generated questions across Tech DB and Protocol DB.

| Metric | Description |
|--------|-------------|
| Faithfulness | Whether the response is grounded in retrieved contexts |
| Answer Relevance | Whether the response addresses the question |
| Context Precision | Proportion of retrieved documents that are relevant |
| Context Recall | Whether all necessary information was retrieved |

### Scenario Evaluation (End-to-End Quality)

Multiple systems are compared across forensic investigation scenarios containing both general knowledge (Type A) and case-specific (Type B) questions:

| System | Model | RAG | ReACT | Cost |
|--------|-------|-----|-------|------|
| DIA | qwen3.5:9b | Yes | Yes | Free (local) |
| RAG-only | qwen3.5:9b | Yes | No | Free (local) |
| Vanilla | qwen3.5:9b | No | No | Free (local) |
| Commercial | GPT-4o-mini / GPT-5.4-mini | No | No | API billing |

Evaluation criteria (LLM-as-a-Judge, scored 1-5):

| Criterion | Description |
|-----------|-------------|
| Factual Accuracy | Technical and legal claims are correct |
| Case Specificity | Case-specific questions answered with concrete case data |
| Completeness | All sub-questions addressed |
| Practical Usefulness | Steps are concrete enough for an investigator to follow |
| Clarity | Response is well-structured and easy to follow |

---

## Project Structure

```
DIA/
├── main.py                  # Streamlit web application
├── react_engine.py          # ReACT agent engine (LangGraph StateGraph)
├── rag_core.py              # FAISS vector search + cross-encoder reranker
├── tools.py                 # Streamlit-cached RAG wrappers
├── config.py                # Directory paths and helpers
├── requirements.txt
├── .env                     # API keys (not committed)
│
├── vector_db/               # Shared FAISS indexes
│   ├── Tech.faiss
│   ├── Tech_metadata.json
│   ├── Protocol.faiss
│   └── Protocol_metadata.json
│
├── cases/                   # Per-case investigation data
│   ├── #001_Scenario/
│   │   ├── case_db/         # Case-specific FAISS index
│   │   └── docs/            # Source evidence documents
│   ├── #002_Scenario/
│   └── #003_Scenario/
│
├── models/                  # Local models (fully offline)
│   ├── all-mpnet-base-v2/   # Sentence embedding model
│   └── ms-marco-MiniLM-L-6-v2/  # Cross-encoder reranker
│
├── history/                 # Session chat history (JSON)
├── attachments/             # Uploaded files per session
├── checkpoints.db           # Conversation memory (SQLite)
├── figures/                 # Architecture diagrams
│
└── eval/                    # Evaluation suite
    ├── eval_batch_run.py    # Batch answer generation for RAGAS
    ├── eval_ragas.py        # RAGAS evaluation (4 metrics)
    ├── eval_scenario.py     # Scenario evaluation (LLM-as-a-Judge)
    ├── results/             # Evaluation output (JSON, CSV)
    └── scenarios/           # Scenario definitions and cached responses
```

---

## Acknowledgement

This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No.RS-2024-00441762, Global Advanced Cybersecurity Human Resources Development)
