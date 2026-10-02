# Engineering Decisions

This document details the architectural rationale, rejected alternatives, and operational trade-offs for the VoltPro Diagnostic Copilot.

## 1. Industrial inverter domain over generic consumer PDF chat

**Chose:** VoltPro-X9000 industrial hybrid inverter service documentation.  
**Rejected:** General knowledge bases or consumer-grade PDF Q&A.

**Rationale:** Industrial support desks face catastrophic costs during system outages. Field technicians require high-precision retrieval where error codes, wiring pinouts, and torque specifications must be cited verbatim rather than approximated. Grounding the assistant in equipment documentation with strict refusal constraints addresses a high-consequence operational domain.

## 2. Structure-aware chunking over arbitrary character/token splitting

**Chose:** Custom regex splitting along `=== SECTION N ===` and `[Error Code E-xxx]` boundaries (`core/tools.py`).  
**Rejected:** Fixed-size recursive character text splitters (e.g. 500-token chunks with 50-token overlap).

**Rationale:** Technical service manuals contain highly coupled diagnostic procedures. Splitting by arbitrary character length routinely cuts an error code's trigger condition from its safety remediation steps, silently degrading diagnostic accuracy. Preserving structural blocks guarantees complete procedural integrity while allowing explicit metadata tagging (`section`, `error_code`) for primary source citations.

## 3. Local ONNX embeddings (FastEmbed) over cloud embedding APIs

**Chose:** `BAAI/bge-small-en-v1.5` running locally via FastEmbed and ONNX Runtime.  
**Rejected:** Cloud embedding endpoints (e.g. OpenAI `text-embedding-3-small` or HuggingFace Inference API).

**Rationale:** Local inference eliminates external network latency, per-query API expenditure, and third-party rate limits. Crucially, it removes authentication failures (such as HTTP 401 token expiration errors) during query time, ensuring 100% offline retrieval reliability for field technicians.

## 4. Embedded local Qdrant instance over hosted cloud cluster

**Chose:** Embedded Qdrant storage (`QdrantClient(path="./qdrant_storage")`).  
**Rejected:** Provisioning a standalone managed Qdrant Cloud or Milvus cluster.

**Rationale:** The technical manual parses into fewer than 100 structured chunks. A local file-backed vector database provides zero-setup, self-contained execution without recurring cloud infrastructure costs. For multi-worker production deployments, the client configuration in `core/tools.py` swaps cleanly to remote Qdrant endpoints without touching downstream retrieval logic.

## 5. In-memory simulated telemetry and spare parts catalogs

**Chose:** In-memory dictionary lookup tools (`lookup_device_telemetry`, `lookup_spare_part`).  
**Rejected:** Direct live integration with proprietary SCADA and ERP databases in this demonstration.

**Rationale:** Simulates enterprise system-of-record integration while preserving deterministic testing. The agent learns to route queries between manual text, real-time sensor metrics (temperature, fan RPM, inverter alarms), and inventory availability without coupling the codebase to proprietary industrial networking protocols.

## 6. Bounded conversation summarization at message threshold

**Chose:** Automated conversation summarization triggered once thread depth exceeds 12 messages (`MAX_MESSAGES_BEFORE_SUMMARY = 12`), retaining the 4 most recent messages.  
**Rejected:** Unbounded conversation history or fixed sliding-window message dropping.

**Rationale:** Unbounded history inflates context window consumption and latency linearly. Simple sliding windows discard critical context (such as the inverter serial number or active error code established in turn 1). The summarization node compresses earlier turns into a structured digest while explicitly retaining technical identifiers.

## 7. Mandatory retrieval before refusal

**Chose:** Prompt directive strictly requiring a tool call before concluding an item is undocumented.  
**Rejected:** Allowing immediate model refusals based on pre-trained parametric memory.

**Rationale:** During early iterations, the model exhibited refusal leakage: upon observing an unanswerable question, it erroneously reused the prior turn's refusal for subsequent answerable questions without searching. Mandating a tool call forces an empirical lookup attempt on every user turn.

## 8. SQLite checkpointer for session state persistence

**Chose:** `SqliteSaver` persisting thread states to `customer_support.db`.  
**Rejected:** In-memory ephemeral checkpointers (`MemorySaver`).

**Rationale:** Real technical support sessions require persistence across browser reloads and network reconnects. SQLite provides zero-dependency transactional storage. For distributed multi-pod Kubernetes deployments, the checkpointer swaps cleanly to `PostgresSaver`.

## 9. First-class dual observability (LangSmith & Langfuse)

**Chose:** Integrated runtime telemetry instrumentation across graph nodes and Streamlit UI via LangSmith and Langfuse callbacks.  
**Rejected:** Unmonitored black-box execution.

**Rationale:** Production field assistants require visibility into latency bottlenecks, token consumption per tool execution, and exact prompt/response audit trails. The telemetry layer initializes non-blocking callbacks when credentials exist while falling back gracefully to zero-overhead offline execution during unit tests.

## 10. Injectable LLM architecture for deterministic control-flow testing

**Chose:** Dependency-injected graph compilation (`build_graph(llm=...)`) tested via `ScriptedLLM`.  
**Rejected:** Testing exclusively against live LLM API endpoints.

**Rationale:** Live LLMs introduce non-determinism, incur API costs, and require network access in CI pipelines. Testing the StateGraph with pre-scripted message sequences proves state transitions, routing logic, tool invocation, and summarization thresholds in under 4 seconds without external secrets.
