import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from fastembed import TextEmbedding

load_dotenv()

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
# Runs fully locally via ONNX Runtime (fastembed) — no API key, no network
# call at query time, no external service to be rate-limited or return an
# auth error. The model is downloaded and cached once on first run.
# BAAI/bge-small-en-v1.5 is fastembed's own default: a small model that
# scores better on retrieval benchmarks than all-MiniLM-L6-v2 while staying
# at the same 384 dimensions, so no other config needs to change.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
_fastembed_model = TextEmbedding(model_name=EMBEDDING_MODEL)


class ReliableEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vec.tolist() for vec in _fastembed_model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(_fastembed_model.embed([text]))).tolist()


embeddings = ReliableEmbeddings()

# ---------------------------------------------------------------------------
# Qdrant (local/embedded — see docs/scaling.md for the cloud migration path)
# ---------------------------------------------------------------------------
QDRANT_DIR = "./qdrant_storage"
client = QdrantClient(path=QDRANT_DIR)
COLLECTION_NAME = "voltpro_manuals"

if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )

vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)

# ---------------------------------------------------------------------------
# Structure-aware chunking
# ---------------------------------------------------------------------------
# The manual isn't generic prose — it's organized into "=== SECTION N: ... ==="
# blocks, and within Section 3 into "[Error Code E-xxx: ...]" blocks. Splitting
# by raw character count cuts these blocks apart (e.g. separating an error's
# Trigger Condition from its Remediation steps), which silently degrades
# answer quality without ever raising an exception. Instead we split on the
# document's own structural boundaries and attach section/error-code metadata
# to each chunk so retrieval results can be cited, not just quoted.

SECTION_PATTERN = re.compile(r"(===\s*SECTION\s+(\d+):[^\n]*===)")
ERROR_CODE_PATTERN = re.compile(r"\[Error Code (E-\d+)")

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "equipment_manual.txt"


def _split_into_structured_chunks(raw_text: str) -> list[Document]:
    parts = SECTION_PATTERN.split(raw_text)
    docs: list[Document] = []

    # re.split with a capturing group interleaves: [pre, header, section_num, body, header, section_num, body, ...]
    preamble = parts[0].strip()
    if preamble:
        docs.append(Document(page_content=preamble, metadata={"section": "0"}))

    for i in range(1, len(parts), 3):
        header, section_num, body = parts[i], parts[i + 1], parts[i + 2]
        body = body.strip()

        # Within Section 3, further split on individual error-code blocks so
        # each error's full trigger/cause/remediation stays together.
        error_blocks = re.split(r"(?=\[Error Code E-\d+)", body)
        for block in error_blocks:
            block = block.strip()
            if not block:
                continue
            match = ERROR_CODE_PATTERN.search(block)
            error_code = match.group(1) if match else None
            metadata = {"section": section_num}
            if error_code:
                metadata["error_code"] = error_code
            docs.append(Document(page_content=f"{header}\n{block}", metadata=metadata))

    return docs


def initialize_knowledge_base():
    """Indexes the manual into Qdrant using structure-aware chunks, if not already indexed."""
    collection_info = client.get_collection(COLLECTION_NAME)
    if collection_info.points_count == 0:
        print("[INFO] Indexing technical documentation into Qdrant (structure-aware chunking)...")
        raw_text = DATA_PATH.read_text(encoding="utf-8")
        docs = _split_into_structured_chunks(raw_text)
        vector_store.add_documents(docs)
        print(f"[SUCCESS] Indexed {len(docs)} structured chunks into Qdrant.")
    else:
        print("[READY] Technical documentation collection already initialized in Qdrant.")


initialize_knowledge_base()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@tool
def search_technical_manual(query: str) -> str:
    """Searches the official VoltPro-X9000 service manual for diagnostic error codes,
    mechanical torque specs, wiring pinouts, and field maintenance procedures.
    Returns each result tagged with its source section and error code, if any."""
    results = vector_store.similarity_search(query, k=3)
    formatted = []
    for doc in results:
        section = doc.metadata.get("section", "?")
        error_code = doc.metadata.get("error_code")
        tag = f"[Source: Section {section}" + (f", {error_code}" if error_code else "") + "]"
        formatted.append(f"{tag}\n{doc.page_content}")
    return "\n\n".join(formatted)


@tool
def lookup_device_telemetry(serial_number: str) -> str:
    """Retrieves live sensor telemetry, thermal status, grid frequency, and active alarm codes
    for an inverter via its serial number."""
    telemetry_db = {
        "SN-9001": "Status: Operational | Temp: 48°C | AC Output: 230V @ 50.0Hz | Battery SoC: 88% | Active Alarm: None",
        "SN-9002": "Status: Critical Warning | Temp: 97°C (Exceeded) | Fan RPM: 0 | Battery SoC: 42% | Active Alarm: Error E-204",
        "SN-9003": "Status: Grid Disconnected | Grid Voltage: 0V | Mode: EPS Backup | Battery SoC: 14% | Active Alarm: Error E-405",
    }
    return telemetry_db.get(
        serial_number,
        f"Device with serial number '{serial_number}' not found in live telemetry network.",
    )


@tool
def lookup_spare_part(part_number: str) -> str:
    """Checks central logistics warehouse stock, unit pricing, and technical specs
    for equipment replacement parts."""
    parts_db = {
        "AF-901": "Part: Washable 50-micron stainless steel air filter (Pack of 2) | Stock: 24 units | Unit Price: $25.00",
        "FAN-4412": "Part: Dual ball-bearing 120mm brushless DC cooling fan | Stock: 12 units | Unit Price: $65.00",
        "SPD-600": "Part: Type II DC Surge Protective Device cartridge (600V DC) | Stock: 8 units | Unit Price: $85.00",
        "MB-9901": "Part: Main Logic Board pre-flashed with firmware v10.4 | Stock: 2 units | Unit Price: $520.00",
    }
    return parts_db.get(
        part_number,
        f"Part number '{part_number}' not recognized in logistics parts inventory catalog.",
    )


tools = [search_technical_manual, lookup_device_telemetry, lookup_spare_part]
