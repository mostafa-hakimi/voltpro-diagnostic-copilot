"""Technical tools and Qdrant RAG retrieval for VoltPro-X9000."""

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

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

_fastembed_model = None

def get_fastembed_model():
    global _fastembed_model
    if _fastembed_model is None:
        _fastembed_model = TextEmbedding(model_name=EMBEDDING_MODEL)
    return _fastembed_model


class ReliableEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = get_fastembed_model()
        return [vec.tolist() for vec in model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        model = get_fastembed_model()
        return next(iter(model.embed([text]))).tolist()


embeddings = ReliableEmbeddings()

QDRANT_DIR = "./qdrant_storage"
COLLECTION_NAME = "voltpro_manuals"

_client = None
_vector_store = None

def get_qdrant_client():
    global _client
    if _client is None:
        _client = QdrantClient(path=QDRANT_DIR)
    return _client

def get_vector_store():
    global _vector_store
    if _vector_store is None:
        client = get_qdrant_client()
        if not client.collection_exists(COLLECTION_NAME):
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
        _vector_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=embeddings,
        )
    return _vector_store


SECTION_PATTERN = re.compile(r"(===\s*SECTION\s+(\d+):[^\n]*===)")
ERROR_CODE_PATTERN = re.compile(r"\[Error Code (E-\d+)")
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "equipment_manual.txt"


def _split_into_structured_chunks(raw_text: str) -> list[Document]:
    parts = SECTION_PATTERN.split(raw_text)
    docs: list[Document] = []

    preamble = parts[0].strip()
    if preamble:
        docs.append(Document(page_content=preamble, metadata={"section": "0"}))

    for i in range(1, len(parts), 3):
        header, section_num, body = parts[i], parts[i + 1], parts[i + 2]
        body = body.strip()

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
    client = get_qdrant_client()
    vs = get_vector_store()
    collection_info = client.get_collection(COLLECTION_NAME)
    if collection_info.points_count == 0:
        print("[INFO] Indexing technical documentation into Qdrant...")
        raw_text = DATA_PATH.read_text(encoding="utf-8")
        docs = _split_into_structured_chunks(raw_text)
        vs.add_documents(docs)
        print(f"[SUCCESS] Indexed {len(docs)} structured chunks into Qdrant.")
    else:
        print("[READY] Technical documentation collection already initialized in Qdrant.")


@tool
def search_technical_manual(query: str) -> str:
    """Searches the official VoltPro-X9000 service manual for diagnostic error codes,
    mechanical torque specs, wiring pinouts, and field maintenance procedures.
    Returns each result tagged with its source section and error code, if any."""
    initialize_knowledge_base()
    vs = get_vector_store()
    results = vs.similarity_search(query, k=3)
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

if __name__ == "__main__":
    initialize_knowledge_base()
