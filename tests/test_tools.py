"""
These are integration tests: search_technical_manual talks to a real
(local) Qdrant collection with fastembed-generated embeddings (fully local,
no API key needed). An already-indexed ./qdrant_storage is required — run
`python -m core.tools` once first if the collection hasn't been built yet.
"""

from core.tools import lookup_device_telemetry, lookup_spare_part, search_technical_manual


def test_telemetry_valid_serial():
    result = lookup_device_telemetry.invoke({"serial_number": "SN-9002"})
    assert "Critical Warning" in result
    assert "Error E-204" in result
    assert "97°C" in result


def test_telemetry_invalid_serial():
    invalid_serial = "SN-9999"
    result = lookup_device_telemetry.invoke({"serial_number": invalid_serial})
    assert f"Device with serial number '{invalid_serial}' not found" in result


def test_spare_part_catalog_lookup():
    result = lookup_spare_part.invoke({"part_number": "FAN-4412"})
    assert "12 units" in result
    assert "$65.00" in result


def test_technical_manual_rag_retrieval():
    query = "torque specification for battery M8 connection lugs"
    result = search_technical_manual.invoke({"query": query})
    assert "12.0 Newton-meters" in result
    assert "106 inch-pounds" in result


def test_technical_manual_returns_citation_tag():
    """Retrieved chunks should carry a [Source: Section ...] tag, not just raw text."""
    query = "Error E-204 thermal overload"
    result = search_technical_manual.invoke({"query": query})
    assert "[Source: Section" in result
