from pathlib import Path

import pytest

from enrichment import MetadataNormalizer, enrich_document
from schemas import DocumentRef, EnrichedDocument, RawDocument


def _raw_document(metadata=None):
    ref = DocumentRef(path=Path("report.txt"))
    return RawDocument(
        ref=ref,
        doc_id="doc-1",
        pages=((0, "Alpha"),),
        text="Alpha",
        metadata=metadata or {},
    )


def test_enrich_document_supports_identity_path():
    raw = _raw_document({"Language": "en"})

    enriched = enrich_document(raw)

    assert enriched == EnrichedDocument(
        raw=raw,
        text="Alpha",
        metadata={"Language": "en"},
        transformations=(),
    )


def test_enrich_document_applies_named_steps_in_order():
    class AddSuffix:
        name = "add_suffix"

        def enrich(self, pages, metadata):
            return [(number, f"{text} Beta") for number, text in pages], {
                **metadata,
                "suffix": "Beta",
            }

    raw = _raw_document({"Language": "EN", "Owner": "Research"})

    enriched = enrich_document(raw, [MetadataNormalizer(), AddSuffix()])

    assert enriched.text == "Alpha Beta"
    assert enriched.metadata == {
        "language": "EN",
        "owner": "Research",
        "suffix": "Beta",
    }
    assert enriched.transformations == ("normalize_metadata", "add_suffix")
    assert raw.metadata == {"Language": "EN", "Owner": "Research"}


@pytest.mark.parametrize(
    ("enricher", "message"),
    [
        (type("Unnamed", (), {"name": "", "enrich": lambda self, pages, metadata: (pages, metadata)})(), "name"),
        (type("BadPair", (), {"name": "bad", "enrich": lambda self, pages, metadata: pages})(), "pair"),
        (type("BadPages", (), {"name": "bad", "enrich": lambda self, pages, metadata: (None, metadata)})(), "pages"),
        (type("BadMetadata", (), {"name": "bad", "enrich": lambda self, pages, metadata: (pages, None)})(), "metadata"),
    ],
)
def test_enrich_document_rejects_invalid_connector_outputs(enricher, message):
    with pytest.raises((TypeError, ValueError), match=message):
        enrich_document(_raw_document(), [enricher])
