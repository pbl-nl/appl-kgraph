import hashlib

from db_storage import Storage
import ingestion
import production_ingestion
from schemas import Entity, Relation, ExtractionResult


class DeterministicEmbedder:
    dimension = 8

    def embed_texts(self, texts, batch_size=64):
        return [
            [value / 255 for value in hashlib.sha256(text.encode()).digest()[:8]]
            for text in texts
        ]


class DeterministicExtractor:
    def __init__(self):
        self.calls = []

    def extract(self, chunks):
        self.calls.append(chunks)
        if not chunks:
            return ExtractionResult()
        source_ids = tuple(chunk.chunk_uuid for chunk in chunks)
        filepath = chunks[0].filepath
        return ExtractionResult(
            entities=[
                Entity("Alpha", "Organization", "An organization", source_ids, filepath)
            ],
            relations=[
                Relation(
                    "Alpha",
                    "Beta",
                    "Works with",
                    "partnership",
                    1.0,
                    source_ids,
                    filepath,
                )
            ],
            diagnostics=[{"chunk_uuid": source_ids[0], "summary": "complete"}],
        )


def configure_ingestion(monkeypatch, extractor=None):
    extractor = extractor or DeterministicExtractor()

    def make_storage(paths=None):
        storage = Storage(paths=paths, embedder=DeterministicEmbedder())
        return storage

    factory = production_ingestion.create_ingestion_pipeline

    def pipeline(storage, paths=None, **kwargs):
        return factory(storage, paths, extractor=extractor, **kwargs)

    monkeypatch.setattr(ingestion, "Storage", make_storage)
    monkeypatch.setattr(ingestion, "create_ingestion_pipeline", pipeline)
    return extractor


class DeterministicChat:
    calls = []

    def __init__(self, system_prompt=None):
        self.system_prompt = system_prompt

    async def generate(self, prompt, *, max_tokens, temperature):
        self.calls.append((self.system_prompt, prompt, max_tokens, temperature))
        return "Grounded answer"
