import lightrag


def test_evidence_lookup_respects_per_entity_chunk_limit():
    class Adapter:
        graph = __import__("networkx").Graph()

        def get_chunk_by_uuid(self, uuid):
            return {
                "chunk_uuid": uuid,
                "doc_id": "doc",
                "filename": "a.txt",
                "text": uuid,
            }

    nodes = [{"name": "Alpha", "chunk_uuids": ["one", "two", "three"]}]
    assert len(lightrag.extract_chunks_from_nodes(Adapter(), nodes, 2)) == 2
    assert len(lightrag.extract_chunks_from_nodes(Adapter(), nodes, 0)) == 0
