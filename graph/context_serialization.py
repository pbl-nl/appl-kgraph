"""Conversions at retrieval storage boundaries; prompts consume typed context."""

from schemas import Chunk, Entity, Relation, RetrievedContext


def context_from_candidates(candidates, windows=(), *, metadata=None):
    entities = [
        Entity(
            name=item.name,
            type=item.type or "unknown",
            description=item.description,
            source_ids=tuple(item.metadata.get("source_ids", ())),
            filepath=item.metadata.get("filepath"),
            metadata={**item.metadata, "score": item.score},
        )
        for item in candidates.entities
    ]
    relations = [
        Relation(
            source_name=item.source_name,
            target_name=item.target_name,
            description=item.description,
            keywords=item.keywords,
            source_ids=tuple(item.metadata.get("source_ids", ())),
            filepath=item.metadata.get("filepath"),
            metadata={**item.metadata, "score": item.score},
        )
        for item in candidates.relations
    ]
    chunks = [
        Chunk(
            chunk_uuid=item.chunk_uuid,
            doc_id=item.document_id,
            chunk_id=int(item.metadata.get("chunk_id", index)),
            filename=item.filename,
            text=item.text,
            char_count=len(item.text),
            start_page=int(item.metadata.get("start_page", 0)),
            end_page=int(item.metadata.get("end_page", 0)),
            filepath=item.metadata.get("filepath"),
            document_language=item.metadata.get("document_language"),
            metadata={**item.metadata, "score": item.score},
        )
        for index, item in enumerate(candidates.chunks)
    ]
    return RetrievedContext(
        candidates.query_plan,
        list(windows),
        entities,
        relations,
        chunks,
        metadata or {},
    )


def light_prompt_rows(context):
    entities = [
        {
            "id": str(index + 1),
            "entity": item.name,
            "type": item.type,
            "description": item.description,
            "file_path": item.filepath or "",
        }
        for index, item in enumerate(context.entities)
    ]
    relations = [
        {
            "id": str(index + 1),
            "entity1": item.source_name,
            "entity2": item.target_name,
            "type": item.metadata.get("type", ""),
            "description": item.description,
            "file_path": item.filepath or "",
        }
        for index, item in enumerate(context.relations)
    ]
    chunks = [
        {
            **item.metadata,
            "id": item.chunk_uuid,
            "chunk_uuid": item.chunk_uuid,
            "document_id": item.doc_id,
            "filename": item.filename,
            "text": item.text,
        }
        for item in context.chunks
    ]
    return entities, relations, chunks
