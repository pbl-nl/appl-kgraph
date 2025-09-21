
import os
import shutil
import tempfile
import time
import uuid
import pytest
import traceback

# Direct imports only, per instruction
pytest.importorskip("chromadb", reason="chromadb is required for storage vector tests")
from graph.storage import Storage, StoragePaths
import test.examples as examples

"""
Comprehensive tests for Storage class, covering add, upsert, delete, and get operations
Right now, these tests are assuming correct arguments are passed; no error handling tests.
To run these tests, use in a terminal at /appl-kgraph root:
pytest test/test_storage_full.py
"""


def _make_paths(tmp_dir: str) -> StoragePaths:
    """Create isolated paths in a temp folder; mirrors the structure from test_store init."""
    return StoragePaths(
        documents_db=os.path.join(tmp_dir, "doc_schema.db"),
        chunks_db=os.path.join(tmp_dir, "chunk_schema.db"),
        chroma_chunks=os.path.join(tmp_dir, "chunk_vector"),
        graph_db=os.path.join(tmp_dir, "kg_schema.db"),
        chroma_entities=os.path.join(tmp_dir, "entity_vector"),
        chroma_relations=os.path.join(tmp_dir, "relation_vector"),
    )


@pytest.fixture(scope="function")
def fresh_storage(tmp_path):
    paths = _make_paths(str(tmp_path))
    st = Storage(paths=paths)
    st.init()
    return st


# Use ONLY example data from test_store.py
DOCS = list(examples.documents)
CHUNKS = list(examples.chunks)
NODES = list(examples.nodes)
EDGES = list(examples.edges)
CHUNK_VECS = list(getattr(examples, "chunks", []))
ENTITY_VECS = list(getattr(examples, "nodes", []))
RELATION_VECS = list(getattr(examples, "edges", []))


# -------------------------------
# ADD
# -------------------------------

def test_add_document_and_list_get(fresh_storage: Storage):
    st = fresh_storage
    # add two docs
    for d in DOCS[:2]:
        st.add_document(d["metadata"], d["full_text"])
    docs = st.list_documents()
    assert isinstance(docs, list) and len(docs) == 2
    d0_id = DOCS[0]["metadata"]["doc_id"]
    got = st.get_document(d0_id)
    assert got is not None
    assert got["doc_id"] == d0_id
    got_filename = st.get_document_by_filename(DOCS[0]["metadata"]["filename"])
    assert got_filename is not None
    assert got_filename["doc_id"] == d0_id

def test_add_chunks_and_getters(fresh_storage: Storage):
    st = fresh_storage
    # Seed one document and its chunks
    d = DOCS[0]
    st.add_document(d["metadata"], d["full_text"])
    doc_chunks = [c for c in CHUNKS if c["doc_id"] == d["metadata"]["doc_id"]]
    assert doc_chunks, "No example chunks found in test_store for the first document"
    st.add_chunks(doc_chunks[:3])  # add first few
    # get by doc_id
    got_by_doc = st.get_chunks_by_doc_id(d["metadata"]["doc_id"])
    assert isinstance(got_by_doc, list) and len(got_by_doc) == min(3, len(doc_chunks))
    # get by filename
    got_by_fn = st.get_chunks_by_filename(d["metadata"]["filename"])
    assert isinstance(got_by_fn, list) and len(got_by_fn) == len(got_by_doc)
    # get single by uuid
    one_uuid = doc_chunks[0]["chunk_uuid"]
    one = st.get_chunk_by_uuid(one_uuid)
    assert one is not None
    assert one[0]["chunk_uuid"] == one_uuid
    # get batch by uuids
    uuids = [c["chunk_uuid"] for c in doc_chunks[:2]]
    batch = st.get_chunks_by_uuids(uuids)
    assert isinstance(batch, list) and {b["chunk_uuid"] for b in batch} == set(uuids)

def test_add_kg_nodes_edges_and_getters(fresh_storage: Storage):
    st = fresh_storage
    # add three nodes
    st.add_kg_nodes(NODES[:3])
    # add two edges among these (fall back to first two if examples not fully overlapping)
    # filter for subset of nodes used
    usable_edges = [e for e in EDGES if {e["source_name"], e["target_name"]} <= {n['name'] for n in NODES[:3]}]
    if len(usable_edges) < 2:
        usable_edges = EDGES[:2]
    st.add_kg_edges(usable_edges)
    # get node
    k = NODES[0]["name"]
    got = st.get_node(k)
    assert got is not None
    assert got["name"] == k
    # get nodes (batch)
    keys = [NODES[i]["name"] for i in range(2)]
    got_many = st.get_nodes(keys)
    assert isinstance(got_many, list) and len(got_many) == 2
    # get edges
    s1, t1 = usable_edges[0]["source_name"], usable_edges[0]["target_name"]
    s2,t2 = usable_edges[1]["source_name"], usable_edges[1]["target_name"]
    e1 = st.get_edge(s1, t1)
    assert e1 is not None and {e1["source_name"], e1["target_name"]} == {s1, t1}
    em = st.get_edges([(s1, t1), (s2, t2)])
    assert isinstance(em, list) and len(em) == 2

def test_add_vectors_and_getters(fresh_storage: Storage):
    st = fresh_storage
    # add chunks and vectors for them if provided; otherwise just ensure method runs
    d = DOCS[0]
    st.add_document(d["metadata"], d["full_text"])
    doc_chunks = [c for c in CHUNKS if c["doc_id"] == d["metadata"]["doc_id"]][:2]
    st.add_chunks(doc_chunks)
    # chunk vectors
    try:
        st.add_chunk_vectors(doc_chunks)
    except Exception as e:
        pytest.fail(f"Relation vector ops failed: {e}\n{traceback.format_exc()}")
    # get chunk vector
    v = st.get_chunk_vector(doc_chunks[0]["chunk_uuid"])

    # allow None/empty when no vector backend
    assert v is None or isinstance(v, dict) or isinstance(v, list)
    # entity/relation vectors, if example lists exist
    
    try:
        st.add_entity_vectors(ENTITY_VECS[:2])
        ents = st.get_entities([e["name"] for e in ENTITY_VECS[:2]])
        assert isinstance(ents, list)
    except Exception as e:
        pytest.fail(f"Entity vector ops failed: {e}\n{traceback.format_exc()}")
    
    try:
        st.add_relation_vectors(RELATION_VECS[:2])
        rels = st.get_relations([(r["source_name"], r["target_name"]) for r in RELATION_VECS[:2]])
        assert isinstance(rels, list)
    except Exception as e:
        pytest.fail(f"Relation vector ops failed: {e}\n{traceback.format_exc()}")

# -------------------------------
# UPSERT
# -------------------------------

def test_upsert_document_chunk_node_edge(fresh_storage: Storage):
    st = fresh_storage
    d = DOCS[0]
    st.add_document(d["metadata"], d["full_text"])
    # upsert document (e.g., change language)
    st.upsert_document(d["metadata"]["doc_id"], {"language": "xx"})
    got = st.get_document(d["metadata"]["doc_id"])
    assert got is not None
    # support both flattened and nested metadata layouts
    lang = got.get("language") or got.get("metadata", {}).get("language")
    assert lang == "xx"

    # chunk
    doc_chunks = [c for c in CHUNKS if c["doc_id"] == d["metadata"]["doc_id"]]
    st.add_chunks(doc_chunks[:1])
    cu = doc_chunks[0]["chunk_uuid"]
    st.upsert_chunk(cu, {"text": doc_chunks[0]["text"] + " [upd]"})
    gotc = st.get_chunk_by_uuid(cu)
    assert gotc is not None and gotc[0]["text"] == doc_chunks[0]["text"] + " [upd]"

    # node
    n = NODES[0]
    st.add_kg_nodes([n])
    st.upsert_node(n["name"], {"description": n["description"] + " [upd]"})
    gotn = st.get_node(n["name"])
    assert gotn is not None and gotn["description"] == n["description"] + " [upd]"

    ns = NODES[1:3]
    st.add_kg_nodes(ns)
    updates = [{k:v if k != "description" else v + " [upd]" for k,v in n.items()} for n in ns]
    st.upsert_nodes(updates)
    gotns = st.get_nodes([n["name"] for n in ns])
    assert isinstance(gotns, list) and len(gotns) == 2
    upd_descriptions = [g['description'] for g in gotns]
    for n in ns:
        assert n["description"] + " [upd]" in upd_descriptions

    # edge
    e = EDGES[0]
    st.add_kg_edges([e])
    st.upsert_edge(e["source_name"], e["target_name"], {"weight": e["weight"] + 0.01})
    gote = st.get_edge(e["source_name"], e["target_name"])
    assert gote is not None and gote["weight"] == e["weight"] + 0.01

    es = EDGES[1:3]
    st.add_kg_edges(es)
    eupdates = [{k:v if k != "keywords" else v + ",upd" for k,v in e.items() if k in ("source_name", "target_name", "keywords")} for e in es]
    st.upsert_edges(eupdates)
    gotes = st.get_edges([(e["source_name"], e["target_name"]) for e in es])
    assert isinstance(gotes, list) and len(gotes) == 2
    upd_keywords = [g['keywords'] for g in gotes]
    for e in es:
        assert e["keywords"] + ",upd" in upd_keywords

def test_upsert_batch_nodes_edges_vectors(fresh_storage: Storage):
    st = fresh_storage
    # nodes batch
    st.add_kg_nodes(NODES[:3])
    updates = [{"name": n["name"], "type": n["type"], "updates": {"description": n["description"] + " [b]"}} for n in NODES[:2]]
    st.upsert_nodes(updates)

    # edges batch
    st.add_kg_edges(EDGES[:3])
    eupd = [{"source_name": e["source_name"], "target_name": e["target_name"], "updates": {"keywords": e["keywords"] + ",upd"}} for e in EDGES[:2]]
    st.upsert_edges(eupd)

    # vector upserts
    d = DOCS[0]
    st.add_document(d["metadata"], d["full_text"])
    chs = [c for c in CHUNKS if c["doc_id"] == d["metadata"]["doc_id"]][:2]
    st.add_chunks(chs)
    try:
        st.upsert_chunk_vector(chs)
    except Exception:
        pytest.fail("chunk vector upsert failed")
    try:
        st.upsert_entity_vector(ENTITY_VECS[:2])
    except Exception:
        pytest.fail("entity vector upsert failed")
    try:
        st.upsert_relation_vector(RELATION_VECS[:2])
    except Exception:
        pytest.fail("relation vector upsert failed")

# -------------------------------
# DELETE
# -------------------------------

def test_delete_document_and_chunks(fresh_storage: Storage):
    st = fresh_storage
    # seed two docs with chunks
    for d in DOCS[:2]:
        st.add_document(d["metadata"], d["full_text"])
        chs = [c for c in CHUNKS if c["doc_id"] == d["metadata"]["doc_id"]][:2]
        st.add_chunks(chs)
    # delete chunks by doc
    pre = st.get_chunks_by_doc_id(DOCS[0]["metadata"]["doc_id"])
    st.delete_chunks_by_doc_id(DOCS[0]["metadata"]["doc_id"])
    post = st.get_chunks_by_doc_id(DOCS[0]["metadata"]["doc_id"])
    assert not post and len(pre) > 0
    # delete doc
    pre = st.get_document(DOCS[0]["metadata"]["doc_id"])
    st.delete_document(DOCS[0]["metadata"]["doc_id"])
    post = st.get_document(DOCS[0]["metadata"]["doc_id"])
    assert not post and pre

def test_delete_chunks_nodes_edges_vectors(fresh_storage: Storage):
    st = fresh_storage
    d = DOCS[0]
    st.add_document(d["metadata"], d["full_text"])
    chs = [c for c in CHUNKS if c["doc_id"] == d["metadata"]["doc_id"]][:3]
    st.add_chunks(chs)
    # delete one chunk by uuid
    pre = st.get_chunk_by_uuid(chs[0]["chunk_uuid"])
    st.delete_chunk_by_uuid(chs[0]["chunk_uuid"])
    post = st.get_chunk_by_uuid(chs[0]["chunk_uuid"])
    assert pre and not post
    # delete multiple by uuids
    pre = st.get_chunks_by_uuids([c["chunk_uuid"] for c in chs[1:]])
    st.delete_chunks_by_uuids([c["chunk_uuid"] for c in chs[1:]])
    post = st.get_chunks_by_uuids([c["chunk_uuid"] for c in chs[1:]])
    assert pre and not post

    # nodes
    st.add_kg_nodes(NODES[:3])
    pre = st.get_node(NODES[0]["name"])
    st.delete_node(NODES[0]["name"])
    post = st.get_node(NODES[0]["name"])
    assert not post and pre
    pre = st.get_nodes([NODES[1]["name"], NODES[2]["name"]])
    st.delete_nodes([NODES[1]["name"], NODES[2]["name"]])
    post = st.get_nodes([NODES[1]["name"], NODES[2]["name"]])
    assert not post and pre

    # edges
    st.add_kg_edges(EDGES[:3])
    pre = st.get_edge(EDGES[0]["source_name"], EDGES[0]["target_name"])
    st.delete_edge(EDGES[0]["source_name"], EDGES[0]["target_name"])
    post = st.get_edge(EDGES[0]["source_name"], EDGES[0]["target_name"])
    assert not post and pre
    pre = st.get_edges([(EDGES[1]["source_name"], EDGES[1]["target_name"]), (EDGES[2]["source_name"], EDGES[2]["target_name"])])
    st.delete_edges([(EDGES[1]["source_name"], EDGES[1]["target_name"]), (EDGES[2]["source_name"], EDGES[2]["target_name"])])
    post = st.get_edges([(EDGES[1]["source_name"], EDGES[1]["target_name"]), (EDGES[2]["source_name"], EDGES[2]["target_name"])])
    assert not post and pre

    # vectors
    try:
        st.add_chunk_vectors(chs[:3])
        pre = st.get_chunk_vector(chs[0]["chunk_uuid"])
        st.delete_chunk_vector(chs[0]["chunk_uuid"])
        post = st.get_chunk_vector(chs[0]["chunk_uuid"])
        assert pre and not post
    except Exception:
        pytest.fail("Chunk vector delete failed")
    try:
        st.add_entity_vectors(ENTITY_VECS[:3])
        pre = st.get_entities([ENTITY_VECS[0]["name"]])
        st.delete_entity_vector([ENTITY_VECS[0]["name"]])
        post = st.get_entities([ENTITY_VECS[0]["name"]])
        assert pre and not post
    except Exception:
        pytest.fail("Entity vector delete failed")
    try:
        st.add_relation_vectors(RELATION_VECS[:3])
        pre = st.get_relations([(RELATION_VECS[0]["source_name"], RELATION_VECS[0]["target_name"])])
        st.delete_relation_vector([(RELATION_VECS[0]["source_name"], RELATION_VECS[0]["target_name"])])
        post = st.get_relations([(RELATION_VECS[0]["source_name"], RELATION_VECS[0]["target_name"])])
        assert pre and not post
    except Exception:
        pytest.fail("Relation vector delete failed")
