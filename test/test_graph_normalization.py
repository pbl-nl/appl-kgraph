import pytest

from graph_normalization import GraphView, normalize_graph
from schemas import Entity, ExtractionResult, GraphDelta, Relation


def test_normalize_graph_merges_duplicates_and_existing_state_deterministically():
    existing = GraphView(
        entities=[
            Entity(
                name="Alpha",
                type="Organization",
                description="Stored description",
                source_ids=("stored",),
                filepath="stored.txt",
            ),
            Entity(name="Beta", type="Person", description="Existing endpoint"),
        ],
        relations=[
            Relation(
                source_name="Alpha",
                target_name="Beta",
                description="Stored relation",
                keywords="stored",
                weight=4,
                source_ids=("stored",),
            )
        ],
    )
    extraction = ExtractionResult(
        entities=[
            Entity(
                name="Alpha",
                type="unknown",
                description="New description",
                source_ids=("chunk-1",),
                filepath="new.txt",
            ),
            Entity(
                name="Alpha",
                type="Person",
                description="New description",
                source_ids=("chunk-2",),
                filepath="new.txt",
            ),
            Entity(name="Alpha", type="Person", description="Third description"),
        ],
        relations=[
            Relation(
                source_name="Beta",
                target_name="Alpha",
                description="New relation",
                keywords="shared, new",
                weight=1,
                source_ids=("chunk-1",),
            ),
            Relation(
                source_name="Alpha",
                target_name="Beta",
                description="New relation",
                keywords="new",
                weight=2,
                source_ids=("chunk-2",),
            ),
        ],
    )

    delta = normalize_graph(extraction, existing)

    assert delta == GraphDelta(
        entities=[
            Entity(
                name="Alpha",
                type="Person",
                description=(
                    "Stored description||New description||Third description"
                ),
                source_ids=("stored", "chunk-1", "chunk-2"),
                filepath="stored.txt||new.txt",
            ),
            Entity(name="Beta", type="Person", description="Existing endpoint",
                   source_ids=("chunk-1", "chunk-2")),
        ],
        relations=[
            Relation(
                source_name="Alpha",
                target_name="Beta",
                description="Stored relation||New relation",
                keywords="stored||shared||new",
                weight=7,
                source_ids=("stored", "chunk-1", "chunk-2"),
            )
        ],
        metadata={"delimiter": "||"},
    )


def test_normalize_graph_creates_missing_relation_endpoints_only():
    extraction = ExtractionResult(
        relations=[
            Relation(source_name="Known", target_name="Missing", description="Link")
        ]
    )
    existing = GraphView(
        entities=[Entity(name="Known", type="Thing", description="Stored")]
    )

    delta = normalize_graph(extraction, existing)

    assert delta.entities == [
        Entity(name="Missing", type="unknown", description="")
    ]
    assert delta.relations[0].source_name == "Known"
    assert delta.relations[0].target_name == "Missing"


def test_normalize_graph_returns_empty_delta_for_empty_extraction():
    existing = GraphView(
        entities=[Entity(name="Unchanged", type="Thing", description="Stored")]
    )

    assert normalize_graph(ExtractionResult(), existing) == GraphDelta(
        metadata={"delimiter": "||"}
    )


def test_normalize_graph_does_not_mutate_inputs():
    entity = Entity(
        name="Alpha",
        type="Thing",
        description="Description",
        metadata={"tags": ["original"]},
    )
    extraction = ExtractionResult(entities=[entity])

    normalize_graph(extraction, GraphView())

    assert entity.metadata == {"tags": ["original"]}


@pytest.mark.parametrize(
    ("extraction", "existing", "message"),
    [
        ({}, GraphView(), "ExtractionResult"),
        (ExtractionResult(), {}, "GraphView"),
    ],
)
def test_normalize_graph_rejects_invalid_connector_inputs(
    extraction,
    existing,
    message,
):
    with pytest.raises(TypeError, match=message):
        normalize_graph(extraction, existing)


def test_relation_only_update_preserves_shared_endpoints_when_old_source_is_removed():
    from graph_normalization import remove_graph_sources
    existing = GraphView(entities=[
        Entity(name="Alpha", type="Organization", description="Original", source_ids=("old",), filepath="old.txt"),
        Entity(name="Beta", type="unknown", description="", source_ids=("old",), filepath="old.txt"),
    ])
    findings = ExtractionResult(relations=[Relation(
        source_name="Alpha", target_name="Beta", description="New relation", source_ids=("new",), filepath="new.txt")])
    upserts = normalize_graph(findings, existing)
    removed = remove_graph_sources(GraphView(upserts.entities, upserts.relations), ["old"], filepath="old.txt")
    assert removed.deleted_entity_names == removed.deleted_relation_pairs == ()
    assert {entity.name for entity in removed.entities} == {"Alpha", "Beta"}
    assert all(entity.source_ids == ("new",) and entity.filepath == "new.txt" for entity in removed.entities)
    assert upserts.relations[0].source_ids == ("new",)


@pytest.mark.parametrize("old_path, current_path", [
    (r"C:\old\ALPHA.txt", "/new/location/alpha.txt"),
    ("/old/location/alpha.txt", r"C:\new\alpha.txt"),
    (r"\\server\share\alpha.txt", ""),
])
def test_remove_graph_sources_prunes_moved_paths_on_shared_records(old_path, current_path):
    from graph_normalization import remove_graph_sources

    provenance = dict(source_ids=("old", "keep"), filepath=f"{old_path}||/new/beta.txt")
    existing = GraphView(
        entities=[Entity(name="Alpha", type="Thing", description="Shared", **provenance)],
        relations=[Relation(source_name="Alpha", target_name="Beta", description="Shared", **provenance)],
    )

    delta = remove_graph_sources(existing, ["old"], filepath=current_path, filename="alpha.txt")

    assert not delta.deleted_entity_names and not delta.deleted_relation_pairs
    for item in (*delta.entities, *delta.relations):
        assert item.source_ids == ("keep",)
        assert item.filepath == "/new/beta.txt"
    assert existing.entities[0].source_ids == ("old", "keep")
