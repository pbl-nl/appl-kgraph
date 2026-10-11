import asyncio

import pytest

from test.rag_fixtures import corpus as corpus_fixture
from test.rag_fixtures import configure_rag
import app
import lightrag
import pathrag


corpus = corpus_fixture
pytestmark = pytest.mark.storage_integration


@pytest.mark.parametrize("strategy", ["pathrag", "lightrag"])
def test_ui_callback_uses_real_typed_query_result(corpus, monkeypatch, strategy):
    module = pathrag if strategy == "pathrag" else lightrag
    configure_rag(module, monkeypatch)
    if strategy == "lightrag":

        async def keywords(*args, **kwargs):
            return ["partnership"], ["Alpha"]

        monkeypatch.setattr(lightrag, "extract_keywords", keywords)
    rag = (pathrag.PathRAG if strategy == "pathrag" else lightrag.LightRAG)(
        project_paths=corpus
    )
    monkeypatch.setattr(app, "_get_" + strategy, lambda folder: rag)
    callback = getattr(app, "create_" + strategy + "_response")
    try:
        cleared, history, sources = asyncio.run(
            callback(
                "Alpha",
                [],
                str(corpus.documents_root),
                "Use Existing Graph",
                [],
                "select documents from graph",
                ["a.txt"],
            )
        )
        assert cleared == ""
        assert history[-1]["content"] == "Grounded answer"
        assert "Evidence from a." in sources
        assert "Evidence from b." not in sources
        assert "error" not in sources.lower()
    finally:
        rag.close()


def test_cli_entrypoints_consume_typed_results_and_close(corpus, monkeypatch, capsys):
    import main

    configure_rag(pathrag, monkeypatch)
    configure_rag(lightrag, monkeypatch)

    async def keywords(*args, **kwargs):
        return ["partnership"], ["Alpha"]

    monkeypatch.setattr(lightrag, "extract_keywords", keywords)
    asyncio.run(
        main.ask_with_pathrag(
            "Alpha", documents_root=corpus.documents_root, verbose=True
        )
    )
    result = asyncio.run(
        main.ask_with_lightrag(
            "Alpha", documents_root=corpus.documents_root, verbose=True
        )
    )
    assert result.context.chunks
    assert "Grounded answer" in capsys.readouterr().out
