from __future__ import annotations

import html
import queue
import tempfile
import threading
from pathlib import Path
from typing import Any, Iterator, List, Optional, Tuple

import gradio as gr
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network

from ingestion import ingest_paths
from lightrag import LightRAG
from pathrag import PathRAG, StorageAdapter as PathStorageAdapter
from graph_pickle import load_graph_from_pickle, save_graph_to_pickle
from project_paths import (
    ProjectPaths,
    ensure_project_dirs,
    list_document_paths,
    resolve_project_paths,
)


mygraph = nx.Graph()
_PATHRAG_CACHE: dict[str, PathRAG] = {}
_LIGHTRAG_CACHE: dict[str, LightRAG] = {}


def generate_dynamic_type_colors(graph: nx.Graph) -> dict[str, str]:
    types = sorted(set(data.get("type", "Unknown") for _, data in graph.nodes(data=True)))
    if not types:
        return {}
    cmap = plt.get_cmap("tab20", len(types))
    return {node_type: mcolors.to_hex(cmap(index)) for index, node_type in enumerate(types)}


def generate_legend_html(graph: Optional[nx.Graph] = None) -> str:
    colors = generate_dynamic_type_colors(graph or nx.Graph())
    html = "<div style='padding:5px;'><b>Legend:</b><br>"
    for key, color in colors.items():
        html += "<div style='display:flex;align-items:center;margin:2px;'>"
        html += (
            f"<div style='width:20px;height:20px;background-color:{color};"
            "margin-right:5px;border:1px solid #fff;'></div>"
            f"{key}</div>"
        )
    html += "</div>"
    return html


def _project_paths_for_folder(folder_path: str) -> Optional[ProjectPaths]:
    if not folder_path:
        return None
    return resolve_project_paths(folder_path)


def _load_graph_from_storage(folder_path: str) -> nx.Graph:
    if not folder_path:
        return nx.Graph()
    project_paths = resolve_project_paths(folder_path)
    if not Path(project_paths.storage.graph_db).exists():
        return nx.Graph()
    adapter = PathStorageAdapter(paths=project_paths.storage)
    return adapter.graph.copy()


def _load_graph_from_pickle(folder_path: str) -> nx.Graph:
    project_paths = _project_paths_for_folder(folder_path)
    if project_paths is None or not project_paths.graph_pickle_file.exists():
        return nx.Graph()
    return load_graph_from_pickle(project_paths.graph_pickle_file)


def _save_graph_pickle(folder_path: str, graph: nx.Graph) -> Optional[Path]:
    project_paths = _project_paths_for_folder(folder_path)
    if project_paths is None:
        return None
    ensure_project_dirs(project_paths)
    return save_graph_to_pickle(graph, project_paths.graph_pickle_file)


def render_graph_iframe(graph: nx.Graph, height_px: int = 650) -> str:
    type_colors = generate_dynamic_type_colors(graph)
    net = Network(height=f"{height_px}px", width="100%", directed=False, bgcolor="#111111", font_color="white")
    for node, data in graph.nodes(data=True):
        node_type = data.get("type", "unknown")
        title = f"type={node_type}\n{data.get('description', '')}"
        color = type_colors.get(node_type, "#0EA5E9")
        label = str(data.get("label", node))
        net.add_node(str(node), label=label, title=title, color=color)

    for source, target, data in graph.edges(data=True):
        title = data.get("description", "") or data.get("keywords", "")
        net.add_edge(str(source), str(target), title=title, value=float(data.get("weight", 1.0) or 1.0))

    net.set_options(
        """
        var options = {
          "nodes": { "shape": "dot", "size": 14, "font": { "strokeWidth": 8 } },
          "edges": { "smooth": false, "color": { "color": "#94A3B8" } },
          "physics": { "enabled": false },
          "interaction": { "hover": true, "navigationButtons": true }
        }
        """
    )

    tmp_path = Path(tempfile.gettempdir()) / "appl_kgraph_graph.html"
    net.save_graph(str(tmp_path))
    rendered = tmp_path.read_text(encoding="utf-8")
    return (
        f'<iframe srcdoc="{html.escape(rendered, quote=True)}" '
        f'style="width:100%;height:{height_px}px;border:none;border-radius:8px;" '
        'sandbox="allow-scripts allow-same-origin"></iframe>'
    )


def _get_pathrag(folder_path: str) -> PathRAG:
    rag = _PATHRAG_CACHE.get(folder_path)
    if rag is None:
        rag = PathRAG(project_paths=resolve_project_paths(folder_path), system_prompt="")
        _PATHRAG_CACHE[folder_path] = rag
    return rag


def _get_lightrag(folder_path: str) -> LightRAG:
    rag = _LIGHTRAG_CACHE.get(folder_path)
    if rag is None:
        rag = LightRAG(project_paths=resolve_project_paths(folder_path), system_prompt="")
        _LIGHTRAG_CACHE[folder_path] = rag
    return rag


def _history_to_turns(chat_history: List[dict]) -> List[Tuple[str, str]]:
    turns: List[Tuple[str, str]] = []
    for message in chat_history or []:
        if isinstance(message, dict):
            role = message.get("role", "")
            content = message.get("content", "")
        else:
            role = getattr(message, "role", "")
            content = getattr(message, "content", "")
        if role and content:
            turns.append((role, content))
    return turns


def _dropdown_choices() -> List[Tuple[str, str]]:
    choices: List[Tuple[str, str]] = []
    for node, data in mygraph.nodes(data=True):
        label = data.get("label", node)
        choices.append((f"{label} ({node})", str(node)))
    return sorted(choices, key=lambda item: item[0].lower())


def update_dropdowns():
    choices = _dropdown_choices()
    return [gr.update(choices=choices, value=None) for _ in range(3)]


def _ingestion_payload(
    graph: nx.Graph,
    status: str,
    active_folder_value: str,
) -> Tuple[str, str, str, str, Any, Any, Any]:
    updates = update_dropdowns()
    return render_graph_iframe(graph), generate_legend_html(graph), status, active_folder_value, *updates


def handle_ingestion(folder_path: str) -> Iterator[Tuple[str, str, str, str, Any, Any, Any]]:
    global mygraph

    if not folder_path or not Path(folder_path).is_dir():
        yield _ingestion_payload(mygraph, "Please provide a valid folder path.", "")
        return

    documents_root = Path(folder_path).expanduser().resolve()
    paths = list_document_paths(documents_root)
    if not paths:
        empty_graph = nx.Graph()
        yield _ingestion_payload(
            empty_graph,
            "No supported files found in the selected folder.",
            str(documents_root),
        )
        return

    progress_messages = [
        f"Preparing ingestion for {documents_root}",
        f"Discovered {len(paths)} supported files",
    ]
    yield _ingestion_payload(mygraph, "\n".join(progress_messages), str(documents_root))

    progress_queue: queue.Queue[str] = queue.Queue()
    outcome: dict[str, Any] = {}
    error: dict[str, Exception] = {}

    def _report_progress(message: str) -> None:
        progress_queue.put(message)

    def _run_ingestion() -> None:
        try:
            outcome["summary"] = ingest_paths(
                paths,
                documents_root=documents_root,
                progress_callback=_report_progress,
            )
        except Exception as exc:  # pragma: no cover - UI integration guard
            error["exception"] = exc
        finally:
            progress_queue.put("__DONE__")

    worker = threading.Thread(target=_run_ingestion, daemon=True)
    worker.start()

    while True:
        try:
            message = progress_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        if message == "__DONE__":
            break

        progress_messages.append(message)
        yield _ingestion_payload(mygraph, "\n".join(progress_messages[-50:]), str(documents_root))

    if "exception" in error:
        progress_messages.append(f"Error: {error['exception']}")
        yield _ingestion_payload(mygraph, "\n".join(progress_messages[-50:]), str(documents_root))
        return

    summary = outcome["summary"]
    mygraph = _load_graph_from_storage(str(documents_root))
    _PATHRAG_CACHE.pop(str(documents_root), None)
    _LIGHTRAG_CACHE.pop(str(documents_root), None)

    pickle_note = ""
    project_paths = resolve_project_paths(documents_root)
    if not project_paths.graph_pickle_file.exists():
        saved_path = _save_graph_pickle(str(documents_root), mygraph)
        if saved_path is not None:
            pickle_note = f"\nSaved baseline graph pickle to {saved_path}"
    else:
        pickle_note = f"\nExisting working graph pickle preserved at {project_paths.graph_pickle_file}"

    status = (
        f"Ingested project at {documents_root}\n"
        f"Processed files: {summary['processed_files']}\n"
        f"Skipped files: {summary['skipped_files']}\n"
        f"Removed files: {summary['removed_files']}\n"
        f"Chunks: {summary['chunk_count']}\n"
        f"Entities: {summary['entity_count']}\n"
        f"Relations: {summary['relation_count']}"
        f"\nRetrieval snapshot: {project_paths.retrieval_graph_pickle_file}"
        f"{pickle_note}\n\n"
        f"{chr(10).join(progress_messages[-20:])}"
    )
    yield _ingestion_payload(mygraph, status, str(documents_root))


def save_current_graph(folder_path: str) -> str:
    if not folder_path:
        return "Select and ingest a document folder first."
    saved_path = _save_graph_pickle(folder_path, mygraph)
    if saved_path is None:
        return "Unable to determine a project path for the current graph."
    return f"Saved working graph pickle to {saved_path}"


def load_saved_graph(folder_path: str) -> Tuple[str, str, str, Any, Any, Any]:
    global mygraph

    if not folder_path:
        updates = update_dropdowns()
        return render_graph_iframe(mygraph), generate_legend_html(mygraph), "Select and ingest a document folder first.", *updates

    project_paths = resolve_project_paths(folder_path)
    if not project_paths.graph_pickle_file.exists():
        updates = update_dropdowns()
        return (
            render_graph_iframe(mygraph),
            generate_legend_html(mygraph),
            f"No saved graph pickle found yet at {project_paths.graph_pickle_file}",
            *updates,
        )

    try:
        mygraph = _load_graph_from_pickle(folder_path)
        message = f"Loaded working graph pickle from {project_paths.graph_pickle_file}"
    except Exception as exc:
        updates = update_dropdowns()
        return render_graph_iframe(mygraph), generate_legend_html(mygraph), f"Failed to load saved graph pickle: {exc}", *updates

    updates = update_dropdowns()
    return render_graph_iframe(mygraph), generate_legend_html(mygraph), message, *updates


def merge_nodes(
    node1: str,
    node2: str,
    active_folder: str,
) -> Tuple[str, str, str, Any, Any, Any]:
    global mygraph

    if not node1 or not node2:
        updates = update_dropdowns()
        return render_graph_iframe(mygraph), generate_legend_html(mygraph), "Select two nodes to merge.", *updates
    if node1 not in mygraph or node2 not in mygraph:
        updates = update_dropdowns()
        return render_graph_iframe(mygraph), generate_legend_html(mygraph), "Both nodes must exist in the current graph.", *updates
    if node1 == node2:
        updates = update_dropdowns()
        return render_graph_iframe(mygraph), generate_legend_html(mygraph), "Cannot merge the same node into itself.", *updates

    new_node = f"{node1}_{node2}"
    suffix = 1
    while new_node in mygraph:
        suffix += 1
        new_node = f"{node1}_{node2}_{suffix}"

    label1 = mygraph.nodes[node1].get("label", node1)
    label2 = mygraph.nodes[node2].get("label", node2)
    mygraph.add_node(
        new_node,
        label=f"{label1} + {label2}",
        type="Merged",
        description=f"Merged from {node1} and {node2}",
        source="Merged",
    )

    for original in (node1, node2):
        for neighbor, attrs in list(mygraph[original].items()):
            if neighbor != new_node and neighbor not in (node1, node2):
                mygraph.add_edge(new_node, neighbor, **attrs)
        mygraph.remove_node(original)

    autosave = ""
    saved_path = _save_graph_pickle(active_folder, mygraph)
    if saved_path is not None:
        autosave = f" Auto-saved to {saved_path}."

    updates = update_dropdowns()
    return render_graph_iframe(mygraph), generate_legend_html(mygraph), f"Merged '{node1}' and '{node2}' into '{new_node}'.{autosave}", *updates


def update_node_attributes(
    node_id: str,
    new_label: str,
    new_type: str,
    new_desc: str,
    new_source: str,
    active_folder: str,
) -> Tuple[str, str, str, Any, Any, Any]:
    global mygraph

    if not node_id:
        updates = update_dropdowns()
        return render_graph_iframe(mygraph), generate_legend_html(mygraph), "Select a node to update.", *updates
    if node_id not in mygraph:
        updates = update_dropdowns()
        return render_graph_iframe(mygraph), generate_legend_html(mygraph), f"Node '{node_id}' was not found in the current graph.", *updates

    if new_label:
        mygraph.nodes[node_id]["label"] = new_label
    if new_type:
        mygraph.nodes[node_id]["type"] = new_type
    if new_desc:
        mygraph.nodes[node_id]["description"] = new_desc
    if new_source:
        mygraph.nodes[node_id]["source"] = new_source

    autosave = ""
    saved_path = _save_graph_pickle(active_folder, mygraph)
    if saved_path is not None:
        autosave = f" Auto-saved to {saved_path}."

    updates = update_dropdowns()
    return render_graph_iframe(mygraph), generate_legend_html(mygraph), f"Updated node '{node_id}'.{autosave}", *updates


async def create_pathrag_response(
    question: str,
    chat_history: List[dict],
    active_folder: str,
) -> Tuple[str, List[dict], str]:
    if not active_folder:
        chat_history = list(chat_history or [])
        chat_history.append({"role": "assistant", "content": "Select and ingest a document folder first."})
        return "", chat_history, ""

    history = list(chat_history or [])
    history.append({"role": "user", "content": question})
    try:
        rag = _get_pathrag(active_folder)
        result = await rag.aretrieve(question, conversation_history=_history_to_turns(history[:-1]))
        history.append({"role": "assistant", "content": result.answer})

        sources = []
        for index, chunk in enumerate(result.chunk_matches, start=1):
            head = chunk.filename or chunk.document_id or "(unknown doc)"
            sources.append(f"{index}. {head} (score={chunk.score:.3f})")
            sources.append(chunk.text)
            sources.append("-" * 46)

        return "", history, "\n".join(sources)
    except Exception as exc:
        history.append({"role": "assistant", "content": f"PathRAG error: {exc}"})
        return "", history, f"PathRAG error: {exc}"


async def create_lightrag_response(
    question: str,
    chat_history: List[dict],
    active_folder: str,
) -> Tuple[str, List[dict], str]:
    if not active_folder:
        chat_history = list(chat_history or [])
        chat_history.append({"role": "assistant", "content": "Select and ingest a document folder first."})
        return "", chat_history, ""

    history = list(chat_history or [])
    history.append({"role": "user", "content": question})
    try:
        rag = _get_lightrag(active_folder)
        result = await rag.aretrieve(question, conversation_history=_history_to_turns(history[:-1]))
        history.append({"role": "assistant", "content": result.answer})

        sources = []
        for index, chunk in enumerate(result.all_chunks, start=1):
            source_type = chunk.get("source_type", "unknown")
            line = f"{index}. {source_type}"
            if source_type == "vector" and chunk.get("score") is not None:
                line += f" (score={float(chunk['score']):.3f})"
            sources.append(line)
            sources.append(chunk.get("text", ""))
            sources.append("-" * 46)

        return "", history, "\n".join(sources)
    except Exception as exc:
        history.append({"role": "assistant", "content": f"LightRAG error: {exc}"})
        return "", history, f"LightRAG error: {exc}"


with gr.Blocks() as demo:
    gr.Markdown("## Interactive Hybrid RAG")

    active_folder = gr.State("")

    with gr.Sidebar():
        folder_path_input = gr.Textbox(label="Document folder", placeholder="C:\\path\\to\\documents")
        go_btn = gr.Button(value="Ingest Folder", variant="primary")
        status_messages = gr.Textbox(label="Status", interactive=False, lines=16)

    with gr.Tabs():
        with gr.Tab("Chat"):
            with gr.Tabs():
                with gr.Tab("PathRAG"):
                    pathrag_chatbot = gr.Chatbot(type="messages", label="PathRAG Chat History", height=420)
                    pathrag_sources = gr.Textbox(label="PathRAG sources", interactive=False, lines=14)
                    with gr.Row():
                        pathrag_msg_input = gr.Textbox(show_label=False, placeholder="Ask a question about the active project...")
                        pathrag_clear_btn = gr.ClearButton(
                            components=[pathrag_msg_input, pathrag_chatbot, pathrag_sources],
                            value="Clear conversation",
                        )

                with gr.Tab("LightRAG"):
                    lightrag_chatbot = gr.Chatbot(type="messages", label="LightRAG Chat History", height=420)
                    lightrag_sources = gr.Textbox(label="LightRAG sources", interactive=False, lines=14)
                    with gr.Row():
                        lightrag_msg_input = gr.Textbox(show_label=False, placeholder="Ask a question about the active project...")
                        lightrag_clear_btn = gr.ClearButton(
                            components=[lightrag_msg_input, lightrag_chatbot, lightrag_sources],
                            value="Clear conversation",
                        )

        with gr.Tab("Knowledge Graph"):
            with gr.Row():
                with gr.Column(scale=9):
                    graph_html = gr.HTML(render_graph_iframe(mygraph))
                with gr.Column(scale=1):
                    legend_html = gr.HTML(generate_legend_html(mygraph))
            with gr.Row():
                save_pickle_btn = gr.Button(value="Save Working Graph", variant="primary")
                load_pickle_btn = gr.Button(value="Load Saved Graph")
            with gr.Tabs():
                with gr.Tab("Edit Node"):
                    edit_node_dropdown = gr.Dropdown(choices=[], label="Select Node")
                    edit_label = gr.Textbox(label="Label")
                    edit_type = gr.Textbox(label="Type")
                    edit_desc = gr.Textbox(label="Description")
                    edit_source = gr.Textbox(label="Source")
                    updatenode_btn = gr.Button(value="Update Node", variant="primary")
                with gr.Tab("Merge Nodes"):
                    m1 = gr.Dropdown(choices=[], label="Node 1")
                    m2 = gr.Dropdown(choices=[], label="Node 2")
                    mergenodes_btn = gr.Button(value="Merge Nodes", variant="primary")

    go_btn.click(
        fn=handle_ingestion,
        inputs=[folder_path_input],
        outputs=[graph_html, legend_html, status_messages, active_folder, m1, m2, edit_node_dropdown],
    )

    save_pickle_btn.click(
        fn=save_current_graph,
        inputs=[active_folder],
        outputs=[status_messages],
    )

    load_pickle_btn.click(
        fn=load_saved_graph,
        inputs=[active_folder],
        outputs=[graph_html, legend_html, status_messages, m1, m2, edit_node_dropdown],
    )

    updatenode_btn.click(
        fn=update_node_attributes,
        inputs=[edit_node_dropdown, edit_label, edit_type, edit_desc, edit_source, active_folder],
        outputs=[graph_html, legend_html, status_messages, m1, m2, edit_node_dropdown],
    )

    mergenodes_btn.click(
        fn=merge_nodes,
        inputs=[m1, m2, active_folder],
        outputs=[graph_html, legend_html, status_messages, m1, m2, edit_node_dropdown],
    )

    pathrag_msg_input.submit(
        fn=create_pathrag_response,
        inputs=[pathrag_msg_input, pathrag_chatbot, active_folder],
        outputs=[pathrag_msg_input, pathrag_chatbot, pathrag_sources],
    )
    lightrag_msg_input.submit(
        fn=create_lightrag_response,
        inputs=[lightrag_msg_input, lightrag_chatbot, active_folder],
        outputs=[lightrag_msg_input, lightrag_chatbot, lightrag_sources],
    )

    pathrag_clear_btn.click(
        fn=lambda: [None, None, None],
        inputs=[],
        outputs=[pathrag_msg_input, pathrag_chatbot, pathrag_sources],
        queue=False,
    )
    lightrag_clear_btn.click(
        fn=lambda: [None, None, None],
        inputs=[],
        outputs=[lightrag_msg_input, lightrag_chatbot, lightrag_sources],
        queue=False,
    )

    demo.load(
        fn=update_dropdowns,
        outputs=[m1, m2, edit_node_dropdown],
    )


if __name__ == "__main__":
    demo.queue()
    demo.launch(inbrowser=True, pwa=True)
