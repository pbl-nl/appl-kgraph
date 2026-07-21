from __future__ import annotations

import html
import queue
import tempfile
import threading
from pathlib import Path
from typing import Any, Iterator, List, Optional, Tuple

import gradio as gr
import textwrap
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network

from ingestion import ingest_paths, remove_document_from_storage
from db_storage import Storage
from existing_graphs import (
    discover_existing_graph_choices as _discover_existing_graph_choices,
    stored_document_names as _stored_document_names,
    stored_document_names_from_database as _stored_document_names_from_database,
)
from lightrag import LightRAG
from pathrag import PathRAG, StorageAdapter as PathStorageAdapter
from settings import settings
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
_DOCS_ROOT_DIRNAME = settings.project.documents_root_dirname
_DOCS_ROOT = Path(__file__).resolve().parents[1] / _DOCS_ROOT_DIRNAME
_GRAPH_PANEL_HEIGHT_PX = 650
_GRAPH_FILTER_TEXT = ""
_GRAPH_FILTER_MODE = "contains"
_APP_CSS = """
#graph-source-inline .wrap,
#graph-filter-match-inline .wrap {
    display: flex !important;
    flex-direction: row !important;
    align-items: center;
    gap: 8px;
    flex-wrap: nowrap !important;
}

#graph-source-inline .wrap > label,
#graph-filter-match-inline .wrap > label {
    display: inline-flex !important;
    align-items: center;
    margin: 0;
}

#existing-graph-doc-filter,
#existing-graph-doc-delete,
#ingest-file-filter {
    overflow: hidden;
}

#existing-graph-doc-filter .wrap,
#existing-graph-doc-delete .wrap,
#ingest-file-filter .wrap {
    display: block !important;
    max-height: 200px;
    overflow-y: auto !important;
    padding-right: 6px;
}
"""


def generate_dynamic_type_colors(graph):
    types = sorted(set(data.get("type", "Unknown") for _, data in graph.nodes(data=True)))
    if not types:
        return {}
    cmap = plt.get_cmap("tab20", len(types))
    return {node_type: mcolors.to_hex(cmap(index)) for index, node_type in enumerate(types)}


def generate_legend_html(graph: Optional[nx.Graph] = None) -> str:
    colors = generate_dynamic_type_colors(graph or nx.Graph())
    html = "<div style='padding:5px;'><b>Legend:</b><br>"
    for key, color in colors.items():
        html += (
            "<div style='display:flex;align-items:center;margin:2px;'>"
            f"<div style='width:20px;height:20px;background-color:{color};"
            f"margin-right:5px;border:1px solid #fff;'></div>{key}</div>"
        )
    html += "</div>"
    return html


def _legend_overlay_html(type_colors: dict[str, str]) -> str:
    if not type_colors:
        return ""
    rows = []
    for node_type, color in sorted(type_colors.items()):
        safe_type = html.escape(str(node_type))
        rows.append(
            "<div style='display:flex;align-items:center;gap:8px;margin:3px 0;'>"
            f"<span style='display:inline-block;width:12px;height:12px;border-radius:2px;background:{color};"
            "border:1px solid rgba(255,255,255,0.35);'></span>"
            f"<span>{safe_type}</span>"
            "</div>"
        )
    rows_html = "".join(rows)
    return (
        "<div id='kg-legend' style='position:fixed;top:12px;right:12px;z-index:9999;padding:10px 12px;"
        "max-width:280px;background:rgba(17,17,17,0.84);border:1px solid rgba(255,255,255,0.2);"
        "border-radius:8px;color:#f8fafc;font-family:Arial,sans-serif;font-size:12px;line-height:1.25;"
        "box-shadow:0 6px 18px rgba(0,0,0,0.28);backdrop-filter:blur(2px);'>"
        "<div style='display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:6px;'>"
        "<div style='font-weight:700;'>Legend</div>"
        "<button id='kg-legend-toggle' type='button' "
        "style='cursor:pointer;background:#0f172a;color:#f8fafc;border:1px solid rgba(255,255,255,0.25);"
        "border-radius:6px;padding:2px 8px;font-size:11px;line-height:1.4;'>Hide</button>"
        "</div>"
        "<div id='kg-legend-content'>"
        f"{rows_html}"
        "</div>"
        "</div>"
        "<script>"
        "(function(){"
        "var btn=document.getElementById('kg-legend-toggle');"
        "var content=document.getElementById('kg-legend-content');"
        "var box=document.getElementById('kg-legend');"
        "if(!btn||!content||!box){return;}"
        "btn.addEventListener('click',function(){"
        "var hidden=content.style.display==='none';"
        "if(hidden){content.style.display='block';btn.textContent='Hide';box.style.padding='10px 12px';}"
        "else{content.style.display='none';btn.textContent='Show';box.style.padding='8px 10px';}"
        "});"
        "})();"
        "</script>"
    )


def _project_paths_for_folder(folder_path: str) -> Optional[ProjectPaths]:
    if not folder_path:
        return None
    return resolve_project_paths(folder_path)


def refresh_existing_graph_dropdown() -> Any:
    choices = _discover_existing_graph_choices()
    value = choices[0][1] if choices else None
    return gr.update(choices=choices, value=value)


def _doc_names_from_node_data(data: dict[str, Any]) -> set[str]:
    raw_filepaths = str(data.get("filepath", "") or "")
    return {
        Path(raw_path.strip()).name
        for raw_path in raw_filepaths.split("||")
        if raw_path.strip()
    }


def _graph_filtered_to_documents(graph: nx.Graph, selected_doc_names: List[str]) -> nx.Graph:
    if not selected_doc_names:
        return graph.copy()

    selected = {name.strip().lower() for name in selected_doc_names if str(name).strip()}
    if not selected:
        return graph.copy()

    matching_nodes = {
        node
        for node, data in graph.nodes(data=True)
        if {name.lower() for name in _doc_names_from_node_data(data)} & selected
    }
    if not matching_nodes:
        return nx.Graph()
    return graph.subgraph(matching_nodes).copy()


def _existing_graph_document_filter_update(selected_docs_folder: str) -> Any:
    if not selected_docs_folder:
        return gr.update(choices=[], value=[])
    doc_names = _stored_document_names(selected_docs_folder)
    return gr.update(choices=doc_names, value=[])


def _ingest_folder_file_filter_update(folder_path: str) -> Any:
    if not folder_path:
        return gr.update(choices=[], value=[])

    folder = Path(folder_path).expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        return gr.update(choices=[], value=[])

    names = sorted({path.name for path in _list_document_paths_direct(folder)}, key=str.lower)
    return gr.update(choices=names, value=[])


def _existing_graph_action_button_updates(action_mode: str) -> Tuple[Any, Any]:
    normalized = (action_mode or "Select documents from graph").strip().lower()
    delete_selected = normalized == "Delete documents from graph"
    return gr.update(visible=not delete_selected), gr.update(visible=delete_selected)


def refresh_existing_graph_controls() -> Tuple[Any, Any]:
    choices = _discover_existing_graph_choices()
    value = choices[0][1] if choices else None
    doc_filter_update = _existing_graph_document_filter_update(value or "")
    return (
        gr.update(choices=choices, value=value),
        doc_filter_update,
    )


def update_source_mode(
    source_mode: str,
    folder_path: str,
    selected_docs_folder: str,
    existing_action_mode: str,
) -> Tuple[Any, Any, Any, str, Any, Any, Any, Any, Any]:
    ingest_visible = source_mode == "Ingest Folder"
    load_visible = source_mode == "Use Existing Graph"
    choices = _discover_existing_graph_choices() if load_visible else []
    selected_value = selected_docs_folder if selected_docs_folder else None
    valid_values = {value for _, value in choices}
    if selected_value not in valid_values:
        selected_value = choices[0][1] if choices else None
    dropdown_update = (
        gr.update(choices=choices, value=selected_value)
        if load_visible
        else gr.update()
    )

    if load_visible:
        count = len(choices)
        status = (
            f"Found {count} existing graph(s) in {_DOCS_ROOT}."
            if count
            else f"No existing graph pickles found in {_DOCS_ROOT}."
        )
    else:
        status = "Ingest mode selected."

    doc_filter_update = _existing_graph_document_filter_update(selected_value or "") if load_visible else gr.update(choices=[], value=[])
    action_value = existing_action_mode if existing_action_mode else "Select documents from graph"
    load_button_update, delete_button_update = _existing_graph_action_button_updates(action_value)
    return (
        gr.update(visible=ingest_visible),
        gr.update(visible=load_visible),
        dropdown_update,
        status,
        gr.update(value=action_value),
        doc_filter_update,
        load_button_update if load_visible else gr.update(visible=False),
        delete_button_update if load_visible else gr.update(visible=False),
        _ingest_folder_file_filter_update(folder_path) if ingest_visible else gr.update(choices=[], value=[]),
    )


def load_existing_graph(selected_docs_folder: str, selected_docs_filter: List[str]) -> Tuple[str, str, str, Any, Any, Any]:
    global mygraph
    updates = update_dropdowns()

    if not selected_docs_folder:
        return render_graph_for_ui(mygraph), "Select an existing graph first.", "", *updates

    try:
        loaded_graph = _load_graph_from_pickle(selected_docs_folder)
    except Exception as exc:
        return render_graph_for_ui(mygraph), f"Failed to load existing graph: {exc}", "", *updates

    selected_docs = [str(name).strip() for name in (selected_docs_filter or []) if str(name).strip()]
    selected_docs_norm = {name.lower() for name in selected_docs}
    available_docs_norm = {name.lower() for name in _stored_document_names(selected_docs_folder)}

    # Selecting every available document should be equivalent to no pre-load filter.
    use_doc_filter = bool(selected_docs_norm) and selected_docs_norm != available_docs_norm
    mygraph = _graph_filtered_to_documents(loaded_graph, selected_docs) if use_doc_filter else loaded_graph.copy()

    _PATHRAG_CACHE.pop(selected_docs_folder, None)
    _LIGHTRAG_CACHE.pop(selected_docs_folder, None)

    selected_path = Path(selected_docs_folder).expanduser().resolve()
    try:
        docs_relative = selected_path.relative_to(_DOCS_ROOT)
        docs_subfolder_name = docs_relative.parts[0] if docs_relative.parts else selected_path.name
    except ValueError:
        docs_subfolder_name = selected_path.name
    if use_doc_filter:
        message = (
            f"Loaded existing graph from {_DOCS_ROOT_DIRNAME}/{docs_subfolder_name} "
            f"with {len(selected_docs)} selected document(s)."
        )
    elif selected_docs:
        message = (
            f"Loaded existing graph from {_DOCS_ROOT_DIRNAME}/{docs_subfolder_name} "
            "with all documents selected (no filtering applied)."
        )
    else:
        message = f"Loaded existing graph from {_DOCS_ROOT_DIRNAME}/{docs_subfolder_name}"
    updates = update_dropdowns()
    return render_graph_for_ui(mygraph), message, selected_docs_folder, *updates


def delete_existing_graph_documents(selected_docs_folder: str, selected_docs_to_delete: List[str]) -> Tuple[str, str, str, Any, Any, Any, Any, Any]:
    global mygraph
    updates = update_dropdowns()
    dropdown_update = gr.update()

    if not selected_docs_folder:
        doc_filter_update = _existing_graph_document_filter_update("")
        return render_graph_for_ui(mygraph), "Select an existing graph first.", "", *updates, dropdown_update, doc_filter_update

    selected_docs = [str(name).strip() for name in (selected_docs_to_delete or []) if str(name).strip()]
    if not selected_docs:
        doc_filter_update = _existing_graph_document_filter_update(selected_docs_folder)
        return (
            render_graph_for_ui(mygraph),
            "Select one or more documents to delete.",
            selected_docs_folder,
            *updates,
            dropdown_update,
            doc_filter_update,
        )

    try:
        project_paths = resolve_project_paths(selected_docs_folder)
        storage = Storage(paths=project_paths.storage)
        storage.init()

        deleted_count = 0
        for filename in selected_docs:
            if storage.get_document_by_filename(filename):
                remove_document_from_storage(storage, filename)
                deleted_count += 1

        remaining_docs = _stored_document_names_from_database(selected_docs_folder)
        graph_removed_from_existing_list = not remaining_docs

        active_folder_value = selected_docs_folder
        if graph_removed_from_existing_list:
            if project_paths.graph_pickle_file.exists():
                project_paths.graph_pickle_file.unlink()
            mygraph = nx.Graph()
            active_folder_value = ""
        else:
            mygraph = _load_graph_from_storage(selected_docs_folder)

        _PATHRAG_CACHE.pop(selected_docs_folder, None)
        _LIGHTRAG_CACHE.pop(selected_docs_folder, None)

        saved_path: Optional[Path] = None
        if not graph_removed_from_existing_list:
            saved_path = _save_graph_pickle(selected_docs_folder, mygraph)
            if saved_path is not None:
                mygraph = _load_graph_from_pickle(selected_docs_folder)

        refreshed_choices = _discover_existing_graph_choices()
        refreshed_values = {value for _, value in refreshed_choices}
        selected_dropdown_value = active_folder_value if active_folder_value in refreshed_values else (refreshed_choices[0][1] if refreshed_choices else None)
        dropdown_update = gr.update(choices=refreshed_choices, value=selected_dropdown_value)
        doc_filter_update = _existing_graph_document_filter_update(selected_dropdown_value or "")
        updates = update_dropdowns()

        if deleted_count:
            if graph_removed_from_existing_list:
                saved_note = "\nNo documents remain in this graph; it was removed from the existing graphs list."
            else:
                saved_note = f"\nUpdated graph pickle saved to {saved_path}." if saved_path is not None else ""
            status = (
                f"Deleted {deleted_count} selected document(s) from the database for {selected_docs_folder}."
                f"{saved_note}"
            )
        else:
            status = "No selected documents were found in the database."

        return render_graph_for_ui(mygraph), status, active_folder_value, *updates, dropdown_update, doc_filter_update
    except Exception as exc:
        doc_filter_update = _existing_graph_document_filter_update(selected_docs_folder)
        return (
            render_graph_for_ui(mygraph),
            f"Failed to delete selected documents: {exc}",
            selected_docs_folder,
            *updates,
            dropdown_update,
            doc_filter_update,
        )


def update_existing_graph_documents_ui(source_mode: str, selected_docs_folder: str) -> Any:
    if source_mode != "Use Existing Graph":
        return gr.update(choices=[], value=[])
    return _existing_graph_document_filter_update(selected_docs_folder)


def update_existing_graph_action_ui(source_mode: str, existing_action_mode: str) -> Tuple[Any, Any]:
    if source_mode != "Use Existing Graph":
        return gr.update(visible=False), gr.update(visible=False)
    return _existing_graph_action_button_updates(existing_action_mode)


def update_ingest_folder_ui(folder_path: str) -> Any:
    return _ingest_folder_file_filter_update(folder_path)


def _load_graph_from_storage(folder_path: str) -> nx.Graph:
    if not folder_path:
        return nx.Graph()
    project_paths = resolve_project_paths(folder_path)
    if not Path(project_paths.storage.graph_db).exists():
        return nx.Graph()
    adapter = PathStorageAdapter(paths=project_paths.storage)
    return adapter.graph.copy()


def _list_document_paths_direct(documents_root: Path) -> List[Path]:
    return [path for path in list_document_paths(documents_root) if path.parent == documents_root]


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


def _graph_filter_match(label: str, query: str, mode: str) -> bool:
    label_norm = label.lower()
    query_norm = query.lower()
    if mode == "exact":
        return label_norm == query_norm
    return query_norm in label_norm


def _filtered_graph_for_display(graph: nx.Graph) -> nx.Graph:
    query = (_GRAPH_FILTER_TEXT or "").strip()
    mode = (_GRAPH_FILTER_MODE or "contains").strip().lower()
    if not query:
        return graph

    matching_nodes = {
        node
        for node, data in graph.nodes(data=True)
        if _graph_filter_match(str(data.get("label", node)), query, mode)
    }
    if not matching_nodes:
        return nx.Graph()

    visible_nodes = set(matching_nodes)
    for node in matching_nodes:
        visible_nodes.update(graph.neighbors(node))

    return graph.subgraph(visible_nodes).copy()


def render_graph_for_ui(graph: nx.Graph) -> str:
    return render_graph_iframe(_filtered_graph_for_display(graph))


def apply_graph_filter(filter_text: str, filter_mode: str) -> str:
    global _GRAPH_FILTER_TEXT, _GRAPH_FILTER_MODE
    _GRAPH_FILTER_TEXT = (filter_text or "").strip()
    _GRAPH_FILTER_MODE = (filter_mode or "contains").strip().lower()
    return render_graph_for_ui(mygraph)


def clear_graph_filter() -> Tuple[str, Any, Any]:
    global _GRAPH_FILTER_TEXT, _GRAPH_FILTER_MODE
    _GRAPH_FILTER_TEXT = ""
    _GRAPH_FILTER_MODE = "contains"
    return render_graph_for_ui(mygraph), gr.update(value=""), gr.update(value="Contains")


def render_graph_iframe(graph: nx.Graph, height_px: int = _GRAPH_PANEL_HEIGHT_PX) -> str:
    type_colors = generate_dynamic_type_colors(graph)
    net = Network(
        height=f"{height_px}px", width="100%",
        directed=False, bgcolor="#111111", font_color="white",
    )

    degrees = dict(graph.degree())
    max_degree = max(degrees.values(), default=1)
    degree_scale = max(1, max_degree)
    min_size, max_size = 8, 40

    for node, data in graph.nodes(data=True):
        node_raw_filepath = data.get("filepath", "") or ""
        node_doc_names = ",\n".join(
            sorted({Path(p.strip()).name for p in node_raw_filepath.split("||") if p.strip()})
        )
        node_doc_line = f"doc(s) = {node_doc_names}" if node_doc_names else ""
        node_label = str(data.get("label", node))
        node_description = f"description = {textwrap.fill(data.get('description', ''), width=80)}"
        node_title = f"{node_label}\n{node_doc_line}\n{node_description}"
        node_type = data.get("type", "unknown")
        node_color = type_colors.get(node_type, "#0EA5E9")
        node_degree = degrees.get(node, 1)
        node_size = min_size + (max_size - min_size) * (node_degree / degree_scale)
        net.add_node(str(node), label=node_label, title=node_title, color=node_color, size=node_size)

    for source, target, data in graph.edges(data=True):
        edge_keywords = data.get("keywords", "") or ""
        edge_description = data.get("description", "") or ""
        net.add_edge(
            str(source), str(target),
            title=edge_description or edge_keywords,
            value=float(data.get("weight", 1.0) or 1.0),
        )

    net.set_options("""
        var options = {
          "nodes": {
            "shape": "dot",
            "font": { "size": 12, "strokeWidth": 2, "strokeColor": "#111111" }
          },
          "edges": {
            "smooth": { "type": "dynamic" },
            "color": { "color": "#94A3B8", "highlight": "#F59E0B" },
            "font": { "size": 9, "strokeWidth": 1, "strokeColor": "#111111", "align": "middle" },
            "scaling": { "min": 1, "max": 6 }
          },
          "physics": {
            "enabled": true,
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
              "gravitationalConstant": -60,
              "centralGravity": 0.005,
              "springLength": 140,
              "springConstant": 0.08,
              "damping": 0.4
            },
            "stabilization": { "enabled": true, "iterations": 250, "fit": true }
          },
          "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": true,
            "multiselect": true,
            "tooltipDelay": 100
          }
        }
    """)

    tmp_path = Path(tempfile.gettempdir()) / "appl_kgraph_graph.html"
    net.save_graph(str(tmp_path))
    rendered = tmp_path.read_text(encoding="utf-8")
    legend_overlay = _legend_overlay_html(type_colors)
    if legend_overlay:
        if "</body>" in rendered:
            rendered = rendered.replace("</body>", f"{legend_overlay}</body>", 1)
        else:
            rendered += legend_overlay
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
        choices.append((str(label), str(node)))
    return sorted(choices, key=lambda item: item[0].lower())


def update_dropdowns() -> Tuple[Any, Any, Any]:
    choices = _dropdown_choices()
    return (
        gr.update(choices=choices, value=None),
        gr.update(choices=choices, value=None),
        gr.update(choices=choices, value=None),
    )


def _ingestion_payload(
    graph: nx.Graph,
    status: str,
    active_folder_value: str,
) -> Tuple[str, str, str, Any, Any, Any]:
    updates = update_dropdowns()
    return render_graph_for_ui(graph), status, active_folder_value, *updates


def handle_ingestion(folder_path: str, selected_ingest_files: List[str]) -> Iterator[Tuple[str, str, str, Any, Any, Any]]:
    global mygraph

    if not folder_path or not Path(folder_path).is_dir():
        yield _ingestion_payload(mygraph, "Please provide a valid folder path.", "")
        return

    documents_root = Path(folder_path).expanduser().resolve()
    all_paths = _list_document_paths_direct(documents_root)
    if not all_paths:
        empty_graph = nx.Graph()
        yield _ingestion_payload(
            empty_graph,
            "No supported files found in the selected folder.",
            str(documents_root),
        )
        return

    selected_names = {str(name).strip().lower() for name in (selected_ingest_files or []) if str(name).strip()}
    paths = (
        [path for path in all_paths if path.name.lower() in selected_names]
        if selected_names
        else all_paths
    )
    prune_missing_documents = False

    if selected_names and not paths:
        yield _ingestion_payload(
            mygraph,
            "No selected files were found in the folder. Refresh file selection and try again.",
            str(documents_root),
        )
        return

    progress_messages: List[str] = []
    if settings.logging.verbosity_enabled:
        selection_line = (
            f"Selected files for ingestion: {len(paths)} of {len(all_paths)}"
            if selected_names
            else f"Selected files for ingestion: all ({len(all_paths)})"
        )
        progress_messages = [
            f"Preparing ingestion for {documents_root}",
            selection_line,
            f"Discovered {len(paths)} supported files",
        ]
        yield _ingestion_payload(mygraph, "\n".join(progress_messages), str(documents_root))
    else:
        yield _ingestion_payload(mygraph, "Ingestion running.", str(documents_root))

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
                prune_missing_documents=prune_missing_documents,
                progress_callback=_report_progress,
            )
        except Exception as exc:
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
        if settings.logging.verbosity_enabled:
            progress_messages.append(message)
            yield _ingestion_payload(mygraph, "\n".join(progress_messages[-50:]), str(documents_root))

    if "exception" in error:
        progress_messages.append(f"Error: {error['exception']}")
        yield _ingestion_payload(mygraph, "\n".join(progress_messages[-50:]), str(documents_root))
        return

    summary = outcome["summary"]
    graph_from_storage = _load_graph_from_storage(str(documents_root))
    _PATHRAG_CACHE.pop(str(documents_root), None)
    _LIGHTRAG_CACHE.pop(str(documents_root), None)

    saved_path = _save_graph_pickle(str(documents_root), graph_from_storage)
    if saved_path is not None:
        mygraph = _load_graph_from_pickle(str(documents_root))
        pickle_note = f"\nSaved working graph pickle to {saved_path}"
    else:
        mygraph = graph_from_storage
        pickle_note = "\nUnable to save working graph pickle; showing graph from storage state."

    project_paths = resolve_project_paths(documents_root)

    status = (
        f"Ingested project at {documents_root}\n"
        f"Processed files: {summary['processed_files']}\n"
        f"Skipped files: {summary['skipped_files']}\n"
        f"Removed files: {summary['removed_files']}\n"
        f"Chunks: {summary['chunk_count']}\n"
        f"Entities: {summary['entity_count']}\n"
        f"Relations: {summary['relation_count']}"
        f"\nRetrieval snapshot: {project_paths.retrieval_graph_pickle_file}"
        f"{pickle_note}"
    )
    if settings.logging.verbosity_enabled and progress_messages:
        status = f"{status}\n\n{chr(10).join(progress_messages[-20:])}"
    yield _ingestion_payload(mygraph, status, str(documents_root))


def save_current_graph(folder_path: str) -> str:
    if not folder_path:
        return "Select and ingest a document folder first."
    saved_path = _save_graph_pickle(folder_path, mygraph)
    if saved_path is None:
        return "Unable to determine a project path for the current graph."
    return f"Saved working graph pickle to {saved_path}"


def load_saved_graph(folder_path: str) -> Tuple[str, str, Any, Any, Any]:
    global mygraph
    updates = update_dropdowns()

    if not folder_path:
        return render_graph_for_ui(mygraph), "Select and ingest a document folder first.", *updates

    project_paths = resolve_project_paths(folder_path)
    if not project_paths.graph_pickle_file.exists():
        return render_graph_for_ui(mygraph), f"No saved graph pickle found yet at {project_paths.graph_pickle_file}", *updates

    try:
        mygraph = _load_graph_from_pickle(folder_path)
        message = f"Loaded working graph pickle from {project_paths.graph_pickle_file}"
    except Exception as exc:
        return render_graph_for_ui(mygraph), f"Failed to load saved graph pickle: {exc}", *updates

    updates = update_dropdowns()
    return render_graph_for_ui(mygraph), message, *updates


def _load_saved_graph_if_available(folder_path: str) -> Optional[str]:
    global mygraph
    if not folder_path:
        return "Select and ingest a document folder first."

    project_paths = resolve_project_paths(folder_path)
    if not project_paths.graph_pickle_file.exists():
        return None

    try:
        mygraph = _load_graph_from_pickle(folder_path)
    except Exception as exc:
        return f"Failed to load saved graph pickle: {exc}"
    return None


def _save_then_reload_graph(folder_path: str) -> Tuple[Optional[Path], Optional[str]]:
    global mygraph
    saved_path = _save_graph_pickle(folder_path, mygraph)
    if saved_path is None:
        return None, "Unable to determine a project path for the current graph."
    try:
        mygraph = _load_graph_from_pickle(folder_path)
    except Exception as exc:
        return saved_path, f"Saved graph to {saved_path}, but failed to reload it: {exc}"
    return saved_path, None


def _query_allowed_documents(
    source_mode: str,
    ingest_selected_files: List[str],
    existing_action_mode: str,
    existing_selected_docs: List[str],
) -> Optional[set[str]]:
    use_existing_filter = (
        source_mode == "Use Existing Graph"
        and (existing_action_mode or "").strip().lower() == "Select documents from graph"
    )
    selected = existing_selected_docs if use_existing_filter else ingest_selected_files
    names = {str(name).strip().lower() for name in (selected or []) if str(name).strip()}
    return names or None


def merge_nodes(node1: str, node2: str, active_folder: str) -> Tuple[str, str, Any, Any, Any]:
    global mygraph
    load_error = _load_saved_graph_if_available(active_folder)
    if load_error:
        updates = update_dropdowns()
        return render_graph_for_ui(mygraph), load_error, *updates

    updates = update_dropdowns()

    if not node1 or not node2:
        return render_graph_for_ui(mygraph), "Select two nodes to merge.", *updates
    if node1 not in mygraph or node2 not in mygraph:
        return render_graph_for_ui(mygraph), "Both nodes must exist in the current graph.", *updates
    if node1 == node2:
        return render_graph_for_ui(mygraph), "Cannot merge the same node into itself.", *updates

    new_node = f"{node1}_{node2}"
    suffix = 1
    while new_node in mygraph:
        suffix += 1
        new_node = f"{node1}_{node2}_{suffix}"

    label1 = mygraph.nodes[node1].get("label", node1)
    label2 = mygraph.nodes[node2].get("label", node2)
    mygraph.add_node(new_node, label=f"{label1} + {label2}", type="Merged",
                     description=f"Merged from {node1} and {node2}", source="Merged")

    for original in (node1, node2):
        for neighbor, attrs in list(mygraph[original].items()):
            if neighbor != new_node and neighbor not in (node1, node2):
                mygraph.add_edge(new_node, neighbor, **attrs)
        mygraph.remove_node(original)

    saved_path, reload_error = _save_then_reload_graph(active_folder)
    if reload_error:
        updates = update_dropdowns()
        return render_graph_for_ui(mygraph), reload_error, *updates

    autosave = f" Saved and reloaded graph from {saved_path}." if saved_path else ""
    updates = update_dropdowns()
    return render_graph_for_ui(mygraph), f"Merged '{node1}' and '{node2}' into '{new_node}'.{autosave}", *updates


def update_node_attributes(node_id: str, new_label: str, new_type: str, new_desc: str, new_source: str, active_folder: str) -> Tuple[str, str, Any, Any, Any]:
    global mygraph
    load_error = _load_saved_graph_if_available(active_folder)
    if load_error:
        updates = update_dropdowns()
        return render_graph_for_ui(mygraph), load_error, *updates

    updates = update_dropdowns()

    if not node_id:
        return render_graph_for_ui(mygraph), "Select a node to update.", *updates
    if node_id not in mygraph:
        return render_graph_for_ui(mygraph), f"Node '{node_id}' was not found in the current graph.", *updates

    if new_label:
        mygraph.nodes[node_id]["label"] = new_label
    if new_type:
        mygraph.nodes[node_id]["type"] = new_type
    if new_desc:
        mygraph.nodes[node_id]["description"] = new_desc
    if new_source:
        mygraph.nodes[node_id]["source"] = new_source

    saved_path, reload_error = _save_then_reload_graph(active_folder)
    if reload_error:
        updates = update_dropdowns()
        return render_graph_for_ui(mygraph), reload_error, *updates

    autosave = f" Saved and reloaded graph from {saved_path}." if saved_path else ""
    updates = update_dropdowns()
    return render_graph_for_ui(mygraph), f"Updated node '{node_id}'.{autosave}", *updates


async def create_pathrag_response(
    question: str,
    chat_history: List[dict],
    active_folder: str,
    source_mode: str,
    ingest_selected_files: List[str],
    existing_action_mode: str,
    existing_selected_docs: List[str],
) -> Tuple[str, List[dict], str]:
    if not active_folder:
        chat_history = list(chat_history or [])
        chat_history.append({"role": "assistant", "content": "Select and ingest a document folder first."})
        return "", chat_history, ""

    history = list(chat_history or [])
    history.append({"role": "user", "content": question})
    try:
        rag = _get_pathrag(active_folder)
        allowed_docs = _query_allowed_documents(source_mode, ingest_selected_files, existing_action_mode, existing_selected_docs)
        result = await rag.aretrieve(
            question,
            conversation_history=_history_to_turns(history[:-1]),
            allowed_document_names=allowed_docs,
        )
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
    source_mode: str,
    ingest_selected_files: List[str],
    existing_action_mode: str,
    existing_selected_docs: List[str],
) -> Tuple[str, List[dict], str]:
    if not active_folder:
        chat_history = list(chat_history or [])
        chat_history.append({"role": "assistant", "content": "Select and ingest a document folder first."})
        return "", chat_history, ""

    history = list(chat_history or [])
    history.append({"role": "user", "content": question})
    try:
        rag = _get_lightrag(active_folder)
        allowed_docs = _query_allowed_documents(source_mode, ingest_selected_files, existing_action_mode, existing_selected_docs)
        result = await rag.aretrieve(
            question,
            conversation_history=_history_to_turns(history[:-1]),
            allowed_document_names=allowed_docs,
        )
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


with gr.Blocks(css=_APP_CSS) as demo:
    gr.Markdown("## Interactive Hybrid RAG")

    active_folder = gr.State("")

    with gr.Sidebar():
        source_mode = gr.Radio(
            choices=["Ingest Folder", "Use Existing Graph"],
            value="Ingest Folder",
            label="Graph Source",
            elem_id="graph-source-inline",
        )
        with gr.Group(visible=True) as ingest_controls:
            folder_path_input = gr.Textbox(label="Document folder", placeholder="C:\\path\\to\\documents")
            ingest_file_filter = gr.CheckboxGroup(
                choices=[],
                value=[],
                label="Files to ingest",
                info="Select one or more files. Leave empty to ingest all supported files.",
                elem_id="ingest-file-filter",
            )
            go_btn = gr.Button(value="Ingest Folder", variant="primary")
        with gr.Group(visible=False) as existing_graph_controls:
            existing_graph_dropdown = gr.Dropdown(
                choices=[],
                label=f"Existing graph in {_DOCS_ROOT_DIRNAME}",
                info=f"Scans {_DOCS_ROOT_DIRNAME}/*/.appl-kgraph/knowledge_graph/kg.pkl",
            )
            existing_graph_action = gr.Radio(
                choices=["Select documents from graph", "Delete documents from graph"],
                value="Select documents from graph",
                label="Existing graph action",
            )
            existing_graph_doc_selection = gr.CheckboxGroup(
                choices=[],
                value=[],
                label="Documents",
                elem_id="existing-graph-doc-filter",
            )
            with gr.Row():
                refresh_existing_btn = gr.Button(value="Refresh Graph List")
                load_existing_btn = gr.Button(value="Load Graph", variant="primary", visible=True)
                delete_existing_btn = gr.Button(value="Delete Selected Documents", variant="primary", visible=False)
        status_messages = gr.Textbox(
            label="Status", 
            interactive=False, 
            lines=10
        )

    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.Tab("PathRAG"):
                    pathrag_chatbot = gr.Chatbot(
                        type="messages",
                        label="PathRAG Chat History",
                        height=int(_GRAPH_PANEL_HEIGHT_PX / 2),
                        resizable=True
                    )
                    pathrag_sources = gr.Textbox(label="PathRAG sources", interactive=False, lines=14)
                    with gr.Row():
                        pathrag_msg_input = gr.Textbox(
                            show_label=False,
                            placeholder="Ask a question about the uploaded documents and Enter...",
                            scale=7,
                        )
                        pathrag_clear_btn = gr.ClearButton(
                            components=[pathrag_msg_input, pathrag_chatbot, pathrag_sources],
                            value="Clear conversation",
                            scale=3,
                        )
                with gr.Tab("LightRAG"):
                    lightrag_chatbot = gr.Chatbot(
                        type="messages",
                        label="LightRAG Chat History",
                        height=int(_GRAPH_PANEL_HEIGHT_PX / 2),
                        resizable=True
                    )
                    lightrag_sources = gr.Textbox(label="LightRAG sources", interactive=False, lines=14)
                    with gr.Row():
                        lightrag_msg_input = gr.Textbox(
                            show_label=False,
                            placeholder="Ask a question about the uploaded documents and Enter...",
                            scale=7,
                        )
                        lightrag_clear_btn = gr.ClearButton(
                            components=[lightrag_msg_input, lightrag_chatbot, lightrag_sources],
                            value="Clear conversation",
                            scale=3,
                        )

        with gr.Column(scale=1):
            graph_html = gr.HTML(render_graph_for_ui(mygraph))
            with gr.Row():
                graph_filter_text = gr.Textbox(
                    label="Node label filter",
                    placeholder="Type label text to filter rendered graph",
                    scale=4,
                )
                graph_filter_mode = gr.Radio(
                    choices=["Contains", "Exact"],
                    value="Contains",
                    label="Match",
                    scale=3,
                    elem_id="graph-filter-match-inline",
                )
            with gr.Row():
                apply_graph_filter_btn = gr.Button(value="Apply Filter", variant="primary")
                clear_graph_filter_btn = gr.Button(value="Clear Filter")
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
        inputs=[folder_path_input, ingest_file_filter],
        outputs=[graph_html, status_messages, active_folder, m1, m2, edit_node_dropdown],
    )
    folder_path_input.change(
        fn=update_ingest_folder_ui,
        inputs=[folder_path_input],
        outputs=[ingest_file_filter],
    )
    source_mode.change(
        fn=update_source_mode,
        inputs=[source_mode, folder_path_input, existing_graph_dropdown, existing_graph_action],
        outputs=[ingest_controls, existing_graph_controls, existing_graph_dropdown, status_messages, existing_graph_action, existing_graph_doc_selection, load_existing_btn, delete_existing_btn, ingest_file_filter],
    )
    existing_graph_dropdown.change(
        fn=update_existing_graph_documents_ui,
        inputs=[source_mode, existing_graph_dropdown],
        outputs=[existing_graph_doc_selection],
    )
    existing_graph_action.change(
        fn=update_existing_graph_action_ui,
        inputs=[source_mode, existing_graph_action],
        outputs=[load_existing_btn, delete_existing_btn],
    )
    refresh_existing_btn.click(
        fn=refresh_existing_graph_controls,
        inputs=[],
        outputs=[existing_graph_dropdown, existing_graph_doc_selection],
    )
    load_existing_btn.click(
        fn=load_existing_graph,
        inputs=[existing_graph_dropdown, existing_graph_doc_selection],
        outputs=[graph_html, status_messages, active_folder, m1, m2, edit_node_dropdown],
    )
    delete_existing_btn.click(
        fn=delete_existing_graph_documents,
        inputs=[existing_graph_dropdown, existing_graph_doc_selection],
        outputs=[graph_html, status_messages, active_folder, m1, m2, edit_node_dropdown, existing_graph_dropdown, existing_graph_doc_selection],
    )
    apply_graph_filter_btn.click(fn=apply_graph_filter, inputs=[graph_filter_text, graph_filter_mode], outputs=[graph_html])
    graph_filter_text.submit(fn=apply_graph_filter, inputs=[graph_filter_text, graph_filter_mode], outputs=[graph_html])
    clear_graph_filter_btn.click(
        fn=clear_graph_filter,
        inputs=[],
        outputs=[graph_html, graph_filter_text, graph_filter_mode],
    )
    updatenode_btn.click(
        fn=update_node_attributes,
        inputs=[edit_node_dropdown, edit_label, edit_type, edit_desc, edit_source, active_folder],
        outputs=[graph_html, status_messages, m1, m2, edit_node_dropdown],
    )
    mergenodes_btn.click(
        fn=merge_nodes,
        inputs=[m1, m2, active_folder],
        outputs=[graph_html, status_messages, m1, m2, edit_node_dropdown],
    )
    pathrag_msg_input.submit(
        fn=create_pathrag_response,
        inputs=[
            pathrag_msg_input,
            pathrag_chatbot,
            active_folder,
            source_mode,
            ingest_file_filter,
            existing_graph_action,
            existing_graph_doc_selection,
        ],
        outputs=[pathrag_msg_input, pathrag_chatbot, pathrag_sources],
    )
    lightrag_msg_input.submit(
        fn=create_lightrag_response,
        inputs=[
            lightrag_msg_input,
            lightrag_chatbot,
            active_folder,
            source_mode,
            ingest_file_filter,
            existing_graph_action,
            existing_graph_doc_selection,
        ],
        outputs=[lightrag_msg_input, lightrag_chatbot, lightrag_sources],
    )
    pathrag_clear_btn.click(fn=lambda: [None, None, None], inputs=[], outputs=[pathrag_msg_input, pathrag_chatbot, pathrag_sources], queue=False)
    lightrag_clear_btn.click(fn=lambda: [None, None, None], inputs=[], outputs=[lightrag_msg_input, lightrag_chatbot, lightrag_sources], queue=False)

    demo.load(fn=update_dropdowns, outputs=[m1, m2, edit_node_dropdown])
    demo.load(fn=refresh_existing_graph_dropdown, outputs=[existing_graph_dropdown])
    demo.load(fn=update_ingest_folder_ui, inputs=[folder_path_input], outputs=[ingest_file_filter])


if __name__ == "__main__":
    demo.queue()
    demo.launch(inbrowser=True, pwa=True)
