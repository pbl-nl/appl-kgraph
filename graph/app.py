import asyncio
import gradio as gr
import networkx as nx
from pyvis.network import Network
import os, json, pickle, random, glob, tempfile, base64
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from win32api import GetSystemMetrics
from gradio import themes
# local imports
from fileparser import FileParser
from ingestion_app import ingest_paths
from pathrag import PathRAG, render_full_context
from lightrag import LightRAG, render_full_context


# =====================================================
# -------- INITIAL RANDOM GRAPH CREATION --------------
# =====================================================
mygraph = nx.Graph()
# COLOR_MODE = "type"

# =====================================================
# -------- VALID EXTENSIONS FOR DOCUMENT SELECTION ----
# =====================================================
VALID_EXTENSIONS = [".pdf", ".docx", ".txt", ".md"]

# =====================================================
# -------- DYNAMIC COLOR GENERATION -------------------
# =====================================================
def generate_dynamic_type_colors(graph):
    types = sorted(set(data.get("type", "Unknown") for _, data in graph.nodes(data=True)))
    cmap = plt.get_cmap("tab20", len(types))
    return {t: mcolors.to_hex(cmap(i)) for i, t in enumerate(types)}

SOURCE_COLORS = {
    "User": "#17becf", "System": "#bcbd22", "External": "#e377c2", "Default": "#7f7f7f"
}

# =====================================================
# -------- GRAPH RENDERING ----------------------------
# =====================================================
# def render_graph_iframe(graph, color_mode="type", height_px=600):
def render_graph_iframe(graph, height_px=600):
    TYPE_COLORS = generate_dynamic_type_colors(graph)
    
    # net = Network(height=f"{height_px}px", width="100%", directed=False, bgcolor="#111111", select_menu=True, filter_menu=True)
    net = Network(height=f"{height_px}px", width="100%", directed=False, bgcolor="#111111")
    # net.set_options("""
    #     var options = {
    #         "nodes": {"font": {"color": "white","size":10,"face":"arial"}, "borderWidth": 2},
    #         "edges": {"color":{"color":"#AAAAAA"}, "smooth": false}
    #     }
    # """)

    # # Important: disable physics BEFORE adding the graph
    # net.barnes_hut()  # optional layout settings
    # net.toggle_physics(False)

    # Load the graph directly (fast)
    graph_path = os.path.join("knowledge_graph", "kg.pkl")
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)
    net.from_nx(graph)

    # for node, data in graph.nodes(data=True):
    #     node_label = data.get("label", "Unknown")
    #     node_type = data.get("type", "Unknown")
    #     node_source = data.get("source", "Unknown")
    #     node_description = data.get("description", "Unknown")
    #     # if color_mode == "type":
    #     color = TYPE_COLORS.get(node_type, "#7f7f7f")
    #     # else:
    #     #     color = SOURCE_COLORS.get(node_source, SOURCE_COLORS["Default"])
    #     title = f"Label: {node_label}, Type: {node_type}, Source: {node_source}, Description: {node_description}"
    #     net.add_node(node, label=data.get("label", node), title=title, color=color)

    # for u, v, data in graph.edges(data=True):
    #     rel = data.get("relation_type", "relation")
    #     desc = data.get("relation_description", "")
    #     weight = data.get("weight", 1)
    #     net.add_edge(u, v, title=f"{rel}, Description: {desc}, Weight: {weight}", value=weight)

    net.set_options("""
        var options = {
            "nodes": {
                "borderWidth": 0,
                "borderWidthSelected": 4,
                "opacity": 1,
                "fixed": {
                    "x": true,
                    "y": true
                },
                "font": {
                    "strokeWidth": 10
                },
                "size": 14
            },
            "edges": {
                "arrows": {
                    "middle": {
                        "enabled": true
                    }
                },
                "selfReferenceSize": null,
                "selfReference": {
                    "angle": 0.7853981633974483
                },
                "smooth": {
                    "forceDirection": "none"
                }
            },
            "interaction": {
                "hover": true,
                "multiselect": true,
                "navigationButtons": true
            },
            "manipulation": {
                "enabled": true,
                "initiallyActive": true
            },
            "physics": {
                "enabled": false,
                "minVelocity": 0.75
            }
        }
    """)

    # net.show_buttons(filter_=['physics'])
    # net.show_buttons()

    # net.from_nx(graph)
    # # Define the custom JS code for hover effect
    # hover_code = """
    #     var highlighted = [];
    #     var neighbors = [];
        
    #     // Event for hovering over nodes
    #     network.on('hoverNode', function(params) {
    #         var node_id = params.node;
            
    #         // Get the node data
    #         highlighted = [];
    #         neighbors = [];
            
    #         // Highlight the hovered node
    #         highlighted.push(node_id);
            
    #         // Get neighbors of the hovered node
    #         var connectedNodes = network.getConnectedNodes(node_id);
    #         neighbors = connectedNodes;

    #         // Style update for highlighting nodes
    #         network.nodes.forEach(function(node) {
    #             if (highlighted.includes(node.id)) {
    #                 node.color = {background: 'orange', border: 'black'};
    #                 node.font = {color: 'white'};
    #             } else if (neighbors.includes(node.id)) {
    #                 node.color = {background: 'yellow', border: 'black'};
    #             } else {
    #                 node.color = {background: 'gray', border: 'gray'};
    #                 node.font = {color: 'gray'};
    #             }
    #         });
            
    #         // Style update for edges
    #         network.edges.forEach(function(edge) {
    #             if (highlighted.includes(edge.from) && highlighted.includes(edge.to)) {
    #                 edge.color = 'orange';
    #                 edge.width = 4;
    #             } else if (neighbors.includes(edge.from) || neighbors.includes(edge.to)) {
    #                 edge.color = 'yellow';
    #                 edge.width = 2;
    #             } else {
    #                 edge.color = 'gray';
    #                 edge.width = 1;
    #             }
    #         });

    #         // Refresh the network to apply the styles
    #         network.redraw();
    #     });
        
    #     // Reset on mouseout
    #     network.on('blurNode', function() {
    #         network.nodes.forEach(function(node) {
    #             node.color = {background: 'lightblue', border: 'black'};
    #             node.font = {color: 'black'};
    #         });
            
    #         network.edges.forEach(function(edge) {
    #             edge.color = 'black';
    #             edge.width = 2;
    #         });
            
    #         network.redraw();
    #     });
    # """

    # # Define the custom JS code for click effect
    # click_code = """
    #     var highlighted = [];
    #     var neighbors = [];
        
    #     // Event for clicking on nodes
    #     network.on('selectNode', function(params) {
    #         var node_id = params.nodes[0];
            
    #         // Get the node data
    #         highlighted = [];
    #         neighbors = [];
            
    #         // Highlight the clicked node
    #         highlighted.push(node_id);
            
    #         // Get neighbors of the clicked node
    #         var connectedNodes = network.getConnectedNodes(node_id);
    #         neighbors = connectedNodes;

    #         // Style update for highlighting nodes
    #         network.nodes.forEach(function(node) {
    #             if (highlighted.includes(node.id)) {
    #                 node.color = {background: 'orange', border: 'black'};
    #                 node.font = {color: 'white'};
    #             } else if (neighbors.includes(node.id)) {
    #                 node.color = {background: 'yellow', border: 'black'};
    #             } else {
    #                 node.color = {background: 'gray', border: 'gray'};
    #                 node.font = {color: 'gray'};
    #             }
    #         });
            
    #         // Style update for edges
    #         network.edges.forEach(function(edge) {
    #             if (highlighted.includes(edge.from) && highlighted.includes(edge.to)) {
    #                 edge.color = 'orange';
    #                 edge.width = 4;
    #             } else if (neighbors.includes(edge.from) || neighbors.includes(edge.to)) {
    #                 edge.color = 'yellow';
    #                 edge.width = 2;
    #             } else {
    #                 edge.color = 'gray';
    #                 edge.width = 1;
    #             }
    #         });

    #         // Refresh the network to apply the styles
    #         network.redraw();
    #     });

    #     // Reset the styling when clicking anywhere outside a node
    #     network.on('deselectNode', function() {
    #         network.nodes.forEach(function(node) {
    #             node.color = {background: 'lightblue', border: 'black'};
    #             node.font = {color: 'black'};
    #         });
            
    #         network.edges.forEach(function(edge) {
    #             edge.color = 'black';
    #             edge.width = 2;
    #         });
            
    #         network.redraw();
    #     });
    # """


    # # Add the click event script
    # net.set_options(click_code)

    # net.show("network_click.html")
    tmp_path = os.path.join(tempfile.gettempdir(), "graph.html")
    net.save_graph(tmp_path)
    with open(tmp_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    b64 = base64.b64encode(html_content.encode("utf-8")).decode("utf-8")
    return f'<iframe src="data:text/html;base64,{b64}" style="width:100%; height:{height_px}px; border:none;"></iframe>'

# =====================================================
# -------- GRAPH INTERACTION --------------------------
# =====================================================
# Function to handle node click event
def node_click(node_id):
    # Return node details
    node_details = {
        1: "Node 1: This is Node 1\nColor: Red\nSize: 15",
        2: "Node 2: This is Node 2\nColor: Green\nSize: 15",
        3: "Node 3: This is Node 3\nColor: Blue\nSize: 15"
    }
    
    # Return the details of the clicked node (or a default message if node not found)
    return node_details.get(node_id, "Node details not found.")

# =====================================================
# -------- LEGEND -------------------------------------
# =====================================================
# def generate_legend_html(color_mode="type", graph=None):
def generate_legend_html(graph=None):
    # if color_mode == "type":
    COLORS = generate_dynamic_type_colors(graph)
    # else:
    #     COLORS = SOURCE_COLORS
    html = "<div style='padding:5px;'><b>Legend:</b><br>"
    for key, color in COLORS.items():
        html += f"<div style='display:flex;align-items:center;margin:2px;'>"
        html += f"<div style='width:20px;height:20px;background-color:{color};margin-right:5px;border:1px solid #fff;'></div>{key}</div>"
    html += "</div>"
    return html

# =====================================================
# -------- NODE OPERATIONS ----------------------------
# =====================================================
# def merge_nodes(node1, node2, color_mode):
def merge_nodes(node1, node2):
    if node1 not in mygraph or node2 not in mygraph:
        # return render_graph_iframe(mygraph, color_mode), generate_legend_html(color_mode, mygraph), "⚠️ Both nodes must exist."
        return render_graph_iframe(mygraph), generate_legend_html(mygraph), "⚠️ Both nodes must exist."
    if node1 == node2:
        # return render_graph_iframe(mygraph, color_mode), generate_legend_html(color_mode, mygraph), "⚠️ Cannot merge same node."
        return render_graph_iframe(mygraph), generate_legend_html(mygraph), "⚠️ Cannot merge same node."
    # define new node: labels are concatenated
    new_node = f"{node1}_{node2}"
    mygraph.add_node(new_node, label=f"{mygraph.nodes[node1].get('label','')} + {mygraph.nodes[node2].get('label','')}", type="Merged", description=f"Merged from {node1} and {node2}", source="Merged")
    # after creating the new merged node, assign the old edges to the new node and delete the old nodes
    for n in [node1, node2]:
        for neighbor, attrs in list(mygraph[n].items()):
            if neighbor not in [node1, node2]:
                mygraph.add_edge(new_node, neighbor, **attrs)
        mygraph.remove_node(n)
    # return render_graph_iframe(mygraph, color_mode), generate_legend_html(color_mode, mygraph), f"🔄 Merged '{node1}'+'{node2}' → '{new_node}'."
    return render_graph_iframe(mygraph), generate_legend_html(mygraph), f"🔄 Merged '{node1}'+'{node2}' → '{new_node}'."


# def update_node_attributes(node_id, new_label, new_type, new_desc, new_source, color_mode):
def update_node_attributes(node_id, new_label, new_type, new_desc, new_source):
    if node_id not in mygraph:
        # return render_graph_iframe(mygraph, color_mode), generate_legend_html(color_mode, mygraph), f"⚠️ Node '{node_id}' not found."
        return render_graph_iframe(mygraph), generate_legend_html(mygraph), f"⚠️ Node '{node_id}' not found."
    if new_label:
        mygraph.nodes[node_id]['label'] = new_label
    if new_type:
        mygraph.nodes[node_id]['type'] = new_type
    if new_desc:
        mygraph.nodes[node_id]['description'] = new_desc
    if new_source:
        mygraph.nodes[node_id]['source'] = new_source
    # return render_graph_iframe(mygraph, color_mode), generate_legend_html(color_mode, mygraph), f"✏️ Node '{node_id}' updated."
    return render_graph_iframe(mygraph), generate_legend_html(mygraph), f"✏️ Node '{node_id}' updated."

# =====================================================
# -------- SAVE GRAPH ---------------------------------
# =====================================================
# def save_graph_json():
#     data = {
#         "nodes": [dict(id=n, **d) for n, d in G.nodes(data=True)],
#         "links": [dict(source=u, target=v, **d) for u, v, d in G.edges(data=True)]
#     }
#     with open("graph.json", "w", encoding="utf-8") as f:
#         json.dump(data, f, indent=2)
#     return "✅ Graph saved as graph.json"

def save_graph_pickle():
    with open("graph.pkl", "wb") as f:
        pickle.dump(mygraph, f)
    return "✅ Graph saved as graph.pkl"

# =====================================================
# -------- FOLDER-BASED DOCUMENT SELECTION ------------
# =====================================================
def list_files_in_folder(folder_path):
    print(folder_path)
    if not folder_path or not os.path.isdir(folder_path):
        return gr.update(choices=[]), "⚠️ Please enter a valid folder path."
    files = []
    for ext in VALID_EXTENSIONS:
        files += glob.glob(os.path.join(folder_path, f"*{ext}"))
    files = sorted(os.path.basename(f) for f in files)
    if not files:
        return gr.update(choices=[]), f"📂 No files found with extensions: {', '.join(VALID_EXTENSIONS)}."
    return gr.update(choices=files, value=[])
    # return gr.update(choices=files), f"✅ Found {len(files)} valid files."


def handle_ingestion(folder_path, selected_files):
    # Initialize an empty list to store status messages
    status_messages = []

    # create list of paths of selected files
    paths = FileParser(Path(folder_path)).filepaths
    if not paths:
        print("No files to ingest.")
        return

    # Iterate over the generator returned by process_files
    for graph, status in ingest_paths(paths):
        status_messages.append(status)
        # Yield the current status messages
        yield graph, "\n".join(status_messages)


async def create_pathrag_response(question: str, chat_history: List[gr.ChatMessage]) -> Tuple[str, List[gr.ChatMessage], str]:
    """
    A response is obtained from Pathrag knowledge graph retrieval

    Parameters
    ----------
    question : str
        the currently user submitted question
    chat_history : List[gr.ChatMessage]
        a list of historical chatmessages, alternating between user and assistant

    Returns
    -------
    List[gr.ChatMessage]
        the updated list of historical chatmessages, alternating between user and assistant
    """
    chat_history.append(gr.ChatMessage(role="user", content=question))
    pathrag = PathRAG(
        system_prompt=""
    )
    result = await pathrag.aretrieve(question)

    # chat_history.append(gr.ChatMessage(role="assistant", content=result.answer, metadata={"title":  "Processing..."}))
    chat_history.append(gr.ChatMessage(role="assistant", content=result.answer))
    # print(f"chat_history = {chat_history}")
    # make sure that the msg_input textbox is cleared again and return the updated chat history
    lines = ""
    if result.chunk_matches:
        for i, chunk in enumerate(result.chunk_matches, 1):
            head = f"{chunk.filename or chunk.document_id or '(unknown doc)'}"
            lines += f"{i}. {head} (score={chunk.score:.3f})\n"
            lines += f"{chunk.text}\n"
            lines += 46*"-" + "\n\n"
    # else:
    #     lines.append("(none)")
    
    print()
    return "", chat_history, lines


async def create_lightrag_response(question: str, chat_history: List[gr.ChatMessage]) -> Tuple[str, List[gr.ChatMessage], str]:
    """
    A response is obtained from Lightrag knowledge graph retrieval

    Parameters
    ----------
    question : str
        the currently user submitted question
    chat_history : List[gr.ChatMessage]
        a list of historical chatmessages, alternating between user and assistant

    Returns
    -------
    List[gr.ChatMessage]
        the updated list of historical chatmessages, alternating between user and assistant
    """
    chat_history.append(gr.ChatMessage(role="user", content=question))
    lightrag = LightRAG(
        system_prompt=""
    )
    result = await lightrag.aretrieve(question)
    
    # chat_history.append(gr.ChatMessage(role="assistant", content=result.answer, metadata={"title":  "Processing..."}))
    chat_history.append(gr.ChatMessage(role="assistant", content=result.answer))
    # print(f"chat_history = {chat_history}")
    # make sure that the msg_input textbox is cleared again and return the updated chat history
    lines = ""
    if result.all_chunks:
        for i, chunk in enumerate(result.all_chunks, 1):
            # head = f"{chunk.filename or chunk.document_id or '(unknown doc)'}"
            # lines += f"{i}. {head} (score={chunk.score:.3f})\n"
            lines += f"{i}. {chunk}\n"
            lines += 46*"-" + "\n\n"

    return "", chat_history, lines

# async def predict(input, history):
#     """
#     Predict the response of the chatbot and complete a running list of chat history.
#     """
#     history.append({"role": "user", "content": input})
#     response = await create_pathrag_response(history)
#     history.append({"role": "assistant", "content": response})
#     messages = [(history[i]["content"], history[i+1]["content"]) for i in range(0, len(history)-1, 2)]
#     return messages, history



    # query = "Who are the authors of LayoutParser and do they overlap any of the other articles?"
    # query = input("Enter your question: ")
    # conversation_history = []
    # while query not in ("exit", "quit"):
    #     print("\n--- PathRAG Response ---\n")
    #     asyncio.run(graph.main.ask_with_pathrag(query, verbose=True))
    #     print("\n---\n")
    #     print("\n--- LightRAG Response ---\n")
    #     result = asyncio.run(graph.main.ask_with_lightrag(query, verbose=True))
    #     conversation_history.append((query, result.answer))
    #     print("\n---\n")
    #     query = input("Enter your question: ")

    # try:
    #     file_path = os.path.join(folder_path, "knowledge_graphs", "azureopenai_gpt-4o_azureopenai_text-embedding-ada-002_NLTKTextSplitter_2000_200", "knowledge_graph.pkl")
    #     with open(file_path, "rb") as f:
    #         newG = pickle.load(f)
    #     if not isinstance(newG, nx.Graph):
    #         raise ValueError("Invalid pickle file.")
    #     G.clear()
    #     G.add_nodes_from(newG.nodes(data=True))
    #     G.add_edges_from(newG.edges(data=True))
    #     msg = f"📦 Loaded Pickle graph from {file_path}"
    #     # return render_graph_iframe(G, color_mode), generate_legend_html(color_mode, G), msg
    #     return render_graph_iframe(G), generate_legend_html(G), msg
    # except Exception as e:
    #     # return render_graph_iframe(G, color_mode), generate_legend_html(color_mode, G), f"❌ Load failed: {e}"
    #     return render_graph_iframe(G), generate_legend_html(G), f"❌ Load failed: {e}"

# =====================================================
# -------- DROPDOWN UPDATE HELPER ---------------------
# =====================================================
# def update_dropdowns():
#     nodes = sorted(G.nodes())
#     return [gr.update(choices=nodes) for _ in range(9)]
def update_dropdowns():
    # Create dropdown entries with label → id mapping
    nodes = sorted(mygraph.nodes())
    labeled_nodes = [
        (f"{data.get('label', n)} ({n})", n)
        for n, data in mygraph.nodes(data=True)
    ]
    return [gr.update(choices=labeled_nodes) for _ in range(3)]

def toggle_checkboxgroup(checkbox_value):
    # Show the checkbox group if the checkbox is checked, hide it otherwise
    return gr.update(visible=checkbox_value)


# =====================================================
# -------- GRADIO INTERFACE ---------------------------
# =====================================================
chathistory = []
screenwidth = GetSystemMetrics(0)
screenheight = GetSystemMetrics(1)

my_theme=themes.Soft(primary_hue="blue",
                     secondary_hue="gray",
                     font=[themes.GoogleFont("Oxanium"), "Arial", "sans-serif"],
                     spacing_size=themes.sizes.spacing_sm,
                     text_size=themes.sizes.text_sm) 

with gr.Blocks(theme="glass", css=".prompt {color: green}") as demo:
# with gr.Blocks(theme=my_theme, css=".prompt {color: green}") as demo:
# with gr.Blocks(theme=my_theme) as demo:
    gr.Markdown("## Interactive Hybrid RAG")

    #sidebar
    with gr.Sidebar():
        # Folder and file(s) selection
        folder_path_input = gr.Textbox(
            label="Enter document folder (complete path)",
            # label="Enter document folder (complete path) and hit Enter",
            submit_btn=True
            # placeholder="e.g. /home/user/Documents",
        )
        file_selection = gr.Checkbox(
            label="Select File(s)",
            value=False
        )
        file_selector = gr.CheckboxGroup(
            visible=False,
            show_label=False,
            choices=[]
        )
        go_btn = gr.Button(value="GO", variant="primary")

        # Output status messages
        status_messages = gr.Textbox(
            label="Status",
            interactive=False,
            lines=8
        )

    with gr.Tabs():
        with gr.Tab("Chat"):
            with gr.Tabs():
                with gr.Tab("PathRag"):
                    with gr.Row():
                        # pathrag chatbot component
                        pathrag_chatbot = gr.Chatbot(
                            type="messages",
                            label="PathRag Chat History",
                            show_copy_button = True,
                            avatar_images=('./images/user.png','./images/bot.png'),
                            height=int(screenheight*0.4),
                            layout='bubble',
                        )
                    with gr.Row():
                        # pathrag chunk sources
                        pathrag_sources = gr.Textbox(
                            label="PathRag sources",
                            interactive=False,
                            lines=15,
                            max_lines=15,
                            show_copy_button=True
                        )
                    with gr.Row():
                        with gr.Column(scale=9):
                            # prompt textbox
                            pathrag_msg_input = gr.Textbox(
                                elem_id="prompt",
                                label="Your Question",
                                show_label=False,
                                placeholder="Type your question here and hit Enter...",
                            )
                        with gr.Column(scale=1):
                            # clear conversation button
                            pathrag_clear_btn = gr.ClearButton(components=[pathrag_msg_input, pathrag_chatbot, pathrag_sources],
                                                    value="Clear conversation",
                                                    variant="primary",
                                                    scale=1)
                with gr.Tab("LightRag"):
                    with gr.Row():
                        # lightrag chatbot component
                        lightrag_chatbot = gr.Chatbot(
                            type="messages",
                            label="LightRag Chat History",
                            show_copy_button = True,
                            avatar_images=('./images/user.png','./images/bot.png'),
                            height=int(screenheight*0.4),
                            layout='bubble'
                        )
                    with gr.Row():
                        # lightrag chunk sources
                        lightrag_sources = gr.Textbox(
                            label="LightRag sources",
                            interactive=False,
                            lines=15,
                            max_lines=15,
                            show_copy_button=True
                        )
                    with gr.Row():
                        with gr.Column(scale=9):
                            # prompt textbox
                            lightrag_msg_input = gr.Textbox(
                                elem_id="prompt",
                                label="Your Question",
                                show_label=False,
                                placeholder="Type your question here and hit Enter...",
                            )
                        with gr.Column(scale=1):
                            # clear conversation button
                            lightrag_clear_btn = gr.ClearButton(components=[lightrag_msg_input, lightrag_chatbot, lightrag_sources],
                                                    value="Clear conversation",
                                                    variant="primary",
                                                    scale=1)

            # state = gr.State([])


        with gr.Tab("Knowledge Graph"):
            with gr.Row():
                with gr.Column(scale=9):
                    graph_html = gr.HTML(render_graph_iframe(mygraph))
                with gr.Column(scale=1):
                    with gr.Row():
                        # show legend
                        legend_html = gr.HTML(generate_legend_html(mygraph))

            with gr.Tabs():
                with gr.Tab("Edit Node"):
                    with gr.Row():
                        edit_node_dropdown = gr.Dropdown(choices=[], label="Select Node")
                        edit_label = gr.Textbox(label="Label")
                        edit_type = gr.Textbox(label="Type")
                        edit_desc = gr.Textbox(label="Description")
                        edit_source = gr.Textbox(label="Source")
                        updatenode_btn = gr.Button(value="Update Node", variant="primary")
                with gr.Tab("Merge Nodes"):
                    with gr.Row():
                        m1 = gr.Dropdown(choices=[], label="Node 1")
                        m2 = gr.Dropdown(choices=[], label="Node 2")
                        mergenodes_btn = gr.Button(value="Merge Nodes", variant="primary")
                with gr.Tab("Save Graph"):
                    save_pickle_btn = gr.Button(value="💾 Save Pickle", variant="primary")

    # --- Bindings ---
    # sidebar components
    # Update the checkbox group when folder path is entered
    folder_path_input.submit(fn=list_files_in_folder,
                             inputs=folder_path_input,
                             outputs=file_selector)
    
    # # Update on losing focus / manual change
    # folder_path_input.change(fn=list_files_in_folder,
    #                          inputs=folder_path_input,
    #                          outputs=file_selector)
    
    # Link checkbox state to the visibility of the checkbox group
    file_selection.select(
        lambda value: gr.update(visible=value),
        inputs=file_selection,
        outputs=file_selector
    )

    # Go button click triggers ingestion process
    go_btn.click(fn=handle_ingestion,
                 inputs=[folder_path_input, file_selector],
                 outputs=[graph_html, status_messages] 
                 ).then(fn=update_dropdowns,
                        outputs=[m1, m2, edit_node_dropdown]
                        )
                        # then(fn=load_graph_from_pkl, inputs=[folder_path_input], outputs=[graph_html]). \

    # submission of prompt triggers respons process
    pathrag_msg_input.submit(fn=create_pathrag_response,
                             inputs=[pathrag_msg_input, pathrag_chatbot],
                             outputs=[pathrag_msg_input, pathrag_chatbot, pathrag_sources])
    lightrag_msg_input.submit(fn=create_lightrag_response,
                              inputs=[lightrag_msg_input, lightrag_chatbot],
                              outputs=[lightrag_msg_input, lightrag_chatbot, lightrag_sources])
    
    # clear prompt and chat history
    # clear_btn.click(fn=lambda: None,
    #                 inputs=None,
    #                 outputs=[pathrag_chatbot, lightrag_chatbot, pathrag_sources, lightrag_sources],
    #                 queue=False)
    pathrag_clear_btn.click(fn=lambda: [None, None, None],
                            inputs=[],
                            outputs=[pathrag_msg_input, pathrag_chatbot, pathrag_sources],
                            queue=False)
    
    lightrag_clear_btn.click(fn=lambda: [None, None, None],
                             inputs=[],
                             outputs=[lightrag_msg_input, lightrag_chatbot, lightrag_sources],
                             queue=False)
    
    # update contents of node
    updatenode_btn.click(fn=update_node_attributes,
                         inputs=[edit_node_dropdown, edit_label, edit_type, edit_desc, edit_source],
                         outputs=[graph_html, legend_html, status_messages])
    # merge nodes
    mergenodes_btn.click(fn=merge_nodes,
                         inputs=[m1, m2],
                         outputs=[graph_html, legend_html, status_messages])
    # save graph
    save_pickle_btn.click(fn=lambda: save_graph_pickle(),
                          outputs=status_messages)

    demo.load(fn=update_dropdowns,
              outputs=[m1, m2, edit_node_dropdown])

demo.launch(inbrowser=True, pwa=True)
