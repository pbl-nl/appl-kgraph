"""
Visualization module for the knowledge graph
"""
import json
import random
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict, Counter
import networkx as nx
import pandas as pd
from pyvis.network import Network
import seaborn as sns
from pathlib import Path
import logging

from config import Config
from knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)

class GraphVisualizer:
    """Visualizer for knowledge graphs"""
    
    def __init__(self, kg: KnowledgeGraph, config: Config = Config):
        self.kg = kg
        self.config = config
        self.color_palette = sns.color_palette("husl", 12).as_hex()
        random.shuffle(self.color_palette)

    # Add these methods to the GraphVisualizer class in visualization.py
    # Place them after the __init__ method

    def _create_node_tooltip(self, node: str, data: Dict, community_id: int, viz_graph: nx.Graph) -> str:
        """Create tooltip text for a node"""
        node_degree = viz_graph.degree(node)
        node_type = data.get('type', 'unknown')
        importance = data.get('importance', 1)
        frequency = data.get('frequency', 1)
        description = data.get('description', '')
        
        tooltip = f"""
        <b>{node}</b><br>
        Type: {node_type}<br>
        Community: {community_id}<br>
        Importance: {importance}/5<br>
        Frequency: {frequency}<br>
        Connections: {node_degree}<br>
        Description: {description[:100]}{'...' if len(description) > 100 else ''}
        """
        return tooltip.strip()

    def _get_edge_color(self, edge_data_list: List[Dict]) -> str:
        """Determine edge color based on relationship types"""
        relationship_colors = {
            'related_to': '#888888',
            'subset_of': '#4CAF50',
            'part_of': '#2196F3',
            'causes': '#F44336',
            'enables': '#FF9800',
            'co-occurs_with': '#9C27B0',
            'semantically_similar': '#00BCD4',
            'contextual': '#607D8B'
        }
        
        # Get the most common relationship type
        relationship_types = [data.get('relationship', 'related_to') for data in edge_data_list]
        most_common = Counter(relationship_types).most_common(1)[0][0]
        
        return relationship_colors.get(most_common, '#888888')

    def _create_edge_tooltip(self, source: str, target: str, edge_data_list: List[Dict]) -> str:
        """Create tooltip text for an edge"""
        relationships = []
        total_strength = 0
        evidence_snippets = []
        
        for data in edge_data_list:
            rel_type = data.get('relationship', 'related_to')
            strength = data.get('strength', 1)
            evidence = data.get('evidence', '')
            
            relationships.append(rel_type)
            total_strength += strength
            if evidence:
                evidence_snippets.append(evidence[:80] + '...' if len(evidence) > 80 else evidence)
        
        avg_strength = total_strength / len(edge_data_list)
        unique_relationships = list(set(relationships))
        
        tooltip = f"""
        <b>{source} ↔ {target}</b><br>
        Relationships: {', '.join(unique_relationships)}<br>
        Average Strength: {avg_strength:.2f}<br>
        Count: {len(edge_data_list)}<br>
        """
        
        if evidence_snippets:
            tooltip += f"Evidence: {'; '.join(evidence_snippets[:3])}"
        
        return tooltip.strip()

    def _create_community_info(self, communities: Dict[str, int], colors: Dict[str, str]) -> Dict[int, Dict]:
        """Create community information for visualization"""
        community_info = defaultdict(lambda: {'nodes': [], 'color': '#808080'})
        
        for node, community_id in communities.items():
            community_info[community_id]['nodes'].append(node)
            if node in colors:
                community_info[community_id]['color'] = colors[node]
        
        return dict(community_info)
    
    def detect_communities(self) -> Dict[str, int]:
        """Detect communities in the graph for coloring"""
        try:
            # Convert to undirected graph for community detection
            undirected = self.kg.graph.to_undirected()
            
            if undirected.number_of_nodes() == 0:
                return {}
            
            # Use Louvain algorithm
            import networkx.algorithms.community as nx_comm
            communities = nx_comm.louvain_communities(undirected, seed=42)
            
            # Create community mapping
            community_map = {}
            for i, community in enumerate(communities):
                for node in community:
                    community_map[node] = i
            
            logger.info(f"Detected {len(communities)} communities")
            return community_map
            
        except Exception as e:
            logger.error(f"Community detection failed: {e}")
            # Fallback: assign communities based on node type
            community_map = {}
            type_to_community = {}
            next_community = 0
            
            for node in self.kg.graph.nodes():
                node_type = self.kg.graph.nodes[node].get('type', 'unknown')
                if node_type not in type_to_community:
                    type_to_community[node_type] = next_community
                    next_community += 1
                community_map[node] = type_to_community[node_type]
            
            return community_map
    
    def assign_colors(self, communities: Dict[str, int]) -> Dict[str, str]:
        """Assign colors to nodes based on communities"""
        color_map = {}
        community_colors = {}
        
        for node, community in communities.items():
            if community not in community_colors:
                color_idx = community % len(self.color_palette)
                community_colors[community] = self.color_palette[color_idx]
            color_map[node] = community_colors[community]
        
        return color_map
    
    def calculate_node_sizes(self, min_size: int = 10, max_size: int = 50) -> Dict[str, int]:
        """Calculate node sizes based on importance and degree"""
        sizes = {}
        
        if self.kg.graph.number_of_nodes() == 0:
            return sizes
        
        # Get node metrics
        degrees = dict(self.kg.graph.degree())
        importances = {node: data.get('importance', 1) 
                      for node, data in self.kg.graph.nodes(data=True)}
        frequencies = {node: data.get('frequency', 1) 
                      for node, data in self.kg.graph.nodes(data=True)}
        
        # Normalize metrics
        max_degree = max(degrees.values()) if degrees else 1
        max_importance = max(importances.values()) if importances else 1
        max_frequency = max(frequencies.values()) if frequencies else 1
        
        for node in self.kg.graph.nodes():
            # Combine metrics
            degree_score = degrees.get(node, 0) / max_degree
            importance_score = importances.get(node, 1) / max_importance
            frequency_score = frequencies.get(node, 1) / max_frequency
            
            # Weighted combination
            combined_score = (0.4 * degree_score + 0.4 * importance_score + 0.2 * frequency_score)
            
            # Scale to size range
            size = min_size + int((max_size - min_size) * combined_score)
            sizes[node] = size
        
        return sizes
    
    def filter_graph_for_visualization(self, max_nodes: int = 200, max_edges: int = 500) -> nx.Graph:
        """Filter graph to manageable size for visualization"""
        if self.kg.graph.number_of_nodes() <= max_nodes:
            return self.kg.graph.copy()
        
        # Get most important nodes
        node_scores = {}
        for node, data in self.kg.graph.nodes(data=True):
            degree = self.kg.graph.degree(node)
            importance = data.get('importance', 1)
            frequency = data.get('frequency', 1)
            
            # Combined score
            score = degree * 0.5 + importance * 0.3 + frequency * 0.2
            node_scores[node] = score
        
        # Select top nodes
        top_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        selected_nodes = [node for node, score in top_nodes]
        
        # Create subgraph
        subgraph = self.kg.graph.subgraph(selected_nodes).copy()
        
        # Filter edges if still too many
        if subgraph.number_of_edges() > max_edges:
            # Keep strongest edges
            edges_with_strength = []
            for u, v, data in subgraph.edges(data=True):
                strength = data.get('strength', 1)
                edges_with_strength.append((u, v, strength))
            
            edges_with_strength.sort(key=lambda x: x[2], reverse=True)
            
            # Create new graph with top edges
            filtered_graph = nx.Graph()
            filtered_graph.add_nodes_from(subgraph.nodes(data=True))
            
            for u, v, strength in edges_with_strength[:max_edges]:
                filtered_graph.add_edge(u, v, **subgraph[u][v])
            
            subgraph = filtered_graph
        
        logger.info(f"Filtered graph to {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges")
        return subgraph
    
    def create_interactive_visualization(self, 
                                       output_path: Optional[Path] = None,
                                       title: str = "Knowledge Graph",
                                       max_nodes: int = 200,
                                       physics_enabled: bool = True) -> str:
        """
        Create interactive HTML visualization using PyVis with community-based clustering
        
        Args:
            output_path: Path to save HTML file
            title: Title for the visualization
            max_nodes: Maximum number of nodes to display
            physics_enabled: Whether to enable physics simulation
            
        Returns:
            Path to generated HTML file
        """
        if output_path is None:
            output_path = self.config.VIZ_OUTPUT_PATH
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Filter graph for visualization
        viz_graph = self.filter_graph_for_visualization(max_nodes=max_nodes)
        
        if viz_graph.number_of_nodes() == 0:
            logger.warning("No nodes to visualize")
            return self._create_empty_visualization(output_path, title)
        
        # Detect communities and assign colors
        communities = self.detect_communities()
        colors = self.assign_colors(communities)
        sizes = self.calculate_node_sizes()
        
        # Try PyVis first, with better error handling
        try:
            return self._create_pyvis_visualization(output_path, title, viz_graph, communities, colors, sizes, physics_enabled)
        except Exception as e:
            logger.error(f"PyVis visualization failed: {e}")
            logger.info("Creating advanced fallback visualization...")
            return self._create_advanced_fallback_visualization(output_path, title, viz_graph, communities, colors, sizes)
    
    def _create_pyvis_visualization(self, output_path: Path, title: str, viz_graph: nx.Graph, 
                                   communities: Dict, colors: Dict, sizes: Dict, physics_enabled: bool) -> str:
        """Create PyVis visualization with better error handling"""
        
        # Create network with minimal settings first
        net = Network(height="900px", width="100%", bgcolor="#1a1a1a", font_color="#ffffff")
        
        # Verify network creation
        if not hasattr(net, 'add_node') or not hasattr(net, 'add_edge'):
            raise ValueError("PyVis Network object not properly initialized")
        
        # Configure physics
        if physics_enabled:
            net.barnes_hut()
        
        # Add nodes with community grouping
        for node, data in viz_graph.nodes(data=True):
            community_id = communities.get(node, 0)
            
            # Enhanced node tooltip
            tooltip = self._create_node_tooltip(node, data, community_id, viz_graph)
            
            net.add_node(
                node,
                label=node,
                title=tooltip,
                color=colors.get(node, "#808080"),
                size=sizes.get(node, 20),
                group=community_id,
                physics=physics_enabled
            )
        
        # Add edges with relationship details
        edge_counts = defaultdict(int)
        relationship_details = defaultdict(list)
        
        for u, v, data in viz_graph.edges(data=True):
            edge_key = tuple(sorted([u, v]))
            edge_counts[edge_key] += 1
            relationship_details[edge_key].append(data)
        
        # Add consolidated edges
        for (u, v), count in edge_counts.items():
            edge_data_list = relationship_details[(u, v)]
            
            # Create edge tooltip
            edge_tooltip = self._create_edge_tooltip(u, v, edge_data_list)
            
            # Edge properties
            avg_strength = sum(d.get('strength', 1) for d in edge_data_list) / len(edge_data_list)
            width = max(1, min(6, avg_strength + count - 1))
            edge_color = self._get_edge_color(edge_data_list)
            
            net.add_edge(
                u, v,
                title=edge_tooltip,
                width=width,
                color=edge_color,
                physics=physics_enabled
            )
        
        # Test if show method exists and works
        if not hasattr(net, 'show'):
            raise ValueError("PyVis Network missing show method")
        
        # Generate HTML
        try:
            net.show(str(output_path))
            
            # Verify file was created
            if not output_path.exists():
                raise ValueError("PyVis failed to create output file")
            
            # Enhance the generated HTML
            self._enhance_visualization_html(output_path, title, viz_graph, communities, colors)
            
            logger.info(f"PyVis visualization created successfully: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"PyVis show() method failed: {e}")
            raise
    
    def _create_advanced_fallback_visualization(self, output_path: Path, title: str, viz_graph: nx.Graph, 
                                            communities: Dict, colors: Dict, sizes: Dict) -> str:
        """Create an advanced interactive fallback visualization using D3.js with proper sizing"""
        
        # Prepare data for D3.js
        nodes_data = []
        for node, data in viz_graph.nodes(data=True):
            community_id = communities.get(node, 0)
            nodes_data.append({
                'id': node,
                'label': node,
                'group': community_id,
                'color': colors.get(node, "#808080"),
                'size': sizes.get(node, 20),
                'type': data.get('type', 'unknown'),
                'importance': data.get('importance', 1),
                'frequency': data.get('frequency', 1),
                'description': data.get('description', '')
            })
        
        edges_data = []
        edge_counts = defaultdict(int)
        relationship_details = defaultdict(list)
        
        for u, v, data in viz_graph.edges(data=True):
            edge_key = tuple(sorted([u, v]))
            edge_counts[edge_key] += 1
            relationship_details[edge_key].append(data)
        
        for (u, v), count in edge_counts.items():
            edge_data_list = relationship_details[(u, v)]
            avg_strength = sum(d.get('strength', 1) for d in edge_data_list) / len(edge_data_list)
            
            edges_data.append({
                'source': u,
                'target': v,
                'strength': avg_strength,
                'count': count,
                'relationships': [d.get('relationship', 'related') for d in edge_data_list],
                'color': self._get_edge_color(edge_data_list)
            })
        
        # Create community info
        community_info = self._create_community_info(communities, colors)
        
        # Generate improved D3.js visualization
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <script src="https://d3js.org/d3.v7.min.js"></script>
            <style>
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}
                
                html, body {{
                    width: 100%;
                    height: 100%;
                    overflow: hidden;
                    background: #1a1a1a;
                    color: #ffffff;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                }}
                
                .header {{
                    position: fixed;
                    top: 0;
                    left: 0;
                    right: 0;
                    background: rgba(26, 26, 26, 0.95);
                    padding: 15px 20px;
                    border-bottom: 1px solid #333;
                    z-index: 1000;
                    backdrop-filter: blur(10px);
                }}
                
                .header h1 {{
                    margin: 0 0 8px 0;
                    font-size: 24px;
                    font-weight: 600;
                }}
                
                .header p {{
                    margin: 0 0 12px 0;
                    font-size: 14px;
                    color: #ccc;
                }}
                
                .graph-container {{
                    position: absolute;
                    top: 140px;
                    left: 0;
                    right: 0;
                    bottom: 0;
                    width: 100vw;
                    height: calc(100vh - 140px);
                }}
                
                #graph {{
                    width: 100%;
                    height: 100%;
                    display: block;
                }}
                
                .tooltip {{
                    position: absolute;
                    background: rgba(0, 0, 0, 0.9);
                    color: white;
                    padding: 12px;
                    border-radius: 6px;
                    pointer-events: none;
                    font-size: 12px;
                    max-width: 300px;
                    z-index: 1001;
                    border: 1px solid #444;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                }}
                
                .community-legend {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 8px;
                    margin-bottom: 12px;
                }}
                
                .community-item {{
                    display: flex;
                    align-items: center;
                    background: rgba(255,255,255,0.1);
                    padding: 4px 8px;
                    border-radius: 4px;
                    font-size: 11px;
                    border: 1px solid rgba(255,255,255,0.2);
                }}
                
                .community-color {{
                    width: 12px;
                    height: 12px;
                    border-radius: 50%;
                    margin-right: 6px;
                    border: 1px solid rgba(255,255,255,0.3);
                }}
                
                .controls {{
                    display: flex;
                    align-items: center;
                    gap: 12px;
                    font-size: 12px;
                }}
                
                .controls button {{
                    background: #333;
                    color: #fff;
                    border: 1px solid #555;
                    padding: 6px 12px;
                    cursor: pointer;
                    border-radius: 4px;
                    font-size: 11px;
                    transition: all 0.2s;
                }}
                
                .controls button:hover {{
                    background: #555;
                    border-color: #777;
                }}
                
                .controls button:active {{
                    background: #222;
                }}
                
                .help-text {{
                    color: #999;
                    font-size: 11px;
                    margin-left: auto;
                }}
                
                /* Responsive design */
                @media (max-width: 768px) {{
                    .header {{
                        padding: 10px 15px;
                    }}
                    
                    .header h1 {{
                        font-size: 20px;
                    }}
                    
                    .community-legend {{
                        gap: 6px;
                    }}
                    
                    .community-item {{
                        padding: 3px 6px;
                        font-size: 10px;
                    }}
                    
                    .controls {{
                        flex-wrap: wrap;
                        gap: 8px;
                    }}
                    
                    .controls button {{
                        padding: 4px 8px;
                        font-size: 10px;
                    }}
                    
                    .help-text {{
                        margin-left: 0;
                        margin-top: 6px;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{title}</h1>
                <p>Interactive knowledge graph with {viz_graph.number_of_nodes()} concepts, {viz_graph.number_of_edges()} relationships, and {len(community_info)} communities</p>
                
                <div class="community-legend">
                    {self._create_community_legend_html(community_info)}
                </div>
                
                <div class="controls">
                    <button onclick="restartSimulation()">🔄 Restart Physics</button>
                    <button onclick="centerGraph()">🎯 Center Graph</button>
                    <button onclick="togglePhysics()">⚡ Toggle Physics</button>
                    <button onclick="fitToScreen()">📐 Fit to Screen</button>
                    <div class="help-text">🖱️ Drag nodes • 🎯 Click to focus • 🔍 Scroll to zoom</div>
                </div>
            </div>
            
            <div class="graph-container">
                <svg id="graph"></svg>
            </div>
            
            <div class="tooltip" id="tooltip" style="display: none;"></div>
            
            <script>
                // Graph data
                const nodes = {json.dumps(nodes_data)};
                const links = {json.dumps(edges_data)};
                
                // Set up responsive SVG
                const container = d3.select("#graph");
                const graphContainer = d3.select(".graph-container");
                
                let width, height;
                
                function updateDimensions() {{
                    const containerNode = graphContainer.node();
                    width = containerNode.clientWidth;
                    height = containerNode.clientHeight;
                    
                    container
                        .attr("width", width)
                        .attr("height", height)
                        .attr("viewBox", `0 0 ${{width}} ${{height}}`);
                }}
                
                // Initialize dimensions
                updateDimensions();
                
                // Handle window resize
                window.addEventListener('resize', () => {{
                    updateDimensions();
                    if (simulation) {{
                        simulation.force("center", d3.forceCenter(width / 2, height / 2));
                        simulation.alpha(0.3).restart();
                    }}
                }});
                
                // Create zoom behavior
                const zoom = d3.zoom()
                    .scaleExtent([0.1, 10])
                    .on("zoom", (event) => {{
                        g.attr("transform", event.transform);
                    }});
                
                container.call(zoom);
                
                // Create main group
                const g = container.append("g");
                
                // Create simulation
                let simulation = d3.forceSimulation(nodes)
                    .force("link", d3.forceLink(links).id(d => d.id).distance(d => Math.max(50, 100 - d.strength * 10)))
                    .force("charge", d3.forceManyBody().strength(-400))
                    .force("center", d3.forceCenter(width / 2, height / 2))
                    .force("collision", d3.forceCollide().radius(d => d.size + 10))
                    .force("x", d3.forceX(width / 2).strength(0.1))
                    .force("y", d3.forceY(height / 2).strength(0.1));
                
                // Create links
                const link = g.append("g")
                    .selectAll("line")
                    .data(links)
                    .enter().append("line")
                    .style("stroke", d => d.color)
                    .style("stroke-width", d => Math.max(1, Math.min(5, d.strength * 2)))
                    .style("opacity", 0.7)
                    .style("stroke-linecap", "round");
                
                // Create nodes
                const node = g.append("g")
                    .selectAll("circle")
                    .data(nodes)
                    .enter().append("circle")
                    .attr("r", d => Math.max(8, Math.min(25, d.size)))
                    .style("fill", d => d.color)
                    .style("stroke", "#fff")
                    .style("stroke-width", 2)
                    .style("cursor", "pointer")
                    .call(d3.drag()
                        .on("start", dragstarted)
                        .on("drag", dragged)
                        .on("end", dragended));
                
                // Add labels
                const labels = g.append("g")
                    .selectAll("text")
                    .data(nodes)
                    .enter().append("text")
                    .text(d => d.label.length > 20 ? d.label.substring(0, 20) + "..." : d.label)
                    .style("font-size", d => Math.max(10, Math.min(14, d.size * 0.6)) + "px")
                    .style("fill", "white")
                    .style("text-anchor", "middle")
                    .style("pointer-events", "none")
                    .style("text-shadow", "1px 1px 2px rgba(0,0,0,0.8)")
                    .attr("dy", d => Math.max(8, Math.min(25, d.size)) + 15);
                
                // Add tooltip
                const tooltip = d3.select("#tooltip");
                
                node.on("mouseover", (event, d) => {{
                    tooltip.style("display", "block")
                        .html(`
                            <strong>${{d.label}}</strong><br>
                            <small>Community: ${{d.group}} | Type: ${{d.type}}</small><br>
                            Importance: ${{d.importance}}/5 | Frequency: ${{d.frequency}}<br>
                            <div style="margin-top: 6px; font-size: 11px; color: #ccc;">
                                ${{d.description.substring(0, 150)}}${{d.description.length > 150 ? "..." : ""}}
                            </div>
                        `)
                        .style("left", (event.pageX + 15) + "px")
                        .style("top", (event.pageY - 10) + "px");
                    
                    // Highlight connected nodes
                    const connectedNodes = new Set();
                    const connectedLinks = new Set();
                    
                    links.forEach(l => {{
                        if (l.source.id === d.id || l.target.id === d.id) {{
                            connectedNodes.add(l.source.id);
                            connectedNodes.add(l.target.id);
                            connectedLinks.add(l);
                        }}
                    }});
                    
                    node.style("opacity", n => connectedNodes.has(n.id) ? 1 : 0.3)
                        .style("stroke-width", n => connectedNodes.has(n.id) ? 3 : 2);
                    
                    link.style("opacity", l => connectedLinks.has(l) ? 1 : 0.1)
                        .style("stroke-width", l => connectedLinks.has(l) ? Math.max(2, l.strength * 2) : 1);
                    
                    labels.style("opacity", n => connectedNodes.has(n.id) ? 1 : 0.5);
                }});
                
                node.on("mouseout", () => {{
                    tooltip.style("display", "none");
                    node.style("opacity", 1).style("stroke-width", 2);
                    link.style("opacity", 0.7).style("stroke-width", d => Math.max(1, Math.min(5, d.strength * 2)));
                    labels.style("opacity", 1);
                }});
                
                link.on("mouseover", (event, d) => {{
                    tooltip.style("display", "block")
                        .html(`
                            <strong>${{d.source.id}} ↔ ${{d.target.id}}</strong><br>
                            Relationships: ${{d.relationships.slice(0, 3).join(", ")}}${{d.relationships.length > 3 ? "..." : ""}}<br>
                            Strength: ${{d.strength.toFixed(2)}} | Count: ${{d.count}}
                        `)
                        .style("left", (event.pageX + 15) + "px")
                        .style("top", (event.pageY - 10) + "px");
                }});
                
                link.on("mouseout", () => {{
                    tooltip.style("display", "none");
                }});
                
                // Update positions on simulation tick
                simulation.on("tick", () => {{
                    link
                        .attr("x1", d => d.source.x)
                        .attr("y1", d => d.source.y)
                        .attr("x2", d => d.target.x)
                        .attr("y2", d => d.target.y);
                    
                    node
                        .attr("cx", d => Math.max(30, Math.min(width - 30, d.x)))
                        .attr("cy", d => Math.max(30, Math.min(height - 30, d.y)));
                    
                    labels
                        .attr("x", d => Math.max(30, Math.min(width - 30, d.x)))
                        .attr("y", d => Math.max(30, Math.min(height - 30, d.y)));
                }});
                
                // Drag functions
                function dragstarted(event, d) {{
                    if (!event.active) simulation.alphaTarget(0.3).restart();
                    d.fx = d.x;
                    d.fy = d.y;
                }}
                
                function dragged(event, d) {{
                    d.fx = event.x;
                    d.fy = event.y;
                }}
                
                function dragended(event, d) {{
                    if (!event.active) simulation.alphaTarget(0);
                    d.fx = null;
                    d.fy = null;
                }}
                
                // Control functions
                function restartSimulation() {{
                    simulation.alpha(1).restart();
                }}
                
                function centerGraph() {{
                    const transform = d3.zoomIdentity.translate(width / 2, height / 2).scale(1);
                    container.transition().duration(750).call(zoom.transform, transform);
                }}
                
                function fitToScreen() {{
                    const bounds = g.node().getBBox();
                    const fullWidth = bounds.width;
                    const fullHeight = bounds.height;
                    const scale = Math.min(width / fullWidth, height / fullHeight) * 0.8;
                    const translate = [width / 2 - scale * (bounds.x + fullWidth / 2), height / 2 - scale * (bounds.y + fullHeight / 2)];
                    
                    container.transition().duration(750).call(
                        zoom.transform,
                        d3.zoomIdentity.translate(translate[0], translate[1]).scale(scale)
                    );
                }}
                
                let physicsEnabled = true;
                function togglePhysics() {{
                    physicsEnabled = !physicsEnabled;
                    if (physicsEnabled) {{
                        simulation.alpha(1).restart();
                    }} else {{
                        simulation.stop();
                    }}
                }}
                
                // Auto-fit on load
                setTimeout(() => {{
                    fitToScreen();
                }}, 1000);
            </script>
        </body>
        </html>
        """
        
        # Write HTML file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Advanced D3.js visualization created: {output_path}")
        return str(output_path)

    def _create_community_legend_html(self, community_info: Dict) -> str:
        """Create HTML for community legend"""
        html = ""
        for community_id, info in list(community_info.items())[:10]:  # Show first 10
            color = info["color"]
            node_count = len(info["nodes"])
            html += f"""
            <div class="community-item">
                <div class="community-color" style="background: {color};"></div>
                <span>Community {community_id} ({node_count} nodes)</span>
            </div>
            """
        return html
    
    def _enhance_visualization_html(self, output_path: Path, title: str, graph: nx.Graph, 
                                communities: Dict, colors: Dict):
        """Enhance PyVis generated HTML with additional features"""
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Add community info header
            community_info = self._create_community_info(communities, colors)
            header = f"""
            <div style="padding: 20px; background: rgba(26,26,26,0.9); color: #fff; position: fixed; top: 0; left: 0; right: 0; z-index: 1000;">
                <h1>{title}</h1>
                <p>Interactive knowledge graph with {graph.number_of_nodes()} concepts, {graph.number_of_edges()} relationships, and {len(community_info)} communities</p>
                <div style="display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px;">
                    {self._create_community_legend_html(community_info)}
                </div>
            </div>
            <div style="height: 120px;"></div>
            """
            
            html_content = html_content.replace('<body>', f'<body>{header}')
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
        except Exception as e:
            logger.warning(f"Failed to enhance HTML: {e}")
                
    def _create_empty_visualization(self, output_path: Path, title: str) -> str:
        """Create empty visualization message"""
        html_content = f"""
        <html>
        <head><title>{title}</title></head>
        <body style="background: #1a1a1a; color: #ffffff; font-family: Arial; padding: 40px; text-align: center;">
            <h1>{title}</h1>
            <p>No nodes found in the knowledge graph to visualize.</p>
            <p>Try building the knowledge graph first with: <code>python main.py --build</code></p>
        </body>
        </html>
        """
        with open(output_path, 'w') as f:
            f.write(html_content)
        return str(output_path)
        
    def create_concept_overview(self) -> Dict[str, Any]:
        """Create overview statistics for the knowledge graph"""
        stats = self.kg.get_graph_statistics()
        
        # Add visualization-specific stats
        communities = self.detect_communities()
        community_sizes = {}
        for node, community in communities.items():
            community_sizes[community] = community_sizes.get(community, 0) + 1
        
        stats['communities'] = {
            'count': len(set(communities.values())),
            'sizes': community_sizes
        }
        
        # Top concepts by different metrics
        degrees = dict(self.kg.graph.degree())
        importances = {node: data.get('importance', 1) 
                    for node, data in self.kg.graph.nodes(data=True)}
        
        stats['top_concepts'] = {
            'by_degree': sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10],
            'by_importance': sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
        }
        
        return stats
        
    def create_subgraph_visualization(self, 
                                    central_concepts: List[str],
                                    depth: int = 2,
                                    output_path: Optional[Path] = None) -> str:
        """
        Create visualization focused on specific concepts
        
        Args:
            central_concepts: List of concepts to focus on
            depth: How many hops to include from central concepts
            output_path: Path to save HTML file
            
        Returns:
            Path to generated HTML file
        """
        # Get subgraph around central concepts
        relevant_nodes = set(central_concepts)
        for concept in central_concepts:
            if self.kg.graph.has_node(concept):
                neighbors = self.kg.get_concept_neighbors(concept, max_depth=depth)
                relevant_nodes.update(neighbors)
        
        if not relevant_nodes:
            logger.warning("No relevant nodes found for subgraph visualization")
            return ""
        
        # Create subgraph
        subgraph = self.kg.graph.subgraph(relevant_nodes).copy()
        
        # Temporarily replace the main graph
        original_graph = self.kg.graph
        self.kg.graph = subgraph
        
        try:
            # Create visualization
            title = f"Knowledge Subgraph: {', '.join(central_concepts[:3])}"
            if len(central_concepts) > 3:
                title += f" (+{len(central_concepts) - 3} more)"
            
            output_file = self.create_interactive_visualization(
                output_path=output_path,
                title=title,
                max_nodes=len(relevant_nodes) + 50,  # Allow all relevant nodes
                physics_enabled=True
            )
            
            return output_file
            
        finally:
            # Restore original graph
            self.kg.graph = original_graph

def main():
    """Test the visualization module"""
    from knowledge_graph import KnowledgeGraphBuilder
    import asyncio
    
    async def test_viz():
        # Build knowledge graph
        builder = KnowledgeGraphBuilder()
        kg = await builder.build_from_documents()
        
        # Create visualizer
        visualizer = GraphVisualizer(kg)
        
        # Create overview
        stats = visualizer.create_concept_overview()
        print("Knowledge Graph Overview:")
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        # Create visualization
        output_file = visualizer.create_interactive_visualization()
        print(f"Visualization saved to: {output_file}")
    
    asyncio.run(test_viz())

if __name__ == "__main__":
    main()