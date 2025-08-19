"""
Knowledge graph construction and management module
"""
import json
import pickle
import asyncio
from typing import List, Dict, Any, Set, Tuple, Optional
import pandas as pd
import networkx as nx
import numpy as np
from collections import defaultdict, Counter
from sklearn.metrics.pairwise import cosine_similarity
import logging

from config import Config
from llm_service import LLMService, ExtractedConcept, ExtractedRelationship
from document_processor import DocumentProcessor

logger = logging.getLogger(__name__)

class KnowledgeGraph:
    """Knowledge graph construction and management"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.graph = nx.MultiDiGraph()  # Allow multiple edges between nodes
        self.embeddings = {}  # Store embeddings for semantic similarity
        self.concept_index = defaultdict(set)  # Index concepts by type
        self.chunk_to_concepts = defaultdict(set)  # Map chunks to concepts
        self.concept_frequencies = Counter()  # Track concept frequencies
        
        # Load existing graph if available
        self.load_graph()
    
    def normalize_entity(self, entity: str) -> str:
        """Normalize entity names for consistency"""
        return entity.lower().strip()
    
    def add_concept(self, concept: ExtractedConcept, chunk_id: str = None):
        """Add a concept to the knowledge graph"""
        entity = self.normalize_entity(concept.entity)
        
        # Add or update node
        if self.graph.has_node(entity):
            # Update existing node attributes
            existing_data = self.graph.nodes[entity]
            existing_data['importance'] = max(existing_data.get('importance', 0), concept.importance)
            existing_data['frequency'] = existing_data.get('frequency', 0) + 1
            # Combine descriptions
            descriptions = set([existing_data.get('description', ''), concept.description])
            existing_data['description'] = '; '.join(filter(None, descriptions))
        else:
            # Add new node
            self.graph.add_node(
                entity,
                type=concept.type,
                importance=concept.importance,
                description=concept.description,
                frequency=1,
                chunk_ids=set()
            )
        
        # Track concept-chunk relationship
        if chunk_id:
            self.graph.nodes[entity]['chunk_ids'].add(chunk_id)
            self.chunk_to_concepts[chunk_id].add(entity)
        
        # Update indexes
        self.concept_index[concept.type].add(entity)
        self.concept_frequencies[entity] += 1
    
    def add_relationship(self, relationship: ExtractedRelationship, chunk_id: str = None):
        """Add a relationship to the knowledge graph"""
        source = self.normalize_entity(relationship.source)
        target = self.normalize_entity(relationship.target)
        
        # Ensure nodes exist (add as unknown type if not)
        if not self.graph.has_node(source):
            self.graph.add_node(source, type='unknown', importance=1, description='', frequency=1, chunk_ids=set())
        if not self.graph.has_node(target):
            self.graph.add_node(target, type='unknown', importance=1, description='', frequency=1, chunk_ids=set())
        
        # Add edge with relationship information
        edge_key = self.graph.add_edge(
            source,
            target,
            relationship=relationship.relationship,
            strength=relationship.strength,
            evidence=relationship.evidence,
            chunk_id=chunk_id
        )
        
        logger.debug(f"Added relationship: {source} -> {target} ({relationship.relationship})")
    
    def add_contextual_relationships(self, chunk_concepts: List[str], chunk_id: str):
        """Add implicit relationships based on co-occurrence in same chunk"""
        chunk_concepts = [self.normalize_entity(c) for c in chunk_concepts]
        
        # Add co-occurrence edges between all concept pairs in the chunk
        for i, concept1 in enumerate(chunk_concepts):
            for concept2 in chunk_concepts[i+1:]:
                if concept1 != concept2:
                    # Add bidirectional co-occurrence edges
                    self.graph.add_edge(
                        concept1,
                        concept2,
                        relationship='co-occurs_with',
                        strength=1,
                        evidence=f'Co-occur in chunk {chunk_id}',
                        chunk_id=chunk_id,
                        edge_type='contextual'
                    )
                    self.graph.add_edge(
                        concept2,
                        concept1,
                        relationship='co-occurs_with',
                        strength=1,
                        evidence=f'Co-occur in chunk {chunk_id}',
                        chunk_id=chunk_id,
                        edge_type='contextual'
                    )
    
    def calculate_semantic_similarities(self, embeddings: Dict[str, List[float]]):
        """Calculate and add semantic similarity edges"""
        if not embeddings:
            return
        
        entities = list(embeddings.keys())
        embedding_matrix = np.array([embeddings[entity] for entity in entities])
        
        # Calculate cosine similarities
        similarities = cosine_similarity(embedding_matrix)
        
        # Add similarity edges for highly similar concepts
        similarity_threshold = 0.8
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i+1:], i+1):
                similarity = similarities[i][j]
                if similarity > similarity_threshold:
                    self.graph.add_edge(
                        entity1,
                        entity2,
                        relationship='semantically_similar',
                        strength=int(similarity * 5),  # Scale to 1-5
                        evidence=f'Semantic similarity: {similarity:.3f}',
                        edge_type='semantic'
                    )
        
        # Store embeddings
        self.embeddings.update(embeddings)
    
    def merge_similar_concepts(self, similarity_threshold: float = 0.9):
        """Merge concepts that are very similar"""
        # This is a simplified version - in practice, you'd want more sophisticated merging
        to_merge = []
        entities = list(self.graph.nodes())
        
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                if entity1 in self.embeddings and entity2 in self.embeddings:
                    similarity = cosine_similarity(
                        [self.embeddings[entity1]], 
                        [self.embeddings[entity2]]
                    )[0][0]
                    
                    if similarity > similarity_threshold:
                        to_merge.append((entity1, entity2, similarity))
        
        # Merge highly similar concepts
        for entity1, entity2, similarity in to_merge:
            if self.graph.has_node(entity1) and self.graph.has_node(entity2):
                # Merge entity2 into entity1
                entity1_data = self.graph.nodes[entity1]
                entity2_data = self.graph.nodes[entity2]
                
                # Combine attributes
                entity1_data['frequency'] += entity2_data.get('frequency', 0)
                entity1_data['importance'] = max(entity1_data['importance'], entity2_data.get('importance', 0))
                entity1_data['chunk_ids'].update(entity2_data.get('chunk_ids', set()))
                
                # Redirect edges
                for successor in self.graph.successors(entity2):
                    for edge_data in self.graph[entity2][successor].values():
                        self.graph.add_edge(entity1, successor, **edge_data)
                
                for predecessor in self.graph.predecessors(entity2):
                    for edge_data in self.graph[predecessor][entity2].values():
                        self.graph.add_edge(predecessor, entity1, **edge_data)
                
                # Remove old node
                self.graph.remove_node(entity2)
                logger.info(f"Merged {entity2} into {entity1} (similarity: {similarity:.3f})")
    
    def detect_communities(self) -> Dict[str, int]:
        """Detect communities in the graph"""
        # Convert to undirected graph for community detection
        undirected = self.graph.to_undirected()
        
        try:
            # Use Louvain algorithm for community detection
            import networkx.algorithms.community as nx_comm
            communities = nx_comm.louvain_communities(undirected, seed=42)
            
            # Create community mapping
            community_map = {}
            for i, community in enumerate(communities):
                for node in community:
                    community_map[node] = i
            
            return community_map
            
        except Exception as e:
            logger.error(f"Community detection failed: {e}")
            return {}
    
    def get_concept_neighbors(self, concept: str, max_depth: int = 2) -> Set[str]:
        """Get neighboring concepts within specified depth"""
        concept = self.normalize_entity(concept)
        if not self.graph.has_node(concept):
            return set()
        
        neighbors = set()
        current_level = {concept}
        
        for depth in range(max_depth):
            next_level = set()
            for node in current_level:
                # Get both successors and predecessors
                node_neighbors = set(self.graph.successors(node)) | set(self.graph.predecessors(node))
                next_level.update(node_neighbors)
            
            neighbors.update(next_level)
            current_level = next_level - neighbors  # Only new nodes
            
            if not current_level:
                break
        
        neighbors.discard(concept)  # Remove the original concept
        return neighbors
    
    def get_concept_by_type(self, concept_type: str) -> Set[str]:
        """Get all concepts of a specific type"""
        return self.concept_index.get(concept_type, set())
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph"""
        stats = {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'avg_degree': np.mean([d for n, d in self.graph.degree()]) if self.graph.nodes() else 0,
            'density': nx.density(self.graph),
            'concept_types': dict(Counter(
                self.graph.nodes[node].get('type', 'unknown') 
                for node in self.graph.nodes()
            )),
            'most_frequent_concepts': self.concept_frequencies.most_common(10),
            'isolated_nodes': list(nx.isolates(self.graph))
        }
        return stats
    
    def save_graph(self):
        """Save the knowledge graph to disk"""
        try:
            # Save graph structure
            graph_data = {
                'nodes': [
                    {
                        'id': node,
                        **{k: (list(v) if isinstance(v, set) else v) for k, v in data.items()}
                    }
                    for node, data in self.graph.nodes(data=True)
                ],
                'edges': [
                    {
                        'source': source,
                        'target': target,
                        'key': key,
                        **edge_data
                    }
                    for source, target, key, edge_data in self.graph.edges(keys=True, data=True)
                ]
            }
            
            with open(self.config.GRAPH_STORAGE_PATH, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, indent=2, ensure_ascii=False)
            
            # Save embeddings separately
            if self.embeddings:
                with open(self.config.EMBEDDINGS_STORAGE_PATH, 'wb') as f:
                    pickle.dump(self.embeddings, f)
            
            logger.info(f"Graph saved with {len(graph_data['nodes'])} nodes and {len(graph_data['edges'])} edges")
            
        except Exception as e:
            logger.error(f"Failed to save graph: {e}")
    
    def load_graph(self):
        """Load the knowledge graph from disk"""
        try:
            if self.config.GRAPH_STORAGE_PATH.exists():
                with open(self.config.GRAPH_STORAGE_PATH, 'r', encoding='utf-8') as f:
                    graph_data = json.load(f)
                
                # Rebuild graph
                self.graph = nx.MultiDiGraph()
                
                # Add nodes
                for node_data in graph_data.get('nodes', []):
                    node_id = node_data.pop('id')
                    # Convert lists back to sets where needed
                    if 'chunk_ids' in node_data:
                        node_data['chunk_ids'] = set(node_data['chunk_ids'])
                    self.graph.add_node(node_id, **node_data)
                
                # Add edges
                for edge_data in graph_data.get('edges', []):
                    source = edge_data.pop('source')
                    target = edge_data.pop('target')
                    key = edge_data.pop('key', None)
                    self.graph.add_edge(source, target, key=key, **edge_data)
                
                # Rebuild indexes
                self._rebuild_indexes()
                
                logger.info(f"Loaded graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
            
            # Load embeddings
            if self.config.EMBEDDINGS_STORAGE_PATH.exists():
                with open(self.config.EMBEDDINGS_STORAGE_PATH, 'rb') as f:
                    self.embeddings = pickle.load(f)
                logger.info(f"Loaded {len(self.embeddings)} embeddings")
                
        except Exception as e:
            logger.error(f"Failed to load graph: {e}")
            # Initialize empty graph on failure
            self.graph = nx.MultiDiGraph()
    
    def _rebuild_indexes(self):
        """Rebuild internal indexes after loading"""
        self.concept_index = defaultdict(set)
        self.concept_frequencies = Counter()
        self.chunk_to_concepts = defaultdict(set)
        
        for node, data in self.graph.nodes(data=True):
            concept_type = data.get('type', 'unknown')
            self.concept_index[concept_type].add(node)
            
            frequency = data.get('frequency', 1)
            self.concept_frequencies[node] = frequency
            
            chunk_ids = data.get('chunk_ids', set())
            for chunk_id in chunk_ids:
                self.chunk_to_concepts[chunk_id].add(node)

class KnowledgeGraphBuilder:
    """High-level interface for building knowledge graphs"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.kg = KnowledgeGraph(config)
        self.llm_service = LLMService(config)
        self.doc_processor = DocumentProcessor(config)
    
    async def build_from_documents(self, force_rebuild: bool = False) -> KnowledgeGraph:
        """
        Build knowledge graph from documents
        
        Args:
            force_rebuild: If True, rebuild from scratch
            
        Returns:
            KnowledgeGraph instance
        """
        logger.info("Starting knowledge graph construction...")
        
        # If force_rebuild, clear existing graph
        if force_rebuild:
            logger.info("Force rebuild requested - clearing existing graph")
            self.kg = KnowledgeGraph(self.config)
        
        # Process documents (only new/changed ones unless force_rebuild)
        df = self.doc_processor.process_documents(force_reload=force_rebuild)
        
        if df.empty:
            logger.info("No new documents to process")
            return self.kg
        
        logger.info(f"Processing {len(df)} document chunks...")
        
        # Extract concepts and relationships
        texts = df['text'].tolist()
        metadatas = df[['chunk_id', 'source', 'filename']].to_dict('records')
        
        # Process in smaller batches for better reliability
        batch_size = 5
        all_concepts = []
        all_relationships = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
            
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            # Extract concepts
            try:
                concept_results = await self.llm_service.batch_extract_concepts(batch_texts, batch_metadatas)
                all_concepts.extend(concept_results)
            except Exception as e:
                logger.error(f"Concept extraction failed for batch {i//batch_size + 1}: {e}")
                all_concepts.extend([[] for _ in batch_texts])  # Empty results for failed batch
            
            # Extract relationships
            try:
                relationship_results = await self.llm_service.batch_extract_relationships(batch_texts, batch_metadatas)
                all_relationships.extend(relationship_results)
            except Exception as e:
                logger.error(f"Relationship extraction failed for batch {i//batch_size + 1}: {e}")
                all_relationships.extend([[] for _ in batch_texts])  # Empty results for failed batch
        
        # Generate embeddings for unique concepts
        logger.info("Generating embeddings...")
        unique_concepts = set()
        for concepts in all_concepts:
            unique_concepts.update([c.entity for c in concepts])
        
        unique_concepts = list(unique_concepts)
        concept_embeddings = {}
        
        if unique_concepts:
            try:
                embeddings = await self.llm_service.generate_embeddings(unique_concepts)
                if embeddings and len(embeddings) == len(unique_concepts):
                    concept_embeddings = dict(zip(unique_concepts, embeddings))
                    logger.info(f"Generated embeddings for {len(concept_embeddings)} concepts")
                else:
                    logger.warning("Embedding generation failed or returned incorrect number of embeddings")
            except Exception as e:
                logger.error(f"Embedding generation failed: {e}")
        
        # Build graph
        logger.info("Building knowledge graph...")
        concepts_added = 0
        relationships_added = 0
        
        for i, (concepts, relationships, metadata) in enumerate(zip(all_concepts, all_relationships, metadatas)):
            chunk_id = metadata['chunk_id']
            
            # Add concepts
            chunk_concepts = []
            for concept in concepts:
                try:
                    self.kg.add_concept(concept, chunk_id)
                    chunk_concepts.append(concept.entity)
                    concepts_added += 1
                except Exception as e:
                    logger.warning(f"Failed to add concept {concept.entity}: {e}")
            
            # Add explicit relationships
            for relationship in relationships:
                try:
                    self.kg.add_relationship(relationship, chunk_id)
                    relationships_added += 1
                except Exception as e:
                    logger.warning(f"Failed to add relationship {relationship.source} -> {relationship.target}: {e}")
            
            # Add contextual relationships
            if len(chunk_concepts) > 1:
                try:
                    self.kg.add_contextual_relationships(chunk_concepts, chunk_id)
                except Exception as e:
                    logger.warning(f"Failed to add contextual relationships for chunk {chunk_id}: {e}")
        
        # Add semantic similarities
        if concept_embeddings:
            try:
                self.kg.calculate_semantic_similarities(concept_embeddings)
            except Exception as e:
                logger.warning(f"Failed to calculate semantic similarities: {e}")
        
        # Merge similar concepts (optional, can be disabled if causing issues)
        try:
            self.kg.merge_similar_concepts()
        except Exception as e:
            logger.warning(f"Failed to merge similar concepts: {e}")
        
        # Save graph
        try:
            self.kg.save_graph()
        except Exception as e:
            logger.error(f"Failed to save graph: {e}")
        
        # Print statistics
        stats = self.kg.get_graph_statistics()
        logger.info(f"Knowledge graph built successfully:")
        logger.info(f"  - Nodes: {stats['total_nodes']}")
        logger.info(f"  - Edges: {stats['total_edges']}")
        logger.info(f"  - Concepts added: {concepts_added}")
        logger.info(f"  - Relationships added: {relationships_added}")
        logger.info(f"  - Average degree: {stats['avg_degree']:.2f}")
        logger.info(f"  - Concept types: {stats['concept_types']}")
        
        return self.kg

async def main():
    """Test the knowledge graph builder"""
    builder = KnowledgeGraphBuilder()
    kg = await builder.build_from_documents(force_rebuild=True)
    
    stats = kg.get_graph_statistics()
    print("Knowledge Graph Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    asyncio.run(main())