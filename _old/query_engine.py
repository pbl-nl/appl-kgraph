"""
Query engine for searching and retrieving information from the knowledge graph
"""
import re
import asyncio
from typing import List, Dict, Any, Set, Tuple, Optional
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
import logging

from config import Config
from knowledge_graph import KnowledgeGraph
from llm_service import LLMService

logger = logging.getLogger(__name__)

class QueryResult:
    """Container for query results"""
    
    def __init__(self, 
                 concepts: List[str] = None,
                 relationships: List[Dict] = None,
                 paths: List[List[str]] = None,
                 context: str = "",
                 relevance_score: float = 0.0):
        self.concepts = concepts or []
        self.relationships = relationships or []
        self.paths = paths or []
        self.context = context
        self.relevance_score = relevance_score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'concepts': self.concepts,
            'relationships': self.relationships,
            'paths': self.paths,
            'context': self.context,
            'relevance_score': self.relevance_score
        }

class QueryEngine:
    """Engine for querying the knowledge graph"""
    
    def __init__(self, kg: KnowledgeGraph, llm_service: LLMService = None):
        self.kg = kg
        self.llm_service = llm_service or LLMService()
    
    def extract_query_concepts(self, query: str) -> List[str]:
        """Extract potential concepts from user query"""
        # Simple keyword extraction - could be enhanced with NLP
        query_lower = query.lower()
        
        # Remove common stop words and punctuation
        stop_words = {'the', 'is', 'are', 'was', 'were', 'what', 'how', 'why', 'when', 'where', 'who',
                     'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'about'}
        
        # Extract words
        words = re.findall(r'\b\w+\b', query_lower)
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Find concepts that match or contain these keywords
        matched_concepts = []
        for concept in self.kg.graph.nodes():
            concept_lower = concept.lower()
            for keyword in keywords:
                if keyword in concept_lower or concept_lower in keyword:
                    matched_concepts.append(concept)
                    break
        
        return list(set(matched_concepts))
    
    async def find_concepts_by_similarity(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Find concepts similar to query using embeddings"""
        if not self.kg.embeddings:
            return []
        
        try:
            # Get query embedding
            query_embeddings = await self.llm_service.generate_embeddings([query])
            if not query_embeddings:
                return []
            
            query_embedding = query_embeddings[0]
            
            # Calculate similarities
            similarities = []
            for concept, concept_embedding in self.kg.embeddings.items():
                similarity = cosine_similarity([query_embedding], [concept_embedding])[0][0]
                similarities.append((concept, similarity))
            
            # Sort by similarity and return top k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
    
    def find_shortest_paths(self, source_concepts: List[str], target_concepts: List[str], 
                           max_length: int = 4) -> List[List[str]]:
        """Find shortest paths between concept sets"""
        paths = []
        
        for source in source_concepts:
            for target in target_concepts:
                if source != target and self.kg.graph.has_node(source) and self.kg.graph.has_node(target):
                    try:
                        # Find shortest path
                        path = nx.shortest_path(self.kg.graph.to_undirected(), source, target)
                        if len(path) <= max_length:
                            paths.append(path)
                    except nx.NetworkXNoPath:
                        continue
        
        # Remove duplicates and sort by length
        unique_paths = list(set(tuple(path) for path in paths))
        unique_paths.sort(key=len)
        
        return [list(path) for path in unique_paths[:5]]  # Return top 5 shortest paths
    
    def get_concept_subgraph(self, concepts: List[str], depth: int = 2) -> nx.Graph:
        """Extract subgraph around specified concepts"""
        if not concepts:
            return nx.Graph()
        
        # Get all neighbors within specified depth
        relevant_nodes = set(concepts)
        for concept in concepts:
            if self.kg.graph.has_node(concept):
                neighbors = self.kg.get_concept_neighbors(concept, max_depth=depth)
                relevant_nodes.update(neighbors)
        
        # Create subgraph
        subgraph = self.kg.graph.subgraph(relevant_nodes).copy()
        return subgraph
    
    def rank_concepts_by_relevance(self, concepts: List[str], query: str) -> List[Tuple[str, float]]:
        """Rank concepts by relevance to query"""
        rankings = []
        query_lower = query.lower()
        
        for concept in concepts:
            if not self.kg.graph.has_node(concept):
                continue
            
            node_data = self.kg.graph.nodes[concept]
            score = 0.0
            
            # Exact match bonus
            if concept.lower() in query_lower:
                score += 2.0
            
            # Partial match bonus
            concept_words = set(concept.lower().split())
            query_words = set(query_lower.split())
            overlap = len(concept_words.intersection(query_words))
            if overlap > 0:
                score += overlap / len(concept_words.union(query_words))
            
            # Importance and frequency weighting
            score += node_data.get('importance', 1) * 0.1
            score += min(node_data.get('frequency', 1) * 0.05, 0.5)  # Cap frequency bonus
            
            # Description match
            description = node_data.get('description', '').lower()
            if any(word in description for word in query_words):
                score += 0.5
            
            rankings.append((concept, score))
        
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings
    
    def extract_relationships_context(self, concepts: List[str], paths: List[List[str]]) -> List[Dict[str, Any]]:
        """Extract relationship information for context"""
        relationships = []
        processed_edges = set()
        
        # Direct relationships between query concepts
        for i, concept1 in enumerate(concepts):
            for concept2 in concepts[i+1:]:
                if self.kg.graph.has_edge(concept1, concept2):
                    edge_key = (concept1, concept2)
                    if edge_key not in processed_edges:
                        edge_data = self.kg.graph[concept1][concept2]
                        for edge_info in edge_data.values():
                            relationships.append({
                                'source': concept1,
                                'target': concept2,
                                'relationship': edge_info.get('relationship', 'related_to'),
                                'strength': edge_info.get('strength', 1),
                                'evidence': edge_info.get('evidence', '')
                            })
                        processed_edges.add(edge_key)
        
        # Relationships in paths
        for path in paths:
            for i in range(len(path) - 1):
                source, target = path[i], path[i + 1]
                edge_key = (source, target)
                if edge_key not in processed_edges and self.kg.graph.has_edge(source, target):
                    edge_data = self.kg.graph[source][target]
                    for edge_info in edge_data.values():
                        relationships.append({
                            'source': source,
                            'target': target,
                            'relationship': edge_info.get('relationship', 'related_to'),
                            'strength': edge_info.get('strength', 1),
                            'evidence': edge_info.get('evidence', '')
                        })
                    processed_edges.add(edge_key)
        
        return relationships
    
    def generate_context_text(self, result: QueryResult) -> str:
        """Generate natural language context from query results"""
        context_parts = []
        
        # Add concept information
        if result.concepts:
            context_parts.append("Relevant concepts:")
            for concept in result.concepts[:10]:  # Limit to top 10
                node_data = self.kg.graph.nodes.get(concept, {})
                concept_type = node_data.get('type', 'unknown')
                description = node_data.get('description', '')
                importance = node_data.get('importance', 1)
                
                concept_info = f"- {concept} ({concept_type})"
                if description:
                    concept_info += f": {description}"
                if importance > 3:
                    concept_info += " [High importance]"
                context_parts.append(concept_info)
        
        # Add relationship information
        if result.relationships:
            context_parts.append("\nRelevant relationships:")
            for rel in result.relationships[:15]:  # Limit to top 15
                rel_text = f"- {rel['source']} {rel['relationship']} {rel['target']}"
                if rel.get('evidence'):
                    rel_text += f" (Evidence: {rel['evidence'][:100]}...)"
                context_parts.append(rel_text)
        
        # Add path information
        if result.paths:
            context_parts.append("\nConnection paths:")
            for path in result.paths[:5]:  # Limit to top 5
                path_text = " -> ".join(path)
                context_parts.append(f"- {path_text}")
        
        return "\n".join(context_parts)
    
    async def search(self, query: str, max_results: int = 20) -> QueryResult:
        """
        Search the knowledge graph for relevant information
        
        Args:
            query: User query string
            max_results: Maximum number of results to return
            
        Returns:
            QueryResult with relevant information
        """
        logger.info(f"Searching knowledge graph for: {query}")
        
        # Extract concepts from query
        query_concepts = self.extract_query_concepts(query)
        logger.debug(f"Extracted query concepts: {query_concepts}")
        
        # Find similar concepts using embeddings
        similar_concepts = await self.find_concepts_by_similarity(query, top_k=max_results)
        embedding_concepts = [concept for concept, score in similar_concepts if score > 0.3]
        
        # Combine and rank all relevant concepts
        all_concepts = list(set(query_concepts + embedding_concepts))
        ranked_concepts = self.rank_concepts_by_relevance(all_concepts, query)
        
        # Get top concepts
        top_concepts = [concept for concept, score in ranked_concepts[:max_results] if score > 0]
        
        # Find connection paths between concepts
        paths = []
        if len(top_concepts) > 1:
            paths = self.find_shortest_paths(top_concepts[:5], top_concepts[:5])
        
        # Extract relationships
        relationships = self.extract_relationships_context(top_concepts, paths)
        
        # Calculate overall relevance score
        relevance_score = np.mean([score for _, score in ranked_concepts[:10]]) if ranked_concepts else 0.0
        
        # Create result
        result = QueryResult(
            concepts=top_concepts,
            relationships=relationships,
            paths=paths,
            relevance_score=relevance_score
        )
        
        # Generate context text
        result.context = self.generate_context_text(result)
        
        logger.info(f"Found {len(top_concepts)} relevant concepts, {len(relationships)} relationships")
        return result
    
    async def answer_query(self, query: str) -> str:
        """
        Answer a query using the knowledge graph and LLM
        
        Args:
            query: User question
            
        Returns:
            Generated answer
        """
        # Search knowledge graph
        search_result = await self.search(query)
        
        if not search_result.concepts and not search_result.relationships:
            return "I couldn't find relevant information in the knowledge graph to answer your question. Please try rephrasing your query or check if the information is available in the processed documents."
        
        # Generate answer using LLM
        answer = await self.llm_service.answer_query(query, search_result.context)
        
        return answer
    
    def get_concept_details(self, concept: str) -> Dict[str, Any]:
        """Get detailed information about a specific concept"""
        concept = self.kg.normalize_entity(concept)
        
        if not self.kg.graph.has_node(concept):
            return {"error": f"Concept '{concept}' not found in knowledge graph"}
        
        node_data = self.kg.graph.nodes[concept]
        
        # Get connected concepts
        neighbors = list(self.kg.get_concept_neighbors(concept, max_depth=1))
        
        # Get relationships
        relationships = []
        for neighbor in neighbors:
            if self.kg.graph.has_edge(concept, neighbor):
                edge_data = self.kg.graph[concept][neighbor]
                for edge_info in edge_data.values():
                    relationships.append({
                        'target': neighbor,
                        'relationship': edge_info.get('relationship', 'related_to'),
                        'strength': edge_info.get('strength', 1),
                        'evidence': edge_info.get('evidence', '')
                    })
        
        return {
            'concept': concept,
            'type': node_data.get('type', 'unknown'),
            'description': node_data.get('description', ''),
            'importance': node_data.get('importance', 1),
            'frequency': node_data.get('frequency', 1),
            'chunk_ids': list(node_data.get('chunk_ids', set())),
            'neighbors': neighbors,
            'relationships': relationships
        }

async def main():
    """Test the query engine"""
    from knowledge_graph import KnowledgeGraphBuilder
    
    # Build knowledge graph
    builder = KnowledgeGraphBuilder()
    kg = await builder.build_from_documents()
    
    # Create query engine
    query_engine = QueryEngine(kg)
    
    # Test queries
    test_queries = [
        "What is machine learning?",
        "How does artificial intelligence work?",
        "Tell me about neural networks"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        answer = await query_engine.answer_query(query)
        print(f"Answer: {answer}")

if __name__ == "__main__":
    asyncio.run(main())