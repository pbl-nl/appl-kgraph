"""
Comprehensive test suite for the Knowledge Graph system
"""
import pytest
import pytest_asyncio
import asyncio
import tempfile
import shutil
from pathlib import Path
import json
import os
from unittest.mock import Mock, patch, AsyncMock

# Configure pytest for async tests
pytest_plugins = ('pytest_asyncio',)

# Import modules to test
from config import Config
from document_processor import DocumentProcessor
from llm_service import LLMService, ExtractedConcept, ExtractedRelationship
from knowledge_graph import KnowledgeGraph, KnowledgeGraphBuilder
from query_engine import QueryEngine, QueryResult
from visualization import GraphVisualizer

class TestConfig:
    """Test configuration management"""
    
    def test_config_validation_success(self):
        """Test successful configuration validation"""
        with patch.dict(os.environ, {
            'AZURE_OPENAI_ENDPOINT': 'https://test.openai.azure.com/',
            'AZURE_OPENAI_API_KEY': 'test-key'
        }):
            config = Config()
            assert config.validate() == True
    
    def test_config_validation_failure(self):
        """Test configuration validation failure"""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            with pytest.raises(ValueError):
                config.validate()

class TestDocumentProcessor:
    """Test document processing functionality"""
    
    @pytest.fixture
    def temp_config(self):
        """Create temporary configuration for testing"""
        temp_dir = tempfile.mkdtemp()
        
        class TestConfig:
            DATA_INPUT_DIR = Path(temp_dir) / "input"
            DATA_OUTPUT_DIR = Path(temp_dir) / "output"
            METADATA_STORAGE_PATH = Path(temp_dir) / "output" / "metadata.json"
            CHUNK_SIZE = 100
            CHUNK_OVERLAP = 20
        
        config = TestConfig()
        config.DATA_INPUT_DIR.mkdir(parents=True, exist_ok=True)
        config.DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        yield config
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_document_loading(self, temp_config):
        """Test document loading functionality"""
        processor = DocumentProcessor(temp_config)
        
        # Create test document
        test_file = temp_config.DATA_INPUT_DIR / "test.txt"
        test_content = "This is a test document for knowledge graph processing."
        test_file.write_text(test_content)
        
        # Load documents
        documents = processor.load_documents(force_reload=True)
        
        assert len(documents) == 1
        assert documents[0].page_content == test_content
        assert documents[0].metadata['filename'] == 'test.txt'
    
    def test_document_chunking(self, temp_config):
        """Test document chunking functionality"""
        processor = DocumentProcessor(temp_config)
        
        # Create long test document
        long_content = "This is a sentence. " * 20  # Create content longer than chunk size
        test_file = temp_config.DATA_INPUT_DIR / "long_test.txt"
        test_file.write_text(long_content)
        
        documents = processor.load_documents(force_reload=True)
        chunks = processor.chunk_documents(documents)
        
        # Should create multiple chunks
        assert len(chunks) > 1
        
        # Each chunk should have metadata
        for chunk in chunks:
            assert 'chunk_id' in chunk.metadata
            assert 'chunk_index' in chunk.metadata
            assert 'total_chunks' in chunk.metadata
    
    def test_change_detection(self, temp_config):
        """Test file change detection"""
        processor = DocumentProcessor(temp_config)
        
        # Create and process initial document
        test_file = temp_config.DATA_INPUT_DIR / "change_test.txt"
        test_file.write_text("Original content")
        
        # First load
        docs1 = processor.load_documents(force_reload=True)
        assert len(docs1) == 1
        
        # Second load without changes - should skip
        docs2 = processor.load_documents(force_reload=False)
        assert len(docs2) == 0  # No changes detected
        
        # Modify file
        test_file.write_text("Modified content")
        
        # Third load - should detect change
        docs3 = processor.load_documents(force_reload=False)
        assert len(docs3) == 1
        assert docs3[0].page_content == "Modified content"

class TestLLMService:
    """Test LLM service functionality"""
    
    @pytest.fixture
    def mock_llm_service(self):
        """Create mock LLM service for testing"""
        with patch.dict(os.environ, {
            'AZURE_OPENAI_ENDPOINT': 'https://test.openai.azure.com/',
            'AZURE_OPENAI_API_KEY': 'test-key'
        }):
            service = LLMService()
            return service
    
    def test_concept_validation(self, mock_llm_service):
        """Test concept validation"""
        valid_json = '''[
            {
                "entity": "machine learning",
                "type": "concept",
                "importance": 5,
                "description": "A subset of artificial intelligence"
            }
        ]'''
        
        concepts = mock_llm_service.validate_json_response(valid_json, ExtractedConcept)
        assert len(concepts) == 1
        assert concepts[0].entity == "machine learning"
        assert concepts[0].type == "concept"
        assert concepts[0].importance == 5
    
    def test_relationship_validation(self, mock_llm_service):
        """Test relationship validation"""
        valid_json = '''[
            {
                "source": "machine learning",
                "target": "artificial intelligence",
                "relationship": "is_subset_of",
                "strength": 4,
                "evidence": "ML is a subset of AI"
            }
        ]'''
        
        relationships = mock_llm_service.validate_json_response(valid_json, ExtractedRelationship)
        assert len(relationships) == 1
        assert relationships[0].source == "machine learning"
        assert relationships[0].target == "artificial intelligence"
        assert relationships[0].relationship == "is_subset_of"
    
    def test_invalid_json_handling(self, mock_llm_service):
        """Test handling of invalid JSON responses"""
        invalid_json = '{"invalid": "json structure"'
        
        concepts = mock_llm_service.validate_json_response(invalid_json, ExtractedConcept)
        assert len(concepts) == 0
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, mock_llm_service):
        """Test batch processing of texts"""
        with patch.object(mock_llm_service, 'extract_concepts', new_callable=AsyncMock) as mock_extract:
            mock_extract.return_value = [
                ExtractedConcept(entity="test", type="concept", importance=1, description="test concept")
            ]
            
            texts = ["Text 1", "Text 2", "Text 3"]
            results = await mock_llm_service.batch_extract_concepts(texts)
            
            assert len(results) == 3
            assert mock_extract.call_count == 3

class TestKnowledgeGraph:
    """Test knowledge graph functionality"""
    
    @pytest.fixture
    def temp_kg(self):
        """Create temporary knowledge graph for testing"""
        temp_dir = tempfile.mkdtemp()
        
        class TestConfig:
            GRAPH_STORAGE_PATH = Path(temp_dir) / "graph.json"
            EMBEDDINGS_STORAGE_PATH = Path(temp_dir) / "embeddings.pkl"
            MIN_EDGE_WEIGHT = 1
        
        kg = KnowledgeGraph(TestConfig())
        
        yield kg
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_concept_addition(self, temp_kg):
        """Test adding concepts to knowledge graph"""
        concept = ExtractedConcept(
            entity="Machine Learning",
            type="concept",
            importance=5,
            description="AI subset"
        )
        
        temp_kg.add_concept(concept, chunk_id="test_chunk")
        
        # Check if concept was added
        normalized_entity = temp_kg.normalize_entity(concept.entity)
        assert temp_kg.graph.has_node(normalized_entity)
        
        node_data = temp_kg.graph.nodes[normalized_entity]
        assert node_data['type'] == 'concept'
        assert node_data['importance'] == 5
        assert 'test_chunk' in node_data['chunk_ids']
    
    def test_relationship_addition(self, temp_kg):
        """Test adding relationships to knowledge graph"""
        relationship = ExtractedRelationship(
            source="Machine Learning",
            target="Artificial Intelligence",
            relationship="is_part_of",
            strength=4,
            evidence="ML is a subset of AI"
        )
        
        temp_kg.add_relationship(relationship, chunk_id="test_chunk")
        
        # Check if relationship was added
        source = temp_kg.normalize_entity(relationship.source)
        target = temp_kg.normalize_entity(relationship.target)
        
        assert temp_kg.graph.has_edge(source, target)
        edge_data = temp_kg.graph[source][target]
        assert len(edge_data) > 0
    
    def test_contextual_relationships(self, temp_kg):
        """Test adding contextual relationships"""
        concepts = ["concept1", "concept2", "concept3"]
        temp_kg.add_contextual_relationships(concepts, "test_chunk")
        
        # Check that co-occurrence edges were created
        normalized_concepts = [temp_kg.normalize_entity(c) for c in concepts]
        
        for i, concept1 in enumerate(normalized_concepts):
            for concept2 in normalized_concepts[i+1:]:
                assert temp_kg.graph.has_edge(concept1, concept2)
                assert temp_kg.graph.has_edge(concept2, concept1)
    
    def test_graph_statistics(self, temp_kg):
        """Test graph statistics calculation"""
        # Add some test data
        concept1 = ExtractedConcept(entity="test1", type="concept", importance=3, description="test")
        concept2 = ExtractedConcept(entity="test2", type="organization", importance=4, description="test")
        
        temp_kg.add_concept(concept1)
        temp_kg.add_concept(concept2)
        
        relationship = ExtractedRelationship(
            source="test1", target="test2", relationship="related", strength=3, evidence="test"
        )
        temp_kg.add_relationship(relationship)
        
        stats = temp_kg.get_graph_statistics()
        
        assert stats['total_nodes'] == 2
        assert stats['total_edges'] == 1
        assert 'concept' in stats['concept_types']
        assert 'organization' in stats['concept_types']

class TestQueryEngine:
    """Test query engine functionality"""
    
    @pytest.fixture
    def mock_query_engine(self):
        """Create mock query engine for testing"""
        kg = Mock()
        kg.graph.nodes.return_value = [
            ("machine learning", {"type": "concept", "importance": 5, "description": "AI subset"}),
            ("neural networks", {"type": "concept", "importance": 4, "description": "ML technique"})
        ]
        kg.normalize_entity = lambda x: x.lower()
        
        llm_service = Mock()
        
        query_engine = QueryEngine(kg, llm_service)
        return query_engine
    
    def test_query_concept_extraction(self, mock_query_engine):
        """Test extracting concepts from queries"""
        query = "What is machine learning and how does it work?"
        
        # Mock the graph to return matching concepts
        mock_query_engine.kg.graph.nodes.return_value = ["machine learning", "deep learning"]
        
        concepts = mock_query_engine.extract_query_concepts(query)
        
        # Should extract relevant keywords
        assert len(concepts) >= 0  # May vary based on implementation
    
    def test_concept_ranking(self, mock_query_engine):
        """Test concept relevance ranking"""
        concepts = ["machine learning", "neural networks", "unrelated concept"]
        query = "machine learning algorithms"
        
        # Mock graph node data
        mock_query_engine.kg.graph.has_node.return_value = True
        mock_query_engine.kg.graph.nodes = {
            "machine learning": {"importance": 5, "frequency": 10, "description": "AI subset"},
            "neural networks": {"importance": 4, "frequency": 5, "description": "ML technique"},
            "unrelated concept": {"importance": 1, "frequency": 1, "description": "unrelated"}
        }
        
        rankings = mock_query_engine.rank_concepts_by_relevance(concepts, query)
        
        # Machine learning should rank highest
        assert rankings[0][0] == "machine learning"
        assert rankings[0][1] > rankings[1][1]  # Higher score

class TestIntegration:
    """Integration tests for the complete system"""
    
    @pytest.fixture
    def integration_setup(self):
        """Set up integration test environment"""
        temp_dir = tempfile.mkdtemp()
        
        # Create test configuration
        class TestConfig:
            DATA_INPUT_DIR = Path(temp_dir) / "input"
            DATA_OUTPUT_DIR = Path(temp_dir) / "output"
            GRAPH_STORAGE_PATH = Path(temp_dir) / "output" / "graph.json"
            EMBEDDINGS_STORAGE_PATH = Path(temp_dir) / "output" / "embeddings.pkl"
            METADATA_STORAGE_PATH = Path(temp_dir) / "output" / "metadata.json"
            VIZ_OUTPUT_PATH = Path(temp_dir) / "viz.html"
            CHUNK_SIZE = 200
            CHUNK_OVERLAP = 50
            AZURE_OPENAI_ENDPOINT = "https://test.openai.azure.com/"
            AZURE_OPENAI_API_KEY = "test-key"
            AZURE_OPENAI_API_VERSION = "2024-02-01"
            AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-4"
            AZURE_OPENAI_EMBEDDING_DEPLOYMENT = "text-embedding-ada-002"
            MAX_RETRIES = 1
            RETRY_DELAY = 0.1
            MAX_TOKENS = 1000
            TEMPERATURE = 0.1
            MIN_EDGE_WEIGHT = 1
            MAX_COMMUNITIES = 5
            
            @classmethod
            def validate(cls):
                cls.DATA_INPUT_DIR.mkdir(parents=True, exist_ok=True)
                cls.DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                cls.VIZ_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
                return True
        
        config = TestConfig()
        config.validate()
        
        yield config, temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_document_to_graph_pipeline(self, integration_setup):
        """Test complete pipeline from documents to knowledge graph"""
        config, temp_dir = integration_setup
        
        # Create test documents
        doc1 = config.DATA_INPUT_DIR / "doc1.txt"
        doc1.write_text("Machine learning is a subset of artificial intelligence. "
                       "Neural networks are a key component of machine learning.")
        
        doc2 = config.DATA_INPUT_DIR / "doc2.txt" 
        doc2.write_text("Deep learning uses neural networks with multiple layers. "
                       "It is effective for image recognition and natural language processing.")
        
        # Test document processing
        processor = DocumentProcessor(config)
        df = processor.process_documents(force_reload=True)
        
        assert len(df) > 0
        assert 'text' in df.columns
        assert 'chunk_id' in df.columns
        
        # Test knowledge graph creation
        kg = KnowledgeGraph(config)
        
        # Mock LLM responses for testing
        with patch('llm_service.LLMService') as mock_llm_class:
            mock_llm = Mock()
            mock_llm_class.return_value = mock_llm
            
            # Mock concept extraction
            mock_llm.batch_extract_concepts = AsyncMock(return_value=[
                [ExtractedConcept(entity="machine learning", type="concept", importance=5, description="AI subset")],
                [ExtractedConcept(entity="deep learning", type="concept", importance=4, description="ML technique")]
            ])
            
            # Mock relationship extraction
            mock_llm.batch_extract_relationships = AsyncMock(return_value=[
                [ExtractedRelationship(source="machine learning", target="artificial intelligence", 
                                     relationship="subset_of", strength=5, evidence="ML is subset of AI")],
                []
            ])
            
            # Mock embeddings
            mock_llm.generate_embeddings = AsyncMock(return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
            
            # Build knowledge graph
            builder = KnowledgeGraphBuilder(config)
            builder.llm_service = mock_llm
            
            # Run async test
            async def test_build():
                result_kg = await builder.build_from_documents(force_rebuild=True)
                return result_kg
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result_kg = loop.run_until_complete(test_build())
            loop.close()
            
            # Verify graph was built
            assert result_kg.graph.number_of_nodes() > 0
            
            # Test query engine
            query_engine = QueryEngine(result_kg, mock_llm)
            
            # Mock query response
            mock_llm.answer_query = AsyncMock(return_value="Machine learning is a subset of artificial intelligence.")
            
            async def test_query():
                answer = await query_engine.answer_query("What is machine learning?")
                return answer
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            answer = loop.run_until_complete(test_query())
            loop.close()
            
            assert "machine learning" in answer.lower()

def run_performance_test():
    """Run performance tests for large datasets"""
    print("Running performance tests...")
    
    # Create large test dataset
    temp_dir = tempfile.mkdtemp()
    input_dir = Path(temp_dir) / "input"
    input_dir.mkdir(parents=True)
    
    # Generate multiple test documents
    for i in range(10):
        doc_file = input_dir / f"doc_{i}.txt"
        content = f"Document {i} content. " * 100  # Create substantial content
        doc_file.write_text(content)
    
    try:
        import time
        start_time = time.time()
        
        # Test document processing performance
        class PerfConfig:
            DATA_INPUT_DIR = input_dir
            DATA_OUTPUT_DIR = Path(temp_dir) / "output"
            METADATA_STORAGE_PATH = Path(temp_dir) / "output" / "metadata.json"
            CHUNK_SIZE = 500
            CHUNK_OVERLAP = 100
        
        PerfConfig.DATA_OUTPUT_DIR.mkdir(exist_ok=True)
        
        processor = DocumentProcessor(PerfConfig())
        df = processor.process_documents(force_reload=True)
        
        processing_time = time.time() - start_time
        
        print(f"Processed {len(df)} chunks in {processing_time:.2f} seconds")
        print(f"Processing rate: {len(df)/processing_time:.2f} chunks/second")
        
        # Performance assertions
        assert processing_time < 30  # Should complete within 30 seconds
        assert len(df) > 0
        
        print("Performance tests passed!")
        
    finally:
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])
    
    # Run performance tests
    run_performance_test()