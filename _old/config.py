"""
Configuration management for the Knowledge Graph system
"""
import os
from typing import Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the Knowledge Graph system"""
    
    # Azure OpenAI Configuration
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
    
    # Document Processing
    DATA_INPUT_DIR = Path("data_input")
    DATA_OUTPUT_DIR = Path("data_output")
    CHUNK_SIZE = 1500
    CHUNK_OVERLAP = 150
    
    # Knowledge Graph
    GRAPH_STORAGE_PATH = Path("data_output/knowledge_graph.json")
    EMBEDDINGS_STORAGE_PATH = Path("data_output/embeddings.pkl")
    METADATA_STORAGE_PATH = Path("data_output/metadata.json")
    
    # LLM Processing
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0
    MAX_TOKENS = 4000
    TEMPERATURE = 0.1
    
    # Graph Analysis
    MIN_EDGE_WEIGHT = 2
    MAX_COMMUNITIES = 10
    
    # Visualization
    VIZ_OUTPUT_PATH = Path("docs/knowledge_graph.html")
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that required configuration is present"""
        required_vars = [
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_KEY"
        ]
        
        missing_vars = []
        for var in required_vars:
            # Check environment variables directly for testing compatibility
            env_value = os.getenv(var)
            if not env_value:
                missing_vars.append(var)
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")
        
        # Create directories if they don't exist
        cls.DATA_INPUT_DIR.mkdir(exist_ok=True)
        cls.DATA_OUTPUT_DIR.mkdir(exist_ok=True)
        cls.VIZ_OUTPUT_PATH.parent.mkdir(exist_ok=True)
        
        return True

# System prompts for LLM
SYSTEM_PROMPTS = {
    "concept_extraction": """You are an expert at extracting key concepts and entities from text.
Your task is to extract the most important concepts, entities, and terms from the given text.
Focus on:
- Key entities (people, organizations, locations, products, services)
- Important concepts and ideas
- Technical terms and domain-specific vocabulary
- Events and processes

Return ONLY a JSON array with the following structure:
[
    {
        "entity": "concept or entity name",
        "type": "person|organization|location|concept|event|object|condition|misc",
        "importance": 1-5,
        "description": "brief description of the entity"
    }
]

Important: Return only the JSON array, no other text or explanation.""",

    "relationship_extraction": """You are an expert at identifying relationships between concepts in text.
Your task is to extract meaningful relationships between entities and concepts mentioned in the text.
Focus on:
- Direct relationships explicitly mentioned
- Implicit relationships that can be inferred
- Causal relationships
- Hierarchical relationships
- Temporal relationships

Return ONLY a JSON array with the following structure:
[
    {
        "source": "source entity/concept",
        "target": "target entity/concept", 
        "relationship": "relationship type or description",
        "strength": 1-5,
        "evidence": "text snippet supporting this relationship"
    }
]

Important: Return only the JSON array, no other text or explanation.""",

    "query_response": """You are a helpful assistant that answers questions based on knowledge graph information.
You will be provided with:
1. A user question
2. Relevant information retrieved from a knowledge graph

Your task is to:
- Provide a comprehensive answer based on the retrieved information
- Cite the relevant knowledge when possible
- If the information is insufficient, clearly state what's missing
- Be accurate and don't hallucinate information not present in the context

Keep your response informative but concise."""
}