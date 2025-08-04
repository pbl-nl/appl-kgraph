"""
LLM service module for Azure OpenAI integration
"""
import json
import asyncio
from typing import List, Dict, Any, Optional, Union
import openai
from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import tiktoken
from pydantic import BaseModel, ValidationError
from config import Config, SYSTEM_PROMPTS
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExtractedConcept(BaseModel):
    """Pydantic model for extracted concepts"""
    entity: str
    type: str
    importance: int
    description: str

class ExtractedRelationship(BaseModel):
    """Pydantic model for extracted relationships"""
    source: str
    target: str
    relationship: str
    strength: int
    evidence: str

class LLMService:
    """Service for interacting with Azure OpenAI"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        config.validate()
        
        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            api_key=config.AZURE_OPENAI_API_KEY,
            api_version=config.AZURE_OPENAI_API_VERSION,
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT
        )
        
        # Initialize tokenizer for token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        except:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))
    
    def trim_text_to_token_limit(self, text: str, max_tokens: int = 3000) -> str:
        """Trim text to fit within token limit"""
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        trimmed_tokens = tokens[:max_tokens]
        return self.tokenizer.decode(trimmed_tokens)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def _make_api_call(self, messages: List[Dict], max_tokens: int = None) -> str:
        """Make API call to Azure OpenAI with retry logic"""
        try:
            # Force JSON format for concept and relationship extraction
            is_extraction_task = any("JSON" in str(msg.get("content", "")) for msg in messages)
            
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.config.AZURE_OPENAI_DEPLOYMENT_NAME,
                messages=messages,
                max_tokens=max_tokens or self.config.MAX_TOKENS,
                temperature=self.config.TEMPERATURE,
                response_format={"type": "json_object"} if is_extraction_task else None
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"API call failed: {e}")
            raise
    
    def validate_json_response(self, response: str, model_class: BaseModel) -> List[BaseModel]:
        """Validate and parse JSON response"""
        try:
            # Clean response
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            
            # Parse JSON
            data = json.loads(response)
            
            # Handle different response formats
            if isinstance(data, dict):
                # Check if it's wrapped in entities/relationships key
                if "entities" in data:
                    data = data["entities"]
                elif "relationships" in data:
                    data = data["relationships"]
                else:
                    # Single object
                    data = [data]
            
            # Validate with Pydantic
            if isinstance(data, list):
                results = []
                for item in data:
                    if item and isinstance(item, dict):
                        try:
                            results.append(model_class(**item))
                        except ValidationError as ve:
                            logger.warning(f"Skipping invalid item: {ve}")
                            continue
                return results
            else:
                return [model_class(**data)]
                
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"JSON validation failed: {e}")
            logger.error(f"Response was: {response}")
            return []
    
    async def extract_concepts(self, text: str, metadata: Dict = None) -> List[ExtractedConcept]:
        """
        Extract concepts from text using Azure OpenAI
        
        Args:
            text: Input text to process
            metadata: Additional metadata for context
            
        Returns:
            List of ExtractedConcept objects
        """
        # Trim text if too long
        text = self.trim_text_to_token_limit(text)
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPTS["concept_extraction"]},
            {"role": "user", "content": f"Extract concepts from this text:\n\n{text}"}
        ]
        
        try:
            response = await self._make_api_call(messages)
            concepts = self.validate_json_response(response, ExtractedConcept)
            
            logger.info(f"Extracted {len(concepts)} concepts")
            return concepts
            
        except Exception as e:
            logger.error(f"Concept extraction failed: {e}")
            return []
    
    async def extract_relationships(self, text: str, metadata: Dict = None) -> List[ExtractedRelationship]:
        """
        Extract relationships from text using Azure OpenAI
        
        Args:
            text: Input text to process
            metadata: Additional metadata for context
            
        Returns:
            List of ExtractedRelationship objects
        """
        # Trim text if too long
        text = self.trim_text_to_token_limit(text)
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPTS["relationship_extraction"]},
            {"role": "user", "content": f"Extract relationships from this text:\n\n{text}"}
        ]
        
        try:
            response = await self._make_api_call(messages)
            relationships = self.validate_json_response(response, ExtractedRelationship)
            
            # Add metadata
            for rel in relationships:
                if metadata:
                    rel.metadata = metadata
            
            logger.info(f"Extracted {len(relationships)} relationships")
            return relationships
            
        except Exception as e:
            logger.error(f"Relationship extraction failed: {e}")
            return []
    
    async def batch_extract_concepts(self, texts: List[str], metadatas: List[Dict] = None) -> List[List[ExtractedConcept]]:
        """
        Extract concepts from multiple texts concurrently
        
        Args:
            texts: List of input texts
            metadatas: List of metadata dicts for each text
            
        Returns:
            List of concept lists for each input text
        """
        if metadatas is None:
            metadatas = [{}] * len(texts)
        
        tasks = [
            self.extract_concepts(text, metadata)
            for text, metadata in zip(texts, metadatas)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to process text {i}: {result}")
                valid_results.append([])
            else:
                valid_results.append(result)
        
        return valid_results
    
    async def batch_extract_relationships(self, texts: List[str], metadatas: List[Dict] = None) -> List[List[ExtractedRelationship]]:
        """
        Extract relationships from multiple texts concurrently
        
        Args:
            texts: List of input texts
            metadatas: List of metadata dicts for each text
            
        Returns:
            List of relationship lists for each input text
        """
        if metadatas is None:
            metadatas = [{}] * len(texts)
        
        tasks = [
            self.extract_relationships(text, metadata)
            for text, metadata in zip(texts, metadatas)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to process text {i}: {result}")
                valid_results.append([])
            else:
                valid_results.append(result)
        
        return valid_results
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts using Azure OpenAI
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
            
        try:
            # Filter out empty texts
            valid_texts = [text.strip() for text in texts if text and text.strip()]
            if not valid_texts:
                return []
            
            response = await asyncio.to_thread(
                self.client.embeddings.create,
                model=self.config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
                input=valid_texts  # Send list directly
            )
            embeddings = [data.embedding for data in response.data]
            logger.info(f"Generated embeddings for {len(valid_texts)} texts")
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return []
    
    async def answer_query(self, query: str, context: str) -> str:
        """
        Answer a query based on provided context
        
        Args:
            query: User question
            context: Retrieved context from knowledge graph
            
        Returns:
            Generated answer
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPTS["query_response"]},
            {"role": "user", "content": f"Question: {query}\n\nContext from Knowledge Graph:\n{context}\n\nPlease provide a comprehensive answer based on the context."}
        ]
        
        try:
            response = await self._make_api_call(messages, max_tokens=1000)
            return response
            
        except Exception as e:
            logger.error(f"Query answering failed: {e}")
            return "I apologize, but I'm unable to generate a response at this time due to a technical issue."

async def main():
    """Test the LLM service"""
    llm = LLMService()
    
    test_text = "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed."
    
    print("Testing concept extraction...")
    concepts = await llm.extract_concepts(test_text)
    for concept in concepts:
        print(f"- {concept.entity} ({concept.type}): {concept.description}")
    
    print("\nTesting relationship extraction...")
    relationships = await llm.extract_relationships(test_text)
    for rel in relationships:
        print(f"- {rel.source} -> {rel.target}: {rel.relationship}")

if __name__ == "__main__":
    asyncio.run(main())