"""
Main application for the Knowledge Graph system
"""
import asyncio
import argparse
import sys
from pathlib import Path
import json
import logging
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('knowledge_graph.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import modules
from config import Config
from document_processor import DocumentProcessor
from llm_service import LLMService
from knowledge_graph import KnowledgeGraph, KnowledgeGraphBuilder
from query_engine import QueryEngine
from visualization import GraphVisualizer

class KnowledgeGraphApp:
    """Main application class for the Knowledge Graph system"""
    
    def __init__(self):
        self.config = Config()
        self.kg_builder = None
        self.kg = None
        self.query_engine = None
        self.visualizer = None
        
        # Validate configuration
        try:
            self.config.validate()
            logger.info("Configuration validated successfully")
        except ValueError as e:
            logger.error(f"Configuration error: {e}")
            raise
    
    async def initialize(self):
        """Initialize the application components"""
        logger.info("Initializing Knowledge Graph application...")
        
        # Initialize components
        self.kg_builder = KnowledgeGraphBuilder(self.config)
        self.kg = KnowledgeGraph(self.config)
        
        # Load existing knowledge graph
        self.kg.load_graph()
        
        # Initialize query engine and visualizer
        llm_service = LLMService(self.config)
        self.query_engine = QueryEngine(self.kg, llm_service)
        self.visualizer = GraphVisualizer(self.kg, self.config)
        
        logger.info("Application initialized successfully")
    
    async def build_knowledge_graph(self, force_rebuild: bool = False):
        """Build or update the knowledge graph from documents"""
        logger.info(f"Building knowledge graph (force_rebuild={force_rebuild})...")
        
        try:
            # Build knowledge graph
            self.kg = await self.kg_builder.build_from_documents(force_rebuild=force_rebuild)
            
            # Update query engine and visualizer with new graph
            self.query_engine.kg = self.kg
            self.visualizer.kg = self.kg
            
            # Print statistics
            stats = self.kg.get_graph_statistics()
            logger.info("Knowledge Graph built successfully:")
            logger.info(f"  - Total nodes: {stats['total_nodes']}")
            logger.info(f"  - Total edges: {stats['total_edges']}")
            logger.info(f"  - Average degree: {stats['avg_degree']:.2f}")
            logger.info(f"  - Concept types: {stats['concept_types']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to build knowledge graph: {e}")
            return False
    
    async def query_knowledge_graph(self, query: str, detailed: bool = False):
        """Query the knowledge graph and return results"""
        if not self.query_engine:
            logger.error("Query engine not initialized")
            return None
        
        try:
            logger.info(f"Processing query: {query}")
            
            if detailed:
                # Return detailed search results
                search_result = await self.query_engine.search(query)
                return {
                    'query': query,
                    'answer': await self.query_engine.answer_query(query),
                    'concepts': search_result.concepts[:10],
                    'relationships': search_result.relationships[:10],
                    'paths': search_result.paths[:5],
                    'relevance_score': search_result.relevance_score
                }
            else:
                # Return just the answer
                answer = await self.query_engine.answer_query(query)
                return {
                    'query': query,
                    'answer': answer
                }
                
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                'query': query,
                'answer': f"Error processing query: {e}",
                'error': True
            }
    
    def create_visualization(self, output_path: Optional[str] = None):
        """Create interactive visualization of the knowledge graph"""
        if not self.visualizer:
            logger.error("Visualizer not initialized")
            return None
        
        try:
            output_file = self.visualizer.create_interactive_visualization(
                output_path=Path(output_path) if output_path else None
            )
            logger.info(f"Visualization created: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Visualization creation failed: {e}")
            return None
    
    def get_statistics(self):
        """Get knowledge graph statistics"""
        if not self.kg:
            return {"error": "Knowledge graph not loaded"}
        
        stats = self.kg.get_graph_statistics()
        viz_stats = self.visualizer.create_concept_overview() if self.visualizer else {}
        
        return {
            **stats,
            **viz_stats
        }
    
    def get_concept_details(self, concept: str):
        """Get detailed information about a specific concept"""
        if not self.query_engine:
            return {"error": "Query engine not initialized"}
        
        return self.query_engine.get_concept_details(concept)
    
    async def interactive_mode(self):
        """Run interactive query mode"""
        print("\n" + "="*60)
        print("Knowledge Graph Interactive Query System")
        print("="*60)
        print("Commands:")
        print("  - Type your question to query the knowledge graph")
        print("  - 'stats' to see graph statistics")
        print("  - 'viz' to create a visualization")
        print("  - 'concept <name>' to get concept details")
        print("  - 'rebuild' to rebuild the knowledge graph")
        print("  - 'quit' or 'exit' to quit")
        print("="*60)
        
        while True:
            try:
                user_input = input("\n🤖 Enter your query: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                elif user_input.lower() == 'stats':
                    stats = self.get_statistics()
                    print("\n📊 Knowledge Graph Statistics:")
                    print(json.dumps(stats, indent=2, default=str))
                
                elif user_input.lower() == 'viz':
                    print("🎨 Creating visualization...")
                    viz_file = self.create_visualization()
                    if viz_file:
                        print(f"✅ Visualization saved to: {viz_file}")
                    else:
                        print("❌ Failed to create visualization")
                
                elif user_input.lower().startswith('concept '):
                    concept_name = user_input[8:].strip()
                    details = self.get_concept_details(concept_name)
                    print(f"\n🔍 Details for '{concept_name}':")
                    print(json.dumps(details, indent=2, default=str))
                
                elif user_input.lower() == 'rebuild':
                    print("🔨 Rebuilding knowledge graph...")
                    success = await self.build_knowledge_graph(force_rebuild=True)
                    if success:
                        print("✅ Knowledge graph rebuilt successfully")
                    else:
                        print("❌ Failed to rebuild knowledge graph")
                
                else:
                    # Process as query
                    print("🔍 Searching knowledge graph...")
                    result = await self.query_knowledge_graph(user_input)
                    
                    if result:
                        print(f"\n💡 Answer:")
                        print(result['answer'])
                        
                        if result.get('error'):
                            print("\n⚠️  There was an error processing your query.")
                    else:
                        print("❌ Failed to process query")
            
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Knowledge Graph System')
    parser.add_argument('--build', action='store_true', help='Build knowledge graph from documents')
    parser.add_argument('--rebuild', action='store_true', help='Rebuild knowledge graph from scratch')
    parser.add_argument('--query', type=str, help='Query the knowledge graph')
    parser.add_argument('--detailed', action='store_true', help='Return detailed query results')
    parser.add_argument('--visualize', action='store_true', help='Create knowledge graph visualization')
    parser.add_argument('--stats', action='store_true', help='Show knowledge graph statistics')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--concept', type=str, help='Get details about a specific concept')
    parser.add_argument('--output', type=str, help='Output file path for visualization')
    
    args = parser.parse_args()
    
    # Initialize application
    app = KnowledgeGraphApp()
    await app.initialize()
    
    # Handle different modes
    if args.build or args.rebuild:
        success = await app.build_knowledge_graph(force_rebuild=args.rebuild)
        if not success:
            sys.exit(1)
    
    if args.stats:
        stats = app.get_statistics()
        print("Knowledge Graph Statistics:")
        print(json.dumps(stats, indent=2, default=str))
    
    if args.concept:
        details = app.get_concept_details(args.concept)
        print(f"Concept Details for '{args.concept}':")
        print(json.dumps(details, indent=2, default=str))
    
    if args.query:
        result = await app.query_knowledge_graph(args.query, detailed=args.detailed)
        if result:
            if args.detailed:
                print("Detailed Query Result:")
                print(json.dumps(result, indent=2, default=str))
            else:
                print("Answer:")
                print(result['answer'])
        else:
            print("Failed to process query")
            sys.exit(1)
    
    if args.visualize:
        viz_file = app.create_visualization(args.output)
        if viz_file:
            print(f"Visualization saved to: {viz_file}")
        else:
            print("Failed to create visualization")
            sys.exit(1)
    
    if args.interactive:
        await app.interactive_mode()
    
    # If no specific action, show help
    if not any([args.build, args.rebuild, args.query, args.visualize, 
               args.stats, args.interactive, args.concept]):
        parser.print_help()
        print("\nTip: Use --interactive for an interactive query session")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)