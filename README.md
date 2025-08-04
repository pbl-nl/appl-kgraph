# appl-kgraph
Basic Knowledge Graph Implementation

# Acknowledgement
This is an extension of [rahulnyk](https://github.com/rahulnyk/knowledge_graph/tree/main)'s codebase of knowledge graph by Claude AI.


### Preparation
1. Clone this repo to a folder of your choice
2. In a folder of your choice, create a file named ".env"
3. Using Azure OpenAI Services, enter the variables in the .env file:<br>
    AZURE_OPENAI_API_KEY = "..."<br>
    AZURE_OPENAI_ENDPOINT = "..."<br>
    AZURE_OPENAI_API_VERSION = "..."<br>
    AZURE_OPENAI_DEPLOYMENT_NAME = "..."<br>
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT = "..."<br>
The value of this variable can be found in your Azure OpenAI Services subscription

### Pip virtual environment setup
1. Open an Anaconda prompt or other command prompt
2. Go to the root folder of the project and create a Python environment with pip with <code>python -m venv venv</code><br>
This will create a basic virtual environment folder named venv in the root of your project folder
NB: The chosen name of the environment folder is here venv. It can be changed to a name of your choice
3. Activate this environment with <code>venv\Scripts\activate</code>
4. All required packages can now be installed with <code>pip install -r requirements.txt</code>

### How to use

<code>
    
    # First time (builds from scratch)
    python main.py --build
    
    # Add new documents to data_input/ and run (incremental update)
    python main.py --build
    
    # Force complete rebuild
    python main.py --rebuild
    
    # Query the system
    python main.py --query "What are the main causes of India's health workforce shortages?"

    # 4. Get detailed results
    python main.py --query "What are the main causes of India's health workforce shortages?" --detailed

    # Interactive mode for multiple questions
    python main.py --interactive
    # >>> Enter your query: What are India's health workforce shortages?
    # >>> Enter your query: How do public and private sectors differ in healthcare delivery?
    # >>> Enter your query: What policies address these workforce shortages?
    # >>> Enter your query: viz  # Create visualization
    # >>> Enter your query: stats  # See graph statistics
    # >>> Enter your query: quit  # Exit

    # Run all tests
    python -m pytest test_system.py -v
    
    # Run only unit tests (skip slow integration tests)
    python -m pytest test_system.py -v -m "not slow"
    
    # Run with coverage
    pip install pytest-cov
    python -m pytest test_system.py --cov=. --cov-report=html

    # Alternative test command
    python test_system.py

    # Generate interactive graph visualization
    python main.py --visualize

    # See graph statistics
    python main.py --stats

</code>

