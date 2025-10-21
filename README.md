# appl-kgraph
Knowledge Graph Implementation

# Acknowledgement
This is an reimplementation of [LightRAG](https://github.com/HKUDS/LightRAG) and [PathRAG](https://github.com/BUPT-GAMMA/PathRAG).


# Preparation
1. Clone this repo to a folder of your choice
2. In a folder of your choice, create a file named ".env"
3. Using Azure OpenAI Services, enter the variables in the .env file:<br>
    AZURE_OPENAI_API_KEY = "..."<br>
    AZURE_OPENAI_ENDPOINT = "..."<br>
    AZURE_OPENAI_API_VERSION = "..."<br>
    AZURE_OPENAI_LLM_DEPLOYMENT_NAME = "..."<br>
    AZURE_OPENAI_EMB_DEPLOYMENT_NAME = "..."<br>
    LLM_PROVIDER = "azure"
The value of this variable can be found in your Azure OpenAI Services subscription
4. Using OpenAI Services, enter the variables in the .env file:<br>
    OPENAI_API_KEY = "..."<br>
    OPENAI_BASE_URL = "..."<br>
    OPENAI_LLM_MODEL = "..."<br>
    OPENAI_EMBEDDINGS_MODEL = "..."<br>
    LLM_PROVIDER = "openai"
5. In case your documents include .docx files, make sure that Microsoft Word is installed.


# Pip virtual environment setup
1. Open an Anaconda prompt or other command prompt
2. Go to the root folder of the project and create a Python environment with pip with <code>python -m venv venv</code><br>
This will create a basic virtual environment folder named venv in the root of your project folder
NB: The chosen name of the environment folder is here venv. It can be changed to a name of your choice
3. Activate this environment with <code>venv\Scripts\activate</code> for windows or <code>source venv/bin/activate</code> for MacOS
4. All required packages can now be installed with <code>pip install -r requirements.txt</code>

# How to use
Go to the root folder of the project and run <code>python graph/main.py</code> on the terminal.