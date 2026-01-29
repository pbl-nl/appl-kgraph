# appl-kgraph

**appl-kgraph** is a modular **graph-based Retrieval-Augmented Generation (RAG)** system with pluggable retrieval strategies. It is designed for querying large document collections by combining classic vector-based retrieval with an explicit **knowledge graph** over entities and relations extracted from the source texts.

The project builds on ideas from recent graph-based RAG research, most notably **LightRAG** and **PathRAG**, and provides a shared graph + vector indexing layer on top of which multiple retrieval strategies can be implemented, compared, and extended.

---

## What this project is for

Traditional RAG systems rely on chunking documents and retrieving the most similar text fragments using vector embeddings. While effective for local, factual questions, this approach often struggles with:

* Global or cross-document questions
* Redundant or noisy retrieval results
* Capturing relationships between entities across chunks

**appl-kgraph** addresses these limitations by:

* Building a **knowledge graph** from documents (entities as nodes, relations as edges)
* Storing textual evidence alongside nodes and edges
* Combining **vector search** with **graph-aware retrieval**

This enables richer retrieval for:

* Exploratory questions
* Summarization and sensemaking tasks
* Queries that require reasoning over multiple documents or concepts

---

## Supported retrieval strategies

The system supports multiple retrieval paradigms through a pluggable retriever design. All retrievers share the same underlying document ingestion, embedding, and graph construction pipeline, but differ in how retrieved context is selected and structured before being passed to the language model.

At a minimum, the system can operate in a **vanilla vector-based RAG** mode. In this configuration, documents are chunked, embedded, and retrieved purely based on embedding similarity. This approach is simple, fast, and often effective for local factual questions. However, because it treats the document collection as a flat set of chunks, it does not explicitly model relationships between entities or concepts. As a result, it can struggle with cross-document questions, global summaries, and queries that require reasoning over connections rather than isolated passages. In this project, vanilla RAG is primarily included as a baseline for comparison and ablation.

### LightRAG-style retrieval

Inspired by *LightRAG: Simple and Fast Retrieval-Augmented Generation*, this retrieval strategy augments vector-based retrieval with a lightweight knowledge graph over entities and relations extracted from the documents. Instead of retrieving raw chunks directly, LightRAG-style retrieval operates over graph elements that are associated with textual descriptions.

Concretely, the query is first analyzed to extract both **low-level (specific)** and **high-level (abstract)** keywords. Low-level retrieval focuses on precise entities and relations relevant to the query, while high-level retrieval targets broader themes and concepts. Retrieved graph elements are then expanded with their immediate neighbors (ego networks), allowing the retriever to capture local structure while still incorporating global context.

This dual-level design makes LightRAG-style retrieval efficient and flexible. It performs well on large corpora and mixed query types, supports incremental updates to the document collection, and typically incurs lower token overhead than community-based or exhaustive graph traversal methods.

### PathRAG-style retrieval

Inspired by *PathRAG: Pruning Graph-Based Retrieval Augmented Generation with Relational Paths*, this retrieval strategy further refines graph-based retrieval by explicitly focusing on **relational paths** between query-relevant entities.

Rather than retrieving all neighbors of relevant nodes, PathRAG-style retrieval first identifies a set of candidate nodes using vector similarity. It then searches for connecting paths between these nodes in the knowledge graph and applies a pruning procedure to discard weak or noisy connections. Only a small number of high-quality paths are retained and converted into structured textual representations.

These paths are passed to the language model in an order that favors logical coherence, helping the model reason over chains of relationships instead of a flat list of facts. This approach is particularly effective for global, multi-hop questions, long-form answers, and summarization tasks, where redundancy reduction and structured context matter more than raw recall.

---

## High-level architecture

At a conceptual level, the pipeline consists of:

1. **Ingestion & indexing**

   * Document chunking
   * Entity and relation extraction via LLMs
   * Knowledge graph construction
   * Vector embedding of entities, relations, and/or chunks

2. **Retrieval**

   * Query analysis (keyword extraction, embedding)
   * Vector-based candidate selection
   * Graph-based expansion or path selection (LightRAG or PathRAG)

3. **Generation**

   * Structured prompt construction
   * Answer generation using an LLM

The codebase is intentionally modular so that different retrievers and prompt strategies can be swapped, compared, or combined.

---

## Acknowledgement

This project is a re-implementation and adaptation of ideas from:

* *LightRAG: Simple and Fast Retrieval-Augmented Generation* (Guo et al.)
* *PathRAG: Pruning Graph-Based Retrieval Augmented Generation with Relational Paths* (Chen et al.)

The goal here is not a verbatim reproduction, but a practical, inspectable system that can be extended and experimented with.

---

## Getting started

### Preparation

1. Clone this repository to a folder of your choice.
2. In the root folder, create a file named `.env`.

### LLM configuration

You can use either **Azure OpenAI** or **OpenAI** directly.

**Azure OpenAI**:

```
AZURE_OPENAI_API_KEY="..."
AZURE_OPENAI_ENDPOINT="..."
AZURE_OPENAI_API_VERSION="..."
AZURE_OPENAI_LLM_DEPLOYMENT_NAME="..."
AZURE_OPENAI_EMB_DEPLOYMENT_NAME="..."
LLM_PROVIDER="azure"
```

**OpenAI**:

```
OPENAI_API_KEY="..."
OPENAI_BASE_URL="..."
OPENAI_LLM_MODEL="..."
OPENAI_EMBEDDINGS_MODEL="..."
LLM_PROVIDER="openai"
```

If your document collection includes `.docx` files, ensure that **Microsoft Word** is installed.

---

### Python environment setup

1. Open a terminal (Anaconda Prompt or standard shell)
2. Navigate to the project root
3. Create a virtual environment:

   ```
   python -m venv venv
   ```
4. Activate it:

   * Windows: `venv\Scripts\activate`
   * macOS/Linux: `source venv/bin/activate`
5. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

---

### Running the project

From the project root, run:

```
python graph/main.py
```

This will execute the current end-to-end pipeline using the configured retrieval strategy.
Upon running main.py, user input wil be requested as the query, while the ingestion will be performed on the documents located in the *docs* folder.

---

## Project status and scope

This repository implements a **general graph-based RAG system**. Retrieval strategies (e.g. naive vector-based RAG, LightRAG-style graph expansion, PathRAG-style path pruning) are treated as interchangeable components.

In addition to graph-based retrievers, the system also supports a **naive vector-based RAG** setup (chunking + embedding similarity) as a baseline. This mode is primarily intended for comparison and ablation, rather than as the main focus of the project.

The system is currently in development and the roadmap includes implementing a UI and broader multi-modal support.

---

## Citation and contact

If you use ideas, code, or design patterns from this project in academic or applied work, please cite the relevant original papers (LightRAG, PathRAG) and reference this repository where appropriate.

For questions, feedback, or collaboration inquiries, you can contact the maintainers at:

📧 <a href='mailto:stefan.troost@pbl.nl, k.wittenberg@scp.nl'>Contact link</a>



