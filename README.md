# Enterprise AI Governance & Strategy Knowledge Assistant

A Retrieval-Augmented Generation (RAG) system that enables executive Q&A across enterprise AI governance, cybersecurity risk, and AI business strategy frameworks.

Built using LangChain, OpenAI, ChromaDB, and Streamlit, this project demonstrates how modern data leaders can operationalize structured knowledge systems over authoritative frameworks.

![Architecture Diagram](architecture_diagram.png)

The system implements a standard Retrieval-Augmented Generation (RAG) pipeline:

PDFs → Text Chunking → Embeddings → Vector Store (ChromaDB)
      → Retriever → LLM → Structured Answer + Source Citation

## Tech Stack

* **Language:** Python 3.10+
* **Orchestrator:** LangChain (LCEL Architecture)
* **LLM:** OpenAI gpt-4o-mini
* **Embeddings:** OpenAI text-embedding-3-small
* **Vector Database:** ChromaDB (Local Persistence)
* **Frontend:** Streamlit

## How it works
1- All PDFs in /data are ingested.
2- Documents are chunked into semantic segments.
3- Chunks are embedded and stored in ChromaDB.
4- User queries retrieve the most relevant context.
5- The LLM answers strictly using retrieved content.
6- Source references are appended with page numbers.

The assistant does not hallucinate external knowledge; it answers only from the ingested frameworks.


## Setup & Installation
1. Prerequisites

    Python 3.10 or higher

    An OpenAI API Key

2. Installation

Clone the repository and install dependencies:
	
	git clone https://github.com/danielrodriguezrivera/enterprise-ai-governance-rag
	cd enterprise-ai-governance-rag
	pip install -r requirements.txt

3. Environment Configuration

Create a .env file in the root directory and add your API key:

	OPENAI_API_KEY=sk-proj-your-actual-api-key-here

Usage
Step 1: Ingest Data

The ingestion script is located in the src folder. Run it from the root directory to ensure relative paths to ./data and ./vector_db resolve correctly.
Bash

	python src/ingestion.py

This will scan the data/ folder, chunk the PDFs, and populate vector_db/.

Step 2: Launch Application

Start the Streamlit interface to query your documents.

	streamlit run app.py


## System Architecture

Ingestion Pipeline (src/ingestion.py)

    Extract: Loads PDFs from data/ using PyPDFLoader.

    Chunk: Splits text into 1000-character chunks with 200-character overlap using RecursiveCharacterTextSplitter.

    Embed: Converts text to vectors using text-embedding-3-small.

    Store: Persists vectors to vector_db/.

Retrieval Pipeline (app.py)

    Query: User submits a question via Streamlit.

    Search: System performs MMR Search (k=12, fetch_k=50) on vector_db to find diverse context.

    Generate: GPT-4o-mini synthesizes an answer using the retrieved context.

    Cite: The UI parses metadata to display the source document name and page number.

Author

Daniel Edgardo Rodriguez Rivera

    LinkedIn Profile: https://www.linkedin.com/in/daniel-rodriguez-sv
    GitHub Repository: https://github.com/danielrodriguezrivera/