# Ollama RAG System with ChromaDB

A simple Retrieval Augmented Generation (RAG) system using Ollama for embeddings and text generation, with ChromaDB as the vector store.

## Overview

This system demonstrates how to:
- Embed documents using Ollama's nomic-embed-text model
- Store embeddings in ChromaDB
- Retrieve relevant documents based on query similarity
- Generate natural language responses using Ollama's gemma3 model

## Requirements

```python
pip install ollama chromadb
```

You'll also need Ollama running locally with the following models:
	•	nomic-embed-text (for embeddings)
	•	gemma3 (for text generation)

## Usage

The system comes pre-loaded with sample documents about llamas. To use:
	1.	Run the script:
```python
python main.py
```
	2.	The system will:
	▪	Create a new ChromaDB collection
	▪	Embed and store the sample documents
	▪	Run example queries to demonstrate functionality

## Key Components
	•	⁠setup_collection(): Creates and populates the vector database
	•	⁠query_and_respond(): Handles query processing and response generation
	•	Sample documents about llamas for demonstration

## Example Output
Question: What animals are llamas related to?
Answer: Llamas are related to vicuñas and camels.

Question: How tall can llamas grow?
Answer: Llamas can grow as much as 6 feet tall.

## Features
	•	Vector similarity search for relevant document retrieval
	•	Chat-based response generation
	•	Comprehensive logging
	•	Error handling and collection management
	•	Support for multiple relevant documents per query

## Limitations
	•	Currently uses in-memory storage for ChromaDB
	•	Limited to pre-loaded sample documents
	•	Single-model configuration

## Future Improvements
	•	Add persistent storage support
	•	Add persistent storage support
	•	Implement document loading from files
	•	Add support for different embedding and generation models
	•	Add interactive query interface
