from typing import List, Dict, Any
import logging
import ollama
from ollama import chat
import chromadb
from chromadb.api.models.Collection import Collection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("rag_system")

# Sample documents about llamas
documents: List[str] = [
    "Llamas are members of the camelid family meaning they're pretty closely related to vicuñas and camels",
    "Llamas were first domesticated and used as pack animals 4,000 to 5,000 years ago in the Peruvian highlands",
    "Llamas can grow as much as 6 feet tall though the average llama between 5 feet 6 inches and 5 feet 9 inches tall",
    "Llamas weigh between 280 and 450 pounds and can carry 25 to 30 percent of their body weight",
    "Llamas are vegetarians and have very efficient digestive systems",
    "Llamas live to be about 20 years old, though some only live for 15 years and others live to be 30 years old",
]

def setup_collection(name: str = "docs") -> Collection:
    """Set up a ChromaDB collection and populate it with document embeddings."""
    client = chromadb.Client()
    
    # Always delete collection if it exists to avoid conflicts
    try:
        client.delete_collection(name)
        logger.info(f"Deleted existing collection '{name}'")
    except Exception as e:
        logger.info(f"No existing collection to delete: {e}")
        
    # Create a new collection
    collection = client.create_collection(name=name)
    logger.info(f"Created new collection '{name}'")
    
    # Store each document in the vector embedding database
    for i, doc in enumerate(documents):
        try:
            # Get embedding from Ollama
            response = ollama.embeddings(model="nomic-embed-text", prompt=doc)
            
            # Extract embeddings from response
            if "embedding" in response:
                embeddings = response["embedding"]
            else:
                embeddings = response.get("embeddings", [])
                
            if not embeddings:
                logger.error(f"No embeddings found in response: {response}")
                raise ValueError("No embeddings found in response")
                
            # Add document to collection
            collection.add(
                ids=[str(i)],
                embeddings=[embeddings],
                documents=[doc]
            )
            logger.info(f"Added document {i} to collection")
        except Exception as e:
            logger.error(f"Error embedding document {i}: {e}")
            # Continue with next document instead of failing completely
            continue
    
    return collection

def query_and_respond(collection: Collection, query: str, model: str = "gemma3") -> str:
    """Query the vector database and generate a response using Ollama chat API.
    
    Args:
        collection: ChromaDB collection to query
        query: The user's question
        model: The Ollama model to use for response generation
        
    Returns:
        str: The generated response
    """
    try:
        logger.info(f"Generating embedding for query: {query}")
        response = ollama.embeddings(model="nomic-embed-text", prompt=query)
        
        # Extract embeddings from response
        if "embedding" in response:
            query_embedding = response["embedding"]
        else:
            query_embedding = response.get("embeddings", [])
            
        if not query_embedding:
            logger.error(f"No embeddings found in response: {response}")
            return "Error: Could not generate embeddings for your query."
        
        logger.info("Querying vector database for relevant documents...")
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=2  # Get top 2 most relevant documents
        )
        
        if not results["documents"] or not results["documents"][0]:
            return "I don't have enough information to answer that question."
        
        # Combine relevant documents for context
        relevant_docs = results['documents'][0]
        context = "\n".join(relevant_docs)
        logger.info(f"Found relevant documents: {context}")
        
        logger.info(f"Generating response using {model}...")
        
        # Use the chat API for better responses
        messages = [
            {
                "role": "system", 
                "content": "You are a helpful assistant that answers questions based on the provided context. If the context doesn't contain enough information to answer the question, say so."
            },
            {
                "role": "user",
                "content": f"Context information:\n{context}\n\nQuestion: {query}\n\nPlease answer the question based only on the context provided."
            }
        ]
        
        response = chat(model=model, messages=messages)
        
        # Extract the response content
        if hasattr(response, 'message') and hasattr(response.message, 'content'):
            return response.message.content
        else:
            # Fallback in case the response format changes
            return str(response)
            
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return f"Error processing query: {str(e)}"

def main():
    # Initialize the collection
    logger.info("Setting up collection...")
    collection = setup_collection()
    
    # Check if documents were added successfully
    try:
        count = len(collection.get()["ids"])
        logger.info(f"Collection contains {count} documents")
        if count == 0:
            logger.error("No documents were added to the collection")
            return
    except Exception as e:
        logger.error(f"Error checking collection: {e}")
    
    # Example query
    query = "What animals are llamas related to?"
    logger.info(f"Processing query: {query}")
    response = query_and_respond(collection, query)
    print(f"\nQuestion: {query}\nAnswer: {response}")
    
    # Additional example queries
    additional_queries = [
        "How tall can llamas grow?",
        "What do llamas eat?",
        "How long do llamas live?"
    ]
    
    for q in additional_queries:
        logger.info(f"Processing query: {q}")
        resp = query_and_respond(collection, q)
        print(f"\nQuestion: {q}\nAnswer: {resp}")

if __name__ == "__main__":
    main()
