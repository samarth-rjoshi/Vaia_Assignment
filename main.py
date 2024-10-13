from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import requests
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_chroma import Chroma
import uuid
import os

# Set Hugging Face API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

app = FastAPI()

# Initialize the model for embedding
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize vector database client (e.g., ChromaDB)
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
)

# Initialize the text generation model
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
    max_new_tokens=4096,  # Increased token limit to capture more content
    do_sample=False,
)

# Models for input data
class URLRequest(BaseModel):
    url: str

class QueryRequest(BaseModel):
    query: str

def extract_text_from_url(url: str, depth: int = 1) -> List[str]:
    """
    Extract text from the provided URL, including recursive extraction from links up to a specified depth.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        soup = BeautifulSoup(response.text, 'html.parser')
        text_chunks = []

        # Extract text at the current depth level
        for p in soup.find_all('p'):
            text_chunks.append(p.get_text(strip=True))

        # If depth > 1, recursively extract text from links
        if depth > 1:
            for link in soup.find_all('a', href=True):
                sub_url = link['href']
                # Ensure the link is absolute
                if not sub_url.startswith("http"):
                    sub_url = requests.compat.urljoin(url, sub_url)
                
                # Add the sub-url text
                text_chunks.extend(extract_text_from_url(sub_url, depth - 1))
        
        return text_chunks
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error fetching URL: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error extracting text: {str(e)}")


@app.get("/")
def read_root():
    """
    Root endpoint to provide a welcome message.
    """
    return {"message": "Welcome to the URL parsing and querying API!"}

@app.post("/url-parser")
def url_parser(request: URLRequest):
    """
    Extracts text from a URL, converts it into embeddings, and stores them in the vector database.
    """
    # Extract text chunks from the URL
    text_chunks = extract_text_from_url(request.url)
    
    # Generate Document objects for each text chunk
    documents = [Document(page_content=chunk) for chunk in text_chunks]
    ids = [str(uuid.uuid4()) for _ in range(len(documents))]

    # Store the embeddings in the vector store with unique IDs
    vector_store.add_documents(documents=documents, ids=ids)

    return {"message": "Embeddings successfully stored", "chunks_count": len(text_chunks)}

@app.post("/query")
def query(request: QueryRequest):
    """
    Searches for the most relevant text chunk based on a query using stored embeddings and uses the LLM to generate a response.
    """
    try:
        # Perform a similarity search to find relevant text chunks
        results = vector_store.similarity_search(query=request.query)

        if not results:
            raise HTTPException(status_code=404, detail="No relevant content found")

        # Concatenate the top matching chunks to provide more context to the LLM
        context = "\n".join([result.page_content for result in results])

        # Create a prompt combining the query and the context
        prompt = f"Context: {context}\n\nQuery: {request.query}\nAnswer based on the above context:"

        # Generate a response using the LLM
        llm_response = llm(prompt)

        # Return the full LLM response for debugging
        return {"llm_response": llm_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
