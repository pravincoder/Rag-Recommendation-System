import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import uuid
import numpy as np
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")  # e.g., "us-west-2"
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "shl-test-index")

# === STEP 1: Preprocessing CSV & Chunking ===
def pre_processing_csv(csv_path):
    df = pd.read_csv(csv_path)
    df.fillna("", inplace=True)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    documents = []
    metadatas = []

    for idx, row in df.iterrows():
        combined_text = (
            f"Test Name: {row.get('Test Name', '')}\n"
            f"Description: {row.get('Description', '')}\n"
            f"Remote Testing: {row.get('Remote Testing', '')}\n"
            f"Adaptive/IRT: {row.get('Adaptive/IRT', '')}\n"
            f"Test Type: {row.get('Test Type', '')}\n"
        )
        
        chunks = text_splitter.split_text(combined_text)

        for chunk in chunks:
            documents.append(chunk)
            metadatas.append({
                "Test Name": row.get('Test Name', ''),
                "Test Link": row.get('Test Link', ''),
                "Remote Testing": row.get('Remote Testing', ''),
                "Adaptive/IRT": row.get('Adaptive/IRT', ''),
                "Test Type": row.get('Test Type', ''),
                "row_id": idx
            })
    
    return documents, metadatas

# === STEP 2: Embed and Store in Pinecone ===
def build_pinecone_store(documents, metadatas, model, index_name, pinecone_api_key, pinecone_env):
    print("🔍 Embedding documents...")
    embeddings = model.encode(documents, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    print("🔑 Initializing Pinecone client...")
    # Import new classes from the pinecone package
    from pinecone import Pinecone, ServerlessSpec

    # Create a Pinecone client instance
    pc = Pinecone(api_key=pinecone_api_key)
    
    # Check if the index exists; if not, create a new one.
    existing_indexes = pc.list_indexes().names()
    if index_name not in existing_indexes:
        print("📥 Creating new Pinecone index...")
        pc.create_index(
            name=index_name,
            dimension=embeddings.shape[1],
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=pinecone_env)
        )
        # Optionally, you might need to wait a few moments for the new index to be ready.

    # Connect to the index
    index = pc.Index(index_name)

    print("📥 Upserting embeddings to Pinecone index...")
    to_upsert = []
    for i, (vec, meta) in enumerate(zip(embeddings, metadatas)):
        # Create a unique document id
        doc_id = str(uuid.uuid4())
        # Save the document text in metadata to return during queries
        meta_copy = meta.copy()
        meta_copy["document"] = documents[i]
        # Prepare tuple (id, vector, metadata)
        to_upsert.append((doc_id, vec.tolist(), meta_copy))
    
    # Upsert documents as a single batch (for large datasets, consider batching the upserts)
    index.upsert(vectors=to_upsert)

    return index, model, embeddings, documents, metadatas

# === STEP 3: Query the RAG Model using Pinecone ===
def ask_query(query, model, index, k=10):
    print(f"\n💬 Query: {query}")
    # Generate query embedding
    query_embedding = model.encode([query]).tolist()[0]
    # Query Pinecone (retrieve extra candidates to filter duplicates)
    query_response = index.query(vector=query_embedding, top_k=k * 2, include_metadata=True)

    seen_tests = set()
    final_results = []

    # Loop through matches and filter for unique "Test Name"
    for match in query_response['matches']:
        meta = match.get('metadata', {})
        test_name = meta.get("Test Name", "")
        if test_name in seen_tests:
            continue
        seen_tests.add(test_name)
        # Retrieve the stored document text from metadata
        doc = meta.get("document", "")
        final_results.append((doc, meta))
        if len(final_results) >= k:
            break

    return final_results

# === Example Usage ===
if __name__ == "__main__":
    # Path to your CSV file
    csv_path = "shl_products.csv"
    
    # Step 1: Preprocess CSV and create document chunks
    documents, metadatas = pre_processing_csv(csv_path)
    
    # Load the SentenceTransformer model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Step 2: Build the Pinecone vector store
    index, model, embeddings, documents, metadatas = build_pinecone_store(
        documents, metadatas, model, PINECONE_INDEX_NAME, PINECONE_API_KEY, PINECONE_ENV
    )
    
    # Step 3: Query the RAG model
    sample_query = "I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes."
    results = ask_query(sample_query, model, index, k=10)
    
    # Display the results
    print(f"\nResults for query: {sample_query}\n{'='*80}")
    for i, (doc, meta) in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"Test Name: {meta.get('Test Name', '')}")
        print(f"Test Link: https://www.shl.com{meta.get('Test Link', '')}")
        print(f"Chunk: {doc}")
        print("-" * 80)
