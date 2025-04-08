import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid
from chromadb import PersistentClient


# === STEP 1: Preprocessing CSV & Chunking ===
def pre_processing_csv(csv_path):
    df = pd.read_csv(csv_path)
    df.fillna("", inplace=True)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50
    )

    documents = []
    metadatas = []

    for idx, row in df.iterrows():
        combined_text = f"""
        Test Name: {row.get('Test Name', '')}
        Description: {row.get('Description', '')}
        Remote Testing: {row.get('Remote Testing', '')}
        Adaptive/IRT: {row.get('Adaptive/IRT', '')}
        Test Type: {row.get('Test Type', '')}
        """
        
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

# === STEP 2: Embed and Store in ChromaDB ===
def build_chroma_store(documents, metadatas, persist_directory="./chroma_db"):
    client = PersistentClient(path=persist_directory)
    collection = client.get_or_create_collection(name="shl_test_catalog")

    print("🔍 Embedding documents...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(documents, show_progress_bar=True)

    print("📥 Adding to ChromaDB...")
    collection.add(
        documents=documents,
        embeddings=[e.tolist() for e in embeddings],
        ids=[str(uuid.uuid4()) for _ in range(len(documents))],
        metadatas=metadatas
    )

    return collection, model

# === STEP 3: Query the RAG Model ===
def ask_query(query, model, collection, k=10):
    print(f"\n💬 Query: {query}")
    query_embedding = model.encode(query)
    
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=k * 2
    )

    seen_tests = set()
    final_results = []
    
    for i in range(len(results['documents'][0])):
        doc = results['documents'][0][i]
        meta = results['metadatas'][0][i]
        test_name = meta['Test Name']
        
        if test_name in seen_tests:
            continue
            
        seen_tests.add(test_name)
        final_results.append((doc, meta))
        
        if len(final_results) >= k:
            break

    return final_results
