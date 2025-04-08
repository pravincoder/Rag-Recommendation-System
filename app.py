import streamlit as st
from rag import pre_processing_csv, build_pinecone_store, ask_query
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")  # e.g., "us-west-2"
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "shl-test-index")

# Cache data processing and vector store creation for efficiency
@st.cache_resource
def load_data():
    csv_path = "shl_products.csv"  # Update the path if necessary
    # Step 1: Preprocess the CSV and create document chunks
    documents, metadatas = pre_processing_csv(csv_path)
    
    # Load the SentenceTransformer model (this may take some time at first run)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Step 2: Build the Pinecone vector store with embeddings
    index, model, embeddings, documents, metadatas = build_pinecone_store(
        documents, metadatas, model, PINECONE_INDEX_NAME, PINECONE_API_KEY, PINECONE_ENV
    )
    return index, model

# Load and cache the data, index, and model
index, model = load_data()

# Streamlit UI
st.title("🧠 RAG Model Query Interface")
st.write("Enter a query to retrieve relevant SHL assessments.")

# Query input widget
user_query = st.text_input("Your query:")

if st.button("Submit Query"):
    if user_query:
        # Step 3: Query the RAG model using the Pinecone index
        results = ask_query(user_query, model, index, k=10)
        
        if results:
            st.markdown(f"### Results for: `{user_query}`")
            for i, (doc, meta) in enumerate(results, 1):
                st.markdown(f"**Result {i}:**")
                st.markdown(f"**Test Name:** {meta.get('Test Name', '')}")
                st.markdown(f"**Test Link:** [SHL Link](https://www.shl.com{meta.get('Test Link', '')})")
                st.markdown(f"**Chunk:** {doc}")
                st.markdown("---")
        else:
            st.info("No results found. Please try a different query.")
    else:
        st.warning("Please enter a query to search.")
