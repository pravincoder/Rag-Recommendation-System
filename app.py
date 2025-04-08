import streamlit as st
import tempfile

# Use DuckDB instead of SQLite for ChromaDB
import chromadb
from chromadb.config import Settings
# Import your RAG components
from rag import build_chroma_store, pre_processing_csv, ask_query
from sentence_transformers import SentenceTransformer

# Temporary directory for persistence in Streamlit Cloud
temp_dir = tempfile.TemporaryDirectory()

@st.cache_resource
def load_data(csv_path):
    """Load and process data, caching the results."""
    docs, metas = pre_processing_csv(csv_path)

    # Use DuckDB + Parquet instead of SQLite
    client = chromadb.Client(Settings(
        persist_directory=temp_dir.name  
    ))

    collection, model = build_chroma_store(docs, metas, client=client)
    return collection, model

# Load your CSV data
csv_path = "shl_products.csv"  # Update path if needed
collection, model = load_data(csv_path)

# Streamlit UI
st.title("🧠 RAG Model Query Interface")
st.write("Enter a query to get relevant SHL test assessments.")

# Query input
user_query = st.text_input("Enter your query:")

if st.button("Submit"):
    if user_query:
        results = ask_query(user_query, model, collection)
        if results:
            st.write(f"📊 Results for query: {user_query}")
            st.write("=" * 80)
            for i, (doc, meta) in enumerate(results, 1):
                st.markdown(f"🔹 **Result {i}**")
                st.markdown(f"🧪 **Test Name:** {meta['Test Name']}")
                st.markdown(f"🔗 **Link:** [https://www.shl.com{meta['Test Link']}]")
                st.markdown(f"📄 **Chunk:** {doc}")
                st.write("-" * 80)
        else:
            st.warning("No results found.")
    else:
        st.warning("Please enter a query.")
