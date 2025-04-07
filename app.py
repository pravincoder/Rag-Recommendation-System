import streamlit as st
import pysqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from rag import build_chroma_store, pre_processing_csv, ask_query
from sentence_transformers import SentenceTransformer
import chromadb

@st.cache_resource
def load_data(csv_path):
    """Load and process data, caching the results."""
    docs, metas = pre_processing_csv(csv_path)
    collection, model = build_chroma_store(docs, metas)
    return collection, model

# Load data
csv_path = "shl_products.csv"  # replace with your path
collection, model = load_data(csv_path)

# Streamlit app layout
st.title("RAG Model Query Interface")
st.write("Enter a query to get relevant test assessments.")

# User input
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
                st.write("-" * 80)  # Separator for each result
        else:
            st.write("No results found.")
    else:
        st.warning("Please enter a query.")