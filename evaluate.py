import numpy as np
from typing import List, Dict, Set
import pandas as pd
from rag import pre_processing_csv, build_pinecone_store  # updated to use Pinecone store
from sentence_transformers import SentenceTransformer
import os

class Evaluator:
    def __init__(self, ground_truth: Dict[str, Set[str]]):
        """
        Initialize evaluator with ground truth data.
        ground_truth: Dictionary mapping query to set of relevant test IDs
        """
        self.ground_truth = ground_truth
        self.total_queries = len(ground_truth)

    def recall_at_k(self, retrieved: List[str], relevant: Set[str], k: int) -> float:
        """
        Calculate Recall@K for a single query.
        """
        if not relevant:
            return 0.0
        
        retrieved_at_k = set(retrieved[:k])
        relevant_retrieved = retrieved_at_k.intersection(relevant)
        return len(relevant_retrieved) / len(relevant)

    def average_precision_at_k(self, retrieved: List[str], relevant: Set[str], k: int) -> float:
        """
        Calculate Average Precision@K for a single query.
        """
        if not relevant:
            return 0.0

        score = 0.0
        num_relevant = 0
        for i in range(min(k, len(retrieved))):
            if retrieved[i] in relevant:
                num_relevant += 1
                score += num_relevant / (i + 1)
        normalization = min(k, len(relevant)) if len(relevant) > 0 else 1
        return score / normalization

    def evaluate(self, model: SentenceTransformer, index, k: int = 10) -> Dict[str, float]:
        """
        Evaluate the retrieval system using Mean Recall@K and MAP@K.
        """
        mean_recall = 0.0
        mean_ap = 0.0

        # Loop over each query in the ground truth.
        for query, relevant_tests in self.ground_truth.items():
            # Generate the query embedding.
            query_embedding = model.encode([query]).tolist()[0]
            # Query the Pinecone index (top_k matches).
            response = index.query(vector=query_embedding, top_k=k, include_metadata=True)
            
            # Extract the "Test Name" from each match's metadata.
            retrieved_tests = [
                match["metadata"].get("Test Name", "") for match in response.get("matches", [])
            ]
            
            recall = self.recall_at_k(retrieved_tests, relevant_tests, k)
            ap = self.average_precision_at_k(retrieved_tests, relevant_tests, k)
            mean_recall += recall
            mean_ap += ap

        mean_recall /= self.total_queries
        mean_ap /= self.total_queries

        return {
            'Mean Recall@K': mean_recall,
            'MAP@K': mean_ap
        }

def main():
    # Example ground truth data: mapping each query string to a set of relevant test names.
    ground_truth = {
        "I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes.": {            
            "Contact Center Customer Service + 8.0",
            "Contact Center Sales & Service + 8.0",
            "Technician/Technologist Solution",
            ".NET Framework 4.5",
            ".NET MVC (New)",
            ".NET MVVM (New)",
            ".NET WCF (New)",
            ".NET WPF (New)",
        },
        "Looking to hire mid-level professionals who are proficient in Python, SQL and Java Script. Need an assessment package that can test all skills with max duration of 60 minutes.": {
            "Automata - SQL (New)",
            "Automata Front End",
            "JavaScript (New)",
            "Microsoft SQL Server 2014 Programming",
            "Oracle DBA (Advanced Level) (New)",
            "Oracle DBA (Entry Level) (New)",
            "Oracle PL/SQL (New)",
            "Python (New)",
            "SQL (New)",
            "SQL Server (New)",
            "SQL Server Analysis Services (SSAS) (New)",
        },
    }

    evaluator = Evaluator(ground_truth)

    # Load and process data
    csv_path = "shl_products.csv"
    docs, metas = pre_processing_csv(csv_path)

    # Load the SentenceTransformer model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Build the Pinecone vector store.
    # Environment variables are used for the index name, API key, and region.
    index, _, _, _, _ = build_pinecone_store(
        docs,
        metas,
        model,
        os.getenv("PINECONE_INDEX_NAME", "shl-test-index"),
        os.getenv("PINECONE_API_KEY"),
        os.getenv("PINECONE_ENV", "us-west-2")
    )

    # Evaluate with different values of K.
    k_values = [5, 10, 15]
    for k in k_values:
        print(f"\nEvaluating with K={k}")
        metrics = evaluator.evaluate(model, index, k)
        print(f"Mean Recall@{k}: {metrics['Mean Recall@K']:.4f}")
        print(f"MAP@{k}: {metrics['MAP@K']:.4f}")

if __name__ == "__main__":
    main()
