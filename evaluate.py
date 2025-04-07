import numpy as np
from typing import List, Dict, Set
import pandas as pd
from rag import pre_processing_csv, build_chroma_store, ask_query
from sentence_transformers import SentenceTransformer
import chromadb

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
        retrieved: List of retrieved test IDs
        relevant: Set of relevant test IDs
        k: Number of top results to consider
        """
        if not relevant:
            return 0.0
        
        retrieved_at_k = set(retrieved[:k])
        relevant_retrieved = retrieved_at_k.intersection(relevant)
        return len(relevant_retrieved) / len(relevant)

    def average_precision_at_k(self, retrieved: List[str], relevant: Set[str], k: int) -> float:
        """
        Calculate Average Precision@K for a single query.
        retrieved: List of retrieved test IDs
        relevant: Set of relevant test IDs
        k: Number of top results to consider
        """
        if not relevant:
            return 0.0

        score = 0.0
        num_relevant = 0
        relevant_at_k = min(k, len(relevant))

        for i in range(k):
            if retrieved[i] in relevant:
                num_relevant += 1
                precision_at_i = num_relevant / (i + 1)
                score += precision_at_i

        return score / relevant_at_k if relevant_at_k > 0 else 0.0

    def evaluate(self, model: SentenceTransformer, collection: chromadb.Collection, k: int = 10) -> Dict[str, float]:
        """
        Evaluate the system using Mean Recall@K and MAP@K.
        Returns dictionary with evaluation metrics.
        """
        mean_recall = 0.0
        mean_ap = 0.0

        for query, relevant_tests in self.ground_truth.items():
            # Get results from the system
            results = collection.query(
                query_embeddings=[model.encode(query).tolist()],
                n_results=k
            )
            
            retrieved_tests = [meta['Test Name'] for meta in results['metadatas'][0]]
            
            # Calculate metrics for this query
            recall = self.recall_at_k(retrieved_tests, relevant_tests, k)
            ap = self.average_precision_at_k(retrieved_tests, relevant_tests, k)
            
            mean_recall += recall
            mean_ap += ap

        # Calculate means
        mean_recall /= self.total_queries
        mean_ap /= self.total_queries

        return {
            'Mean Recall@K': mean_recall,
            'MAP@K': mean_ap
        }

def main():
    # Example ground truth data
    
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

    # Initialize evaluator
    evaluator = Evaluator(ground_truth)

    # Load and process data
    csv_path = "shl_products.csv"
    docs, metas = pre_processing_csv(csv_path)
    collection, model = build_chroma_store(docs, metas)

    # Evaluate with different K values
    k_values = [5, 10, 15]
    results = {}

    for k in k_values:
        print(f"\nEvaluating with K={k}")
        metrics = evaluator.evaluate(model, collection, k)
        results[k] = metrics
        print(f"Mean Recall@{k}: {metrics['Mean Recall@K']:.4f}")
        print(f"MAP@{k}: {metrics['MAP@K']:.4f}")


if __name__ == "__main__":
    main() 