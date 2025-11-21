"""
evaluation/evaluate.py

Run quick evaluation queries against the search engine.

Usage:
    python evaluation/evaluate.py

Requirements:
    - API must be running on http://127.0.0.1:8000
    - Documents must already be indexed (FastAPI startup does this)

This evaluation script sends a set of example queries to the API
and prints the top results with scores and keyword overlaps.
"""

import requests
import json

API_URL = "http://127.0.0.1:8000/search"

TEST_QUERIES = [
    "machine learning basics",
    "cooking recipes",
    "football strategies",
    "climate change global warming",
    "blockchain cryptography",
]

def run_query(query: str, top_k: int = 3):
    payload = {"query": query, "top_k": top_k}
    res = requests.post(API_URL, json=payload)
    if res.status_code != 200:
        print(f"Error {res.status_code}: {res.text}")
        return
    data = res.json()

    print(f"\n=== QUERY: {query} ===")
    for r in data.get("results", []):
        print(f"Doc: {r['doc_id']} | Score: {r['score']:.3f}")
        print(f"Preview: {r['preview'][:120]}...")
        print(f"Overlap: {r['explanation']['keyword_overlap']}")
        print("---")


def main():
    print("Running evaluation queries...\n")
    for q in TEST_QUERIES:
        run_query(q)
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()