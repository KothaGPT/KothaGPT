"""Embeddings and reranking: a minimal RAG retrieval prototype.

Usage:
    python examples/python/rag.py
"""

from kothagpt import KothaGPT

DOCUMENTS = [
    "বাংলা ভাষা দক্ষিণ এশিয়ার অন্যতম প্রধান ভাষা।",
    "ঢাকা বাংলাদেশের রাজধানী শহর।",
    "রান্নায় হলুদ ও মরিচের ব্যবহার বেশি হয়।",
    "বাংলা ভাষা শেখার জন্য নিয়মিত অনুশীলন প্রয়োজন।",
]

QUERY = "বাংলা ভাষা কীভাবে শিখব?"


def main() -> None:
    with KothaGPT() as client:
        # 1. Embed the documents.
        embedded = client.embeddings.create(DOCUMENTS)
        print(f"Embedded {len(embedded.data)} documents (dim={len(embedded.data[0].embedding)}).")

        # 2. Rerank them against the query.
        reranked = client.rerank.create(QUERY, DOCUMENTS, top_n=2)
        print(f"\nTop results for: {QUERY}\n")
        for result in reranked.results:
            print(f"  [{result.relevance_score:.3f}] {result.document}")


if __name__ == "__main__":
    main()
