from services.rag.chunk import chunk_text
from services.rag.context import build_context
from services.rag.retriever import LexicalRetriever


def test_bangla_chunking_preserves_provenance() -> None:
    chunks = chunk_text(
        "বাংলা ভাষা সুন্দর। এটি একটি পরীক্ষা।",
        document_id="doc-1",
        source="fixture.txt",
        max_chars=30,
        overlap=5,
    )
    assert chunks
    assert all(chunk.document_id == "doc-1" and chunk.source == "fixture.txt" for chunk in chunks)


def test_retrieval_is_ranked_and_context_is_cited() -> None:
    chunks = chunk_text(
        "Python functions return values.", document_id="doc-1", source="python.md"
    ) + chunk_text("Tea is grown in Sylhet.", document_id="doc-2", source="tea.md")
    context = build_context("Python functions", LexicalRetriever(chunks), top_k=2)
    assert context.results[0].chunk.document_id == "doc-1"
    assert context.citations[0].citation_id == "[1]"
    assert "[1]" in context.text
