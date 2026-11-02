"""Tests for the RAG API endpoints wired into ``services/api``."""

from fastapi.testclient import TestClient

from services.api.app import app

client = TestClient(app)


def test_rag_kb_empty_then_ready():
    body = client.get("/v1/rag/kb").json()
    assert body["id"] == "default"
    assert body["status"] == "empty"
    assert body["documents"] == 0
    assert body["chunks"] == 0


def test_rag_ingest_is_idempotent():
    text = "বাংলা ভাষা সুন্দর। এটি একটি নথি থেকে নেওয়া নমুনা।"
    r1 = client.post(
        "/v1/rag/ingest",
        json={"text": text, "source": "smoke", "document_id": "doc-smoke-1"},
    )
    assert r1.status_code == 200
    rec1 = r1.json()
    assert rec1["status"] == "ready"
    assert rec1["chunks"] >= 1
    assert rec1["document_id"] == "doc-smoke-1"

    # Re-ingesting the same content digest returns the existing record.
    r2 = client.post(
        "/v1/rag/ingest",
        json={"text": text, "source": "smoke", "document_id": "doc-smoke-1"},
    )
    assert r2.status_code == 200
    assert r2.json()["document_id"] == rec1["document_id"]
    assert r2.json()["digest"] == rec1["digest"]

    kb = client.get("/v1/rag/kb").json()
    assert kb["documents"] == 1
    assert kb["chunks"] == rec1["chunks"]


def test_rag_search_returns_ranked_chunks():
    client.post(
        "/v1/rag/ingest",
        json={
            "text": "সিলেটে চা উৎপাদিত হয় এবং এটি বাংলাদেশের উদ্যানে বিখ্যাত।",
            "source": "tea.md",
            "document_id": "doc-tea",
        },
    )
    client.post(
        "/v1/rag/ingest",
        json={
            "text": "পাইথন একটি প্রোগ্রামিং ভাষা যা ফাংশন রিটার্ন করে।",
            "source": "python.md",
            "document_id": "doc-python",
        },
    )

    r = client.post("/v1/rag/search", json={"query": "চা", "top_k": 5})
    assert r.status_code == 200
    results = r.json()["results"]
    assert results
    assert results[0]["document_id"] == "doc-tea"
    assert all("score" in item and "chunk_id" in item for item in results)


def test_rag_context_builds_citations():
    client.post(
        "/v1/rag/ingest",
        json={
            "text": "সিলেটে চা উৎপাদিত হয়।",
            "source": "tea.md",
            "document_id": "doc-tea",
        },
    )
    r = client.post("/v1/rag/context", json={"query": "চা", "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["query"] == "চা"
    assert body["text"]
    assert len(body["citations"]) >= 1
    assert body["citations"][0]["citation_id"] == "[1]"


def test_rag_grounding_headers_on_chat():
    client.post(
        "/v1/rag/ingest",
        json={
            "text": "সিলেটে চা উৎপাদিত হয়।",
            "source": "tea.md",
            "document_id": "doc-tea",
        },
    )
    r = client.post(
        "/v1/chat/completions",
        json={"model": "kothagpt", "messages": [{"role": "user", "content": "চা কোথায় উৎপাদিত হয়?"}]},
        headers={"X-Kotha-RAG": "true"},
    )
    assert r.status_code == 200
    assert r.headers.get("X-Kotha-RAG-Citations")
    body = r.json()
    assert body["choices"][0]["message"]["content"]
