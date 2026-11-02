"""Dependency-light retrieval augmented generation primitives."""

from .chunk import Chunk, chunk_text
from .retriever import LexicalRetriever, SearchResult

__all__ = ["Chunk", "LexicalRetriever", "SearchResult", "chunk_text"]
