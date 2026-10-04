"""Vietnamese BM25, dense Qdrant retrieval and reciprocal rank fusion."""
from dataclasses import dataclass
import warnings
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)

@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str

def segment_vietnamese(text: str) -> str:
    from underthesea import word_tokenize
    return word_tokenize(text, format="text").replace("_", " ")

class BM25Search:
    def __init__(self):
        self.corpus_tokens, self.documents, self.bm25 = [], [], None

    def index(self, chunks):
        from rank_bm25 import BM25Okapi
        self.documents = list(chunks)
        self.corpus_tokens = [segment_vietnamese(c["text"].lower()).split() for c in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens) if any(self.corpus_tokens) else None

    def search(self, query, top_k=BM25_TOP_K):
        if self.bm25 is None or top_k <= 0 or not query.strip():
            return []
        scores = self.bm25.get_scores(segment_vietnamese(query.lower()).split())
        indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [SearchResult(self.documents[i]["text"], float(scores[i]),
            dict(self.documents[i].get("metadata", {})), "bm25")
            for i in indices[:top_k] if scores[i] > 0]

class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        try:
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=2)
            self.client.get_collections()
            self.backend = "server"
        except Exception:
            warnings.warn("Qdrant server unavailable; using the scaffold's in-memory fallback.")
            self.client = QdrantClient(":memory:")
            self.backend = "memory"
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks, collection=COLLECTION_NAME):
        from qdrant_client.models import Distance, VectorParams, PointStruct
        if self.client.collection_exists(collection):
            self.client.delete_collection(collection)
        self.client.create_collection(collection, vectors_config=VectorParams(
            size=EMBEDDING_DIM, distance=Distance.COSINE))
        for start in range(0, len(chunks), 32):
            batch = chunks[start:start + 32]
            vectors = self._get_encoder().encode([c["text"] for c in batch],
                batch_size=8, show_progress_bar=False, normalize_embeddings=True)
            points = [PointStruct(id=start + i, vector=v.tolist(),
                payload={**c.get("metadata", {}), "text": c["text"]})
                for i, (c, v) in enumerate(zip(batch, vectors))]
            self.client.upsert(collection, points=points, wait=True)

    def search(self, query, top_k=DENSE_TOP_K, collection=COLLECTION_NAME):
        if top_k <= 0 or not query.strip() or not self.client.collection_exists(collection):
            return []
        vector = self._get_encoder().encode(query, normalize_embeddings=True).tolist()
        response = self.client.query_points(collection, query=vector, limit=top_k, with_payload=True)
        results = []
        for point in response.points:
            payload = dict(point.payload or {})
            text = payload.pop("text", "")
            results.append(SearchResult(text, float(point.score), payload, "dense"))
        return results

def reciprocal_rank_fusion(results_list, k=60, top_k=HYBRID_TOP_K):
    if k < 0:
        raise ValueError("k must be non-negative")
    if top_k <= 0:
        return []
    scores, documents = {}, {}
    for ranked in results_list:
        seen = set()
        for rank, result in enumerate(ranked):
            key = (result.metadata.get("chunk_id"), result.metadata.get("source"), result.text)
            if key in seen:
                continue
            seen.add(key)
            documents.setdefault(key, result)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
    keys = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [SearchResult(documents[key].text, scores[key], dict(documents[key].metadata), "hybrid")
            for key in keys]

class HybridSearch:
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks):
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query, top_k=HYBRID_TOP_K):
        return reciprocal_rank_fusion([self.bm25.search(query, BM25_TOP_K),
            self.dense.search(query, DENSE_TOP_K)], top_k=top_k)
