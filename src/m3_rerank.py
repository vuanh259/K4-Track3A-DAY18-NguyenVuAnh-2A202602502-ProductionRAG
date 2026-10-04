"""Cross-encoder reranking and measured warm latency."""
from dataclasses import dataclass
from functools import lru_cache
import time
import numpy as np
from config import RERANK_TOP_K

@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int

@lru_cache(maxsize=2)
def _cached_model(name):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(name)

class CrossEncoderReranker:
    def __init__(self, model_name="BAAI/bge-reranker-v2-m3"):
        self.model_name, self._model = model_name, None

    def _load_model(self):
        if self._model is None:
            self._model = _cached_model(self.model_name)
        return self._model

    def rerank(self, query, documents, top_k=RERANK_TOP_K):
        if not documents or top_k <= 0:
            return []
        pairs = [(query, doc["text"]) for doc in documents]
        scores = np.asarray(self._load_model().predict(pairs)).reshape(-1)
        if len(scores) != len(documents) or not np.isfinite(scores).all():
            raise ValueError("Reranker must return one finite score per document")
        ranked = sorted(zip(scores, documents), key=lambda item: item[0], reverse=True)
        return [RerankResult(doc["text"], float(doc.get("score", 0)), float(score),
                dict(doc.get("metadata", {})), rank)
                for rank, (score, doc) in enumerate(ranked[:top_k])]

class FlashrankReranker:
    """Optional ONNX alternative, loaded only when selected."""
    def __init__(self):
        self._model = None

    def rerank(self, query, documents, top_k=RERANK_TOP_K):
        if not documents or top_k <= 0:
            return []
        from flashrank import Ranker, RerankRequest
        if self._model is None:
            self._model = Ranker()
        passages = [{"id": i, "text": doc["text"]} for i, doc in enumerate(documents)]
        results = self._model.rerank(RerankRequest(query=query, passages=passages))
        return [RerankResult(item["text"], float(documents[item["id"]].get("score", 0)),
                float(item["score"]), dict(documents[item["id"]].get("metadata", {})), rank)
                for rank, item in enumerate(results[:top_k])]

def benchmark_reranker(reranker, query, documents, n_runs=5):
    if n_runs <= 0:
        raise ValueError("n_runs must be positive")
    reranker.rerank(query, documents)
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        times.append((time.perf_counter() - start) * 1000)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}
