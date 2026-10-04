from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


from functools import lru_cache

@lru_cache(maxsize=1)
def _semantic_encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def _bounded_split(text: str, size: int) -> list[str]:
    pieces = []
    while text:
        if len(text) <= size:
            if text.strip():
                pieces.append(text)
            break
        window = text[:size]
        cut = 0
        for separator in ("\n\n", "\n", ". ", " "):
            index = window.rfind(separator)
            if index >= size // 2:
                cut = index + len(separator)
                break
        cut = cut or size
        piece, text = text[:cut], text[cut:]
        if piece.strip():
            pieces.append(piece)
    return pieces


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """Group adjacent complete sentences by cosine similarity."""
    import numpy as np
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n\s*\n', text) if s.strip()]
    if not sentences:
        return []
    if not -1 <= threshold <= 1:
        raise ValueError('threshold must be between -1 and 1')
    vectors = _semantic_encoder().encode(sentences)
    groups, current = [], [sentences[0]]
    for i in range(1, len(sentences)):
        a, b = vectors[i - 1], vectors[i]
        similarity = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        if similarity < threshold:
            groups.append(current)
            current = []
        current.append(sentences[i])
    groups.append(current)
    return [Chunk(' '.join(group), {**(metadata or {}), 'strategy': 'semantic',
                  'chunk_index': i}) for i, group in enumerate(groups)]


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """Bounded, lossless parent/child splits with document-specific identifiers."""
    from hashlib import sha256
    if parent_size <= 0 or child_size <= 0:
        raise ValueError('Chunk sizes must be positive')
    metadata = dict(metadata or {})
    namespace = sha256((str(metadata.get('source', '')) + '\0' + text).encode()).hexdigest()[:16]
    parents, children = [], []
    for i, parent_text in enumerate(_bounded_split(text, parent_size)):
        pid = f'{namespace}_parent_{i}'
        parents.append(Chunk(parent_text, {**metadata, 'chunk_type': 'parent',
                            'parent_id': pid}, parent_id=pid))
        for j, child_text in enumerate(_bounded_split(parent_text, min(child_size, parent_size))):
            children.append(Chunk(child_text, {**metadata, 'chunk_type': 'child',
                'chunk_id': f'{pid}_child_{j}', 'parent_id': pid}, parent_id=pid))
    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """Split at Markdown headings outside fenced code; preserve tables and lists."""
    chunks, lines, section, fence = [], [], '', None
    def flush():
        body = ''.join(lines).strip()
        if body:
            chunks.append(Chunk(body, {**(metadata or {}), 'section': section,
                                'strategy': 'structure', 'chunk_index': len(chunks)}))
    for line in text.splitlines(keepends=True):
        marker = re.match(r'^\s{0,3}(`{3,}|~{3,})', line)
        if marker:
            token = marker.group(1)
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
        header = re.match(r'^#{1,3}\s+(.+)', line) if fence is None and not marker else None
        if header:
            flush()
            lines = []
            section = header.group(1).strip()
        lines.append(line)
    flush()
    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    # Match the real pipeline: never merge evidence from different documents.
    basic, semantic, parents, children, structure = [], [], [], [], []
    for doc in documents:
        text, meta = doc['text'], doc.get('metadata', {})
        basic.extend(chunk_basic(text, metadata=meta))
        semantic.extend(chunk_semantic(text, metadata=meta))
        doc_parents, doc_children = chunk_hierarchical(text, metadata=meta)
        parents.extend(doc_parents)
        children.extend(doc_children)
        structure.extend(chunk_structure_aware(text, metadata=meta))

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
