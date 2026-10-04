"""
Basic RAG Baseline — Chạy TRƯỚC để có scores so sánh.
=====================================================
Basic = paragraph chunking + dense-only search (không hybrid, không rerank, không enrichment).
Đây là RAG đã học ở buổi trước — hôm nay sẽ cải thiện từng bước.
"""

import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import NAIVE_COLLECTION
from src.m1_chunking import chunk_basic, load_documents
from src.m2_search import DenseSearch
from src.m4_eval import evaluate_ragas, load_test_set, save_report


def main():
    print("=" * 60)
    print("BASIC RAG BASELINE")
    print("(paragraph chunking + dense-only, no rerank, no enrichment)")
    print("=" * 60)

    docs = load_documents()
    chunks = []
    for doc in docs:
        for c in chunk_basic(doc["text"], metadata=doc["metadata"]):
            chunks.append({"text": c.text, "metadata": c.metadata})
    print(f"  {len(chunks)} basic paragraph chunks")

    search = DenseSearch()
    search.index(chunks, collection=NAIVE_COLLECTION)

    test_set = load_test_set()
    questions, answers, all_contexts, ground_truths = [], [], [], []

    llm_client = None
    from src.llm import (
        active_provider,
        chat_model_name,
        create_chat_client,
        normalize_abstention,
        wait_for_gemini_request,
    )
    if active_provider():
        llm_client = create_chat_client()

    for i, item in enumerate(test_set):
        results = search.search(item["question"], top_k=3, collection=NAIVE_COLLECTION)
        contexts = [r.text for r in results]

        if llm_client and contexts:
            from openai import APIError

            try:
                context_str = "\n\n".join(contexts)
                wait_for_gemini_request()
                request_options = (
                    {"reasoning_effort": "none"}
                    if active_provider() == "gemini"
                    else {}
                )
                resp = llm_client.chat.completions.create(model=chat_model_name(), temperature=0, messages=[
                    {"role": "system", "content": "Trả lời CHỈ dựa trên context. Nếu không có → nói 'Không tìm thấy.'"},
                    {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {item['question']}"},
                ], **request_options)
                answer = normalize_abstention(resp.choices[0].message.content)
            except APIError as error:
                if active_provider() == "ollama":
                    raise
                print(f"  Generation failed: {type(error).__name__}: {error}", flush=True)
                answer = contexts[0]
        else:
            answer = contexts[0] if contexts else "Không tìm thấy."

        answers.append(answer)
        questions.append(item["question"])
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)

    results = evaluate_ragas(questions, answers, all_contexts, ground_truths, cache_dir='.cache/ragas')
    save_report(results, [], path="reports/naive_baseline_report.json")
    if results.get("evaluation_status") != "completed":
        print(
            "\nBaseline evaluation did not complete; see "
            "reports/naive_baseline_report.json for the error.",
            flush=True,
        )
        return results

    print("\nBASIC BASELINE SCORES")
    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        print(f"  {metric}: {results[metric]:.4f}")
    print("\nBaseline evaluation completed.")
    return results


if __name__ == "__main__":
    start = time.time()
    result = main()
    print(f"Total: {time.time() - start:.1f}s")
    if result.get("evaluation_status") != "completed":
        raise SystemExit(1)
