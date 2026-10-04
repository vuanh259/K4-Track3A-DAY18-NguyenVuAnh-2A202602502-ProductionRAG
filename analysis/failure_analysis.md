# Failure Analysis — Lab 18: Production RAG

**Học viên:** Nguyen Vu Anh · **MSSV:** 2A202602502

## Execution status

The baseline pipeline completed on the local Ollama provider for all 20 test
questions, and its RAGAS report is saved to
`reports/naive_baseline_report.json`. The production pipeline was not run, so
there is no production score, comparison, or end-to-end latency report. The
submission checker correctly remains blocked because
`reports/ragas_report.json` has not been generated.

The baseline was evaluated with `qwen2.5:3b-instruct`,
`nomic-embed-text:latest`, and RAGAS 0.1.x. These local-model measurements are
not directly comparable with results from a different evaluator/provider.

## RAGAS Scores

| Metric | Naive Baseline (Ollama) | Production | Δ |
|--------|-------------------------:|-----------:|--:|
| Faithfulness | 0.4167 | N/A | N/A |
| Answer Relevancy | 0.3087 | N/A | N/A |
| Context Precision | 0.8083 | N/A | N/A |
| Context Recall | 0.6458 | N/A | N/A |

## Baseline observations

- Context precision is the strongest baseline metric (0.8083); the dense
  retriever usually returned focused evidence.
- Answer relevancy (0.3087) and faithfulness (0.4167) are comparatively low.
  Inspecting answers and retrieved contexts is necessary before attributing
  these scores to a specific prompt or retrieval defect.
- The saved baseline report contains 20 per-question results. It does not
  include a baseline Bottom-5 diagnosis.

## Validation status

- Focused tests excluding the memory-heavy reranker tests: 65 passed.
- M4 tests run separately: 7 passed.
- The full suite run in this Windows session: 61 passed, 9 failed. Five
  reranker tests failed with Windows paging-file allocation errors; three M2
  tests hit `MemoryError` while loading NLP resources under memory pressure.
  The remaining M4 test failure occurred in that full-suite run but passed when
  M4 was run on its own.
- Whole-repository Ruff reports 15 lint findings; targeted Ruff checks on the
  changed provider/evaluation paths passed.

## Remaining work

1. Run the production pipeline and generate `reports/ragas_report.json` and
   `reports/latency_report.json`.
2. Re-run the full test suite with sufficient Windows paging-file capacity,
   especially the reranker tests.
3. Review production per-question answers/contexts and update the comparison
   and failure analysis using measured results only.
