# Lab 18: Production RAG Pipeline — Bài tập cá nhân

**Thời gian:** 2 giờ implement + 30 phút reflection
**Điểm:** 100 + 10 bonus

---

## Tổng quan

Implement **toàn bộ 5 modules** của Production RAG Pipeline, chạy RAGAS evaluation, và phân tích kết quả.

```
naive_baseline.py (chạy trước)
        ↓ so sánh
M1 Chunking → M5 Enrichment → M2 Hybrid Search → M3 Reranking → LLM Answer → M4 RAGAS Eval
```

---

## Timeline (Thời lượng ước tính)

| Thời lượng | Hoạt động |
|------------|-----------|
| 10 phút | Setup môi trường: `docker compose up -d`, `pip install`, chạy `naive_baseline.py` |
| 20 phút | **M1 Chunking** — 3 strategies (semantic, hierarchical, structure-aware) |
| 20 phút | **M2 Search** — BM25 Vietnamese + Dense + RRF |
| 15 phút | **M3 Rerank** — CrossEncoder load + rerank |
| 15 phút | **M4 Eval** — RAGAS + failure analysis |
| 20 phút | **M5 Enrichment** — combined single-call hoặc 4 techniques riêng |
| 20 phút | Chạy `python src/pipeline.py` → RAGAS scores → failure analysis |
| 30 phút | **Reflection:** map lecture concepts → project plan (xem bên dưới) |

---

## Setup (10 phút)

**Linux / macOS / Git Bash:**
```bash
docker compose up -d                    # Khởi động Qdrant
pip install -r requirements.txt
cp .env.example .env                    # Tạo file .env và điền GEMINI_API_KEY
python naive_baseline.py                # Khởi tạo baseline (sẽ cập nhật điểm thật sau khi xong M2 & M4)
```

**Windows (PowerShell):**
```powershell
docker compose up -d                    # Khởi động Qdrant
pip install -r requirements.txt
Copy-Item .env.example .env             # Tạo file .env và điền GEMINI_API_KEY (CMD: copy .env.example .env)
python naive_baseline.py                # Khởi tạo baseline (sẽ cập nhật điểm thật sau khi xong M2 & M4)
```

---

## Module 1: Advanced Chunking (20 phút)

**File:** `src/m1_chunking.py` · **Test:** `pytest tests/test_m1.py`

Implement 3 strategies, so sánh với basic baseline:

| Strategy | Hàm | Mô tả |
|----------|-----|-------|
| Semantic | `chunk_semantic()` | Nhóm câu theo cosine similarity |
| Hierarchical | `chunk_hierarchical()` | Parent (2048) + Child (256), retrieve child → return parent |
| Structure-Aware | `chunk_structure_aware()` | Parse markdown headers → chunk theo section |

**Pass criteria:**
- [ ] Semantic: `list[Chunk]` không rỗng
- [ ] Hierarchical: children có `parent_id` hợp lệ, nhỏ hơn parents
- [ ] Structure-Aware: giữ headers, có `section` trong metadata

---

## Module 2: Hybrid Search (20 phút)

**File:** `src/m2_search.py` · **Test:** `pytest tests/test_m2.py`

| Component | Hàm | Mô tả |
|-----------|-----|-------|
| Vietnamese segmentation | `segment_vietnamese()` | underthesea + replace `_` |
| BM25 | `BM25Search.index()` + `.search()` | BM25Okapi trên text đã segment |
| Dense | `DenseSearch.index()` + `.search()` | bge-m3 + Qdrant `query_points()` |
| RRF | `reciprocal_rank_fusion()` | score(d) = Σ 1/(k + rank + 1) |

**Pass criteria:**
- [ ] BM25 search trả về results với `method="bm25"`
- [ ] RRF merge → results với `method="hybrid"`
- [ ] Query "nghỉ phép" → kết quả liên quan

---

## Module 3: Reranking (15 phút)

**File:** `src/m3_rerank.py` · **Test:** `pytest tests/test_m3.py`

| Component | Hàm | Mô tả |
|-----------|-----|-------|
| Cross-encoder | `CrossEncoderReranker._load_model()` + `.rerank()` | bge-reranker-v2-m3 via `sentence_transformers.CrossEncoder` |

**Pass criteria:**
- [ ] Rerank 5 docs → trả về ≤ 3 `RerankResult`
- [ ] Sorted by `rerank_score` descending
- [ ] Doc "nghỉ phép" ranked cao hơn "VPN"

---

## Module 4: RAGAS Evaluation (15 phút)

**File:** `src/m4_eval.py` · **Test:** `pytest tests/test_m4.py`

| Component | Hàm | Mô tả |
|-----------|-----|-------|
| Evaluate | `evaluate_ragas()` | 4 metrics, wrap trong try/except |
| Failure analysis | `failure_analysis()` | Bottom-N, Diagnostic Tree mapping |

**Pass criteria:**
- [ ] `evaluate_ragas()` trả về dict với 4 metric keys
- [ ] `failure_analysis()` trả về list với `diagnosis` + `suggested_fix`

---

## Module 5: Enrichment (20 phút)

**File:** `src/m5_enrichment.py` · **Test:** `pytest tests/test_m5.py`

**Chọn 1 trong 2 mode:**

| Mode | Hàm | API calls | Mô tả |
|------|-----|-----------|-------|
| Combined (khuyến khích) | `_enrich_single_call()` | 1 call/chunk | 1 prompt → summary + questions + context + metadata |
| Riêng lẻ (để học) | 4 hàm riêng | 4 calls/chunk | `summarize_chunk()`, `generate_hypothesis_questions()`, `contextual_prepend()`, `extract_metadata()` |

**Pass criteria:**
- [ ] `enrich_chunks()` trả về `list[EnrichedChunk]`
- [ ] `enriched_text` khác `original_text` (nếu có API key)
- [ ] Fallback hoạt động khi không có API key

---

## Chạy Pipeline (20 phút)

```bash
python src/pipeline.py
```

Điền bảng so sánh:

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|-----------|---|
| Faithfulness | ? | ? | ? |
| Answer Relevancy | ? | ? | ? |
| Context Precision | ? | ? | ? |
| Context Recall | ? | ? | ? |

Mở `reports/ragas_report.json` → tìm bottom-5 worst questions → điền `analysis/failure_analysis.md`.

---

## Reflection: Lecture → Project (30 phút)

Viết file `analysis/reflections/reflection_[HọTên].md` (hoặc copy từ `analysis/reflections/reflection_TEMPLATE.md`) gồm **3 phần**:

### Phần 1: Mapping bài giảng (10 phút)
Map từng concept trong lecture vào code bạn vừa viết:

| Lecture Concept | Module | Hàm cụ thể | Observation |
|----------------|--------|-------------|-------------|
| Semantic chunking | M1 | `chunk_semantic()` | "Threshold 0.85 tạo X chunks vs basic Y chunks" |
| BM25 + Dense fusion | M2 | `reciprocal_rank_fusion()` | "RRF giải quyết..." |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | "Latency Xms, precision..." |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | "Metric X thấp nhất vì..." |
| Contextual embeddings | M5 | `contextual_prepend()` | "Giảm retrieval failure bằng..." |

### Phần 2: Khó khăn & giải quyết (10 phút)
- Lỗi gặp phải (exact error message)
- Cách debug
- Kiến thức thiếu → cách bổ sung

### Phần 3: Action Plan cho project (10 phút)
Dựa trên những gì học được hôm nay, viết plan cụ thể cho project của bạn:

```markdown
## Project: [Tên project]

### Hiện tại
- RAG pipeline hiện tại: [mô tả ngắn]
- Known issues: [vấn đề đang gặp]

### Plan áp dụng
1. [ ] Chunking strategy: [chọn gì, tại sao]
2. [ ] Search: [BM25/Dense/Hybrid, tại sao]
3. [ ] Reranking: [có/không, model nào]
4. [ ] Evaluation: [RAGAS hay custom metrics]
5. [ ] Enrichment: [technique nào phù hợp nhất]

### Timeline
- Tuần X: ...
- Tuần Y: ...
```

---

## Quy chuẩn đặt tên Repository bài nộp

Theo quy ước chung Khóa 4 (Track 3A) cho bài tập cá nhân:
- **Cấu trúc đặt tên repo:**  
  `K4-Track3A-DAY18-<HoVaTen>-<MSSV>-ProductionRAG`
- **Quy tắc:**
  - Viết tiếng Việt **không dấu**, **không khoảng trắng**.
  - Ngăn cách giữa các thành phần bằng dấu gạch nối `-`.
  - `<HoVaTen>` viết dạng PascalCase (ví dụ: `NguyenVanAn`).
  - `<MSSV>` là mã số học viên (ví dụ: `AI20K001`).
- **Ví dụ chuẩn:**  
  `K4-Track3A-DAY18-NguyenVanAn-AI20K001-ProductionRAG`

## Hạn nộp bài (Deadline)

- **Hạn chót:** **23h59 ngày diễn ra bài lab (GMT+7)**.
- **Nơi nộp:** Nộp link GitHub repository cá nhân (để chế độ Public) lên cổng VLearn LMS / Codelab.

---

## Deliverable

Cấu trúc repository cá nhân khi push lên GitHub:

```
<Tên-Repo-Cá-Nhân>/
├── src/                        # ★ 5 modules đã implement
│   ├── m1_chunking.py
│   ├── m2_search.py
│   ├── m3_rerank.py
│   ├── m4_eval.py
│   ├── m5_enrichment.py
│   └── pipeline.py
├── analysis/
│   ├── failure_analysis.md     # ★ Bottom-5 analysis
│   └── reflections/
│       └── reflection_[HọTên].md  # ★ Mapping + Plan
└── reports/                    # ★ Bắt buộc: kết quả đánh giá pipeline
    └── ragas_report.json
```

### Trước khi nộp

**1. Kiểm tra tổng thể bằng script (chạy được trên mọi hệ điều hành Windows / Linux / macOS):**
```bash
python check_lab.py                     # Kiểm tra đầy đủ: files, reports, reflections, TODOs, tests
```

**2. Các câu lệnh kiểm tra chi tiết:**
```bash
pytest tests/ -v                        # Tất cả tests pass?
python main.py                          # Pipeline chạy end-to-end, sinh reports/ragas_report.json?

# Kiểm tra số lượng TODOs còn lại (mục tiêu: 0):
# Linux / macOS:
grep -r "# TODO" src/m*.py | wc -l

# Windows (PowerShell):
(Select-String -Path src/*.py -Pattern "# TODO").Count
```
