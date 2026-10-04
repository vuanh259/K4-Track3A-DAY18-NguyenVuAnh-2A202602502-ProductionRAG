# Reflection — Lab 18: Production RAG

**Họ và tên:** Nguyen Vu Anh
**MSSV:** 2A202602502
**Khóa:** K4 - Track 3A
**Ngày hoàn thành:** 04/10/2026

---

## Phần 1: Mapping bài giảng

| Lecture Concept | Module | Hàm cụ thể | Observation |
|---|---|---|---|
| Semantic chunking | M1 | `chunk_semantic()` | Đo trên từng tài liệu: threshold 0.85 tạo 208 chunks, trung bình 99 ký tự; basic tạo 57 chunks, trung bình 366 ký tự. Threshold cao làm semantic chia nhỏ hơn, không mặc nhiên tốt hơn. Số liệu: `reports/chunking_comparison.json`. |
| Hierarchical chunking | M1 | `chunk_hierarchical()` | Đo được 26 parents và 110 children; child trung bình 191 ký tự, tối đa 256. Child giữ `parent_id`; pipeline tra cứu bằng child rồi trả lại parent làm context. |
| Structure-aware chunking | M1 | `chunk_structure_aware()` | Đo được 107 chunks, trung bình 194 ký tự; giữ `section`, bảng và fenced code. So sánh chạy riêng từng tài liệu để không trộn nguồn. |
| BM25 + Dense fusion | M2 | `BM25Search`, `DenseSearch`, `reciprocal_rank_fusion()` | BM25 dùng underthesea và chuẩn hóa `_` thành khoảng trắng; Dense dùng bge-m3/Qdrant; RRF gộp theo thứ hạng thay vì so sánh hai thang điểm khác nhau. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Chấm lại các cặp query-document và lấy top 3; model đã tải, xác minh SHA-256 và test rerank pass. Latency toàn pipeline còn chờ chạy end-to-end. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()`, `failure_analysis()` | Baseline Ollama trên 20 câu đạt Faithfulness 0.4167, Answer Relevancy 0.3087, Context Precision 0.8083, Context Recall 0.6458. Chưa có production score; không coi số 0 mặc định là điểm đo được. |
| Contextual prepend + HyQA | M5 | `_enrich_single_call()`, `enrich_chunks()` | Một lần gọi tạo summary, câu hỏi giả thuyết, context và metadata; fallback trích xuất giúp chạy offline, còn provenance và `parent_id` gốc được giữ lại. |

## Phần 2: Khó khăn và cách debug

- **Docker/Qdrant:** ban đầu Docker Desktop/Linux engine chưa chạy (`failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine`). Sau khi khởi động Docker Desktop và chạy `docker compose up -d`, Qdrant trả readiness thành công.
- **Cài dependencies:** `ReadTimeoutError: HTTPSConnectionPool(host='files.pythonhosted.org', port=443): Read timed out.` — tải gói bị timeout. Cài lại với timeout và số lần retry cao hơn đã cài thành công `requirements.txt`.
- **Gemini quota/model:** model chat ban đầu trả `503 UNAVAILABLE`; reasoning chiếm output budget làm JSON M5 bị cắt dù có JSON mode. Key thay thế chạy prompt M5 thật với `gemini-3.6-flash`; một số model 3.5/3.7/3.8 có 503 high demand. Project hiện tại báo free-tier 20 request/ngày/model; API yêu cầu chờ khoảng 14 giờ 57 phút sau khi 23/110 chunk được checkpoint. Enrichment dừng rõ khi gặp 429/5xx, cache có thể resume; limiter 13 giây chỉ xử lý RPM, không vượt quota ngày.
- **Model embedding/reranker:** trọng số `BAAI/bge-m3` đã tải, SHA-256 khớp metadata chính thức và khởi tạo trả dimension 1024. `BAAI/bge-reranker-v2-m3` đã tải đủ, SHA-256 xác minh thành công và test M3 pass.
- **Lỗi RAGAS thực tế:** `ValueError: RAGAS returned non-finite scores for question 4: faithfulness`. Kiểm tra mã RAGAS 0.1 cho thấy bộ tách câu chỉ giữ câu kết thúc bằng dấu chấm; model nhỏ cũng có thể trả danh sách claim rỗng. Local evaluator dùng prompt trích claim rõ ràng và giữ tất cả câu không rỗng, kể cả bullet/câu thiếu dấu chấm. Câu từng lỗi đã được đánh giá lại thành công; không thay NaN bằng điểm giả.
- **Latency:** trộn `time.time()` với `time.perf_counter()` tạo thời gian âm khoảng -1,79 tỷ giây. Sửa sang cùng đồng hồ monotonic; thực sự tải reranker trước khi kết thúc phép đo load. Regression test dùng hai đồng hồ lệch nhau để bắt lỗi tái diễn.
- **Timeout khi đánh giá:** lần chạy local tiếp theo gặp `TimeoutError` sau 51/80 phép đo. Giới hạn output evaluator ở 2048 tokens và lưu checkpoint sau mỗi câu, theo hash của input và cấu hình evaluator. Chỉ retry khi lỗi, không chọn điểm cao nhất trong nhiều lần chạy. Điểm hữu hạn được giữ nguyên; lỗi tiếp diễn vẫn làm pipeline thất bại.
- **Cách xác minh:** 65 test tập trung (không gồm test M3 nặng bộ nhớ) và 7 test M4 riêng lẻ pass. Full suite chạy dưới áp lực bộ nhớ có 61 passed/9 failed: năm test reranker gặp lỗi Windows paging file, ba test M2 gặp `MemoryError`, một test M4 chỉ lỗi trong lượt chạy toàn suite nhưng pass riêng. Ruff toàn repo có 15 findings; targeted Ruff cho các đường provider/evaluation đã kiểm tra pass.
- **Kiến thức cần bổ sung:** quản lý quota/rate limit cho LLM, chi phí khi đánh giá nhiều câu, và cách ước lượng kích thước/thời gian tải model trước khi chạy pipeline.

## Phần 3: Action Plan cho project

### Project: Vietnamese Internal Policy RAG (corpus Lab 18)

#### Hiện trạng

- **Pipeline:** dữ liệu quy chế tiếng Việt → hierarchical chunking → enrichment → BM25 + Dense/Qdrant → cross-encoder → tổng hợp câu trả lời có trích context.
- **Điểm cần đo:** baseline local đã có điểm RAGAS trên 20 câu; production RAGAS, so sánh hai hệ thống và latency end-to-end chưa được sinh.

#### Kế hoạch cải tiến

1. **Chunking:** dùng structure-aware cho Markdown có heading/bảng; hierarchical parent-child làm mặc định cho retrieval để giữ đầy đủ ngữ cảnh khi sinh câu trả lời.
2. **Search:** kết hợp BM25 tiếng Việt với Dense search và RRF để vừa bắt đúng từ khóa/chỉ số, vừa tìm nội dung diễn đạt tương đương.
3. **Reranking:** dùng `BAAI/bge-reranker-v2-m3` cho top 20 candidate, lấy top 3; xác minh test khi Windows paging file đủ dung lượng và đo latency bằng đồng hồ monotonic.
4. **Evaluation:** chạy production trên cùng 20 câu hỏi để so với baseline Ollama; lưu kết quả theo từng câu, rồi kiểm tra thủ công trước khi kết luận nguyên nhân. Điểm model local không so sánh trực tiếp với provider khác.
5. **Enrichment:** giữ combined one-call mode để bổ sung context và HyQA; dùng Ollama local để tránh quota Gemini, cache phân biệt provider/model, giữ nguyên văn bản gốc khi tạo context trả lời.

#### Timeline

- **Tuần 1:** khởi động Qdrant, chạy baseline, xác nhận test M1–M3 và lưu latency.
- **Tuần 2:** chạy pipeline production bằng Ollama hoặc provider có quota đủ; phân tích kết quả đo được và so với baseline.
- **Tuần 3:** sửa retrieval/prompt theo lỗi đo được, chạy lại cùng test set và ghi nhận delta.
