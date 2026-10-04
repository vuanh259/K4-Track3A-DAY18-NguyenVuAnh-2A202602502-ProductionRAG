"""
Kiểm tra định dạng bài nộp trước khi submit.
Chạy: python check_lab.py

⚠️ Lỗi định dạng khiến script chấm tự động không chạy → trừ 5 điểm thủ tục.
"""

import json
import math
import os
import subprocess
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def check_file(path: str, required: bool = True) -> bool:
    if os.path.exists(path):
        print(f"  ✅ {path}")
        return True
    elif required:
        print(f"  ❌ THIẾU: {path}")
        return False
    else:
        print(f"  ⚠️  Optional: {path}")
        return True


def check_json(path: str, required_keys: list[str]) -> bool:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        missing = [k for k in required_keys if k not in data]
        if missing:
            print(f"  ❌ {path} thiếu keys: {missing}")
            return False
        if path.endswith("report.json") and data.get("evaluation_status") != "completed":
            print(f"  ❌ {path} — evaluation did not complete; scores are unmeasured")
            return False
        if path.endswith("report.json"):
            from src.m4_eval import METRICS, load_test_set
            rows = data.get('per_question', [])
            expected = len(load_test_set())
            if data.get('num_questions') != expected or len(rows) != expected:
                print(f"  ❌ {path} — expected {expected} evaluated questions")
                return False
            for metric in METRICS:
                values = [data.get('aggregate', {}).get(metric)] + [r.get(metric) for r in rows]
                if any(not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
                    print(f"  ❌ {path} — invalid {metric} scores")
                    return False
                if not math.isclose(values[0], sum(values[1:]) / expected, abs_tol=1e-6):
                    print(f"  ❌ {path} — aggregate differs from question scores")
                    return False
        print(f"  ✅ {path} — keys OK")
        return True
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"  ❌ {path} — {e}")
        return False


def check_todos() -> int:
    """Count remaining TODO markers in src/."""
    count = 0
    for root, _, files in os.walk("src"):
        for f in files:
            if f.endswith(".py"):
                with open(os.path.join(root, f), encoding="utf-8") as fh:
                    for line in fh:
                        if "# TODO:" in line:
                            count += 1
    return count


def run_tests() -> tuple[int, int]:
    """Run pytest and return (passed, total)."""
    try:
        import re
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=no", "-q"],
            capture_output=True, text=True, timeout=1800, encoding="utf-8",
            errors="replace", check=False,
        )
        lines = result.stdout.strip().split("\n")
        summary = lines[-1] if lines else ""
        m_pass = re.search(r"(\d+)\s+passed", summary)
        m_fail = re.search(r"(\d+)\s+failed", summary)
        m_error = re.search(r"(\d+)\s+errors?", summary)
        passed = int(m_pass.group(1)) if m_pass else 0
        failed = int(m_fail.group(1)) if m_fail else 0
        errors = int(m_error.group(1)) if m_error else 0
        total = passed + failed + errors
        if result.returncode and total == passed:
            total += 1
        return passed, total
    except (OSError, subprocess.TimeoutExpired) as e:
        print(f"  ⚠️  pytest error: {e}")
        return 0, 0


def validate():
    print("🔍 Kiểm tra bài nộp Lab 18: Production RAG\n")
    errors = 0

    # 1. Source files
    print("📁 Source code:")
    for f in ["src/m1_chunking.py", "src/m2_search.py", "src/m3_rerank.py",
              "src/m4_eval.py", "src/m5_enrichment.py", "src/pipeline.py"]:
        if not check_file(f):
            errors += 1

    # 2. Reports
    print("\n📊 Reports:")
    if check_file("reports/ragas_report.json"):
        if not check_json("reports/ragas_report.json", ["aggregate", "num_questions"]):
            errors += 1
    else:
        errors += 1
    baseline_path = "reports/naive_baseline_report.json"
    if os.path.exists(baseline_path):
        if not check_json(baseline_path, ["aggregate", "num_questions"]):
            errors += 1
    else:
        check_file(baseline_path, required=False)

    # 3. Analysis
    print("\n📝 Analysis:")
    if not check_file("analysis/failure_analysis.md"):
        errors += 1

    # 4. Individual reflections
    print("\n👤 Individual reflections:")
    reflections = []
    ref_dir = "analysis/reflections"
    if os.path.isdir(ref_dir):
        reflections.extend([f"{ref_dir}/{f}" for f in os.listdir(ref_dir)
                            if f.startswith("reflection_") and f.endswith(".md") and f != "reflection_TEMPLATE.md"])
    if os.path.isdir("analysis"):
        reflections.extend([f"analysis/{f}" for f in os.listdir("analysis")
                            if f.startswith("reflection_") and f.endswith(".md") and f != "reflection_TEMPLATE.md"])

    if reflections:
        for r in set(reflections):
            print(f"  ✅ {r}")
    else:
        print(f"  ⚠️  Chưa có file reflection cá nhân (đặt tại {ref_dir}/reflection_[HọTên].md hoặc analysis/reflection_[HọTên].md)")
        errors += 1

    # 5. TODO count
    print("\n🔧 TODO markers:")
    todo_count = check_todos()
    if todo_count == 0:
        print("  ✅ Không còn TODO nào")
    else:
        print(f"  ⚠️  Còn {todo_count} TODO chưa implement")
        errors += 1

    # 6. Tests
    print("\n🧪 Auto-tests:")
    passed, total = run_tests()
    if total > 0:
        pct = passed / total * 100
        print(f"  {'✅' if pct >= 80 else '⚠️'} {passed}/{total} tests passed ({pct:.0f}%)")
        if passed != total:
            errors += 1
    else:
        print("  ⚠️  Không chạy được tests")
        errors += 1

    # 7. Summary
    print("\n" + "=" * 50)
    if errors == 0:
        print("🚀 Bài lab sẵn sàng để nộp!")
    else:
        print(f"❌ Có {errors} lỗi. Sửa trước khi nộp.")
    print("=" * 50)
    return errors == 0


if __name__ == "__main__":
    raise SystemExit(0 if validate() else 1)
