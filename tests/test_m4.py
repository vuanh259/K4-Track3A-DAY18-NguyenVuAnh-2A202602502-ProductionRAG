"""Tests for Module 4: Evaluation."""
import os
import sys
from types import ModuleType, SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.m4_eval import (
    EvalResult,
    _evaluation_metrics,
    evaluate_ragas,
    failure_analysis,
    load_test_set,
)


@pytest.fixture(autouse=True)
def disable_external_evaluation(monkeypatch):
    import config
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    monkeypatch.setattr(config, "OPENAI_API_KEY", "")

def test_load_test_set():
    ts = load_test_set()
    assert len(ts) > 0 and "question" in ts[0] and "ground_truth" in ts[0]

def test_evaluate_returns_metrics():
    r = evaluate_ragas(["q"], ["a"], [["c"]], ["gt"])
    for k in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        assert k in r and isinstance(r[k], (int, float))


def test_evaluate_reports_non_finite_metric_and_question(monkeypatch):
    from src import llm

    class FakeDataset:
        @staticmethod
        def from_dict(_data):
            return _data

    class FakeFrame:
        def __len__(self):
            return 1

        def iterrows(self):
            yield 0, {
                "faithfulness": float("nan"),
                "answer_relevancy": 0.5,
                "context_precision": 0.5,
                "context_recall": 0.5,
            }

    class FakeEvaluation:
        @staticmethod
        def to_pandas():
            return FakeFrame()

    datasets = ModuleType("datasets")
    datasets.Dataset = FakeDataset
    ragas = ModuleType("ragas")
    ragas.evaluate = lambda *args, **kwargs: FakeEvaluation()
    metrics = ModuleType("ragas.metrics")
    for name in (
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "faithfulness",
    ):
        setattr(metrics, name, name)
    run_config = ModuleType("ragas.run_config")
    run_config.RunConfig = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, "datasets", datasets)
    monkeypatch.setitem(sys.modules, "ragas", ragas)
    monkeypatch.setitem(sys.modules, "ragas.metrics", metrics)
    monkeypatch.setitem(sys.modules, "ragas.run_config", run_config)
    monkeypatch.setattr(
        llm,
        "create_ragas_models",
        lambda: (object(), object(), "test"),
    )

    result = evaluate_ragas(["q"], ["a"], [["c"]], ["gt"])

    assert result["evaluation_status"] == "failed"
    assert "question 0: faithfulness" in result["error"]


def test_local_evaluator_uses_simple_faithfulness_extraction():
    from ragas.metrics import faithfulness

    metrics = _evaluation_metrics("ollama")
    local_faithfulness = metrics[0]

    assert local_faithfulness is not faithfulness
    assert "do not omit a sentence" in local_faithfulness.statement_prompt.instruction
    assert len(local_faithfulness.statement_prompt.examples) == 1
    assert metrics[1:] == _evaluation_metrics("gemini")[1:]
    assert faithfulness.statement_prompt.examples[0]["question"].startswith("Who was")


def test_failure_analysis_returns():
    results = [EvalResult("Q1", "A1", ["C1"], "GT1", 0.5, 0.6, 0.4, 0.3)]
    f = failure_analysis(results, bottom_n=1)
    assert len(f) == 1

def test_failure_has_diagnosis():
    results = [EvalResult("Q1", "A1", ["C1"], "GT1", 0.5, 0.6, 0.4, 0.3)]
    f = failure_analysis(results, bottom_n=1)
    if f:
        assert "diagnosis" in f[0] and "suggested_fix" in f[0]


def test_evaluation_checkpoint_reuses_only_identical_inputs(monkeypatch, tmp_path):
    import pandas as pd
    import ragas
    from src import llm, m4_eval

    calls = []
    scores = dict.fromkeys(m4_eval.METRICS, 0.5)

    def fake_evaluate(dataset, **kwargs):
        calls.append(dataset)
        return SimpleNamespace(to_pandas=lambda: pd.DataFrame([scores]))

    monkeypatch.setattr(ragas, 'evaluate', fake_evaluate)
    monkeypatch.setattr(llm, 'create_ragas_models', lambda: (
        SimpleNamespace(model_name='test-model'), SimpleNamespace(model='test-embed'), 'test'))
    first = evaluate_ragas(['q'], ['a'], [['c']], ['gt'], cache_dir=tmp_path)
    second = evaluate_ragas(['q'], ['a'], [['c']], ['gt'], cache_dir=tmp_path)
    assert first['evaluation_status'] == second['evaluation_status'] == 'completed'
    assert len(calls) == 1
    evaluate_ragas(['q'], ['changed answer'], [['c']], ['gt'], cache_dir=tmp_path)
    assert len(calls) == 2
