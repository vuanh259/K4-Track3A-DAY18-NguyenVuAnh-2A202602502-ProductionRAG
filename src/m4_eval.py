"""RAGAS evaluation with explicit failure status and question-level evidence."""
import hashlib
import inspect
import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

from config import TEST_SET_PATH

METRICS = ('faithfulness', 'answer_relevancy', 'context_precision', 'context_recall')

@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float

def load_test_set(path=TEST_SET_PATH):
    with open(path, encoding='utf-8') as stream:
        return json.load(stream)


def _evaluation_metrics(provider):
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    faithfulness_metric = faithfulness
    if provider == 'ollama':
        class CompleteAnswerFaithfulness(type(faithfulness)):
            def _create_statements_prompt(self, row):
                # RAGAS 0.1 drops sentences not ending in a literal period.
                # Keep bullet lists, Vietnamese text and unpunctuated answers.
                sentences = self.sentence_segmenter.segment(row['answer'])
                sentences = [s for s in sentences if s.strip()]
                if not sentences and row['answer'].strip():
                    sentences = [row['answer']]
                return self.statement_prompt.format(
                    question=row['question'], answer=row['answer'],
                    sentences='\n'.join(f'{i}:{s}' for i, s in enumerate(sentences)),
                )

        faithfulness_metric = CompleteAnswerFaithfulness()
        statement_prompt = deepcopy(faithfulness.statement_prompt)
        statement_prompt.instruction = (
            'For each numbered answer sentence, list every independently '
            'verifiable claim as a simpler statement. Preserve all facts, '
            'numbers, and conditions; do not omit a sentence or invent claims. '
            'Use the original language and sentence index. Return only the '
            'required JSON.'
        )
        statement_prompt.examples = [{
            'question': 'Hạn mức bảo hiểm PVI cho nhân viên là bao nhiêu?',
            'answer': (
                'PVI có hạn mức 200.000.000 VNĐ/năm, bao gồm nội trú, '
                'ngoại trú và nha khoa.'
            ),
            'sentences': (
                '0:PVI có hạn mức 200.000.000 VNĐ/năm, bao gồm nội trú, '
                'ngoại trú và nha khoa.'
            ),
            'analysis': [{
                'sentence_index': 0,
                'simpler_statements': [
                    'PVI có hạn mức 200.000.000 VNĐ/năm.',
                    'PVI bao gồm nội trú, ngoại trú và nha khoa.',
                ],
            }],
        }]
        faithfulness_metric.statement_prompt = statement_prompt
    return [
        faithfulness_metric,
        answer_relevancy,
        context_precision,
        context_recall,
    ]


def evaluate_ragas(questions, answers, contexts, ground_truths, cache_dir=None):
    if len({len(questions), len(answers), len(contexts), len(ground_truths)}) != 1:
        raise ValueError('All evaluation columns must have equal lengths')
    inputs = [
        {"question": q, "answer": a, "contexts": c, "ground_truth": g}
        for q, a, c, g in zip(questions, answers, contexts, ground_truths)
    ]
    try:
        if not inputs:
            raise ValueError('Evaluation dataset is empty')
        from datasets import Dataset
        from openai import APIConnectionError
        from ragas import evaluate
        from ragas.run_config import RunConfig

        from src.llm import create_ragas_models
        evaluator_llm, evaluator_embeddings, provider = create_ragas_models()
        rows = []
        from importlib.metadata import version
        evaluator_config = {
            'provider': provider,
            'model': getattr(evaluator_llm, 'model_name', getattr(evaluator_llm, 'model', 'unknown')),
            'embedding_model': getattr(evaluator_embeddings, 'model', 'unknown'),
            'ragas_version': version('ragas'),
            'adapter': hashlib.sha256(inspect.getsource(_evaluation_metrics).encode()).hexdigest(),
        }
        for index, item in enumerate(inputs):
            cache_path = None
            if cache_dir is not None:
                key = hashlib.sha256(json.dumps(
                    [evaluator_config, item], ensure_ascii=False, sort_keys=True,
                ).encode()).hexdigest()
                cache_path = Path(cache_dir) / f'{key}.json'
                if cache_path.exists():
                    cached = json.loads(cache_path.read_text(encoding='utf-8'))
                    if all(isinstance(cached.get(m), (int, float)) and math.isfinite(cached[m]) for m in METRICS):
                        rows.append(EvalResult(**item, **{m: cached[m] for m in METRICS}))
                        print(f'  RAGAS question {index + 1}/{len(inputs)}: cached', flush=True)
                        continue
            # Checkpoint each question; retry only failed calls, never select the
            # highest of multiple valid scores. A persistent failure stays failed.
            for attempt in range(2):
                try:
                    # RAGAS 0.1 creates an event loop per evaluate() call.
                    # Reusing an async HTTP client across these loops fails.
                    evaluator_llm, evaluator_embeddings, _ = create_ragas_models()
                    dataset = Dataset.from_dict({key: [value] for key, value in item.items()})
                    result = evaluate(dataset, metrics=_evaluation_metrics(provider),
                        llm=evaluator_llm, embeddings=evaluator_embeddings,
                        run_config=RunConfig(max_workers=1, max_retries=0, timeout=120),
                        raise_exceptions=True)
                    frame = result.to_pandas()
                    if len(frame) != 1:
                        raise ValueError('RAGAS returned an incomplete evaluation')
                    row = next(frame.iterrows())[1]
                    scores = {metric: float(row[metric]) for metric in METRICS}
                    invalid_metrics = [m for m, score in scores.items() if not math.isfinite(score)]
                    if invalid_metrics:
                        raise ValueError(
                            f'RAGAS returned non-finite scores for question {index}: '
                            f'{", ".join(invalid_metrics)}')
                    break
                except (ValueError, TimeoutError, APIConnectionError):
                    if attempt == 1:
                        raise
                    print(f'  Retrying failed RAGAS question {index + 1}', flush=True)
            rows.append(EvalResult(**item, **scores))
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(scores, allow_nan=False), encoding='utf-8')
            print(f'  RAGAS question {index + 1}/{len(inputs)}: completed', flush=True)
        return {**{m: sum(getattr(r, m) for r in rows) / len(rows) for m in METRICS},
                'per_question': rows, 'evaluation_status': 'completed',
                'evaluator': 'ragas', 'provider': provider, 'evaluator_config': evaluator_config}
    except Exception as error:  # noqa: BLE001 - RAGAS/provider failures must be reported as unscored.
        # The scaffold requires numeric fallbacks. Keep them explicitly unscored,
        # preserve inputs, and never mistake them for measured RAGAS scores.
        reason = f'{type(error).__name__}: {error}'
        print(f'RAGAS unavailable: {reason}')
        return {**dict.fromkeys(METRICS, 0.0), 'per_question': [],
                'evaluation_status': 'failed', 'error': reason,
                'unscored_inputs': inputs}

def failure_analysis(eval_results, bottom_n=10):
    tree = {
        'faithfulness': ('LLM hallucinating', 'Tighten prompt, lower temperature'),
        'context_recall': ('Missing relevant chunks', 'Improve chunking or add BM25'),
        'context_precision': ('Too many irrelevant chunks', 'Add reranking or metadata filter'),
        'answer_relevancy': ("Answer does not match question", 'Improve prompt template'),
    }
    failures = []
    for result in eval_results:
        scores = {m: getattr(result, m) for m in METRICS}
        if not all(math.isfinite(s) for s in scores.values()):
            raise ValueError('Cannot diagnose unmeasured scores')
        worst = min(scores, key=scores.get)
        diagnosis, fix = tree[worst]
        failures.append({'question': result.question, 'answer': result.answer,
            'contexts': result.contexts, 'ground_truth': result.ground_truth,
            'worst_metric': worst, 'score': sum(scores.values()) / 4,
            'metrics': scores, 'diagnosis': diagnosis, 'suggested_fix': fix})
    return sorted(failures, key=lambda f: f['score'])[:max(0, bottom_n)]

def save_report(results, failures, path='reports/ragas_report.json'):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = results.get('per_question', [])
    report = {'aggregate': {m: results.get(m, 0.0) for m in METRICS},
              'num_questions': len(rows), 'per_question': [asdict(row) for row in rows],
              'failures': failures, 'evaluation_status': results.get('evaluation_status', 'unknown')}
    for key in ('error', 'unscored_inputs', 'evaluator', 'provider', 'evaluator_config'):
        if key in results:
            report[key] = results[key]
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(f'Report saved to {path}')


def write_failure_analysis(
    results,
    failures,
    baseline_path='reports/naive_baseline_report.json',
    path='analysis/failure_analysis.md',
):
    """Render measured results and metric-based failure hypotheses for the lab."""
    baseline = {}
    try:
        with open(baseline_path, encoding='utf-8') as stream:
            baseline_report = json.load(stream)
        if baseline_report.get('evaluation_status') == 'completed':
            baseline = baseline_report.get('aggregate', {})
    except FileNotFoundError:
        pass

    lines = [
        '# Failure Analysis — Lab 18: Production RAG',
        '',
        '**Khóa:** K4 - Track 3A',
        '',
        '## RAGAS Scores',
        '',
        '| Metric | Naive Baseline | Production | Δ |',
        '|--------|---------------:|-----------:|--:|',
    ]
    completed = results.get('evaluation_status') == 'completed'
    for metric in METRICS:
        production = results.get(metric)
        naive = baseline.get(metric)
        if completed and production is not None:
            naive_text = f'{naive:.4f}' if naive is not None else 'N/A'
            delta = f'{production - naive:+.4f}' if naive is not None else 'N/A'
            lines.append(f'| {metric} | {naive_text} | {production:.4f} | {delta} |')
        else:
            lines.append(f'| {metric} | N/A | N/A | N/A |')

    lines.extend(['', '## Bottom-5 Failures', ''])
    if not completed:
        lines.extend([
            'Evaluation did not complete; no scores or failure rankings are reported.',
            f'**Reason:** {results.get("error", "Unknown evaluation error")}',
            '',
        ])
    elif not failures:
        lines.extend(['No per-question failures were returned by the evaluator.', ''])
    else:
        for index, failure in enumerate(failures[:5], start=1):
            metrics = failure['metrics']
            lines.extend([
                f'### #{index}',
                f'- **Question:** {failure["question"]}',
                f'- **Expected:** {failure["ground_truth"]}',
                f'- **Got:** {failure["answer"]}',
                f'- **Retrieved context:** {" | ".join(failure["contexts"])}',
                f'- **Worst metric:** {failure["worst_metric"]} ({metrics[failure["worst_metric"]]:.4f})',
                ('- **Error Tree:** Is the answer supported by the retrieved context? '
                'Check faithfulness; does the answer address the question? Check answer relevancy; '
                'does the context contain the expected evidence? Check context recall; '
                'are the retrieved passages focused? Check context precision.'),
                f'- **Root-cause hypothesis:** {failure["diagnosis"]}',
                f'- **Suggested fix:** {failure["suggested_fix"]}',
                '',
            ])

        first = failures[0]
        lines.extend([
            '## Case Study',
            '',
            f'**Question:** {first["question"]}',
            '',
            '**Error Tree walkthrough:**',
            f'1. Faithfulness: {first["metrics"]["faithfulness"]:.4f}; inspect whether each answer claim appears in context.',
            f'2. Context recall: {first["metrics"]["context_recall"]:.4f}; verify the expected evidence was retrieved.',
            f'3. Context precision: {first["metrics"]["context_precision"]:.4f}; inspect irrelevant passages.',
            f'4. Fix the measured bottleneck: {first["suggested_fix"]}.',
            '',
        ])

    lines.extend([
        '> Diagnoses are metric-based hypotheses, not proof of root cause; verify the cited answer and context manually.',
        '',
    ])
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Failure analysis saved to {output}')
