"""Regression tests for boundaries the course smoke tests do not cover."""
import json
from types import SimpleNamespace

import pytest

from src import m5_enrichment as enrichment
from src.m1_chunking import chunk_hierarchical, chunk_structure_aware
from src.m2_search import BM25Search, SearchResult, reciprocal_rank_fusion
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import evaluate_ragas, save_report, write_failure_analysis


def test_hierarchy_long_unbroken_text_is_bounded_and_lossless():
    text = 'A' * 4100
    parents, children = chunk_hierarchical(text, parent_size=200, child_size=70)
    assert ''.join(p.text for p in parents) == text
    assert all(len(p.text) <= 200 for p in parents)
    assert all(len(c.text) <= 70 for c in children)
    for parent in parents:
        assert ''.join(c.text for c in children if c.parent_id == parent.parent_id) == parent.text


def test_hierarchy_ids_do_not_collide_between_documents():
    first, _ = chunk_hierarchical('same text', metadata={'source': 'one.md'})
    second, _ = chunk_hierarchical('same text', metadata={'source': 'two.md'})
    assert first[0].parent_id != second[0].parent_id


@pytest.mark.parametrize('size', [0, -1])
def test_hierarchy_invalid_sizes(size):
    with pytest.raises(ValueError):
        chunk_hierarchical('text', child_size=size)


def test_markdown_fenced_heading_and_table_stay_intact():
    text = '# Real\n\n```python\n# not a section\n```\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n## Next\nBody'
    chunks = chunk_structure_aware(text)
    assert len(chunks) == 2
    assert '# not a section' in chunks[0].text
    assert '| 1 | 2 |' in chunks[0].text


def test_rrf_uses_rank_not_raw_score_and_counts_each_list_once():
    a = SearchResult('A', 100, {}, 'bm25')
    b = SearchResult('B', .1, {}, 'dense')
    result = reciprocal_rank_fusion([[a, a, b], [b]])
    assert result[0].text == 'B'
    assert result[0].score == pytest.approx(1 / 63 + 1 / 61)
    assert result[1].score == pytest.approx(1 / 61)
    assert a.method == 'bm25'


def test_empty_index_does_not_crash():
    search = BM25Search()
    search.index([])
    assert search.search('anything') == []


def test_reranker_keeps_metadata_and_handles_one_score():
    reranker = CrossEncoderReranker()
    reranker._model = SimpleNamespace(predict=lambda pairs: .7)
    result = reranker.rerank('q', [{'text': 'a', 'score': .2, 'metadata': {'source': 'a'}}])
    assert result[0].rank == 0 and result[0].original_score == .2
    assert result[0].metadata['source'] == 'a'
    assert reranker.rerank('q', []) == []


def test_eval_missing_key_is_not_reported_as_measured_zero(monkeypatch, tmp_path):
    import config
    monkeypatch.setattr(config, 'GEMINI_API_KEY', '')
    monkeypatch.setattr(config, 'OPENAI_API_KEY', '')
    result = evaluate_ragas(['q'], ['a'], [['context']], ['truth'])
    assert result['evaluation_status'] == 'failed'
    assert result['per_question'] == []
    assert result['unscored_inputs'][0]['answer'] == 'a'
    save_report(result, [], tmp_path / 'report.json')
    report = json.loads((tmp_path / 'report.json').read_text(encoding='utf-8'))
    assert report['num_questions'] == 0


def test_eval_rejects_mismatched_columns():
    with pytest.raises(ValueError):
        evaluate_ragas(['q'], [], [['c']], ['g'])


def test_failure_analysis_does_not_invent_scores(tmp_path):
    path = tmp_path / 'failure_analysis.md'
    write_failure_analysis(
        {'evaluation_status': 'failed', 'error': 'quota exceeded'},
        [],
        baseline_path=tmp_path / 'missing_baseline.json',
        path=path,
    )
    report = path.read_text(encoding='utf-8')
    assert 'Evaluation did not complete' in report
    assert 'quota exceeded' in report
    assert '| faithfulness | N/A | N/A | N/A |' in report


def test_combined_enrichment_indexes_questions_and_protects_provenance(monkeypatch):
    monkeypatch.setattr(enrichment, '_enrich_single_call', lambda text, source: {
        'summary': 'short', 'questions': ['Which policy?'], 'context': 'Policy context',
        'metadata': {'parent_id': 'wrong', 'source': 'wrong'}})
    result = enrichment.enrich_chunks([{'text': 'Original evidence.',
        'metadata': {'parent_id': 'real', 'source': 'real.md'}}])[0]
    assert 'Which policy?' in result.enriched_text
    assert result.auto_metadata['parent_id'] == 'real'
    assert result.auto_metadata['source'] == 'real.md'
    assert result.original_text == 'Original evidence.'


def test_enrichment_cache_persists_successful_api_results(monkeypatch, tmp_path):
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps({
                        'summary': 'Summary',
                        'questions': ['Question?'],
                        'context': 'Context',
                        'metadata': {'category': 'policy'},
                    })
                )
            )
        ]
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: response,
            )
        )
    )
    monkeypatch.setattr(enrichment, 'ENRICHMENT_CACHE_PATH', tmp_path / 'cache.json')
    monkeypatch.setattr(enrichment, 'active_provider', lambda: 'gemini')
    monkeypatch.setattr(enrichment, 'chat_model_name', lambda: 'gemini-test')
    monkeypatch.setattr(enrichment, 'create_chat_client', lambda: client)
    monkeypatch.setattr(enrichment, 'wait_for_gemini_request', lambda: None)
    enrichment._enrich_single_call.cache_clear()

    first = enrichment._enrich_single_call('Evidence', 'policy.md')
    enrichment._enrich_single_call.cache_clear()
    second = enrichment._enrich_single_call('Evidence', 'policy.md')

    assert first == second
    assert second['summary'] == 'Summary'
    assert (tmp_path / 'cache.json').exists()


def test_pipeline_expands_parent_and_excludes_generated_hyqa(monkeypatch):
    import config
    from src.pipeline import run_query
    monkeypatch.setattr(config, 'GEMINI_API_KEY', '')
    monkeypatch.setattr(config, 'OPENAI_API_KEY', '')
    result = SearchResult('Generated questions', 1, {'parent_id': 'p', 'original_text': 'child'}, 'hybrid')
    search = SimpleNamespace(search=lambda q: [result], parent_map={'p': 'Complete parent evidence'})
    class Reranker:
        def rerank(self, query, documents, top_k):
            assert documents[0]['text'] == 'child'
            return [result]
    answer, contexts = run_query('query', search, Reranker())
    assert contexts == ['Complete parent evidence']
    assert answer == 'Complete parent evidence'


def test_pipeline_build_times_are_monotonic_and_load_model(monkeypatch):
    from src import pipeline
    clock = iter(range(100, 120))
    monkeypatch.setattr(pipeline.time, 'perf_counter', lambda: next(clock))
    monkeypatch.setattr(pipeline.time, 'time', lambda: 1_800_000_000)
    monkeypatch.setattr(pipeline, 'load_documents', lambda: [])
    monkeypatch.setattr(pipeline, 'enrich_chunks', lambda chunks: [])
    monkeypatch.setattr(pipeline, 'HybridSearch', lambda: SimpleNamespace(index=lambda chunks: None))
    loaded = []
    monkeypatch.setattr(pipeline, 'CrossEncoderReranker', lambda: SimpleNamespace(
        _load_model=lambda: loaded.append(True)))
    search, _ = pipeline.build_pipeline()
    assert loaded == [True]
    assert all(0 <= value < 20 for value in search.pipeline_timings.values())


def test_local_faithfulness_keeps_unpunctuated_answer():
    from src.m4_eval import _evaluation_metrics
    metric = _evaluation_metrics('ollama')[0]
    answer = 'Mức lương: 17 triệu VNĐ'
    captured = {}
    metric.statement_prompt = SimpleNamespace(format=lambda **kwargs: captured.update(kwargs))
    metric._create_statements_prompt({'question': 'Lương bao nhiêu?', 'answer': answer})
    assert answer in captured['sentences']


def test_submission_checker_rejects_incomplete_scores(tmp_path):
    from check_lab import check_json
    path = tmp_path / 'report.json'
    path.write_text(json.dumps({'aggregate': {}, 'num_questions': 0,
                               'evaluation_status': 'completed'}), encoding='utf-8')
    assert not check_json(str(path), ['aggregate', 'num_questions'])
