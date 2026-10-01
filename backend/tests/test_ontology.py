"""The ontology is the contract the planner and analyzer both read."""

from eval_factory.domain.models import ScoreMode, SystemType
from eval_factory.domain.ontology import KNOWN_EVALUATORS, PROFILES, profile_for


def test_every_system_type_has_a_profile():
    assert set(PROFILES) == set(SystemType)


def test_metric_weights_sum_to_one():
    for profile in PROFILES.values():
        total = sum(metric.weight for metric in profile.metrics)
        assert abs(total - 1.0) < 1e-9, profile.system_type


def test_each_profile_mixes_both_evaluation_modes():
    for profile in PROFILES.values():
        modes = {metric.mode for metric in profile.metrics}
        assert modes == {ScoreMode.DETERMINISTIC, ScoreMode.PROBABILISTIC}
        assert profile.risks
        assert profile.default_components


def test_metric_ids_are_unique_and_evaluators_are_registered():
    seen: set[str] = set()
    for profile in PROFILES.values():
        for metric in profile.metrics:
            assert metric.id not in seen
            seen.add(metric.id)
            assert metric.evaluator_id in KNOWN_EVALUATORS
            assert 0 <= metric.threshold <= 1


def test_profile_lookup_rejects_unknown_types():
    assert profile_for("rag").system_type is SystemType.RAG
    try:
        profile_for("search")
    except Exception as exc:
        assert "rag, chatbot, agentic" in str(exc)
    else:
        raise AssertionError("expected an error")
