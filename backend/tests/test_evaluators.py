"""Coded checks are parallel and repeatable. Judges run one at a time."""

import threading
import time

from eval_factory.domain.models import (
    EvalCase,
    EvaluationRun,
    LLMRubric,
    MetricPlan,
    RubricCriterion,
    Rule,
    RuleBundle,
    RuleKind,
    SystemOutput,
    SystemType,
)
from eval_factory.evaluators.deterministic import run_deterministic
from eval_factory.evaluators.judges import run_judges
from eval_factory.planning.planner import attach_bundle, attach_rubric, default_plan


def _run(system_type: SystemType = SystemType.RAG) -> EvaluationRun:
    plan = default_plan(system_type)
    return EvaluationRun(name="demo", system_type=system_type, metric_plan=plan)


def _rag_case(response: str) -> tuple[EvalCase, SystemOutput]:
    case = EvalCase(
        id="c1",
        input="What is the refund window?",
        expected="30 days",
        context=["Refunds are accepted within 30 days of purchase."],
    )
    output = SystemOutput(case_id="c1", response=response, contexts=case.context)
    return case, output


def test_grounded_cited_answer_scores_higher_than_an_unsupported_one():
    grounded = _run()
    case, output = _rag_case("Refunds are accepted within 30 days of purchase. [1]")
    grounded.cases = [case]
    grounded.outputs = [output]
    drifted = _run()
    bad_case, bad_output = _rag_case("Everyone worldwide gets a lifetime replacement.")
    drifted.cases = [bad_case]
    drifted.outputs = [bad_output]

    good = {item.metric_id: item for item in run_deterministic(grounded)}
    bad = {item.metric_id: item for item in run_deterministic(drifted)}
    assert good["citation_support"].score == 1
    assert bad["citation_support"].score == 0
    assert good["context_overlap"].score > bad["context_overlap"].score
    assert good["expected_coverage"].score > bad["expected_coverage"].score
    again = run_deterministic(grounded)
    assert [item.model_dump() for item in again] == [item.model_dump() for item in run_deterministic(grounded)]


def test_deterministic_checks_overlap_on_worker_threads():
    run = _run(SystemType.CHATBOT)
    run.cases = [
        EvalCase(id=f"c{index}", input="Help me reset my password please", expected="Use the reset link")
        for index in range(4)
    ]
    run.outputs = [
        SystemOutput(case_id=case.id, response="Use the reset link in the account settings page to continue.")
        for case in run.cases
    ]
    seen: list[str] = []
    lock = threading.Lock()
    started = threading.Event()

    def slow(metric, case, output, current):
        del metric, output, current
        with lock:
            seen.append(threading.current_thread().name)
            if len(seen) >= 4:
                started.set()
        assert started.wait(2)
        from eval_factory.domain.models import EvaluatorResult, ScoreMode

        return EvaluatorResult(
            case_id=case.id,
            metric_id="probe",
            evaluator_id="probe",
            mode=ScoreMode.DETERMINISTIC,
            score=1,
            passed=True,
            rationale="ok",
        )

    run.metric_plan = MetricPlan(
        system_type=SystemType.CHATBOT,
        metrics=[
            run.metric_plan.metrics[0].model_copy(
                update={"id": "probe", "evaluator_id": "probe", "mode": "deterministic"}
            )
        ],
    )
    run_deterministic(run, registry={"probe": slow}, max_workers=4)
    assert len(set(seen)) > 1


def test_custom_rule_bundle_scores_only_its_rules():
    run = _run(SystemType.CHATBOT)
    bundle = RuleBundle(
        name="Ticket language",
        rules=[
            Rule(name="mentions ticket", kind=RuleKind.CONTAINS, params={"value": "ticket"}),
            Rule(name="no refund promise", kind=RuleKind.NOT_CONTAINS, params={"value": "lifetime"}),
        ],
    )
    run.rule_bundles = [bundle]
    run.metric_plan = attach_bundle(default_plan(SystemType.CHATBOT), bundle)
    run.metric_plan.metrics = [metric for metric in run.metric_plan.metrics if metric.evaluator_id == "rule_bundle"]
    run.cases = [EvalCase(id="c1", input="Where is my ticket?")]
    run.outputs = [SystemOutput(case_id="c1", response="Your ticket is in the billing queue.")]
    result = run_deterministic(run)[0]
    assert result.score == 1


def test_judges_are_deterministic_offline_and_run_in_series():
    run = _run()
    case, output = _rag_case("Refunds are accepted within 30 days of purchase. [1]")
    run.cases = [case, EvalCase(id="c2", input="What is the refund window?", expected="30 days", context=case.context)]
    run.outputs = [output, SystemOutput(case_id="c2", response="I am not sure about that.", contexts=case.context)]
    first = run_judges(run)
    second = run_judges(run)
    assert [item.model_dump() for item in first] == [item.model_dump() for item in second]
    by_case = {}
    for item in first:
        by_case.setdefault(item.case_id, {})[item.metric_id] = item.score
    assert by_case["c1"]["faithfulness"] > by_case["c2"]["faithfulness"]

    calls: list[float] = []
    state = {"active": 0}

    class SequentialProvider:
        name = "openai"

        def complete_json(self, *, system, user, max_tokens=1200):
            del system, user, max_tokens
            assert state["active"] == 0
            state["active"] += 1
            calls.append(time.perf_counter())
            time.sleep(0.02)
            state["active"] -= 1
            return {"score": 0.5, "rationale": "reviewed"}

    judged = run_judges(run, provider=SequentialProvider())
    assert len(calls) == len(judged)
    assert all(later - earlier >= 0.015 for earlier, later in zip(calls, calls[1:]))


def test_custom_rubric_is_a_probabilistic_metric():
    run = _run(SystemType.CHATBOT)
    rubric = LLMRubric(
        name="Empathy",
        prompt="Acknowledge the problem.",
        criteria=[RubricCriterion(name="Acknowledgement", description="Names the user's problem clearly.")],
    )
    run.rubrics = [rubric]
    plan = attach_rubric(default_plan(SystemType.CHATBOT), rubric)
    plan.metrics = [metric for metric in plan.metrics if metric.evaluator_id == "llm_rubric"]
    run.metric_plan = plan
    run.cases = [EvalCase(id="c1", input="My invoice is wrong")]
    run.outputs = [SystemOutput(case_id="c1", response="Acknowledgement: your invoice problem is clear and I can fix it.")]
    result = run_judges(run)[0]
    assert result.mode.value == "probabilistic"
    assert result.score > 0.5
