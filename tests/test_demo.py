import json

import pytest

from swarm import Budget, PlanError, Supervisor
from swarm.demo import LLMPlanner, RulePlanner, default_workers


def run(invoice="INV-1001", **kw):
    workers = default_workers(**kw.pop("workers", {}))
    plan = RulePlanner().plan(f"investigate {invoice}")
    return Supervisor(workers, **kw).run(plan)


def test_dispute_investigation_end_to_end():
    r = run()
    assert r.status == "complete"
    draft = r.results["draft"]["draft"]
    assert "Outstanding balance 500.00" in draft  # 1200 invoiced, 400 + 300 paid
    assert "TRK-A3" in draft  # the one undelivered shipment
    assert r.results["review"]["approved"] is True


def test_fan_out_created_one_carrier_check_per_shipment():
    r = run()
    assert sorted(t for t in r.statuses if t.startswith("carrier:")) == ["carrier:TRK-A1", "carrier:TRK-A2", "carrier:TRK-A3"]


def test_the_flaky_reviewer_is_retried_and_the_fallback_is_not_needed():
    r = run()
    events = [e["event"] for e in r.trace]
    assert "retry" in events and "fallback" not in events


def test_a_reviewer_that_keeps_failing_falls_back_to_the_strict_one():
    r = run(workers={"reviewer_failures": 5})
    assert r.status == "complete"
    assert r.results["review"]["approved"] is False
    assert "fallback" in [e["event"] for e in r.trace]


def test_an_unknown_invoice_fails_its_branch_and_skips_the_dependents():
    r = run("INV-9999")
    assert r.statuses["invoice"] == "failed"
    assert r.statuses["analysis"] == r.statuses["draft"] == r.statuses["review"] == "skipped"
    assert r.statuses["payments"] == "done"  # independent work still finished
    assert r.status == "partial"


def test_a_small_budget_leaves_the_draft_unwritten():
    r = run(budget=Budget(max_steps=3))
    assert r.status == "partial"
    assert r.statuses["draft"] == "skipped"
    assert r.steps == 3


# --- the LLM planner never gets to run an unvalidated plan --------------------------------------

WORKERS = set(default_workers())


def planner(reply):
    return LLMPlanner(lambda prompt: reply, WORKERS)


GOOD = json.dumps([
    {"id": "inv", "worker": "invoice_lookup", "input": {"invoice_id": "INV-1002"}},
    {"id": "pay", "worker": "payment_lookup", "input": {"invoice_id": "INV-1002"}, "deps": []},
])


def test_a_valid_json_plan_is_accepted():
    plan = planner(GOOD).plan("anything")
    assert [t.id for t in plan] == ["inv", "pay"]


def test_json_wrapped_in_prose_is_accepted():
    assert len(planner("Sure! Here is the plan:\n" + GOOD + "\nLet me know.").plan("x")) == 2


@pytest.mark.parametrize(
    "reply,fragment",
    [
        ("I cannot help with that", "JSON array"),
        ("[]", "between 1 and"),
        (json.dumps([{"id": "a", "worker": "rm_rf_everything"}]), "unknown worker"),
        (json.dumps([{"id": "a", "worker": "invoice_lookup", "deps": ["a"]}]), "itself"),
        (json.dumps([{"id": "a", "worker": "invoice_lookup", "deps": ["b"]},
                     {"id": "b", "worker": "invoice_lookup", "deps": ["a"]}]), "cycle"),
        (json.dumps([{"worker": "invoice_lookup"}]), "string 'id'"),
        (json.dumps([{"id": f"t{i}", "worker": "invoice_lookup"} for i in range(13)]), "between 1 and"),
        (json.dumps([{"id": "a", "worker": "invoice_lookup", "input": "not-an-object"}]), "bad 'input'"),
    ],
)
def test_bad_plans_are_rejected(reply, fragment):
    with pytest.raises(PlanError, match=fragment):
        planner(reply).plan("x")


def test_a_validated_llm_plan_runs():
    plan = planner(GOOD).plan("x")
    r = Supervisor(default_workers()).run(plan)
    assert r.status == "complete"
    assert r.results["inv"]["amount"] == 560.0
