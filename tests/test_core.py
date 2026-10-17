import threading
import time

import pytest

from swarm import Budget, PlanError, Result, Subtask, Supervisor, check_schema, validate_plan


class Fn:
    """A worker built from a function, for tests."""

    def __init__(self, name, fn, schema=None, delay=0.0):
        self.name = name
        self.fn = fn
        self.schema = schema if schema is not None else {}
        self.delay = delay
        self.calls = 0
        self._lock = threading.Lock()

    def run(self, task, inputs):
        with self._lock:
            self.calls += 1
            n = self.calls
        if self.delay:
            time.sleep(self.delay)
        return self.fn(task, inputs, n)


def ok(**out):
    return lambda task, inputs, n: Result(True, dict(out), cost=0.1)


def sup(workers, **kw):
    return Supervisor({w.name: w for w in workers}, **kw)


# --- plan validation ---------------------------------------------------------------------------


def test_duplicate_ids_rejected():
    with pytest.raises(PlanError, match="duplicate"):
        validate_plan([Subtask("a", "w"), Subtask("a", "w")], {"w"})


def test_unknown_worker_and_fallback_rejected():
    with pytest.raises(PlanError, match="unknown worker"):
        validate_plan([Subtask("a", "nope")], {"w"})
    with pytest.raises(PlanError, match="fallback"):
        validate_plan([Subtask("a", "w", fallback="nope")], {"w"})


def test_unknown_dependency_and_self_dependency_rejected():
    with pytest.raises(PlanError, match="unknown dependencies"):
        validate_plan([Subtask("a", "w", deps=("zzz",))], {"w"})
    with pytest.raises(PlanError, match="itself"):
        validate_plan([Subtask("a", "w", deps=("a",))], {"w"})


def test_cycle_rejected():
    plan = [Subtask("a", "w", deps=("c",)), Subtask("b", "w", deps=("a",)), Subtask("c", "w", deps=("b",))]
    with pytest.raises(PlanError, match="cycle"):
        validate_plan(plan, {"w"})


def test_supervisor_refuses_an_invalid_plan_before_running_anything():
    w = Fn("w", ok(x=1))
    with pytest.raises(PlanError):
        sup([w]).run([Subtask("a", "w", deps=("a",))])
    assert w.calls == 0


# --- scheduling --------------------------------------------------------------------------------


def test_dependencies_run_before_dependents_and_receive_their_output():
    a = Fn("a", ok(value=21), {"value": int})
    b = Fn("b", lambda t, i, n: Result(True, {"doubled": i["first"]["value"] * 2}), {"doubled": int})
    report = sup([a, b]).run([Subtask("first", "a"), Subtask("second", "b", deps=("first",))])
    assert report.status == "complete"
    assert report.results["second"] == {"doubled": 42}
    order = [e["task"] for e in report.trace if e["event"] == "start"]
    assert order == ["first", "second"]


def test_independent_tasks_really_run_in_parallel():
    w = Fn("w", ok(x=1), {"x": int}, delay=0.25)
    plan = [Subtask(f"t{i}", "w", {"i": i}) for i in range(4)]
    t0 = time.monotonic()
    report = sup([w], max_parallel=4).run(plan)
    elapsed = time.monotonic() - t0
    assert report.status == "complete"
    assert elapsed < 0.6, f"4 x 0.25s tasks took {elapsed:.2f}s, so they did not overlap"


def test_max_parallel_one_serialises():
    w = Fn("w", ok(x=1), {"x": int}, delay=0.1)
    plan = [Subtask(f"t{i}", "w", {"i": i}) for i in range(3)]
    t0 = time.monotonic()
    sup([w], max_parallel=1).run(plan)
    assert time.monotonic() - t0 >= 0.3


# --- typed handoffs ----------------------------------------------------------------------------


def test_schema_violation_is_a_failure_and_blocks_dependents():
    bad = Fn("bad", ok(amount="12"), {"amount": float})  # a string where a float was promised
    nxt = Fn("nxt", ok(x=1), {"x": int})
    report = sup([bad, nxt]).run([Subtask("a", "bad"), Subtask("b", "nxt", deps=("a",))])
    assert report.statuses == {"a": "failed", "b": "skipped"}
    assert "schema violation" in report.reasons["a"]
    assert nxt.calls == 0


def test_check_schema_details():
    assert check_schema({"a": int}, {"a": 1}) is None
    assert "missing" in check_schema({"a": int}, {})
    assert "should be int" in check_schema({"a": int}, {"a": True})  # bool is not an int here
    assert check_schema({"a": float}, {"a": 3}) is None  # an int is acceptable where a float is promised


def test_a_crashing_worker_is_a_failed_attempt_not_a_crashed_supervisor():
    def boom(t, i, n):
        raise RuntimeError("kaput")

    report = sup([Fn("w", boom)]).run([Subtask("a", "w")])
    assert report.statuses["a"] == "failed"
    assert "RuntimeError" in report.reasons["a"]


# --- failure policy ----------------------------------------------------------------------------


def test_retry_then_success():
    def flaky(t, i, n):
        return Result(False, error="timeout") if n == 1 else Result(True, {"x": 1})

    w = Fn("w", flaky, {"x": int})
    report = sup([w]).run([Subtask("a", "w", retries=2)])
    assert report.status == "complete"
    assert w.calls == 2
    assert any(e["event"] == "retry" for e in report.trace)


def test_fallback_worker_after_retries_run_out():
    primary = Fn("primary", lambda t, i, n: Result(False, error="down"))
    backup = Fn("backup", ok(x=1), {"x": int})
    report = sup([primary, backup]).run([Subtask("a", "primary", retries=1, fallback="backup")])
    assert report.status == "complete"
    assert primary.calls == 2 and backup.calls == 1
    assert any(e["event"] == "fallback" for e in report.trace)


def test_critical_failure_skips_dependents_but_not_unrelated_work():
    bad = Fn("bad", lambda t, i, n: Result(False, error="no"))
    good = Fn("good", ok(x=1), {"x": int})
    plan = [Subtask("a", "bad"), Subtask("b", "good", deps=("a",)), Subtask("c", "good", {"k": 1})]
    report = sup([bad, good]).run(plan)
    assert report.statuses == {"a": "failed", "b": "skipped", "c": "done"}
    assert report.status == "partial"


def test_optional_failure_lets_dependents_run_without_it():
    bad = Fn("bad", lambda t, i, n: Result(False, error="no"))
    seen = {}

    def final(t, i, n):
        seen.update(i)
        return Result(True, {"x": 1})

    plan = [Subtask("opt", "bad", critical=False), Subtask("end", "final", deps=("opt",))]
    report = sup([bad, Fn("final", final, {"x": int})]).run(plan)
    assert report.statuses["end"] == "done"
    assert "opt" not in seen
    assert report.status == "partial"  # one task failed even though the run reached its end


# --- budgets -----------------------------------------------------------------------------------


def test_step_budget_stops_the_run_and_marks_the_rest_skipped():
    w = Fn("w", ok(x=1), {"x": int})
    plan = [Subtask("a", "w", {"n": 1}), Subtask("b", "w", {"n": 2}, deps=("a",)),
            Subtask("c", "w", {"n": 3}, deps=("b",)), Subtask("d", "w", {"n": 4}, deps=("c",))]
    report = sup([w], budget=Budget(max_steps=2)).run(plan)
    assert report.steps == 2
    assert report.statuses == {"a": "done", "b": "done", "c": "skipped", "d": "skipped"}
    assert report.reasons["c"] == "step budget exhausted"


def test_cost_budget():
    w = Fn("w", lambda t, i, n: Result(True, {"x": 1}, cost=1.0), {"x": int})
    plan = [Subtask(f"t{i}", "w", {"i": i}, deps=(f"t{i-1}",) if i else ()) for i in range(5)]
    report = sup([w], budget=Budget(max_cost=2.0)).run(plan)
    assert sum(1 for s in report.statuses.values() if s == "done") == 2
    assert report.reasons["t2"] == "cost budget exhausted"


def test_deadline_uses_the_injected_clock():
    now = [0.0]

    def tick(t, i, n):
        now[0] += 10.0
        return Result(True, {"x": 1})

    w = Fn("w", tick, {"x": int})
    plan = [Subtask("a", "w", {"n": 1}), Subtask("b", "w", {"n": 2}, deps=("a",)), Subtask("c", "w", {"n": 3}, deps=("b",))]
    report = sup([w], budget=Budget(deadline_s=15.0), clock=lambda: now[0]).run(plan)
    assert report.statuses == {"a": "done", "b": "done", "c": "skipped"}
    assert report.reasons["c"] == "deadline passed"


# --- loops and duplicates ----------------------------------------------------------------------


def test_identical_calls_are_cached_not_repeated():
    w = Fn("w", ok(x=1), {"x": int})
    plan = [Subtask("a", "w", {"q": "same"}), Subtask("b", "w", {"q": "same"})]
    report = sup([w], max_parallel=1).run(plan)
    assert report.status == "complete"
    assert w.calls == 1
    assert any(e["event"] == "dedup" for e in report.trace)


def test_a_call_that_keeps_failing_the_same_way_is_stopped_as_a_loop():
    w = Fn("w", lambda t, i, n: Result(False, error="same error"))
    report = sup([w], max_repeats=3).run([Subtask("a", "w", retries=50)])
    assert report.statuses["a"] == "failed"
    assert report.steps == 3, "the loop guard should stop retries long before the retry allowance"
    assert "loop detected" in report.reasons["a"]


# --- dynamic fan-out ---------------------------------------------------------------------------


def _fan_out_workers(n_children):
    def parent(t, i, n):
        kids = [Subtask(f"kid{j}", "kid", {"j": j}) for j in range(n_children)]
        return Result(True, {"count": n_children}, spawn=kids)

    def kid(t, i, n):
        return Result(True, {"j": t.input["j"]})

    def collect(t, i, n):
        return Result(True, {"seen": sorted(k for k in i if k.startswith("kid"))})

    return [Fn("parent", parent, {"count": int}), Fn("kid", kid, {"j": int}),
            Fn("collect", collect, {"seen": list})]


def test_spawned_tasks_run_and_the_downstream_task_waits_for_them():
    report = sup(_fan_out_workers(3)).run(
        [Subtask("p", "parent"), Subtask("end", "collect", deps=("p",))]
    )
    assert report.status == "complete"
    assert report.results["end"]["seen"] == ["kid0", "kid1", "kid2"]
    assert any(e["event"] == "spawned" for e in report.trace)


def test_spawn_depth_is_limited():
    def endless(t, i, n):
        return Result(True, {"x": 1}, spawn=[Subtask(f"{t.id}.c", "endless", {"d": t.id})])

    w = Fn("endless", endless, {"x": int})
    report = sup([w], max_spawn_depth=2).run([Subtask("root", "endless")])
    assert sorted(report.statuses) == ["root", "root.c", "root.c.c"]
    assert any(e["event"] == "spawn_rejected" for e in report.trace)


def test_an_invalid_spawn_is_rejected_but_the_parent_still_counts_as_done():
    def bad_spawn(t, i, n):
        return Result(True, {"x": 1}, spawn=[Subtask("evil", "no-such-worker")])

    report = sup([Fn("w", bad_spawn, {"x": int})]).run([Subtask("a", "w")])
    assert report.statuses == {"a": "done"}
    rejected = [e for e in report.trace if e["event"] == "spawn_rejected"]
    assert rejected and "unknown worker" in rejected[0]["reason"]


def test_task_limit_caps_runaway_fan_out():
    def wide(t, i, n):
        return Result(True, {"x": 1}, spawn=[Subtask(f"k{j}", "leaf", {"j": j}) for j in range(50)])

    leaf = Fn("leaf", ok(x=1), {"x": int})
    report = sup([Fn("wide", wide, {"x": int}), leaf], max_tasks=10).run([Subtask("w", "wide")])
    assert len(report.statuses) == 1
    assert any(e["event"] == "spawn_rejected" and e["reason"] == "task limit" for e in report.trace)
