"""A supervisor that runs a plan of subtasks across specialist workers.

The supervisor owns everything that must not be left to a model: the plan is validated as a DAG,
every worker result is checked against a declared schema before anything downstream sees it,
spend is capped by a shared budget, repeated identical calls are cached or stopped, and failures
follow an explicit policy (retry, fallback worker, critical or optional). All state changes happen
on the supervisor's own thread; workers only compute and return a Result.
"""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


class PlanError(ValueError):
    """The plan is not a valid DAG over known workers."""


@dataclass(frozen=True)
class Subtask:
    id: str
    worker: str
    input: dict = field(default_factory=dict)
    deps: tuple[str, ...] = ()
    critical: bool = True  # if it fails, tasks that depend on it are skipped; otherwise they run without it
    retries: int = 0  # extra attempts on the same worker after a failure
    fallback: str | None = None  # worker to try once after the retries are used up


@dataclass
class Result:
    ok: bool
    output: dict = field(default_factory=dict)
    error: str | None = None
    cost: float = 0.0
    spawn: list[Subtask] = field(default_factory=list)  # new subtasks the supervisor may add (fan-out)


class Worker(Protocol):
    name: str
    schema: dict[str, type]  # the output keys and types this worker promises

    def run(self, task: Subtask, inputs: dict[str, dict]) -> Result: ...


@dataclass
class Budget:
    max_steps: int = 50  # worker invocations, including retries and fallbacks
    max_cost: float = float("inf")
    deadline_s: float | None = None


@dataclass
class RunReport:
    status: str  # complete: every task done | partial: some done | failed: none done
    results: dict[str, dict]
    statuses: dict[str, str]
    reasons: dict[str, str]
    steps: int
    cost: float
    trace: list[dict]


def validate_plan(plan: list[Subtask], worker_names: set[str], known_ids: set[str] = frozenset()) -> None:
    ids = [t.id for t in plan]
    if len(set(ids)) != len(ids):
        raise PlanError("duplicate task ids")
    clash = set(ids) & set(known_ids)
    if clash:
        raise PlanError(f"task ids already in use: {sorted(clash)}")
    universe = set(ids) | set(known_ids)
    for t in plan:
        if t.worker not in worker_names:
            raise PlanError(f"{t.id}: unknown worker {t.worker!r}")
        if t.fallback is not None and t.fallback not in worker_names:
            raise PlanError(f"{t.id}: unknown fallback worker {t.fallback!r}")
        missing = [d for d in t.deps if d not in universe]
        if missing:
            raise PlanError(f"{t.id}: unknown dependencies {missing}")
        if t.id in t.deps:
            raise PlanError(f"{t.id}: depends on itself")
        if t.retries < 0:
            raise PlanError(f"{t.id}: negative retries")
    # Kahn's algorithm over the new tasks only (known tasks are already acyclic and cannot depend on new ones)
    indeg = {t.id: sum(1 for d in t.deps if d in ids) for t in plan}
    children: dict[str, list[str]] = {t.id: [] for t in plan}
    for t in plan:
        for d in t.deps:
            if d in children:
                children[d].append(t.id)
    queue = [i for i, n in indeg.items() if n == 0]
    seen = 0
    while queue:
        cur = queue.pop()
        seen += 1
        for c in children[cur]:
            indeg[c] -= 1
            if indeg[c] == 0:
                queue.append(c)
    if seen != len(plan):
        raise PlanError("the plan contains a cycle")


def check_schema(schema: dict[str, type], output: dict) -> str | None:
    for key, typ in schema.items():
        if key not in output:
            return f"missing key {key!r}"
        value = output[key]
        if typ is float and isinstance(value, int) and not isinstance(value, bool):
            continue
        if not isinstance(value, typ) or (typ is int and isinstance(value, bool)):
            return f"key {key!r} should be {typ.__name__}, got {type(value).__name__}"
    return None


def _dedupe_key(worker: str, task_input: dict, inputs: dict[str, dict]) -> str:
    blob = json.dumps([worker, task_input, inputs], sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


class Supervisor:
    def __init__(
        self,
        workers: dict[str, Worker],
        budget: Budget | None = None,
        max_parallel: int = 4,
        max_spawn_depth: int = 2,
        max_tasks: int = 100,
        max_repeats: int = 3,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.workers = workers
        self.budget = budget or Budget()
        self.max_parallel = max_parallel
        self.max_spawn_depth = max_spawn_depth
        self.max_tasks = max_tasks
        self.max_repeats = max_repeats
        self.clock = clock

    def run(self, plan: list[Subtask]) -> RunReport:
        validate_plan(plan, set(self.workers))
        tasks: dict[str, Subtask] = {t.id: t for t in plan}
        deps: dict[str, set[str]] = {t.id: set(t.deps) for t in plan}
        depth = {t.id: 0 for t in plan}
        status = {t.id: "pending" for t in plan}
        reasons: dict[str, str] = {}
        results: dict[str, Result] = {}
        worker_for = {t.id: t.worker for t in plan}
        retries_left = {t.id: t.retries for t in plan}
        fallback_used: dict[str, bool] = {}
        trace: list[dict] = []
        seen: dict[str, Result] = {}
        repeats: dict[str, int] = {}
        state = {"steps": 0, "cost": 0.0}
        started = self.clock()

        def ev(event: str, **kw: Any) -> None:
            trace.append({"t": round(self.clock() - started, 4), "event": event, **kw})

        def satisfied(d: str) -> bool:
            return status[d] == "done" or (status[d] == "failed" and not tasks[d].critical)

        def blocked(d: str) -> bool:
            return status[d] == "skipped" or (status[d] == "failed" and tasks[d].critical)

        def budget_reason() -> str | None:
            b = self.budget
            if state["steps"] >= b.max_steps:
                return "step budget exhausted"
            if state["cost"] >= b.max_cost:
                return "cost budget exhausted"
            if b.deadline_s is not None and self.clock() - started >= b.deadline_s:
                return "deadline passed"
            return None

        def finish_ok(tid: str, res: Result) -> None:
            status[tid] = "done"
            results[tid] = res
            ev("done", task=tid, worker=worker_for[tid])
            if res.spawn:
                adopt_spawn(tid, res.spawn)

        def adopt_spawn(parent: str, spawned: list[Subtask]) -> None:
            if depth[parent] + 1 > self.max_spawn_depth:
                ev("spawn_rejected", task=parent, reason="spawn depth limit")
                return
            if len(tasks) + len(spawned) > self.max_tasks:
                ev("spawn_rejected", task=parent, reason="task limit")
                return
            try:
                validate_plan(spawned, set(self.workers), known_ids=set(tasks))
            except PlanError as e:
                ev("spawn_rejected", task=parent, reason=str(e))
                return
            new_ids = []
            for s in spawned:
                tasks[s.id] = s
                deps[s.id] = set(s.deps)
                depth[s.id] = depth[parent] + 1
                status[s.id] = "pending"
                worker_for[s.id] = s.worker
                retries_left[s.id] = s.retries
                new_ids.append(s.id)
            # Fan-out then fan-in: whatever was waiting for the parent now also waits for what it spawned.
            for other, d in deps.items():
                if parent in d and status[other] == "pending" and other not in new_ids:
                    d.update(new_ids)
            ev("spawned", task=parent, new=new_ids)

        def fail(tid: str, error: str) -> None:
            # Retry on the same worker, then the fallback worker once, then give up.
            if retries_left[tid] > 0:
                retries_left[tid] -= 1
                status[tid] = "pending"
                ev("retry", task=tid, worker=worker_for[tid], error=error)
                return
            fb = tasks[tid].fallback
            if fb and not fallback_used.get(tid):
                fallback_used[tid] = True
                worker_for[tid] = fb
                status[tid] = "pending"
                ev("fallback", task=tid, worker=fb, error=error)
                return
            status[tid] = "failed"
            reasons[tid] = error
            ev("failed", task=tid, worker=worker_for[tid], error=error)

        def call(worker: Worker, task: Subtask, inputs: dict[str, dict]) -> Result:
            try:
                return worker.run(task, inputs)
            except Exception as e:  # a crashing worker is a failed attempt, not a crashed supervisor
                return Result(ok=False, error=f"exception: {type(e).__name__}: {e}")

        running: dict[Future, tuple[str, str]] = {}

        with ThreadPoolExecutor(max_workers=self.max_parallel) as pool:
            while True:
                # Tasks whose critical dependency failed or was skipped can never run.
                changed = True
                while changed:
                    changed = False
                    for tid, st in status.items():
                        if st == "pending" and any(blocked(d) for d in deps[tid]):
                            status[tid] = "skipped"
                            reasons[tid] = "a dependency failed or was skipped"
                            ev("skipped", task=tid, reason=reasons[tid])
                            changed = True

                reason = budget_reason()
                if reason is None:
                    for tid in list(status):
                        if len(running) >= self.max_parallel:
                            break
                        if status[tid] != "pending" or not all(satisfied(d) for d in deps[tid]):
                            continue
                        task = tasks[tid]
                        worker_name = worker_for[tid]
                        inputs = {d: results[d].output for d in sorted(deps[tid]) if d in results}
                        key = _dedupe_key(worker_name, task.input, inputs)
                        if key in seen:
                            ev("dedup", task=tid, worker=worker_name)
                            # Same output, no cost, and no second round of spawned tasks.
                            finish_ok(tid, Result(ok=True, output=seen[key].output))
                            continue
                        repeats[key] = repeats.get(key, 0) + 1
                        if repeats[key] > self.max_repeats:
                            fail(tid, "loop detected: the same call was repeated too often")
                            continue
                        state["steps"] += 1
                        status[tid] = "running"
                        ev("start", task=tid, worker=worker_name)
                        fut = pool.submit(call, self.workers[worker_name], task, inputs)
                        running[fut] = (tid, key)
                else:
                    for tid, st in status.items():
                        if st == "pending":
                            status[tid] = "skipped"
                            reasons[tid] = reason
                            ev("skipped", task=tid, reason=reason)

                if not running:
                    if any(st == "pending" for st in status.values()) and reason is None:
                        continue  # something was just deduplicated or re-queued; scan again
                    break

                done, _ = wait(list(running), return_when=FIRST_COMPLETED)
                for fut in done:
                    tid, key = running.pop(fut)
                    res: Result = fut.result()
                    state["cost"] += res.cost
                    if res.ok:
                        problem = check_schema(self.workers[worker_for[tid]].schema, res.output)
                        if problem:
                            fail(tid, f"schema violation: {problem}")
                            continue
                        seen[key] = res
                        finish_ok(tid, res)
                    else:
                        fail(tid, res.error or "worker reported failure")

        done_n = sum(1 for s in status.values() if s == "done")
        overall = "complete" if done_n == len(status) else ("partial" if done_n else "failed")
        return RunReport(
            status=overall,
            results={t: r.output for t, r in results.items()},
            statuses=dict(status),
            reasons=reasons,
            steps=state["steps"],
            cost=round(state["cost"], 6),
            trace=trace,
        )
