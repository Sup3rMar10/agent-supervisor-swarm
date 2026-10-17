# agent-supervisor-swarm

![ci](https://github.com/Sup3rMar10/agent-supervisor-swarm/actions/workflows/ci.yml/badge.svg)

A **supervisor** that runs a multi-agent plan: specialist workers, a validated task graph, parallel execution, typed handoffs, a shared budget, loop detection, an explicit failure policy and dynamic fan-out. The idea is that everything a model must not be trusted with lives in ordinary, tested code around it.

> **What is verified, and what is not.** 41 tests pass locally on Python 3.13 (stdlib only; `pytest` to run them). I also broke the code on purpose twice and confirmed the tests catch it (removing the loop guard fails the loop test; forcing one worker thread fails the parallelism test). **There is no live model anywhere in this repo.** The workers are deterministic Python stand-ins over made-up data, so what is tested is the supervision, not whether a model plans or reasons well. The optional Claude transport (`swarm/llm.py`) has never been run against the live API.

## What the supervisor guarantees

| Concern | Behaviour | Tested by |
|---|---|---|
| Plan validity | The plan must be a DAG over known workers: no duplicate ids, unknown workers or fallbacks, unknown or self dependencies, or cycles. Checked **before anything runs**. | `test_core.py` plan tests |
| Parallelism | Independent tasks run concurrently (`max_parallel`). Four 0.25 s tasks finish in under 0.6 s; with `max_parallel=1` they take at least 0.3 s for three. | timing tests |
| Typed handoffs | Each worker declares an output schema. A result that does not match (missing key, string where a float was promised) is a **failure**, so garbage never reaches the next worker. | schema tests |
| Failure policy | Per task: retries on the same worker, then one fallback worker, then fail. A *critical* failure skips its dependents; an *optional* one lets them run without it. A worker that raises is a failed attempt, not a crashed supervisor. | policy tests |
| Budget | A shared cap on steps (including retries), cost and wall-clock deadline (injectable clock). When it is hit, the rest are marked `skipped` with the reason, and the report says `partial`. | budget tests |
| Loops and duplicates | Identical calls (same worker, input and dependency outputs) are cached once. A call repeated more than `max_repeats` times is stopped as a loop, long before a large retry allowance would be used up. | loop tests |
| Fan-out | A worker can return new subtasks. They are validated like any plan, depth-limited, count-limited, and whatever waited for the parent now also waits for them (map, then reduce). Invalid spawns are rejected and recorded; the parent still counts as done. | spawn tests |
| Trace | Every start, retry, fallback, dedup, spawn, skip and failure is an event with a time, for debugging and for the interview. | trace assertions |

All supervisor state changes happen on the supervisor's own thread. Workers only compute and return a `Result`.

## The demo: a synthetic payment dispute

`python -m swarm demo` runs a made-up investigation (all data is invented): three lookups in parallel, an analyst that learns there are three shipments and fans out **one carrier check per shipment**, a drafter that waits for all of it, and a reviewer that fails once (retried) and has a stricter fallback.

```
status: complete   steps: 10   cost: 0.095
draft: Outstanding balance 500.00. Not yet delivered: TRK-A3; hold collection on those lines.
```

`--max-steps 4` shows the budget at work (`status: partial`, the draft is never written).

## The LLM planner never gets to run an unvalidated plan

`LLMPlanner` asks a model for a plan as JSON and trusts none of it: it extracts the JSON array, checks the shape, limits the size (12 tasks), and runs `validate_plan`. Eight kinds of bad reply are tested and rejected: no JSON, an empty plan, an unknown worker (`rm_rf_everything`), a self-dependency, a cycle, a missing id, more than 12 tasks, and a non-object input. The model is injected as a plain function, so all of this is tested without a network or a key. `swarm/llm.py` holds a stdlib-only Claude transport (model id `claude-sonnet-5-5`) that is **untested live**.

## Honest limits

- **No live model, synthetic workers.** This tests orchestration. A real planner will make worse plans than the rule-based one; the validator is what makes that safe, not what makes it good.
- **Dedup assumes workers are pure.** Caching identical calls is right for lookups and wrong for a worker with side effects (sending an email twice should not be silently cached). Side-effecting steps need idempotency keys; see `agent-durable-runtime`.
- **Threads, not processes.** Fine for I/O-bound workers (model and API calls); CPU-bound workers would be limited by the GIL.
- **The budget is checked before a launch.** With `max_parallel > 1`, a few already-running tasks finish after the budget is hit, so spend can overshoot slightly. Running workers are not cancelled.
- **No backoff between retries** and no persistence: if the process dies, the run is gone. That is the next repo.
- Results are returned as plain dicts; there is no streaming and no human-approval step here.

## Run it

```
python -m pytest
python -m swarm demo
python -m swarm demo --max-steps 4 --trace
```

Python 3.11 or later; no dependencies besides `pytest` for the tests.
