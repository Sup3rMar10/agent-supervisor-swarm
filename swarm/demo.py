"""A synthetic payment-dispute investigation: the workers, the data and the planners.

Every record is made up. The point is the shape of the work: three lookups that can run in parallel,
an analyst that discovers how many shipments need a carrier check and fans out (map), a drafter that
waits for all of it (reduce), and a reviewer that is flaky and has a stricter fallback.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Callable

from .core import Result, Subtask, PlanError, validate_plan

INVOICES = {
    "INV-1001": {"customer": "Acme Test Co", "amount": 1200.0},
    "INV-1002": {"customer": "Globex Sample Ltd", "amount": 560.0},
}
PAYMENTS = {"INV-1001": [400.0, 300.0], "INV-1002": []}
SHIPMENTS = {"INV-1001": ["TRK-A1", "TRK-A2", "TRK-A3"], "INV-1002": ["TRK-B1"]}
DELIVERED = {"TRK-A1": True, "TRK-A2": True, "TRK-A3": False, "TRK-B1": True}


class BaseWorker:
    name = "base"
    schema: dict = {}
    delay = 0.0

    def run(self, task: Subtask, inputs: dict) -> Result:
        if self.delay:
            time.sleep(self.delay)
        return self.work(task, inputs)

    def work(self, task: Subtask, inputs: dict) -> Result:  # pragma: no cover - overridden
        raise NotImplementedError


class InvoiceLookup(BaseWorker):
    name = "invoice_lookup"
    schema = {"invoice_id": str, "amount": float, "customer": str}

    def work(self, task, inputs):
        inv = task.input["invoice_id"]
        if inv not in INVOICES:
            return Result(ok=False, error=f"unknown invoice {inv}")
        return Result(ok=True, output={"invoice_id": inv, **INVOICES[inv]}, cost=0.01)


class PaymentLookup(BaseWorker):
    name = "payment_lookup"
    schema = {"payments": list, "paid": float}

    def work(self, task, inputs):
        pays = PAYMENTS.get(task.input["invoice_id"], [])
        return Result(ok=True, output={"payments": pays, "paid": float(sum(pays))}, cost=0.01)


class ShippingLookup(BaseWorker):
    name = "shipping_lookup"
    schema = {"shipments": list}

    def work(self, task, inputs):
        return Result(ok=True, output={"shipments": SHIPMENTS.get(task.input["invoice_id"], [])}, cost=0.01)


class CarrierCheck(BaseWorker):
    name = "carrier_check"
    schema = {"tracking": str, "delivered": bool}

    def work(self, task, inputs):
        t = task.input["tracking"]
        return Result(ok=True, output={"tracking": t, "delivered": DELIVERED.get(t, False)}, cost=0.005)


class Analyst(BaseWorker):
    """Computes the balance, then fans out one carrier check per shipment it just learned about."""

    name = "analyst"
    schema = {"outstanding": float, "shipments": int}

    def work(self, task, inputs):
        invoice = inputs["invoice"]
        paid = inputs["payments"]["paid"]
        shipments = inputs["shipping"]["shipments"]
        spawn = [
            Subtask(id=f"carrier:{t}", worker="carrier_check", input={"tracking": t}, deps=(), critical=False)
            for t in shipments
        ]
        return Result(
            ok=True,
            output={"outstanding": invoice["amount"] - paid, "shipments": len(shipments)},
            cost=0.02,
            spawn=spawn,
        )


class Drafter(BaseWorker):
    name = "drafter"
    schema = {"draft": str}

    def work(self, task, inputs):
        a = inputs["analysis"]
        carriers = [v for k, v in inputs.items() if k.startswith("carrier:")]
        undelivered = [c["tracking"] for c in carriers if not c["delivered"]]
        text = f"Outstanding balance {a['outstanding']:.2f}."
        if undelivered:
            text += f" Not yet delivered: {', '.join(sorted(undelivered))}; hold collection on those lines."
        elif carriers:
            text += " All shipments were delivered."
        return Result(ok=True, output={"draft": text}, cost=0.02)


class FlakyReviewer(BaseWorker):
    """Fails on its first call (a stand-in for a timeout), then behaves."""

    name = "reviewer"
    schema = {"approved": bool, "notes": str}

    def __init__(self, failures: int = 1):
        self._left = failures
        self._lock = threading.Lock()

    def work(self, task, inputs):
        with self._lock:
            if self._left > 0:
                self._left -= 1
                raise TimeoutError("reviewer backend timed out")
        return Result(ok=True, output={"approved": "Outstanding" in inputs["draft"]["draft"], "notes": "ok"}, cost=0.01)


class StrictReviewer(BaseWorker):
    name = "reviewer_strict"
    schema = {"approved": bool, "notes": str}

    def work(self, task, inputs):
        return Result(ok=True, output={"approved": False, "notes": "fallback reviewer: needs a human"}, cost=0.01)


def default_workers(reviewer_failures: int = 1, delay: float = 0.0) -> dict:
    ws = [InvoiceLookup(), PaymentLookup(), ShippingLookup(), CarrierCheck(), Analyst(), Drafter(),
          FlakyReviewer(reviewer_failures), StrictReviewer()]
    for w in ws:
        w.delay = delay
    return {w.name: w for w in ws}


# --- planners ---------------------------------------------------------------------------------


class RulePlanner:
    """Deterministic planner: always the same shape, with the invoice id taken from the goal."""

    def plan(self, goal: str) -> list[Subtask]:
        inv = next((tok.strip(".,") for tok in goal.split() if tok.startswith("INV-")), "INV-1001")
        i = {"invoice_id": inv}
        return [
            Subtask("invoice", "invoice_lookup", i),
            Subtask("payments", "payment_lookup", i),
            Subtask("shipping", "shipping_lookup", i),
            Subtask("analysis", "analyst", {}, deps=("invoice", "payments", "shipping")),
            Subtask("draft", "drafter", {}, deps=("analysis",)),
            Subtask("review", "reviewer", {}, deps=("draft",), retries=1, fallback="reviewer_strict"),
        ]


class LLMPlanner:
    """Asks a model for a plan as JSON and trusts none of it: the result must pass validate_plan.

    `transport` is any function prompt -> text (see llm.py for a Claude one). It is injected so the
    parsing and the rejection of bad plans can be tested without a network or a key.
    """

    MAX_TASKS = 12

    def __init__(self, transport: Callable[[str], str], worker_names: set[str]):
        self.transport = transport
        self.worker_names = worker_names

    def prompt(self, goal: str) -> str:
        return (
            "Return ONLY a JSON array of subtasks for this goal. Each item: "
            '{"id": str, "worker": str, "input": object, "deps": [str], "critical": bool}. '
            f"Allowed workers: {sorted(self.worker_names)}. Goal: {goal}"
        )

    def plan(self, goal: str) -> list[Subtask]:
        raw = self.transport(self.prompt(goal))
        try:
            start, end = raw.index("["), raw.rindex("]") + 1
            items = json.loads(raw[start:end])
        except ValueError as e:
            raise PlanError(f"planner did not return a JSON array: {e}") from e
        if not isinstance(items, list) or not items or len(items) > self.MAX_TASKS:
            raise PlanError(f"plan must have between 1 and {self.MAX_TASKS} tasks")
        plan = []
        for it in items:
            if not isinstance(it, dict) or not isinstance(it.get("id"), str) or not isinstance(it.get("worker"), str):
                raise PlanError("each task needs string 'id' and 'worker'")
            inp = it.get("input", {})
            deps = it.get("deps", [])
            if not isinstance(inp, dict) or not isinstance(deps, list) or not all(isinstance(d, str) for d in deps):
                raise PlanError(f"{it['id']}: bad 'input' or 'deps'")
            plan.append(Subtask(it["id"], it["worker"], inp, tuple(deps), bool(it.get("critical", True))))
        validate_plan(plan, self.worker_names)
        return plan
