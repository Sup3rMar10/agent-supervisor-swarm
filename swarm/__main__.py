"""python -m swarm demo [--invoice INV-1001] [--max-steps N] [--delay S]"""

from __future__ import annotations

import argparse
import json

from .core import Budget, Supervisor
from .demo import RulePlanner, default_workers


def main() -> int:
    p = argparse.ArgumentParser(prog="swarm")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="run the synthetic payment-dispute investigation")
    d.add_argument("--invoice", default="INV-1001")
    d.add_argument("--max-steps", type=int, default=50)
    d.add_argument("--delay", type=float, default=0.05, help="seconds each worker pretends to work")
    d.add_argument("--trace", action="store_true", help="print the full event trace")
    args = p.parse_args()

    plan = RulePlanner().plan(f"investigate {args.invoice}")
    sup = Supervisor(default_workers(delay=args.delay), budget=Budget(max_steps=args.max_steps))
    report = sup.run(plan)
    print(f"status: {report.status}   steps: {report.steps}   cost: {report.cost}")
    for tid, st in report.statuses.items():
        extra = f"  ({report.reasons[tid]})" if tid in report.reasons else ""
        print(f"  {tid:<18} {st}{extra}")
    if "draft" in report.results:
        print("draft:", report.results["draft"]["draft"])
    if args.trace:
        print(json.dumps(report.trace, indent=2))
    return 0 if report.status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
