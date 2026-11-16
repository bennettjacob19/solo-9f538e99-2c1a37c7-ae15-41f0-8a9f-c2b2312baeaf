"""End-to-end smoke test against running gateway + mock backend.

Usage:
    python scripts/smoke_test.py [--gateway URL] [--backend URL]

Requires the compose stack (or anything exposing the same endpoints) to be
up.  Prints a PASS/FAIL line per check and exits non-zero if any check fails.
The run leaves the mock backend in ``mode=ok``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import httpx


async def main(gateway: str, backend: str) -> int:
    failures = 0

    def check(label: str, condition: bool, detail: str = "") -> None:
        nonlocal failures
        status = "PASS" if condition else "FAIL"
        if not condition:
            failures += 1
        print(f"[{status}] {label}{(' :: ' + detail) if detail else ''}")

    async with httpx.AsyncClient(timeout=30) as client:
        for service, url in (("gateway", gateway), ("backend", backend)):
            health = await client.get(f"{url}/healthz")
            check(f"{service} healthz", health.status_code == 200)

        docs = await client.get(f"{gateway}/docs")
        check("gateway /docs reachable", docs.status_code == 200)

        # 1. Merge 4+4 texts into one batch and check index ordering.
        await client.post(f"{backend}/admin/reset")
        await client.post(
            f"{backend}/admin/config", json={"delay_ms": 250, "mode": "ok"}
        )
        groups = [["a1", "a2", "a3", "a4"], ["b1", "b2", "b3", "b4"]]
        responses = await asyncio.gather(
            *[
                client.post(
                    f"{gateway}/v1/embeddings",
                    json={"input": texts, "timeout_ms": 5000},
                )
                for texts in groups
            ]
        )
        check("merged batch returns 200", all(r.status_code == 200 for r in responses))
        bodies = [r.json() for r in responses]
        check(
            "both callers share one backend batch",
            bodies[0]["batch_id"] == bodies[1]["batch_id"],
            f"{bodies[0]['batch_id']} vs {bodies[1]['batch_id']}",
        )
        check(
            "batch size is 8",
            bodies[0]["batch_size"] == 8,
            str(bodies[0]["batch_size"]),
        )
        check(
            "indices preserved per caller",
            all(
                [item["index"] for item in body["data"]] == list(range(4))
                for body in bodies
            ),
        )
        stats = (await client.get(f"{backend}/admin/stats")).json()
        check(
            "backend saw one batch of 8",
            [len(entry["texts"]) for entry in stats["requests"]] == [8],
            str([len(entry["texts"]) for entry in stats["requests"]]),
        )
        check(
            "single-slot backend never ran two batches concurrently",
            stats["max_concurrency"] == 1 and stats["concurrency_violations"] == 0,
            f"max={stats['max_concurrency']} violations={stats['concurrency_violations']}",
        )

        # 2. Deadline expiry: late vectors are discarded, next batch waits.
        await client.post(f"{backend}/admin/reset")
        await client.post(
            f"{backend}/admin/config", json={"delay_ms": 500, "mode": "ok"}
        )
        expired = await client.post(
            f"{gateway}/v1/embeddings",
            json={"input": ["slow"], "timeout_ms": 80},
        )
        check("expired request returns 504", expired.status_code == 504, expired.text)
        started = time.monotonic()
        nxt = await client.post(
            f"{gateway}/v1/embeddings",
            json={"input": ["next"], "timeout_ms": 5000},
        )
        waited = time.monotonic() - started
        check("follow-up request succeeds", nxt.status_code == 200, nxt.text)
        check(
            "follow-up waited for the abandoned batch",
            waited >= 0.3,
            f"waited {waited:.2f}s",
        )
        gstats = (await client.get(f"{gateway}/stats")).json()
        check(
            "late vectors not counted as completed", gstats["completed_requests"] >= 1
        )
        check(
            "no stuck requests",
            gstats["pending_requests"] == 0 and not gstats["in_flight"],
        )
        bstats = (await client.get(f"{backend}/admin/stats")).json()
        check(
            "backend batches never overlap",
            bstats["max_concurrency"] == 1 and bstats["concurrency_violations"] == 0,
            f"max={bstats['max_concurrency']}",
        )

        # 3. Backend failure fails every waiter, then service recovers.
        await client.post(f"{backend}/admin/reset")
        await client.post(
            f"{backend}/admin/config", json={"delay_ms": 50, "mode": "error"}
        )
        failed = await asyncio.gather(
            *[
                client.post(
                    f"{gateway}/v1/embeddings",
                    json={"input": texts, "timeout_ms": 5000},
                )
                for texts in (["f1"], ["f2", "f3"])
            ]
        )
        check(
            "whole batch fails with 502",
            all(r.status_code == 502 for r in failed),
            str([r.status_code for r in failed]),
        )
        check(
            "a complete backend error does not raise a fault",
            (await client.get(f"{gateway}/admin/fault")).json()["active"] is False,
        )
        await client.post(f"{backend}/admin/config", json={"mode": "ok"})
        recovered = await client.post(
            f"{gateway}/v1/embeddings", json={"input": ["after"], "timeout_ms": 5000}
        )
        check("requests still served after backend error", recovered.status_code == 200)
        gstats = (await client.get(f"{gateway}/stats")).json()
        check(
            "no slot leak after error",
            gstats["pending_requests"] == 0 and not gstats["in_flight"],
            json.dumps(
                {
                    "pending": gstats["pending_requests"],
                    "in_flight": gstats["in_flight"],
                }
            ),
        )

        # 3b. Accepted-then-aborted response: the slot must be quarantined.
        await client.post(f"{backend}/admin/reset")
        await client.post(
            f"{backend}/admin/config",
            json={
                "mode": "accept_then_abort",
                "delay_ms": 0,
                "compute_after_abort_ms": 3000,
            },
        )
        lost = await client.post(
            f"{gateway}/v1/embeddings",
            json={"input": ["accepted-then-lost"], "timeout_ms": 5000},
        )
        check(
            "unconfirmed batch still answers 502",
            lost.status_code == 502,
            lost.text,
        )
        lost_error = lost.json()["error"]
        check(
            "502 carries fault id and reason",
            bool(lost_error.get("fault_id")) and bool(lost_error.get("reason")),
            json.dumps(lost_error),
        )
        fault_id = lost_error.get("fault_id")

        rejected = await client.post(
            f"{gateway}/v1/embeddings", json={"input": ["during-fault"]}
        )
        check(
            "new request during fault gets 503",
            rejected.status_code == 503,
            rejected.text,
        )
        rejected_error = rejected.json()["error"]
        check(
            "503 carries the same fault id",
            rejected_error.get("code") == "backend_unconfirmed"
            and rejected_error.get("fault_id") == fault_id,
            json.dumps(rejected_error),
        )
        slot = (await client.get(f"{backend}/slot")).json()
        check(
            "backend still reports the accepted batch as busy",
            slot["busy"] is True,
            json.dumps({"busy": slot["busy"]}),
        )
        gstats = (await client.get(f"{gateway}/stats")).json()
        check(
            "no second batch was dispatched while unconfirmed",
            slot["total_requests"] == 1
            and gstats["last_batch"]["outcome"] == "unconfirmed",
            json.dumps(
                {
                    "backend_requests": slot["total_requests"],
                    "last_batch": gstats["last_batch"],
                }
            ),
        )

        busy_recovery = await client.post(
            f"{gateway}/admin/fault/recover", json={"fault_id": fault_id}
        )
        check(
            "recovery is refused while the backend is busy",
            busy_recovery.status_code == 409,
            busy_recovery.text,
        )

        for _ in range(60):
            slot = (await client.get(f"{backend}/slot")).json()
            if not slot["busy"]:
                break
            await asyncio.sleep(0.25)
        check("backend eventually reports idle", slot["busy"] is False)

        recovered_fault = await client.post(
            f"{gateway}/admin/fault/recover",
            json={"fault_id": fault_id, "operator": "smoke-test", "reason": "idle"},
        )
        check(
            "operator recovery succeeds once idle",
            recovered_fault.status_code == 200,
            recovered_fault.text,
        )
        stale_recovery = await client.post(
            f"{gateway}/admin/fault/recover", json={"fault_id": fault_id}
        )
        check(
            "replaying the same recovery is rejected",
            stale_recovery.status_code == 409,
            stale_recovery.text,
        )
        await client.post(
            f"{backend}/admin/config", json={"mode": "ok", "delay_ms": 50}
        )
        served = await client.post(
            f"{gateway}/v1/embeddings", json={"input": ["after-recovery"]}
        )
        check("requests served after recovery", served.status_code == 200, served.text)
        check(
            "fault state visible in /docs-adjacent endpoints",
            (await client.get(f"{gateway}/admin/fault")).status_code == 200
            and (await client.get(f"{gateway}/openapi.json")).status_code == 200,
        )

        # 4. Validation limits.
        too_many = await client.post(
            f"{gateway}/v1/embeddings", json={"input": list("abcde")}
        )
        check(
            "5 texts rejected", too_many.status_code == 422, str(too_many.status_code)
        )
        zero_timeout = await client.post(
            f"{gateway}/v1/embeddings", json={"input": ["x"], "timeout_ms": 0}
        )
        check("timeout_ms=0 rejected", zero_timeout.status_code == 422)

    print()
    print("RESULT:", "ALL CHECKS PASSED" if failures == 0 else f"{failures} FAILURE(S)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway", default="http://127.0.0.1:8000")
    parser.add_argument("--backend", default="http://127.0.0.1:9000")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.gateway, args.backend)))
