#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live, deterministic proof of OpenShell per-job isolation and cleanup.

This intentionally bypasses the research LLM. It exercises the production AI-Q
OpenShell provider against a real gateway, using fixed shell commands as the oracle.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import grpc

from aiq_agent.agents.deep_researcher.sandbox import SandboxConfig
from aiq_agent.agents.deep_researcher.sandbox import create_sandbox_backend
from aiq_agent.agents.deep_researcher.sandbox.providers.openshell import _policy_network_hosts
from aiq_agent.agents.deep_researcher.sandbox.providers.openshell import _read_policy_data


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway", help="Registered OpenShell gateway name; default uses the active gateway")
    parser.add_argument(
        "--policy",
        default="configs/openshell/aiq-research-policy.yaml",
        help="Policy applied to both probe sandboxes",
    )
    parser.add_argument(
        "--image",
        default=os.getenv("AIQ_OPENSHELL_IMAGE", "aiq-openshell-demo:latest"),
        help="Prebuilt OpenShell sandbox image",
    )
    parser.add_argument(
        "--expected-gateway-version",
        default="0.0.72",
        help="Fail unless the live gateway reports this exact version",
    )
    parser.add_argument(
        "--allow-best-effort-landlock",
        action="store_true",
        help="Permit a local-demo best_effort policy; never use this for production acceptance",
    )
    return parser.parse_args()


def _assert_deleted(client: object, name: str) -> None:
    try:
        client.get(name)  # type: ignore[attr-defined]
    except grpc.RpcError as exc:
        if isinstance(exc, grpc.Call) and exc.code() == grpc.StatusCode.NOT_FOUND:
            return
        raise
    raise AssertionError(f"sandbox still exists after cleanup: {name}")


def _attested(events: list[dict[str, object]]) -> bool:
    for event in events:
        if event.get("type") != "sandbox.attestation":
            continue
        data = event.get("data")
        if isinstance(data, dict) and data.get("status") == "succeeded":
            return isinstance(data.get("policy_version"), int) and data["policy_version"] > 0
    return False


def main() -> int:
    args = _args()
    policy_path = Path(args.policy).resolve()
    require_hard_landlock = not args.allow_best_effort_landlock
    policy_data = _read_policy_data(str(policy_path), require_hard_landlock=require_hard_landlock)
    hosts = tuple(sorted(_policy_network_hosts(policy_data)))
    network = {"mode": "allowlist", "allow": hosts} if hosts else {"mode": "blocked"}

    try:
        from openshell.sandbox import SandboxClient
    except ImportError as exc:
        raise RuntimeError("Run scripts/setup_openshell.sh before this smoke test") from exc

    with SandboxClient.from_active_cluster(cluster=args.gateway) as client:
        version = client.health().version
    if version != args.expected_gateway_version:
        raise AssertionError(
            f"gateway is {version}, expected {args.expected_gateway_version}; "
            "upgrade the gateway service before claiming acceptance"
        )
    print(f"PASS gateway version: {version}")

    suffix = uuid4().hex[:10]
    job_ids = (f"aiq-isolation-a-{suffix}", f"aiq-isolation-b-{suffix}")
    events: tuple[list[dict[str, object]], list[dict[str, object]]] = ([], [])

    def provider(job_id: str, sink: list[dict[str, object]]):
        config = SandboxConfig(
            provider="openshell",
            workdir="/sandbox",
            network=network,
            providers={
                "openshell": {
                    "gateway": args.gateway,
                    "policy": str(policy_path),
                    "image": args.image,
                    "require_hard_landlock": require_hard_landlock,
                }
            },
        )
        created = create_sandbox_backend(config, job_id)
        created.set_event_emitter(sink.append)
        return created

    sandboxes = (provider(job_ids[0], events[0]), provider(job_ids[1], events[1]))
    markers = (f"owner-a-{suffix}", f"owner-b-{suffix}")

    def initialize(index: int) -> str:
        sandbox = sandboxes[index]
        marker_file = f"{sandbox.workdir}/owner.txt"
        command = (
            f"printf %s {shlex.quote(markers[index])} > {shlex.quote(marker_file)}; cat {shlex.quote(marker_file)}"
        )
        result = sandbox.execute(command, timeout=30)
        if result.exit_code != 0 or result.output.strip() != markers[index]:
            raise AssertionError(f"job {index} workspace probe failed: exit={result.exit_code}")
        return sandbox.physical_sandbox_name

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            names = tuple(pool.map(initialize, range(2)))
        if names[0] == names[1]:
            raise AssertionError(f"jobs attached to the same physical sandbox: {names[0]}")

        with SandboxClient.from_active_cluster(cluster=args.gateway) as client:
            refs = (client.get(names[0]), client.get(names[1]))
            if refs[0].id == refs[1].id:
                raise AssertionError(f"jobs share physical sandbox id {refs[0].id}")
            if not all(ref.current_policy_version > 0 for ref in refs):
                raise AssertionError("a live sandbox has no loaded policy revision")
            if not all(_attested(job_events) for job_events in events):
                raise AssertionError("AI-Q did not emit successful positive-revision attestation events")
            print(
                "PASS distinct attested sandboxes: "
                f"A={names[0]}@r{refs[0].current_policy_version}, "
                f"B={names[1]}@r{refs[1].current_policy_version}"
            )

            sandboxes[0].terminate()
            _assert_deleted(client, names[0])
            if client.get(names[1]).id != refs[1].id:
                raise AssertionError("job B changed after cancelling job A")
            result = sandboxes[1].execute("printf %s still-alive", timeout=30)
            if result.exit_code != 0 or result.output.strip() != "still-alive":
                raise AssertionError("job B stopped working after job A was cancelled")
            print("PASS cancellation isolation: A deleted; B remained usable")

            sandboxes[1].close()
            _assert_deleted(client, names[1])
            print("PASS terminal cleanup: both probe sandboxes deleted")
    finally:
        for sandbox in sandboxes:
            sandbox.terminate()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - smoke script should print one direct failure
        print(f"FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
