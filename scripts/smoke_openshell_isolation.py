#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live, deterministic proof of OpenShell per-job isolation and cleanup.

This intentionally bypasses the research LLM. It exercises the production AI-Q
OpenShell provider against a real gateway, using fixed shell commands as the oracle.
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import logging
import os
import shlex
import sys
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import grpc
import yaml

from aiq_agent.agents.deep_researcher.sandbox import SandboxConfig
from aiq_agent.agents.deep_researcher.sandbox import create_sandbox_backend
from aiq_agent.agents.deep_researcher.sandbox.providers.openshell import _build_sandbox_spec
from aiq_agent.agents.deep_researcher.sandbox.providers.openshell import _deterministic_policy_hash
from aiq_agent.agents.deep_researcher.sandbox.providers.openshell import _parse_policy_proto
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


def _attested(events: list[dict[str, object]], *, expected_hash: str) -> bool:
    for event in events:
        if event.get("type") != "sandbox.attestation":
            continue
        data = event.get("data")
        if isinstance(data, dict) and data.get("status") == "succeeded":
            return (
                isinstance(data.get("policy_version"), int)
                and data["policy_version"] > 0
                and data.get("policy_hash") == expected_hash
                and data.get("policy_source") == 1
                and data.get("assurance") == "strict"
                and data.get("reason_code") is None
            )
    return False


def _loaded_policy_revision(client: object, name: str) -> int:
    """Return the loaded revision, including the OpenShell 0.0.72 status-RPC fallback."""
    from openshell._proto import openshell_pb2

    sandbox = client.get(name)  # type: ignore[attr-defined]
    if sandbox.current_policy_version > 0:
        return sandbox.current_policy_version

    # OpenShell 0.0.72 can leave SandboxStatus.current_policy_version at zero even
    # after the authoritative policy status RPC records the initial revision as loaded.
    stub = getattr(client, "_stub", None)
    if stub is None or not hasattr(stub, "GetSandboxPolicyStatus"):
        return 0
    response = stub.GetSandboxPolicyStatus(
        openshell_pb2.GetSandboxPolicyStatusRequest(name=name, version=0),
        timeout=30,
    )
    revision = response.revision
    if revision.status != openshell_pb2.POLICY_STATUS_LOADED:
        return 0
    return revision.version


def main() -> int:
    args = _args()
    policy_path = Path(args.policy).resolve()
    require_hard_landlock = not args.allow_best_effort_landlock
    policy_data = _read_policy_data(str(policy_path), require_hard_landlock=require_hard_landlock)
    expected_policy = _parse_policy_proto(policy_data, policy_path=str(policy_path))
    expected_policy_hash = _deterministic_policy_hash(expected_policy)
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

    def provider(
        job_id: str,
        emitter: Callable[[dict[str, object]], None],
        *,
        configured_policy: Path = policy_path,
        shared_name: str | None = None,
    ):
        openshell_config: dict[str, object] = {
            "gateway": args.gateway,
            "policy": str(configured_policy),
            "image": args.image,
            "require_hard_landlock": require_hard_landlock,
        }
        if shared_name is not None:
            openshell_config.update(
                existing_sandbox_name=shared_name,
                allow_shared_sandbox=True,
            )
        config = SandboxConfig(
            provider="openshell",
            workdir="/sandbox",
            network=network,
            providers={"openshell": openshell_config},
        )
        created = create_sandbox_backend(config, job_id)
        created.set_event_emitter(emitter)
        return created

    sandboxes = (provider(job_ids[0], events[0].append), provider(job_ids[1], events[1].append))
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
            revisions = tuple(_loaded_policy_revision(client, name) for name in names)
            if not all(revision > 0 for revision in revisions):
                raise AssertionError("a live sandbox has no loaded policy revision")
            if not all(_attested(job_events, expected_hash=expected_policy_hash) for job_events in events):
                raise AssertionError("AI-Q did not emit strict source/content/hash attestation events")
            print(f"PASS distinct attested sandboxes: A={names[0]}@r{revisions[0]}, B={names[1]}@r{revisions[1]}")

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

    # Prove that a failed job still reaches terminal deletion and that an exception
    # containing a credential-like canary cannot enter AI-Q logs or events.
    secret_canary = f"credential=aiq-smoke-{suffix}"
    captured_logs = io.StringIO()
    capture_handler = logging.StreamHandler(captured_logs)
    capture_handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(capture_handler)
    redaction_events: list[dict[str, object]] = []

    def fail_event_delivery(event: dict[str, object]) -> None:
        redaction_events.append(event)
        raise RuntimeError(secret_canary)

    failed_sandbox = provider(f"aiq-isolation-failure-{suffix}", fail_event_delivery)
    failed_name: str | None = None
    try:
        result = failed_sandbox.execute("exit 23", timeout=30)
        failed_name = failed_sandbox.physical_sandbox_name
        if result.exit_code != 23:
            raise AssertionError(f"failure probe returned unexpected exit code {result.exit_code}")
        failed_sandbox.close()
        with SandboxClient.from_active_cluster(cluster=args.gateway) as client:
            _assert_deleted(client, failed_name)
    finally:
        failed_sandbox.terminate()
        root_logger.removeHandler(capture_handler)
        capture_handler.close()
    if secret_canary in captured_logs.getvalue() or secret_canary in json.dumps(redaction_events):
        raise AssertionError("credential canary appeared in captured AI-Q logs or events")
    print("PASS failure cleanup: failed job sandbox deleted")
    print("PASS log redaction: credential canary absent from captured logs and events")

    # Create a shared debug sandbox with the submitted policy, then prove that an
    # attachment claiming a structurally different policy fails strict attestation.
    mismatch_data = copy.deepcopy(policy_data)
    filesystem = mismatch_data.get("filesystem_policy") or mismatch_data.get("filesystem")
    if not isinstance(filesystem, dict):
        raise AssertionError("policy has no filesystem section for the mismatch probe")
    read_only = list(filesystem.get("read_only") or [])
    read_only.append("/__aiq_attestation_mismatch__")
    filesystem["read_only"] = read_only

    shared_name: str | None = None
    mismatch_provider = None
    try:
        with tempfile.TemporaryDirectory(prefix="aiq-openshell-smoke-") as temp_dir:
            mismatch_path = Path(temp_dir) / "mismatched-policy.yaml"
            mismatch_path.write_text(yaml.safe_dump(mismatch_data, sort_keys=False), encoding="utf-8")
            with SandboxClient.from_active_cluster(cluster=args.gateway) as client:
                shared_ref = client.create(
                    spec=_build_sandbox_spec(
                        policy=expected_policy,
                        image=args.image,
                        job_id=f"aiq-shared-mismatch-{suffix}",
                    )
                )
                shared_name = shared_ref.name
                client.wait_ready(shared_name)

            mismatch_events: list[dict[str, object]] = []
            mismatch_provider = provider(
                f"aiq-mismatched-attach-{suffix}",
                mismatch_events.append,
                configured_policy=mismatch_path,
                shared_name=shared_name,
            )
            try:
                mismatch_provider.execute("true", timeout=30)
            except RuntimeError as exc:
                if "policy_content_mismatch" not in str(exc):
                    raise AssertionError("shared policy mismatch failed for an unexpected reason") from exc
            else:
                raise AssertionError("shared sandbox accepted a mismatched policy")

            with SandboxClient.from_active_cluster(cluster=args.gateway) as client:
                client.get(shared_name)
            if any(
                isinstance(event.get("data"), dict) and event["data"].get("status") == "succeeded"
                for event in mismatch_events
            ):
                raise AssertionError("mismatched shared policy emitted attestation success")
    finally:
        if mismatch_provider is not None:
            mismatch_provider.terminate()
        if shared_name is not None:
            with SandboxClient.from_active_cluster(cluster=args.gateway) as client:
                try:
                    client.get(shared_name)
                except grpc.RpcError as exc:
                    if not isinstance(exc, grpc.Call) or exc.code() != grpc.StatusCode.NOT_FOUND:
                        raise
                else:
                    if not client.delete(shared_name):
                        raise AssertionError("shared mismatch probe could not be deleted")
                    client.wait_deleted(shared_name)
    if shared_name is None:
        raise AssertionError("shared mismatch probe was not created")
    with SandboxClient.from_active_cluster(cluster=args.gateway) as client:
        _assert_deleted(client, shared_name)
    print("PASS shared-policy rejection: mismatched attachment denied and probe deleted")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - smoke script should print one direct failure
        print(f"FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
