# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OpenShell provider tests: creation/attestation plus confined file transfer."""

from __future__ import annotations

import base64
import importlib.util
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from aiq_agent.agents.deep_researcher.sandbox.base import SandboxProvider
from aiq_agent.agents.deep_researcher.sandbox.config import SandboxConfig
from aiq_agent.agents.deep_researcher.sandbox.providers.openshell import OpenShellSandboxProvider
from aiq_agent.agents.deep_researcher.sandbox.providers.openshell import _build_sandbox_spec
from aiq_agent.agents.deep_researcher.sandbox.providers.openshell import _parse_policy_proto
from aiq_agent.agents.deep_researcher.sandbox.providers.openshell import _read_policy_data
from aiq_agent.agents.deep_researcher.sandbox.providers.openshell import _validate_policy_network


@dataclass
class _ExecResult:
    exit_code: int
    stdout: str = ""
    stderr: str = ""


class _FakeOpenShellSandbox:
    """Records exec calls and returns scripted results."""

    id = "fake-os-id"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.result = _ExecResult(exit_code=0)
        self.exit_calls = 0

    def exec(self, command, **kwargs):  # noqa: ANN001 - mirrors openshell.Sandbox.exec
        self.calls.append({"command": list(command), **kwargs})
        return self.result

    def __exit__(self, *_args: object) -> None:
        self.exit_calls += 1


def _provider(**openshell_config: Any) -> OpenShellSandboxProvider:
    cfg = SandboxConfig(
        provider="openshell",
        network={"mode": "blocked"},
        providers={"openshell": openshell_config},
    )
    # Construct through the provider-neutral base so these unit tests run even when
    # the optional OpenShell SDK/adapter are absent from normal CI.
    provider = object.__new__(OpenShellSandboxProvider)
    SandboxProvider.__init__(provider, cfg, "job-1")
    provider._os_context = None
    # Avoid real session creation: a non-None _session short-circuits _session_or_create.
    provider._session = MagicMock()  # type: ignore[assignment]
    return provider


class _FakeCreatedContext(_FakeOpenShellSandbox):
    """Entered SDK context with a public sandbox reference for attestation."""

    def __init__(self, *, phase: int = 2, policy_version: int = 1, name: str = "generated") -> None:
        super().__init__()
        self.enter_calls = 0
        self.sandbox = SimpleNamespace(
            phase=phase,
            current_policy_version=policy_version,
            name=name,
        )

    def __enter__(self):
        self.enter_calls += 1
        return self


class _FakeAdapter:
    """Minimal langchain-nvidia-openshell adapter used by provider lifecycle tests."""

    def __init__(self, *, sandbox: _FakeCreatedContext, timeout: int, shell: tuple[str, ...]) -> None:
        self.id = sandbox.id
        self.sandbox = sandbox
        self.timeout = timeout
        self.shell = shell


@contextmanager
def _fake_optional_modules(context_factory, adapter_cls: type = _FakeAdapter):
    openshell_module = ModuleType("openshell")
    openshell_module.Sandbox = context_factory  # type: ignore[attr-defined]
    adapter_module = ModuleType("langchain_nvidia_openshell")
    adapter_module.OpenShellSandbox = adapter_cls  # type: ignore[attr-defined]
    proto_module = ModuleType("openshell._proto")
    proto_module.openshell_pb2 = SimpleNamespace(SANDBOX_PHASE_READY=2)  # type: ignore[attr-defined]
    with patch.dict(
        sys.modules,
        {
            "openshell": openshell_module,
            "openshell._proto": proto_module,
            "langchain_nvidia_openshell": adapter_module,
        },
    ):
        yield


def _write_policy(path: Path, *, compatibility: str = "hard_requirement", extra: str = "") -> None:
    path.write_text(
        "\n".join(
            [
                "version: 1",
                "filesystem_policy:",
                "  include_workdir: true",
                "  read_write: [/sandbox]",
                "landlock:",
                f"  compatibility: {compatibility}",
                "process:",
                "  run_as_user: sandbox",
                "  run_as_group: sandbox",
                "network_policies: {}",
                extra,
            ]
        ),
        encoding="utf-8",
    )


def test_policy_reader_normalizes_filesystem_alias_and_requires_hard_landlock(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.yaml"
    _write_policy(policy_path)

    policy = _read_policy_data(str(policy_path), require_hard_landlock=True)

    assert "filesystem" in policy
    assert "filesystem_policy" not in policy
    assert policy["landlock"]["compatibility"] == "hard_requirement"


def test_policy_reader_rejects_best_effort_in_production(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.yaml"
    _write_policy(policy_path, compatibility="best_effort")

    with pytest.raises(ValueError, match="hard_requirement"):
        _read_policy_data(str(policy_path), require_hard_landlock=True)


def test_policy_reader_rejects_missing_file_and_wrong_version(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Could not read"):
        _read_policy_data(str(tmp_path / "missing.yaml"), require_hard_landlock=True)

    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text("version: 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly 1"):
        _read_policy_data(str(policy_path), require_hard_landlock=True)


@pytest.mark.parametrize(
    ("policy", "message"),
    [
        (
            "version: 1\nlandlock: {compatibility: hard_requirement}\n"
            "process: {run_as_user: sandbox, run_as_group: sandbox}\n",
            "filesystem",
        ),
        (
            "version: 1\nfilesystem_policy: {read_write: [/sandbox]}\n"
            "landlock: {compatibility: hard_requirement}\n"
            "process: {run_as_user: root, run_as_group: sandbox}\n",
            "non-root",
        ),
        (
            "version: 1\nfilesystem_policy: {read_write: [/sandbox]}\n"
            "landlock: {compatibility: hard_requirement}\n"
            "process: {run_as_user: sandbox, run_as_group: sandbox}\n"
            "network_policies: {bad: {endpoints: [{host: example.com, enforcement: audit, access: read-write}]}}\n",
            "enforcement=enforce",
        ),
    ],
)
def test_policy_reader_enforces_production_floor(tmp_path: Path, policy: str, message: str) -> None:
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(policy, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _read_policy_data(str(policy_path), require_hard_landlock=True)


def test_checked_policy_matches_installed_sdk_schema() -> None:
    if importlib.util.find_spec("openshell") is None:
        pytest.skip("optional OpenShell SDK is not installed")

    policy_path = "configs/openshell/aiq-research-policy.yaml"
    policy_data = _read_policy_data(policy_path, require_hard_landlock=True)
    policy = _parse_policy_proto(policy_data, policy_path=policy_path)

    assert policy.version == 1

    spec = _build_sandbox_spec(policy=policy, image="aiq:test", job_id="job-1")
    assert spec.template.image == "aiq:test"
    assert dict(spec.template.labels) == {"aiq": "deep-research", "aiq-job-id": "job-1"}
    assert not spec.template.environment
    assert not spec.environment
    assert not spec.providers


def test_sandbox_spec_normalizes_long_job_id_label() -> None:
    if importlib.util.find_spec("openshell") is None:
        pytest.skip("optional OpenShell SDK is not installed")
    from openshell._proto import sandbox_pb2

    spec = _build_sandbox_spec(
        policy=sandbox_pb2.SandboxPolicy(version=1),
        image="aiq:test",
        job_id="JOB_WITH_UNSAFE_CHARS_" * 8,
    )
    job_label = spec.template.labels["aiq-job-id"]
    assert len(job_label) <= 63
    assert job_label == job_label.lower()
    assert "_" not in job_label


def test_policy_network_must_not_exceed_declared_allowlist() -> None:
    policy = {"network_policies": {"github": {"endpoints": [{"host": "api.github.com"}, {"host": "github.com"}]}}}

    _validate_policy_network(
        policy,
        mode="allowlist",
        allow=("api.github.com", "github.com"),
    )
    with pytest.raises(ValueError, match="github.com"):
        _validate_policy_network(policy, mode="allowlist", allow=("api.github.com",))
    with pytest.raises(ValueError, match="network endpoints"):
        _validate_policy_network(policy, mode="blocked", allow=())


def test_per_job_session_uses_policy_spec_and_attests_before_return(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.yaml"
    _write_policy(policy_path)
    created: list[tuple[_FakeCreatedContext, dict[str, Any]]] = []

    def context_factory(**kwargs: Any) -> _FakeCreatedContext:
        context = _FakeCreatedContext()
        created.append((context, kwargs))
        return context

    with (
        _fake_optional_modules(context_factory),
        patch(
            "aiq_agent.agents.deep_researcher.sandbox.providers.openshell._parse_policy_proto",
            return_value="policy-proto",
        ),
        patch(
            "aiq_agent.agents.deep_researcher.sandbox.providers.openshell._build_sandbox_spec",
            return_value="job-spec",
        ),
    ):
        provider = OpenShellSandboxProvider(
            SandboxConfig(
                provider="openshell",
                providers={"openshell": {"policy": str(policy_path), "image": "aiq:test"}},
            ),
            "job-123",
        )
        backend = provider._create_session()

    context, kwargs = created[0]
    assert backend.id == context.id
    assert context.enter_calls == 1
    assert kwargs["spec"] == "job-spec"
    assert kwargs["delete_on_exit"] is True
    assert "sandbox" not in kwargs
    assert provider.physical_sandbox_name == "generated"


def test_two_jobs_create_distinct_specs_without_named_attachment(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.yaml"
    _write_policy(policy_path)
    created: list[dict[str, Any]] = []
    spec_jobs: list[str] = []

    def context_factory(**kwargs: Any) -> _FakeCreatedContext:
        created.append(kwargs)
        return _FakeCreatedContext(name=f"generated-{len(created)}")

    def build_spec(*, policy: Any, image: str, job_id: str) -> str:
        del policy, image
        spec_jobs.append(job_id)
        return f"spec-{job_id}"

    with (
        _fake_optional_modules(context_factory),
        patch(
            "aiq_agent.agents.deep_researcher.sandbox.providers.openshell._parse_policy_proto",
            return_value="policy-proto",
        ),
        patch(
            "aiq_agent.agents.deep_researcher.sandbox.providers.openshell._build_sandbox_spec",
            side_effect=build_spec,
        ),
    ):
        for job_id in ("job-a", "job-b"):
            provider = OpenShellSandboxProvider(
                SandboxConfig(
                    provider="openshell",
                    providers={"openshell": {"policy": str(policy_path), "image": "aiq:test"}},
                ),
                job_id,
            )
            provider._create_session()

    assert spec_jobs == ["job-a", "job-b"]
    assert [kwargs["spec"] for kwargs in created] == ["spec-job-a", "spec-job-b"]
    assert all("sandbox" not in kwargs for kwargs in created)


@pytest.mark.parametrize(
    ("phase", "policy_version", "expected", "message"),
    [
        (3, 1, None, "not READY"),
        (2, 0, None, "no loaded policy"),
        (2, 2, 1, "expected 1"),
    ],
)
def test_attestation_failure_deletes_before_raising(
    tmp_path: Path,
    phase: int,
    policy_version: int,
    expected: int | None,
    message: str,
) -> None:
    policy_path = tmp_path / "policy.yaml"
    _write_policy(policy_path)
    context = _FakeCreatedContext(phase=phase, policy_version=policy_version)

    with (
        _fake_optional_modules(lambda **_kwargs: context),
        patch(
            "aiq_agent.agents.deep_researcher.sandbox.providers.openshell._parse_policy_proto",
            return_value="policy-proto",
        ),
        patch(
            "aiq_agent.agents.deep_researcher.sandbox.providers.openshell._build_sandbox_spec",
            return_value="job-spec",
        ),
    ):
        provider = OpenShellSandboxProvider(
            SandboxConfig(
                provider="openshell",
                providers={
                    "openshell": {
                        "policy": str(policy_path),
                        "expected_policy_version": expected,
                    }
                },
            ),
            "job-123",
        )
        with pytest.raises(RuntimeError, match=message):
            provider._create_session()

    assert context.exit_calls == 1
    assert provider._os_context is None


def test_shared_attachment_requires_explicit_debug_opt_in() -> None:
    with pytest.raises(ValueError, match="allow_shared_sandbox"):
        SandboxConfig(provider="openshell", providers={"openshell": {"existing_sandbox_name": "shared"}})

    config = SandboxConfig(
        provider="openshell",
        providers={
            "openshell": {
                "existing_sandbox_name": "shared",
                "allow_shared_sandbox": True,
                "delete_on_exit": False,
            }
        },
    )
    assert config.providers.openshell.shared_sandbox_name == "shared"


def test_per_job_mode_rejects_disabled_attestation() -> None:
    with pytest.raises(ValueError, match="attest=true"):
        SandboxConfig(
            provider="openshell",
            providers={"openshell": {"policy": "policy.yaml", "attest": False}},
        )


def test_adapter_construction_failure_deletes_sandbox(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.yaml"
    _write_policy(policy_path)
    context = _FakeCreatedContext()

    class FailingAdapter:
        def __init__(self, **_kwargs: Any) -> None:
            raise ValueError("bad adapter configuration")

    with (
        _fake_optional_modules(lambda **_kwargs: context, FailingAdapter),
        patch(
            "aiq_agent.agents.deep_researcher.sandbox.providers.openshell._parse_policy_proto",
            return_value="policy-proto",
        ),
        patch(
            "aiq_agent.agents.deep_researcher.sandbox.providers.openshell._build_sandbox_spec",
            return_value="job-spec",
        ),
    ):
        provider = OpenShellSandboxProvider(
            SandboxConfig(provider="openshell", providers={"openshell": {"policy": str(policy_path)}}),
            "job-123",
        )
        with pytest.raises(ValueError, match="bad adapter"):
            provider._create_session()

    assert context.exit_calls == 1
    assert provider._os_context is None


def test_upload_passes_path_via_argv_and_data_via_stdin_no_env() -> None:
    provider = _provider()
    fake = _FakeOpenShellSandbox()
    provider._os_context = fake

    result = provider.upload_files([("/sandbox/x.py", b"print('hi')")])

    assert result[0].error is None
    call = fake.calls[0]
    assert call["command"][0] == "python3" and call["command"][-1] == "/sandbox/x.py"
    assert call["stdin"] == base64.b64encode(b"print('hi')")
    assert "env" not in call  # the whole point: never rely on exec env propagation


def test_upload_classifies_failure() -> None:
    provider = _provider()
    fake = _FakeOpenShellSandbox()
    fake.result = _ExecResult(exit_code=1, stderr="No such file or directory")
    provider._os_context = fake

    result = provider.upload_files([("/sandbox/x.py", b"data")])
    assert result[0].error == "file_not_found"


def test_upload_rejects_relative_path() -> None:
    provider = _provider()
    provider._os_context = _FakeOpenShellSandbox()
    result = provider.upload_files([("relative.py", b"data")])
    assert result[0].error == "invalid_path"


def test_download_passes_path_via_argv_and_decodes_base64() -> None:
    provider = _provider()
    fake = _FakeOpenShellSandbox()
    fake.result = _ExecResult(exit_code=0, stdout=base64.b64encode(b"chart-bytes").decode())
    provider._os_context = fake

    artifact_path = f"{provider.artifact_dir}/chart.png"
    result = provider.download_files([artifact_path])

    assert result[0].error is None
    assert result[0].content == b"chart-bytes"
    call = fake.calls[0]
    # Path is passed positionally via argv (with the size cap appended); never via env.
    assert artifact_path in call["command"]
    assert call["command"][-1] == provider.artifact_dir
    assert "env" not in call


def test_download_uses_confined_shim_when_adapter_transfer_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _provider()
    fake = _FakeOpenShellSandbox()
    fake.result = _ExecResult(exit_code=0, stdout=base64.b64encode(b"chart-bytes").decode())
    provider._os_context = fake
    provider._session.download_files.side_effect = AssertionError("adapter download bypassed confinement")  # type: ignore[union-attr]
    monkeypatch.setenv("AIQ_OPENSHELL_ADAPTER_FILE_TRANSFER", "1")

    result = provider.download_files([f"{provider.artifact_dir}/chart.png"])

    assert result[0].content == b"chart-bytes"
    assert fake.calls[0]["command"][-1] == provider.artifact_dir


def test_download_is_directory_exit_code() -> None:
    provider = _provider()
    fake = _FakeOpenShellSandbox()
    fake.result = _ExecResult(exit_code=3, stderr="")
    provider._os_context = fake

    result = provider.download_files(["/sandbox"])
    assert result[0].content is None
    assert result[0].error == "is_directory"  # _DOWNLOAD_CODE exits 3 specifically for a directory


def test_download_passes_size_cap_via_argv() -> None:
    provider = _provider()
    fake = _FakeOpenShellSandbox()
    fake.result = _ExecResult(exit_code=0, stdout=base64.b64encode(b"x").decode())
    provider._os_context = fake

    provider.download_files(["/sandbox/aiq-artifacts/chart.png"])

    # The artifact size cap is passed to the bootstrap so it can refuse oversized files
    # before reading them into host memory.
    assert str(provider.config.artifact_capture.max_file_bytes) in fake.calls[0]["command"]


def test_download_rejects_oversized_and_symlink() -> None:
    provider = _provider()
    fake = _FakeOpenShellSandbox()
    provider._os_context = fake

    fake.result = _ExecResult(exit_code=4, stderr="")
    assert provider.download_files(["/sandbox/aiq-artifacts/huge.bin"])[0].error == "too_large"

    fake.result = _ExecResult(exit_code=5, stderr="")
    assert provider.download_files(["/sandbox/aiq-artifacts/evil.png"])[0].error == "symlink_rejected"


def test_download_rejects_non_base64_stdout() -> None:
    provider = _provider()
    fake = _FakeOpenShellSandbox()
    fake.result = _ExecResult(exit_code=0, stdout="not valid base64 !!!")
    provider._os_context = fake

    result = provider.download_files(["/sandbox/aiq-artifacts/chart.png"])
    assert result[0].content is None
    assert result[0].error == "invalid_content"


def test_terminate_exits_openshell_context_once() -> None:
    provider = _provider()
    fake = _FakeOpenShellSandbox()
    provider._os_context = fake

    provider.terminate()
    provider.terminate()

    assert fake.exit_calls == 1
    assert provider._os_context is None


def test_context_exit_failure_marks_cleanup_failed() -> None:
    provider = _provider()

    class FailingContext:
        def __exit__(self, *_args: object) -> None:
            raise RuntimeError("gateway delete failed")

    provider._os_context = FailingContext()

    provider.close()

    assert provider.cleanup_succeeded is False
    assert provider._os_context is None
