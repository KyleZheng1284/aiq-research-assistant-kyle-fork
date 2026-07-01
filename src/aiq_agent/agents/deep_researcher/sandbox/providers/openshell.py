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

"""OpenShell sandbox provider (enterprise/on-prem).

Governed, policy-enforced execution on local Docker/Podman/Kubernetes/microVM via
the OpenShell gateway. The deepagents ``BaseSandbox`` adapter is the official
``langchain-nvidia-openshell`` partner package (``OpenShellSandbox``), the same
adapter AI-Q PR #274 integrates. Both the ``openshell`` SDK and the adapter are
intentionally NOT declared in ``pyproject``; they are optional, ad-hoc
dependencies imported lazily, so this provider is never force-installed.

Until ``langchain-ai/langchain-nvidia`` PR #303 publishes the adapter to PyPI,
install it from a git spec (see ``scripts/setup_openshell.sh`` /
``LANGCHAIN_NVIDIA_REPO``).
"""

from __future__ import annotations

import base64
import logging
import os
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

from deepagents.backends.protocol import FileDownloadResponse
from deepagents.backends.protocol import FileUploadResponse
from deepagents.backends.sandbox import BaseSandbox

from ..base import SandboxProvider
from ..base import SandboxTerminatedError
from ..capabilities import SandboxCapabilities
from ..registry import register_sandbox_provider

if TYPE_CHECKING:
    from ..config import SandboxConfig

logger = logging.getLogger(__name__)

# Migration switch: when set truthy, delegate file transfer to the official adapter's
# upload_files/download_files instead of the local env-free shim. Use this to validate the
# upstream argv fix (langchain-ai/langchain-nvidia PR #303); once that ships, the shim and
# this switch can be removed and the adapter used unconditionally.
_ADAPTER_FILE_TRANSFER_ENV = "AIQ_OPENSHELL_ADAPTER_FILE_TRANSFER"


def _adapter_file_transfer_enabled() -> bool:
    """True only when the toggle env var is an explicit truthy value (not just any string)."""
    return os.getenv(_ADAPTER_FILE_TRANSFER_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


# File-transfer bootstraps pass the path via argv (not env): OpenShell <=0.0.67 strips
# OPENSHELL_* env before exec, breaking the adapter's env-based transfer. We keep the
# adapter for execute and override only these two methods until the SDK propagates env.
# The download bootstrap fails closed before reading untrusted bytes: reject a symlink
# leaf (exit 5) or directory (exit 3), and read at most cap+1 bytes (exit 4 if over) so
# an oversized/out-of-tree file is never pulled into host memory.
_UPLOAD_CODE = (
    "import base64,os,sys;"
    "p=sys.argv[1];"
    "d=os.path.dirname(p);"
    "(os.makedirs(d,exist_ok=True) if d else None);"
    "open(p,'wb').write(base64.b64decode(sys.stdin.buffer.read()))"
)
_DOWNLOAD_CODE = (
    "import base64,os,sys;"
    "p=sys.argv[1];"
    "limit=int(sys.argv[2]);"
    "root=os.path.realpath(sys.argv[3]);"
    "rp=os.path.realpath(p);"
    "(sys.exit(5) if not (rp==root or rp.startswith(root+os.sep)) else None);"
    "(sys.exit(3) if os.path.isdir(rp) else None);"
    "b=open(rp,'rb').read(limit+1);"
    "(sys.exit(4) if len(b)>limit else None);"
    "sys.stdout.write(base64.b64encode(b).decode())"
)

# Bootstrap exit codes mapped to a download error reason (see _DOWNLOAD_CODE).
_DOWNLOAD_EXIT_ERRORS = {3: "is_directory", 4: "too_large", 5: "symlink_rejected"}


def _classify_fs_error(text: str) -> str:
    """Map sandbox-side stderr to a deepagents FileOperationError literal."""
    lowered = text.lower()
    if "no such file" in lowered or "file not found" in lowered or "filenotfounderror" in lowered:
        return "file_not_found"
    if "is a directory" in lowered or "isadirectoryerror" in lowered:
        return "is_directory"
    if "invalid" in lowered and "path" in lowered:
        return "invalid_path"
    return "permission_denied"


_OPENSHELL_IMPORT_HINT = (
    "The OpenShell sandbox provider requires the `openshell>=0.0.72,<0.1` SDK and the "
    "`langchain-nvidia-openshell` adapter (published on PyPI). They are optional, ad-hoc "
    "dependencies. Install them with `./scripts/setup_openshell.sh` (which installs "
    "`langchain-nvidia-openshell` from PyPI; override the source via `LANGCHAIN_NVIDIA_REPO`), "
    "and configure an OpenShell gateway before enabling this provider."
)


def _normalize_openshell_name(job_id: str, prefix: str = "aiq-deep-research") -> str:
    """Normalize a job id into a DNS-style, length-bounded OpenShell sandbox name."""
    raw = f"{prefix}-{job_id}" if prefix else job_id
    normalized = re.sub(r"[^a-z0-9-]+", "-", raw.lower())
    normalized = re.sub(r"-+", "-", normalized).strip("-")
    return (normalized[:63].rstrip("-")) or prefix


def _is_openshell_not_found_error(exc: Exception) -> bool:
    """Best-effort classification of OpenShell stale-sandbox errors."""
    text = str(exc).lower()
    return "not found" in text and ("sandbox" in text or exc.__class__.__module__.startswith("openshell"))


def _read_policy_data(policy_path: str, *, require_hard_landlock: bool) -> dict[str, Any]:
    """Read and normalize OpenShell policy YAML without importing the optional SDK."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(_OPENSHELL_IMPORT_HINT) from exc

    path = Path(policy_path).expanduser()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Could not read OpenShell policy file: {path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid OpenShell policy YAML: {path}") from exc

    if not isinstance(raw, dict):
        raise ValueError(f"OpenShell policy must be a YAML mapping: {path}")
    if raw.get("version") != 1:
        raise ValueError("OpenShell policy version must be exactly 1")

    landlock = raw.get("landlock")
    compatibility = landlock.get("compatibility") if isinstance(landlock, dict) else None
    if require_hard_landlock and compatibility != "hard_requirement":
        raise ValueError(
            "OpenShell production policy requires landlock.compatibility=hard_requirement; "
            "set require_hard_landlock=false only for an explicit local demo."
        )

    filesystem = raw.get("filesystem_policy", raw.get("filesystem"))
    if not isinstance(filesystem, dict) or not any(filesystem.get(key) for key in ("read_only", "read_write")):
        raise ValueError("OpenShell production policy requires non-empty filesystem read_only or read_write rules")

    process = raw.get("process")
    if not isinstance(process, dict):
        raise ValueError("OpenShell production policy requires a process policy")
    for field in ("run_as_user", "run_as_group"):
        identity = process.get(field)
        if not isinstance(identity, str) or not identity.strip() or identity.strip().lower() in {"0", "root"}:
            raise ValueError(f"OpenShell production policy requires a non-root process.{field}")

    network_policies = raw.get("network_policies") or {}
    if not isinstance(network_policies, dict):
        raise ValueError("OpenShell network_policies must be a mapping")
    for policy_name, network_policy in network_policies.items():
        if not isinstance(network_policy, dict):
            raise ValueError(f"OpenShell network policy {policy_name!r} must be a mapping")
        endpoints = network_policy.get("endpoints") or []
        if not isinstance(endpoints, list):
            raise ValueError(f"OpenShell network policy {policy_name!r} endpoints must be a list")
        for endpoint in endpoints:
            if not isinstance(endpoint, dict):
                raise ValueError(f"OpenShell network policy {policy_name!r} contains an invalid endpoint")
            if endpoint.get("enforcement") != "enforce" or endpoint.get("access") != "read-only":
                raise ValueError(
                    f"OpenShell network policy {policy_name!r} endpoints must use "
                    "enforcement=enforce and access=read-only"
                )

    # OpenShell's YAML schema calls this field filesystem_policy while the 0.0.72
    # Python proto calls it filesystem. Keep this compatibility translation in one
    # place and reject every other unknown field through ParseDict below.
    policy_data = dict(raw)
    if "filesystem_policy" in policy_data:
        if "filesystem" in policy_data:
            raise ValueError("OpenShell policy cannot contain both filesystem_policy and filesystem")
        policy_data["filesystem"] = policy_data.pop("filesystem_policy")
    return policy_data


def _parse_policy_proto(policy_data: dict[str, Any], *, policy_path: str) -> Any:
    """Parse one validated policy snapshot into the SDK proto with strict field validation."""
    try:
        from google.protobuf.json_format import ParseDict
        from openshell._proto import sandbox_pb2
    except ImportError as exc:
        raise ImportError(_OPENSHELL_IMPORT_HINT) from exc

    try:
        return ParseDict(policy_data, sandbox_pb2.SandboxPolicy(), ignore_unknown_fields=False)
    except Exception as exc:  # noqa: BLE001 - protobuf raises several parse exception types
        raise ValueError(f"OpenShell policy does not match the installed SDK schema: {policy_path}") from exc


def _policy_network_hosts(policy_data: dict[str, Any]) -> set[str]:
    """Return every hostname authorized by an OpenShell network policy."""
    policies = policy_data.get("network_policies") or {}
    if not isinstance(policies, dict):
        return set()
    hosts: set[str] = set()
    for policy in policies.values():
        if not isinstance(policy, dict):
            continue
        endpoints = policy.get("endpoints") or []
        if not isinstance(endpoints, list):
            continue
        for endpoint in endpoints:
            if isinstance(endpoint, dict) and isinstance(endpoint.get("host"), str):
                hosts.add(endpoint["host"].lower().rstrip("."))
    return hosts


def _validate_policy_network(policy_data: dict[str, Any], *, mode: str, allow: tuple[str, ...]) -> None:
    """Fail closed when the policy grants more egress than the public config declares."""
    policy_hosts = _policy_network_hosts(policy_data)
    if mode == "blocked" and policy_hosts:
        raise ValueError(
            f"OpenShell policy grants network endpoints while sandbox.network is 'blocked': {sorted(policy_hosts)}"
        )
    if mode == "allowlist":
        configured_hosts = {host.lower().rstrip(".") for host in allow}
        unexpected = policy_hosts - configured_hosts
        if unexpected:
            raise ValueError(f"OpenShell policy grants hosts outside sandbox.network.allow: {sorted(unexpected)}")


def _build_sandbox_spec(*, policy: Any, image: str, job_id: str) -> Any:
    """Build a secret-free per-job OpenShell spec using the installed SDK schema."""
    try:
        from openshell._proto import openshell_pb2
    except ImportError as exc:
        raise ImportError(_OPENSHELL_IMPORT_HINT) from exc

    template = openshell_pb2.SandboxTemplate(
        image=image,
        labels={"aiq": "deep-research", "aiq-job-id": _normalize_openshell_name(job_id, prefix="")},
    )
    # Deliberately omit environment and providers. Research/model credentials stay
    # on the host unless a future explicit credential-provider feature is configured.
    return openshell_pb2.SandboxSpec(template=template, policy=policy)


class OpenShellSandboxProvider(SandboxProvider):
    """OpenShell backend that creates and attests a policy-bound sandbox per job."""

    provider_name = "openshell"

    def __init__(self, config: SandboxConfig, job_id: str) -> None:
        """Initialize the provider, requiring the OpenShell SDK and adapter to import."""
        super().__init__(config, job_id)
        self._os_context: object | None = None
        self._os_context_entering = False
        self._os_context_exit_requested = False
        try:
            import langchain_nvidia_openshell  # noqa: F401
            import openshell  # noqa: F401
        except ImportError as exc:
            raise ImportError(_OPENSHELL_IMPORT_HINT) from exc

    @classmethod
    def _scoped_name(cls, job_id: str) -> str:
        """Return the OpenShell-safe sandbox name derived from the job id."""
        return _normalize_openshell_name(job_id)

    @property
    def capabilities(self) -> SandboxCapabilities:
        """Declare the gateway-enforced guarantees this provider supports."""
        return SandboxCapabilities(
            supports_network_policy=True,
            supports_network_allowlist=True,
            supports_filesystem_policy=True,
            supports_process_policy=True,
            supports_artifact_download=True,
            supports_cleanup=True,
            supports_terminate=True,
        )

    def is_recoverable_error(self, exc: Exception) -> bool:
        """Return whether the error is a missing-sandbox condition worth one retry."""
        return _is_openshell_not_found_error(exc)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload files. Uses the local env-free shim by default (OpenShell <=0.0.67 strips
        ``OPENSHELL_*`` env); set ``AIQ_OPENSHELL_ADAPTER_FILE_TRANSFER`` to delegate to the
        official adapter (validates the upstream argv fix)."""
        if _adapter_file_transfer_enabled():
            return self._call("upload_files", lambda session: session.upload_files(files), idempotent=True)
        return self._call("upload_files", lambda _s: self._upload_files_envfree(files), idempotent=True)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download artifacts through the bounded, job-confined local shim."""
        return self._call("download_files", lambda _s: self._download_files_envfree(paths), idempotent=True)

    def _upload_files_envfree(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload files via argv + stdin so no ``OPENSHELL_*`` env is required."""
        sandbox = self._active_os_context()
        responses: list[FileUploadResponse] = []
        for path, content in files:
            if not path.startswith("/"):
                responses.append(FileUploadResponse(path=path, error="invalid_path"))
                continue
            result = sandbox.exec(  # type: ignore[union-attr]
                ["python3", "-c", _UPLOAD_CODE, path],
                stdin=base64.b64encode(content),
                timeout_seconds=self.config.timeout,
            )
            exit_code = getattr(result, "exit_code", 1)
            error = None if exit_code == 0 else _classify_fs_error(getattr(result, "stderr", "") or "")
            responses.append(FileUploadResponse(path=path, error=error))
        return responses

    def _download_files_envfree(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files via an argv bootstrap that enforces size/symlink limits in-sandbox."""
        sandbox = self._active_os_context()
        # Cap passed to the bootstrap so oversized files are refused before transfer.
        max_bytes = self.config.artifact_capture.max_file_bytes
        responses: list[FileDownloadResponse] = []
        for path in paths:
            if not path.startswith("/"):
                responses.append(FileDownloadResponse(path=path, content=None, error="invalid_path"))
                continue
            result = sandbox.exec(  # type: ignore[union-attr]
                # Confine resolved paths to this job's artifact directory. The configured
                # workdir may be shared by several jobs in an attached named sandbox.
                ["python3", "-c", _DOWNLOAD_CODE, path, str(max_bytes), self.artifact_dir],
                timeout_seconds=self.config.timeout,
            )
            exit_code = getattr(result, "exit_code", 1)
            if exit_code != 0:
                error = _DOWNLOAD_EXIT_ERRORS.get(exit_code) or _classify_fs_error(getattr(result, "stderr", "") or "")
                responses.append(FileDownloadResponse(path=path, content=None, error=error))
                continue
            # Validate base64 so stray stdout fails closed rather than storing corrupt bytes.
            try:
                content = base64.b64decode((getattr(result, "stdout", "") or "").strip().encode("ascii"), validate=True)
            except ValueError:
                responses.append(FileDownloadResponse(path=path, content=None, error="invalid_content"))
                continue
            responses.append(FileDownloadResponse(path=path, content=content, error=None))
        return responses

    def _create_session(self) -> BaseSandbox:
        """Create and attest a per-job OpenShell sandbox, or explicitly attach for debug."""
        try:
            import openshell
            from langchain_nvidia_openshell import OpenShellSandbox
        except ImportError as exc:
            raise ImportError(_OPENSHELL_IMPORT_HINT) from exc

        cfg = self.config
        oscfg = cfg.providers.openshell

        if not oscfg.shell or any(not part.strip() for part in oscfg.shell):
            raise ValueError("OpenShell shell must contain at least one non-empty argv element")

        # Release any prior context (covers the recoverable-error reset path).
        self._exit_context()

        sandbox_kwargs: dict[str, object] = {
            "cluster": oscfg.gateway,
            "ready_timeout_seconds": oscfg.ready_timeout_seconds,
        }
        shared_name = oscfg.shared_sandbox_name
        if shared_name is not None:
            logger.warning(
                "OpenShell shared-sandbox debug attachment enabled: sandbox=%s job=%s; physical job isolation is off",
                shared_name,
                self.job_id,
            )
            # Attachment does not transfer ownership: never delete a shared sandbox
            # that this job did not create.
            sandbox_kwargs.update(sandbox=shared_name, delete_on_exit=False)
        else:
            if not oscfg.policy:
                raise ValueError("Per-job OpenShell creation requires a policy file")
            policy_data = _read_policy_data(oscfg.policy, require_hard_landlock=oscfg.require_hard_landlock)
            _validate_policy_network(
                policy_data,
                mode=cfg.network.mode,
                allow=cfg.network.allow,
            )
            policy = _parse_policy_proto(policy_data, policy_path=oscfg.policy)
            sandbox_kwargs.update(
                spec=_build_sandbox_spec(policy=policy, image=oscfg.image, job_id=self.job_id),
                delete_on_exit=oscfg.delete_on_exit,
            )

        os_sandbox = openshell.Sandbox(**sandbox_kwargs)
        self._enter_context(os_sandbox)
        backend: BaseSandbox | None = None
        try:
            self._ensure_context_active(os_sandbox)
            self.physical_sandbox_name = getattr(os_sandbox.sandbox, "name", None)
            phase, policy_version = self._attest(os_sandbox)
            self._ensure_context_active(os_sandbox)
            backend = OpenShellSandbox(sandbox=os_sandbox, timeout=cfg.timeout, shell=oscfg.shell)
            self._ensure_context_active(os_sandbox)
            sandbox_ref = os_sandbox.sandbox
            logger.info(
                "OpenShell sandbox attested: id=%s name=%s policy_version=%s shared=%s",
                backend.id,
                getattr(sandbox_ref, "name", None),
                policy_version,
                shared_name is not None,
            )
            self._emit_attestation(phase=phase, policy_version=policy_version, status="succeeded")
            return backend
        except BaseException:
            self._safe_close(backend)
            self._exit_context()
            raise

    def _attest(self, os_sandbox: Any) -> tuple[object, object]:
        """Fail closed unless the entered sandbox is READY with the expected policy revision."""
        try:
            from openshell._proto import openshell_pb2
        except ImportError as exc:
            raise ImportError(_OPENSHELL_IMPORT_HINT) from exc

        sandbox_ref = os_sandbox.sandbox
        phase = getattr(sandbox_ref, "phase", None)
        policy_version = getattr(sandbox_ref, "current_policy_version", 0)
        if phase != openshell_pb2.SANDBOX_PHASE_READY:
            self._emit_attestation(phase=phase, policy_version=policy_version, status="failed")
            raise RuntimeError(f"OpenShell sandbox attestation failed: phase={phase!r} is not READY")

        oscfg = self.config.providers.openshell
        if oscfg.attest and (not isinstance(policy_version, int) or policy_version <= 0):
            phase, policy_version = self._wait_for_loaded_policy(os_sandbox, sandbox_ref)
            if phase != openshell_pb2.SANDBOX_PHASE_READY:
                self._emit_attestation(phase=phase, policy_version=policy_version, status="failed")
                raise RuntimeError(f"OpenShell sandbox attestation failed: phase={phase!r} is not READY")
        if oscfg.attest and (not isinstance(policy_version, int) or policy_version <= 0):
            self._emit_attestation(phase=phase, policy_version=policy_version, status="failed")
            raise RuntimeError("OpenShell sandbox attestation failed: no loaded policy revision")
        if oscfg.expected_policy_version is not None and policy_version != oscfg.expected_policy_version:
            self._emit_attestation(phase=phase, policy_version=policy_version, status="failed")
            raise RuntimeError(
                "OpenShell sandbox attestation failed: "
                f"policy revision {policy_version!r} != expected {oscfg.expected_policy_version}"
            )
        return phase, policy_version

    def _wait_for_loaded_policy(self, os_sandbox: Any, sandbox_ref: Any) -> tuple[object, object]:
        """Wait for READY plus a loaded policy revision on OpenShell 0.0.72."""
        from openshell._proto import openshell_pb2

        oscfg = self.config.providers.openshell
        phase = getattr(sandbox_ref, "phase", None)
        policy_version = getattr(sandbox_ref, "current_policy_version", 0)
        client = getattr(os_sandbox, "_client", None)
        # OpenShell 0.0.72 does not expose policy status on SandboxClient even though the
        # generated RPC is authoritative; isolate that compatibility access here.
        stub = getattr(client, "_stub", None)
        if client is None or stub is None or not hasattr(stub, "GetSandboxPolicyStatus"):
            return phase, policy_version

        request = openshell_pb2.GetSandboxPolicyStatusRequest(name=sandbox_ref.name, version=0)
        deadline = time.monotonic() + oscfg.ready_timeout_seconds
        while True:
            self._ensure_context_active(os_sandbox)
            sandbox_ref = client.get(sandbox_ref.name)
            phase = getattr(sandbox_ref, "phase", None)
            policy_version = getattr(sandbox_ref, "current_policy_version", 0)
            if phase == openshell_pb2.SANDBOX_PHASE_READY and policy_version > 0:
                return phase, policy_version
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return phase, policy_version

            response = stub.GetSandboxPolicyStatus(request, timeout=min(5.0, remaining))
            revision = getattr(response, "revision", None)
            revision_status = getattr(revision, "status", openshell_pb2.POLICY_STATUS_UNSPECIFIED)
            if revision_status == openshell_pb2.POLICY_STATUS_LOADED:
                policy_version = getattr(revision, "version", 0)
                if phase == openshell_pb2.SANDBOX_PHASE_READY and policy_version > 0:
                    return phase, policy_version
            elif revision_status == openshell_pb2.POLICY_STATUS_FAILED:
                return phase, 0
            if phase == openshell_pb2.SANDBOX_PHASE_ERROR:
                return phase, policy_version
            time.sleep(min(0.5, remaining))

    def _emit_attestation(self, *, phase: object, policy_version: object, status: str) -> None:
        """Emit a secret-free OpenShell attestation outcome."""
        self._emit_event(
            {
                "type": "sandbox.attestation",
                "data": {
                    "provider": self.provider_name,
                    "sandbox": getattr(self, "physical_sandbox_name", None) or self.sandbox_name,
                    "phase": phase,
                    "policy_version": policy_version,
                    "status": status,
                },
            }
        )

    def close(self) -> None:
        """Terminate the session and exit the OpenShell context manager."""
        super().close()
        self._exit_context()

    def _terminate_session(self, session: BaseSandbox | None) -> None:
        """Close the adapter session and the owning OpenShell context on cancellation."""
        super()._terminate_session(session)
        self._exit_context()

    def _active_os_context(self) -> Any:
        """Return the current SDK context without racing an out-of-band teardown."""
        with self._state_lock:
            ctx = self._os_context
        if ctx is None:
            raise SandboxTerminatedError(f"OpenShell sandbox {self.sandbox_name} has no active context")
        return ctx

    def _enter_context(self, ctx: object) -> None:
        """Enter an SDK context while allowing terminate() to request deferred cleanup."""
        with self._state_lock:
            if self._terminated:
                raise SandboxTerminatedError(f"Sandbox {self.sandbox_name} has been terminated")
            if self._os_context is not None:
                raise RuntimeError(f"OpenShell sandbox {self.sandbox_name} context creation is already in progress")
            self._os_context = ctx
            self._os_context_entering = True
            self._os_context_exit_requested = False

        try:
            ctx.__enter__()  # type: ignore[attr-defined]
        except BaseException:
            self._finish_context_entry(ctx, entered=False)
            raise

        if not self._finish_context_entry(ctx, entered=True):
            raise SandboxTerminatedError(f"Sandbox {self.sandbox_name} was closed during creation")

    def _finish_context_entry(self, ctx: object, *, entered: bool) -> bool:
        """Publish an entered context, or honor a pending teardown exactly once."""
        with self._state_lock:
            owns_context = self._os_context is ctx
            if owns_context:
                self._os_context_entering = False
                release_context = not entered or self._os_context_exit_requested or self._terminated
                if release_context:
                    self._os_context = None
                    self._os_context_exit_requested = False
            else:
                release_context = entered

        if release_context:
            self._close_os_context(ctx)
        return entered and owns_context and not release_context

    def _ensure_context_active(self, ctx: object) -> None:
        """Abort session publication when teardown won a creation race."""
        with self._state_lock:
            active = self._os_context is ctx and not self._os_context_exit_requested and not self._terminated
        if not active:
            raise SandboxTerminatedError(f"Sandbox {self.sandbox_name} was closed during creation")

    def _exit_context(self) -> None:
        """Exit once, deferring deletion to the creator while ``__enter__`` is in flight."""
        with self._state_lock:
            ctx = self._os_context
            if ctx is None:
                return
            if self._os_context_entering:
                self._os_context_exit_requested = True
                return
            self._os_context = None
            self._os_context_exit_requested = False
        self._close_os_context(ctx)

    def _close_os_context(self, ctx: object) -> None:
        """Drive one detached SDK context exit without replacing the job result."""
        if hasattr(ctx, "__exit__"):
            try:
                ctx.__exit__(None, None, None)
            except Exception:  # noqa: BLE001 - cleanup must never raise on the terminal path
                self._cleanup_failed = True
                logger.warning("OpenShell sandbox %s context cleanup failed", self.sandbox_name, exc_info=True)


register_sandbox_provider("openshell", OpenShellSandboxProvider)
