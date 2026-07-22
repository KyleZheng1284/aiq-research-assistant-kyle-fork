# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider-neutral regressions for quantitative artifact publication.

These tests deliberately use an in-memory sandbox stand-in.  The artifact
contract must not depend on Modal or OpenShell implementation details; the
provider-specific profile test at the bottom verifies only workflow ownership
and checkpoint wiring.
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
import yaml
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool

from aiq_agent.agents.deep_researcher.custom_middleware import ArtifactHarvestMiddleware
from aiq_agent.agents.deep_researcher.custom_middleware import SourceRegistryMiddleware
from aiq_agent.agents.deep_researcher.custom_middleware import WriterExecuteBudgetMiddleware
from aiq_agent.agents.deep_researcher.factory import build_deep_research_middleware_set
from aiq_agent.agents.deep_researcher.factory import build_deep_research_tool_set
from aiq_agent.agents.deep_researcher.sandbox.artifacts import ArtifactManager
from aiq_agent.agents.deep_researcher.sandbox.artifacts import SqlArtifactStore
from aiq_agent.agents.deep_researcher.sandbox.config import ArtifactCaptureConfig

_REPO_ROOT = Path(__file__).parents[4]
_ARTIFACT_DIR = "/sandbox/aiq-artifacts"
_CSV_PATH = f"{_ARTIFACT_DIR}/revenue.csv"
_PNG_PATH = f"{_ARTIFACT_DIR}/revenue.png"
_MANIFEST_PATH = f"{_ARTIFACT_DIR}/manifest.json"
_CSV_BYTES = b"quarter,revenue_usd_billions\r\nFY2025-Q1,22.6\r\nFY2025-Q2,26.3\r\n"
_CSV_DIGEST = sha256(_CSV_BYTES).hexdigest()
_PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
_REPORT = """# Revenue analysis

| quarter | revenue_usd_billions |
| --- | ---: |
| FY2025-Q1 | 22.6 |
| FY2025-Q2 | 26.3 |
"""
_MARKDOWN_TABLE = """| quarter | revenue_usd_billions |
| --- | ---: |
| FY2025-Q1 | 22.6 |
| FY2025-Q2 | 26.3 |"""


class _FakeSandbox:
    """Small sandbox double that records the exact CSV consumed by a chart."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.chart_exit_codes: list[int] = []
        self.chart_calls: list[str] = []
        self.chart_inputs: list[bytes] = []
        self.scan_calls: list[str] = []

    def download_files(self, paths: list[str]) -> list[Any]:
        return [
            SimpleNamespace(path=path, content=self.files.get(path), error=None if path in self.files else "not found")
            for path in paths
        ]

    def execute(self, command: str, *, timeout: int | None = None) -> Any:
        del timeout
        if command.startswith("find "):
            self.scan_calls.append(command)
            return SimpleNamespace(output="\n".join(self.files), exit_code=0, truncated=False)

        self.chart_calls.append(command)
        self.chart_inputs.append(self.files[_CSV_PATH])
        exit_code = self.chart_exit_codes.pop(0) if self.chart_exit_codes else 0
        if exit_code == 0:
            self.files[_PNG_PATH] = _PNG_BYTES
            self.files[_MANIFEST_PATH] = _manifest_bytes(include_png=True)
        return SimpleNamespace(output="chart output", exit_code=exit_code, truncated=False)


def _manifest_bytes(*, include_png: bool = False, expected_sha256: str | None = _CSV_DIGEST) -> bytes:
    dataset: dict[str, object] = {
        "path": _CSV_PATH,
        "kind": "dataset",
        "inline": False,
    }
    if expected_sha256 is not None:
        dataset["expected_sha256"] = expected_sha256
    artifacts: list[dict[str, object]] = [dataset]
    if include_png:
        artifacts.append(
            {
                "path": _PNG_PATH,
                "kind": "image",
                "inline": True,
                "caption": "Revenue by quarter",
            }
        )
    return json.dumps({"version": 1, "artifacts": artifacts}).encode()


def _manager(tmp_path: Path, sandbox: _FakeSandbox) -> tuple[ArtifactManager, SqlArtifactStore]:
    store = SqlArtifactStore(f"sqlite:///{tmp_path}/jobs.db")
    manager = ArtifactManager(
        job_id="job-publication-regression",
        backend=sandbox,
        store=store,
        config=ArtifactCaptureConfig(enabled=True),
        artifact_dir=_ARTIFACT_DIR,
    )
    return manager, store


def _request(
    tool_name: str,
    *,
    tool_call_id: str = "tc1",
    args: dict[str, object] | None = None,
    files: dict[str, object] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        tool_call={"name": tool_name, "id": tool_call_id, "args": args or {}},
        state={"files": files or {}},
    )


def _writer_state_files() -> dict[str, object]:
    note = {
        "query_topic": "Quarterly revenue",
        "target_components": ["revenue_table"],
        "summary": "Revenue increased.",
        "findings": [],
        "gaps": [],
        "sources": [
            {
                "id": 1,
                "title": "Quarterly results",
                "source_type": "url",
                "locator": "https://example.test/results",
            }
        ],
        "narrative_notes": "Canonical quarterly values.",
        "language": "English",
        "quantitative_datasets": [
            {
                "dataset_id": "revenue",
                "title": "Quarterly revenue",
                "csv_text": _CSV_BYTES.decode("utf-8"),
                "markdown_table": _MARKDOWN_TABLE,
                "summary": "Revenue increased.",
                "source_ids": [1],
                "caveats": [],
                "csv_sha256": "0" * 64,
            }
        ],
    }
    return {
        "/shared/output.md": {"content": _REPORT},
        "/shared/research_note_01_revenue.json": {"content": json.dumps(note)},
    }


async def _publish_csv(
    *,
    manager: ArtifactManager,
    sandbox: _FakeSandbox,
    expected_sha256: str | None = _CSV_DIGEST,
    state_files: dict[str, object] | None = None,
) -> ToolMessage:
    """Simulate writer byte-copy + manifest write, then run the real checkpoint."""
    checkpoint = ArtifactHarvestMiddleware(manager)
    publication_guard = WriterExecuteBudgetMiddleware(max_attempts=3, artifact_manager=manager)
    request = _request(
        "write_file",
        args={
            "file_path": _MANIFEST_PATH,
            "content": _manifest_bytes(expected_sha256=expected_sha256).decode("utf-8"),
        },
        files=state_files if state_files is not None else _writer_state_files(),
    )

    async def _write(_request: object) -> ToolMessage:
        sandbox.files[_CSV_PATH] = _CSV_BYTES
        sandbox.files[_MANIFEST_PATH] = _manifest_bytes(expected_sha256=expected_sha256)
        return ToolMessage(content="manifest written", tool_call_id="tc1", name="write_file", status="success")

    return await checkpoint.awrap_tool_call(
        request,
        lambda current_request: publication_guard.awrap_tool_call(current_request, _write),
    )


def _deepagents_execute_message(response: Any, *, tool_call_id: str) -> ToolMessage:
    """Format the fake response exactly as DeepAgents FilesystemMiddleware does."""
    command_status = "succeeded" if response.exit_code == 0 else "failed"
    return ToolMessage(
        content=f"{response.output}\n[Command {command_status} with exit code {response.exit_code}]",
        tool_call_id=tool_call_id,
        name="execute",
        status="success",
    )


@pytest.mark.asyncio
async def test_csv_only_publishes_exact_bytes_without_chart_execution(tmp_path: Path) -> None:
    sandbox = _FakeSandbox()
    manager, store = _manager(tmp_path, sandbox)
    manager.register_canonical_digests([_CSV_DIGEST])

    result = await _publish_csv(manager=manager, sandbox=sandbox)

    artifacts = store.list("job-publication-regression")
    assert [artifact.filename for artifact in artifacts] == ["revenue.csv"]
    assert artifacts[0].sha256 == _CSV_DIGEST
    assert b"".join(store.open_bytes("job-publication-regression", artifacts[0].artifact_id)) == _CSV_BYTES
    assert "revenue.csv (downloadable" in result.content
    assert sandbox.chart_calls == []
    assert sandbox.scan_calls == []


@pytest.mark.asyncio
async def test_missing_canonical_state_blocks_physical_publication_write(tmp_path: Path) -> None:
    sandbox = _FakeSandbox()
    manager, store = _manager(tmp_path, sandbox)
    manager.register_canonical_digests([_CSV_DIGEST])

    result = await _publish_csv(
        manager=manager,
        sandbox=sandbox,
        state_files={"/shared/output.md": {"content": "# Text fallback"}},
    )

    assert result.status == "error"
    assert "writer_canonical_dataset_missing" in result.content
    assert _CSV_PATH not in sandbox.files
    assert _MANIFEST_PATH not in sandbox.files
    assert store.list("job-publication-regression") == []


@pytest.mark.asyncio
async def test_csv_plus_png_renders_from_already_published_exact_csv(tmp_path: Path) -> None:
    sandbox = _FakeSandbox()
    manager, store = _manager(tmp_path, sandbox)
    manager.register_canonical_digests([_CSV_DIGEST])
    await _publish_csv(manager=manager, sandbox=sandbox)
    checkpoint = ArtifactHarvestMiddleware(manager)
    limiter = WriterExecuteBudgetMiddleware(max_attempts=3, artifact_manager=manager)
    request = _request(
        "execute",
        args={"command": "python /sandbox/make_chart.py /sandbox/aiq-artifacts"},
        files=_writer_state_files(),
    )

    async def _execute(current_request: Any) -> ToolMessage:
        response = sandbox.execute(current_request.tool_call["args"]["command"])
        return _deepagents_execute_message(response, tool_call_id=current_request.tool_call["id"])

    result = await checkpoint.awrap_tool_call(
        request,
        lambda current_request: limiter.awrap_tool_call(current_request, _execute),
    )

    artifacts = {artifact.filename: artifact for artifact in store.list("job-publication-regression")}
    assert sandbox.chart_inputs == [_CSV_BYTES]
    assert len(sandbox.chart_calls) == 1
    assert set(artifacts) == {"revenue.csv", "revenue.png"}
    assert artifacts["revenue.csv"].sha256 == _CSV_DIGEST
    assert "artifact://revenue.png" in result.content


@pytest.mark.asyncio
async def test_three_nonzero_chart_exits_preserve_report_table_and_csv_and_block_fourth(tmp_path: Path) -> None:
    sandbox = _FakeSandbox()
    sandbox.chart_exit_codes = [1, 1, 1, 0]
    manager, store = _manager(tmp_path, sandbox)
    manager.register_canonical_digests([_CSV_DIGEST])
    await _publish_csv(manager=manager, sandbox=sandbox)
    checkpoint = ArtifactHarvestMiddleware(manager)
    limiter = WriterExecuteBudgetMiddleware(max_attempts=3, artifact_manager=manager)
    state_files = _writer_state_files()

    async def _execute(current_request: Any) -> ToolMessage:
        response = sandbox.execute(current_request.tool_call["args"]["command"])
        return _deepagents_execute_message(response, tool_call_id=current_request.tool_call["id"])

    results: list[ToolMessage] = []
    for attempt in range(4):
        request = _request(
            "execute",
            tool_call_id=f"chart-{attempt + 1}",
            args={"command": "python /sandbox/make_chart.py /sandbox/aiq-artifacts"},
            files=state_files,
        )
        results.append(
            await checkpoint.awrap_tool_call(
                request,
                lambda current_request: limiter.awrap_tool_call(current_request, _execute),
            )
        )

    artifacts = store.list("job-publication-regression")
    assert len(sandbox.chart_calls) == 3
    assert limiter.attempts == 3
    assert "writer_execute_budget_exhausted" in results[2].content
    assert "writer_execute_budget_exhausted" in results[3].content
    assert state_files["/shared/output.md"]["content"] == _REPORT
    assert "| quarter | revenue_usd_billions |" in _REPORT
    assert [artifact.filename for artifact in artifacts] == ["revenue.csv"]
    assert b"".join(store.open_bytes("job-publication-regression", artifacts[0].artifact_id)) == _CSV_BYTES


@pytest.mark.asyncio
async def test_missing_registered_digest_rejects_capture_and_blocks_chart_execution(tmp_path: Path) -> None:
    sandbox = _FakeSandbox()
    manager, store = _manager(tmp_path, sandbox)

    publication_result = await _publish_csv(manager=manager, sandbox=sandbox, expected_sha256=None)
    limiter = WriterExecuteBudgetMiddleware(max_attempts=3, artifact_manager=manager)
    chart_result = await limiter.awrap_tool_call(
        _request(
            "execute",
            args={"command": "python /sandbox/make_chart.py /sandbox/aiq-artifacts"},
            files={"/shared/output.md": {"content": _REPORT}},
        ),
        AsyncMock(return_value=ToolMessage(content="unexpected", tool_call_id="tc1")),
    )

    assert "canonical_dataset_digest_missing" in publication_result.content
    assert "writer_canonical_dataset_unpublished" in chart_result.content
    assert store.list("job-publication-regression") == []
    assert sandbox.chart_calls == []


@tool
def _provider_neutral_source(query: str) -> str:
    """Return a deterministic source result for middleware construction."""
    return query


def test_openshell_profile_uses_provider_neutral_writer_publication_checkpoint() -> None:
    config = yaml.safe_load((_REPO_ROOT / "configs/config_openshell.yml").read_text(encoding="utf-8"))
    sandbox_config = config["functions"]["deep_research_sandbox"]
    skills_config = config["functions"]["deep_research_skills"]
    manager = SimpleNamespace(artifact_dir=_ARTIFACT_DIR)
    registry = SourceRegistryMiddleware(source_tool_names={_provider_neutral_source.name})
    tool_set = build_deep_research_tool_set(
        [_provider_neutral_source],
        source_registry_middleware=registry,
        max_concurrent_source_tool_calls=1,
        max_source_tool_batch_size=1,
    )

    middleware_set = build_deep_research_middleware_set(
        tool_set=tool_set,
        source_registry_middleware=registry,
        artifact_manager=manager,
    )

    assert sandbox_config["provider"] == "openshell"
    assert skills_config["agents"] == {
        "researcher-agent": ["research"],
        "writer-agent": ["synthesis"],
    }
    assert skills_config["require_sandbox"] == ["research", "synthesis"]
    assert not any(isinstance(item, ArtifactHarvestMiddleware) for item in middleware_set.researcher)
    assert not any(isinstance(item, ArtifactHarvestMiddleware) for item in middleware_set.planner)
    checkpoint = next(item for item in middleware_set.writer if isinstance(item, ArtifactHarvestMiddleware))
    limiter = next(item for item in middleware_set.writer if isinstance(item, WriterExecuteBudgetMiddleware))
    assert checkpoint.artifact_manager is manager
    assert limiter.artifact_manager is manager
