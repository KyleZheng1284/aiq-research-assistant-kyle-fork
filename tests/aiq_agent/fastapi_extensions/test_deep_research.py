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

"""
Tests for the async job API routes.

Module under test: frontends/aiq_api/src/aiq_api/routes/jobs.py

API endpoints tested:
    GET  /v1/jobs/async/agents                         - List available agent types
    POST /v1/jobs/async/submit                         - Submit a new job
    GET  /v1/jobs/async/job/{id}                       - Get job status
    GET  /v1/jobs/async/job/{id}/stream                - SSE stream from beginning
    POST /v1/jobs/async/job/{id}/cancel                - Cancel a running job
    GET  /v1/jobs/async/job/{id}/state                 - Get artifacts from event store
    GET  /v1/jobs/async/job/{id}/report                - Get final report

Test coverage:
    TestJobSubmitRequest:
        - Valid request with defaults
        - Custom job_id and expiry_seconds
        - Empty input rejected (min_length=1)
        - Expiry validation (ge=600, le=604800)

    TestJobStatusResponse:
        - Minimal response (job_id, status)
        - Full response with error, created_at

    TestJobStateResponse:
        - Response without state (has_state=False)
        - Response with artifacts

    TestJobReportResponse:
        - Response without report (has_report=False)
        - Response with report content

    TestRegisterRoutes:
        - Routes not registered when Dask unavailable
        - Routes not registered without job_store
        - Routes registered when infrastructure available
"""

import json
from hashlib import sha256
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from aiq_api.routes.jobs import JobReportResponse
from aiq_api.routes.jobs import JobStateResponse
from aiq_api.routes.jobs import JobStatusResponse
from aiq_api.routes.jobs import JobSubmitRequest


class TestJobSubmitRequest:
    """Tests for the JobSubmitRequest model."""

    def test_valid_request(self):
        """Test creating a valid submit request."""
        req = JobSubmitRequest(agent_type="deep_researcher", input="What is CUDA?")

        assert req.input == "What is CUDA?"
        assert req.agent_type == "deep_researcher"
        assert req.job_id is None
        assert req.expiry_seconds is None
        assert req.data_sources is None

    def test_with_data_sources(self):
        """Test submit request with selected data sources."""
        req = JobSubmitRequest(agent_type="deep_researcher", input="query", data_sources=["web_search"])

        assert req.data_sources == ["web_search"]

    def test_null_data_sources_defaults_to_all_sources(self):
        """Test null data_sources preserves all-source behavior."""
        req = JobSubmitRequest(agent_type="deep_researcher", input="query", data_sources=None)

        assert req.data_sources is None

    def test_empty_data_sources_accepted(self):
        """Test that empty data_sources is accepted as a deliberate 'no data-source tools' signal."""
        req = JobSubmitRequest(agent_type="deep_researcher", input="query", data_sources=[])

        assert req.data_sources == []

    def test_with_custom_job_id(self):
        """Test submit request with custom job ID."""
        req = JobSubmitRequest(agent_type="deep_researcher", input="query", job_id="custom-123")

        assert req.job_id == "custom-123"

    def test_with_custom_expiry(self):
        """Test submit request with custom expiry."""
        req = JobSubmitRequest(agent_type="deep_researcher", input="query", expiry_seconds=7200)

        assert req.expiry_seconds == 7200

    def test_empty_input_rejected(self):
        """Test that empty input is rejected."""
        with pytest.raises(ValueError):
            JobSubmitRequest(agent_type="deep_researcher", input="")

    def test_expiry_too_low_rejected(self):
        """Test that expiry below 600 is rejected."""
        with pytest.raises(ValueError):
            JobSubmitRequest(agent_type="deep_researcher", input="query", expiry_seconds=300)

    def test_expiry_too_high_rejected(self):
        """Test that expiry above 604800 is rejected."""
        with pytest.raises(ValueError):
            JobSubmitRequest(agent_type="deep_researcher", input="query", expiry_seconds=700000)


class TestJobStatusResponse:
    """Tests for the JobStatusResponse model."""

    def test_minimal_response(self):
        """Test minimal job response."""
        resp = JobStatusResponse(job_id="123", status="running")

        assert resp.job_id == "123"
        assert resp.status == "running"
        assert resp.error is None
        assert resp.created_at is None

    def test_full_response(self):
        """Test full job response."""
        resp = JobStatusResponse(
            job_id="123",
            status="success",
            error="some error",
            created_at="2026-01-20T10:00:00",
        )

        assert resp.job_id == "123"
        assert resp.status == "success"
        assert resp.error == "some error"
        assert resp.created_at == "2026-01-20T10:00:00"


class TestJobStateResponse:
    """Tests for the JobStateResponse model."""

    def test_without_state(self):
        """Test state response without state."""
        resp = JobStateResponse(job_id="123", has_state=False)

        assert resp.job_id == "123"
        assert resp.has_state is False
        assert resp.state is None

    def test_with_artifacts(self):
        """Test state response with artifacts."""
        artifacts = {"tools": [], "outputs": []}
        resp = JobStateResponse(job_id="123", has_state=True, artifacts=artifacts)

        assert resp.has_state is True
        assert resp.artifacts == artifacts


class TestJobReportResponse:
    """Tests for the JobReportResponse model."""

    def test_without_report(self):
        """Test report response without report."""
        resp = JobReportResponse(job_id="123", has_report=False)

        assert resp.job_id == "123"
        assert resp.has_report is False
        assert resp.report is None

    def test_with_report(self):
        """Test report response with report."""
        resp = JobReportResponse(job_id="123", has_report=True, report="# Report\n\nContent here")

        assert resp.has_report is True
        assert resp.report == "# Report\n\nContent here"


class TestRegisterRoutes:
    """Tests for the register_routes function."""

    @pytest.mark.asyncio
    async def test_routes_not_registered_without_dask(self):
        """Test that routes are not registered when Dask is not available."""
        from aiq_api.routes.jobs import register_job_routes

        mock_app = MagicMock()
        mock_builder = MagicMock()
        mock_builder.get_function_config.side_effect = KeyError("Not found")
        mock_worker = MagicMock()
        mock_worker._dask_available = False
        mock_worker._job_store = None

        await register_job_routes(mock_app, mock_builder, mock_worker)

        # Async job submission/control routes require Dask + a job store and must
        # NOT be registered. The always-on control-plane routes (data sources,
        # agent list, and per-user MCP auth status/connect/callback) are still
        # registered regardless of Dask availability.
        post_paths = [c.args[0] for c in mock_app.post.call_args_list if c.args]
        assert "/v1/jobs/async/submit" not in post_paths
        assert "/v1/auth/mcp/{source_id}/connect" in post_paths
        get_paths = [c.args[0] for c in mock_app.get.call_args_list if c.args]
        assert "/v1/jobs/async/agents" in get_paths
        assert "/v1/data_sources" in get_paths
        assert "/v1/auth/mcp/{source_id}/status" in get_paths
        assert "/v1/auth/mcp/{source_id}/callback" in get_paths

    @pytest.mark.asyncio
    async def test_routes_not_registered_without_job_store(self):
        """Test that routes are not registered without job store."""
        from aiq_api.routes.jobs import register_job_routes

        mock_app = MagicMock()
        mock_builder = MagicMock()
        mock_builder.get_function_config.side_effect = KeyError("Not found")
        mock_worker = MagicMock()
        mock_worker._dask_available = True
        mock_worker._job_store = None

        await register_job_routes(mock_app, mock_builder, mock_worker)

        # Async job submission/control routes require Dask + a job store and must
        # NOT be registered. The always-on control-plane routes (data sources,
        # agent list, and per-user MCP auth status/connect/callback) are still
        # registered regardless of Dask availability.
        post_paths = [c.args[0] for c in mock_app.post.call_args_list if c.args]
        assert "/v1/jobs/async/submit" not in post_paths
        assert "/v1/auth/mcp/{source_id}/connect" in post_paths
        get_paths = [c.args[0] for c in mock_app.get.call_args_list if c.args]
        assert "/v1/jobs/async/agents" in get_paths
        assert "/v1/data_sources" in get_paths
        assert "/v1/auth/mcp/{source_id}/status" in get_paths
        assert "/v1/auth/mcp/{source_id}/callback" in get_paths

    @pytest.mark.asyncio
    async def test_routes_registered_with_dask(self):
        """Test that routes are registered when Dask is available."""
        from aiq_api.routes.jobs import register_job_routes

        mock_app = MagicMock()
        mock_builder = MagicMock()
        mock_builder.get_function_config.side_effect = KeyError("Not found")
        mock_worker = MagicMock()
        mock_worker._dask_available = True
        mock_worker._job_store = MagicMock()
        mock_worker._scheduler_address = "tcp://localhost:8786"
        mock_worker._db_url = "sqlite:///./test.db"
        mock_worker._config_file_path = "/path/to/config.yml"
        mock_worker._log_level = 20
        mock_worker._use_dask_threads = False
        mock_worker._front_end_config = MagicMock(expiry_seconds=86400)

        await register_job_routes(mock_app, mock_builder, mock_worker)

        assert mock_app.post.call_count >= 2
        assert mock_app.get.call_count >= 6


class TestArtifactRoutes:
    """End-to-end route contract for a runtime-verified canonical CSV."""

    @pytest.mark.asyncio
    async def test_canonical_csv_sha_and_exact_bytes_round_trip_through_job_api(self, tmp_path):
        """The original artifacts API surface returns the canonical digest and bytes."""
        from fastapi import FastAPI
        from httpx import ASGITransport
        from httpx import AsyncClient

        from aiq_agent.agents.deep_researcher.sandbox.artifacts import ArtifactManager
        from aiq_agent.agents.deep_researcher.sandbox.artifacts import SqlArtifactStore
        from aiq_agent.agents.deep_researcher.sandbox.config import ArtifactCaptureConfig
        from aiq_api.routes.jobs import register_job_routes

        job_id = "canonical-api-job"
        artifact_dir = "/sandbox/canonical-api-job/aiq-artifacts"
        csv_path = f"{artifact_dir}/revenue.csv"
        manifest_path = f"{artifact_dir}/manifest.json"
        csv_bytes = b"quarter,revenue\r\nQ1,22.6\r\nQ2,26.3\r\n"
        digest = sha256(csv_bytes).hexdigest()
        manifest = json.dumps(
            {
                "version": 1,
                "artifacts": [
                    {
                        "path": csv_path,
                        "kind": "dataset",
                        "inline": False,
                        "expected_sha256": digest,
                    }
                ],
            }
        ).encode("utf-8")

        class _Backend:
            files = {manifest_path: manifest, csv_path: csv_bytes}

            def download_files(self, paths):
                return [
                    SimpleNamespace(
                        path=path,
                        content=self.files.get(path),
                        error=None if path in self.files else "not found",
                    )
                    for path in paths
                ]

        db_url = f"sqlite:///{tmp_path / 'jobs.db'}"
        store = SqlArtifactStore(db_url)
        manager = ArtifactManager(
            job_id=job_id,
            backend=_Backend(),
            store=store,
            config=ArtifactCaptureConfig(enabled=True),
            artifact_dir=artifact_dir,
        )
        manager.register_canonical_digests([digest])
        [captured] = manager.harvest_after_execute()

        app = FastAPI()
        worker = MagicMock()
        worker._dask_available = True
        worker._job_store = MagicMock()
        worker._scheduler_address = "tcp://localhost:8786"
        worker._db_url = db_url
        worker._config_file_path = "/path/to/config.yml"
        worker._log_level = 20
        worker._use_dask_threads = False
        worker._front_end_config = SimpleNamespace(expiry_seconds=86400)

        with (
            patch(
                "aiq_api.mcp_auth.factory.build_mcp_auth_provider",
                new=AsyncMock(return_value=MagicMock()),
            ),
            patch("aiq_api.routes.auth.register_mcp_auth_routes"),
            patch(
                "aiq_api.jobs.crypto.validate_content_encryption_startup_async",
                new=AsyncMock(),
            ),
            patch(
                "aiq_api.jobs.access.authorize_job_access",
                new=AsyncMock(return_value=SimpleNamespace(job_id=job_id)),
            ),
            patch(
                "aiq_api.routes.jobs.require_verified_principal",
                return_value=SimpleNamespace(subject="test-owner"),
            ),
        ):
            await register_job_routes(app, MagicMock(), worker)
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                listing = await client.get(f"/v1/jobs/async/job/{job_id}/artifacts")
                content = await client.get(f"/v1/jobs/async/job/{job_id}/artifacts/{captured.artifact_id}/content")

        assert listing.status_code == 200
        [metadata] = listing.json()["artifacts"]
        assert metadata["sha256"] == digest
        assert metadata["filename"] == "revenue.csv"
        assert content.status_code == 200
        assert content.content == csv_bytes


class TestArtifactHelpers:
    """Tests for artifact extraction helper functions."""

    def test_extract_event_metadata_with_data(self):
        """Test extracting metadata from event with data dict."""
        from aiq_api.routes.jobs import _extract_event_metadata

        event = {
            "type": "tool.start",
            "data": {"name": "test", "input": "query"},
            "metadata": {"workflow": "agent-1"},
        }

        data, metadata = _extract_event_metadata(event)

        assert data == {"name": "test", "input": "query"}
        assert metadata == {"workflow": "agent-1"}

    def test_extract_event_metadata_fallback_to_nested(self):
        """Test extracting metadata from nested data.metadata."""
        from aiq_api.routes.jobs import _extract_event_metadata

        event = {
            "type": "tool.start",
            "data": {"name": "test", "metadata": {"workflow": "nested"}},
        }

        data, metadata = _extract_event_metadata(event)

        assert metadata == {"workflow": "nested"}

    def test_extract_event_metadata_handles_non_dict(self):
        """Test extracting metadata handles non-dict data."""
        from aiq_api.routes.jobs import _extract_event_metadata

        event = {"type": "test", "data": "string_data"}

        data, metadata = _extract_event_metadata(event)

        assert data == {}
        assert metadata == {}

    def test_process_tool_start(self):
        """Test processing tool.start event."""
        from aiq_api.routes.jobs import _process_tool_start

        event = {"timestamp": "2026-01-22T10:00:00"}
        data = {"id": "tool-1", "name": "search", "data": {"input": "query"}}
        metadata = {"workflow": "agent-1"}
        tool_call_map: dict = {}

        _process_tool_start(event, data, metadata, tool_call_map)

        assert "tool-1" in tool_call_map
        assert tool_call_map["tool-1"]["name"] == "search"
        assert tool_call_map["tool-1"]["status"] == "running"

    def test_process_tool_end_updates_existing(self):
        """Test processing tool.end updates existing tool."""
        from aiq_api.routes.jobs import _process_tool_end

        event = {"timestamp": "2026-01-22T10:00:01"}
        data = {"id": "tool-1", "name": "search", "data": {"output": "result"}}
        metadata = {"workflow": "agent-1"}
        tool_call_map = {
            "tool-1": {
                "id": "tool-1",
                "name": "search",
                "input": "query",
                "output": None,
                "status": "running",
            }
        }

        _process_tool_end(event, data, metadata, tool_call_map)

        assert tool_call_map["tool-1"]["output"] == "result"
        assert tool_call_map["tool-1"]["status"] == "completed"

    def test_process_tool_end_creates_new(self):
        """Test processing tool.end creates new entry if missing."""
        from aiq_api.routes.jobs import _process_tool_end

        event = {"timestamp": "2026-01-22T10:00:01"}
        data = {"id": "tool-2", "name": "other", "data": {"output": "result"}}
        metadata = {"workflow": "agent-1"}
        tool_call_map: dict = {}

        _process_tool_end(event, data, metadata, tool_call_map)

        assert "tool-2" in tool_call_map
        assert tool_call_map["tool-2"]["status"] == "completed"

    def test_process_artifact_update(self):
        """Test processing artifact.update event."""
        from aiq_api.routes.jobs import _process_artifact_update

        event = {"name": "output.md", "timestamp": "2026-01-22T10:00:00"}
        data = {"type": "output", "content": "Report content", "extra": "value"}
        metadata = {"workflow": "agent-1"}
        outputs: list = []
        sources_found: set = set()
        sources_cited: set = set()

        _process_artifact_update(event, data, metadata, outputs, sources_found, sources_cited)

        assert len(outputs) == 1
        assert outputs[0]["type"] == "output"
        assert outputs[0]["content"] == "Report content"
        assert outputs[0]["extra"] == "value"

    def test_process_artifact_update_skips_empty_content(self):
        """Test that empty content is skipped."""
        from aiq_api.routes.jobs import _process_artifact_update

        event = {"name": "empty.md", "timestamp": "2026-01-22T10:00:00"}
        data = {"type": "output", "content": None}
        metadata = {}
        outputs: list = []
        sources_found: set = set()
        sources_cited: set = set()

        _process_artifact_update(event, data, metadata, outputs, sources_found, sources_cited)

        assert len(outputs) == 0

    def test_process_artifact_update_tracks_citation_source(self):
        """Test that citation_source events are tracked."""
        from aiq_api.routes.jobs import _process_artifact_update

        event = {"name": "https://example.com", "timestamp": "2026-01-22T10:00:00"}
        data = {"type": "citation_source", "content": "https://example.com", "url": "https://example.com"}
        metadata = {}
        outputs: list = []
        sources_found: set = set()
        sources_cited: set = set()

        _process_artifact_update(event, data, metadata, outputs, sources_found, sources_cited)

        assert len(sources_found) == 1
        assert "https://example.com" in sources_found
        assert len(sources_cited) == 0

    def test_process_artifact_update_tracks_citation_use(self):
        """Test that citation_use events are tracked."""
        from aiq_api.routes.jobs import _process_artifact_update

        event = {"name": "https://example.com", "timestamp": "2026-01-22T10:00:00"}
        data = {"type": "citation_use", "content": "https://example.com", "url": "https://example.com"}
        metadata = {}
        outputs: list = []
        sources_found: set = set()
        sources_cited: set = set()

        _process_artifact_update(event, data, metadata, outputs, sources_found, sources_cited)

        assert len(sources_cited) == 1
        assert "https://example.com" in sources_cited
        assert len(sources_found) == 0
