# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in end-to-end checks against a separately deployed NeMo Retriever service."""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
import zipfile
from pathlib import Path

import pytest
from knowledge_layer.nemo_retriever.adapter import NemoRetrieverIngestor
from knowledge_layer.nemo_retriever.adapter import NemoRetrieverRetriever

from aiq_agent.knowledge import JobState
from aiq_agent.knowledge.schema import FileStatus

pytestmark = pytest.mark.skipif(
    os.environ.get("AIQ_NRL_LIVE_TESTS") != "1",
    reason="Set AIQ_NRL_LIVE_TESTS=1 to run against a live NeMo Retriever deployment",
)


def _bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _config(*, scope: str | None = None, token: str | None = None) -> dict:
    selected_scope = scope or os.environ.get("NRL_SCOPE")
    if not selected_scope:
        pytest.fail("NRL_SCOPE is required for the live NeMo Retriever test")
    return {
        "base_url": os.environ.get("NRL_BASE_URL", "http://127.0.0.1:7670"),
        "api_token": token if token is not None else os.environ.get("NRL_API_TOKEN"),
        "scope": selected_scope,
        "connect_timeout_s": float(os.environ.get("NRL_CONNECT_TIMEOUT_S", "30")),
        "request_timeout_s": float(os.environ.get("NRL_REQUEST_TIMEOUT_S", "300")),
        "max_retries": int(os.environ.get("NRL_MAX_RETRIES", "5")),
        "max_concurrency": int(os.environ.get("NRL_MAX_CONCURRENCY", "3")),
        "verify_ssl": _bool_env("NRL_VERIFY_SSL", True),
        "ca_bundle": os.environ.get("NRL_CA_BUNDLE"),
        "collection_ttl_hours": float(os.environ.get("NRL_COLLECTION_TTL_HOURS", "24")),
    }


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * percentile))
    return ordered[index] * 1000


def _assert_no_physical_storage_identifiers(value: object) -> None:
    serialized = json.dumps(value, default=str).lower()
    for forbidden in ("lancedb_uri", "table_name", "physical_table", "table_path", "vdb_uri"):
        assert forbidden not in serialized


def test_nemo_retriever_live_end_to_end(tmp_path):
    config = _config()
    ingestor = NemoRetrieverIngestor(config)
    retriever = NemoRetrieverRetriever(config)
    collection_name = f"aiq-nrl-live-{uuid.uuid4().hex[:10]}"
    timeout_s = float(os.environ.get("AIQ_NRL_LIVE_TIMEOUT_S", "1800"))

    text_file = tmp_path / "live.txt"
    html_file = tmp_path / "live.html"
    pdf_file = tmp_path / "multimodal_test.pdf"
    text_file.write_text(
        "The AIQ NeMo Retriever live validation phrase is separation of inputs and outputs.",
        encoding="utf-8",
    )
    html_file.write_text(
        "<html><body><h1>AIQ validation</h1><p>NeMo Retriever is an external knowledge service.</p></body></html>",
        encoding="utf-8",
    )
    archive = Path(__file__).parent / "data" / "Knowledge_Layer_Test_Data.zip"
    with zipfile.ZipFile(archive) as fixture_zip:
        pdf_file.write_bytes(fixture_zip.read("multimodal_test.pdf"))

    assert asyncio.run(ingestor.health_check())
    ingestor.create_collection(collection_name, description="AIQ NeMo Retriever live adapter validation")
    try:
        job_id = ingestor.submit_job(
            [str(text_file), str(html_file), str(pdf_file)],
            collection_name,
            {"original_filenames": [text_file.name, html_file.name, pdf_file.name]},
        )
        deadline = time.monotonic() + timeout_s
        status = ingestor.get_job_status(job_id)
        while not status.is_terminal and time.monotonic() < deadline:
            time.sleep(2)
            status = ingestor.get_job_status(job_id)
        assert status.status == JobState.COMPLETED, status.model_dump()
        assert status.processed_files == 3
        assert all(item.status == FileStatus.SUCCESS for item in status.file_details)
        assert all(item.file_id for item in status.file_details)
        assert set(status.metadata["attempt_ids"]) == {item.file_id for item in status.file_details}
        assert all(status.metadata["attempt_ids"][item.file_id] != item.file_id for item in status.file_details)

        files = ingestor.list_files(collection_name)
        assert len(files) == 3
        assert {item.file_id for item in files} == {item.file_id for item in status.file_details}
        _assert_no_physical_storage_identifiers([item.model_dump() for item in files])

        result = asyncio.run(
            retriever.retrieve(
                "What is the AIQ NeMo Retriever live validation phrase?",
                collection_name,
                top_k=5,
            )
        )
        assert result.success, result.error_message
        assert result.chunks
        assert all(chunk.display_citation for chunk in result.chunks)
        _assert_no_physical_storage_identifiers(result.model_dump())

        # Re-instantiation represents an AIQ process restart; NRL-owned data must remain available.
        restarted_ingestor = NemoRetrieverIngestor(config)
        assert {item.file_id for item in restarted_ingestor.list_files(collection_name)} == {
            item.file_id for item in files
        }

        deleted_id = files[0].file_id
        assert restarted_ingestor.delete_file(deleted_id, collection_name)
        assert deleted_id not in {item.file_id for item in restarted_ingestor.list_files(collection_name)}

        second_scope = os.environ.get("NRL_SECOND_SCOPE")
        if second_scope:
            second = NemoRetrieverIngestor(_config(scope=second_scope, token=os.environ.get("NRL_SECOND_API_TOKEN")))
            assert second.get_collection(collection_name) is None

        samples = max(3, int(os.environ.get("AIQ_NRL_LATENCY_SAMPLES", "10")))
        direct_ms: list[float] = []
        adapter_ms: list[float] = []
        query_payload = {"query": "AIQ validation", "collection_name": collection_name, "top_k": 5}
        for _ in range(samples):
            started = time.perf_counter()
            ingestor._transport.request_json("POST", "/v1/query", operation="query", json=query_payload)
            direct_ms.append(time.perf_counter() - started)
            started = time.perf_counter()
            asyncio.run(retriever.retrieve("AIQ validation", collection_name, top_k=5))
            adapter_ms.append(time.perf_counter() - started)
        print(
            "NRL latency ms "
            f"direct(p50={_percentile(direct_ms, 0.50):.1f}, p95={_percentile(direct_ms, 0.95):.1f}, "
            f"p99={_percentile(direct_ms, 0.99):.1f}) "
            f"adapter(p50={_percentile(adapter_ms, 0.50):.1f}, p95={_percentile(adapter_ms, 0.95):.1f}, "
            f"p99={_percentile(adapter_ms, 0.99):.1f})"
        )
    finally:
        assert ingestor.delete_collection(collection_name)
