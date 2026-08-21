# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import Mock

import pytest
from fastapi import APIRouter
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.testclient import TestClient

from aiq_agent.knowledge.schema import Chunk
from aiq_agent.knowledge.schema import ContentType
from aiq_agent.knowledge.schema import RetrievalResult
from aiq_api.routes import collections as routes_module
from aiq_api.routes.collections import KnowledgeQueryRequest
from aiq_api.routes.collections import _require_ingestor
from aiq_api.routes.collections import add_collection_routes


def _query_endpoint():
    router = APIRouter()
    add_collection_routes(router)
    return next(route.endpoint for route in router.routes if route.path == "/v1/knowledge/query")


def _request(**overrides) -> KnowledgeQueryRequest:
    values = {"query": "What was revenue?", "collection_name": "reports", "top_k": 3}
    values.update(overrides)
    return KnowledgeQueryRequest(**values)


def _result(*, success: bool = True) -> RetrievalResult:
    return RetrievalResult(
        query="What was revenue?",
        backend="nemo_retriever_local",
        success=success,
        error_message=None if success else "private backend detail at /private/data",
        chunks=[
            Chunk(
                chunk_id="chunk-1",
                content="Revenue increased.",
                score=0.0,
                distance=-0.125,
                file_name="report.pdf",
                display_citation="report.pdf, p.1",
                page_number=1,
                content_type=ContentType.TEXT,
            )
        ]
        if success
        else [],
    )


def _adapters(*, get_collection=lambda _name: object(), result=None):
    ingestor = SimpleNamespace(
        backend_name="nemo_retriever_local",
        config={"data_dir": "/private/data"},
        get_collection=get_collection,
    )
    retriever = SimpleNamespace(
        retrieve=AsyncMock(return_value=result or _result()),
        close=Mock(),
    )
    return ingestor, retriever


def test_direct_query_preserves_native_distance_and_closes_retriever(monkeypatch) -> None:
    endpoint = _query_endpoint()
    ingestor, retriever = _adapters()
    monkeypatch.setattr(routes_module, "get_retriever", Mock(return_value=retriever))

    result = asyncio.run(endpoint(request=_request(filters={"team": "finance"}), ingestor=ingestor))

    assert result.chunks[0].score == 0.0
    assert result.chunks[0].distance == -0.125
    retriever.retrieve.assert_awaited_once_with(
        query="What was revenue?",
        collection_name="reports",
        top_k=3,
        filters={"team": "finance"},
    )
    retriever.close.assert_called_once_with()


def test_direct_query_returns_404_before_creating_retriever(monkeypatch) -> None:
    endpoint = _query_endpoint()
    ingestor, _ = _adapters(get_collection=lambda _name: None)
    factory = Mock()
    monkeypatch.setattr(routes_module, "get_retriever", factory)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(endpoint(request=_request(), ingestor=ingestor))

    assert exc_info.value.status_code == 404
    factory.assert_not_called()


def test_direct_query_collection_lookup_does_not_block_event_loop(monkeypatch) -> None:
    endpoint = _query_endpoint()
    release = threading.Event()

    def get_collection(_name):
        release.wait(timeout=5)
        return object()

    ingestor, retriever = _adapters(get_collection=get_collection)
    monkeypatch.setattr(routes_module, "get_retriever", Mock(return_value=retriever))

    async def exercise() -> None:
        started = time.monotonic()
        task = asyncio.create_task(endpoint(request=_request(), ingestor=ingestor))
        await asyncio.sleep(0.05)
        assert time.monotonic() - started < 0.5
        assert not task.done()
        release.set()
        await task

    try:
        asyncio.run(exercise())
    finally:
        release.set()


def test_direct_query_redacts_backend_failure(monkeypatch) -> None:
    endpoint = _query_endpoint()
    ingestor, retriever = _adapters(result=_result(success=False))
    monkeypatch.setattr(routes_module, "get_retriever", Mock(return_value=retriever))

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(endpoint(request=_request(), ingestor=ingestor))

    assert exc_info.value.status_code == 502
    assert "private backend detail" not in str(exc_info.value.detail)
    assert "/private/data" not in str(exc_info.value.detail)
    retriever.close.assert_called_once_with()


def test_direct_query_cleanup_logs_only_safe_error_metadata(monkeypatch, caplog) -> None:
    endpoint = _query_endpoint()
    ingestor, retriever = _adapters()
    retriever.close.side_effect = RuntimeError("secret-token at /private/data")
    monkeypatch.setattr(routes_module, "get_retriever", Mock(return_value=retriever))

    result = asyncio.run(endpoint(request=_request(), ingestor=ingestor))

    assert result.success is True
    assert "error_type=RuntimeError" in caplog.text
    assert "secret-token" not in caplog.text
    assert "/private/data" not in caplog.text


def test_direct_query_http_contract_and_validation(monkeypatch) -> None:
    app = FastAPI()
    add_collection_routes(app)
    ingestor, retriever = _adapters()
    app.dependency_overrides[_require_ingestor] = lambda: ingestor
    monkeypatch.setattr(routes_module, "get_retriever", Mock(return_value=retriever))

    with TestClient(app) as client:
        response = client.post(
            "/v1/knowledge/query",
            json={"query": "What was revenue?", "collection_name": "reports", "top_k": 3},
        )
        blank_response = client.post(
            "/v1/knowledge/query",
            json={"query": "   ", "collection_name": "reports"},
        )

    assert response.status_code == 200
    assert response.json()["chunks"][0]["score"] == 0.0
    assert response.json()["chunks"][0]["distance"] == -0.125
    assert blank_response.status_code == 422
