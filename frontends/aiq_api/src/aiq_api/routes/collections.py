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

"""Collection management endpoints."""

import asyncio
import logging
from typing import Any

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator

from aiq_agent.knowledge.base import BaseIngestor
from aiq_agent.knowledge.factory import get_retriever
from aiq_agent.knowledge.schema import CollectionInfo
from aiq_agent.knowledge.schema import RetrievalResult

from ..models.requests import CreateCollectionRequest

logger = logging.getLogger(__name__)


class KnowledgeQueryRequest(BaseModel):
    """Request body for retrieval without invoking an agent LLM."""

    query: str = Field(..., min_length=1, max_length=32768)
    collection_name: str = Field(..., min_length=1)
    top_k: int = Field(default=10, ge=1, le=100)
    filters: dict[str, Any] | None = None

    @field_validator("query", "collection_name")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


def _require_ingestor() -> BaseIngestor:
    from aiq_agent.knowledge.factory import get_active_ingestor

    ingestor = get_active_ingestor()
    if ingestor is None:
        raise HTTPException(status_code=503, detail="Knowledge API not configured")
    return ingestor


def add_collection_routes(router: APIRouter):
    """Add collection management routes to the FastAPI app."""

    @router.post(
        "/v1/collections",
        response_model=CollectionInfo,
        status_code=201,
        tags=["collections"],
        summary="Create a new collection",
    )
    async def create_collection(
        request: CreateCollectionRequest,
        ingestor: BaseIngestor = Depends(_require_ingestor),
    ) -> CollectionInfo:
        """Create a new collection for storing documents."""
        try:
            return await asyncio.to_thread(
                ingestor.create_collection,
                name=request.name,
                description=request.description,
                metadata=request.metadata,
            )
        except Exception as e:
            logger.error(f"Failed to create collection '{request.name}': {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get(
        "/v1/collections",
        response_model=list[CollectionInfo],
        tags=["collections"],
        summary="List all collections",
    )
    async def list_collections(
        ingestor: BaseIngestor = Depends(_require_ingestor),
    ) -> list[CollectionInfo]:
        """List all available collections."""
        try:
            return await asyncio.to_thread(ingestor.list_collections)
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get(
        "/v1/collections/{name}",
        response_model=CollectionInfo,
        tags=["collections"],
        summary="Get collection details",
    )
    async def get_collection(
        name: str,
        ingestor: BaseIngestor = Depends(_require_ingestor),
    ) -> CollectionInfo:
        """Get details for a specific collection."""
        collection = await asyncio.to_thread(ingestor.get_collection, name)
        if collection is None:
            raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")
        return collection

    @router.delete(
        "/v1/collections/{name}",
        tags=["collections"],
        summary="Delete a collection",
    )
    async def delete_collection(
        name: str,
        ingestor: BaseIngestor = Depends(_require_ingestor),
    ) -> dict:
        """Delete a collection and all its contents."""
        try:
            success = await asyncio.to_thread(ingestor.delete_collection, name)
            if not success:
                raise HTTPException(status_code=500, detail=f"Failed to delete collection '{name}'")
            return {"success": True, "collection": name}
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to delete collection '{name}': {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get(
        "/v1/knowledge/health",
        tags=["health"],
        summary="Check knowledge backend health",
    )
    async def health_check(
        ingestor: BaseIngestor = Depends(_require_ingestor),
    ) -> dict:
        """Check if the knowledge backend is healthy and reachable."""
        try:
            healthy = await ingestor.health_check()
            if not healthy:
                raise HTTPException(status_code=503, detail="Knowledge backend unhealthy")
            return {"status": "healthy", "backend": ingestor.backend_name}
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise HTTPException(status_code=503, detail=str(e))

    @router.post(
        "/v1/knowledge/query",
        response_model=RetrievalResult,
        tags=["knowledge"],
        summary="Query an ingested knowledge collection",
        description="Retrieve normalized chunks directly without invoking a generative model.",
    )
    async def query_knowledge(
        request: KnowledgeQueryRequest,
        ingestor: BaseIngestor = Depends(_require_ingestor),
    ) -> RetrievalResult:
        """Retrieve chunks from the configured backend and collection."""
        try:
            collection = await asyncio.to_thread(ingestor.get_collection, request.collection_name)
        except Exception as exc:
            logger.warning(
                "Knowledge collection lookup failed (backend=%s, error_type=%s)",
                ingestor.backend_name,
                type(exc).__name__,
            )
            raise HTTPException(status_code=502, detail="Knowledge backend query failed") from exc

        if collection is None:
            raise HTTPException(status_code=404, detail=f"Collection '{request.collection_name}' not found")

        retriever = None
        try:
            retriever = await asyncio.to_thread(get_retriever, ingestor.backend_name, ingestor.config)
            result = await retriever.retrieve(
                query=request.query,
                collection_name=request.collection_name,
                top_k=request.top_k,
                filters=request.filters,
            )
        except Exception as exc:
            logger.warning(
                "Knowledge retrieval failed (backend=%s, error_type=%s)",
                ingestor.backend_name,
                type(exc).__name__,
            )
            raise HTTPException(status_code=502, detail="Knowledge backend query failed") from exc
        finally:
            close = getattr(retriever, "close", None)
            if callable(close):
                try:
                    await asyncio.to_thread(close)
                except Exception:
                    logger.warning("Failed to close knowledge retriever", exc_info=True)

        if not result.success:
            logger.warning("Knowledge retrieval returned a failure (backend=%s)", ingestor.backend_name)
            raise HTTPException(status_code=502, detail="Knowledge backend query failed")
        return result
