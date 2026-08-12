# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AIQ Knowledge Layer adapter for a separately deployed NeMo Retriever service."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.parse import urlparse

from pydantic import ValidationError

from aiq_agent.knowledge import BaseIngestor
from aiq_agent.knowledge import BaseRetriever
from aiq_agent.knowledge import Chunk
from aiq_agent.knowledge import ContentType
from aiq_agent.knowledge import FileProgress
from aiq_agent.knowledge import IngestionJobStatus
from aiq_agent.knowledge import JobState
from aiq_agent.knowledge import RetrievalResult
from aiq_agent.knowledge import clear_collection_summaries
from aiq_agent.knowledge import get_available_documents
from aiq_agent.knowledge import list_summary_collections
from aiq_agent.knowledge import register_ingestor
from aiq_agent.knowledge import register_retriever
from aiq_agent.knowledge import register_summary
from aiq_agent.knowledge import unregister_summary
from aiq_agent.knowledge.base import TTLCleanupMixin
from aiq_agent.knowledge.schema import CollectionInfo
from aiq_agent.knowledge.schema import FileInfo
from aiq_agent.knowledge.schema import FileStatus

from ._models import CollectionDeleteWire
from ._models import CollectionPageWire
from ._models import CollectionWire
from ._models import DocumentDeleteWire
from ._models import DocumentPageWire
from ._models import DocumentWire
from ._models import JobAggregateWire
from ._models import JobCreatedWire
from ._models import JobDocumentsPageWire
from ._models import JobDocumentWire
from ._models import QueryHitWire
from ._models import QueryResponseWire
from ._models import UploadAcceptedWire
from ._transport import NemoRetrieverError
from ._transport import NemoRetrieverHTTPError
from ._transport import _NRLTransport

logger = logging.getLogger(__name__)

_BACKEND_NAME = "nemo_retriever"
_RESOURCE_PAGE_SIZE = 100
_JOB_PAGE_SIZE = 1000
_PHYSICAL_STORAGE_KEYS = frozenset(
    {
        "database_uri",
        "lance_uri",
        "lancedb_uri",
        "physical_table",
        "table_name",
        "table_path",
        "vdb_uri",
    }
)
_SUCCESS_STATUSES = frozenset({"completed", "indexed", "ready", "success", "succeeded"})
_FAILED_STATUSES = frozenset({"failed", "error"})
_PLACEHOLDER_SUMMARY = "No Summary Available"
# Shared with the other knowledge backends so one setting paces every TTL cleanup thread.
TTL_CLEANUP_INTERVAL_SECONDS = int(os.environ.get("AIQ_TTL_CLEANUP_INTERVAL_SECONDS", "3600"))


@dataclass(frozen=True)
class _Settings:
    base_url: str
    api_token: str | None
    scope: str
    connect_timeout_s: float
    request_timeout_s: float
    max_retries: int
    max_concurrency: int
    verify_ssl: bool
    ca_bundle: str | None
    collection_ttl_hours: float
    warm_start: bool
    start_ttl_cleanup: bool


@dataclass(frozen=True)
class _FileDescriptor:
    position: int
    path: Path
    filename: str
    file_size: int
    content_sha256: str
    manifest_entry_id: str


_SHARED_TRANSPORTS: dict[tuple[Any, ...], _NRLTransport] = {}
_SHARED_TRANSPORTS_LOCK = threading.Lock()


def _secret_value(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "get_secret_value"):
        value = value.get_secret_value()
    normalized = str(value).strip()
    return normalized or None


def _settings(config: dict[str, Any]) -> _Settings:
    base_url = str(config.get("base_url") or "http://127.0.0.1:7670").strip().rstrip("/")
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("nrl_base_url must be an absolute HTTP(S) URL")
    scope = str(config.get("scope") or "").strip()
    if not scope:
        raise ValueError("nemo_retriever requires an explicit nrl_scope")
    connect_timeout_s = float(config.get("connect_timeout_s", 30))
    request_timeout_s = float(config.get("request_timeout_s", 300))
    max_retries = int(config.get("max_retries", 5))
    max_concurrency = int(config.get("max_concurrency", 8))
    collection_ttl_hours = float(config.get("collection_ttl_hours", 24))
    if connect_timeout_s <= 0 or request_timeout_s <= 0:
        raise ValueError("NeMo Retriever timeouts must be greater than zero")
    if max_retries < 0:
        raise ValueError("nrl_max_retries must be zero or greater")
    if max_concurrency < 1:
        raise ValueError("nrl_max_concurrency must be at least one")
    if collection_ttl_hours <= 0:
        raise ValueError("nrl_collection_ttl_hours must be greater than zero")
    return _Settings(
        base_url=base_url,
        api_token=_secret_value(config.get("api_token")),
        scope=scope,
        connect_timeout_s=connect_timeout_s,
        request_timeout_s=request_timeout_s,
        max_retries=max_retries,
        max_concurrency=max_concurrency,
        verify_ssl=bool(config.get("verify_ssl", True)),
        ca_bundle=str(config["ca_bundle"]) if config.get("ca_bundle") else None,
        collection_ttl_hours=collection_ttl_hours,
        warm_start=bool(config.get("warm_start", True)),
        start_ttl_cleanup=bool(config.get("start_ttl_cleanup", True)),
    )


def _transport_for(config: dict[str, Any], settings: _Settings) -> _NRLTransport:
    injected = config.get("_transport")
    if injected is not None:
        return injected
    token_fingerprint = hashlib.sha256((settings.api_token or "").encode()).hexdigest()
    key = (
        settings.base_url,
        settings.scope,
        token_fingerprint,
        settings.connect_timeout_s,
        settings.request_timeout_s,
        settings.max_retries,
        settings.verify_ssl,
        settings.ca_bundle,
    )
    with _SHARED_TRANSPORTS_LOCK:
        transport = _SHARED_TRANSPORTS.get(key)
        if transport is None:
            transport = _NRLTransport(
                base_url=settings.base_url,
                scope=settings.scope,
                api_token=settings.api_token,
                connect_timeout_s=settings.connect_timeout_s,
                request_timeout_s=settings.request_timeout_s,
                max_retries=settings.max_retries,
                verify_ssl=settings.verify_ssl,
                ca_bundle=settings.ca_bundle,
            )
            _SHARED_TRANSPORTS[key] = transport
        return transport


def _wire(model: Any, payload: Any, operation: str) -> Any:
    try:
        return model.model_validate(payload)
    except (ValidationError, ValueError, TypeError) as error:
        raise NemoRetrieverError(
            f"NeMo Retriever {operation} response did not match the public API contract"
        ) from error


def _resource_path(value: str) -> str:
    return quote(value, safe="")


def _parse_timestamp(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def _scrub_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            normalized = str(key).lower()
            if normalized in _PHYSICAL_STORAGE_KEYS or "lancedb" in normalized:
                continue
            result[str(key)] = _scrub_metadata(item)
        return result
    if isinstance(value, list):
        return [_scrub_metadata(item) for item in value]
    if isinstance(value, tuple):
        return [_scrub_metadata(item) for item in value]
    return value


def _normalize_content_type(value: str | None) -> ContentType:
    normalized = (value or "").strip().lower()
    if "table" in normalized:
        return ContentType.TABLE
    if "chart" in normalized or "plot" in normalized or "graph" in normalized:
        return ContentType.CHART
    if any(token in normalized for token in ("image", "figure", "graphic")):
        return ContentType.IMAGE
    return ContentType.TEXT


def _status_to_file_status(status: str) -> FileStatus:
    normalized = status.lower()
    if normalized in _SUCCESS_STATUSES:
        return FileStatus.SUCCESS
    if normalized in _FAILED_STATUSES:
        return FileStatus.FAILED
    if normalized == "pending":
        return FileStatus.UPLOADING
    return FileStatus.INGESTING


def _safe_structured_data(metadata: dict[str, Any]) -> str | None:
    for key in ("structured_data", "table_content", "chart_data"):
        value = metadata.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return None


def _bbox(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _http_url(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    parsed = urlparse(value)
    return value if parsed.scheme in {"http", "https"} and parsed.netloc else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _descriptors(file_paths: list[str], original_filenames: list[str]) -> list[_FileDescriptor]:
    if not file_paths:
        raise ValueError("At least one file is required for NeMo Retriever ingestion")
    descriptors: list[_FileDescriptor] = []
    for position, raw_path in enumerate(file_paths):
        path = Path(raw_path)
        if not path.is_file():
            raise ValueError(f"Ingestion file does not exist or is not a regular file: {path}")
        supplied_name = original_filenames[position] if position < len(original_filenames) else path.name
        filename = Path(str(supplied_name)).name or path.name
        content_sha256 = _sha256_file(path)
        manifest_entry_id = hashlib.sha256(f"{position}\0{filename}\0{content_sha256}".encode()).hexdigest()
        descriptors.append(
            _FileDescriptor(
                position=position,
                path=path,
                filename=filename,
                file_size=path.stat().st_size,
                content_sha256=content_sha256,
                manifest_entry_id=manifest_entry_id,
            )
        )
    return descriptors


def _idempotency_key(collection_name: str, descriptors: list[_FileDescriptor]) -> str:
    canonical = {
        "collection_name": collection_name,
        "documents": [
            {
                "position": item.position,
                "filename": item.filename,
                "content_sha256": item.content_sha256,
            }
            for item in descriptors
        ],
    }
    encoded = json.dumps(canonical, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return f"aiq-{hashlib.sha256(encoded).hexdigest()}"


def _collection_info(item: CollectionWire) -> CollectionInfo:
    metadata = _scrub_metadata(item.metadata)
    metadata.update({"nrl_status": item.status, "expires_at": item.expires_at.isoformat() if item.expires_at else None})
    return CollectionInfo(
        name=item.name,
        description=item.description,
        backend=_BACKEND_NAME,
        created_at=_parse_timestamp(item.created_at),
        updated_at=_parse_timestamp(item.updated_at),
        metadata=metadata,
    )


def _collection_expiration(info: CollectionInfo) -> datetime | None:
    """Read back the expiration that ``_collection_info`` carries as an ISO string in metadata."""
    raw = info.metadata.get("expires_at")
    if not raw:
        return None
    try:
        return _parse_timestamp(datetime.fromisoformat(str(raw)))
    except ValueError:
        logger.warning("NeMo Retriever reported a collection expiration that could not be parsed")
        return None


def _document_info(item: DocumentWire) -> FileInfo:
    status = _status_to_file_status(item.status)
    return FileInfo(
        file_id=item.document_id,
        file_name=item.filename,
        collection_name=item.collection_name,
        status=status,
        chunk_count=item.chunk_count,
        uploaded_at=_parse_timestamp(item.created_at),
        ingested_at=_parse_timestamp(item.updated_at) if status == FileStatus.SUCCESS else None,
        error_message=item.error,
        metadata={
            "content_sha256": item.content_sha256,
            "document_version": item.document_version,
            "job_id": item.job_id,
            "nrl_status": item.status,
        },
    )


@register_retriever(_BACKEND_NAME)
class NemoRetrieverRetriever(BaseRetriever):
    """Retrieve citation-ready chunks through NeMo Retriever's public REST API."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self._settings = _settings(self.config)
        self._transport = _transport_for(self.config, self._settings)

    @property
    def backend_name(self) -> str:
        return _BACKEND_NAME

    async def retrieve(
        self,
        query: str,
        collection_name: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        if filters:
            return RetrievalResult(
                query=query,
                backend=_BACKEND_NAME,
                chunks=[],
                success=False,
                error_message="NeMo Retriever metadata filters are not supported by the public query contract",
            )
        try:
            payload = await self._transport.arequest_json(
                "POST",
                "/v1/query",
                operation="query",
                json={"query": query, "collection_name": collection_name, "top_k": top_k},
            )
            response = _wire(QueryResponseWire, payload, "query")
            if len(response.results) != 1:
                raise NemoRetrieverError("NeMo Retriever query returned an unexpected number of result sets")
            chunks = [self.normalize(hit) for hit in response.results[0].hits]
            return RetrievalResult(
                query=query,
                backend=_BACKEND_NAME,
                chunks=chunks,
                total_tokens=sum(max(1, len(chunk.content) // 4) for chunk in chunks if chunk.content),
                success=True,
            )
        except NemoRetrieverError as error:
            logger.warning("NeMo Retriever query failed: %s", error)
            return RetrievalResult(
                query=query,
                backend=_BACKEND_NAME,
                chunks=[],
                success=False,
                error_message=str(error),
            )

    def normalize(self, raw_result: Any) -> Chunk:
        hit = raw_result if isinstance(raw_result, QueryHitWire) else _wire(QueryHitWire, raw_result, "query hit")
        metadata = _scrub_metadata(hit.metadata)
        bounding_box = _bbox(hit.bbox if hit.bbox is not None else hit.bbox_xyxy_norm)
        metadata.update(
            {
                "document_id": hit.document_id,
                "source": _scrub_metadata(hit.source),
                "source_id": hit.source_id,
                "bounding_box": bounding_box,
            }
        )
        content_type = _normalize_content_type(hit.content_type)
        page_number = hit.page_number
        citation = f"{hit.filename}, p.{page_number}" if page_number else hit.filename
        image_storage_uri = hit.stored_image_uri or None
        image_url = _http_url(image_storage_uri)
        if image_url is None and content_type == ContentType.IMAGE:
            image_url = _http_url(hit.source)
        return Chunk(
            chunk_id=hit.chunk_id,
            content=hit.text or "",
            distance=hit.distance,
            file_name=hit.filename,
            page_number=page_number,
            display_citation=citation,
            content_type=content_type,
            content_subtype=hit.content_type if hit.content_type and hit.content_type != content_type.value else None,
            structured_data=_safe_structured_data(metadata),
            image_storage_uri=image_storage_uri,
            image_url=image_url,
            metadata=metadata,
        )

    async def health_check(self) -> bool:
        try:
            payload = await self._transport.arequest_json("GET", "/v1/health", operation="health check")
            return isinstance(payload, dict) and payload.get("status") in {"ok", "healthy"}
        except NemoRetrieverError:
            return False


@register_ingestor(_BACKEND_NAME)
class NemoRetrieverIngestor(TTLCleanupMixin, BaseIngestor):
    """Manage NRL collections and accepted ingestion jobs through REST."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self._settings = _settings(self.config)
        self._transport = _transport_for(self.config, self._settings)
        self._tracking_lock = threading.Lock()
        self._accepted_by_job: dict[str, list[UploadAcceptedWire]] = {}
        self._file_jobs: dict[str, str] = {}
        self._file_sizes: dict[str, int] = {}
        self._job_collections: dict[str, str] = {}
        # Needed because we want the summary to show the filename, but we work with document_ids
        self._document_id_to_filename: dict[str, str] = {}
        self._summarized: defaultdict[str, set[str]] = defaultdict(set)
        self._collection_expirations: dict[str, datetime] = {}
        if self._settings.warm_start:
            self._start_warm_start_task()
        if self._settings.start_ttl_cleanup:
            self._start_ttl_cleanup_task(self._settings.collection_ttl_hours, TTL_CLEANUP_INTERVAL_SECONDS)

    def _start_warm_start_task(self) -> None:
        """Reconcile on a daemon thread so startup neither blocks on NRL nor fails with it."""
        thread = threading.Thread(
            target=self._warm_start,
            daemon=True,
            name=f"{self.backend_name}-warm-start",
        )
        thread.start()

    def _warm_start(self) -> None:
        """Reconcile with NRL, retire collections that expired while down, and warm the caches."""
        try:
            self._reconcile_summaries()
            self._cleanup_expired_collections()
        except Exception:
            logger.exception("NeMo Retriever warm start did not complete")

    def _cleanup_expired_collections(self) -> None:
        """Drop the summaries and cached state of collections whose NRL expiration has passed.

        NRL expires the collection itself, so cleanup never issues a delete: it only retires the
        state AI-Q keeps alongside it, which would otherwise keep offering agents documents that
        can no longer be retrieved.

        It also replaces the mixin's idle-time policy: NRL is given an absolute expiration at
        creation and enforces it, so inferring one from ``updated_at`` would push AI-Q's view of a
        collection past the deadline the service will actually act on.

        Each deadline is the one NRL last reported, recorded at create, update, or reconcile time.
        """
        now = datetime.now(UTC)
        with self._tracking_lock:
            due = [name for name, expires_at in self._collection_expirations.items() if expires_at <= now]
        for name in due:
            self._forget_collection(name)
            logger.info("Dropped summaries for an expired NeMo Retriever collection")

    def _track_expiration(self, name: str, expires_at: datetime | None) -> None:
        """Record when NRL will expire a collection, or stop tracking one that no longer expires."""
        with self._tracking_lock:
            if expires_at is None:
                self._collection_expirations.pop(name, None)
            else:
                self._collection_expirations[name] = expires_at

    def _forget_collection(self, name: str) -> None:
        """Drop a collection's summaries and the adapter state that tracked its documents."""
        clear_collection_summaries(name)
        with self._tracking_lock:
            self._collection_expirations.pop(name, None)
            for document_id in self._summarized.pop(name, set()):
                self._document_id_to_filename.pop(document_id, None)

    def _reconcile_summaries(self) -> None:
        """Make the summary store match the documents NRL actually holds.

        NRL owns document lifetime, including server-side collection expiration, so it is the
        authority here: documents it serves must have a summary row, and rows it no longer backs
        are dropped. A collection is cleared only once NRL positively reports it absent, so a
        transport failure leaves the store untouched instead of deleting live summaries. The
        expirations NRL reports are adopted as well, which is what lets TTL cleanup act on
        collections this process did not create itself.
        """
        collections = self.list_collections()
        for collection in collections:
            self._track_expiration(collection.name, _collection_expiration(collection))
        live_collections = {collection.name for collection in collections}
        for name in live_collections | set(list_summary_collections()):
            try:
                if name in live_collections:
                    self._reconcile_collection(name)
                elif self.get_collection(name) is None:
                    self._forget_collection(name)
                    logger.info("Dropped summaries for a collection NeMo Retriever no longer holds")
            except NemoRetrieverError:
                logger.warning("Skipped summary reconciliation for one collection (NeMo Retriever unavailable)")

    def _reconcile_collection(self, collection_name: str) -> None:
        """Reconcile one collection that NRL still holds against its stored summaries.

        The store is read before NRL so that a document ingested mid-reconcile is treated as new
        rather than as an orphan: an extra idempotent write is harmless, a wrong delete is not.

        Every document NRL lists gets its filename mapped, because a delete can arrive for one that
        is still ingesting. Only successfully ingested documents count as summarized.
        """
        stored = {document.file_name for document in get_available_documents(collection_name)}
        documents = self.list_files(collection_name)
        known: set[str] = set()
        for document in documents:
            filename = document.file_name or document.file_id
            known.add(filename)
            with self._tracking_lock:
                self._document_id_to_filename[document.file_id] = filename
            if document.status != FileStatus.SUCCESS:
                continue
            if filename not in stored:
                self._register_summary(
                    collection_name, document.status, document.file_id, filename, _PLACEHOLDER_SUMMARY
                )
            else:
                with self._tracking_lock:
                    self._summarized[collection_name].add(document.file_id)
        for orphan in stored - known:
            unregister_summary(collection_name, orphan)

    @property
    def backend_name(self) -> str:
        return _BACKEND_NAME

    def create_collection(
        self,
        name: str,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CollectionInfo:
        expires_at = datetime.now(UTC) + timedelta(hours=self._settings.collection_ttl_hours)
        payload = self._transport.request_json(
            "POST",
            "/v1/collections",
            operation="create collection",
            json={
                "name": name,
                "description": description,
                "metadata": _scrub_metadata(metadata or {}),
                "expires_at": expires_at.isoformat(),
            },
        )
        info = _collection_info(_wire(CollectionWire, payload, "create collection"))
        self._track_expiration(name, _collection_expiration(info) or expires_at)
        return info

    def update_collection(
        self,
        name: str,
        *,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
        expires_at: datetime | None = None,
    ) -> CollectionInfo:
        body: dict[str, Any] = {}
        if description is not None:
            body["description"] = description
        if metadata is not None:
            body["metadata"] = _scrub_metadata(metadata)
        if expires_at is not None:
            body["expires_at"] = expires_at.isoformat()
        payload = self._transport.request_json(
            "PATCH",
            f"/v1/collections/{_resource_path(name)}",
            operation="update collection",
            json=body,
        )
        info = _collection_info(_wire(CollectionWire, payload, "update collection"))
        self._track_expiration(name, _collection_expiration(info) or expires_at)
        return info

    def delete_collection(self, name: str) -> bool:
        payload = self._transport.request_json(
            "DELETE",
            f"/v1/collections/{_resource_path(name)}",
            operation="delete collection",
            params={"if_exists": "true"},
        )
        result = _wire(CollectionDeleteWire, payload, "delete collection")
        self._forget_collection(name)
        return result.deleted or not result.existed or result.cleanup_pending

    def list_collections(self) -> list[CollectionInfo]:
        items: list[CollectionInfo] = []
        token: str | None = None
        seen_tokens: set[str] = set()
        while True:
            params: dict[str, Any] = {"limit": _RESOURCE_PAGE_SIZE}
            if token:
                params["continuation_token"] = token
            payload = self._transport.request_json(
                "GET", "/v1/collections", operation="list collections", params=params
            )
            page = _wire(CollectionPageWire, payload, "list collections")
            items.extend(_collection_info(item) for item in page.items)
            if not page.next_token:
                return items
            if page.next_token in seen_tokens:
                raise NemoRetrieverError("NeMo Retriever collection pagination repeated a continuation token")
            seen_tokens.add(page.next_token)
            token = page.next_token

    def get_collection(self, name: str) -> CollectionInfo | None:
        try:
            payload = self._transport.request_json(
                "GET", f"/v1/collections/{_resource_path(name)}", operation="get collection"
            )
        except NemoRetrieverHTTPError as error:
            if error.status_code == 404:
                return None
            raise
        return _collection_info(_wire(CollectionWire, payload, "get collection"))

    def submit_job(
        self,
        file_paths: list[str],
        collection_name: str,
        config: dict[str, Any] | None = None,
    ) -> str:
        job_config = dict(config or {})
        originals = [str(value) for value in job_config.get("original_filenames", [])]
        cleanup_files = bool(job_config.get("cleanup_files", False))
        descriptors = _descriptors(file_paths, originals)
        manifest = [
            {
                "manifest_entry_id": item.manifest_entry_id,
                "filename": item.filename,
                "content_sha256": item.content_sha256,
            }
            for item in descriptors
        ]
        idempotency_key = str(job_config.get("idempotency_key") or _idempotency_key(collection_name, descriptors))
        created_payload = self._transport.request_json(
            "POST",
            "/v1/ingest/job",
            operation="job creation",
            json={
                "expected_documents": len(descriptors),
                "collection_name": collection_name,
                "operation": "append",
                "retain_results": False,
                "idempotency_key": idempotency_key,
                "document_manifest": manifest,
            },
        )
        created = _wire(JobCreatedWire, created_payload, "job creation")
        file_metadata = _scrub_metadata(job_config.get("metadata") or {})

        def upload(item: _FileDescriptor) -> UploadAcceptedWire:
            metadata: dict[str, Any] = {"filename": item.filename}
            if file_metadata:
                metadata["metadata"] = file_metadata
            payload = self._transport.upload_document(
                job_id=created.job_id,
                file_path=item.path,
                filename=item.filename,
                manifest_entry_id=item.manifest_entry_id,
                metadata=json.dumps(metadata, separators=(",", ":"), sort_keys=True),
            )
            return _wire(UploadAcceptedWire, payload, "job document upload")

        accepted_by_position: dict[int, UploadAcceptedWire] = {}
        try:
            with ThreadPoolExecutor(max_workers=min(self._settings.max_concurrency, len(descriptors))) as executor:
                futures = {executor.submit(upload, item): item.position for item in descriptors}
                for future in as_completed(futures):
                    accepted_by_position[futures[future]] = future.result()
        finally:
            if cleanup_files:
                for item in descriptors:
                    try:
                        item.path.unlink(missing_ok=True)
                    except OSError:
                        logger.warning("Failed to remove temporary upload file %s", item.path)

        accepted = [accepted_by_position[index] for index in range(len(descriptors))]
        sizes = {item.position: item.file_size for item in descriptors}
        with self._tracking_lock:
            self._accepted_by_job[created.job_id] = accepted
            self._job_collections[created.job_id] = collection_name
            for position, item in enumerate(accepted):
                self._file_jobs[item.document_id] = created.job_id
                self._file_sizes[item.document_id] = sizes[position]
        return created.job_id

    def get_job_status(self, job_id: str) -> IngestionJobStatus:
        payload = self._transport.request_json(
            "GET", f"/v1/ingest/job/{_resource_path(job_id)}", operation="job status"
        )
        aggregate = _wire(JobAggregateWire, payload, "job status")
        documents = self._list_job_documents(job_id)
        state = {
            "pending": JobState.PENDING,
            "processing": JobState.PROCESSING,
            "completed": JobState.COMPLETED,
            "failed": JobState.FAILED,
            "partial_success": JobState.COMPLETED,
        }.get(aggregate.status.lower(), JobState.PROCESSING)
        file_details: list[FileProgress] = []
        attempt_ids: dict[str, str] = {}
        terminal = 0
        for item in documents:
            status = _status_to_file_status(item.status)
            if status in {FileStatus.SUCCESS, FileStatus.FAILED}:
                terminal += 1
            attempt_ids[item.document_id] = item.attempt_id
            # Always register at least the filename. We may want to gate this behind the generate_summary flag later.
            # TODO: Get the summary for the document
            self._register_summary(
                aggregate.collection_name,
                status,
                item.document_id,
                item.filename,
            )
            file_details.append(
                FileProgress(
                    file_id=item.document_id,
                    file_name=item.filename or item.document_id,
                    status=status,
                    progress_percent=100.0 if status in {FileStatus.SUCCESS, FileStatus.FAILED} else 50.0,
                    error_message=item.error,
                    chunks_created=max(0, item.result_rows or 0),
                )
            )
        collection_name = aggregate.collection_name
        if not collection_name:
            with self._tracking_lock:
                collection_name = self._job_collections.get(job_id, "")
        error_message = None
        if aggregate.status.lower() == "failed":
            error_message = "NeMo Retriever ingestion job failed"
        elif aggregate.status.lower() == "partial_success":
            error_message = "NeMo Retriever ingestion job completed with one or more failed documents"
        return IngestionJobStatus(
            job_id=aggregate.job_id,
            status=state,
            submitted_at=_parse_timestamp(aggregate.created_at) or datetime.now(UTC),
            started_at=_parse_timestamp(aggregate.started_at),
            completed_at=_parse_timestamp(aggregate.finalized_at),
            total_files=aggregate.expected_documents,
            processed_files=terminal,
            file_details=file_details,
            collection_name=collection_name or "",
            backend=_BACKEND_NAME,
            error_message=error_message,
            metadata={
                "nrl_status": aggregate.status,
                "counts": dict(aggregate.counts),
                "operation": aggregate.operation,
                "trace_id": aggregate.trace_id,
                "attempt_ids": attempt_ids,
            },
        )

    def _list_job_documents(self, job_id: str) -> list[JobDocumentWire]:
        items: list[JobDocumentWire] = []
        offset = 0
        while True:
            payload = self._transport.request_json(
                "GET",
                f"/v1/ingest/job/{_resource_path(job_id)}/documents",
                operation="job document listing",
                params={"offset": offset, "limit": _JOB_PAGE_SIZE},
            )
            page = _wire(JobDocumentsPageWire, payload, "job document listing")
            items.extend(page.items)
            offset += len(page.items)
            if offset >= page.total_filtered:
                return items
            if not page.items:
                raise NemoRetrieverError("NeMo Retriever job document pagination made no progress")

    def upload_file(
        self,
        file_path: str,
        collection_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> FileInfo:
        job_id = self.submit_job(
            [file_path],
            collection_name,
            config={"original_filenames": [Path(file_path).name], "metadata": metadata or {}},
        )
        with self._tracking_lock:
            accepted = list(self._accepted_by_job.get(job_id, []))
        if not accepted:
            raise NemoRetrieverError("NeMo Retriever accepted the job without a document identifier")
        item = accepted[0]
        return FileInfo(
            file_id=item.document_id,
            file_name=Path(file_path).name,
            collection_name=collection_name,
            status=_status_to_file_status(item.status),
            file_size=Path(file_path).stat().st_size,
            uploaded_at=_parse_timestamp(item.created_at),
            metadata={
                "job_id": job_id,
                "attempt_id": item.attempt_id,
                "content_sha256": item.content_sha256,
                "nrl_status": item.status,
            },
        )

    def delete_file(self, file_id: str, collection_name: str) -> bool:
        payload = self._transport.request_json(
            "DELETE",
            f"/v1/collections/{_resource_path(collection_name)}/documents/{_resource_path(file_id)}",
            operation="delete document",
            params={"if_exists": "true"},
        )
        result = _wire(DocumentDeleteWire, payload, "delete document")
        with self._tracking_lock:
            summary_key = self._document_id_to_filename.get(file_id, file_id)
        unregister_summary(collection_name, summary_key)
        with self._tracking_lock:
            self._document_id_to_filename.pop(file_id, None)
            self._summarized[collection_name].discard(file_id)
        return result.deleted or not result.existed or result.cleanup_pending

    def list_files(self, collection_name: str) -> list[FileInfo]:
        items: list[FileInfo] = []
        token: str | None = None
        seen_tokens: set[str] = set()
        while True:
            params: dict[str, Any] = {"limit": _RESOURCE_PAGE_SIZE}
            if token:
                params["continuation_token"] = token
            payload = self._transport.request_json(
                "GET",
                f"/v1/collections/{_resource_path(collection_name)}/documents",
                operation="list documents",
                params=params,
            )
            page = _wire(DocumentPageWire, payload, "list documents")
            items.extend(_document_info(item) for item in page.items)
            if not page.next_token:
                return items
            if page.next_token in seen_tokens:
                raise NemoRetrieverError("NeMo Retriever document pagination repeated a continuation token")
            seen_tokens.add(page.next_token)
            token = page.next_token

    def get_file_status(self, file_id: str, collection_name: str) -> FileInfo | None:
        try:
            payload = self._transport.request_json(
                "GET",
                f"/v1/collections/{_resource_path(collection_name)}/documents/{_resource_path(file_id)}",
                operation="get document",
            )
            return _document_info(_wire(DocumentWire, payload, "get document"))
        except NemoRetrieverHTTPError as error:
            if error.status_code != 404:
                raise
        with self._tracking_lock:
            job_id = self._file_jobs.get(file_id)
            file_size = self._file_sizes.get(file_id)
        if not job_id:
            return None
        for item in self._list_job_documents(job_id):
            if item.document_id != file_id:
                continue
            status = _status_to_file_status(item.status)
            return FileInfo(
                file_id=item.document_id,
                file_name=item.filename or item.document_id,
                collection_name=item.collection_name or collection_name,
                status=status,
                file_size=file_size,
                chunk_count=max(0, item.result_rows or 0),
                uploaded_at=_parse_timestamp(item.submitted_at),
                ingested_at=_parse_timestamp(item.completed_at) if status == FileStatus.SUCCESS else None,
                error_message=item.error,
                metadata={
                    "job_id": item.job_id,
                    "attempt_id": item.attempt_id,
                    "content_sha256": item.content_sha256,
                    "nrl_status": item.status,
                },
            )
        return None

    async def health_check(self) -> bool:
        try:
            payload = await self._transport.arequest_json("GET", "/v1/health", operation="health check")
            return isinstance(payload, dict) and payload.get("status") in {"ok", "healthy"}
        except NemoRetrieverError:
            return False

    def _register_summary(
        self,
        collection_name: str,
        status: FileStatus,
        document_id: str,
        filename: str | None,
        summary: str | None = None,
    ) -> None:
        """Register a summary for a document if it has been successfully ingested."""
        if status == FileStatus.SUCCESS and document_id not in self._summarized[collection_name]:
            summary = summary or _PLACEHOLDER_SUMMARY
            logger.info(f"registering summary for {filename or document_id}")
            filename = filename or document_id
            with self._tracking_lock:
                self._document_id_to_filename[document_id] = filename
                self._summarized[collection_name].add(document_id)
            register_summary(collection_name, filename, summary)
