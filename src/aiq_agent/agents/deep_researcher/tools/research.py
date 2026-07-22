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

"""Researcher runnable and batched research tool construction."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from contextvars import ContextVar
from typing import Any
from typing import cast

from langchain.agents.structured_output import StructuredOutputValidationError
from langchain.tools import ToolRuntime
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langchain_core.tools import tool

from ..models import ResearchNotes
from ..models import ResearchQuery

_NO_TOOL_RUNTIME = cast(ToolRuntime, None)
logger = logging.getLogger(__name__)
_NOTE_SLUG_MAX_LENGTH = 64
_MAX_STRUCTURED_OUTPUT_CORRECTIONS = 2
_STRUCTURED_OUTPUT_ERROR_MAX_CHARS = 4000
_structured_output_correction_state: ContextVar[dict[str, Any] | None] = ContextVar(
    "researcher_structured_output_correction_state",
    default=None,
)
RESEARCHER_AGENT_NAME = "researcher-agent"


def _salvage_textual_research_notes(error: StructuredOutputValidationError) -> ResearchNotes | None:
    """Return valid non-quantitative notes when only a dataset payload is invalid.

    The structured-output exception retains the model's exact tool arguments.
    Removing ``quantitative_datasets`` is deliberately the only repair made by
    the runtime: all textual fields still have to satisfy ``ResearchNotes``.
    """
    for tool_call in error.ai_message.tool_calls:
        if tool_call.get("name") != error.tool_name:
            continue
        arguments = tool_call.get("args")
        if not isinstance(arguments, dict) or "quantitative_datasets" not in arguments:
            continue

        textual_arguments = dict(arguments)
        textual_arguments["quantitative_datasets"] = []
        try:
            return ResearchNotes.model_validate(textual_arguments)
        except Exception:  # noqa: BLE001 - only a fully valid textual contract is salvageable
            return None
    return None


def handle_research_notes_structured_error(error: Exception) -> str:
    """Provide at most two task-local validation-feedback turns to the researcher."""
    correction_state = _structured_output_correction_state.get()
    if correction_state is None:
        raise error
    if isinstance(error, StructuredOutputValidationError):
        fallback_note = _salvage_textual_research_notes(error)
        if fallback_note is not None:
            correction_state["fallback_note"] = fallback_note
    correction_count = correction_state["count"]
    if correction_count >= _MAX_STRUCTURED_OUTPUT_CORRECTIONS:
        raise error

    correction_count += 1
    correction_state["count"] = correction_count
    validation_detail = str(error)
    if len(validation_detail) > _STRUCTURED_OUTPUT_ERROR_MAX_CHARS:
        validation_detail = f"{validation_detail[:_STRUCTURED_OUTPUT_ERROR_MAX_CHARS]}..."
    return (
        "The ResearchNotes response failed validation. "
        f"This is bounded correction {correction_count} of {_MAX_STRUCTURED_OUTPUT_CORRECTIONS}. "
        "Preserve valid textual findings and sources, correct the quantitative dataset fields, and return "
        f"ResearchNotes without doing more research. Validation error: {validation_detail}"
    )


def format_research_request(query: ResearchQuery) -> str:
    """Create the single-query researcher task text used by the batch tool."""
    query_json = json.dumps(query.model_dump(mode="json"), indent=2, ensure_ascii=False)
    return (
        "Batch research invocation. Execute this ResearchQuery and return a structured ResearchNotes response. "
        "Do not call write_file or edit_file; run_research_batch will persist the returned ResearchNotes under "
        "/shared/ after you return.\n\n"
        "ResearchQuery JSON:\n"
        f"{query_json}"
    )


def researcher_invoke_state(query: ResearchQuery, runtime: ToolRuntime | None) -> dict[str, Any]:
    """Build nested researcher state, carrying parent files for StateBackend-backed skills."""
    invoke_state: dict[str, Any] = {
        "messages": [HumanMessage(content=format_research_request(query))],
    }
    parent_state = getattr(runtime, "state", None) if runtime is not None else None
    if isinstance(parent_state, dict) and "files" in parent_state:
        invoke_state["files"] = parent_state["files"]
    return invoke_state


def researcher_invoke_config(runtime: ToolRuntime | None, callbacks: list[Any]) -> dict[str, Any]:
    """Build child-run config while preserving the active callback lineage."""
    config = dict(runtime.config) if runtime is not None else {}
    config.pop("run_id", None)
    config.pop("configurable", None)
    config["run_name"] = RESEARCHER_AGENT_NAME
    if not config.get("callbacks") and callbacks:
        config["callbacks"] = callbacks
    return config


async def _run_research_query(
    *,
    query: ResearchQuery,
    researcher_runnable: Any,
    runtime: ToolRuntime | None,
    callbacks: list[Any],
    semaphore: asyncio.Semaphore,
) -> ResearchNotes:
    """Run one researcher worker and return its structured notes."""
    async with semaphore:
        correction_state: dict[str, Any] = {"count": 0, "fallback_note": None}
        correction_token = _structured_output_correction_state.set(correction_state)
        try:
            try:
                result = await researcher_runnable.ainvoke(
                    researcher_invoke_state(query, runtime),
                    config=researcher_invoke_config(runtime, callbacks),
                )
            except StructuredOutputValidationError as exc:
                note = _salvage_textual_research_notes(exc) or correction_state["fallback_note"]
                if note is not None:
                    logger.warning(
                        "Omitting invalid quantitative dataset publication after bounded structured-output corrections "
                        "for query %r",
                        query.query,
                    )
                    return note
                raise RuntimeError(f"researcher worker failed for query {query.query!r}: {exc}") from exc
            except Exception as exc:  # noqa: BLE001 - captured as per-item failure
                raise RuntimeError(f"researcher worker failed for query {query.query!r}: {exc}") from exc

            try:
                structured = result.get("structured_response") if isinstance(result, dict) else None
                if structured is None:
                    raise ValueError("researcher worker did not return structured ResearchNotes")
                note = ResearchNotes.model_validate(structured)
            except Exception as exc:  # noqa: BLE001 - captured as per-item failure
                raise ValueError(
                    f"researcher worker returned invalid ResearchNotes for query {query.query!r}: {exc}"
                ) from exc

            return note
        finally:
            _structured_output_correction_state.reset(correction_token)


def _research_note_slug(text: str) -> str:
    """Return a compact filesystem-safe slug for a research note."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    slug = slug[:_NOTE_SLUG_MAX_LENGTH].strip("_")
    return slug or "research_note"


def _research_note_path(query: ResearchQuery, note: ResearchNotes, index: int) -> str:
    """Build a stable /shared path for a returned research note."""
    digest_input = json.dumps(query.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha1(digest_input.encode("utf-8")).hexdigest()[:8]
    slug = _research_note_slug(note.query_topic or query.query)
    return f"/shared/research_note_{index:02d}_{slug}_{digest}.json"


def _research_note_files(queries: list[ResearchQuery], notes: list[ResearchNotes]) -> list[tuple[str, bytes]]:
    """Serialize returned research notes as shared JSON files."""
    return [
        (
            _research_note_path(query, note, index),
            json.dumps(note.model_dump(mode="json", exclude_none=True), indent=2, ensure_ascii=False).encode("utf-8"),
        )
        for index, (query, note) in enumerate(zip(queries, notes, strict=False), start=1)
    ]


def _persist_research_notes(
    *,
    backend: Any | None,
    queries: list[ResearchQuery],
    notes: list[ResearchNotes],
) -> bool:
    """Persist returned ResearchNotes into parent /shared state.

    Returns whether the notes were durably handed to the configured backend.
    """
    if backend is None or not notes:
        return False

    note_files = _research_note_files(queries, notes)
    responses = list(backend.upload_files(note_files))
    if len(responses) != len(note_files):
        raise RuntimeError("failed to persist every research note file")
    errors = [f"{response.path}: {response.error}" for response in responses if getattr(response, "error", None)]
    if errors:
        raise RuntimeError(f"failed to persist research note file(s): {'; '.join(errors)}")
    return True


def _canonical_dataset_digests(notes: list[ResearchNotes]) -> list[str]:
    """Recompute and verify canonical CSV digests at the publication trust boundary."""
    digests: list[str] = []
    for note in notes:
        for dataset in note.quantitative_datasets:
            runtime_digest = hashlib.sha256(dataset.csv_text.encode("utf-8")).hexdigest()
            if dataset.csv_sha256 != runtime_digest:
                raise ValueError("canonical dataset digest no longer matches its validated csv_text")
            digests.append(runtime_digest)
    return digests


def _register_canonical_dataset_digests(
    *,
    artifact_manager: Any | None,
    notes: list[ResearchNotes],
    expected_digests: list[str] | None = None,
) -> None:
    """Register reverified canonical CSV digests with the job artifact manager."""
    if artifact_manager is None:
        return
    digests = _canonical_dataset_digests(notes)
    if expected_digests is not None and digests != expected_digests:
        raise ValueError("canonical dataset digests changed before artifact registration")
    if digests:
        artifact_manager.register_canonical_digests(digests)


async def _run_research_queries(
    *,
    queries: list[ResearchQuery],
    researcher_runnable: Any,
    runtime: ToolRuntime | None,
    callbacks: list[Any],
    max_concurrency: int,
) -> tuple[list[ResearchQuery], list[ResearchNotes], list[str]]:
    """Run researcher workers concurrently and collect successful query/note pairs plus surfaced errors."""
    semaphore = asyncio.Semaphore(min(max_concurrency, len(queries)))
    raw_results = await asyncio.gather(
        *(
            _run_research_query(
                query=query,
                researcher_runnable=researcher_runnable,
                runtime=runtime,
                callbacks=callbacks,
                semaphore=semaphore,
            )
            for query in queries
        ),
        return_exceptions=True,
    )

    successful_queries: list[ResearchQuery] = []
    notes: list[ResearchNotes] = []
    errors: list[str] = []
    for query, raw_result in zip(queries, raw_results, strict=False):
        if isinstance(raw_result, BaseException):
            error = str(raw_result) or raw_result.__class__.__name__
            errors.append(f"{query.query}: {error}")
        else:
            successful_queries.append(query)
            notes.append(raw_result)
    return successful_queries, notes, errors


def build_research_batch_tool(
    *,
    researcher_runnable: Any,
    callbacks: list[Any],
    max_research_concurrency: int,
    backend: Any | None = None,
    source_registry_middleware: Any | None = None,
    artifact_manager: Any | None = None,
) -> BaseTool:
    """Build an orchestrator-only tool that runs researcher tasks concurrently."""

    @tool
    async def run_research_batch(
        queries: list[ResearchQuery],
        runtime: ToolRuntime = _NO_TOOL_RUNTIME,
    ) -> str:
        """Run planned research queries in parallel and return ResearchNotes JSON."""
        if not queries:
            return "[]"

        if len(queries) > max_research_concurrency:
            raise ValueError(
                f"run_research_batch accepts at most {max_research_concurrency} curated queries. "
                f"Received {len(queries)}. Rank, merge, or drop lower-priority queries and call again."
            )
        successful_queries, notes, errors = await _run_research_queries(
            queries=queries,
            researcher_runnable=researcher_runnable,
            runtime=runtime,
            callbacks=callbacks,
            max_concurrency=max_research_concurrency,
        )
        if source_registry_middleware is not None:
            source_registry_middleware.register_research_note_sources(notes)
        canonical_digests = _canonical_dataset_digests(notes)
        notes_persisted = _persist_research_notes(backend=backend, queries=successful_queries, notes=notes)
        if canonical_digests and artifact_manager is not None and not notes_persisted:
            raise RuntimeError("canonical dataset digests cannot be registered before ResearchNotes persistence")
        _register_canonical_dataset_digests(
            artifact_manager=artifact_manager,
            notes=notes,
            expected_digests=canonical_digests,
        )

        if errors:
            retained_detail = ""
            if notes:
                retained_actions = []
                if source_registry_middleware is not None:
                    retained_actions.append("registered")
                if backend is not None:
                    retained_actions.append("persisted under /shared/")
                retained_text = " and ".join(retained_actions) if retained_actions else "retained"
                retained_detail = (
                    f" {len(notes)} successful researcher worker(s) were {retained_text}; "
                    "resubmit only the failed queries."
                )
            raise RuntimeError(
                f"run_research_batch failed for {len(errors)} of {len(queries)} researcher worker(s). "
                f"Errors: {'; '.join(errors)}.{retained_detail}"
            )

        return json.dumps(
            [note.model_dump(mode="json", exclude_none=True) for note in notes],
            indent=2,
            ensure_ascii=False,
        )

    return run_research_batch
