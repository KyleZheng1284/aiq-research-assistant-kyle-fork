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

"""Deep research agent using deepagents library for multi-phase workflow."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.tools import BaseTool

from aiq_agent.common import LLMProvider
from aiq_agent.common import load_prompt
from aiq_agent.common.citation_verification import EmptySourceRegistryError
from aiq_agent.common.citation_verification import sanitize_report
from aiq_agent.common.citation_verification import source_entries_from_parent_context
from aiq_agent.common.citation_verification import verify_citations

from .custom_middleware import FinalReportCommitTracker
from .custom_middleware import SourceRegistryMiddleware
from .custom_middleware import validated_research_notes_from_state
from .deepagents_runtime import DeepAgentsRuntime
from .deepagents_runtime import DeepResearchSandboxConfig
from .deepagents_runtime import DeepResearchSkillsConfig
from .factory import build_deep_research_graph
from .factory import build_deep_research_middleware_set
from .factory import build_deep_research_tool_set
from .models import DeepResearchAgentState
from .models import QuantitativeDataset
from .models import ResearchNotes
from .tools.source_tool_batching import DEFAULT_MAX_CONCURRENT_SOURCE_TOOL_CALLS
from .tools.source_tool_batching import DEFAULT_MAX_SOURCE_TOOL_BATCH_SIZE

logger = logging.getLogger(__name__)

DEFAULT_MAX_RESEARCH_CONCURRENCY = 6
DEFAULT_MAX_RESEARCHER_EXECUTE_ATTEMPTS: int | None = None
DEFAULT_MAX_WRITER_EXECUTE_ATTEMPTS: int | None = None
PARENT_REPORT_CONTEXT_PATH = "/shared/parent_report_context.json"

# Path to this agent's directory (for loading prompts)
AGENT_DIR = Path(__file__).parent

_MAX_FALLBACK_REPORT_CHARS = 1024 * 1024
_MAX_REQUIRED_TABLE_CHARS = 3 * _MAX_FALLBACK_REPORT_CHARS // 4
_MAX_REQUIRED_PUBLISHED_DATASETS = 16
_MAX_REQUIRED_UNPUBLISHED_DATASETS = 4
_FALLBACK_OMISSION_NOTICE_RESERVE = 512
_FALLBACK_TRUNCATION_MARKER = "\n\n_[Additional supporting evidence was truncated.]_"


class DeepResearcherAgent:
    """
    Deep research agent using deepagents library for multi-phase workflow.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        tools: Sequence[BaseTool] | None = None,
        *,
        verbose: bool = True,
        callbacks: list[Any] | None = None,
        domain_catalog_path: str | None = None,
        enable_source_router: bool = True,
        enable_citation_verification: bool = True,
        skills: DeepResearchSkillsConfig | None = None,
        sandbox: DeepResearchSandboxConfig | None = None,
        job_id: str | None = None,
        artifact_db_url: str | None = None,
        artifact_emit: Callable[[dict[str, Any]], None] | None = None,
        max_research_concurrency: int = DEFAULT_MAX_RESEARCH_CONCURRENCY,
        max_concurrent_source_tool_calls: int = DEFAULT_MAX_CONCURRENT_SOURCE_TOOL_CALLS,
        max_source_tool_batch_size: int = DEFAULT_MAX_SOURCE_TOOL_BATCH_SIZE,
        max_researcher_execute_attempts: int | None = DEFAULT_MAX_RESEARCHER_EXECUTE_ATTEMPTS,
        max_writer_execute_attempts: int | None = DEFAULT_MAX_WRITER_EXECUTE_ATTEMPTS,
    ) -> None:
        """
        Initialize the deep researcher agent.

        Args:
            llm_provider: LLMProvider for role-based LLM access.
            tools: Optional sequence of LangChain tools for research.
            verbose: Enable detailed logging.
            callbacks: Optional list of callbacks.
            domain_catalog_path: Optional YAML/JSON domain catalog path for source-router-agent.
            enable_source_router: Enable the advisory source-router-agent before planning.
            enable_citation_verification: Verify generated citations against the captured source registry.
            skills: Optional DeepAgents skills config.
            sandbox: Optional DeepAgents sandbox config.
            job_id: Optional async job identifier used to scope sandbox backends.
            max_research_concurrency: Maximum ResearchQuery items accepted and run concurrently per
                run_research_batch call.
            max_concurrent_source_tool_calls: Shared source-tool concurrency limit across researcher workers.
            max_source_tool_batch_size: Maximum concrete inputs per batch-capable source tool call.
            max_researcher_execute_attempts: Optional hard limit on physical researcher code-execution attempts.
            max_writer_execute_attempts: Optional hard limit on writer-side chart execution attempts,
                including physical retries. ``None`` disables each corresponding limit.
        """
        self.llm_provider = llm_provider
        self.tools = list(tools) if tools else []
        self.verbose = verbose
        self.callbacks = callbacks or []
        self.max_research_concurrency = max_research_concurrency
        self.max_concurrent_source_tool_calls = max_concurrent_source_tool_calls
        self.max_source_tool_batch_size = max_source_tool_batch_size
        self.max_researcher_execute_attempts = max_researcher_execute_attempts
        self.max_writer_execute_attempts = max_writer_execute_attempts
        self.domain_catalog_path = domain_catalog_path
        self.enable_source_router = enable_source_router
        self.enable_citation_verification = enable_citation_verification
        self.job_id = str(job_id) if job_id is not None else str(uuid4())

        self.deepagents_runtime = DeepAgentsRuntime(
            skills=skills,
            sandbox=sandbox,
            job_id=self.job_id,
            artifact_db_url=artifact_db_url,
            artifact_emit=artifact_emit,
        )

        try:
            self._prompts = self._load_prompts()
            source_tool_names = {tool.name for tool in self.tools}
            self.source_registry_middleware = SourceRegistryMiddleware(source_tool_names=source_tool_names)
            self.tool_set = build_deep_research_tool_set(
                self.tools,
                source_registry_middleware=self.source_registry_middleware,
                max_concurrent_source_tool_calls=self.max_concurrent_source_tool_calls,
                max_source_tool_batch_size=self.max_source_tool_batch_size,
            )
            self.middleware_set = build_deep_research_middleware_set(
                tool_set=self.tool_set,
                source_registry_middleware=self.source_registry_middleware,
                enable_source_router=self.enable_source_router,
                artifact_manager=self.deepagents_runtime.artifact_manager,
                max_researcher_execute_attempts=self.max_researcher_execute_attempts,
                max_writer_execute_attempts=self.max_writer_execute_attempts,
            )

            self.source_tool_names = self.tool_set.source_tool_names
            self.tools_info = self.tool_set.tools_info
            self.non_search_tools = self.tool_set.helper_tools
            self.all_tools = self.tool_set.all_tools
            self.research_source_tools = self.tool_set.research_source_tools
            self.researcher_tools = self.tool_set.researcher_tools
            self.writer_tools = self.tool_set.writer_tools
            self.researcher_middleware = self.middleware_set.researcher
            self.writer_middleware = self.middleware_set.writer
            self.orchestrator_middleware = self.middleware_set.orchestrator
            self.middleware = self.researcher_middleware
        except Exception:
            try:
                cleanup_succeeded = self.deepagents_runtime.finalize(interrupted=False)
            except Exception as cleanup_error:  # noqa: BLE001 - preserve the original construction failure
                logger.warning(
                    "Deep research runtime cleanup failed during agent construction (%s)",
                    type(cleanup_error).__name__,
                )
            else:
                if not cleanup_succeeded:
                    logger.warning("Deep research runtime cleanup reported failure during agent construction")
            raise

    def finalize(self, *, interrupted: bool) -> bool:
        """Release this request's sandbox runtime exactly once."""
        return self.deepagents_runtime.finalize(interrupted=interrupted)

    def _load_prompts(self) -> dict[str, str]:
        """Load all prompts for subagents."""
        prompts = {}
        prompt_names = ["planner", "researcher", "orchestrator", "writer", "source_router"]

        for name in prompt_names:
            prompts[name] = load_prompt(AGENT_DIR / "prompts", name)

        return prompts

    def _build_orchestrator_agent(
        self,
        state: DeepResearchAgentState,
        *,
        final_report_tracker: FinalReportCommitTracker,
    ) -> Any:
        """Build the orchestrator graph for the current state."""
        return build_deep_research_graph(
            llm_provider=self.llm_provider,
            state=state,
            prompts=self._prompts,
            tools=self.tools,
            runtime=self.deepagents_runtime,
            tool_set=self.tool_set,
            middleware_set=self.middleware_set,
            source_registry_middleware=self.source_registry_middleware,
            callbacks=self.callbacks,
            domain_catalog_path=self.domain_catalog_path,
            enable_source_router=self.enable_source_router,
            max_research_concurrency=self.max_research_concurrency,
            final_report_tracker=final_report_tracker,
            max_researcher_execute_attempts=self.max_researcher_execute_attempts,
        )

    def _extract_final_markdown(
        self,
        result: dict | Any,
        files: dict[str, Any] | None = None,
        *,
        final_report_tracker: FinalReportCommitTracker,
    ) -> str | None:
        """Extract the current run's writer commit or a validated ResearchNotes fallback."""
        # Resolve result files first, then fall back to the passed-in files (state.files) and
        # finally an empty dict. Without the explicit grouping, `or files or {}` bound only to
        # the else branch, so a dict result lacking a usable "files" key silently discarded
        # the fallback even when output files existed.
        result_files = result.get("files", None) if isinstance(result, dict) else getattr(result, "files", None)
        files = result_files or files or {}
        committed = final_report_tracker.committed_text(files)
        notes = self._validated_research_notes(files)
        required_tables = self._required_quantitative_tables(notes)
        if committed is not None:
            report = committed.strip()
            if report and (not required_tables or self._contains_all_canonical_tables(report, required_tables)):
                return report

        fallback = self._build_research_notes_fallback(files, notes=notes)
        return fallback[0] if fallback is not None else None

    @staticmethod
    def _validated_research_notes(files: object) -> list[ResearchNotes]:
        """Load schema-valid notes through the shared researcher/writer trust boundary."""
        return list(validated_research_notes_from_state({"files": files}))

    def _required_quantitative_tables(self, notes: list[ResearchNotes]) -> tuple[str, ...]:
        """Return the bounded canonical tables that the accepted report must preserve."""
        selected, _ = self._select_required_quantitative_datasets(notes)
        return tuple(dataset.markdown_table.strip() for _, dataset in selected)

    def _select_required_quantitative_datasets(
        self,
        notes: list[ResearchNotes],
    ) -> tuple[tuple[tuple[ResearchNotes, QuantitativeDataset], ...], int]:
        """Select report tables by durable publication, then relevance, within hard limits.

        Once a canonical CSV has been durably published, its runtime-trusted digest is the
        job-lifetime signal that its table belongs in the report. Before any CSV is published,
        the highest-relevance validated notes provide a deterministic fallback signal. Duplicate
        CSV digests represent the same canonical serialization and are required only once.
        """
        candidates = [
            (note_index, dataset_index, note, dataset)
            for note_index, note in enumerate(notes)
            for dataset_index, dataset in enumerate(note.quantitative_datasets)
        ]
        published_digests = self._published_canonical_digests()
        if published_digests:
            candidates = [candidate for candidate in candidates if candidate[3].csv_sha256 in published_digests]
            max_datasets = _MAX_REQUIRED_PUBLISHED_DATASETS
        else:
            candidates.sort(
                key=lambda candidate: (
                    -(
                        candidate[2].evidence_judgment.relevance_score
                        if candidate[2].evidence_judgment is not None
                        else 0
                    ),
                    candidate[0],
                    candidate[1],
                )
            )
            max_datasets = _MAX_REQUIRED_UNPUBLISHED_DATASETS

        unique_candidates: list[tuple[int, int, ResearchNotes, QuantitativeDataset]] = []
        seen_digests: set[str] = set()
        for candidate in candidates:
            digest = candidate[3].csv_sha256
            if digest in seen_digests:
                continue
            seen_digests.add(digest)
            unique_candidates.append(candidate)

        selected: list[tuple[ResearchNotes, QuantitativeDataset]] = []
        selected_chars = 0
        for _, _, note, dataset in unique_candidates:
            section_chars = len(self._canonical_table_section(dataset)) + (2 if selected else 0)
            if len(selected) >= max_datasets or selected_chars + section_chars > _MAX_REQUIRED_TABLE_CHARS:
                continue
            selected.append((note, dataset))
            selected_chars += section_chars

        return tuple(selected), len(unique_candidates) - len(selected)

    def _published_canonical_digests(self) -> frozenset[str]:
        """Read the artifact manager's job-lifetime set of published canonical digests."""
        manager = self.deepagents_runtime.artifact_manager
        reader = getattr(manager, "published_canonical_digests", None) if manager is not None else None
        if not callable(reader):
            return frozenset()
        try:
            digests = reader()
        except Exception:  # noqa: BLE001 - report fallback must survive best-effort artifact state reads
            logger.warning("Unable to read published canonical dataset digests", exc_info=True)
            return frozenset()
        if not isinstance(digests, (set, frozenset, list, tuple)):
            return frozenset()
        return frozenset(digest for digest in digests if isinstance(digest, str))

    @staticmethod
    def _canonical_table_section(dataset: QuantitativeDataset) -> str:
        """Render an exact canonical table with only its validated title added."""
        return f"### {dataset.title.strip()}\n\n{dataset.markdown_table.strip()}"

    @staticmethod
    def _contains_all_canonical_tables(report: str, tables: tuple[str, ...]) -> bool:
        """Check exact canonical table inclusion while tolerating newline style only."""
        normalized_report = report.replace("\r\n", "\n").replace("\r", "\n")
        return all(table.replace("\r\n", "\n").replace("\r", "\n") in normalized_report for table in tables)

    def _build_research_notes_fallback(
        self,
        files: object,
        *,
        notes: list[ResearchNotes] | None = None,
    ) -> tuple[str, bool] | None:
        """Build a bounded report directly from validated researcher evidence.

        This is the terminal safety net when the writer fails to create
        ``output.md``. It performs no new analysis: quantitative tables and
        summaries are copied from validated ``ResearchNotes`` and source
        references are assigned deterministically.
        """
        notes = notes if notes is not None else self._validated_research_notes(files)
        if not notes:
            return None

        source_numbers: dict[tuple[str, str], int] = {}
        source_rows: list[tuple[int, str, str]] = []
        local_source_numbers: list[dict[int, int]] = []
        for note in notes:
            note_numbers: dict[int, int] = {}
            for source in note.sources:
                key = (source.locator.strip(), source.title.strip())
                number = source_numbers.get(key)
                if number is None:
                    number = len(source_rows) + 1
                    source_numbers[key] = number
                    source_rows.append((number, source.title.strip() or source.locator, source.locator.strip()))
                note_numbers[source.id] = number
            local_source_numbers.append(note_numbers)

        sections = [
            "# Research Report",
            (
                "> The final synthesis step did not complete. This bounded fallback preserves "
                "the validated researcher evidence and canonical quantitative tables without rerunning analysis."
            ),
        ]
        selected_datasets, omitted_table_count = self._select_required_quantitative_datasets(notes)
        selected_dataset_objects = {id(dataset) for _, dataset in selected_datasets}
        if selected_datasets:
            sections.append("## Canonical quantitative evidence")
            sections.extend(self._canonical_table_section(dataset) for _, dataset in selected_datasets)

        rendered_chars = sum(len(section) for section in sections) + 2 * (len(sections) - 1)
        optional_limit = _MAX_FALLBACK_REPORT_CHARS - _FALLBACK_OMISSION_NOTICE_RESERVE
        optional_evidence_omitted = False

        def append_optional(section: str) -> None:
            """Append optional evidence without consuming the omission-notice reserve."""
            nonlocal optional_evidence_omitted
            nonlocal rendered_chars
            section = section.strip()
            if not section:
                return
            separator_chars = 2 if sections else 0
            available = optional_limit - rendered_chars - separator_chars
            if available <= 0:
                optional_evidence_omitted = True
                return
            if len(section) > available:
                optional_evidence_omitted = True
                if available <= len(_FALLBACK_TRUNCATION_MARKER):
                    return
                section = section[: available - len(_FALLBACK_TRUNCATION_MARKER)].rstrip()
                section += _FALLBACK_TRUNCATION_MARKER
            sections.append(section)
            rendered_chars += separator_chars + len(section)

        if source_rows:
            sources = ["## Sources"]
            for number, title, locator in source_rows:
                sources.append(f"[{number}] {title}: {locator}")
            append_optional("\n".join(sources))

        for note, note_numbers in zip(notes, local_source_numbers, strict=True):
            note_sections = [f"## {note.query_topic.strip() or 'Research findings'}"]
            if note.summary.strip():
                note_sections.append(note.summary.strip())
            if note.findings:
                findings = ["### Findings"]
                for finding in note.findings:
                    citations = sorted(
                        {note_numbers[source_id] for source_id in finding.source_ids if source_id in note_numbers}
                    )
                    citation_text = "".join(f"[{number}]" for number in citations)
                    evidence = finding.evidence.strip()
                    line = f"- **{finding.claim.strip()}**"
                    if evidence:
                        line += f" {evidence}"
                    if citation_text:
                        line += f" {citation_text}"
                    findings.append(line)
                note_sections.append("\n".join(findings))
            for dataset in note.quantitative_datasets:
                if id(dataset) not in selected_dataset_objects:
                    continue
                dataset_section = [f"### {dataset.title} — analysis", dataset.summary.strip()]
                if dataset.caveats:
                    dataset_section.extend(["#### Caveats", *[f"- {caveat.strip()}" for caveat in dataset.caveats]])
                note_sections.append("\n\n".join(part for part in dataset_section if part))
            if note.gaps:
                note_sections.append("### Remaining gaps\n" + "\n".join(f"- {gap.description}" for gap in note.gaps))
            append_optional("\n\n".join(note_sections))

        if omitted_table_count or optional_evidence_omitted:
            omitted = []
            if omitted_table_count:
                noun = "table" if omitted_table_count == 1 else "tables"
                verb = "was" if omitted_table_count == 1 else "were"
                omitted.append(f"{omitted_table_count} additional validated quantitative {noun} {verb} omitted")
            if optional_evidence_omitted:
                omitted.append("some supporting evidence was truncated or omitted")
            notice = (
                "> Bounded fallback notice: " + "; ".join(omitted) + " to keep this report within its safety limit."
            )
            sections.append(notice)

        return "\n\n".join(sections).strip(), bool(selected_datasets)

    @staticmethod
    def _read_seed_file_text(files: dict[str, Any], path: str) -> str | None:
        entry = files.get(path)
        if isinstance(entry, dict):
            entry = entry.get("content")
        if isinstance(entry, bytes):
            entry = entry.decode("utf-8")
        return entry if isinstance(entry, str) and entry.strip() else None

    def _seed_parent_sources(self, files: dict[str, Any]) -> None:
        """Register parent report sources so preserved citations verify in delta reports."""
        context_text = self._read_seed_file_text(files, PARENT_REPORT_CONTEXT_PATH)
        if not context_text:
            return
        parent_sources = source_entries_from_parent_context(context_text)
        seeded = self.source_registry_middleware.register_compact_sources(parent_sources)
        if seeded:
            logger.info("Seeded %d parent report source(s) into citation registry", seeded)

    @staticmethod
    def _replace_last_message_content(result: dict | Any, content: str) -> None:
        """Overwrite the final message content in-place with post-processed Markdown."""
        messages = result.get("messages") if isinstance(result, dict) else getattr(result, "messages", None)
        if not messages:
            return
        last_msg = messages[-1]
        if hasattr(last_msg, "model_copy"):
            messages[-1] = last_msg.model_copy(update={"content": content})
        else:
            messages[-1] = type(last_msg)(content=content)

    async def run(self, state: DeepResearchAgentState) -> DeepResearchAgentState:
        """
        Execute deep research with multi-phase workflow.
        """
        prepared_files = self.deepagents_runtime.prepare_state_files(dict(state.files))
        if prepared_files != state.files:
            state = state.model_copy(update={"files": prepared_files})
        self._seed_parent_sources(state.files)
        final_report_tracker = FinalReportCommitTracker()
        agent = self._build_orchestrator_agent(state, final_report_tracker=final_report_tracker)

        messages = state.messages
        if messages:
            query_content = messages[-1].content
            query = query_content if isinstance(query_content, str) else str(query_content)
            logger.info("=" * 80)
            logger.info("Deep Research Subagent: Starting workflow")
            logger.info("Query: %s...", query[:100])
            logger.info("=" * 80)

        try:
            result = await agent.ainvoke(state, config={"callbacks": self.callbacks} if self.callbacks else None)

            final_message = self._extract_final_markdown(
                result,
                state.files,
                final_report_tracker=final_report_tracker,
            )
            if final_message is None:
                raise RuntimeError("writer_output_not_committed")

            # Post-process: verify citations against source registry
            if self.enable_citation_verification and self.source_registry_middleware.has_sources():
                registry = self.source_registry_middleware.active_registry()
                verification = verify_citations(
                    final_message,
                    registry,
                    reference_sources=self.source_registry_middleware.get_source_entries(mode="compact"),
                )
                if verification.removed_citations:
                    removed_details = []
                    for c in verification.removed_citations:
                        url_match = re.search(r"https?://\S+", c.get("line", ""))
                        url_str = url_match.group(0).rstrip(".,;)") if url_match else "(no url)"
                        removed_details.append(f"[{c['number']}] {c['reason']}: {url_str}")
                    logger.info(
                        "Citation verification removed %d invalid citation(s):\n  %s",
                        len(verification.removed_citations),
                        "\n  ".join(removed_details),
                    )
                final_message = verification.verified_report
                if not verification.valid_citations:
                    logger.warning(
                        "Citation verification found no valid citations in writer-agent output; "
                        "returning the generated report without failing the job. "
                        "This may indicate unsupported citation formatting or over-aggressive verification."
                    )
            elif self.enable_citation_verification:
                from aiq_agent.common.tool_validation import validate_tool_availability

                _, available_count, unavailable = validate_tool_availability(
                    self.tools,
                    research_type="deep research",
                    enable_logging=False,
                )
                raise EmptySourceRegistryError(
                    "deep research",
                    unavailable_tools=unavailable,
                    available_count=available_count,
                )

            # Post-process: sanitize report (strip body URLs, shortened URLs, unsafe URLs)
            sanitization = sanitize_report(final_message)
            final_message = sanitization.sanitized_report

            # Post-process: harvest sandbox artifacts and resolve artifact:// references so
            # generated charts/files render in the report. Inert (manager is None) unless a
            # sandbox + artifact_capture + db_url are configured. Blocking I/O off the loop.
            manager = self.deepagents_runtime.artifact_manager
            if manager is not None:
                try:
                    await asyncio.to_thread(manager.final_harvest)
                    produced = await asyncio.to_thread(manager.store.list, manager.job_id)
                    final_message = await asyncio.to_thread(manager.resolve_report_references, final_message, produced)
                    final_message = await asyncio.to_thread(
                        manager.ensure_inline_artifacts_embedded, final_message, produced
                    )
                    final_message = await asyncio.to_thread(manager.append_artifact_index, final_message, produced)
                except Exception:
                    # Best-effort: never discard an already verified/sanitized report because
                    # artifact harvest or embedding failed. final_message stays as-is.
                    logger.warning(
                        "Artifact post-processing failed; returning report without embedded artifacts",
                        exc_info=True,
                    )

            # Re-emit the verified/sanitized report so the frontend overwrites
            # the raw version that on_llm_end auto-emitted during ainvoke().
            for cb in self.callbacks:
                if hasattr(cb, "emit_final_report"):
                    cb.emit_final_report(final_message)
                    break

            self._replace_last_message_content(result, final_message)

            logger.info("=" * 80)
            logger.info("Deep Research Subagent: Workflow complete")
            logger.info("Final answer length: %d characters", len(final_message))
            logger.info("=" * 80)
            return DeepResearchAgentState.model_validate(result)

        except Exception as ex:
            logger.error("Deep Research Subagent failed: %s", ex, exc_info=True)
            raise
