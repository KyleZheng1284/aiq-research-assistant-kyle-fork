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

"""Structured response contracts for deep researcher planning, research, and synthesis."""

from __future__ import annotations

import csv
import hashlib
import io
from typing import ClassVar
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator

_DATASET_ID_PATTERN = r"^[a-z0-9][a-z0-9_-]{0,63}$"
_MAX_DATASETS_PER_NOTE = 4
_MAX_CSV_BYTES_PER_DATASET = 64 * 1024
_MAX_CSV_BYTES_PER_NOTE = 128 * 1024
_MAX_CSV_DATA_ROWS = 5_000
_MAX_CSV_COLUMNS = 128
_MAX_DATASET_TITLE_CHARS = 256
_MAX_MARKDOWN_TABLE_CHARS = 64 * 1024
_MAX_DATASET_SUMMARY_CHARS = 8 * 1024
_MAX_DATASET_CAVEATS = 32
_MAX_DATASET_CAVEAT_CHARS = 2 * 1024
_MAX_DATASET_SOURCE_IDS = 128
_ALLOWED_CSV_CONTROLS = frozenset({"\t", "\n", "\r"})


class _StrictContract(BaseModel):
    """Base model for structured response schemas."""

    model_config: ClassVar[ConfigDict] = {"extra": "forbid"}


class TaskAnalysis(_StrictContract):
    """Planner analysis of the user's research request."""

    user_intent: str = Field(description="Brief statement of what the user wants to achieve.")
    explicit_requirements: list[str] = Field(description="Requirements explicitly stated by the user.")
    implicit_requirements: list[str] = Field(description="Requirements implied by the request.")
    out_of_scope: list[str] = Field(description="Tangential topics that should be excluded from the report.")
    language: str = Field(description="Language to use for the plan, notes, and final report.")


class AnswerComponent(_StrictContract):
    """Required evidence or synthesis component for the final answer."""

    id: str = Field(description="Stable component identifier, such as 'latest_price_anchor'.")
    name: str = Field(description="Short human-readable component name.")
    description: str = Field(description="What the writer must cover for this component.")


class AnswerStrategy(_StrictContract):
    """Planner guidance for the final answer shape and synthesis logic."""

    answer_type: Literal[
        "long_form_report",
        "brief_answer",
        "table",
        "comparison",
        "prediction",
        "multiple_choice",
        "data_extraction",
        "custom",
    ] = Field(description="The intended final output shape.")
    title: str = Field(description="Concise human-facing title for the final output.")
    required_components: list[AnswerComponent] = Field(
        description="Evidence and synthesis components that must be covered in the final answer."
    )


class Constraint(_StrictContract):
    """Lightweight final-answer requirement."""

    category: Literal["content", "source", "structure", "depth", "format", "exclusion"] = Field(
        description="Constraint category."
    )
    constraint: str = Field(description="Specific, actionable constraint text.")
    rationale: str = Field(description="Why this constraint exists.")


class SourceRecommendation(_StrictContract):
    """A source-router recommendation for the planner."""

    source_id: str = Field(description="Configured data source ID to use.")
    tool_names: list[str] = Field(description="Exact available source tool names under this source.")
    priority: int = Field(ge=1, le=3, description="Priority rank for this source: 1 is highest, 3 is lowest.")
    rationale: str = Field(description="Why this source should support the request.")


class SourceRoutingPlan(_StrictContract):
    """Advisory source route produced before planning."""

    domain_id: str = Field(description="Best-fit configured domain route for this request.")
    domain_name: str = Field(description="Human-readable domain name.")
    routing_reason: str = Field(description="Why this domain/source route fits the user request.")
    recommendations: list[SourceRecommendation] = Field(description="Primary source recommendations.")
    fallback_sources: list[SourceRecommendation] = Field(description="Fallback sources if primary sources are weak.")
    planner_guidance: str = Field(description="Concise instructions the planner should apply when writing queries.")


class ResearchQuery(_StrictContract):
    """Self-contained research query for a researcher worker."""

    query: str = Field(description="Specific, self-contained search or document query.")
    subqueries: list[str] = Field(
        default_factory=list,
        description=(
            "Optional ordered concrete search angles for distinct facets unlikely to be covered by the main query. "
            "Prefer leaving this empty for focused queries and creating separate ResearchQuery items for independent "
            "evidence needs."
        ),
    )
    preferred_tools: list[str] = Field(
        min_length=1,
        description=(
            "Ordered exact available source tool names to prioritize for this query. "
            "The first item is the primary tool the researcher should use first."
        ),
    )
    fallback_tools: list[str] = Field(
        default_factory=list,
        description="Ordered exact available source tool names to use for corroboration or gaps.",
    )
    target_components: list[str] = Field(description="Answer components this query is intended to support.")
    rationale: str = Field(description="Why this query is needed.")


class ResearchPlan(_StrictContract):
    """Structured plan produced by the planner subagent."""

    task_analysis: TaskAnalysis = Field(description="Planner analysis of the user's request.")
    answer_strategy: AnswerStrategy = Field(description="Final answer shape and synthesis strategy.")
    constraints: list[Constraint] = Field(description="Lightweight requirements for the final answer.")
    queries: list[ResearchQuery] = Field(description="Queries for researcher workers to execute.")


class ResearchSource(_StrictContract):
    """Source used by a researcher worker."""

    id: int = Field(description="Integer source identifier used by findings in this note.")
    title: str = Field(description="Source title or document name.")
    source_type: Literal["url", "internal_document", "tool"] = Field(
        description="Kind of source referenced by locator."
    )
    locator: str = Field(
        description=(
            "URL for web sources, document/page citation for internal documents, "
            "or raw tool name for URL-less structured tool results."
        )
    )


class ResearchFinding(_StrictContract):
    """Atomic finding captured from one or more sources."""

    claim: str = Field(description="Concise factual claim or analytical conclusion.")
    evidence: str = Field(description="Detailed supporting evidence, including dates, figures, names, and context.")
    source_ids: list[int] = Field(description="IDs from the sources list that support this finding.")
    confidence: Literal["low", "medium", "high"] = Field(description="Confidence in the finding.")
    caveats: list[str] = Field(description="Limitations, disagreements, or context needed to use this finding.")


class ResearchGap(_StrictContract):
    """Information gap identified during research."""

    description: str = Field(description="Missing or weakly supported information.")
    impact: str = Field(description="Why the gap matters for the final report.")
    suggested_follow_up_queries: list[str] = Field(description="Queries that could close the gap.")


class EvidenceJudgment(_StrictContract):
    """Post-research judgment attached to a research note."""

    relevance_score: int = Field(
        ge=0,
        le=100,
        description="How useful this note is for the final answer, from 0 to 100.",
    )
    confidence: Literal["low", "medium", "high"] = Field(description="Confidence in this judgment.")
    rationale: str = Field(description="Concise explanation of the relevance score and confidence.")


class QuantitativeDataset(_StrictContract):
    """Canonical quantitative evidence produced by a researcher worker.

    ``csv_text`` is the sole canonical serialization. Validation inspects it but
    never normalizes or reserializes it, and ``csv_sha256`` is always recomputed
    from its exact UTF-8 bytes rather than trusted from model output.
    """

    model_config: ClassVar[ConfigDict] = {
        "extra": "forbid",
        "frozen": True,
        "revalidate_instances": "always",
    }

    dataset_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=_DATASET_ID_PATTERN,
        description="Stable lowercase identifier containing letters, digits, underscores, or hyphens.",
    )
    title: str = Field(
        min_length=1,
        max_length=_MAX_DATASET_TITLE_CHARS,
        description="Concise human-facing dataset title.",
    )
    csv_text: str = Field(description="Exact UTF-8 CSV serialization of the canonical dataset.")
    markdown_table: str = Field(
        min_length=1,
        max_length=_MAX_MARKDOWN_TABLE_CHARS,
        description="Markdown rendering derived from the same canonical dataset.",
    )
    summary: str = Field(
        min_length=1,
        max_length=_MAX_DATASET_SUMMARY_CHARS,
        description="Computed statistics and interpretation of the canonical dataset.",
    )
    source_ids: list[int] = Field(
        default_factory=list,
        max_length=_MAX_DATASET_SOURCE_IDS,
        description="IDs from the enclosing ResearchNotes sources that support this dataset.",
    )
    caveats: list[str] = Field(
        default_factory=list,
        max_length=_MAX_DATASET_CAVEATS,
        description="Dataset-specific limitations and normalization caveats.",
    )
    csv_sha256: str = Field(
        default="",
        frozen=True,
        description="Runtime-computed SHA-256 of the exact UTF-8 csv_text; model-provided values are ignored.",
    )

    @field_validator("title", "markdown_table", "summary")
    @classmethod
    def _reject_blank_descriptive_fields(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("dataset descriptive fields must not be blank")
        return value

    @field_validator("caveats")
    @classmethod
    def _validate_caveats(cls, caveats: list[str]) -> list[str]:
        for caveat in caveats:
            if not caveat.strip():
                raise ValueError("dataset caveats must not be blank")
            if len(caveat) > _MAX_DATASET_CAVEAT_CHARS:
                raise ValueError(f"dataset caveats must not exceed {_MAX_DATASET_CAVEAT_CHARS} characters")
        return caveats

    @field_validator("source_ids")
    @classmethod
    def _validate_source_ids(cls, source_ids: list[int]) -> list[int]:
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("dataset source_ids must be unique")
        return source_ids

    @field_validator("csv_sha256", mode="before")
    @classmethod
    def _ignore_model_digest(cls, _value: object) -> str:
        return ""

    @model_validator(mode="after")
    def _validate_csv_and_compute_digest(self) -> QuantitativeDataset:
        csv_bytes = self.csv_text.encode("utf-8")
        if not csv_bytes:
            raise ValueError("csv_text must not be empty")
        if len(csv_bytes) > _MAX_CSV_BYTES_PER_DATASET:
            raise ValueError(f"csv_text must not exceed {_MAX_CSV_BYTES_PER_DATASET} UTF-8 bytes")
        if self.csv_text.startswith("\ufeff") or "\ufeff" in self.csv_text:
            raise ValueError("csv_text must not contain a byte-order mark")
        if any(_is_forbidden_csv_control(character) for character in self.csv_text):
            raise ValueError("csv_text contains a forbidden control character")
        _validate_csv_quote_placement(self.csv_text)

        try:
            reader = csv.reader(io.StringIO(self.csv_text, newline=""), strict=True)
            header = next(reader)
            if not header:
                raise ValueError("csv_text must contain a header row")
            if len(header) > _MAX_CSV_COLUMNS:
                raise ValueError(f"csv_text must not exceed {_MAX_CSV_COLUMNS} columns")
            if any(not column or column != column.strip() for column in header):
                raise ValueError("CSV header names must be non-empty and trimmed")
            if len(header) != len(set(header)):
                raise ValueError("CSV header names must be unique")

            row_count = 0
            for row in reader:
                row_count += 1
                if row_count > _MAX_CSV_DATA_ROWS:
                    raise ValueError(f"csv_text must not exceed {_MAX_CSV_DATA_ROWS} data rows")
                if not row or all(not cell.strip() for cell in row):
                    raise ValueError("csv_text must not contain blank records")
                if len(row) != len(header):
                    raise ValueError("every CSV data row must match the header width")
            if row_count == 0:
                raise ValueError("csv_text must contain at least one data row")
        except csv.Error as exc:
            raise ValueError(f"csv_text is not valid strict CSV: {exc}") from exc

        object.__setattr__(self, "csv_sha256", hashlib.sha256(csv_bytes).hexdigest())
        return self


def _is_forbidden_csv_control(character: str) -> bool:
    """Return whether a character is unsafe in canonical CSV text."""
    codepoint = ord(character)
    return (codepoint < 32 and character not in _ALLOWED_CSV_CONTROLS) or codepoint == 127


def _validate_csv_quote_placement(csv_text: str) -> None:
    """Reject quote placement that ``csv.reader(strict=True)`` accepts leniently.

    The standard-library parser treats a quote inside an unquoted field as a
    literal character even in strict mode. Such input is not portable CSV and
    downstream readers may interpret it differently. This state check inspects
    the original text without normalizing or reserializing it.
    """
    field_start = "field_start"
    unquoted = "unquoted"
    quoted = "quoted"
    after_quote = "after_quote"
    state = field_start

    for character in csv_text:
        if state == field_start:
            if character == '"':
                state = quoted
            elif character not in {",", "\r", "\n"}:
                state = unquoted
        elif state == unquoted:
            if character == '"':
                raise ValueError("csv_text contains malformed quote placement")
            if character in {",", "\r", "\n"}:
                state = field_start
        elif state == quoted:
            if character == '"':
                state = after_quote
        elif character == '"':
            state = quoted
        elif character in {",", "\r", "\n"}:
            state = field_start
        else:
            raise ValueError("csv_text contains malformed quote placement")


class ResearchNotes(_StrictContract):
    """Structured notes produced by a researcher worker."""

    query_topic: str = Field(description="Short topic label for this research note.")
    target_components: list[str] = Field(description="Answer components these notes support.")
    summary: str = Field(description="Brief synthesis of the research results.")
    findings: list[ResearchFinding] = Field(description="Detailed findings supported by cited sources.")
    gaps: list[ResearchGap] = Field(description="Open gaps or weak spots discovered during research.")
    sources: list[ResearchSource] = Field(description="Every source used by these notes.")
    narrative_notes: str = Field(description="Detailed synthesis preserving nuance for final answer writing.")
    language: str = Field(description="Language used in these research notes.")
    evidence_judgment: EvidenceJudgment | None = Field(
        default=None,
        description="Researcher self-assessment of this note's usefulness for final synthesis.",
    )
    quantitative_datasets: list[QuantitativeDataset] = Field(
        default_factory=list,
        max_length=_MAX_DATASETS_PER_NOTE,
        description="Validated canonical quantitative evidence for durable publication by the writer.",
    )

    @model_validator(mode="after")
    def _validate_quantitative_dataset_references(self) -> ResearchNotes:
        if not self.quantitative_datasets:
            return self

        source_ids = [source.id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("ResearchNotes source IDs must be unique when quantitative datasets are present")

        known_source_ids = set(source_ids)
        dataset_ids = [dataset.dataset_id for dataset in self.quantitative_datasets]
        if len(dataset_ids) != len(set(dataset_ids)):
            raise ValueError("quantitative dataset IDs must be unique within ResearchNotes")

        unknown_source_ids = sorted(
            {
                source_id
                for dataset in self.quantitative_datasets
                for source_id in dataset.source_ids
                if source_id not in known_source_ids
            }
        )
        if unknown_source_ids:
            raise ValueError(f"quantitative datasets reference unknown source IDs: {unknown_source_ids}")

        total_csv_bytes = sum(len(dataset.csv_text.encode("utf-8")) for dataset in self.quantitative_datasets)
        if total_csv_bytes > _MAX_CSV_BYTES_PER_NOTE:
            raise ValueError(
                f"aggregate quantitative CSV content must not exceed {_MAX_CSV_BYTES_PER_NOTE} UTF-8 bytes"
            )
        return self
