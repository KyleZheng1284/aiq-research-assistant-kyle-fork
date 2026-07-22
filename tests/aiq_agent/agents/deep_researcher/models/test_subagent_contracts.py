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

"""Tests for deep researcher structured response contracts."""

import hashlib

import pytest
from pydantic import ValidationError

from aiq_agent.agents.deep_researcher.models import AnswerStrategy
from aiq_agent.agents.deep_researcher.models import Constraint
from aiq_agent.agents.deep_researcher.models import EvidenceJudgment
from aiq_agent.agents.deep_researcher.models import QuantitativeDataset
from aiq_agent.agents.deep_researcher.models import ResearchNotes
from aiq_agent.agents.deep_researcher.models import ResearchPlan
from aiq_agent.agents.deep_researcher.models import SourceRoutingPlan


def _answer_strategy() -> dict:
    return {
        "answer_type": "comparison",
        "title": "CUDA and OpenCL Trade-offs",
        "required_components": [
            {
                "id": "programming_model",
                "name": "Programming model",
                "description": "Compare kernel, memory, and execution models.",
            }
        ],
    }


def _task_analysis() -> dict:
    return {
        "user_intent": "Understand CUDA and OpenCL trade-offs.",
        "explicit_requirements": ["Compare CUDA and OpenCL"],
        "implicit_requirements": ["Cover ecosystem and portability"],
        "out_of_scope": ["General GPU purchasing advice"],
        "language": "English",
    }


def _quantitative_dataset(**updates) -> dict:
    dataset = {
        "dataset_id": "quarterly_revenue",
        "title": "Quarterly revenue",
        "csv_text": "quarter,revenue_usd_billions\nFY2025-Q1,22.6\nFY2025-Q2,26.3\n",
        "markdown_table": "| Quarter | Revenue |\n|---|---:|\n| FY2025-Q1 | 22.6 |",
        "summary": "Mean revenue was USD 24.45 billion.",
        "source_ids": [1],
        "caveats": ["Figures are rounded to one decimal place."],
    }
    dataset.update(updates)
    return dataset


def _research_notes_with_datasets(datasets: list[dict], **updates) -> dict:
    notes = {
        "query_topic": "NVIDIA quarterly revenue",
        "target_components": ["revenue_table"],
        "summary": "Revenue increased across the period.",
        "findings": [],
        "gaps": [],
        "sources": [
            {
                "id": 1,
                "title": "NVIDIA quarterly results",
                "source_type": "url",
                "locator": "https://example.test/results",
            }
        ],
        "narrative_notes": "The canonical table preserves the reported values.",
        "language": "English",
        "quantitative_datasets": datasets,
    }
    notes.update(updates)
    return notes


def test_research_plan_contract_validates_expected_shape():
    plan = ResearchPlan.model_validate(
        {
            "task_analysis": _task_analysis(),
            "answer_strategy": _answer_strategy(),
            "constraints": [
                {
                    "category": "content",
                    "constraint": "Compare portability, performance, and ecosystem maturity.",
                    "rationale": "These dimensions determine practical adoption.",
                }
            ],
            "queries": [
                {
                    "query": "CUDA OpenCL portability performance ecosystem comparison",
                    "subqueries": ["CUDA OpenCL portability", "CUDA OpenCL benchmark comparison"],
                    "preferred_tools": ["web_search_tool"],
                    "fallback_tools": [],
                    "target_components": ["programming_model"],
                    "rationale": "Supports the comparison component.",
                }
            ],
        }
    )

    assert plan.answer_strategy.required_components[0].id == "programming_model"
    assert plan.constraints[0].category == "content"
    assert plan.queries[0].target_components == ["programming_model"]
    assert plan.queries[0].subqueries == ["CUDA OpenCL portability", "CUDA OpenCL benchmark comparison"]
    assert plan.queries[0].preferred_tools == ["web_search_tool"]
    assert plan.queries[0].fallback_tools == []


def test_research_plan_contract_accepts_prediction_answer_type():
    answer_strategy = _answer_strategy()
    answer_strategy["answer_type"] = "prediction"
    answer_strategy["title"] = "Election Forecast"

    plan = ResearchPlan.model_validate(
        {
            "task_analysis": _task_analysis(),
            "answer_strategy": answer_strategy,
            "constraints": [],
            "queries": [
                {
                    "query": "Example election forecast evidence",
                    "subqueries": [],
                    "preferred_tools": ["polymarket_search_tool"],
                    "fallback_tools": [],
                    "target_components": ["programming_model"],
                    "rationale": "Supports the forecast evidence component.",
                }
            ],
        }
    )

    assert plan.answer_strategy.answer_type == "prediction"
    assert plan.queries[0].preferred_tools == ["polymarket_search_tool"]


def test_reduced_answer_strategy_contract_validates():
    strategy = AnswerStrategy.model_validate(_answer_strategy())

    assert strategy.answer_type == "comparison"
    assert strategy.title == "CUDA and OpenCL Trade-offs"
    assert strategy.required_components[0].id == "programming_model"


def test_constraint_contract_rejects_verification_field():
    with pytest.raises(ValidationError):
        Constraint.model_validate(
            {
                "category": "content",
                "constraint": "Compare portability, performance, and ecosystem maturity.",
                "rationale": "These dimensions determine practical adoption.",
                "verification": "Each dimension appears in the final answer.",
            }
        )


def test_research_notes_contract_validates_expected_shape():
    notes = ResearchNotes.model_validate(
        {
            "query_topic": "CUDA vs OpenCL portability",
            "target_components": ["programming_model"],
            "summary": "CUDA is NVIDIA-specific while OpenCL targets cross-vendor portability.",
            "findings": [
                {
                    "claim": "OpenCL is designed for cross-vendor heterogeneous compute.",
                    "evidence": "The source describes OpenCL as an open standard for heterogeneous platforms.",
                    "source_ids": [1],
                    "confidence": "high",
                    "caveats": ["Portability does not guarantee equal performance across vendors."],
                }
            ],
            "gaps": [
                {
                    "description": "Recent benchmark coverage is sparse.",
                    "impact": "Limits quantitative comparison.",
                    "suggested_follow_up_queries": ["CUDA OpenCL benchmark 2026"],
                }
            ],
            "sources": [
                {
                    "id": 1,
                    "title": "OpenCL Overview",
                    "source_type": "url",
                    "locator": "https://example.test/opencl",
                }
            ],
            "narrative_notes": "OpenCL offers broader portability, while CUDA typically has deeper vendor tooling.",
            "language": "English",
        }
    )

    assert notes.target_components == ["programming_model"]
    assert notes.findings[0].source_ids == [1]
    assert notes.sources[0].source_type == "url"
    assert notes.sources[0].locator == "https://example.test/opencl"
    assert notes.evidence_judgment is None


def test_research_notes_contract_accepts_evidence_judgment():
    notes = ResearchNotes.model_validate(
        {
            "query_topic": "CUDA vs OpenCL portability",
            "target_components": ["programming_model"],
            "summary": "CUDA is NVIDIA-specific while OpenCL targets portability.",
            "findings": [],
            "gaps": [],
            "sources": [],
            "narrative_notes": "OpenCL offers broader portability.",
            "language": "English",
            "evidence_judgment": {
                "relevance_score": 85,
                "confidence": "high",
                "rationale": "Directly supports the programming model component.",
            },
        }
    )

    assert notes.evidence_judgment is not None
    assert notes.evidence_judgment.relevance_score == 85
    assert notes.evidence_judgment.confidence == "high"


def test_quantitative_dataset_preserves_exact_csv_and_computes_runtime_digest():
    csv_text = 'name,note\r\n"Alpha","comma, quote ""ok"""\r\n"β","line 1\nline 2"\r\n'
    supplied_digest = "f" * 64

    dataset = QuantitativeDataset.model_validate(_quantitative_dataset(csv_text=csv_text, csv_sha256=supplied_digest))

    assert dataset.csv_text == csv_text
    assert dataset.csv_sha256 == hashlib.sha256(csv_text.encode("utf-8")).hexdigest()
    assert dataset.csv_sha256 != supplied_digest
    assert dataset.model_dump(mode="json")["csv_text"] == csv_text


@pytest.mark.parametrize(
    ("csv_text", "message"),
    [
        ("", "must not be empty"),
        ("\n1,2\n", "header row"),
        ("a,b\n", "at least one data row"),
        ("a,a\n1,2\n", "header names must be unique"),
        (" a,b\n1,2\n", "header names must be non-empty and trimmed"),
        ("a,\n1,2\n", "header names must be non-empty and trimmed"),
        ("a,b\n1\n", "must match the header width"),
        ("a,b\n1,2,3\n", "must match the header width"),
        ("a,b\n1,2\n\n3,4\n", "must not contain blank records"),
        ("a,b\n,\n", "must not contain blank records"),
        ('a,b\n"unterminated,2\n', "not valid strict CSV"),
        ('a,b\n1"2,3\n', "malformed quote placement"),
        ('a,b\n1, "quoted after space"\n', "malformed quote placement"),
        ("\ufeffa,b\n1,2\n", "byte-order mark"),
        ("a,b\n1,\x00\n", "forbidden control character"),
        ("a,b\n1,\x7f\n", "forbidden control character"),
    ],
)
def test_quantitative_dataset_rejects_invalid_csv(csv_text: str, message: str):
    with pytest.raises(ValidationError, match=message):
        QuantitativeDataset.model_validate(_quantitative_dataset(csv_text=csv_text))


def test_quantitative_dataset_accepts_lf_crlf_unicode_empty_cells_and_missing_final_newline():
    csv_variants = (
        "label,value,note\nGPU,12,\nCPU,8,baseline",
        "label,value\r\nGráficos,12\r\n数据,8\r\n",
    )

    datasets = [QuantitativeDataset.model_validate(_quantitative_dataset(csv_text=value)) for value in csv_variants]

    assert [dataset.csv_text for dataset in datasets] == list(csv_variants)


def test_quantitative_dataset_is_frozen_after_runtime_digest_computation():
    dataset = QuantitativeDataset.model_validate(_quantitative_dataset())

    with pytest.raises(ValidationError, match="frozen"):
        dataset.csv_text = "quarter,revenue_usd_billions\nFY2025-Q1,99\n"


def test_quantitative_dataset_enforces_csv_byte_row_and_column_limits():
    oversized_csv = "label,value\nrow," + ("é" * (32 * 1024)) + "\n"
    too_many_rows = "value\n" + "\n".join(str(index) for index in range(5_001)) + "\n"
    headers = [f"c{index}" for index in range(129)]
    too_many_columns = ",".join(headers) + "\n" + ",".join("1" for _ in headers) + "\n"

    with pytest.raises(ValidationError, match="65536 UTF-8 bytes"):
        QuantitativeDataset.model_validate(_quantitative_dataset(csv_text=oversized_csv))
    with pytest.raises(ValidationError, match="5000 data rows"):
        QuantitativeDataset.model_validate(_quantitative_dataset(csv_text=too_many_rows))
    with pytest.raises(ValidationError, match="128 columns"):
        QuantitativeDataset.model_validate(_quantitative_dataset(csv_text=too_many_columns))


def test_quantitative_dataset_accepts_exact_csv_and_note_boundaries():
    exact_size_csv = "value\n" + ("x" * (64 * 1024 - len("value\n") - 1)) + "\n"
    exact_rows_csv = "value\n" + ("1\n" * 5_000)
    headers = [f"c{index}" for index in range(128)]
    exact_columns_csv = ",".join(headers) + "\n" + ",".join("1" for _ in headers) + "\n"

    assert len(exact_size_csv.encode("utf-8")) == 64 * 1024
    assert QuantitativeDataset.model_validate(_quantitative_dataset(csv_text=exact_size_csv))
    assert QuantitativeDataset.model_validate(_quantitative_dataset(csv_text=exact_rows_csv))
    assert QuantitativeDataset.model_validate(_quantitative_dataset(csv_text=exact_columns_csv))

    exact_aggregate = [
        _quantitative_dataset(dataset_id=f"dataset_{index}", csv_text=exact_size_csv) for index in range(2)
    ]
    notes = ResearchNotes.model_validate(_research_notes_with_datasets(exact_aggregate))
    assert sum(len(dataset.csv_text.encode("utf-8")) for dataset in notes.quantitative_datasets) == 128 * 1024


def test_quantitative_dataset_rejects_duplicate_source_ids_and_invalid_descriptions():
    with pytest.raises(ValidationError, match="source_ids must be unique"):
        QuantitativeDataset.model_validate(_quantitative_dataset(source_ids=[1, 1]))
    with pytest.raises(ValidationError, match="must not be blank"):
        QuantitativeDataset.model_validate(_quantitative_dataset(title=" "))
    with pytest.raises(ValidationError, match="must not be blank"):
        QuantitativeDataset.model_validate(_quantitative_dataset(caveats=[" "]))


def test_research_notes_validates_quantitative_dataset_identity_sources_and_aggregate_size():
    duplicate_id = _quantitative_dataset()
    duplicate_id_2 = _quantitative_dataset()
    with pytest.raises(ValidationError, match="dataset IDs must be unique"):
        ResearchNotes.model_validate(_research_notes_with_datasets([duplicate_id, duplicate_id_2]))

    with pytest.raises(ValidationError, match="unknown source IDs"):
        ResearchNotes.model_validate(_research_notes_with_datasets([_quantitative_dataset(source_ids=[2])]))

    duplicate_sources = [
        {
            "id": 1,
            "title": "First",
            "source_type": "url",
            "locator": "https://example.test/first",
        },
        {
            "id": 1,
            "title": "Duplicate",
            "source_type": "url",
            "locator": "https://example.test/duplicate",
        },
    ]
    with pytest.raises(ValidationError, match="source IDs must be unique"):
        ResearchNotes.model_validate(
            _research_notes_with_datasets([_quantitative_dataset()], sources=duplicate_sources)
        )

    large_cell = "x" * 44_000
    datasets = [
        _quantitative_dataset(dataset_id=f"dataset_{index}", csv_text=f"value\n{large_cell}\n") for index in range(3)
    ]
    with pytest.raises(ValidationError, match="aggregate quantitative CSV content"):
        ResearchNotes.model_validate(_research_notes_with_datasets(datasets))


def test_research_notes_enforces_dataset_count_and_accepts_four_distinct_datasets():
    four = [_quantitative_dataset(dataset_id=f"dataset_{index}") for index in range(4)]
    notes = ResearchNotes.model_validate(_research_notes_with_datasets(four))
    assert len(notes.quantitative_datasets) == 4

    five = [_quantitative_dataset(dataset_id=f"dataset_{index}") for index in range(5)]
    with pytest.raises(ValidationError):
        ResearchNotes.model_validate(_research_notes_with_datasets(five))


@pytest.mark.parametrize("dataset_id", ["Uppercase", "with space", "-leading", "x" * 65])
def test_quantitative_dataset_rejects_invalid_dataset_id(dataset_id: str):
    with pytest.raises(ValidationError, match="dataset_id"):
        QuantitativeDataset.model_validate(_quantitative_dataset(dataset_id=dataset_id))


def test_evidence_judgment_contract_rejects_invalid_score():
    with pytest.raises(ValidationError):
        EvidenceJudgment.model_validate(
            {
                "relevance_score": 101,
                "confidence": "high",
                "rationale": "Score must stay within the configured range.",
            }
        )


def test_source_routing_plan_contract_validates_expected_shape():
    route = SourceRoutingPlan.model_validate(
        {
            "domain_id": "current_news",
            "domain_name": "Current News",
            "routing_reason": "The user asks for recent developments.",
            "recommendations": [
                {
                    "source_id": "news_search",
                    "tool_names": ["duckduckgo_news_search_tool"],
                    "priority": 1,
                    "rationale": "Best fit for recent news.",
                }
            ],
            "fallback_sources": [
                {
                    "source_id": "web_search",
                    "tool_names": ["web_search_tool"],
                    "priority": 2,
                    "rationale": "Broad web fallback.",
                }
            ],
            "planner_guidance": "Use news_search first, then web_search if coverage is weak.",
        }
    )

    assert route.domain_id == "current_news"
    assert route.recommendations[0].tool_names == ["duckduckgo_news_search_tool"]


def test_subagent_contracts_reject_extra_fields_and_old_plan_shape():
    with pytest.raises(ValidationError):
        ResearchPlan.model_validate(
            {
                "task_analysis": _task_analysis(),
                "answer_strategy": _answer_strategy(),
                "constraints": [],
                "queries": [],
                "unexpected": "value",
            }
        )

    with pytest.raises(ValidationError):
        ResearchPlan.model_validate(
            {
                "task_analysis": _task_analysis(),
                "report_title": "Title",
                "report_toc": [],
                "constraints": [],
                "queries": [],
            }
        )

    with pytest.raises(ValidationError):
        ResearchNotes.model_validate(
            {
                "query_topic": "CUDA vs OpenCL portability",
                "target_sections": ["Programming Model Differences"],
                "summary": "Old field should fail.",
                "findings": [],
                "gaps": [],
                "sources": [],
                "narrative_notes": "",
                "language": "English",
            }
        )

    for removed_field, value in (
        ("assembly_instruction", "Synthesize evidence into a comparison."),
        ("selection_mode", "none"),
        ("expected_count", None),
        ("options", []),
    ):
        old_strategy = _answer_strategy()
        old_strategy[removed_field] = value
        with pytest.raises(ValidationError):
            AnswerStrategy.model_validate(old_strategy)
