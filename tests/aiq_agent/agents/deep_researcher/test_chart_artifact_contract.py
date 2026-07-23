# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static contracts for canonical quantitative evidence and durable publication."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from aiq_agent.common import render_prompt_template

_REPO_ROOT = Path(__file__).parents[4]
_AGENT_ROOT = _REPO_ROOT / "src" / "aiq_agent" / "agents" / "deep_researcher"
_RESEARCH_SKILLS = _AGENT_ROOT / "skills" / "research"
_SYNTHESIS_SKILLS = _AGENT_ROOT / "skills" / "synthesis"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_chart_skill_is_owned_by_synthesis() -> None:
    assert not (_RESEARCH_SKILLS / "chart-generation" / "SKILL.md").exists()
    assert (_SYNTHESIS_SKILLS / "chart-generation" / "SKILL.md").is_file()
    assert (_RESEARCH_SKILLS / "data-table-analysis" / "SKILL.md").is_file()


def test_data_table_skill_returns_canonical_evidence_without_publishing() -> None:
    skill = _read(_RESEARCH_SKILLS / "data-table-analysis" / "SKILL.md")

    assert "ResearchNotes.quantitative_datasets" in skill
    assert "`csv_text` is the sole canonical serialization" in skill
    assert "from the same final DataFrame and column order" in skill
    assert "Never call `write_file` or `edit_file`" in skill
    assert "Do not calculate, guess, or return `csv_sha256`" in skill


def test_chart_skill_publishes_exact_csv_and_only_visualizes_existing_values() -> None:
    skill = _read(_SYNTHESIS_SKILLS / "chart-generation" / "SKILL.md")

    assert "do not add, drop, reorder, filter" in skill
    assert "Do not parse and reserialize" in skill
    assert "A CSV-only request ends here. It requires no `execute` call." in skill
    assert '"expected_sha256"' in skill
    assert "df = pd.read_csv(CSV_PATH)" in skill
    assert "hashlib.sha256(csv_bytes).hexdigest()" in skill
    assert "pd.DataFrame(rows)" not in skill
    assert ".groupby(" not in skill
    assert ".pct_change(" not in skill
    assert ".sort_values(" not in skill


def test_chart_skill_uses_runtime_argument_instead_of_executable_placeholder() -> None:
    skill = _read(_SYNTHESIS_SKILLS / "chart-generation" / "SKILL.md")

    assert 'ARTIFACT_DIR = "<sandbox_artifact_dir>"' not in skill
    assert '"path": "<sandbox_artifact_dir>' not in skill
    assert "ARTIFACT_DIR = Path(sys.argv[1])" in skill
    assert 'ARTIFACT_DIR / "manifest.json"' in skill


def test_researcher_hands_off_canonical_dataset_without_publishing() -> None:
    prompt = _read(_AGENT_ROOT / "prompts" / "researcher.j2")
    rendered = render_prompt_template(
        prompt,
        current_datetime="2026-07-09",
        execution_enabled=True,
        sandbox_workdir="/sandbox/job-123",
        sandbox_artifact_dir="/sandbox/job-123/aiq-artifacts",
        max_researcher_execute_attempts=3,
        user_info=None,
        tools=[],
        available_documents=None,
    )

    assert "return each final DataFrame as one `quantitative_datasets` entry" in rendered
    assert "Do not render charts" in rendered
    assert "add a concise recommendation to `ResearchNotes.narrative_notes`" in rendered
    assert "canonical dataset, chart type, axes or series, and rationale" in rendered
    assert "Never write to `/sandbox/job-123/aiq-artifacts`" in rendered
    assert "the writer alone decides whether to render and publish final charts" in rendered
    assert "Do not calculate or guess `csv_sha256`" in rendered


def test_researcher_prompt_omits_execute_budget_when_disabled() -> None:
    prompt = _read(_AGENT_ROOT / "prompts" / "researcher.j2")
    rendered = render_prompt_template(
        prompt,
        current_datetime="2026-07-09",
        execution_enabled=True,
        sandbox_workdir="/sandbox/job-123",
        sandbox_artifact_dir="/sandbox/job-123/aiq-artifacts",
        max_researcher_execute_attempts=None,
        user_info=None,
        tools=[],
        available_documents=None,
    )

    assert "physical `execute` attempts" not in rendered
    assert "None physical" not in rendered


def test_writer_writes_baseline_before_publication_and_bounds_chart_failure() -> None:
    prompt = _read(_AGENT_ROOT / "prompts" / "writer.j2")
    rendered = render_prompt_template(
        prompt,
        current_datetime="2026-07-09",
        execution_enabled=True,
        parent_report_context_available=False,
        sandbox_workdir="/sandbox/job-123",
        sandbox_artifact_dir="/sandbox/job-123/aiq-artifacts",
        user_info=None,
    )

    baseline = "Write the complete baseline to `/shared/output.md`"
    publication = "Publish the exact canonical CSV first"
    assert rendered.index(baseline) < rendered.index(publication)
    assert "A CSV-only request requires no `execute` call." in rendered
    assert "render it only from the already-published CSV" in rendered
    assert "python3 /sandbox/job-123/make_chart.py /sandbox/job-123/aiq-artifacts" in rendered
    assert "Never put a literal `<sandbox_workdir>` or `<sandbox_artifact_dir>`" in rendered
    assert "On CSV rejection, chart failure, or `writer_execute_budget_exhausted`" in rendered
    assert "Never reconstruct a missing or invalid dataset" in rendered


def test_orchestrator_delegates_durable_publication_exclusively_to_writer() -> None:
    prompt = _read(_AGENT_ROOT / "prompts" / "orchestrator.j2")
    rendered = render_prompt_template(
        prompt,
        current_datetime="2026-07-09",
        execution_enabled=True,
        enable_source_router=False,
        max_research_concurrency=6,
        parent_report_context_available=False,
        sandbox_workdir="/sandbox/job-123",
        sandbox_artifact_dir="/sandbox/job-123/aiq-artifacts",
        clarifier_result=None,
        user_info=None,
        tools=[],
        available_documents=None,
    )

    assert "/sandbox/job-123/aiq-artifacts/` is reserved for writer-agent durable publication" in rendered
    assert "writer-agent alone publishes CSVs, charts, and manifests" in rendered
    assert "source router, orchestrator, planner, and researcher must never generate or modify files there" in rendered


@pytest.mark.parametrize(
    "config_path",
    [
        "configs/config_domain_routing_and_skills.yml",
        "configs/config_openshell.yml",
    ],
)
def test_sandbox_profiles_assign_analysis_to_researcher_and_publication_to_writer(config_path: str) -> None:
    config = yaml.safe_load(_read(_REPO_ROOT / config_path))
    skills = config["functions"]["deep_research_skills"]
    agent = config["functions"]["deep_research_agent"]

    assert skills["agents"] == {
        "researcher-agent": ["research"],
        "writer-agent": ["synthesis"],
    }
    assert skills["require_sandbox"] == ["research", "synthesis"]
    assert "max_researcher_execute_attempts" not in agent
    assert "max_writer_execute_attempts" not in agent
    assert "workflow_timeout_seconds" not in agent
