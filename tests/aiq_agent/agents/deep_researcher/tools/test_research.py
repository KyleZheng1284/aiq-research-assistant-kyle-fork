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

"""Tests for canonical quantitative evidence persistence and registration."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain.agents.middleware import ModelRetryMiddleware
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages import ToolMessage

from aiq_agent.agents.deep_researcher.factory import build_researcher_runnable
from aiq_agent.agents.deep_researcher.models import ResearchNotes
from aiq_agent.agents.deep_researcher.models import ResearchQuery
from aiq_agent.agents.deep_researcher.tools.research import _canonical_dataset_digests
from aiq_agent.agents.deep_researcher.tools.research import _register_canonical_dataset_digests
from aiq_agent.agents.deep_researcher.tools.research import _research_note_files
from aiq_agent.agents.deep_researcher.tools.research import _run_research_query
from aiq_agent.agents.deep_researcher.tools.research import build_research_batch_tool


class _CountingToolBindingChatModel(FakeMessagesListChatModel):
    """Scripted chat model that counts and accepts structured-output tool binding."""

    call_count: int = 0
    correction_messages: list[list[str]] = []

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.call_count += 1
        self.correction_messages.append(
            [str(message.content) for message in messages if isinstance(message, ToolMessage)]
        )
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def _quantitative_note(csv_text: str) -> ResearchNotes:
    return ResearchNotes.model_validate(
        {
            "query_topic": "Quarterly revenue",
            "target_components": ["revenue_table"],
            "summary": "Revenue increased.",
            "findings": [],
            "gaps": [],
            "sources": [
                {
                    "id": 1,
                    "title": "Quarterly results",
                    "source_type": "url",
                    "locator": "https://example.test/results",
                }
            ],
            "narrative_notes": "Canonical quarterly values.",
            "language": "English",
            "quantitative_datasets": [
                {
                    "dataset_id": "quarterly_revenue",
                    "title": "Quarterly revenue",
                    "csv_text": csv_text,
                    "markdown_table": "| Quarter | Revenue |\n|---|---:|\n| Q1 | 22.6 |",
                    "summary": "Mean revenue was 24.45.",
                    "source_ids": [1],
                    "caveats": [],
                }
            ],
        }
    )


@pytest.mark.asyncio
async def test_repeated_invalid_quantitative_output_is_bounded_and_salvages_textual_notes():
    """Invalid datasets cannot loop; bounded retry exhaustion drops only publication data."""
    invalid_payload = _quantitative_note("quarter,revenue\nQ1,22.6\n").model_dump(mode="json")
    invalid_payload["quantitative_datasets"][0]["csv_text"] = "quarter,revenue\n"
    invalid_response = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "ResearchNotes",
                "args": invalid_payload,
                "id": "invalid-research-notes",
                "type": "tool_call",
            }
        ],
    )
    model = _CountingToolBindingChatModel(responses=[invalid_response])
    runnable = build_researcher_runnable(
        researcher_model=model,
        researcher_tools=[],
        researcher_middleware=[
            ModelRetryMiddleware(
                max_retries=2,
                initial_delay=0,
                jitter=False,
            )
        ],
        system_prompt="Return ResearchNotes.",
    )
    query = ResearchQuery(
        query="NVIDIA quarterly revenue",
        preferred_tools=["web_search_tool"],
        target_components=["revenue_table"],
        rationale="Build the quantitative evidence.",
    )

    note = await _run_research_query(
        query=query,
        researcher_runnable=runnable,
        runtime=None,
        callbacks=[],
        semaphore=asyncio.Semaphore(1),
    )

    assert model.call_count == 3
    assert model.correction_messages[0] == []
    assert "bounded correction 1 of 2" in model.correction_messages[1][-1]
    assert "bounded correction 2 of 2" in model.correction_messages[2][-1]
    assert note.summary == "Revenue increased."
    assert note.narrative_notes == "Canonical quarterly values."
    assert note.quantitative_datasets == []


def test_research_note_persistence_preserves_canonical_csv_and_digest_exactly():
    csv_text = 'quarter,revenue,note\r\nQ1,22.6,"reported, rounded"\r\nQ2,26.3,"line 1\nline 2"'
    note = _quantitative_note(csv_text)
    query = ResearchQuery(
        query="NVIDIA quarterly revenue",
        preferred_tools=["web_search_tool"],
        target_components=["revenue_table"],
        rationale="Build the quantitative evidence.",
    )

    [(_path, content)] = _research_note_files([query], [note])
    persisted = json.loads(content.decode("utf-8"))
    dataset = persisted["quantitative_datasets"][0]

    assert dataset["csv_text"] == csv_text
    assert dataset["csv_sha256"] == note.quantitative_datasets[0].csv_sha256
    assert ResearchNotes.model_validate(persisted).quantitative_datasets[0].csv_text == csv_text


def test_validated_canonical_digests_are_registered_with_job_artifact_manager():
    note = _quantitative_note("quarter,revenue\nQ1,22.6\n")
    artifact_manager = MagicMock()

    _register_canonical_dataset_digests(artifact_manager=artifact_manager, notes=[note])

    artifact_manager.register_canonical_digests.assert_called_once_with([note.quantitative_datasets[0].csv_sha256])


def test_digest_registration_recomputes_and_rejects_a_stale_model_copy():
    note = _quantitative_note("quarter,revenue\nQ1,22.6\n")
    stale_dataset = note.quantitative_datasets[0].model_copy(update={"csv_text": "quarter,revenue\nQ1,99\n"})
    stale_note = note.model_copy(update={"quantitative_datasets": [stale_dataset]})

    with pytest.raises(ValueError, match="no longer matches"):
        _canonical_dataset_digests([stale_note])


@pytest.mark.asyncio
async def test_research_batch_registers_validated_digest_before_returning_notes():
    note = _quantitative_note("quarter,revenue\nQ1,22.6\n")
    artifact_manager = MagicMock()

    class _ResearcherRunnable:
        async def ainvoke(self, _state, config=None):
            return {"structured_response": note.model_dump(mode="json")}

    class _Backend:
        def upload_files(self, files):
            return [SimpleNamespace(path=path, error=None) for path, _content in files]

    batch_tool = build_research_batch_tool(
        researcher_runnable=_ResearcherRunnable(),
        callbacks=[],
        max_research_concurrency=1,
        backend=_Backend(),
        artifact_manager=artifact_manager,
    )
    await batch_tool.ainvoke(
        {
            "queries": [
                {
                    "query": "NVIDIA quarterly revenue",
                    "preferred_tools": ["web_search_tool"],
                    "target_components": ["revenue_table"],
                    "rationale": "Build the quantitative evidence.",
                }
            ]
        }
    )

    artifact_manager.register_canonical_digests.assert_called_once_with([note.quantitative_datasets[0].csv_sha256])


@pytest.mark.asyncio
async def test_research_batch_refuses_digest_registration_without_note_persistence():
    note = _quantitative_note("quarter,revenue\nQ1,22.6\n")

    class _ResearcherRunnable:
        async def ainvoke(self, _state, config=None):
            return {"structured_response": note.model_dump(mode="json")}

    artifact_manager = MagicMock()
    batch_tool = build_research_batch_tool(
        researcher_runnable=_ResearcherRunnable(),
        callbacks=[],
        max_research_concurrency=1,
        artifact_manager=artifact_manager,
    )

    with pytest.raises(RuntimeError, match="before ResearchNotes persistence"):
        await batch_tool.ainvoke(
            {
                "queries": [
                    {
                        "query": "NVIDIA quarterly revenue",
                        "preferred_tools": ["web_search_tool"],
                        "target_components": ["revenue_table"],
                        "rationale": "Build the quantitative evidence.",
                    }
                ]
            }
        )

    artifact_manager.register_canonical_digests.assert_not_called()


@pytest.mark.asyncio
async def test_research_batch_persists_notes_before_registering_digest():
    note = _quantitative_note("quarter,revenue\nQ1,22.6\n")
    events: list[str] = []

    class _ResearcherRunnable:
        async def ainvoke(self, _state, config=None):
            return {"structured_response": note.model_dump(mode="json")}

    class _Backend:
        def upload_files(self, files):
            events.append("persist")
            return [SimpleNamespace(path=path, error=None) for path, _content in files]

    artifact_manager = MagicMock()
    artifact_manager.register_canonical_digests.side_effect = lambda _digests: events.append("register")
    batch_tool = build_research_batch_tool(
        researcher_runnable=_ResearcherRunnable(),
        callbacks=[],
        max_research_concurrency=1,
        backend=_Backend(),
        artifact_manager=artifact_manager,
    )

    await batch_tool.ainvoke(
        {
            "queries": [
                {
                    "query": "NVIDIA quarterly revenue",
                    "preferred_tools": ["web_search_tool"],
                    "target_components": ["revenue_table"],
                    "rationale": "Build the quantitative evidence.",
                }
            ]
        }
    )

    assert events == ["persist", "register"]


@pytest.mark.asyncio
async def test_research_batch_does_not_register_digest_when_persistence_fails():
    note = _quantitative_note("quarter,revenue\nQ1,22.6\n")

    class _ResearcherRunnable:
        async def ainvoke(self, _state, config=None):
            return {"structured_response": note.model_dump(mode="json")}

    class _Backend:
        def upload_files(self, _files):
            raise RuntimeError("persistence unavailable")

    artifact_manager = MagicMock()
    batch_tool = build_research_batch_tool(
        researcher_runnable=_ResearcherRunnable(),
        callbacks=[],
        max_research_concurrency=1,
        backend=_Backend(),
        artifact_manager=artifact_manager,
    )

    with pytest.raises(RuntimeError, match="persistence unavailable"):
        await batch_tool.ainvoke(
            {
                "queries": [
                    {
                        "query": "NVIDIA quarterly revenue",
                        "preferred_tools": ["web_search_tool"],
                        "target_components": ["revenue_table"],
                        "rationale": "Build the quantitative evidence.",
                    }
                ]
            }
        )

    artifact_manager.register_canonical_digests.assert_not_called()


def test_digest_registration_is_a_noop_without_datasets_or_manager():
    note = _quantitative_note("quarter,revenue\nQ1,22.6\n").model_copy(update={"quantitative_datasets": []})
    artifact_manager = MagicMock()

    _register_canonical_dataset_digests(artifact_manager=artifact_manager, notes=[note])
    _register_canonical_dataset_digests(artifact_manager=None, notes=[_quantitative_note("a\n1\n")])

    artifact_manager.register_canonical_digests.assert_not_called()
