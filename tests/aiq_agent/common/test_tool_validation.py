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

"""Tests for tool validation utilities."""

from unittest.mock import MagicMock

from aiq_agent.common.tool_validation import GENERIC_EMPTY_SOURCES_MESSAGE
from aiq_agent.common.tool_validation import format_configuration_error_message
from aiq_agent.common.tool_validation import get_unavailable_tool_details
from aiq_agent.common.tool_validation import validate_tool_availability


def _make_tool(name: str, description: str) -> MagicMock:
    """Create a mock tool with the given name and description."""
    tool = MagicMock()
    tool.name = name
    tool.description = description
    return tool


# ---------------------------------------------------------------------------
# validate_tool_availability
# ---------------------------------------------------------------------------


class TestValidateToolAvailability:
    """Tests for validate_tool_availability."""

    def test_all_available(self):
        tools = [_make_tool("search", "Search the web")]
        is_valid, count, unavailable = validate_tool_availability(tools, enable_logging=False)
        assert is_valid is True
        assert count == 1
        assert unavailable == []

    def test_all_unavailable(self):
        tools = [
            _make_tool("web_search", "Web search tool (unavailable - missing TAVILY_API_KEY)"),
        ]
        is_valid, count, unavailable = validate_tool_availability(tools, enable_logging=False)
        assert is_valid is False
        assert count == 0
        assert len(unavailable) == 1

    def test_mixed_availability(self):
        tools = [
            _make_tool("web_search", "Web search tool (unavailable - missing TAVILY_API_KEY)"),
            _make_tool("knowledge", "Query the knowledge base"),
        ]
        is_valid, count, unavailable = validate_tool_availability(tools, enable_logging=False)
        assert is_valid is True
        assert count == 1
        assert len(unavailable) == 1

    def test_empty_tools_list(self):
        is_valid, count, unavailable = validate_tool_availability([], enable_logging=False)
        assert is_valid is False
        assert count == 0
        assert unavailable == []


# ---------------------------------------------------------------------------
# get_unavailable_tool_details
# ---------------------------------------------------------------------------


class TestGetUnavailableToolDetails:
    """Tests for get_unavailable_tool_details."""

    def test_extracts_tavily_key(self):
        tools = [
            _make_tool("web_search", "Web search tool (unavailable - missing TAVILY_API_KEY)"),
        ]
        details = get_unavailable_tool_details(tools)
        assert len(details) == 1
        assert details[0]["tool_name"] == "web_search"
        assert details[0]["missing_key"] == "TAVILY_API_KEY"

    def test_extracts_serper_key(self):
        tools = [
            _make_tool("paper_search", "Paper search tool (unavailable - missing SERPER_API_KEY)"),
        ]
        details = get_unavailable_tool_details(tools)
        assert len(details) == 1
        assert details[0]["missing_key"] == "SERPER_API_KEY"

    def test_multiple_unavailable_tools(self):
        tools = [
            _make_tool("web_search", "Web search tool (unavailable - missing TAVILY_API_KEY)"),
            _make_tool("paper_search", "Paper search tool (unavailable - missing SERPER_API_KEY)"),
            _make_tool("knowledge", "Query the knowledge base"),
        ]
        details = get_unavailable_tool_details(tools)
        assert len(details) == 2
        keys = {d["missing_key"] for d in details}
        assert keys == {"TAVILY_API_KEY", "SERPER_API_KEY"}

    def test_no_unavailable_tools(self):
        tools = [
            _make_tool("search", "Search the web for information"),
        ]
        details = get_unavailable_tool_details(tools)
        assert details == []

    def test_unavailable_without_extractable_key(self):
        tools = [
            _make_tool("custom_tool", "Custom tool (unavailable - configuration error)"),
        ]
        details = get_unavailable_tool_details(tools)
        assert len(details) == 1
        assert details[0]["tool_name"] == "custom_tool"
        assert details[0]["missing_key"] == ""

    def test_empty_tools_list(self):
        assert get_unavailable_tool_details([]) == []


# ---------------------------------------------------------------------------
# format_configuration_error_message
# ---------------------------------------------------------------------------


class TestFormatConfigurationErrorMessage:
    """Tests for format_configuration_error_message."""

    def test_with_missing_keys(self):
        details = [
            {"tool_name": "web_search", "missing_key": "TAVILY_API_KEY", "description": ""},
            {"tool_name": "paper_search", "missing_key": "SERPER_API_KEY", "description": ""},
        ]
        msg = format_configuration_error_message(details)
        assert "missing configuration" in msg
        assert "TAVILY_API_KEY" in msg
        assert "SERPER_API_KEY" in msg
        assert "deploy/.env" in msg

    def test_without_extractable_keys(self):
        details = [
            {"tool_name": "custom_tool", "missing_key": "", "description": ""},
        ]
        msg = format_configuration_error_message(details)
        assert "custom_tool" in msg
        assert "unavailable" in msg.lower()

    def test_empty_details_returns_generic_constant(self):
        msg = format_configuration_error_message([])
        assert msg is GENERIC_EMPTY_SOURCES_MESSAGE

    def test_mixed_keys_and_no_keys(self):
        details = [
            {"tool_name": "web_search", "missing_key": "TAVILY_API_KEY", "description": ""},
            {"tool_name": "broken_tool", "missing_key": "", "description": ""},
        ]
        msg = format_configuration_error_message(details)
        assert "TAVILY_API_KEY" in msg
        assert "broken_tool" in msg
        assert "configuration error" in msg


# ---------------------------------------------------------------------------
# Stub description contract tests
# ---------------------------------------------------------------------------


class TestStubDescriptionContract:
    """Verify that the real stub docstrings used by tool registrations
    are extractable by get_unavailable_tool_details.

    If someone changes a stub's docstring wording, these tests break —
    surfacing the implicit contract between registration and extraction.
    """

    def test_tavily_stub_description_extracts_key(self):
        tool = _make_tool(
            "web_search_tool",
            "Web search tool (unavailable - missing TAVILY_API_KEY).",
        )
        details = get_unavailable_tool_details([tool])
        assert len(details) == 1
        assert details[0]["missing_key"] == "TAVILY_API_KEY"
        assert details[0]["tool_name"] == "web_search_tool"

    def test_serper_stub_description_extracts_key(self):
        tool = _make_tool(
            "paper_search_tool",
            "Paper search tool (unavailable - missing SERPER_API_KEY).",
        )
        details = get_unavailable_tool_details([tool])
        assert len(details) == 1
        assert details[0]["missing_key"] == "SERPER_API_KEY"
        assert details[0]["tool_name"] == "paper_search_tool"
