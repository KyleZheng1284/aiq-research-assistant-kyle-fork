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

"""Tool validation utilities for checking tool availability."""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_API_KEY_PATTERN = re.compile(r"missing\s+([A-Z][A-Z0-9_]*(?:_KEY|_TOKEN|_SECRET))", re.IGNORECASE)

GENERIC_EMPTY_SOURCES_MESSAGE = (
    "The search tools did not return any results for this question. "
    "This may be due to a temporary issue or the question may need to be rephrased. "
    "Please try again."
)


def validate_tool_availability(
    tools: list[Any],
    research_type: str = "research",
    enable_logging: bool = True,
) -> tuple[bool, int, list[str]]:
    """
    Validate that at least one tool is available.

    Args:
        tools: List of tools to validate
        research_type: Type of research (e.g., "shallow research", "deep research") for logging
        enable_logging: Whether to log tool availability information

    Returns:
        Tuple of (is_valid, available_count, unavailable_tools):
        - is_valid: True if at least one tool is available
        - available_count: Number of available tools
        - unavailable_tools: List of unavailable tool names with reasons
    """
    available_tools_count = 0
    unavailable_tools = []

    if enable_logging:
        logger.info("Checking %d tools for %s", len(tools), research_type)

    for tool in tools:
        tool_name = getattr(tool, "name", "").lower()
        tool_desc = getattr(tool, "description", "").lower() or ""

        # Check if tool is unavailable (stub)
        is_unavailable = "unavailable" in tool_desc or "missing" in tool_desc

        if is_unavailable:
            reason = "missing or invalid API key or config error"
            if enable_logging:
                logger.info("Tool %s is unavailable: %s", tool_name, reason)
            unavailable_tools.append(f"{tool_name} - {reason}")
        else:
            available_tools_count += 1
            if enable_logging:
                logger.info("Found available tool: %s", tool_name)

    if enable_logging:
        logger.info(
            "Tool availability check: %d available tools out of %d",
            available_tools_count,
            len(tools),
        )

    return available_tools_count > 0, available_tools_count, unavailable_tools


def get_unavailable_tool_details(tools: list[Any]) -> list[dict[str, str]]:
    """Extract structured details about unavailable (stub) tools.

    Inspects tool descriptions for "unavailable"/"missing" markers set at
    registration time and extracts the specific API key name when present.

    Args:
        tools: List of tools to inspect.

    Returns:
        List of dicts with keys ``tool_name``, ``missing_key`` (empty string
        if not extractable), and ``description``.
    """
    details: list[dict[str, str]] = []
    for tool in tools:
        tool_desc = getattr(tool, "description", "") or ""
        desc_lower = tool_desc.lower()
        if "unavailable" not in desc_lower and "missing" not in desc_lower:
            continue

        tool_name = getattr(tool, "name", "unknown")
        match = _API_KEY_PATTERN.search(tool_desc)
        missing_key = match.group(1) if match else ""

        details.append(
            {
                "tool_name": tool_name,
                "missing_key": missing_key,
                "description": tool_desc,
            }
        )
    return details


def format_configuration_error_message(
    unavailable_details: list[dict[str, str]],
) -> str:
    """Format a user-facing error for research failures caused by missing configuration.

    Produces an actionable message that tells the user exactly which API keys
    are missing and how to fix the problem, instead of a generic "try again".

    Args:
        unavailable_details: Output of :func:`get_unavailable_tool_details`.

    Returns:
        User-friendly error message string.
    """
    if not unavailable_details:
        return GENERIC_EMPTY_SOURCES_MESSAGE

    has_missing_keys = any(d.get("missing_key") for d in unavailable_details)

    if has_missing_keys:
        key_lines = []
        for detail in unavailable_details:
            if detail.get("missing_key"):
                key_lines.append(f"  - {detail['missing_key']} (required by {detail['tool_name']})")
            else:
                key_lines.append(f"  - {detail['tool_name']} (configuration error)")

        return (
            "Research could not be completed because required search tools are "
            "unavailable due to missing configuration.\n\n"
            "Missing API keys:\n" + "\n".join(key_lines) + "\n\n"
            "Please set these in deploy/.env and restart the application."
        )

    tool_names = [d["tool_name"] for d in unavailable_details]
    return (
        "Research could not be completed because the following search tools "
        "are unavailable:\n" + "\n".join(f"  - {name}" for name in tool_names) + "\n\n"
        "Please check your configuration and restart the application."
    )
