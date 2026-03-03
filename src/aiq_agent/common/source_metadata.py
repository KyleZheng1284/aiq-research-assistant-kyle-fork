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
"""Structured source metadata for tool returns.

Tools return strings (NAT requirement) but can embed typed source metadata
as an HTML comment at the end.  The SourceRegistryMiddleware extracts this
metadata directly instead of regex-parsing the free-form string, then strips
the block so the LLM never sees it.

Usage in a tool:
    from aiq_agent.common.source_metadata import SourceRef, encode_source_metadata

    refs = [SourceRef(url="https://example.com", title="Example")]
    return encode_source_metadata(formatted_string, refs)

Usage in middleware:
    from aiq_agent.common.source_metadata import extract_source_metadata

    clean_content, refs = extract_source_metadata(tool_message.content)
"""

from __future__ import annotations

import json
import logging
import re

from pydantic import BaseModel

logger = logging.getLogger(__name__)

_METADATA_TAG = "TOOL_SOURCE_METADATA"
_METADATA_RE = re.compile(
    rf"\n?\n?<!-- {_METADATA_TAG}:(.*) -->$",
    re.DOTALL,
)


class SourceRef(BaseModel):
    """A single source reference from a tool result."""

    url: str | None = None
    title: str | None = None
    citation_key: str | None = None


def encode_source_metadata(content: str, sources: list[SourceRef]) -> str:
    """Append structured source metadata to a tool's string return.

    The metadata is encoded as an HTML comment that the middleware will
    extract and strip before the LLM sees the content.
    """
    if not sources:
        return content
    payload = json.dumps([s.model_dump(exclude_none=True) for s in sources], separators=(",", ":"))
    return f"{content}\n\n<!-- {_METADATA_TAG}:{payload} -->"


def extract_source_metadata(content: str) -> tuple[str, list[SourceRef] | None]:
    """Extract and strip structured source metadata from tool content.

    Returns:
        Tuple of (clean_content, sources_or_none).
        If no metadata block is found, returns (original_content, None).
    """
    match = _METADATA_RE.search(content)
    if not match:
        return content, None
    try:
        raw = json.loads(match.group(1))
        refs = [SourceRef(**item) for item in raw]
        clean = content[: match.start()]
        return clean, refs
    except Exception:
        logger.warning("Failed to parse tool source metadata; falling back to regex", exc_info=True)
        return content, None
