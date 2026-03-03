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

"""Tests for source_metadata encode/extract helpers."""

from aiq_agent.common.source_metadata import SourceRef
from aiq_agent.common.source_metadata import encode_source_metadata
from aiq_agent.common.source_metadata import extract_source_metadata


class TestEncodeSourceMetadata:
    """Tests for encode_source_metadata."""

    def test_appends_metadata_block(self):
        refs = [SourceRef(url="https://example.com", title="Example")]
        result = encode_source_metadata("Hello world", refs)
        assert result.startswith("Hello world")
        assert "TOOL_SOURCE_METADATA" in result
        assert "https://example.com" in result

    def test_empty_sources_returns_content_unchanged(self):
        result = encode_source_metadata("Hello world", [])
        assert result == "Hello world"

    def test_multiple_sources(self):
        refs = [
            SourceRef(url="https://a.com", title="A"),
            SourceRef(url="https://b.com", title="B"),
        ]
        result = encode_source_metadata("content", refs)
        assert "https://a.com" in result
        assert "https://b.com" in result

    def test_citation_key_sources(self):
        refs = [SourceRef(citation_key="report.pdf, p.5", title="report.pdf")]
        result = encode_source_metadata("content", refs)
        assert "report.pdf" in result
        assert "citation_key" in result

    def test_excludes_none_fields(self):
        refs = [SourceRef(url="https://x.com")]
        result = encode_source_metadata("content", refs)
        assert "title" not in result
        assert "citation_key" not in result


class TestExtractSourceMetadata:
    """Tests for extract_source_metadata."""

    def test_extracts_metadata_and_strips(self):
        refs = [SourceRef(url="https://example.com", title="Example")]
        encoded = encode_source_metadata("Hello world", refs)
        clean, extracted = extract_source_metadata(encoded)
        assert clean == "Hello world"
        assert extracted is not None
        assert len(extracted) == 1
        assert extracted[0].url == "https://example.com"
        assert extracted[0].title == "Example"

    def test_no_metadata_returns_none(self):
        clean, extracted = extract_source_metadata("Just plain text")
        assert clean == "Just plain text"
        assert extracted is None

    def test_roundtrip_multiple_sources(self):
        refs = [
            SourceRef(url="https://a.com", title="A"),
            SourceRef(url="https://b.com", title="B"),
            SourceRef(citation_key="doc.pdf, p.3"),
        ]
        encoded = encode_source_metadata("content", refs)
        clean, extracted = extract_source_metadata(encoded)
        assert clean == "content"
        assert len(extracted) == 3
        assert extracted[0].url == "https://a.com"
        assert extracted[2].citation_key == "doc.pdf, p.3"

    def test_malformed_json_returns_none(self):
        bad = "content\n\n<!-- TOOL_SOURCE_METADATA:{not valid json -->"
        clean, extracted = extract_source_metadata(bad)
        assert clean == bad
        assert extracted is None

    def test_empty_string(self):
        clean, extracted = extract_source_metadata("")
        assert clean == ""
        assert extracted is None

    def test_content_with_html_comments_not_confused(self):
        text = "Some <!-- regular comment --> text"
        clean, extracted = extract_source_metadata(text)
        assert clean == text
        assert extracted is None

    def test_preserves_multiline_content(self):
        content = "Line 1\n\nLine 2\n\nLine 3"
        refs = [SourceRef(url="https://x.com")]
        encoded = encode_source_metadata(content, refs)
        clean, extracted = extract_source_metadata(encoded)
        assert clean == content
        assert len(extracted) == 1
