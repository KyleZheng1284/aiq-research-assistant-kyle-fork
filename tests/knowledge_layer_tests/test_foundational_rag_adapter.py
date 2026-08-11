# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for the Foundational RAG Knowledge Layer adapter."""

from unittest.mock import patch

from knowledge_layer.foundational_rag.adapter import COLLECTION_TTL_HOURS
from knowledge_layer.foundational_rag.adapter import TTL_CLEANUP_INTERVAL_SECONDS
from knowledge_layer.foundational_rag.adapter import FoundationalRagIngestor
from knowledge_layer.register import KnowledgeRetrievalConfig
from knowledge_layer.register import _setup_backend


def test_ttl_cleanup_starts_by_default() -> None:
    """Preserve automatic cleanup for existing deployments that omit the option."""
    with patch.object(FoundationalRagIngestor, "_start_ttl_cleanup_task") as start_cleanup:
        FoundationalRagIngestor()

    start_cleanup.assert_called_once_with(COLLECTION_TTL_HOURS, TTL_CLEANUP_INTERVAL_SECONDS)


def test_ttl_cleanup_can_be_disabled() -> None:
    """Allow externally managed shared collections to opt out of client-side deletion."""
    with patch.object(FoundationalRagIngestor, "_start_ttl_cleanup_task") as start_cleanup:
        FoundationalRagIngestor({"start_ttl_cleanup": False})

    start_cleanup.assert_not_called()


def test_foundational_rag_config_forwards_ttl_cleanup_setting() -> None:
    """Expose the adapter option without changing the existing default."""
    default_config = KnowledgeRetrievalConfig(backend="foundational_rag")
    _, default_backend_config = _setup_backend(default_config)

    assert default_backend_config["start_ttl_cleanup"] is True

    config = KnowledgeRetrievalConfig(backend="foundational_rag", start_ttl_cleanup=False)
    backend, backend_config = _setup_backend(config)

    assert backend == "foundational_rag"
    assert backend_config["start_ttl_cleanup"] is False
