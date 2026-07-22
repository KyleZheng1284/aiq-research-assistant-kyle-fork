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

"""Tests for the artifact runtime: manifest parsing, MIME sniffing, the harvest
validation pipeline (traversal/extension/size/quota/dedup/scan), and SQL or S3-compatible stores."""

from __future__ import annotations

import json
import logging
from hashlib import sha256
from io import BytesIO
from types import SimpleNamespace
from typing import Any

import pytest

from aiq_agent.agents.deep_researcher.sandbox.artifacts import Artifact
from aiq_agent.agents.deep_researcher.sandbox.artifacts import ArtifactKind
from aiq_agent.agents.deep_researcher.sandbox.artifacts import ArtifactManager
from aiq_agent.agents.deep_researcher.sandbox.artifacts import ArtifactStatus
from aiq_agent.agents.deep_researcher.sandbox.artifacts import S3ArtifactBlobStore
from aiq_agent.agents.deep_researcher.sandbox.artifacts import SqlArtifactStore
from aiq_agent.agents.deep_researcher.sandbox.artifacts import build_artifact_store
from aiq_agent.agents.deep_researcher.sandbox.artifacts import parse_manifest
from aiq_agent.agents.deep_researcher.sandbox.artifacts.manager import _normalize_posix
from aiq_agent.agents.deep_researcher.sandbox.artifacts.manager import _sniff_mime
from aiq_agent.agents.deep_researcher.sandbox.config import ArtifactCaptureConfig

_ARTIFACT_DIR = "/workspace/aiq-artifacts"
_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
_PNG_SHA256 = sha256(_PNG).hexdigest()


class _FakeBackend:
    """Minimal BaseSandbox stand-in mapping sandbox paths to bytes."""

    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files
        self.execute_calls: list[str] = []
        self.download_calls: list[list[str]] = []

    def download_files(self, paths: list[str]) -> list[Any]:
        self.download_calls.append(list(paths))
        return [
            SimpleNamespace(path=p, content=self.files.get(p), error=None if p in self.files else "not found")
            for p in paths
        ]

    def execute(self, command: str, *, timeout: int | None = None) -> Any:
        self.execute_calls.append(command)
        return SimpleNamespace(output="\n".join(self.files), exit_code=0)


def _manifest_bytes(path: str, kind: str = "image", expected_sha256: str | None = None) -> bytes:
    entry = {"path": path, "kind": kind, "inline": True}
    if expected_sha256 is not None:
        entry["expected_sha256"] = expected_sha256
    return json.dumps({"version": 1, "artifacts": [entry]}).encode("utf-8")


def _make_manager(store: Any, files: dict[str, bytes], **capture: Any) -> tuple[ArtifactManager, list]:
    emitted: list = []
    config = ArtifactCaptureConfig(enabled=True, **capture)
    manager = ArtifactManager(
        job_id="job-1",
        backend=_FakeBackend(files),
        store=store,
        config=config,
        artifact_dir=_ARTIFACT_DIR,
        emit=emitted.append,
    )
    return manager, emitted


class TestManifest:
    def test_parse_valid(self) -> None:
        manifest = parse_manifest(_manifest_bytes(f"{_ARTIFACT_DIR}/c.png").decode())
        assert manifest is not None
        assert manifest.artifacts[0].path == f"{_ARTIFACT_DIR}/c.png"

    def test_parse_invalid_returns_none(self) -> None:
        assert parse_manifest("not json{") is None

    def test_parse_expected_sha256(self) -> None:
        digest = "a" * 64
        manifest = parse_manifest(_manifest_bytes(f"{_ARTIFACT_DIR}/data.csv", "dataset", digest).decode())
        assert manifest is not None
        assert manifest.artifacts[0].expected_sha256 == digest


class TestSniffMime:
    def test_png_by_magic(self) -> None:
        assert _sniff_mime(_PNG, "x.bin") == "image/png"

    def test_csv_by_extension(self) -> None:
        assert _sniff_mime(b"a,b\n1,2\n", "data.csv") == "text/csv"


class TestNormalizePosix:
    def test_absolute_path_has_single_leading_slash(self) -> None:
        from pathlib import PurePosixPath

        # The absolute-root sentinel must not be re-appended as a path segment.
        assert _normalize_posix(PurePosixPath("/sandbox/aiq-artifacts")) == "/sandbox/aiq-artifacts"

    def test_collapses_dot_and_parent_segments(self) -> None:
        from pathlib import PurePosixPath

        assert _normalize_posix(PurePosixPath("/sandbox/./sub/../aiq-artifacts")) == "/sandbox/aiq-artifacts"

    def test_relative_path_has_no_leading_slash(self) -> None:
        from pathlib import PurePosixPath

        assert _normalize_posix(PurePosixPath("sub/aiq-artifacts")) == "sub/aiq-artifacts"


class TestHarvest:
    def test_store_and_scan_failures_do_not_log_exception_text(self, caplog: pytest.LogCaptureFixture) -> None:
        store = SimpleNamespace(list=lambda _job_id: (_ for _ in ()).throw(RuntimeError("credential=do-not-log")))
        manager, _ = _make_manager(store, {})

        with caplog.at_level(logging.WARNING):
            assert manager.resolve_report_references("report") == "report"
            manager.backend.execute = lambda *_args, **_kwargs: (_ for _ in ()).throw(  # type: ignore[method-assign]
                RuntimeError("token=do-not-log")
            )
            assert manager._scan_dir() == []

        assert "RuntimeError" in caplog.text
        assert "credential=do-not-log" not in caplog.text
        assert "token=do-not-log" not in caplog.text

    def test_captures_manifest_artifact(self, tmp_path: Any) -> None:
        store = SqlArtifactStore(f"sqlite:///{tmp_path}/jobs.db")
        png_path = f"{_ARTIFACT_DIR}/chart.png"
        files = {f"{_ARTIFACT_DIR}/manifest.json": _manifest_bytes(png_path), png_path: _PNG}
        manager, emitted = _make_manager(store, files)

        captured = manager.final_harvest()

        assert len(captured) == 1
        assert captured[0].mime_type == "image/png"
        assert captured[0].kind == ArtifactKind.IMAGE
        assert store.list("job-1")[0].filename == "chart.png"
        assert emitted and emitted[0]["type"] == "artifact.update"
        assert emitted[0]["name"] == "chart.png"
        assert emitted[0]["data"]["type"] == "file"
        assert emitted[0]["data"]["artifact_id"] == captured[0].artifact_id
        assert emitted[0]["data"]["content_url"].endswith(f"/{captured[0].artifact_id}/content")
        assert "content" not in emitted[0]["data"]  # bytes and URL-as-text never enter the payload

    def test_rejects_path_traversal(self, tmp_path: Any) -> None:
        store = SqlArtifactStore(f"sqlite:///{tmp_path}/jobs.db")
        evil = "/etc/passwd.png"
        files = {f"{_ARTIFACT_DIR}/manifest.json": _manifest_bytes(evil), evil: _PNG}
        manager, _ = _make_manager(store, files)
        assert manager.final_harvest() == []

    def test_enforces_extension_allowlist(self, tmp_path: Any) -> None:
        store = SqlArtifactStore(f"sqlite:///{tmp_path}/jobs.db")
        exe = f"{_ARTIFACT_DIR}/evil.exe"
        files = {f"{_ARTIFACT_DIR}/manifest.json": _manifest_bytes(exe), exe: _PNG}
        manager, _ = _make_manager(store, files)
        assert manager.final_harvest() == []

    def test_enforces_size_cap(self, tmp_path: Any) -> None:
        store = SqlArtifactStore(f"sqlite:///{tmp_path}/jobs.db")
        png_path = f"{_ARTIFACT_DIR}/chart.png"
        files = {f"{_ARTIFACT_DIR}/manifest.json": _manifest_bytes(png_path), png_path: _PNG}
        manager, _ = _make_manager(store, files, max_file_bytes=8)
        assert manager.final_harvest() == []

    def test_enforces_quota(self, tmp_path: Any) -> None:
        store = SqlArtifactStore(f"sqlite:///{tmp_path}/jobs.db")
        a = f"{_ARTIFACT_DIR}/a.png"
        b = f"{_ARTIFACT_DIR}/b.png"
        manifest = json.dumps(
            {"version": 1, "artifacts": [{"path": a, "kind": "image"}, {"path": b, "kind": "image"}]}
        ).encode("utf-8")
        files = {f"{_ARTIFACT_DIR}/manifest.json": manifest, a: _PNG, b: _PNG + b"x"}
        manager, _ = _make_manager(store, files, max_file_count=1)
        captured = manager.final_harvest()
        assert len(captured) == 1

    def test_dedups_identical_content(self, tmp_path: Any) -> None:
        store = SqlArtifactStore(f"sqlite:///{tmp_path}/jobs.db")
        png_path = f"{_ARTIFACT_DIR}/chart.png"
        files = {f"{_ARTIFACT_DIR}/manifest.json": _manifest_bytes(png_path), png_path: _PNG}
        manager, _ = _make_manager(store, files)
        manager.final_harvest()
        first_downloads = list(manager.backend.download_calls)
        manager.final_harvest()  # same bytes again
        assert len(store.list("job-1")) == 1
        assert manager.backend.download_calls == first_downloads

    def test_checkpoint_harvest_uses_manifest_without_directory_scan(self, tmp_path: Any) -> None:
        store = SqlArtifactStore(f"sqlite:///{tmp_path}/jobs.db")
        png_path = f"{_ARTIFACT_DIR}/chart.png"
        files = {f"{_ARTIFACT_DIR}/manifest.json": _manifest_bytes(png_path), png_path: _PNG}
        manager, _ = _make_manager(store, files)

        captured = manager.harvest_after_execute()

        assert [artifact.filename for artifact in captured] == ["chart.png"]
        assert manager.backend.execute_calls == []

    def test_final_harvest_after_checkpoint_does_not_reemit(self, tmp_path: Any) -> None:
        store = SqlArtifactStore(f"sqlite:///{tmp_path}/jobs.db")
        png_path = f"{_ARTIFACT_DIR}/chart.png"
        csv_path = f"{_ARTIFACT_DIR}/chart.csv"
        files = {
            f"{_ARTIFACT_DIR}/manifest.json": _manifest_bytes(png_path),
            png_path: _PNG,
            csv_path: b"state,pop\nCA,39431263\n",
        }
        manager, emitted = _make_manager(store, files)

        checkpointed = manager.harvest_after_execute()
        assert [artifact.filename for artifact in checkpointed] == ["chart.png"]

        finalized = manager.final_harvest()
        # chart.png was already captured at checkpoint; only the scan-discovered CSV is new.
        assert [artifact.filename for artifact in finalized] == ["chart.csv"]
        assert len(emitted) == 2

    def test_scan_fallback_without_manifest(self, tmp_path: Any) -> None:
        store = SqlArtifactStore(f"sqlite:///{tmp_path}/jobs.db")
        png_path = f"{_ARTIFACT_DIR}/chart.png"
        files = {png_path: _PNG}  # no manifest
        manager, _ = _make_manager(store, files)
        captured = manager.final_harvest()  # job_end allows scan
        assert len(captured) == 1
        assert captured[0].filename == "chart.png"

    def test_noncanonical_csv_scan_without_manifest_remains_backward_compatible(self, tmp_path: Any) -> None:
        store = SqlArtifactStore(f"sqlite:///{tmp_path}/jobs.db")
        csv_path = f"{_ARTIFACT_DIR}/legacy.csv"
        csv_bytes = b"label,value\nA,1\n"
        manager, _ = _make_manager(store, {csv_path: csv_bytes})

        captured = manager.final_harvest()

        assert [artifact.filename for artifact in captured] == ["legacy.csv"]
        assert captured[0].kind == ArtifactKind.TABLE
        assert captured[0].sha256 == sha256(csv_bytes).hexdigest()

    def test_registered_canonical_path_is_digest_verified_without_manifest(self, tmp_path: Any) -> None:
        store = SqlArtifactStore(f"sqlite:///{tmp_path}/jobs.db")
        csv_path = f"{_ARTIFACT_DIR}/quarterly_revenue.csv"
        csv_bytes = b"quarter,revenue\nQ1,10\n"
        digest = sha256(csv_bytes).hexdigest()
        manager, _ = _make_manager(store, {csv_path: csv_bytes})
        manager.register_canonical_datasets([("quarterly_revenue", digest)])

        captured = manager.final_harvest()

        assert [artifact.filename for artifact in captured] == ["quarterly_revenue.csv"]
        assert captured[0].kind == ArtifactKind.DATASET
        assert captured[0].sha256 == digest

    def test_registered_canonical_path_rejects_scan_bytes_with_wrong_digest(self, tmp_path: Any) -> None:
        store = SqlArtifactStore(f"sqlite:///{tmp_path}/jobs.db")
        csv_path = f"{_ARTIFACT_DIR}/quarterly_revenue.csv"
        expected = sha256(b"quarter,revenue\nQ1,10\n").hexdigest()
        manager, _ = _make_manager(store, {csv_path: b"quarter,revenue\nQ1,999\n"})
        manager.register_canonical_datasets([("quarterly_revenue", expected)])

        assert manager.final_harvest() == []
        assert manager.last_harvest_rejections() == (("quarterly_revenue.csv", "canonical_dataset_digest_mismatch"),)
        assert store.list("job-1") == []

    def test_registered_digest_cannot_move_to_a_different_canonical_path(self, tmp_path: Any) -> None:
        store = SqlArtifactStore(f"sqlite:///{tmp_path}/jobs.db")
        registered_bytes = b"quarter,revenue\nQ1,10\n"
        digest = sha256(registered_bytes).hexdigest()
        unknown_path = f"{_ARTIFACT_DIR}/unknown.csv"
        files = {
            f"{_ARTIFACT_DIR}/manifest.json": _manifest_bytes(unknown_path, "dataset", digest),
            unknown_path: registered_bytes,
        }
        manager, _ = _make_manager(store, files)
        manager.register_canonical_datasets([("quarterly_revenue", digest)])

        assert manager.final_harvest() == []
        assert manager.last_harvest_rejections() == (("unknown.csv", "canonical_dataset_digest_unregistered"),)
        assert store.list("job-1") == []

    def test_final_harvest_unions_manifest_and_scan(self, tmp_path: Any) -> None:
        # A manifest that declares only the PNG must not hide a sibling CSV at job end.
        store = SqlArtifactStore(f"sqlite:///{tmp_path}/jobs.db")
        png_path = f"{_ARTIFACT_DIR}/chart.png"
        csv_path = f"{_ARTIFACT_DIR}/chart.csv"
        files = {
            f"{_ARTIFACT_DIR}/manifest.json": _manifest_bytes(png_path),
            png_path: _PNG,
            csv_path: b"state,pop\nCA,39431263\n",
        }
        manager, _ = _make_manager(store, files)

        captured = manager.final_harvest()

        names = sorted(a.filename for a in captured)
        assert names == ["chart.csv", "chart.png"]

    def test_canonical_dataset_capture_requires_registered_exact_digest(self, tmp_path: Any) -> None:
        store = SqlArtifactStore(f"sqlite:///{tmp_path}/jobs.db")
        csv_path = f"{_ARTIFACT_DIR}/revenue.csv"
        csv_bytes = b"quarter,revenue\nQ1,22.6\nQ2,26.3\n"
        digest = sha256(csv_bytes).hexdigest()
        files = {
            f"{_ARTIFACT_DIR}/manifest.json": _manifest_bytes(csv_path, "dataset", digest),
            csv_path: csv_bytes,
        }
        manager, emitted = _make_manager(store, files)
        manager.register_canonical_digests([digest])

        captured = manager.final_harvest()

        assert len(captured) == 1
        assert captured[0].kind == ArtifactKind.DATASET
        assert captured[0].sha256 == digest
        assert b"".join(store.open_bytes("job-1", captured[0].artifact_id)) == csv_bytes
        assert emitted[0]["data"]["sha256"] == digest
        assert manager.last_harvest_rejections() == ()
        assert manager.has_published_canonical_dataset() is True
        assert manager.published_canonical_digests() == frozenset({digest})

    def test_accepted_canonical_path_cannot_later_downgrade_to_legacy_capture(self, tmp_path: Any) -> None:
        store = SqlArtifactStore(f"sqlite:///{tmp_path}/jobs.db")
        csv_path = f"{_ARTIFACT_DIR}/revenue.csv"
        manifest_path = f"{_ARTIFACT_DIR}/manifest.json"
        original_bytes = b"quarter,revenue\nQ1,22.6\n"
        digest = sha256(original_bytes).hexdigest()
        files = {
            manifest_path: _manifest_bytes(csv_path, "dataset", digest),
            csv_path: original_bytes,
        }
        manager, _ = _make_manager(store, files)
        manager.register_canonical_digests([digest])

        assert [artifact.filename for artifact in manager.harvest_after_execute()] == ["revenue.csv"]
        files[csv_path] = b"quarter,revenue\nQ1,999\n"
        files[manifest_path] = _manifest_bytes(csv_path, "table")

        assert manager.harvest_after_execute() == []
        assert manager.last_harvest_rejections() == (("revenue.csv", "canonical_dataset_digest_missing"),)
        [stored] = store.list("job-1")
        assert b"".join(store.open_bytes("job-1", stored.artifact_id)) == original_bytes

    def test_legacy_duplicate_before_canonical_entry_cannot_bypass_digest_validation(self, tmp_path: Any) -> None:
        store = SqlArtifactStore(f"sqlite:///{tmp_path}/jobs.db")
        csv_path = f"{_ARTIFACT_DIR}/revenue.csv"
        manifest_path = f"{_ARTIFACT_DIR}/manifest.json"
        manifest = json.dumps(
            {
                "version": 1,
                "artifacts": [
                    {"path": csv_path, "kind": "table"},
                    {"path": csv_path, "kind": "dataset", "expected_sha256": "a" * 64},
                ],
            }
        ).encode("utf-8")
        manager, _ = _make_manager(
            store,
            {manifest_path: manifest, csv_path: b"quarter,revenue\nQ1,22.6\n"},
        )

        assert manager.final_harvest() == []
        assert manager.last_harvest_rejections() == (("revenue.csv", "canonical_dataset_digest_unregistered"),)
        assert store.list("job-1") == []

    def test_conflicting_canonical_declarations_for_one_path_fail_closed(self, tmp_path: Any) -> None:
        store = SqlArtifactStore(f"sqlite:///{tmp_path}/jobs.db")
        csv_path = f"{_ARTIFACT_DIR}/revenue.csv"
        manifest_path = f"{_ARTIFACT_DIR}/manifest.json"
        csv_bytes = b"quarter,revenue\nQ1,22.6\n"
        digest = sha256(csv_bytes).hexdigest()
        other_digest = "a" * 64
        manifest = json.dumps(
            {
                "version": 1,
                "artifacts": [
                    {"path": csv_path, "kind": "dataset", "expected_sha256": digest},
                    {"path": csv_path, "kind": "dataset", "expected_sha256": other_digest},
                ],
            }
        ).encode("utf-8")
        manager, _ = _make_manager(store, {manifest_path: manifest, csv_path: csv_bytes})
        manager.register_canonical_digests([digest, other_digest])

        assert manager.final_harvest() == []
        assert manager.last_harvest_rejections() == (("revenue.csv", "canonical_dataset_digest_mismatch"),)
        assert store.list("job-1") == []

    def test_final_scan_does_not_reject_a_checkpointed_canonical_dataset(self, tmp_path: Any) -> None:
        store = SqlArtifactStore(f"sqlite:///{tmp_path}/jobs.db")
        csv_path = f"{_ARTIFACT_DIR}/revenue.csv"
        csv_bytes = b"quarter,revenue\nQ1,22.6\n"
        digest = sha256(csv_bytes).hexdigest()
        files = {
            f"{_ARTIFACT_DIR}/manifest.json": _manifest_bytes(csv_path, "dataset", digest),
            csv_path: csv_bytes,
        }
        manager, emitted = _make_manager(store, files)
        manager.register_canonical_digests([digest])

        assert [artifact.filename for artifact in manager.harvest_after_execute()] == ["revenue.csv"]
        assert manager.final_harvest() == []

        assert manager.last_harvest_rejections() == ()
        assert [event for event in emitted if event["type"] == "artifact.warning"] == []

    def test_canonical_path_with_download_failure_cannot_downgrade_to_legacy_capture(self, tmp_path: Any) -> None:
        store = SqlArtifactStore(f"sqlite:///{tmp_path}/jobs.db")
        csv_path = f"{_ARTIFACT_DIR}/revenue.csv"
        manifest_path = f"{_ARTIFACT_DIR}/manifest.json"
        csv_bytes = b"quarter,revenue\nQ1,22.6\n"
        digest = sha256(csv_bytes).hexdigest()
        files = {manifest_path: _manifest_bytes(csv_path, "dataset", digest)}
        manager, _ = _make_manager(store, files)
        manager.register_canonical_digests([digest])

        assert manager.harvest_after_execute() == []
        files[csv_path] = csv_bytes
        files[manifest_path] = _manifest_bytes(csv_path, "table")

        assert manager.harvest_after_execute() == []
        assert manager.last_harvest_rejections() == (("revenue.csv", "canonical_dataset_digest_missing"),)
        assert store.list("job-1") == []

    @pytest.mark.parametrize("failure", ["exception", "empty", "unreadable"])
    def test_unreadable_manifest_fails_closed_without_directory_scan(self, tmp_path: Any, failure: str) -> None:
        store = SqlArtifactStore(f"sqlite:///{tmp_path}/jobs.db")
        csv_path = f"{_ARTIFACT_DIR}/legacy.csv"
        manifest_path = f"{_ARTIFACT_DIR}/manifest.json"
        manager, _ = _make_manager(store, {csv_path: b"label,value\nA,1\n"})

        def fail_manifest_download(paths: list[str]) -> list[Any]:
            assert paths == [manifest_path]
            if failure == "exception":
                raise RuntimeError("transport unavailable")
            if failure == "empty":
                return []
            return [SimpleNamespace(path=manifest_path, content=None, error="permission denied")]

        original_download = manager.backend.download_files

        def download(paths: list[str]) -> list[Any]:
            if paths != [manifest_path]:
                return original_download(paths)
            return fail_manifest_download(paths)

        manager.backend.download_files = download  # type: ignore[method-assign]

        assert manager.final_harvest() == []
        assert manager.last_harvest_rejections() == (("manifest.json", "artifact_manifest_invalid"),)
        assert store.list("job-1") == []

    def test_expected_digest_on_non_csv_does_not_mark_canonical_dataset_published(self, tmp_path: Any) -> None:
        store = SqlArtifactStore(f"sqlite:///{tmp_path}/jobs.db")
        json_path = f"{_ARTIFACT_DIR}/data.json"
        data = b'{"quarter":"Q1","revenue":22.6}'
        digest = sha256(data).hexdigest()
        files = {
            f"{_ARTIFACT_DIR}/manifest.json": _manifest_bytes(json_path, "dataset", digest),
            json_path: data,
        }
        manager, _ = _make_manager(store, files)
        manager.register_canonical_digests([digest])

        assert manager.harvest_after_execute() == []
        assert manager.last_harvest_rejections() == (("data.json", "canonical_dataset_digest_missing"),)
        assert manager.published_canonical_digests() == frozenset()
        assert manager.has_published_canonical_dataset() is False
        assert store.list("job-1") == []

    def test_invalid_present_manifest_fails_closed_without_scan_downgrade(self, tmp_path: Any) -> None:
        store = SqlArtifactStore(f"sqlite:///{tmp_path}/jobs.db")
        csv_path = f"{_ARTIFACT_DIR}/revenue.csv"
        manifest_path = f"{_ARTIFACT_DIR}/manifest.json"
        manifest = json.dumps(
            {
                "version": 1,
                "artifacts": [
                    {"path": csv_path, "kind": "dataset", "expected_sha256": "a" * 64},
                    {"path": f"{_ARTIFACT_DIR}/bad.png", "kind": "not-a-kind"},
                ],
            }
        ).encode("utf-8")
        manager, _ = _make_manager(
            store,
            {
                manifest_path: manifest,
                csv_path: b"quarter,revenue\nQ1,22.6\n",
            },
        )

        assert manager.harvest_after_execute() == []
        assert manager.last_harvest_rejections() == (("manifest.json", "artifact_manifest_invalid"),)
        manager.backend.files[manifest_path] = json.dumps({"version": 1, "artifacts": []}).encode("utf-8")
        assert manager.harvest_after_execute() == []
        assert manager.final_harvest() == []
        assert store.list("job-1") == []
        assert manager.has_published_canonical_dataset() is False

    def test_invalid_manifest_recovers_only_through_a_valid_explicit_entry(self, tmp_path: Any) -> None:
        store = SqlArtifactStore(f"sqlite:///{tmp_path}/jobs.db")
        csv_path = f"{_ARTIFACT_DIR}/revenue.csv"
        manifest_path = f"{_ARTIFACT_DIR}/manifest.json"
        csv_bytes = b"quarter,revenue\nQ1,22.6\n"
        digest = sha256(csv_bytes).hexdigest()
        files = {
            manifest_path: b"not-json",
            csv_path: csv_bytes,
        }
        manager, _ = _make_manager(store, files)

        assert manager.harvest_after_execute() == []
        manager.register_canonical_digests([digest])
        files[manifest_path] = _manifest_bytes(csv_path, "dataset", digest)

        captured = manager.harvest_after_execute()

        assert [artifact.filename for artifact in captured] == ["revenue.csv"]
        assert manager.has_published_canonical_dataset() is True

    @pytest.mark.parametrize(
        ("manifest_digest", "registered_digest", "reason"),
        [
            (None, None, "canonical_dataset_digest_missing"),
            ("a" * 64, None, "canonical_dataset_digest_unregistered"),
            ("a" * 64, "a" * 64, "canonical_dataset_digest_mismatch"),
        ],
    )
    def test_rejects_untrusted_canonical_dataset_with_stable_reason(
        self,
        tmp_path: Any,
        manifest_digest: str | None,
        registered_digest: str | None,
        reason: str,
    ) -> None:
        store = SqlArtifactStore(f"sqlite:///{tmp_path}/jobs.db")
        csv_path = f"{_ARTIFACT_DIR}/revenue.csv"
        files = {
            f"{_ARTIFACT_DIR}/manifest.json": _manifest_bytes(csv_path, "dataset", manifest_digest),
            csv_path: b"quarter,revenue\nQ1,22.6\n",
        }
        manager, emitted = _make_manager(store, files)
        if registered_digest is not None:
            manager.register_canonical_digests([registered_digest])

        assert manager.harvest_after_execute() == []

        warnings = [event for event in emitted if event["type"] == "artifact.warning"]
        assert warnings[-1]["data"] == {"path": csv_path, "reason": reason}
        assert store.list("job-1") == []

    def test_rejected_canonical_path_cannot_be_recaptured_by_final_scan(self, tmp_path: Any) -> None:
        store = SqlArtifactStore(f"sqlite:///{tmp_path}/jobs.db")
        csv_path = f"{_ARTIFACT_DIR}/revenue.csv"
        manifest_path = f"{_ARTIFACT_DIR}/manifest.json"
        files = {
            manifest_path: _manifest_bytes(csv_path, "dataset"),
            csv_path: b"quarter,revenue\nQ1,22.6\n",
        }
        manager, _ = _make_manager(store, files)

        assert manager.harvest_after_execute() == []
        del manager.backend.files[manifest_path]
        assert manager.final_harvest() == []
        assert store.list("job-1") == []

    def test_rejected_canonical_path_cannot_downgrade_and_can_recover_with_valid_digest(self, tmp_path: Any) -> None:
        store = SqlArtifactStore(f"sqlite:///{tmp_path}/jobs.db")
        csv_path = f"{_ARTIFACT_DIR}/revenue.csv"
        manifest_path = f"{_ARTIFACT_DIR}/manifest.json"
        csv_bytes = b"quarter,revenue\nQ1,22.6\n"
        digest = sha256(csv_bytes).hexdigest()
        files = {
            manifest_path: _manifest_bytes(csv_path, "dataset"),
            csv_path: csv_bytes,
        }
        manager, _ = _make_manager(store, files)

        assert manager.harvest_after_execute() == []
        assert manager.last_harvest_rejections() == (
            (csv_path.rsplit("/", maxsplit=1)[-1], "canonical_dataset_digest_missing"),
        )

        files[manifest_path] = _manifest_bytes(csv_path, "table")
        assert manager.harvest_after_execute() == []
        assert manager.last_harvest_rejections() == (("revenue.csv", "canonical_dataset_digest_missing"),)
        assert store.list("job-1") == []

        manager.register_canonical_digests([digest])
        files[manifest_path] = _manifest_bytes(csv_path, "dataset", digest)
        captured = manager.harvest_after_execute()

        assert [artifact.filename for artifact in captured] == ["revenue.csv"]
        assert manager.last_harvest_rejections() == ()
        assert manager.has_published_canonical_dataset() is True

    def test_harvest_rejection_diagnostics_are_bounded(self, tmp_path: Any) -> None:
        store = SqlArtifactStore(f"sqlite:///{tmp_path}/jobs.db")
        manifest_path = f"{_ARTIFACT_DIR}/manifest.json"
        entries = [{"path": f"{_ARTIFACT_DIR}/dataset_{index}.csv", "kind": "dataset"} for index in range(40)]
        manager, _ = _make_manager(
            store,
            {manifest_path: json.dumps({"version": 1, "artifacts": entries}).encode("utf-8")},
        )

        assert manager.harvest_after_execute() == []
        diagnostics = manager.last_harvest_rejections()
        assert len(diagnostics) == 32
        assert all(reason == "canonical_dataset_digest_missing" for _path, reason in diagnostics)

    def test_noncanonical_manifest_csv_remains_backward_compatible(self, tmp_path: Any) -> None:
        store = SqlArtifactStore(f"sqlite:///{tmp_path}/jobs.db")
        csv_path = f"{_ARTIFACT_DIR}/legacy.csv"
        csv_bytes = b"label,value\nA,1\n"
        files = {
            f"{_ARTIFACT_DIR}/manifest.json": _manifest_bytes(csv_path, "table"),
            csv_path: csv_bytes,
        }
        manager, _ = _make_manager(store, files)

        captured = manager.final_harvest()

        assert [artifact.filename for artifact in captured] == ["legacy.csv"]
        assert captured[0].sha256 == sha256(csv_bytes).hexdigest()

    def test_csv_dataset_manifest_kind_is_reserved_for_canonical_publication(self, tmp_path: Any) -> None:
        store = SqlArtifactStore(f"sqlite:///{tmp_path}/jobs.db")
        csv_path = f"{_ARTIFACT_DIR}/dataset.csv"
        files = {
            f"{_ARTIFACT_DIR}/manifest.json": _manifest_bytes(csv_path, "dataset"),
            csv_path: b"label,value\nA,1\n",
        }
        manager, _ = _make_manager(store, files)

        assert manager.final_harvest() == []
        assert manager.last_harvest_rejections() == (("dataset.csv", "canonical_dataset_digest_missing"),)

    def test_canonical_digest_registration_is_strict_and_idempotent(self, tmp_path: Any) -> None:
        store = SqlArtifactStore(f"sqlite:///{tmp_path}/jobs.db")
        manager, _ = _make_manager(store, {})
        digest = "a" * 64

        manager.register_canonical_digests([digest, digest])
        assert manager._trusted_canonical_digests == {digest}
        with pytest.raises(ValueError, match="lowercase SHA-256"):
            manager.register_canonical_digests(["A" * 64])

    def test_canonical_path_registration_is_strict_and_rejects_conflicts(self, tmp_path: Any) -> None:
        store = SqlArtifactStore(f"sqlite:///{tmp_path}/jobs.db")
        manager, _ = _make_manager(store, {})
        first_digest = "a" * 64

        manager.register_canonical_datasets([("quarterly_revenue", first_digest)])
        manager.register_canonical_datasets([("quarterly_revenue", first_digest)])

        expected_path = f"{_ARTIFACT_DIR}/quarterly_revenue.csv"
        assert manager._trusted_canonical_paths == {expected_path: first_digest}
        with pytest.raises(ValueError, match="conflicting digests"):
            manager.register_canonical_datasets([("quarterly_revenue", "b" * 64)])
        with pytest.raises(ValueError, match="path-safe"):
            manager.register_canonical_datasets([("../quarterly_revenue", first_digest)])

    def test_rejects_mime_spoof(self, tmp_path: Any) -> None:
        # A file claiming .png but with non-image content must be rejected.
        store = SqlArtifactStore(f"sqlite:///{tmp_path}/jobs.db")
        png_path = f"{_ARTIFACT_DIR}/chart.png"
        files = {f"{_ARTIFACT_DIR}/manifest.json": _manifest_bytes(png_path), png_path: b"#!/bin/sh\nrm -rf /\n"}
        manager, _ = _make_manager(store, files)
        assert manager.final_harvest() == []

    def test_rejects_svg_fail_closed(self, tmp_path: Any) -> None:
        # SVG cannot be reliably sanitized (javascript: URIs, <foreignObject>, CSS payloads),
        # and the content endpoint serves bytes as the stored MIME, so SVG is rejected at
        # harvest rather than partially cleaned and stored (stored-XSS prevention).
        store = SqlArtifactStore(f"sqlite:///{tmp_path}/jobs.db")
        svg_path = f"{_ARTIFACT_DIR}/diagram.svg"
        svg = b'<svg onload="steal()"><script>alert(1)</script><rect/></svg>'
        files = {f"{_ARTIFACT_DIR}/manifest.json": _manifest_bytes(svg_path), svg_path: svg}
        manager, _ = _make_manager(store, files)

        assert manager.final_harvest() == []


class TestStore:
    def _artifact(self) -> Artifact:
        return Artifact(
            artifact_id="art_" + "a" * 32,
            job_id="job-1",
            kind=ArtifactKind.IMAGE,
            mime_type="image/png",
            filename="chart.png",
            sandbox_path=f"{_ARTIFACT_DIR}/chart.png",
            storage_uri="",
            sha256=_PNG_SHA256,
            size_bytes=len(_PNG),
        )

    def test_put_get_list_open(self, tmp_path: Any) -> None:
        store = SqlArtifactStore(f"sqlite:///{tmp_path}/jobs.db")
        stored = store.put(self._artifact(), _PNG)
        assert stored.status.value == "available"
        assert store.get("job-1", stored.artifact_id) is not None
        assert len(store.list("job-1")) == 1
        assert b"".join(store.open_bytes("job-1", stored.artifact_id)) == _PNG

    def test_dedup_by_digest(self, tmp_path: Any) -> None:
        store = SqlArtifactStore(f"sqlite:///{tmp_path}/jobs.db")
        first = store.put(self._artifact(), _PNG)
        # Same content/digest but a DIFFERENT artifact_id: dedup must key on (job_id, sha256),
        # not on artifact_id, so the second put returns the existing row instead of inserting
        # a duplicate. Reusing the same id here would let an id-based regression pass silently.
        duplicate = self._artifact().model_copy(update={"artifact_id": "art_" + "b" * 32})
        again = store.put(duplicate, _PNG)
        assert again.artifact_id == first.artifact_id
        assert len(store.list("job-1")) == 1

    def test_cleanup_removes_old_artifacts(self, tmp_path: Any) -> None:
        from sqlalchemy import text

        store = SqlArtifactStore(f"sqlite:///{tmp_path}/jobs.db")
        stored = store.put(self._artifact(), _PNG)
        with store._engine.connect() as conn:
            conn.execute(
                text("UPDATE artifacts SET created_at = datetime('now', '-2 seconds') WHERE artifact_id = :id"),
                {"id": stored.artifact_id},
            )
            conn.commit()

        assert store.cleanup_old_artifacts(1) == 1
        assert store.get(stored.job_id, stored.artifact_id) is None


class _FakeStreamingBody:
    def __init__(self, data: bytes) -> None:
        self._stream = BytesIO(data)
        self.closed = False

    def iter_chunks(self, chunk_size: int) -> Any:
        while chunk := self._stream.read(chunk_size):
            yield chunk

    def close(self) -> None:
        self.closed = True
        self._stream.close()


class _FakeS3Client:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.fail_put = False
        self.fail_delete = False
        self.head_bucket_calls: list[str] = []

    def put_object(self, **kwargs: Any) -> None:
        if self.fail_put:
            raise RuntimeError("upload failed")
        location = (kwargs["Bucket"], kwargs["Key"])
        self.objects[location] = kwargs["Body"]

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        return {"Body": _FakeStreamingBody(self.objects[(kwargs["Bucket"], kwargs["Key"])])}

    def delete_object(self, **kwargs: Any) -> None:
        if self.fail_delete:
            raise RuntimeError("delete failed")
        location = (kwargs["Bucket"], kwargs["Key"])
        self.objects.pop(location, None)

    def head_bucket(self, **kwargs: Any) -> None:
        self.head_bucket_calls.append(kwargs["Bucket"])


class TestS3Store:
    def _artifact(self) -> Artifact:
        return Artifact(
            artifact_id="art_" + "a" * 32,
            job_id="job-1",
            kind=ArtifactKind.IMAGE,
            mime_type="image/png",
            filename="chart.png",
            sandbox_path=f"{_ARTIFACT_DIR}/chart.png",
            storage_uri="",
            sha256=_PNG_SHA256,
            size_bytes=len(_PNG),
        )

    def _store(
        self,
        tmp_path: Any,
        client: _FakeS3Client,
    ) -> SqlArtifactStore:
        blob_store = S3ArtifactBlobStore(
            bucket="aiq-artifacts",
            prefix="artifacts/v1",
            endpoint_url="http://minio:9000",
            client=client,
        )
        return SqlArtifactStore(f"sqlite:///{tmp_path}/jobs.db", blob_store=blob_store)

    def test_put_open_and_delete(self, tmp_path: Any) -> None:
        client = _FakeS3Client()
        store = self._store(tmp_path, client)

        stored = store.put(self._artifact(), _PNG)

        assert stored.storage_uri == f"s3://aiq-artifacts/artifacts/v1/job-1/{stored.artifact_id}"
        assert stored.status == ArtifactStatus.AVAILABLE
        assert b"".join(store.open_bytes("job-1", stored.artifact_id)) == _PNG
        assert store.delete_job("job-1") == 1
        assert client.objects == {}
        assert store.get("job-1", stored.artifact_id) is None

    def test_validate_checks_configured_bucket(self, tmp_path: Any) -> None:
        client = _FakeS3Client()
        store = self._store(tmp_path, client)

        store.validate()

        assert client.head_bucket_calls == ["aiq-artifacts"]

    def test_s3_bytes_leave_sql_content_null(self, tmp_path: Any) -> None:
        from sqlalchemy import text

        store = self._store(tmp_path, _FakeS3Client())
        stored = store.put(self._artifact(), _PNG)
        with store._engine.connect() as conn:
            content = conn.execute(
                text("SELECT content FROM artifacts WHERE artifact_id = :artifact_id"),
                {"artifact_id": stored.artifact_id},
            ).scalar_one()
        assert content is None

    def test_failed_upload_removes_metadata(self, tmp_path: Any, caplog: pytest.LogCaptureFixture) -> None:
        client = _FakeS3Client()
        client.fail_put = True
        store = self._store(tmp_path, client)
        artifact = self._artifact()

        with caplog.at_level(logging.WARNING):
            with pytest.raises(RuntimeError, match="upload failed"):
                store.put(artifact, _PNG)

        assert store.get(artifact.job_id, artifact.artifact_id) is None
        assert b"".join(store.open_bytes(artifact.job_id, artifact.artifact_id)) == b""
        assert "RuntimeError" in caplog.text
        assert "upload failed" not in caplog.text

    def test_failed_delete_retains_metadata_for_retry(self, tmp_path: Any) -> None:
        client = _FakeS3Client()
        store = self._store(tmp_path, client)
        stored = store.put(self._artifact(), _PNG)
        client.fail_delete = True

        assert store.delete_job(stored.job_id) == 0
        assert store.get(stored.job_id, stored.artifact_id) is not None

    def test_failed_metadata_delete_is_retried(self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _FakeS3Client()
        store = self._store(tmp_path, client)
        stored = store.put(self._artifact(), _PNG)

        def fail_connect() -> None:
            raise RuntimeError("database unavailable")

        with monkeypatch.context() as patch:
            patch.setattr(store._engine, "connect", fail_connect)
            assert store._delete_artifact(stored) == 0

        assert client.objects == {}
        assert store.get(stored.job_id, stored.artifact_id) is not None
        assert store._delete_artifact(stored) == 1
        assert store.get(stored.job_id, stored.artifact_id) is None


class TestArtifactStoreFactory:
    def test_sql_is_the_default_provider(self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AIQ_ARTIFACT_BLOB_PROVIDER", raising=False)

        store = build_artifact_store(f"sqlite:///{tmp_path}/jobs.db")

        assert isinstance(store, SqlArtifactStore)
        assert store._blob_store.scheme == "db"

    def test_s3_provider_uses_configured_blob_store(
        self,
        tmp_path: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from aiq_agent.agents.deep_researcher.sandbox.artifacts import factory

        fake_blob_store = S3ArtifactBlobStore(bucket="aiq-artifacts", client=_FakeS3Client())
        monkeypatch.setenv("AIQ_ARTIFACT_BLOB_PROVIDER", "s3")
        monkeypatch.setenv("AIQ_ARTIFACT_S3_BUCKET", "aiq-artifacts")
        monkeypatch.setattr(factory, "S3ArtifactBlobStore", lambda **_kwargs: fake_blob_store)
        factory._build_artifact_store.cache_clear()

        store = build_artifact_store(f"sqlite:///{tmp_path}/jobs.db")

        assert isinstance(store, SqlArtifactStore)
        assert store._blob_store is fake_blob_store
        factory._build_artifact_store.cache_clear()

    def test_rejects_unknown_provider(self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        from aiq_agent.agents.deep_researcher.sandbox.artifacts import factory

        monkeypatch.setenv("AIQ_ARTIFACT_BLOB_PROVIDER", "unknown")
        factory._build_artifact_store.cache_clear()

        with pytest.raises(ValueError, match="Unsupported AIQ_ARTIFACT_BLOB_PROVIDER"):
            build_artifact_store(f"sqlite:///{tmp_path}/jobs.db")

        factory._build_artifact_store.cache_clear()


class TestEnsureInlineArtifactsEmbedded:
    """The safety net that surfaces produced inline figures the report forgot to embed."""

    def _store_with(self, tmp_path: Any, *artifacts: Artifact) -> SqlArtifactStore:
        store = SqlArtifactStore(f"sqlite:///{tmp_path}/jobs.db")
        for index, artifact in enumerate(artifacts):
            store.put(artifact, _PNG + bytes([index]))
        return store

    def _image(self, artifact_id: str, *, inline: bool = True) -> Artifact:
        return Artifact(
            artifact_id=artifact_id,
            job_id="job-1",
            kind=ArtifactKind.IMAGE,
            mime_type="image/png",
            filename=f"{artifact_id}.png",
            sandbox_path=f"{_ARTIFACT_DIR}/{artifact_id}.png",
            storage_uri="",
            sha256=artifact_id.ljust(64, "0"),
            size_bytes=len(_PNG),
            title="Chart Title",
            caption="A caption",
            inline=inline,
        )

    def test_appends_orphan_inline_image(self, tmp_path: Any) -> None:
        artifact_id = "art_" + "a" * 32
        store = self._store_with(tmp_path, self._image(artifact_id))
        manager, _ = _make_manager(store, {})

        result = manager.ensure_inline_artifacts_embedded("# Report\n\nNo figures here.\n")

        assert "## Figures" in result
        assert f"![A caption](artifact://{artifact_id})" in result

    def test_does_not_duplicate_referenced_image(self, tmp_path: Any) -> None:
        artifact_id = "art_" + "b" * 32
        store = self._store_with(tmp_path, self._image(artifact_id))
        manager, _ = _make_manager(store, {})
        markdown = f"# Report\n\n![Inline](artifact://{artifact_id})\n"

        result = manager.ensure_inline_artifacts_embedded(markdown)

        assert result == markdown
        assert "## Figures" not in result

    def test_append_artifact_index_lists_all_artifacts(self, tmp_path: Any) -> None:
        png = self._image("art_" + "e" * 32)
        csv = Artifact(
            artifact_id="art_" + "f" * 32,
            job_id="job-1",
            kind=ArtifactKind.TABLE,
            mime_type="text/csv",
            filename="chart.csv",
            sandbox_path=f"{_ARTIFACT_DIR}/chart.csv",
            storage_uri="",
            sha256="f" * 64,
            size_bytes=8,
            caption="Plotted values",
        )
        store = self._store_with(tmp_path, png, csv)
        manager, _ = _make_manager(store, {})

        result = manager.append_artifact_index("# Report\n\nBody.\n")

        assert "## Generated Artifacts" in result
        assert "`art_eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee.png`" in result
        assert "`chart.csv` - Plotted values (generated in the analysis sandbox)" in result

    def test_append_artifact_index_noop_without_artifacts(self, tmp_path: Any) -> None:
        store = SqlArtifactStore(f"sqlite:///{tmp_path}/jobs.db")
        manager, _ = _make_manager(store, {})
        assert manager.append_artifact_index("# Report\n") == "# Report\n"

    def test_skips_non_inline_and_non_image(self, tmp_path: Any) -> None:
        non_inline = self._image("art_" + "c" * 32, inline=False)
        csv = Artifact(
            artifact_id="art_" + "d" * 32,
            job_id="job-1",
            kind=ArtifactKind.TABLE,
            mime_type="text/csv",
            filename="data.csv",
            sandbox_path=f"{_ARTIFACT_DIR}/data.csv",
            storage_uri="",
            sha256="d" * 64,
            size_bytes=8,
            inline=True,
        )
        store = self._store_with(tmp_path, non_inline, csv)
        manager, _ = _make_manager(store, {})

        result = manager.ensure_inline_artifacts_embedded("# Report\n")

        assert result == "# Report\n"
