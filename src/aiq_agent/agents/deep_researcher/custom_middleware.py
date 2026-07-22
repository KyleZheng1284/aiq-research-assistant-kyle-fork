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

"""Custom middleware for the deep research agent."""

import asyncio
import hashlib
import json
import logging
import posixpath
import re
import shlex
import threading
from pathlib import Path
from pathlib import PurePosixPath

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware import hook_config
from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.messages import ToolMessage

from aiq_agent.common import get_source_id_for_tool
from aiq_agent.common import load_prompt
from aiq_agent.common import render_prompt_template
from aiq_agent.common.citation_verification import SourceEntry
from aiq_agent.common.citation_verification import SourceRegistry
from aiq_agent.common.citation_verification import extract_sources_from_tool_result

from .models import ResearchNotes
from .sandbox.artifacts.manifest import parse_manifest

logger = logging.getLogger(__name__)

# Path to this agent's prompts directory
_PROMPTS_DIR = Path(__file__).parent / "prompts"
_SOURCE_ROUTING_PATH = "/shared/source_routing.json"
# When a sandbox provider is configured, CompositeBackend strips the /shared/ route
# before delegating to StateBackend, so the router's file is stored under the
# route-local key. The guard reads raw state, so it must accept both forms or it
# blocks the orchestrator forever on sandboxed runs.
_SOURCE_ROUTING_STATE_KEYS = (_SOURCE_ROUTING_PATH, "/source_routing.json")
FINAL_REPORT_PATH = "/shared/output.md"
FINAL_REPORT_STATE_PATHS = (FINAL_REPORT_PATH, "/output.md")
_UNRESOLVED_SANDBOX_PATH_PATTERN = re.compile(
    r"<\s*sandbox_(?:artifact_dir|workdir)\s*>|\{\{\s*sandbox_(?:artifact_dir|workdir)\s*\}\}"
)
_EXECUTE_RESULT_RE = re.compile(
    r"\n\[Command (?P<outcome>succeeded|failed) with exit code (?P<exit_code>-?\d+)\]"
    r"(?:\n\[Output was truncated due to size limits\])?\Z"
)
_MARKDOWN_TABLE_SEPARATOR_CELL_RE = re.compile(r":?-{3,}:?")


def _tool_result_failed(result: object) -> bool:
    """Return whether a tool result represents a failed operation.

    DeepAgents deliberately returns sandbox command results as successful
    ``ToolMessage`` objects even when the command's process exit code is non-zero.
    Parse its terminal status marker in addition to the ordinary message status so
    failed chart commands cannot be checkpointed or mistaken for a successful
    writer execution attempt.
    """
    status = result.get("status") if isinstance(result, dict) else getattr(result, "status", None)
    if status == "error":
        return True

    exit_code = result.get("exit_code") if isinstance(result, dict) else getattr(result, "exit_code", None)
    if isinstance(exit_code, int):
        return exit_code != 0

    content = result.get("content") if isinstance(result, dict) else getattr(result, "content", None)
    if not isinstance(content, str):
        return False
    match = _EXECUTE_RESULT_RE.search(content)
    return bool(match and int(match.group("exit_code")) != 0)


def _entry_text(entry: object) -> str | None:
    """Read text from a DeepAgents state-file entry without changing it."""
    if isinstance(entry, dict):
        entry = entry.get("content")
    if isinstance(entry, bytes):
        try:
            return entry.decode("utf-8")
        except UnicodeDecodeError:
            return None
    if isinstance(entry, str):
        return entry
    if isinstance(entry, list) and all(isinstance(line, str) for line in entry):
        return "\n".join(entry)
    return None


def _normalize_guard_path(path: str) -> str:
    """Normalize a POSIX path for artifact-directory ownership checks."""
    normalized = posixpath.normpath(path)
    if normalized.startswith("//"):
        normalized = "/" + normalized.lstrip("/")
    return normalized


def _path_targets_directory(path: object, directory: str) -> bool:
    """Return whether a path resolves to or below a configured directory."""
    if not isinstance(path, str) or not path or not directory:
        return False
    normalized_path = _normalize_guard_path(path)
    normalized_dir = _normalize_guard_path(directory)
    if normalized_path == normalized_dir or normalized_path.startswith(normalized_dir.rstrip("/") + "/"):
        return True
    if not posixpath.isabs(normalized_path):
        # Sandbox execute starts in the per-job workdir, whose direct artifact
        # child is conventionally addressed as ``aiq-artifacts/...``. Treat any
        # relative occurrence of that reserved directory name conservatively as
        # an artifact target, including ``./`` and parent-relative spellings.
        artifact_basename = PurePosixPath(normalized_dir).name
        return artifact_basename in PurePosixPath(normalized_path).parts
    return False


def _command_targets_directory(command: object, directory: str) -> bool:
    """Return whether a shell command directly names the configured directory."""
    if not isinstance(command, str) or not command.strip() or not directory:
        return False

    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        tokens = command.split()
    for token in tokens:
        candidates = {token, token.strip("<>|&;()")}
        if "=" in token:
            candidates.add(token.split("=", maxsplit=1)[1].strip("<>|&;()"))
        if any(_path_targets_directory(candidate, directory) for candidate in candidates):
            return True

    # Inline scripts can directly open a literal artifact path inside one shell
    # token (for example ``python -c 'open("/path/file", "w")'``). Match the
    # configured absolute directory only at path boundaries to avoid siblings.
    normalized_dir = _normalize_guard_path(directory).rstrip("/")
    boundary_pattern = rf"(?<![\w.-]){re.escape(normalized_dir)}(?=$|[/\s'\";|&()<>=,])"
    return re.search(boundary_pattern, command) is not None


def validated_research_notes_from_state(state: object) -> tuple[ResearchNotes, ...]:
    """Load only schema-valid persisted research notes from DeepAgents state.

    The state-file channel is the durable handoff between researcher and writer.
    Revalidating here recomputes every canonical CSV digest from the exact
    persisted ``csv_text`` and prevents stale or model-supplied digest values
    from opening the writer execution gate.
    """
    files = state.get("files", {}) if isinstance(state, dict) else getattr(state, "files", {})
    if not isinstance(files, dict):
        return ()

    notes: list[ResearchNotes] = []
    for path, entry in sorted(files.items(), key=lambda item: str(item[0])):
        filename = PurePosixPath(str(path)).name
        if not filename.startswith("research_note_") or not filename.endswith(".json"):
            continue
        text = _entry_text(entry)
        if text is None:
            logger.warning("Ignoring unreadable persisted research note %s", filename)
            continue
        try:
            payload = json.loads(text)
            notes.append(ResearchNotes.model_validate(payload))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("Ignoring invalid persisted research note %s (%s)", filename, type(exc).__name__)
    return tuple(notes)


def _contains_markdown_table(text: str) -> bool:
    """Return whether text contains a Markdown table with at least one data row."""

    def cells(line: str) -> list[str]:
        stripped = line.strip()
        if stripped.startswith("|"):
            stripped = stripped[1:]
        if stripped.endswith("|"):
            stripped = stripped[:-1]
        return [cell.strip() for cell in stripped.split("|")]

    lines = text.splitlines()
    for index in range(len(lines) - 2):
        if any("|" not in lines[offset] for offset in (index, index + 1, index + 2)):
            continue
        header_cells = cells(lines[index])
        separator_cells = cells(lines[index + 1])
        data_cells = cells(lines[index + 2])
        if (
            len(header_cells) >= 1
            and len(header_cells) == len(separator_cells) == len(data_cells)
            and all(header_cells)
            and all(_MARKDOWN_TABLE_SEPARATOR_CELL_RE.fullmatch(cell) for cell in separator_cells)
        ):
            return True
    return False


def _normalized_virtual_path(path: object) -> str | None:
    """Return a canonical virtual path without weakening backend validation."""
    if not isinstance(path, str) or not path:
        return None
    normalized = posixpath.normpath(path.replace("\\", "/"))
    return normalized if normalized.startswith("/") else f"/{normalized}"


def _tool_file_path(tool_call: object) -> str | None:
    """Read and normalize a filesystem tool's target path."""
    if not isinstance(tool_call, dict):
        return None
    args = tool_call.get("args")
    if not isinstance(args, dict):
        return None
    return _normalized_virtual_path(args.get("file_path", args.get("path")))


class FinalReportCommitTracker:
    """Run-local proof of the writer's most recent successful report mutation."""

    def __init__(self) -> None:
        self._digest: str | None = None
        self._lock = threading.Lock()

    @staticmethod
    def _digest_text(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def record(self, content: str) -> str:
        """Record the exact UTF-8 digest after a successful writer mutation."""
        digest = self._digest_text(content)
        with self._lock:
            self._digest = digest
        return digest

    @property
    def digest(self) -> str | None:
        """Return the most recently committed digest for this run."""
        with self._lock:
            return self._digest

    def committed_text(
        self,
        files: object,
        *,
        paths: tuple[str, ...] = FINAL_REPORT_STATE_PATHS,
    ) -> str | None:
        """Return the non-empty state file matching the writer's exact digest."""
        digest = self.digest
        if digest is None or not isinstance(files, dict):
            return None
        for path in paths:
            content = _entry_text(files.get(path))
            if content is not None and content.strip() and self._digest_text(content) == digest:
                return content
        return None


class SourceRoutingGuardMiddleware(AgentMiddleware):
    """Require the source-router handoff before other orchestrator tool calls."""

    def __init__(self, *, enabled: bool, required_subagent: str = "source-router-agent") -> None:
        self.enabled = enabled
        self.required_subagent = required_subagent

    @staticmethod
    def _routing_complete(state: object) -> bool:
        files = state.get("files", {}) if isinstance(state, dict) else getattr(state, "files", {})
        return isinstance(files, dict) and any(key in files for key in _SOURCE_ROUTING_STATE_KEYS)

    async def awrap_tool_call(self, request, handler):
        """Block out-of-order calls until the source router writes its route file."""
        if not self.enabled or self._routing_complete(request.state):
            return await handler(request)

        tool_call = request.tool_call
        args = tool_call.get("args") or {}
        if tool_call.get("name") == "task" and args.get("subagent_type") == self.required_subagent:
            return await handler(request)

        return ToolMessage(
            content=(
                "Source routing is required before any other tool call. "
                f"Call task with subagent_type={self.required_subagent!r}."
            ),
            tool_call_id=tool_call.get("id", "source-routing-guard"),
            name=tool_call.get("name"),
            status="error",
        )


class EmptyContentFixMiddleware(AgentMiddleware):
    """
    Middleware that fixes empty ToolMessage content.

    Some LLM APIs (e.g., NVIDIA, OpenAI) reject messages with empty content.
    This middleware ensures all ToolMessages have non-empty content by
    replacing empty strings with a placeholder.
    """

    def __init__(self, placeholder: str = "empty content received."):
        """
        Initialize the middleware.

        Args:
            placeholder: Text to use when ToolMessage content is empty.
        """
        self.placeholder = placeholder

    async def awrap_model_call(self, request, handler):
        """Fix empty ToolMessage content before sending to the model."""
        fixed_messages = []
        for msg in request.messages:
            if isinstance(msg, ToolMessage) and not msg.content:
                # Create a new ToolMessage with placeholder content
                fixed_messages.append(
                    ToolMessage(
                        content=self.placeholder,
                        tool_call_id=msg.tool_call_id,
                        name=getattr(msg, "name", None),
                        id=msg.id,
                    )
                )
            else:
                fixed_messages.append(msg)

        return await handler(request.override(messages=fixed_messages))


class ExecuteTimeoutClampMiddleware(AgentMiddleware):
    """Clamp the sandbox ``execute`` tool's per-call timeout to a configured ceiling.

    The deepagents ``execute`` tool forwards a model-supplied ``timeout`` straight to the
    sandbox backend, and providers cap it (OpenShell rejects an ``exec`` timeout above the
    gateway maximum). LLMs routinely pass an oversized value -- e.g. milliseconds, where the
    backend expects seconds, or an arbitrarily large round number -- so an unclamped timeout
    makes every ``execute`` fail with a "timeout exceeds maximum" error and no sandbox code
    ever runs. Bound the argument to the configured sandbox lifetime (seconds).

    This guards a different boundary than ``SandboxProvider._clamp_timeout`` in
    ``sandbox/base.py``: that clamp covers AI-Q's own provider-mediated calls (e.g. workspace
    prep), whereas the deepagents ``execute`` tool reaches the backend without passing through
    it, so the untrusted agent argument must be sanitized here at the tool-call boundary.
    """

    def __init__(self, *, max_timeout_seconds: int) -> None:
        """Store the ceiling (in seconds) that a single ``execute`` call may request."""
        self.max_timeout_seconds = max(1, int(max_timeout_seconds))

    async def awrap_tool_call(self, request, handler):
        """Clamp an oversized ``timeout`` argument on ``execute`` tool calls."""
        tool_call = request.tool_call
        if tool_call.get("name") != "execute":
            return await handler(request)
        args = tool_call.get("args")
        if not isinstance(args, dict) or not isinstance(args.get("timeout"), (int, float)):
            return await handler(request)
        requested = int(args["timeout"])
        # A non-positive value means "no timeout" to the backend; leave it alone.
        if requested <= 0 or requested <= self.max_timeout_seconds:
            return await handler(request)
        logger.warning(
            "Clamping execute timeout %ss -> %ss (agent-supplied value exceeds the sandbox ceiling)",
            requested,
            self.max_timeout_seconds,
        )
        modified = {**tool_call, "args": {**args, "timeout": self.max_timeout_seconds}}
        return await handler(request.override(tool_call=modified))


class FilesystemToolCallGuardMiddleware(AgentMiddleware):
    """Normalize safe filesystem aliases and reject unresolved sandbox path templates."""

    async def awrap_tool_call(self, request, handler):
        """Repair ``read_file(path=...)`` and fail before executing placeholder paths."""
        tool_call = request.tool_call
        if not isinstance(tool_call, dict):
            return await handler(request)
        args = tool_call.get("args")
        if not isinstance(args, dict):
            return await handler(request)

        if tool_call.get("name") == "read_file" and isinstance(args.get("path"), str):
            normalized_args = {key: value for key, value in args.items() if key != "path"}
            normalized_args.setdefault("file_path", args["path"])
            modified = {**tool_call, "args": normalized_args}
            return await handler(request.override(tool_call=modified))

        if tool_call.get("name") == "execute" and isinstance(args.get("command"), str):
            command = args["command"]
            unresolved = _UNRESOLVED_SANDBOX_PATH_PATTERN.search(command)
            if unresolved is not None:
                return ToolMessage(
                    content=(
                        f"Command not executed: unresolved sandbox path placeholder {unresolved.group(0)}. "
                        "Use the exact sandbox_workdir or sandbox_artifact_dir path from your instructions."
                    ),
                    tool_call_id=tool_call.get("id", "filesystem-tool-call-guard"),
                    name="execute",
                    status="error",
                )

        return await handler(request)


class FinalReportOwnershipGuardMiddleware(AgentMiddleware):
    """Reserve final-report mutation for the writer role."""

    async def awrap_tool_call(self, request, handler):
        """Reject non-writer mutations of either final-report state path."""
        tool_call = request.tool_call if isinstance(getattr(request, "tool_call", None), dict) else {}
        if tool_call.get("name") not in {"write_file", "edit_file"}:
            return await handler(request)
        if _tool_file_path(tool_call) not in FINAL_REPORT_STATE_PATHS:
            return await handler(request)
        return ToolMessage(
            content=(
                "final_report_writer_only: only writer-agent may write or edit "
                f"{FINAL_REPORT_PATH}; hand off evidence through the normal research workflow."
            ),
            tool_call_id=tool_call.get("id", "final-report-ownership"),
            name=tool_call.get("name"),
            status="error",
        )


class FinalReportCommitMiddleware(AgentMiddleware):
    """Commit writer-owned output with overwrite and exact-digest verification."""

    def __init__(self, *, backend: object, tracker: FinalReportCommitTracker) -> None:
        self.backend = backend
        self.tracker = tracker
        self._mutation_lock = asyncio.Lock()

    @staticmethod
    def _tool_error(tool_call: dict[str, object], reason: str, guidance: str) -> ToolMessage:
        return ToolMessage(
            content=f"{reason}: {guidance}",
            tool_call_id=tool_call.get("id", "final-report-commit"),
            name=tool_call.get("name"),
            status="error",
        )

    @staticmethod
    def _response_error(response: object) -> object:
        return response.get("error") if isinstance(response, dict) else getattr(response, "error", None)

    async def _commit_write(self, tool_call: dict[str, object]) -> ToolMessage:
        args = tool_call.get("args")
        content = args.get("content") if isinstance(args, dict) else None
        if not isinstance(content, str):
            return self._tool_error(tool_call, "writer_output_commit_failed", "report content must be text")
        try:
            responses = await self.backend.aupload_files([(FINAL_REPORT_PATH, content.encode("utf-8"))])
        except Exception as exc:  # noqa: BLE001 - return a stable, sanitized tool error
            logger.warning("Writer final-report commit failed (%s)", type(exc).__name__)
            return self._tool_error(tool_call, "writer_output_commit_failed", "the backend rejected the write")
        if not isinstance(responses, list) or len(responses) != 1 or self._response_error(responses[0]):
            logger.warning("Writer final-report commit returned an unsuccessful upload response")
            return self._tool_error(tool_call, "writer_output_commit_failed", "the backend rejected the write")
        self.tracker.record(content)
        return ToolMessage(
            content=f"Updated file {FINAL_REPORT_PATH}",
            tool_call_id=tool_call.get("id", "final-report-commit"),
            name="write_file",
            status="success",
        )

    async def _refresh_after_edit(self, tool_call: dict[str, object], result: object) -> object:
        if _tool_result_failed(result):
            return result
        try:
            responses = await self.backend.adownload_files([FINAL_REPORT_PATH])
        except Exception as exc:  # noqa: BLE001 - return a stable, sanitized tool error
            logger.warning("Writer final-report verification failed (%s)", type(exc).__name__)
            return self._tool_error(
                tool_call,
                "writer_output_commit_failed",
                "the edited report could not be verified",
            )
        response = responses[0] if isinstance(responses, list) and len(responses) == 1 else None
        content = response.get("content") if isinstance(response, dict) else getattr(response, "content", None)
        if response is None or self._response_error(response) or not isinstance(content, bytes):
            logger.warning("Writer final-report verification returned an unsuccessful download response")
            return self._tool_error(
                tool_call,
                "writer_output_commit_failed",
                "the edited report could not be verified",
            )
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            return self._tool_error(tool_call, "writer_output_commit_failed", "the edited report is not UTF-8")
        self.tracker.record(text)
        return result

    async def awrap_tool_call(self, request, handler):
        """Upsert writer output and track successful writes or edits only."""
        tool_call = request.tool_call if isinstance(getattr(request, "tool_call", None), dict) else {}
        tool_name = tool_call.get("name")
        if tool_name not in {"write_file", "edit_file"}:
            return await handler(request)
        target = _tool_file_path(tool_call)
        if target not in FINAL_REPORT_STATE_PATHS:
            return await handler(request)
        if target != FINAL_REPORT_PATH:
            return self._tool_error(
                tool_call,
                "writer_output_path_invalid",
                f"write the final report to {FINAL_REPORT_PATH}",
            )

        async with self._mutation_lock:
            if tool_name == "write_file":
                return await self._commit_write(tool_call)
            try:
                result = await handler(request)
            except Exception as exc:  # noqa: BLE001 - return a stable, sanitized tool error
                logger.warning("Writer final-report edit failed (%s)", type(exc).__name__)
                return self._tool_error(tool_call, "writer_output_commit_failed", "the backend rejected the edit")
            return await self._refresh_after_edit(tool_call, result)


class RequiredOutputFileMiddleware(AgentMiddleware):
    """Verify a model's file-backed completion marker before ending its run.

    A model can claim that it wrote a file without making the filesystem tool call.
    Keep recovery local to that agent: request one corrective model turn, then fail
    with a stable reason code instead of restarting the surrounding workflow.
    """

    def __init__(
        self,
        *,
        tracker: FinalReportCommitTracker,
        paths: tuple[str, ...] = FINAL_REPORT_STATE_PATHS,
        completion_marker: str = "Wrote /shared/output.md",
        max_retries: int = 1,
        reason_code: str = "writer_output_not_committed",
    ) -> None:
        """Configure the accepted state paths and bounded corrective turns."""
        if not paths:
            raise ValueError("paths must not be empty")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.tracker = tracker
        self.paths = paths
        self.completion_marker = completion_marker
        self.max_retries = max_retries
        self.reason_code = reason_code
        self._retry_message = (
            "The final report is missing, empty, or was not committed by this writer run. "
            "Do not repeat research or regenerate artifacts. "
            f"Call write_file with file_path={paths[0]} and the complete final Markdown, confirm the tool "
            f"succeeds, and only then return `{completion_marker}`."
        )

    @staticmethod
    def _files_from_state(state: object) -> object:
        return state.get("files", {}) if isinstance(state, dict) else getattr(state, "files", {})

    def _required_output_is_committed(self, state: object) -> bool:
        files = self._files_from_state(state)
        return self.tracker.committed_text(files, paths=self.paths) is not None

    def _retry_count(self, messages: list[object]) -> int:
        return sum(isinstance(message, HumanMessage) and message.content == self._retry_message for message in messages)

    def _check_after_model(self, state: object) -> dict[str, object] | None:
        messages = state.get("messages", []) if isinstance(state, dict) else getattr(state, "messages", [])
        if not isinstance(messages, list) or not messages:
            return None
        last_message = messages[-1]
        if not isinstance(last_message, AIMessage) or last_message.tool_calls:
            return None
        if last_message.text.strip() != self.completion_marker:
            return None
        if self._required_output_is_committed(state):
            return None

        retry_count = self._retry_count(messages)
        if retry_count >= self.max_retries:
            raise RuntimeError(self.reason_code)

        logger.warning("Agent reported completion before committing the required output; requesting corrective turn")
        return {
            "messages": [HumanMessage(content=self._retry_message)],
            "jump_to": "model",
        }

    @hook_config(can_jump_to=["model"])
    def after_model(self, state, runtime):
        """Verify synchronous writer completion and request one local repair when needed."""
        return self._check_after_model(state)

    @hook_config(can_jump_to=["model"])
    async def aafter_model(self, state, runtime):
        """Verify asynchronous writer completion and request one local repair when needed."""
        return self._check_after_model(state)


# Common hallucinated tool name mappings
_TOOL_NAME_ALIASES: dict[str, str] = {
    "open_file": "read_file",
    "find": "grep",
    "find_file": "glob",
}


class ToolNameSanitizationMiddleware(AgentMiddleware):
    """
    Middleware that sanitizes corrupted tool names in LLM responses.

    LLMs sometimes generate malformed tool calls with suffixes like
    <|channel|>commentary or .exec, or hallucinate tool names like
    open_file or find. This middleware intercepts the model response
    and fixes tool names before the framework dispatches them.
    """

    def __init__(self, valid_tool_names: list[str]):
        """Store the set of valid tool names used to correct malformed tool calls."""
        self.valid_tool_names = set(valid_tool_names)

    def _sanitize_tool_name(self, name: str) -> str:
        """Sanitize a potentially corrupted tool name.

        Returns the cleaned name if it maps to a valid tool,
        otherwise returns the original name unchanged.
        """
        # 1. Strip <|channel|> and everything after
        if "<|channel|>" in name:
            candidate = name.split("<|channel|>", maxsplit=1)[0]
            if candidate in self.valid_tool_names:
                logger.info("Sanitized tool name: '%s' -> '%s'", name, candidate)
                return candidate

        # 2. Strip dot suffix if base name is valid
        if "." in name:
            candidate = name.split(".", maxsplit=1)[0]
            if candidate in self.valid_tool_names:
                logger.info("Sanitized tool name: '%s' -> '%s'", name, candidate)
                return candidate

        # 3. Map common hallucinated names
        if name in _TOOL_NAME_ALIASES:
            mapped = _TOOL_NAME_ALIASES[name]
            if mapped in self.valid_tool_names:
                logger.info("Mapped tool name: '%s' -> '%s'", name, mapped)
                return mapped

        return name

    async def awrap_model_call(self, request, handler):
        """Intercept model response and sanitize tool names."""
        response = await handler(request)

        needs_fix = False
        for msg in response.result:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    sanitized = self._sanitize_tool_name(tc["name"])
                    if sanitized != tc["name"]:
                        needs_fix = True
                        break
                if needs_fix:
                    break

        if not needs_fix:
            return response

        new_result = []
        for msg in response.result:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                new_tool_calls = []
                for tc in msg.tool_calls:
                    new_tool_calls.append({**tc, "name": self._sanitize_tool_name(tc["name"])})
                new_msg = AIMessage(
                    content=msg.content,
                    tool_calls=new_tool_calls,
                    id=msg.id,
                )
                new_result.append(new_msg)
            else:
                new_result.append(msg)

        return ModelResponse(result=new_result, structured_response=response.structured_response)


def _request_tool_name(tool: object) -> str | None:
    """Return a LangChain model-request tool name across common tool shapes."""
    name = getattr(tool, "name", None)
    if isinstance(name, str):
        return name
    if isinstance(tool, dict):
        dict_name = tool.get("name")
        if isinstance(dict_name, str):
            return dict_name
        function = tool.get("function")
        if isinstance(function, dict):
            function_name = function.get("name")
            if isinstance(function_name, str):
                return function_name
    return None


class ToolVisibilityMiddleware(AgentMiddleware):
    """Hide selected tools from model requests without removing scaffolding middleware."""

    def __init__(self, hidden_tool_names: set[str]) -> None:
        """Store the tool names to hide from model requests."""
        self.hidden_tool_names = hidden_tool_names

    def _filter_tools(self, tools: list[object]) -> list[object]:
        """Return the tool list with hidden tools removed."""
        if not self.hidden_tool_names:
            return tools
        return [tool for tool in tools if _request_tool_name(tool) not in self.hidden_tool_names]

    def wrap_model_call(self, request, handler):
        """Filter hidden tools before a synchronous model call."""
        return handler(request.override(tools=self._filter_tools(request.tools)))

    async def awrap_model_call(self, request, handler):
        """Filter hidden tools before an asynchronous model call."""
        return await handler(request.override(tools=self._filter_tools(request.tools)))


class TodoSuppressionMiddleware(AgentMiddleware):
    """Strip the framework's ``write_todos`` tool and its injected prompt for a subagent.

    deepagents attaches ``TodoListMiddleware`` to every subagent, which adds the
    ``write_todos`` tool plus a system-prompt block telling the agent to use it.
    Agents that own no progress list - e.g. the planner, which returns a single
    structured ``ResearchPlan`` - should not have it. Placed after the framework's
    ``TodoListMiddleware`` in the stack, this removes both the tool and the injected
    prompt block from the model request, keeping todo tracking solely with the
    orchestrator. It is a no-op when neither is present.
    """

    _TODO_TOOL = "write_todos"
    _TODO_PROMPT_MARKER = "## `write_todos`"

    def _clean_request(self, request: object) -> object:
        """Return the request with the write_todos tool and its prompt block removed."""
        overrides: dict[str, object] = {
            "tools": [tool for tool in request.tools if _request_tool_name(tool) != self._TODO_TOOL]
        }
        system_message = getattr(request, "system_message", None)
        if system_message is not None:
            blocks = system_message.content_blocks
            kept = [
                block
                for block in blocks
                if not (isinstance(block, dict) and self._TODO_PROMPT_MARKER in str(block.get("text", "")))
            ]
            if len(kept) != len(blocks):
                overrides["system_message"] = SystemMessage(content=kept)
        return request.override(**overrides)

    def wrap_model_call(self, request, handler):
        """Strip write_todos and its prompt before a synchronous model call."""
        return handler(self._clean_request(request))

    async def awrap_model_call(self, request, handler):
        """Strip write_todos and its prompt before an asynchronous model call."""
        return await handler(self._clean_request(request))


class ToolRetryMiddleware(AgentMiddleware):
    """Retries failed tool calls with exponential backoff.

    Provides uniform retry coverage for all tools. Some tools (e.g., Tavily)
    have their own internal retry; this middleware wraps the outer call so
    tools without retry (knowledge layer, paper search) are also covered.
    """

    def __init__(
        self,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        initial_delay: float = 1.0,
    ):
        """Configure retry count and exponential backoff for failed tool calls."""
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.initial_delay = initial_delay

    async def awrap_tool_call(self, request, handler):
        """Retry tool calls on failure with exponential backoff."""
        delay = self.initial_delay
        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                return await handler(request)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    tool_name = request.tool_call.get("name", "?") if hasattr(request, "tool_call") else "?"
                    logger.warning(
                        "Tool %s failed (attempt %d/%d): %s",
                        tool_name,
                        attempt + 1,
                        self.max_retries + 1,
                        e,
                    )
                    await asyncio.sleep(delay)
                    delay *= self.backoff_factor
        raise last_exception


class SourceRegistryMiddleware(AgentMiddleware):
    """Intercepts tool call results to build a registry of actual sources.

    Two responsibilities:
    1. awrap_tool_call: Capture URLs/citation keys from tool results
    2. awrap_model_call: Inject a consolidated source list into the LLM context
       so the orchestrator has a single, authoritative reference list when
       writing the final report (no manual reconciliation across research-note files)

    Source capture is gated only by the agent's loaded tool set
    (``source_tool_names``). Internal scratchpad/runtime tools (think,
    write_file, read_file, etc.) are added by deepagents itself and never
    appear in that set, so they are implicitly excluded. Tools registered as
    configured data sources additionally carry a ``source_id`` label, but a
    tool does *not* have to be declared under ``data_sources`` to contribute
    sources — agents can be passed citable tools directly.

    The registry is also used by verify_citations() to strip fabricated,
    stale, or intermediate-artifact citations from the final report.
    """

    def __init__(self, source_tool_names: set[str] | None = None) -> None:
        """Create a source registry scoped to the given source-producing tool names."""
        self.registry = SourceRegistry()
        self._source_tool_names = source_tool_names or set()
        self._compact_source_keys: set[str] = set()
        self._lock = asyncio.Lock()

    def active_registry(self) -> SourceRegistry:
        """Return the session-scoped registry if set, otherwise the instance registry."""
        from aiq_agent.common.citation_verification import get_session_registry

        return get_session_registry() or self.registry

    def has_sources(self) -> bool:
        """Return True when the active source registry contains captured sources."""
        return bool(self.active_registry().all_sources())

    @staticmethod
    def _locator_key(locator: str) -> str:
        """Return the comparable key used for source locators and registry entries."""
        locator = locator.strip()
        if locator.startswith(("http://", "https://")):
            from aiq_agent.common.citation_verification import _normalize_url

            return _normalize_url(locator)
        return locator

    @classmethod
    def _entry_key(cls, entry: SourceEntry) -> str | None:
        """Return the comparable key for a registered source entry."""
        if entry.url:
            return cls._locator_key(entry.url)
        if entry.citation_key:
            return entry.citation_key.strip()
        return None

    def register_research_note_sources(self, notes: list[object]) -> None:
        """Mark ResearchNotes source locators as the compact writer-facing citation set."""
        for note in notes:
            sources = getattr(note, "sources", None) or []
            for source in sources:
                locator = getattr(source, "locator", "")
                if isinstance(locator, str) and locator.strip():
                    self._compact_source_keys.add(self._locator_key(locator))

    def register_compact_sources(self, sources: list[SourceEntry]) -> int:
        """Register seeded sources and expose them in the compact citation source list."""
        registry = self.active_registry()
        registered = 0
        for source in sources:
            key = self._entry_key(source)
            if not key:
                continue
            registry.add(source)
            self._compact_source_keys.add(key)
            registered += 1
        return registered

    async def awrap_tool_call(self, request, handler):
        """Capture sources from tool results after execution.

        Capture is gated only by the agent's loaded tool set
        (``source_tool_names``). Internal scratchpad/runtime tools (think,
        write_file, read_file, etc.) are added by deepagents itself and never
        appear in that set, so they are implicitly excluded.

        Tools that resolve to a configured data source via
        :func:`get_source_id_for_tool` get a ``source_id`` label. Tools passed
        directly to the agent without a data-source declaration are still
        captured — their results are real, citable evidence even when
        ``data_source_registry`` does not know about them — but their entries
        carry no ``source_id``.
        """
        result = await handler(request)
        if isinstance(result, ToolMessage) and result.content:
            tool_name = ""
            if hasattr(request, "tool_call") and isinstance(request.tool_call, dict):
                tool_name = request.tool_call.get("name", "")
            if tool_name not in self._source_tool_names:
                return result
            source_id = get_source_id_for_tool(tool_name)
            sources = extract_sources_from_tool_result(tool_name, str(result.content), source_id=source_id)
            async with self._lock:
                active_registry = self.active_registry()
                for source in sources:
                    active_registry.add(source)
            if sources:
                logger.info(
                    "[CitationRegistry] Captured %d source(s) from %s: %s",
                    len(sources),
                    tool_name,
                    [s.url or s.citation_key for s in sources],
                )
        return result

    def _render_source_list_text(self, sources: list[SourceEntry]) -> str | None:
        """Render a consolidated source list from registry entries.

        Returns rendered template text, or None if no sources captured.
        Used by agent.run() to include the source list in retry messages
        when citation quality is poor.
        """
        from urllib.parse import urlparse

        from aiq_agent.common.citation_verification import _normalize_url

        if not sources:
            return None

        seen: set[str] = set()
        template_sources = []
        for entry in sources:
            if entry.url:
                normalized = _normalize_url(entry.url)
                if normalized in seen:
                    continue
                seen.add(normalized)
                if entry.title:
                    title = entry.title
                else:
                    try:
                        title = urlparse(entry.url).netloc.replace("www.", "")
                    except Exception:
                        title = entry.url
                template_sources.append({"title": title, "url": entry.url})
            elif entry.citation_key:
                key = entry.citation_key
                if key in seen:
                    continue
                seen.add(key)
                template_sources.append({"title": key, "url": key})

        if not template_sources:
            return None

        try:
            template = load_prompt(_PROMPTS_DIR, "source_registry")
            return render_prompt_template(template, sources=template_sources)
        except Exception:
            logger.warning("Failed to load source_registry prompt template", exc_info=True)
            return None

    def get_source_entries(self, mode: str = "compact") -> list[SourceEntry]:
        """Return the source entries represented by the writer-facing source list."""
        sources = self.active_registry().all_sources()
        if mode == "full" or not self._compact_source_keys:
            return sources
        compact_sources = [source for source in sources if self._entry_key(source) in self._compact_source_keys]
        return compact_sources or sources

    def get_source_list_text(self, mode: str = "compact") -> str | None:
        """Build a writer-facing verified source list.

        Compact mode returns the subset of registered sources that researcher
        workers actually carried forward in structured ResearchNotes. Full mode
        returns the complete registry.
        """
        return self._render_source_list_text(self.get_source_entries(mode=mode))


class ArtifactDirectoryOwnershipGuardMiddleware(AgentMiddleware):
    """Reserve durable artifact-directory mutation for the writer role.

    This middleware is installed only on researcher, planner, and orchestrator
    stacks. It is deliberately independent of artifact capture so ownership is
    enforced even when a sandbox is configured without a durable store.
    """

    def __init__(self, artifact_dir: str) -> None:
        if not artifact_dir:
            raise ValueError("artifact_dir must not be empty")
        self.artifact_dir = artifact_dir

    def _targets_artifact_directory(self, tool_call: dict[str, object]) -> bool:
        tool_name = tool_call.get("name")
        args = tool_call.get("args")
        if not isinstance(args, dict):
            return False
        if tool_name in {"write_file", "edit_file"}:
            return _path_targets_directory(args.get("file_path", args.get("path")), self.artifact_dir)
        if tool_name == "execute":
            return _command_targets_directory(args.get("command"), self.artifact_dir)
        return False

    async def awrap_tool_call(self, request, handler):
        """Reject non-writer artifact publication before physical tool dispatch."""
        tool_call = request.tool_call if isinstance(getattr(request, "tool_call", None), dict) else {}
        if not self._targets_artifact_directory(tool_call):
            return await handler(request)
        tool_name = str(tool_call.get("name") or "tool")
        return ToolMessage(
            content=(
                "durable_artifact_publication_writer_only: researcher, planner, and orchestrator roles must "
                "handoff canonical evidence through ResearchNotes; only writer-agent may publish durable artifacts."
            ),
            tool_call_id=tool_call.get("id", "artifact-directory-ownership"),
            name=tool_name,
            status="error",
        )


class ArtifactHarvestMiddleware(AgentMiddleware):
    """Checkpoint durable artifacts after writer publication milestones."""

    def __init__(self, artifact_manager: object) -> None:
        """Store the artifact manager used for best-effort checkpoints."""
        self.artifact_manager = artifact_manager

    async def awrap_tool_call(self, request, handler):
        """Checkpoint after chart execution or a successful artifact-manifest write."""
        result = await handler(request)
        tool_name = ""
        tool_args: dict[str, object] = {}
        if hasattr(request, "tool_call") and isinstance(request.tool_call, dict):
            tool_name = request.tool_call.get("name", "")
            tool_args = request.tool_call.get("args") or {}
        should_checkpoint = tool_name == "execute" or (
            tool_name == "write_file" and self._is_artifact_manifest_write(tool_args)
        )
        if should_checkpoint and not _tool_result_failed(result):
            try:
                atomic_harvest = getattr(type(self.artifact_manager), "harvest_after_execute_with_diagnostics", None)
                if callable(atomic_harvest):
                    checkpoint = await asyncio.to_thread(self.artifact_manager.harvest_after_execute_with_diagnostics)
                    captured = getattr(checkpoint, "artifacts", ())
                    rejections = getattr(checkpoint, "rejections", ())
                else:
                    # Compatibility with older/test managers that expose only
                    # the pre-atomic pair of methods.
                    captured = await asyncio.to_thread(self.artifact_manager.harvest_after_execute)
                    rejections = ()
                    rejection_reader = getattr(self.artifact_manager, "last_harvest_rejections", None)
                    if callable(rejection_reader):
                        rejections = rejection_reader()
            except Exception as exc:  # noqa: BLE001 - artifact capture must not fail the agent
                logger.warning("Artifact checkpoint harvest failed (%s)", type(exc).__name__)
            else:
                result = self._append_checkpoint_result(result, captured, rejections)
        return result

    def _is_artifact_manifest_write(self, args: dict[str, object]) -> bool:
        """Return whether a write targets this job's artifact manifest exactly."""
        path = args.get("file_path", args.get("path"))
        artifact_dir = getattr(self.artifact_manager, "artifact_dir", None)
        if not isinstance(path, str) or not isinstance(artifact_dir, str):
            return False
        return path == f"{artifact_dir.rstrip('/')}/manifest.json"

    @staticmethod
    def _append_checkpoint_result(result: object, captured: object, rejections: object = ()) -> object:
        """Tell the model which safe filenames were captured or rejected."""
        if not isinstance(result, ToolMessage) or not isinstance(result.content, str):
            return result

        sections: list[str] = []
        if isinstance(captured, (list, tuple)) and captured:
            lines = ["Artifact checkpoint captured these exact filenames:"]
            for artifact in captured[:10]:
                filename = PurePosixPath(str(getattr(artifact, "filename", ""))).name
                if not filename:
                    continue
                if bool(getattr(artifact, "inline", False)):
                    lines.append(f"- {filename} (inline): embed as ![caption](artifact://{filename})")
                else:
                    lines.append(f"- {filename} (downloadable; not marked inline)")
            if len(lines) > 1:
                sections.append("\n".join(lines))

        if isinstance(rejections, (list, tuple)) and rejections:
            lines = ["Artifact checkpoint rejected these files; do not claim they were published:"]
            for rejection in rejections[:10]:
                if not isinstance(rejection, (list, tuple)) or len(rejection) != 2:
                    continue
                filename = PurePosixPath(str(rejection[0])).name
                reason = str(rejection[1])
                if filename and reason:
                    lines.append(f"- {filename}: {reason}")
            if len(lines) > 1:
                sections.append("\n".join(lines))

        if not sections:
            return result
        content = result.content.rstrip() + "\n\n" + "\n\n".join(sections)
        return result.model_copy(update={"content": content})


class WriterExecuteBudgetMiddleware(AgentMiddleware):
    """Guard durable publication and bound writer-side chart rendering.

    The limiter sits inside generic tool retry middleware, so every call that can
    physically reach the sandbox consumes one attempt. Once exhausted, a later
    model request to execute is ended before tool dispatch. The writer must create
    the report baseline before its first execution, preserving useful text, table,
    and already-published CSV output when chart rendering fails. Artifact-directory
    writes are also refused until persisted researcher output contains canonical
    data and the complete report/table baseline is present; ordinary report writes
    remain unaffected.
    """

    def __init__(
        self,
        *,
        max_attempts: int = 3,
        output_paths: tuple[str, ...] = ("/shared/output.md", "/output.md"),
        artifact_manager: object | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        if not output_paths:
            raise ValueError("output_paths must not be empty")
        self.max_attempts = max_attempts
        self.output_paths = output_paths
        self.artifact_manager = artifact_manager
        self._attempts = 0
        self._attempt_lock = asyncio.Lock()

    @property
    def attempts(self) -> int:
        """Return the number of physical execute attempts admitted so far."""
        return self._attempts

    def _report_baseline_text(self, state: object) -> str | None:
        files = state.get("files", {}) if isinstance(state, dict) else getattr(state, "files", {})
        if not isinstance(files, dict):
            return None
        for path in self.output_paths:
            text = _entry_text(files.get(path))
            if isinstance(text, str) and text.strip() and _contains_markdown_table(text):
                return text
        return None

    def _published_canonical_digests(self) -> frozenset[str]:
        reader = getattr(self.artifact_manager, "published_canonical_digests", None)
        if not callable(reader):
            return frozenset()
        try:
            values = reader()
            if not isinstance(values, (set, frozenset, list, tuple)):
                return frozenset()
            return frozenset(digest for digest in values if isinstance(digest, str))
        except Exception as exc:  # noqa: BLE001 - the safe response is to block execution
            logger.warning("Canonical publication status check failed (%s)", type(exc).__name__)
            return frozenset()

    def _targets_artifact_directory(self, tool_call: dict[str, object]) -> bool:
        """Return whether a write/edit tool targets the configured artifact directory."""
        if tool_call.get("name") not in {"write_file", "edit_file"}:
            return False
        args = tool_call.get("args")
        if not isinstance(args, dict):
            return False
        path = args.get("file_path", args.get("path"))
        artifact_dir = getattr(self.artifact_manager, "artifact_dir", None)
        return isinstance(artifact_dir, str) and _path_targets_directory(path, artifact_dir)

    def _targets_artifact_execute(self, tool_call: dict[str, object]) -> bool:
        """Return whether an execute tool directly targets the artifact directory."""
        if tool_call.get("name") != "execute":
            return False
        args = tool_call.get("args")
        artifact_dir = getattr(self.artifact_manager, "artifact_dir", None)
        if not isinstance(args, dict) or not isinstance(artifact_dir, str):
            return False
        return _command_targets_directory(args.get("command"), artifact_dir)

    @staticmethod
    def _state_has_canonical_dataset(state: object) -> bool:
        """Return whether persisted state contains schema-valid canonical evidence."""
        return any(note.quantitative_datasets for note in validated_research_notes_from_state(state))

    def _required_publication_tables(
        self,
        state: object,
        tool_call: dict[str, object],
    ) -> tuple[str, ...] | None:
        """Resolve the exact canonical tables required by one publication write.

        ``None`` means the target expresses canonical publication intent that
        cannot be tied back to validated researcher output and must fail closed.
        An empty tuple means the target is a non-dataset artifact and the generic
        report-first baseline is sufficient.
        """
        args = tool_call.get("args")
        if not isinstance(args, dict):
            return None
        raw_path = args.get("file_path", args.get("path"))
        if not isinstance(raw_path, str):
            return None
        filename = PurePosixPath(_normalize_guard_path(raw_path)).name
        datasets = [
            dataset for note in validated_research_notes_from_state(state) for dataset in note.quantitative_datasets
        ]

        if filename.endswith(".csv"):
            dataset_id = filename.removesuffix(".csv")
            matches = [dataset for dataset in datasets if dataset.dataset_id == dataset_id]
            if not matches:
                return None
            return tuple(dict.fromkeys(dataset.markdown_table for dataset in matches))

        if filename != "manifest.json":
            return ()
        if tool_call.get("name") != "write_file":
            # An edit patch does not contain the complete post-edit manifest, so
            # its canonical intent cannot be validated deterministically.
            return None
        manifest_text = _entry_text(args.get("content"))
        if manifest_text is None:
            return None
        manifest = parse_manifest(manifest_text)
        if manifest is None:
            return None

        canonical_entries = [
            entry
            for entry in manifest.artifacts
            if entry.kind.value == "dataset" and PurePosixPath(entry.path).suffix.lower() == ".csv"
        ]
        if not canonical_entries:
            return None

        required_tables: list[str] = []
        for entry in canonical_entries:
            stem = PurePosixPath(entry.path).stem
            id_matches = {index for index, dataset in enumerate(datasets) if dataset.dataset_id == stem}
            if entry.expected_sha256 is None:
                selected = id_matches
            else:
                digest_matches = {
                    index for index, dataset in enumerate(datasets) if dataset.csv_sha256 == entry.expected_sha256
                }
                if not digest_matches:
                    return None
                selected = digest_matches & id_matches if id_matches else digest_matches
            if not selected:
                return None
            required_tables.extend(datasets[index].markdown_table for index in sorted(selected))
        return tuple(dict.fromkeys(required_tables))

    @staticmethod
    def _baseline_contains_tables(baseline: str, tables: tuple[str, ...]) -> bool:
        """Return whether the baseline contains each required table byte-for-text."""
        normalized_baseline = baseline.replace("\r\n", "\n").replace("\r", "\n")
        return all(table.replace("\r\n", "\n").replace("\r", "\n") in normalized_baseline for table in tables)

    @staticmethod
    def _contains_all_published_canonical_tables(
        state: object,
        baseline: str,
        published_digests: frozenset[str],
    ) -> bool:
        tables_by_digest: dict[str, set[str]] = {}
        for note in validated_research_notes_from_state(state):
            for dataset in note.quantitative_datasets:
                if dataset.csv_sha256 in published_digests:
                    tables_by_digest.setdefault(dataset.csv_sha256, set()).add(dataset.markdown_table)

        if not published_digests.issubset(tables_by_digest):
            return False
        return all(table in baseline for digest in published_digests for table in tables_by_digest[digest])

    @staticmethod
    def _tool_message(request, content: str, *, name: str = "execute") -> ToolMessage:
        tool_call = request.tool_call if isinstance(getattr(request, "tool_call", None), dict) else {}
        return ToolMessage(
            content=content,
            tool_call_id=tool_call.get("id", "writer-execute-budget"),
            name=name,
            status="error",
        )

    async def awrap_tool_call(self, request, handler):
        """Admit at most ``max_attempts`` physical writer execute calls."""
        tool_call = request.tool_call if isinstance(getattr(request, "tool_call", None), dict) else {}
        publication_write = self._targets_artifact_directory(tool_call)
        if publication_write:
            tool_name = str(tool_call.get("name") or "write_file")
            if not self._state_has_canonical_dataset(request.state):
                return self._tool_message(
                    request,
                    "writer_canonical_dataset_missing: durable artifact publication requires a schema-valid "
                    "canonical quantitative dataset in persisted ResearchNotes.",
                    name=tool_name,
                )
            baseline = self._report_baseline_text(request.state)
            if baseline is None:
                return self._tool_message(
                    request,
                    "writer_report_baseline_missing: write the complete report and Markdown-table baseline "
                    "to /shared/output.md before durable artifact publication.",
                    name=tool_name,
                )
            required_tables = self._required_publication_tables(request.state, tool_call)
            if required_tables is None:
                return self._tool_message(
                    request,
                    "writer_canonical_dataset_missing: the artifact publication target could not be resolved "
                    "to a schema-valid canonical dataset in persisted ResearchNotes.",
                    name=tool_name,
                )
            if not self._baseline_contains_tables(baseline, required_tables):
                return self._tool_message(
                    request,
                    "writer_canonical_table_missing: copy the exact canonical Markdown table for this published "
                    "dataset into /shared/output.md before durable artifact publication.",
                    name=tool_name,
                )
        if tool_call.get("name") != "execute":
            return await handler(request)
        if not self._targets_artifact_execute(tool_call):
            return await handler(request)
        baseline = self._report_baseline_text(request.state)
        if baseline is None:
            return self._tool_message(
                request,
                "writer_report_baseline_missing: write the complete report and Markdown-table baseline "
                "to /shared/output.md before chart execution.",
            )
        published_digests = self._published_canonical_digests()
        if not published_digests:
            return self._tool_message(
                request,
                "writer_canonical_dataset_unpublished: publish and checkpoint the validated canonical CSV "
                "before chart execution.",
            )
        if not self._contains_all_published_canonical_tables(request.state, baseline, published_digests):
            return self._tool_message(
                request,
                "writer_canonical_table_missing: copy the exact canonical Markdown table from the validated "
                "research note into /shared/output.md before chart execution.",
            )

        async with self._attempt_lock:
            if self._attempts >= self.max_attempts:
                return self._tool_message(
                    request,
                    "writer_execute_budget_exhausted: chart execution is disabled; preserve the report, table, "
                    "and published CSV and finish without a PNG.",
                )
            self._attempts += 1
        result = await handler(request)
        if self._attempts >= self.max_attempts and _tool_result_failed(result):
            return self._tool_message(
                request,
                "writer_execute_budget_exhausted: chart execution failed at the attempt limit; preserve the "
                "report, table, and published CSV and finish without a PNG.",
            )
        if _tool_result_failed(result) and isinstance(result, ToolMessage) and result.status != "error":
            return result.model_copy(update={"status": "error"})
        return result

    @staticmethod
    def _last_ai_message(state: object) -> AIMessage | None:
        messages = state.get("messages", []) if isinstance(state, dict) else getattr(state, "messages", [])
        return next((message for message in reversed(messages) if isinstance(message, AIMessage)), None)

    @hook_config(can_jump_to=["end"])
    async def aafter_model(self, state, runtime):
        """End a post-exhaustion execute request before it reaches the sandbox."""
        if self._attempts < self.max_attempts:
            return None
        last_message = self._last_ai_message(state)
        if last_message is None or not any(self._targets_artifact_execute(call) for call in last_message.tool_calls):
            return None

        tool_messages = [
            ToolMessage(
                content=(
                    "writer_execute_budget_exhausted: no further chart execution is allowed; "
                    "the existing report, table, and CSV remain the final output."
                    if self._targets_artifact_execute(call)
                    else "Tool call cancelled because the writer execution budget is exhausted."
                ),
                tool_call_id=call.get("id", "writer-execute-budget"),
                name=call.get("name"),
                status="error",
            )
            for call in last_message.tool_calls
        ]
        return {
            "jump_to": "end",
            "messages": [
                *tool_messages,
                AIMessage(content="Wrote /shared/output.md\n\nwriter_execute_budget_exhausted"),
            ],
        }


class PlanPersistenceMiddleware(AgentMiddleware):
    """Persists the planner's structured ResearchPlan to the shared filesystem.

    The planner returns a schema-validated ``ResearchPlan`` (``response_format``).
    This middleware writes that plan to ``/shared/plan.json`` deterministically via
    the overwrite-safe ``backend.upload_files`` (the same state-channel write
    ``run_research_batch`` uses for ResearchNotes), so the planner never performs
    file I/O itself. Keeping the write off the LLM removes the ``write_file`` /
    ``edit_file`` loop the planner otherwise hits when ``/shared/plan.json`` already
    exists, since the LLM ``write_file`` tool refuses to overwrite while
    ``upload_files`` overwrites in place.

    Persistence failures propagate so the planner task fails before the
    orchestrator reads a missing or stale ``/shared/plan.json``.
    """

    def __init__(self, backend: object, *, path: str = "/shared/plan.json") -> None:
        """Initialize the middleware.

        Args:
            backend: Shared filesystem backend exposing ``upload_files``.
            path: Shared path the serialized plan is written to.
        """
        self.backend = backend
        self.path = path

    @staticmethod
    def _plan_from_state(state: object) -> object:
        """Extract the planner's ``structured_response`` from dict or attribute state."""
        if isinstance(state, dict):
            return state.get("structured_response")
        return getattr(state, "structured_response", None)

    def _persist_plan(self, plan: object) -> None:
        """Serialize a structured ResearchPlan and upload it to shared state."""
        if plan is None:
            return
        if hasattr(plan, "model_dump"):
            payload = plan.model_dump(mode="json", exclude_none=True)
        elif isinstance(plan, dict):
            payload = plan
        else:
            return
        content = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
        responses = self.backend.upload_files([(self.path, content)])
        errors = [f"{response.path}: {response.error}" for response in responses if getattr(response, "error", None)]
        if errors:
            # Raw backend detail stays in logs; the raised error reaches the job status /
            # caller, so it must not echo backend-internal strings (hostnames, paths, etc.).
            logger.error("Failed to persist plan to %s: %s", self.path, "; ".join(errors))
            raise RuntimeError(f"Failed to persist the research plan to {self.path}")

    def after_agent(self, state, runtime):
        """Persist the plan once the synchronous planner run completes."""
        self._persist_plan(self._plan_from_state(state))

    async def aafter_agent(self, state, runtime):
        """Persist the plan once the asynchronous planner run completes."""
        await asyncio.to_thread(self._persist_plan, self._plan_from_state(state))


class ToolResultPruningMiddleware(AgentMiddleware):
    """Truncates older tool results to keep context manageable.

    Keeps the last N tool results intact and truncates older ones to
    reduce "lost in the middle" degradation. Operates on awrap_model_call
    so the full results are still available for SourceRegistryMiddleware.
    """

    def __init__(self, keep_last_n: int = 3, max_chars: int = 500):
        """Configure how many recent tool results to keep intact and the truncation cap."""
        self.keep_last_n = keep_last_n
        self.max_chars = max_chars

    async def awrap_model_call(self, request, handler):
        """Truncate older ToolMessage content before sending to the model."""
        # Find all ToolMessage indices
        tool_indices = [i for i, msg in enumerate(request.messages) if isinstance(msg, ToolMessage)]

        if len(tool_indices) <= self.keep_last_n:
            return await handler(request)

        # Indices to truncate: all but the last keep_last_n
        truncate_indices = set(tool_indices[: -self.keep_last_n])

        pruned_messages = []
        for i, msg in enumerate(request.messages):
            if i in truncate_indices and isinstance(msg, ToolMessage) and msg.content:
                content = str(msg.content)
                if len(content) > self.max_chars:
                    truncated_content = content[: self.max_chars] + "\n\n[... truncated ...]"
                    pruned_messages.append(
                        ToolMessage(
                            content=truncated_content,
                            tool_call_id=msg.tool_call_id,
                            name=getattr(msg, "name", None),
                            id=msg.id,
                        )
                    )
                else:
                    pruned_messages.append(msg)
            else:
                pruned_messages.append(msg)

        return await handler(request.override(messages=pruned_messages))
