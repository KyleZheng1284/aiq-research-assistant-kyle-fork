# AI-Q sandbox and artifact follow-up plan

Updated 2026-07-01 after `develop` advanced to `09da539` and sandbox/artifact PR #280 merged as `e9c0d4f`.

Implementation status on `codex/aiq-sandbox-followups`:

- **Implemented:** per-job OpenShell creation, strict policy parsing, readiness/revision
  attestation, fail-closed Landlock and network-policy checks, secret-free creation specs,
  and an explicit shared-debug escape hatch.
- **Implemented:** manifest-only artifact checkpoints, idempotent terminal finalization on
  success/failure/cancellation, canonical durable `artifact.update` events, and final event
  flushing before teardown completes.
- **Implemented:** sanitized attestation, policy-denial, and cleanup events plus fake-SDK,
  middleware, runtime, runner, and EventStore contract coverage.
- **Pending external validation/scope:** live-gateway CPU/memory resource mapping, a portable
  disk quota mechanism, effective-policy hash/field attestation, and the two-job live smoke.

Tickets in scope:

- AIQ-3531 — OpenShell fail-closed policy attestation and per-job isolation
- AIQ-3425 — ARTIFACTS-5 harvest hook
- AIQ-3432 — SANDBOX-5 security controls
- AIQ-3433 — SANDBOX-6 lifecycle tests

## What the merged PR already provides

- Provider-neutral `SandboxProvider`, registry, capability gate, lazy session creation, idempotency-gated retry, and `close()` / `terminate()` lifecycle.
- Job-scoped workspace and artifact directories inside a sandbox.
- Durable artifact validation, quota enforcement, SQL persistence, report reference resolution, UI/API content endpoints, and retention cleanup.
- Successful-run artifact harvesting and best-effort report post-processing.
- OpenShell adapter integration, bounded/confined artifact transfer, setup script, policy YAML, and normal/cancel teardown seams.
- Unit coverage for provider lifecycle, cancellation preemption, sandbox disappearance, artifact validation/quotas, and OpenShell file transfer.

These were the foundations inherited from #280. The gaps below drove this follow-up; the
branch status above records which items are now implemented and which require live gateway
validation or an upstream/scope decision.

## Ticket gap matrix

| Ticket | Covered by #280 | Remaining work | Disposition |
| --- | --- | --- | --- |
| AIQ-3531 | Job-scoped logical name/path, lifecycle seam, policy file, OpenShell SDK 0.0.72 | Live two-job smoke and deeper effective-policy attestation only | Core implementation complete in this branch |
| AIQ-3425 | Durable manager, successful-run final harvest, quotas/dedup/store, best-effort report behavior | Optional richer binary file card if product UX requires it | Core implementation and durable event round trip complete |
| AIQ-3432 | Host-side tools keep model/search credentials out of sandbox by default; provider-neutral network gate; CPU/memory model; execution timeout; artifact quotas | Live-validate CPU/memory mapping; decide portable disk quota and total-lifetime watchdog scope | Isolation, policy floor, network bound, secret-free spec, and telemetry complete |
| AIQ-3433 | Many component tests already exist | Live two-job smoke and any acceptance tests tied to the resource/disk decisions | Automated regression matrix expanded in this branch |

## Architecture decisions that replace the old AIQ-3531 plan

### 1. Create a physical OpenShell sandbox per job

Default to in-process SDK creation, not attachment to `providers.openshell.sandbox_name`:

1. Load and strictly validate the configured policy YAML.
2. Convert the YAML to the OpenShell 0.0.72 `SandboxPolicy` proto. The only schema alias required by the checked-in policy is top-level `filesystem_policy` to SDK field `filesystem`; use protobuf `ParseDict(..., ignore_unknown_fields=False)` so unknown or misspelled controls fail closed.
3. Construct `SandboxSpec` with the selected OpenShell image, empty environment/providers, policy proto, and labels containing the AI-Q job id.
4. Enter `openshell.Sandbox(spec=..., delete_on_exit=True, ...)`. This creates a fresh gateway sandbox with an auto-generated physical name. The existing `self.sandbox_name` remains the deterministic logical job name for pre-session identity/logging; the live backend id/name comes from the created sandbox.

This is preferable to the old CLI subprocess proposal because it avoids shell/process lifecycle coupling and uses the SDK session that AI-Q already owns. If a deterministic gateway-side name is a hard acceptance criterion, treat that as a narrow follow-up: OpenShell 0.0.72's public Python `SandboxClient.create()` does not accept a name, whereas the CLI does. Physical per-job isolation does not require a deterministic gateway name.

Do not silently fall back to a shared sandbox. Keep shared attachment only behind an explicit debug-only setting such as `existing_sandbox_name` plus `allow_shared_sandbox: true`, with a startup warning and documentation that it does not satisfy AIQ-3531.

### 2. Attest before returning the backend

After entering the SDK context and before constructing/returning `OpenShellSandbox`:

- Require the returned sandbox phase to be `SANDBOX_PHASE_READY` (the context's wait-ready step should already establish this; verify it explicitly for the attestation record).
- When attestation is enabled, require `current_policy_version > 0`.
- Change `expected_policy_version` from the old plan's `str | None` to `int | None`, matching OpenShell 0.0.72.
- If an expected version is configured, require an exact match.
- Log/emit only sandbox identity, observed phase, observed policy version, and outcome; never policy contents or environment values.
- On any failure, exit the SDK context so the newly created sandbox is deleted, then raise. Never return an unattested backend.

Version attestation proves that a policy revision loaded, not that every effective field matches the source YAML. Full `GetSandboxConfig` policy-hash/field comparison remains a separate hardening item because the 0.0.72 Python client does not expose that RPC publicly.

### 3. Fix both config layers and secure defaults

The merged code has two config surfaces, and both must change:

- `DeepResearchSandboxConfig` in `deepagents_runtime.py` (the actual AI-Q YAML surface).
- `OpenShellProviderConfig` / `SandboxConfig` in `sandbox/config.py` (the provider-neutral runtime model).

Required changes:

- Add `attest: bool = True` and `expected_policy_version: int | None = None` to both layers and map them in `_create_sandbox_backend`.
- Make per-job creation the default and make `delete_on_exit=True` the public default. In per-job mode, reject `delete_on_exit=False` rather than allowing leaked job sandboxes.
- Replace the overloaded shared `sandbox_name` behavior with explicit fields: a logical `name_prefix` if needed, and a separately named debug-only `existing_sandbox_name`.
- Resolve an OpenShell image for per-job creation. The current public `image` field is documented/mapped only for Modal even though `OpenShellProviderConfig` also has an image field. Either make `image` mean the selected provider's image with provider-specific defaults, or add an explicit `openshell_image`; do not accidentally create OpenShell jobs from the Modal default image.
- Expose provider-neutral CPU/memory resource settings on `DeepResearchSandboxConfig` and map them; currently they exist only in the inner `SandboxConfig` and cannot be set by real AI-Q YAML.
- Expand the public network shape to `blocked | allowlist | open` (plus allow entries) or explicitly make the OpenShell policy file authoritative. The current example says `network: blocked` while its policy allows named endpoints, which conflicts with the inner model's documented meaning of `blocked` as no outbound network.

### 4. Enforce a production policy floor

Attesting a weak policy is not sufficient.

- Production mode must require `landlock.compatibility: hard_requirement` so lack of filesystem enforcement fails sandbox creation.
- `best_effort` remains an explicitly opted-in local demo mode and must be labeled as not satisfying the production security ticket.
- Validate that no host secrets are copied into `SandboxSpec.environment`, no OpenShell credential providers are attached by default, and emitted logs/events cannot include environment values.
- A deny-by-default network policy may have an explicit allowlist, but an unrestricted policy must require an explicit insecure/debug override.

## AIQ-3425 harvest completion

### 1. Restore and actually wire checkpoint harvesting

The original PR briefly contained `ArtifactHarvestMiddleware`, but it was removed because it was never wired. Add a wired checkpoint seam after successful sandbox `execute` tool calls:

- Call a manifest-only `harvest_after_execute()` off the event loop.
- Keep failures best-effort and non-propagating.
- Preserve deduplication so checkpoint plus final scan cannot double-store or double-emit.

Checkpoint harvesting is important for cancellation: OpenShell teardown deletes the physical sandbox to interrupt an in-flight command, so artifacts already persisted before cancellation survive even when a final sandbox scan cannot.

### 2. Centralize terminal finalization

Add one idempotent runtime finalizer used by the job runner:

- Success: the agent keeps final harvest + report reference rewriting; finalizer flushes any remaining harvest/event work and closes.
- Agent failure: best-effort final harvest, flush events, then close.
- Cancellation/timeout: attempt a short bounded checkpoint/final harvest when the provider is responsive, then terminate. Never wait indefinitely behind an in-flight execute; incremental checkpoints are the durable fallback.
- Harvest exceptions and cleanup exceptions are separately logged/emitted and never replace the original job result/error.

Remove the current runner assertion that teardown never harvests and replace it with ordering assertions.

### 3. Emit the canonical event envelope

Change generated-artifact events from the current top-level `type: artifact` payload to the API/UI contract:

```text
type: artifact.update
name: <filename>
data:
  type: file
  content: <authenticated content URL or stable artifact reference>
  artifact_id: ...
  kind: ...
  mime_type: ...
  size_bytes: ...
  sha256: ...
  inline: ...
```

Update the UI adapter/types if binary generated artifacts need a richer file card than the existing text-file `content` contract. Add an SSE/API round-trip test; a manager-only unit assertion is not enough.

## AIQ-3432 remaining controls

### Already acceptable after AIQ-3531

- Physical job isolation.
- Policy bound at creation and attested before execution.
- Host secrets absent by default.
- Default-deny network/filesystem/process policy with production Landlock enforcement.
- Command timeout and deterministic teardown.

### Still requiring work or an explicit scope decision

- CPU and memory: expose the existing provider-neutral config and map it to OpenShell creation. Validate the exact OpenShell 0.0.72 SDK `SandboxTemplate.resources` shape against a live gateway before declaring the capability. Until that validation passes, OpenShell must continue to report resource limits as unsupported and fail closed when limits are requested.
- Disk: OpenShell 0.0.72 exposes CPU/memory CLI controls but no portable disk-quota field in the checked Python SDK. Artifact byte quotas are not a sandbox disk quota. Either implement a compute-driver-specific disk limit with attestation or record an upstream dependency/scope change; do not claim disk enforcement from artifact quotas.
- Time: retain per-execute clamping, add a terminal lifecycle test for job timeout, and document whether `timeout` is command lifetime or total sandbox/job lifetime. If total lifetime is required, add a runner watchdog rather than relying only on adapter command timeout.
- Policy denial events: add a provider-to-runtime event callback. Emit sanitized `sandbox.policy_denied` when an OpenShell execution result contains a typed/known denial. Gateway log streaming is not exposed by the 0.0.72 Python client, so full denial-log forwarding is a separate CLI/RPC integration; document that boundary.
- Cleanup events: emit `sandbox.cleanup` with `started | succeeded | failed`, provider, logical job sandbox name, and no secrets. Flush the event before closing the event store.

## AIQ-3433 acceptance matrix

Use fake SDK/client/context objects in normal CI; no live gateway is required. Remove the module-level `pytest.importorskip` pattern for the core OpenShell lifecycle tests so missing optional dependencies do not silently skip the ticket's acceptance coverage.

Required tests:

1. Two job ids create two distinct physical sandbox contexts/spec labels and never attach to the configured legacy name.
2. Valid READY + positive policy version returns a backend.
3. Non-READY, zero policy version, and expected-version mismatch each delete and raise before any execute.
4. Policy YAML unknown fields, wrong types, missing file, and insecure Landlock mode fail before creation.
5. Normal success ordering: final harvest/event flush precedes close/delete.
6. Agent failure ordering: best-effort harvest precedes close/delete and the original agent error remains authoritative.
7. Cancellation: previously checkpointed artifacts remain available; bounded terminal harvest does not block termination; delete occurs exactly once.
8. Sandbox disappearance: idempotent reads/downloads may recreate once; non-idempotent execute is never replayed.
9. Quota exceeded: accepted artifacts persist, warning event is emitted, report/job result is not corrupted, and total/count accounting is stable across repeated harvests.
10. Secret test: with representative host keys set, created spec/env, logs, and lifecycle events contain none of their names/values unless an explicit provider is configured.
11. Artifact event contract: stored `artifact.update` survives EventStore -> SSE -> UI adapter/API reconstruction with the correct job-scoped content URL.
12. Optional manual/live smoke: create two concurrent jobs, verify distinct OpenShell sandboxes and loaded revisions, cancel one, verify its sandbox disappears and the other remains usable.

## Original recommended PR sequence

1. **AIQ-3531 / isolation-attestation PR** — per-job SDK creation, strict policy loader, attestation, secure config defaults, setup/config/docs migration, fake lifecycle tests.
2. **AIQ-3425 / harvest-terminal PR** — wired checkpoints, idempotent terminal finalizer, failure/cancel harvest behavior, canonical `artifact.update`, event flushing.
3. **AIQ-3432 / controls-telemetry PR** — public resource mapping, live CPU/memory validation, disk decision, secret assertions, policy-denial/cleanup events.
4. **AIQ-3433 closure** — keep tests in the behavior PRs; use this ticket to track and run the final cross-layer matrix plus the live two-job smoke.

The current branch combines the tightly coupled isolation, terminal-harvest, telemetry, and
test work so the runner has one coherent lifecycle. CPU/memory, disk, and deeper effective
policy attestation remain separate because they require live gateway evidence or an upstream
scope decision.

## Verification gates

- `ruff check` and `ruff format --check` on changed Python files.
- Targeted sandbox, artifact, runtime, agent, and runner pytest suites.
- API/UI event contract tests for `artifact.update`.
- `bash -n scripts/setup_openshell.sh`.
- Config parsing test for `configs/config_openshell.yml`.
- Optional-dependency-absent test for the fake OpenShell suite.
- Manual OpenShell 0.0.72 smoke with two simultaneous jobs, version attestation, cancellation, and cleanup.

## Risks and follow-ups

- The primary risk is no longer YAML parsing; the checked policy maps cleanly to the 0.0.72 proto. The private proto import is still version-sensitive, so isolate it in one helper, pin/test the supported SDK range, and fail with a clear compatibility error.
- Auto-generated gateway sandbox names are physically isolated but not deterministic. If operators require deterministic names, request/consume a public SDK name parameter or add a tightly scoped CLI creation adapter.
- A positive policy revision is weaker than a policy hash/full effective-policy comparison. Keep the deeper attestation follow-up visible.
- Cancellation can destroy a sandbox before an end-only scan. Checkpoint harvesting plus a bounded terminal attempt is the practical guarantee with the current SDK; document that artifacts from the command actively being killed may be unrecoverable.
- CPU/memory support must not be declared until the SDK resource shape is live-validated. Disk quota needs an explicit upstream or driver-specific solution.
