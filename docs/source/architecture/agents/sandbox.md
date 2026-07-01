<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Deep Research Sandbox Notes

Deep research can optionally run DeepAgents `execute` calls through a sandbox provider
(Modal, OpenShell, or any registered provider). Modal and OpenShell create a fresh physical
sandbox per job. OpenShell binds the configured policy at creation and attests readiness plus
the loaded policy revision before exposing the execution backend.

The sandbox is an internal execution detail. There are no sandbox-specific API
endpoints, and job-level auth remains responsible for submit, stream, status,
cancel, state, and report access. The one user-visible surface is the artifact
runtime (`.../job/{job_id}/artifacts`), which is also auth-scoped to the job.

> **Developer reference:** the full architecture, provider contract, config schema,
> artifact pipeline, and troubleshooting live next to the code in
> [`src/aiq_agent/agents/deep_researcher/sandbox/README.md`](../../../../src/aiq_agent/agents/deep_researcher/sandbox/README.md).

## Current Behavior

- Modal and OpenShell use one physical sandbox per deep research job. OpenShell shared
  attachment is available only through an explicit debug-only opt-in and is not job-isolated.
- Synchronous sandbox-enabled runs use an internal per-agent runtime ID.
- Providers are selected by config (`sandbox.provider` + `providers.<name>`); the
  provider is validated against the registry and gated by its declared capabilities.
  OpenShell policy YAML is parsed strictly against the installed SDK schema, applied in the
  job's creation spec, checked against the declared network upper bound, and attested before use.
- Job IDs must satisfy each provider's object-name rules (Modal: 64 chars or fewer,
  alphanumeric plus dash/period/underscore).
- `timeout` bounds individual execution. Other lifecycle controls are provider-dependent.
- Files written inside the workdir are temporary scratch state. Durable text should
  be written through DeepAgents virtual paths such as `/shared/`; durable binaries
  (charts, CSVs) are captured by the artifact runtime.

## Operational Notes

- High-concurrency Modal and OpenShell runs create one sandbox per job. Optional submit-path
  caps (`AIQ_MAX_SANDBOXES_PER_PRINCIPAL` / `AIQ_MAX_SANDBOXES_GLOBAL`, default-off) bound
  concurrency and cost.
- Custom client-supplied job IDs must not be reused for a new job.
- The runtime closes provider sessions on success, failure, cancellation, and timeout.
- Per-job OpenShell mode requires `delete_on_exit: true`. A persistent shared sandbox is
  possible only through the explicit debug attachment settings.

## Current Safeguards

The following safeguards are in place:

- Explicit sandbox cleanup on success, failure, cancellation, and timeout.
- Idempotency-gated retry-on-stale-container handling.
- Artifact capture for generated charts/binaries (validate -> store -> serve/embed),
  with MIME-from-bytes spoof rejection, SVG sanitization, and an inline-render allowlist.
- Sandbox quota and concurrency controls, and artifact retention via job-expiry cleanup.
- Structured lifecycle logging for sandbox create, reuse, failure, and cleanup.
- Structured `sandbox.attestation` and `sandbox.cleanup` events.
