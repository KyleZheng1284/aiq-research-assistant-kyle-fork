<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# OpenShell Deployment

This is the canonical operator guide for running AI-Q deep-research code in
[NVIDIA OpenShell](https://docs.nvidia.com/openshell/latest/about/installation).
It owns the setup, deployment, acceptance, and troubleshooting contract. The
architecture and implementation pages describe invariants and extension points;
they intentionally link here for operational steps.

## Purpose and Security Boundary

OpenShell executes code generated during deep research. AI-Q orchestration,
inference, retrieval tools, credentials, checkpoints, and report state remain on
the host side. AI-Q does not copy the host environment or model credentials into
the sandbox creation specification.

In production, every job receives a distinct physical sandbox bound to the
submitted policy. AI-Q verifies the authoritative policy source, content, hash,
and revision before exposing the execution adapter. Successful, failed,
timed-out, and cancelled jobs must delete the sandboxes they own.

Attaching to an existing shared sandbox is an explicit debug escape hatch. It is
not job-isolated and is not a production mode. OpenShell is an external runtime
and authentication boundary, not merely a Python dependency: an operator must
own the gateway service, registration, credentials, version, and availability.

## Supported Platforms

| Path | Intended use | Landlock | Gateway owner | AI-Q status |
|---|---|---|---|---|
| Linux + Docker | Production acceptance | `hard_requirement` | systemd or external operator | Tested path after the live suite passes |
| macOS + Docker Desktop | Local demo | `best_effort` permitted explicitly | Homebrew | Demo only |
| Linux + Podman | Operator-managed evaluation | `hard_requirement` | external operator/system service | Supported upstream, not automated or certified by PR #298 |
| Remote authenticated gateway | Managed deployment | Gateway-host dependent | external operator | Accepted only after the AI-Q live suite passes |
| Windows/WSL | -- | -- | -- | Outside current AI-Q setup-script support |

OpenShell supports Docker and rootless Podman upstream. AI-Q's provisioning
script and PR #298 acceptance certify only the exercised Docker path. Follow the
[official OpenShell installation documentation](https://docs.nvidia.com/openshell/latest/about/installation)
for gateway-host prerequisites and upstream runtime support.

## Version Compatibility

`setup_openshell.sh` installs one exact OpenShell CLI/SDK version and reasserts
that pin after installing the DeepAgents adapter. The strict gateway launcher
then requires the CLI, SDK, and gateway versions to match. An operator may select
another exact release with `--openshell-version`, but a version is production
accepted only after the pytest-owned live suite passes against that release on
Linux with hard Landlock enforcement.

Runtime security decisions are capability-based, not version-specific. A zero
generic current or active policy version is treated as unreported only when the
positive loaded revision and effective config agree exactly. Every positive
reported version must match.

## Responsibility and Lifecycle Ownership

| Component | Owns | Must not do |
|---|---|---|
| `setup_openshell.sh` | SDK/adapter install, policy generation, image build | Start, stop, register, select, probe, or kill gateways |
| Homebrew/systemd/external operator | Long-running gateway service and credentials | Delegate process ownership to AI-Q setup |
| `start_openshell_gateway.sh` | Validate registration/auth, optionally start a packaged service, select the gateway, and run the strict disposable capability probe | Launch raw gateway binaries, stop externally managed services, persist credentials |
| AI-Q runtime | Per-job create, readiness, attestation, execution, and terminal deletion | Reuse a shared sandbox without explicit debug opt-in |
| Live pytest fixtures | Acceptance-test resources and verified teardown | Leave resources for manual cleanup |

Provisioning and long-running service lifecycle are deliberately separate. E2E
shutdown never stops a Homebrew-, systemd-, or operator-managed gateway.

## Policy and AI-Q Config Pairing

Both policy layers are enforced:

- The OpenShell policy is authoritative at the gateway.
- `network` and `network_allow` in the AI-Q config are an upper bound on that
  policy. They never grant additional access.
- `network: blocked` permits no policy endpoint.
- `network: allowlist` requires every endpoint to have a non-empty normalized
  host, and every host must appear in `network_allow`.
- Hostless endpoints and `allowed_ips` or CIDR exceptions are rejected because
  the public AI-Q policy does not model those exceptions.
- Production requires both `landlock.compatibility: hard_requirement` in the
  policy and `require_hard_landlock: true` in the AI-Q config.
- A local demo using `best_effort` requires both the policy value
  `best_effort` and `require_hard_landlock: false` in a local config copy.
- Custom policies must explicitly include OpenShell's proxy filesystem baseline,
  including read-only `/proc`. Otherwise the supervisor creates an enriched
  revision whose content and hash correctly fail AI-Q's exact attestation. The
  generated policy from `setup_openshell.sh` already includes this baseline.

Any mismatch fails closed before the execution adapter is available. Keep policy
files and local config copies out of commits when they contain environment-specific
details. Never put credentials in either file.

## Environment Contract

The gateway launcher, AI-Q runtime, and live suite use these non-secret settings:

| Variable | Default | Purpose |
|---|---|---|
| `AIQ_OPENSHELL_LIVE_TESTS` | unset | Must equal `1` to enable live tests |
| `AIQ_OPENSHELL_GATEWAY_NAME` | active gateway | Registered gateway name |
| `AIQ_OPENSHELL_POLICY_FILE` | `configs/openshell/generated/aiq-openshell-policy.yaml` | Policy submitted and attested |
| `AIQ_OPENSHELL_IMAGE` | `aiq-openshell-demo:latest` | Prebuilt sandbox image |
| `AIQ_OPENSHELL_EXPECTED_GATEWAY_VERSION` | installed SDK version | Optional exact live-test override |
| `AIQ_OPENSHELL_LIVE_ALLOW_BEST_EFFORT` | unset | Explicit non-production macOS/demo opt-in |

## Linux Production Acceptance

First install and register an authenticated packaged gateway, or arrange an
externally operated gateway, using the official OpenShell documentation. The
registration must use HTTPS and mTLS, OIDC, or trusted edge authentication.
Do not launch a raw `openshell-gateway` process.

From the AI-Q repository root, provision the pinned SDK, hard policy, and image:

```bash
./scripts/setup_openshell.sh \
  --policy offline \
  --landlock-compatibility hard_requirement
```

Select the authenticated registration and prove version, policy, selector,
execution, and deletion capabilities. Omit `--reuse-existing` only when the gateway is a local packaged
service that the launcher may start through systemd.

```bash
./scripts/start_openshell_gateway.sh \
  --gateway-name openshell \
  --image-name aiq-openshell-demo:latest \
  --policy-file configs/openshell/generated/aiq-openshell-policy.yaml
```

Export the same image and policy for the AI-Q process:

```bash
export AIQ_OPENSHELL_GATEWAY_NAME=openshell
export AIQ_OPENSHELL_IMAGE=aiq-openshell-demo:latest
export AIQ_OPENSHELL_POLICY_FILE="$PWD/configs/openshell/generated/aiq-openshell-policy.yaml"
```

Validate and start AI-Q with the production pairing in
`configs/config_openshell.yml`:

```bash
.venv/bin/nat validate --config_file configs/config_openshell.yml
./scripts/start_e2e.sh --config_file configs/config_openshell.yml
```

In a separate shell with the same exported settings, run the required acceptance
suite:

```bash
AIQ_OPENSHELL_LIVE_TESTS=1 \
AIQ_OPENSHELL_GATEWAY_NAME=openshell \
AIQ_OPENSHELL_POLICY_FILE=configs/openshell/generated/aiq-openshell-policy.yaml \
AIQ_OPENSHELL_IMAGE=aiq-openshell-demo:latest \
.venv/bin/python -m pytest -m integration -vv \
  tests/aiq_agent/agents/deep_researcher/sandbox/test_openshell_live.py
```

Only this Linux, hard-Landlock run can be recorded as production acceptance.

## macOS Local Demo

Use Docker Desktop and a Homebrew-managed OpenShell service. Install and register
OpenShell as described by the official guide. The launcher can start the packaged
service with `brew services`; it never starts the raw gateway binary.

macOS ships Bash 3.2. Install Bash 5 when the setup script reports unsupported
Bash behavior:

```bash
brew install bash
/opt/homebrew/bin/bash ./scripts/setup_openshell.sh \
  --policy offline \
  --landlock-compatibility best_effort
```

Create an untracked local config and set only its OpenShell
`require_hard_landlock` field to `false`:

```bash
cp configs/config_openshell.yml configs/config_openshell.local.yml
```

Then validate the gateway and start AI-Q:

```bash
./scripts/start_openshell_gateway.sh \
  --gateway-name openshell \
  --image-name aiq-openshell-demo:latest \
  --policy-file configs/openshell/generated/aiq-openshell-policy.yaml

AIQ_OPENSHELL_POLICY_FILE=configs/openshell/generated/aiq-openshell-policy.yaml \
AIQ_OPENSHELL_IMAGE=aiq-openshell-demo:latest \
./scripts/start_e2e.sh --config_file configs/config_openshell.local.yml
```

Run the same mechanics through the convenience wrapper with the explicit demo
opt-in:

```bash
.venv/bin/python scripts/smoke_openshell_isolation.py \
  --gateway openshell \
  --policy configs/openshell/generated/aiq-openshell-policy.yaml \
  --image aiq-openshell-demo:latest \
  --allow-best-effort-landlock
```

A passing macOS run is useful local evidence, but it does not satisfy Linux
production acceptance.

## Existing Remote Gateway

The remote gateway must already be registered over HTTPS with mTLS, OIDC, or
trusted edge authentication. The launcher validates the registration and refuses
to substitute a local gateway if the remote service is unavailable:

```bash
./scripts/start_openshell_gateway.sh \
  --gateway-name enterprise \
  --reuse-existing \
  --image-name aiq-openshell-demo:latest \
  --policy-file configs/openshell/generated/aiq-openshell-policy.yaml
```

The disposable strict capability probe is mandatory. After it passes,
export `AIQ_OPENSHELL_GATEWAY_NAME=enterprise` and run the live suite. Never fall
back to a plaintext registration, insecure TLS, or a local raw gateway.

## Shared Debug Attachment

Create a named shared sandbox only when debugging requires it:

```bash
./scripts/start_openshell_gateway.sh \
  --gateway-name openshell \
  --create-shared-debug-sandbox \
  --sandbox-name aiq-openshell-demo
```

Attachment also requires `allow_shared_sandbox: true` and an explicit
`existing_sandbox_name` in a local AI-Q config. When a policy is supplied, AI-Q
requires strict effective-policy attestation and rejects `attest: false`. Without
a policy, the attachment still requires READY/loaded-version checks and emits
`assurance=reduced`. The attaching job never owns or deletes the shared sandbox;
the operator or test fixture that created it remains responsible.

## Expected Runtime Behavior

The human-readable contract is:

- One running deep-research job creates one physical OpenShell sandbox.
- Two concurrent jobs create two distinct sandbox names and physical IDs.
- Active jobs are discoverable with `--selector aiq=deep-research`, with a
  distinct `aiq-job-id` gateway label for each job.
- Attestation succeeds before the execution adapter is exposed.
- Success, command failure, timeout, and cancellation delete owned sandboxes.
- Cancelling one job does not delete or replace another job's sandbox.
- `sandbox.attestation` reports sanitized status, policy version, hash, source,
  assurance, and reason code.
- `sandbox.cleanup` reports sanitized completion and stable failure reason codes.
- Credentials, policy contents, SDK response bodies, and exception messages are
  not emitted in lifecycle events or failure logs.

The final job state is separate from physical cleanup: verify the cleanup event
and absence from the gateway rather than assuming a terminal job status deleted
the resource.

## Acceptance Tests

The canonical acceptance entry point is pytest:

```bash
AIQ_OPENSHELL_LIVE_TESTS=1 \
.venv/bin/python -m pytest -m integration -vv \
  tests/aiq_agent/agents/deep_researcher/sandbox/test_openshell_live.py
```

The suite contains three independently reported tests:

- `test_live_per_job_isolation_attestation_and_cancellation` proves distinct
  sandboxes, authoritative source/content/hash/revision attestation, isolated
  cancellation, selector membership, continued execution, and terminal deletion.
- `test_live_failure_cleanup_and_log_redaction` proves cleanup after a deterministic
  failed command and verifies that a credential-shaped exception canary reaches
  neither logs nor events.
- `test_live_shared_policy_mismatch_is_rejected` proves that a structurally
  different claimed policy cannot attach successfully, while the directly owned
  shared sandbox remains usable until fixture teardown.

Every fixture registers resources immediately, tears them down in reverse order,
and verifies deletion through the gateway. A teardown failure fails the test even
when the test body also failed. Without `AIQ_OPENSHELL_LIVE_TESTS=1`, all three
tests are collected and skipped before optional OpenShell imports or gateway
connections.

`scripts/smoke_openshell_isolation.py` is a convenience wrapper only. It maps its
arguments to the environment contract, enables the live gate, and returns pytest's
exit code unchanged. Pytest owns every assertion and cleanup fixture.

Record the non-secret gateway version, policy path, image tag, platform, and
Landlock mode with acceptance results. Do not record registrations, environment
values, policy contents, response bodies, or credentials.

## Inspection and Troubleshooting

Inspect only registered resources and sanitized AI-Q lifecycle events:

```bash
.venv/bin/openshell status
.venv/bin/openshell gateway list -o json
.venv/bin/openshell sandbox list
.venv/bin/openshell sandbox list --selector aiq=deep-research -o json
```

The selected gateway registration must be HTTPS and report `mtls`, `oidc`, or
trusted edge authentication. During a job, the selector must show one owned
sandbox per active deep-research job. After termination, each owned name must be
absent from direct and selector listings. Use the sandbox name from sanitized `sandbox.attestation` or
`sandbox.cleanup` events; do not expose full SDK payloads to logs.

| Failure | Safe action |
|---|---|
| Generated policy or image is missing | Run `setup_openshell.sh` and reuse the exact paths it prints. |
| CLI, SDK, and gateway versions differ | Install the same exact OpenShell release for all three surfaces and rerun the strict launcher. |
| `request_labels_unsupported` | The installed Python SDK cannot persist gateway labels required for AI-Q ownership and selectors; install a supported release. |
| `policy_status_inconsistent` | The effective policy matches but its revision never became `LOADED`. This is an OpenShell lifecycle failure, not a Landlock-mode mismatch. Do not disable attestation. |
| `policy_content_mismatch` | Regenerate the policy with `setup_openshell.sh`, or add the required OpenShell proxy filesystem baseline (including read-only `/proc`) to a custom policy. Do not weaken exact attestation. |
| `selector_mismatch` | The probe was not discoverable through gateway metadata. Do not rely on Docker/template labels as a substitute. |
| Registration is plaintext or unauthenticated | Register an HTTPS gateway with mTLS, OIDC, or trusted edge authentication. Do not bypass the launcher check. |
| Docker daemon is unavailable | Start the operator-owned Docker service and rerun provisioning/probe. |
| Podman is selected | Follow upstream OpenShell guidance; do not report the path as PR #298-certified. |
| Landlock policy/config mismatch | Pair `hard_requirement` with `require_hard_landlock: true`, or use both demo settings explicitly. |
| Policy is broader than `network_allow` | Remove the endpoint or add its exact normalized hostname to the declared upper bound. Do not add CIDR exceptions. |
| Sandbox never becomes Ready | Inspect the owning gateway/runtime service, image availability, and sanitized sandbox status; do not dump SDK bodies. |
| Probe or job deletion cannot be verified | Treat acceptance as failed, identify the exact sandbox, and retry explicit deletion through the registered gateway. |
| macOS reports Bash 3.2 incompatibility | Install Bash 5 and invoke the setup script with its absolute path. |

For a named sandbox that this operator owns, use explicit cleanup and verify its
absence:

```bash
.venv/bin/openshell sandbox delete <identified-sandbox-name>
.venv/bin/openshell sandbox list
```

Manage a packaged gateway only through its owner:

```bash
brew services restart openshell
systemctl --user restart openshell-gateway
```

Never use broad `pkill`, launch the raw gateway binary, enable insecure TLS, or
perform destructive cleanup without first identifying the owned resource. Do not
stop an externally managed gateway from AI-Q shutdown logic.
