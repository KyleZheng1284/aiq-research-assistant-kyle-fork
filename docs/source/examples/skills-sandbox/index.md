<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Example: Deep Research Skills and Sandbox

This example shows how to run AI-Q deep research with DeepAgents skills and a provider-backed sandbox. The reference
profile uses Modal; AI-Q also includes an experimental OpenShell profile that creates a policy-bound sandbox per job.

Skills let a research agent discover task-specific instructions only when they are relevant. A skill can teach the agent a repeatable workflow, such as extracting numeric facts, normalizing a table, running calculations, and producing reusable text artifacts. The sandbox runs code-based work outside the AI-Q process. Both shipped profiles create a distinct sandbox for each job. OpenShell shared-sandbox attachment is a debug-only escape hatch and is not a production isolation mode.

For more background, refer to the LangChain DeepAgents docs:

- [Deep Agents overview](https://docs.langchain.com/oss/python/deepagents/overview)
- [DeepAgents skills](https://docs.langchain.com/oss/python/deepagents/skills)

## What This Example Enables

The example config enables:

- built-in DeepAgents skills from `src/aiq_agent/agents/deep_researcher/skills/`
- a fresh per-job Modal sandbox for Python execution
- Python packages useful for analysis, including `pandas`, `numpy`, `matplotlib`, and `pillow`
- virtual `/shared/` files for text artifacts that the orchestrator and subagents can read during the report workflow
- durable capture of supported files, including canonical CSV and chart publication, through the async job artifact API

The researcher-side `data-table-analysis` skill normalizes facts, performs calculations, and returns runtime-validated
canonical CSV plus a Markdown table that the skill contract requires to come from the same DataFrame. The writer-side
`chart-generation` skill publishes those exact CSV bytes and optionally renders a chart from the published CSV. The
runtime intentionally does not compare or regenerate the Markdown table, and the writer does not repeat the analysis.

**Models and report quality:** For clearer tables, stronger reasoning over numbers, and more reliable use of the data-table-analysis skill end-to-end, prefer **frontier-class models** for the orchestrator, planner, and researcher in your config ([Swapping models](../../customization/swapping-models.md)). Smaller or faster models may complete runs but often produce weaker structured outputs and more formatting mistakes in long reports.

## Prerequisites

Install and configure AI-Q as usual, then make sure these credentials are available to the process running AI-Q:

```bash
export NVIDIA_API_KEY="nvapi-..."              # pragma: allowlist secret
export TAVILY_API_KEY="tvly-..."               # pragma: allowlist secret
```

For sandbox execution, create a Modal account and configure Modal credentials. Modal uses a token ID and token secret:

```bash
export MODAL_TOKEN_ID="ak-..."                 # pragma: allowlist secret
export MODAL_TOKEN_SECRET="as-..."             # pragma: allowlist secret
```

You can also configure Modal locally with:

```bash
modal token set --token-id "$MODAL_TOKEN_ID" --token-secret "$MODAL_TOKEN_SECRET"
```

Refer to Modal's token configuration docs for details: [modal.config](https://modal.com/docs/reference/modal.config).

## Configuration

Use `configs/config_domain_routing_and_skills.yml`. The relevant section is:

```yaml
functions:
  deep_research_skills:
    _type: deep_research_skills
    agents:
      researcher-agent:
        - research
      writer-agent:
        - synthesis
    require_sandbox:
      - research
      - synthesis

  deep_research_sandbox:
    _type: deep_research_sandbox
    provider: modal
    app_name: aiq-deep-research
    image: python:3.13-slim
    packages:
      - matplotlib
      - numpy
      - pandas
      - pillow
    network: blocked
    artifact_capture:
      enabled: true
      max_file_bytes: 50000000
      allow_extensions: [.png, .jpg, .jpeg, .webp, .csv, .json, .md, .ipynb, .pdf]

  deep_research_agent:
    _type: deep_research_agent
    enable_citation_verification: false
    skills: deep_research_skills
    sandbox: deep_research_sandbox
```

AI-Q validates the public skill collection names (`research`, `synthesis`) and resolves them to DeepAgents source paths internally. When skills are configured, AI-Q mounts the configured built-in skill collections into the DeepAgents virtual filesystem. When the sandbox ref is present, DeepAgents `execute` calls run in the configured provider. Modal creates a fresh sandbox named for the job.

In the reference async API flow, artifact capture uses the job database configured by
`general.front_end.db_url` (`NAT_JOB_STORE_DB_URL`) for metadata. Without that job-scoped database URL, as in a direct
`nat run`, Modal execution still works but durable capture remains inactive. Artifact bytes use SQL BLOB storage in the
job database by default. For production, use S3-compatible object storage by setting `AIQ_ARTIFACT_BLOB_PROVIDER=s3`,
`AIQ_ARTIFACT_S3_BUCKET`, and the standard AWS credentials; set `AIQ_ARTIFACT_S3_ENDPOINT_URL` for MinIO or another
compatible service. See [Production Artifact Storage](../../deployment/production.md#artifact-storage) for all options.

```{important}
`chart-generation` moved from the `research` collection to `synthesis`. When upgrading a custom profile, assign
`researcher-agent: [research]` and `writer-agent: [synthesis]`, and include both collections under `require_sandbox`.
Remove `research` from the writer; durable publication is writer-owned, while analysis remains researcher-owned.
```

To evaluate OpenShell instead, use `configs/config_openshell.yml` after running
`scripts/openshell/setup_openshell.sh`. That experimental profile creates one policy-bound sandbox per job and deletes
owned sandboxes during terminal cleanup. Attaching to an existing shared sandbox requires the explicit debug settings
`allow_shared_sandbox: true` and `existing_sandbox_name`; that mode is not a multi-tenant isolation boundary.

## Run AI-Q

```bash
dotenv -f deploy/.env run .venv/bin/nat run \
  --config_file configs/config_domain_routing_and_skills.yml \
  --input "Compare the top 10 publicly traded semiconductor companies by 2024 revenue. Build a markdown table with revenue, YoY growth, market cap, and gross margin. Then rank them and compute summary statistics. Use the data analysis tool for all calculations."
```

This CLI path validates sandbox analysis and report generation, but it does not expose the async job artifact API. Start
the served API below and submit an async job to validate durable artifact capture, listing, and download.

For API or UI testing:

```bash
dotenv -f deploy/.env run .venv/bin/nat serve \
  --config_file configs/config_domain_routing_and_skills.yml \
  --host 0.0.0.0 \
  --port 8000
```

Then submit a deep research request through the AI-Q API or UI.

## Example Queries

Use queries that require researched numeric facts plus computed tabular analysis.

**Example prompt:**

```text
Compare the top 10 publicly traded semiconductor companies by 2024 revenue. Build a markdown table with revenue, YoY growth, market cap, and gross margin. Then rank them and compute summary statistics. Use the data analysis tool for all calculations.
```

Additional prompts that exercise the same pattern:

```text
Compare AI infrastructure capex for Microsoft, Google, Meta, and Amazon over the last 8 quarters. Include QoQ and YoY growth.
```

```text
Compare R&D spend across the top 10 semiconductor companies and compute R&D as a percent of revenue.
```

Expected behavior:

1. The planner identifies that a skill should be used for structured quantitative analysis.
2. Researchers gather source-grounded input figures.
3. The researcher reads `data-table-analysis`, calls `execute` for Python/pandas calculations, and returns canonical
   `csv_text`, a Markdown table, summary statistics, provenance, and caveats in validated `ResearchNotes`.
4. AI-Q registers the canonical dataset ID, artifact path, and SHA-256 digest over the exact UTF-8 CSV text.
5. The writer writes the complete `/shared/output.md` baseline, then reads `chart-generation` to publish the exact CSV.
6. CSV-only requests finish without chart execution. Chart requests render only from the published CSV. Operators can
   opt into a writer execution budget that stops repeated chart failures while preserving the report, table, and CSV.
7. The final report cites the original sources for input figures and labels computed columns as calculations.

## Skill Files

Built-in deep research skills live under:

```text
src/aiq_agent/agents/deep_researcher/skills/
```

Each skill belongs to the collection for its owning role and has a `SKILL.md` file:

```text
src/aiq_agent/agents/deep_researcher/skills/
|-- research/
|   `-- my-analysis-skill/
|       `-- SKILL.md
`-- synthesis/
    `-- my-publication-skill/
        `-- SKILL.md
```

At minimum, `SKILL.md` needs frontmatter with a stable `name` and a clear `description`:

```markdown
---
name: my-skill
description: >
  Use this skill when the research task requires a specific repeatable workflow.
  Include trigger phrases and expected outputs so the agent can decide when to
  read this skill.
---

# My Skill

## When to Use

Use this skill for ...

## Execution Flow

1. Gather the required inputs.
2. Use the appropriate tools.
3. Write reusable outputs to `/shared/...` when another agent or the final report needs them.
```

Skill descriptions matter because DeepAgents uses the frontmatter description to decide whether the skill applies before reading the full file. Keep descriptions specific, list representative trigger phrases, and explicitly name required tools such as `execute`, `read_file`, or `write_file` when the workflow depends on them.

## Adding More Skills

To add a built-in AI-Q deep research skill:

1. Choose the owning collection: `research` for evidence generation and analysis, or `synthesis` for report writing and
   durable publication. Create the skill directory under that collection.
2. Add a `SKILL.md` file with frontmatter and workflow instructions.
3. Put optional helper scripts, references, or templates inside the same skill directory.
4. Reference any helper files from `SKILL.md` so the agent knows when to read or run them.
5. Keep workflow instructions generic enough to handle variations of the task, but concrete enough to force required tool calls.
6. Run with `configs/config_domain_routing_and_skills.yml` and test a query that should trigger the new skill.

No config change is required only when the chosen collection is already assigned to the owning agent. AI-Q collects
available skill directories at runtime and exposes enabled collections to DeepAgents through an internal `/skills/`
source.

## Notes and Limitations

- The reference Modal and experimental OpenShell profiles create one sandbox per job. OpenShell shared attachment is an
  explicit debug escape hatch and is not a production or multi-tenant isolation mode.
- Plans, research notes, and the report use DeepAgents virtual paths under `/shared/`. Downloadable datasets and charts
  are writer-published under the configured sandbox artifact directory.
- `/shared/` is a virtual DeepAgents filesystem path. Use `ls`, `read_file`, `write_file`, and `edit_file` for `/shared/`; do not inspect `/shared/` with shell commands through `execute`.
- The sandbox is configured with `network: blocked`, so research should happen through AI-Q search tools, not from sandbox code.
- Durable sandbox artifact capture is opt-in (`artifact_capture.enabled: true`) and requires the async API's job-scoped
  artifact store. Direct `nat run` does not provide that store.
  Writer manifest writes and successful chart execution checkpoint declared files, and terminal paths perform one final
  best-effort scan. A busy cancellation skips that scan and preserves earlier checkpoints. Adding a sandbox alone does
  not guarantee that every generated file is persisted or embedded in the report.
- Canonical dataset manifests must carry the runtime-registered path and digest. Missing, moved, unregistered, or
  mismatched identities are rejected. Registered canonical paths remain digest-protected during the final directory
  scan even when no manifest was written.
- The researcher and writer execution budgets and `workflow_timeout_seconds` are disabled by default. Operators may
  enable them independently. Researcher attempts are counted per worker and include generic retries; writer attempts
  count canonical chart rendering and may finish without a PNG when exhausted. When the workflow timeout wins the
  terminal-state race, the job fails with `deep_research_workflow_timeout` and requests forced sandbox termination.
  Artifact recovery after timeout is best effort and is skipped when termination does not return within its bound.
