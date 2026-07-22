---
name: chart-generation
description: >
  Use this synthesis skill to publish a researcher-produced QuantitativeDataset as an
  API-retrievable CSV and, when warranted, render an optional PNG from that exact CSV.
  This skill serializes and visualizes canonical data; it never researches, normalizes,
  filters, aggregates, or derives values. Triggers: "downloadable CSV", "data export",
  "chart", "plot", "graph", "bar chart", "line chart", "visualize", "figure".
  Outputs: a durable CSV, an optional PNG, and manifest.json under aiq-artifacts/.
---

# Canonical Dataset Publication and Chart Generation

Publish validated quantitative evidence prepared by a researcher. The researcher owns
numeric correctness, normalization, calculations, provenance, and caveats. The writer owns
durable serialization, optional visualization, and final-report delivery.

The writer MUST NOT redo the analysis. In particular, do not add, drop, reorder, filter,
aggregate, normalize, round, interpolate, rank, or derive any row or value in `csv_text`.

## Required Input

Read every `ResearchNotes` file and select the `quantitative_datasets` entry that matches the
requested deliverable. A valid entry provides:

- `dataset_id`: safe base name for the published CSV.
- `title`: human-facing dataset title.
- `csv_text`: the sole canonical byte source, encoded as UTF-8.
- `markdown_table`: report-ready representation of the same analyzed DataFrame.
- `summary`, `source_ids`, and `caveats`: interpretation and provenance.
- `csv_sha256`: runtime-computed digest of the exact UTF-8 `csv_text`.

Treat `csv_sha256` as authoritative. Never invent or replace it. If the requested dataset is
missing, invalid, or lacks a runtime digest, do not reconstruct data from findings, prose, a
Markdown table, or model memory. Keep the text/table report and state that the downloadable
dataset could not be published.

## Report-First Safety Contract

Before writing any file under `sandbox_artifact_dir` or calling `execute`:

1. Complete the cited report in memory using the dataset's `markdown_table`, `summary`,
   `source_ids`, and `caveats`.
2. Write the complete baseline report to `/shared/output.md`.
3. If a chart was requested, include one visible fallback sentence where the figure belongs,
   explaining that the table remains authoritative if visualization publication fails.

The baseline is the guaranteed deliverable. Artifact or chart failure must never erase it or
restart research.

## CSV Publication (Always First)

For a requested downloadable dataset, including CSV-only requests:

1. Set the filename to `<dataset_id>.csv` under the exact absolute
   `sandbox_artifact_dir` supplied in the writer instructions.
2. Call `write_file` once with `csv_text` as the entire content. Do not add a code fence,
   explanatory prefix, BOM, or extra newline. Do not parse and reserialize it.
3. Write `manifest.json` with the CSV entry shown below. Copy the exact runtime digest into
   `expected_sha256`.
4. Treat the artifact-checkpoint result as authoritative. If it rejects the CSV, preserve the
   baseline report and stop publication for that dataset.

```json
{
  "version": 1,
  "artifacts": [
    {
      "path": "/absolute/per-job/aiq-artifacts/<dataset_id>.csv",
      "kind": "dataset",
      "title": "Canonical dataset title",
      "inline": false,
      "expected_sha256": "runtime-computed sha256 from ResearchNotes"
    }
  ]
}
```

A CSV-only request ends here. It requires no `execute` call.

## Optional Visualization

Generate a PNG only when the request calls for a figure and the canonical dataset is complete
enough to support one. The durable CSV must already be accepted before chart execution.

1. Write `make_chart.py` under the exact absolute `sandbox_workdir` shown in the writer
   instructions. The script must accept `sandbox_artifact_dir` as its only argument.
2. The script must read the already-published `<dataset_id>.csv` from that directory and
   verify its SHA-256 before plotting.
3. Select existing columns for axes and presentation. Do not mutate the DataFrame or calculate
   new data. If the exact rows cannot be plotted honestly without transformation, skip the PNG
   and retain the CSV plus report table.
4. Use matplotlib's non-interactive `Agg` backend, write `<dataset_id>.png`, and update
   `manifest.json` while retaining the canonical CSV entry and digest.
5. Run the exact command already rendered in the writer instructions. It passes the absolute
   per-job `make_chart.py` path followed by the absolute per-job artifact directory.

6. Use only the exact filename confirmed by the artifact checkpoint. Replace the single
   fallback sentence in `/shared/output.md` with the in-context image reference
   `![<caption>](artifact://<confirmed filename>)`.

Each `execute` starts a fresh shell. Use absolute paths; never use a standalone `cd`, literal
`<sandbox_workdir>` or `<sandbox_artifact_dir>` placeholders, or a stale script from another job.

## Chart Script Shape

The plotting script must follow this shape. Substitute only the canonical dataset metadata,
existing axis columns, chart type, labels, and presentation settings.

```python
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

if len(sys.argv) != 2:
    raise SystemExit("usage: make_chart.py ABSOLUTE_SANDBOX_ARTIFACT_DIR")
ARTIFACT_DIR = Path(sys.argv[1])
if not ARTIFACT_DIR.is_absolute():
    raise SystemExit("artifact directory must be an absolute path")

DATASET_ID = "canonical-dataset-id"
EXPECTED_SHA256 = "runtime-computed sha256 from ResearchNotes"
CSV_PATH = ARTIFACT_DIR / f"{DATASET_ID}.csv"
PNG_PATH = ARTIFACT_DIR / f"{DATASET_ID}.png"

csv_bytes = CSV_PATH.read_bytes()
if hashlib.sha256(csv_bytes).hexdigest() != EXPECTED_SHA256:
    raise SystemExit("canonical dataset digest mismatch")

df = pd.read_csv(CSV_PATH)
# Select existing columns only. Do not filter, sort, aggregate, normalize, or derive values.
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(df["existing_label_column"], df["existing_numeric_column"])
ax.set_xlabel("Label")
ax.set_ylabel("Value (unit from canonical column)")
ax.set_title("Canonical dataset title")
fig.tight_layout()
fig.savefig(PNG_PATH, dpi=150)

manifest = {
    "version": 1,
    "artifacts": [
        {
            "path": str(CSV_PATH),
            "kind": "dataset",
            "title": "Canonical dataset title",
            "inline": False,
            "expected_sha256": EXPECTED_SHA256,
        },
        {
            "path": str(PNG_PATH),
            "kind": "image",
            "title": "Canonical dataset title",
            "caption": "Presentation of the canonical dataset.",
            "inline": True,
            "source_files": [str(CSV_PATH)],
        },
    ],
}
with (ARTIFACT_DIR / "manifest.json").open("w", encoding="utf-8") as handle:
    json.dump(manifest, handle)
```

## Failure and Retry Rules

- The runtime controls the writer's physical `execute` budget. Never try to bypass a blocked
  execution by changing tool names, commands, scripts, or paths.
- After an execution error, correct only presentation code. Never change the CSV or analysis.
- On `writer_execute_budget_exhausted`, stop immediately and finish with the existing report,
  Markdown table, and accepted CSV.
- Do not call `read_file` on a PNG; rely on the checkpoint response.
- If pandas or matplotlib is unavailable, retain the CSV and baseline report and state the
  visualization limitation.
- Reference images only as `artifact://<filename>`; never expose sandbox paths or base64 data.
