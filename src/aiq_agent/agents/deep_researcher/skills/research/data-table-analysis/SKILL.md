---
name: data-table-analysis
description: >
  Use this skill for converting researched facts or user-provided data into structured tables by writing code, then running Python/pandas calculations in the job-scoped sandbox. This skill is for numeric normalization, tabular analysis, rankings, growth rates, summary statistics, CSV/JSON generation, and markdown tables. Triggers: "compute table", "calculate growth", "normalize values", "extract figures", "rank companies", "QoQ", "YoY", "CAGR", "summary statistics", "CSV", "JSON", "markdown table", "standardize quarters", "standardize currencies", "compare over time". Outputs: Markdown tables, CSV text, JSON records, summary statistics, rankings, and data-quality notes.
---

# Data Table Analysis Skill

Generate accurate, source-grounded tables and computed quantitative summaries using Python/pandas.
This skill is the canonical quantitative evidence producer: it returns runtime-validated CSV
text and a Markdown table produced from the same successful DataFrame, plus calculations,
provenance, and caveats in `ResearchNotes`. The runtime does not compare the Markdown table
with the CSV, and this skill never publishes durable files.

## Required Execution Standard

To ensure the calculation is reproducible and useful, you MUST:
1. **Structure Inputs:** Convert facts from research notes or the user request into explicit rows before running pandas.
2. **Preserve Provenance:** Keep source URLs, filing names, or note references in the input table when available.
3. **Normalize Units:** Convert currencies, magnitudes, periods, and date labels into consistent fields before comparing values.
4. **Compute Deterministically:** Call the `execute` tool to run Python/pandas for arithmetic, rankings, growth rates, aggregates, and formatting. Do not hand-compute these values in prose.
5. **Create One Canonical Dataset per Final DataFrame:** Generate `csv_text` and
   `markdown_table` from the same successfully analyzed DataFrame and return them in
   `ResearchNotes.quantitative_datasets`, up to four independently useful datasets per note.
   `csv_text` is the sole canonical serialization for downstream durable publication.
6. **Report Caveats:** Include assumptions, missing values, restatements, estimated figures, or non-comparable metrics in the output notes.
7. **Recommend, Do Not Render:** When a visualization would materially improve the final
   report, add a concise recommendation to `ResearchNotes.narrative_notes` containing the
   canonical `dataset_id`, chart type, axes or series, and rationale. Do not render a chart.
   Never call `write_file` or `edit_file`, write under `sandbox_artifact_dir`, create
   `manifest.json`, or emit an `artifact://` reference. The writer alone decides whether to
   render and publish final charts from the validated canonical dataset.

## QuantitativeDataset Contract

For each independently useful table, append one `QuantitativeDataset` to
`ResearchNotes.quantitative_datasets`:

- `dataset_id`: lowercase stable identifier matching `[a-z0-9][a-z0-9_-]{0,63}`.
- `title`: concise, non-empty human-readable title.
- `csv_text`: exact UTF-8 CSV returned by the successful pandas run.
- `markdown_table`: report-ready table produced from the same DataFrame and columns.
- `summary`: interpretation of the calculations, rankings, or statistics already computed.
- `source_ids`: unique IDs that exist in the enclosing `ResearchNotes.sources` list.
- `caveats`: assumptions, gaps, estimates, restatements, and comparability limitations.

Do not calculate, guess, or return `csv_sha256`; the runtime validates `csv_text` and computes
the trusted digest after the structured response is accepted.

Keep each dataset report-sized: at least one data row, no more than 5,000 rows or 128 columns,
and no more than 64 KiB of UTF-8 CSV. A note may contain at most four datasets. Headers must be
non-empty, unique, and free of surrounding whitespace. Use valid quoted CSV with consistent row
widths and no blank records, BOM, NUL, DEL, or other control characters beyond tab/newlines. The
combined UTF-8 CSV size across every dataset in one note must not exceed 128 KiB.

## Data honesty

The table is the trustworthy, gap-aware deliverable that any downstream chart depends on,
so it must be honest about what is and isn't known:

1. **Per-cell status:** treat each value as reported, estimate, or not disclosed. Leave
   undisclosed cells explicitly empty (e.g. `—`); never fabricate or infer a number to
   fill a gap, and never carry a prior period forward to hide one.
2. **One metric definition:** compare like with like. If sources use different definitions
   (e.g. "cash paid for property and equipment" vs "capital expenditures including finance
   leases"), keep them in separate rows/columns or pick one and label it - do not silently
   blend definitions into a single series.
3. **Surface coverage:** in the notes, state how many cells are reported vs estimated vs
   undisclosed, so the reader (and any chart built from this table) can judge how much
   weight it bears.

## Execution Flow

1. Gather candidate facts from researcher outputs, user-provided data, or source excerpts.

2. Create a normalized input table with one row per comparable observation. Prefer explicit CSV or JSON records embedded in the Python script. If the source rows are in `/shared/...`, call `read_file` first and embed the returned content in the script, or write a sandbox-local input file under your sandbox working directory (`sandbox_workdir`; e.g. `/sandbox` on OpenShell or `/workspace` on Modal). Sandbox code cannot open `/shared/...` directly.

3. Call the `execute` tool with a Python command or script that:
   - imports pandas,
   - builds a DataFrame from the normalized rows,
   - validates data types,
   - standardizes units and period labels,
   - computes the requested metrics,
   - prints `csv_text`, `markdown_table`, summary statistics, and data-quality notes as text,
   - creates `csv_text` and `markdown_table` from the same final DataFrame and column order,
   - uses your sandbox working directory (`sandbox_workdir`) for any sandbox-local input or output files, and writes any script file at the job-unique path your instructions specify (the `<job_id>_<name>.py` form) so a shared sandbox never reuses a stale leftover from another job.
   - does not read from or write to `/shared/...` inside the sandbox process.

4. Inspect the `execute` output. If the code fails, fix the code and call `execute` again. Do
   not continue with hand-computed fallback tables unless the sandbox or pandas is unavailable.
   Do not start a separate repair loop after the structured-output validator rejects a dataset;
   return the supported narrative findings and let the bounded response correction handle it.

5. Return the final outputs from the successful `execute` run as a
   `ResearchNotes.quantitative_datasets` entry. Findings may explain the numbers, but they are
   not a substitute for the canonical dataset. Do not call `write_file`/`edit_file`;
   `run_research_batch` persists the structured notes under `/shared/` automatically.

6. In the response or report, cite the original sources for the input figures. Computed columns should be clearly labeled as calculations.

**Required Tool Use:** For tasks that request calculated tables, growth rates, rankings,
summary statistics, normalization, CSV, or JSON, this skill requires at least one `execute`
call that runs Python/pandas before returning the canonical dataset.

---

## Input Normalization Guidelines

| Input Issue | Required Handling |
|-------------|-------------------|
| Mixed magnitudes | Convert millions/billions/trillions into one numeric unit, such as USD billions. |
| Mixed currencies | Convert to one currency only when an exchange-rate source is available; otherwise keep currencies separate and flag the limitation. |
| Fiscal vs. calendar quarters | Preserve the reported fiscal period and add a normalized sortable period field when possible. |
| Company-specific definitions | Keep metric names explicit, such as "capital expenditures", "PP&E additions", or "cash capex". |
| Missing values | Use null/blank values, not zero, unless the source explicitly reports zero. |
| Approximate figures | Mark estimates with an `is_estimate` column or a notes field. |
| Conflicting figures | Keep both rows with source notes unless one source is clearly authoritative. |

## Calculation Specifications

| Calculation | Formula / Logic Guide |
|-------------|------------------------|
| **QoQ Growth** | `(current_value / prior_quarter_value - 1) * 100` within each entity and metric. |
| **YoY Growth** | `(current_value / value_four_quarters_ago - 1) * 100` within each entity and metric. |
| **CAGR** | `(ending_value / beginning_value) ** (1 / years) - 1`, only when periods are comparable. |
| **Ranking** | Sort by the normalized numeric value and include rank ties deterministically. |
| **Share of Total** | `value / group_total * 100`, computed within the relevant period or category. |
| **Summary Stats** | Include count, mean, median, min, max, and missing-value count when useful. |

## Output Formats

Return one canonical `QuantitativeDataset` per independently useful final DataFrame, up to
four per `ResearchNotes`, for synthesis:

- `markdown_table` for report inclusion.
- exact `csv_text` for writer-owned durable publication.
- `summary`, `source_ids`, and `caveats` for interpretation and provenance.

Additional JSON may appear in findings when useful, but it must not become a second canonical
row representation that can diverge from `csv_text`.

**Note:** Label each output clearly (e.g. an "AI capex 8Q growth" table) so the writer can use it.

---

## Example Code Templates

### A. Normalize Rows and Compute QoQ/YoY

Use this when researched figures need growth calculations.

```python
import pandas as pd

rows = [
    {
        "company": "ExampleCo",
        "period": "FY2025-Q1",
        "period_index": 202501,
        "metric": "capital_expenditures",
        "value_usd_billions": 12.4,
        "source": "https://example.com/filing",
        "notes": "",
    },
]

df = pd.DataFrame(rows)
df = df.sort_values(["company", "metric", "period_index"])
df["qoq_growth_pct"] = (
    df.groupby(["company", "metric"])["value_usd_billions"].pct_change(1) * 100
)
df["yoy_growth_pct"] = (
    df.groupby(["company", "metric"])["value_usd_billions"].pct_change(4) * 100
)

display_cols = [
    "company",
    "period",
    "metric",
    "value_usd_billions",
    "qoq_growth_pct",
    "yoy_growth_pct",
    "source",
    "notes",
]
final_df = df[display_cols]
markdown_table = final_df.to_markdown(index=False, floatfmt=".1f")
csv_text = final_df.to_csv(index=False, lineterminator="\n")
```

### B. Rank Entities by Latest Comparable Period

Use this for company rankings or top-N comparisons.

```python
import pandas as pd

df = pd.DataFrame(rows)
latest_period = df["period_index"].max()
latest = df[df["period_index"] == latest_period].copy()
latest = latest.sort_values(
    ["value_usd_billions", "company"],
    ascending=[False, True],
)
latest["rank"] = range(1, len(latest) + 1)

ranking_table = latest[
    ["rank", "company", "period", "value_usd_billions", "source", "notes"]
].to_markdown(index=False, floatfmt=".1f")
```

### C. Generate Data-Quality Notes

Use this to make limitations explicit before synthesis.

```python
import pandas as pd

df = pd.DataFrame(rows)
notes = []

missing = df["value_usd_billions"].isna().sum()
if missing:
    notes.append(f"{missing} rows have missing normalized values.")

if "is_estimate" in df.columns and df["is_estimate"].fillna(False).any():
    notes.append("Some values are estimates and should be labeled as such.")

if df.duplicated(["company", "period", "metric"]).any():
    notes.append("Some company-period-metric combinations have multiple source rows.")

data_quality_notes = "\n".join(f"- {note}" for note in notes) or "- No major data-quality issues identified."
```

---

## Troubleshooting in the Sandbox

- Missing pandas: If `import pandas` fails, report that the sandbox image needs `pandas` installed. Do not hand-compute large tables in prose.
- Sorting Periods: Do not sort fiscal quarters alphabetically. Create a numeric `period_index` or date column.
- Percent Formatting: Keep computed growth as numeric values in CSV/JSON; format percentages only in markdown tables.
- Zero Division: If a prior period is zero or missing, leave growth blank/null and explain the limitation.
---
