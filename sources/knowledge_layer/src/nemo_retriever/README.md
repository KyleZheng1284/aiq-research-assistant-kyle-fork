# NeMo Retriever Knowledge Layer Backends

AI-Q supports [NVIDIA NeMo Retriever](https://github.com/NVIDIA/NeMo-Retriever)
through two separately registered backends. `nemo_retriever` calls a deployed
Retriever REST API. The experimental `nemo_retriever_local` runs the pinned
Retriever library and LanceDB inside the AI-Q Python process. See the canonical
[knowledge-layer guide](../../../../docs/source/customization/knowledge-layer.md)
for shared Knowledge API behavior.

## Contents

- [Choose a backend](#choose-a-backend)
- [Embedded local backend](#embedded-local-backend-experimental)
  - [Choose an extraction profile](#choose-an-extraction-profile)
  - [Install and start](#install-and-start-the-local-backend)
  - [Local configuration](#local-configuration)
  - [Local operating model](#local-operating-model)
- [Deployed REST backend](#deployed-rest-backend)
  - [Service compatibility](#service-compatibility)
  - [Service configuration](#service-configuration)
  - [Connect to the service](#connect-to-the-service)
  - [Start AI-Q](#start-ai-q-with-the-service-backend)
  - [Service validation](#service-validation)
  - [Service troubleshooting](#service-troubleshooting)

## Choose a backend

Both backends implement the same AI-Q Knowledge API. Choose based on the
deployment and operational model; this is not a runtime mode switch.

| Consideration | Embedded library: `nemo_retriever_local` | Deployed service: `nemo_retriever` |
|---|---|---|
| Topology | AI-Q, Retriever, and LanceDB share one Python 3.12 process | AI-Q calls a separately operated Retriever gateway |
| Infrastructure | No Retriever or vector-database service | Retriever service deployed with Docker Compose or Helm/Kubernetes |
| Storage | Embedded LanceDB in `NRL_LOCAL_DATA_DIR` | Storage and durability are owned by the Retriever deployment |
| Inference | Extraction and embedding can call remote endpoints | Inference is configured and operated by the Retriever service |
| Concurrency | One AI-Q process per data directory | Supports shared service deployments and multiple AI-Q clients |
| Best fit | Laptop evaluation, local development, and zero-deployment workflows | Shared, scaled, or centrally operated environments |
| Support level | Experimental; Apple Silicon macOS, Windows x64, and Linux x64 targets | Existing deployed-service integration |
| Retriever version | Pinned to [`c80f4a5189ee10b98cbdb93e2f853ceb7b699c3b`](https://github.com/NVIDIA/NeMo-Retriever/commit/c80f4a5189ee10b98cbdb93e2f853ceb7b699c3b) | No Python dependency; requires the compatible service contract described below |

## Embedded local backend (experimental)

`nemo_retriever_local` delegates extraction, schemas, collection lifecycle,
storage, and retrieval to the unmodified Retriever library pinned in
[`environments/nemo_retriever_local/pyproject.toml`](../../../../environments/nemo_retriever_local/pyproject.toml).
AI-Q does not start a Retriever HTTP service, a vector-database server, Ray, or
containers for this path.

```text
AI-Q process -> NeMo Retriever operators -> embedded LanceDB
                       |
                       +-> configured extraction and embedding endpoints
```

Direct Knowledge API ingestion and retrieval do not require a generative LLM.
The complete research workflow still requires an OpenAI-compatible agent LLM.

### Choose an extraction profile

AI-Q exposes the two extraction profiles supported by the pinned Retriever
revision: `fast-text` and `auto`. Neither profile is universally preferred.
Select a profile based on corpus characteristics, retrieval requirements,
ingestion latency objectives, and inference usage.

| Profile | Extraction scope | Operational characteristics | Use when |
|---|---|---|---|
| `fast-text` | PDF/document text through PDFium, followed by text embedding | Optimized for ingestion efficiency and lower inference usage; skips images, tables, charts, page images, Page Elements, and OCR | The corpus is text-centric and ingestion throughput or cost is the primary objective |
| `auto` (default) | Text, images, tables, charts, page rendering, Page Elements, and OCR | Optimized for broad extraction coverage and performs additional inference stages; Table Structure remains off unless `NRL_TABLE_STRUCTURE_INVOKE_URL` is configured | Visual or structured content contributes to retrieval quality and the additional processing is acceptable |

Chunk count and ingestion time depend on document content, endpoint load, and
network conditions. Compare the profiles with representative documents before
selecting a production setting. Because the profile is recorded in collection
ownership metadata, use a new collection or data directory when changing it.

### Install and start the local backend

The isolated environment requires Python `>=3.12,<3.13` and keeps Retriever
out of the normal `knowledge-layer[all]` installation.

```bash
uv sync --project environments/nemo_retriever_local --frozen
```

Set `NRL_INFERENCE_API_KEY` in the ignored `deploy/.env` when using Retriever's
default hosted extraction and embedding endpoints. Keep `NVIDIA_API_KEY`
separate when the AI-Q agent LLM uses a different credential.

```bash
uv run --project environments/nemo_retriever_local --frozen \
  dotenv -f deploy/.env run \
  nat serve --config_file configs/config_web_nemo_retriever_local.yml --port 8000
```

This starts the backend at `http://localhost:8000`. The direct query route is
`POST /v1/knowledge/query`; the web UI creates session collection names
automatically.

### Local configuration

Start from
[`configs/config_web_nemo_retriever_local.yml`](../../../../configs/config_web_nemo_retriever_local.yml).
Retriever supplies its default hosted endpoint URLs, so configure individual
invoke URLs only when overriding those defaults. The table below lists every
NeMo Retriever environment variable exposed by the embedded AI-Q backend.

| YAML field | Environment variable | Default | Purpose |
|---|---|---:|---|
| `nrl_scope` | `NRL_SCOPE` | `local` | Logical collection scope |
| `nrl_local_data_dir` | `NRL_LOCAL_DATA_DIR` | `.aiq-data/nemo_retriever` | LanceDB, catalog, recovery, and staging root |
| `nrl_local_profile` | `NRL_LOCAL_PROFILE` | `auto` | `auto` or `fast-text` extraction profile |
| `nrl_inference_api_key` | `NRL_INFERENCE_API_KEY` | Retriever key fallback | Shared credential for extraction plus document/query embedding |
| `nrl_page_elements_invoke_url` | `NRL_PAGE_ELEMENTS_INVOKE_URL` | Retriever default | Optional Page Elements endpoint override |
| `nrl_ocr_invoke_url` | `NRL_OCR_INVOKE_URL` | Retriever default | Optional OCR endpoint override |
| `nrl_table_structure_invoke_url` | `NRL_TABLE_STRUCTURE_INVOKE_URL` | unset | Enables and overrides Table Structure when configured |
| `nrl_embed_invoke_url` | `NRL_EMBED_INVOKE_URL` | Retriever default | Optional embedding endpoint override |
| `nrl_embed_model_name` | `NRL_EMBED_MODEL_NAME` | Retriever default | Optional embedding model override |
| `nrl_embed_model_provider_prefix` | `NRL_EMBED_MODEL_PROVIDER_PREFIX` | Retriever default | Optional model provider-prefix override |
| `nrl_collection_ttl_hours` | `NRL_COLLECTION_TTL_HOURS` | `24` | Expiration applied to new collections |

If `NRL_INFERENCE_API_KEY` is unset, the pinned Retriever revision falls back
to `NVIDIA_API_KEY` and then `NGC_API_KEY`. All configured Retriever inference
endpoints use the resolved Retriever credential; AI-Q does not define a
separate key for each endpoint.

### Local operating model

- Collections, committed documents, chunks, and recovery markers survive an
  AI-Q restart; v1 job history is process-local.
- A process-lifetime file lock permits one AI-Q process per canonical data
  directory. Use separate directories for parallel processes.
- Interrupted work before a document write is not resumed as a job. Collection
  reconciliation repairs committed collection state at startup and periodically.
- Physical LanceDB table names and paths are not exposed through the Knowledge
  API. Retriever and LanceDB can still emit them in process logs, so treat logs
  as operationally sensitive.
- Initial targets are Apple Silicon macOS, Windows x64, and Linux x64 with
  remote inference. Python 3.13, Intel macOS, local GPU inference, and shared
  multi-process data directories are outside the initial support target.

## Deployed REST backend

The `nemo_retriever` backend connects AIQ to an independently deployed NeMo
Retriever service through its public REST API. AIQ never imports the
`nemo-retriever` Python package, opens LanceDB, selects physical tables, or
configures extraction workers.

```text
AIQ -> knowledge_layer.nemo_retriever -> NRL public gateway REST API
```

NeMo Retriever owns authentication, extraction, OCR, tokenization, embedding,
indexing, durable collections, and document lifecycle. AIQ owns only logical
collection/document input, job polling, retrieval queries, and conversion to
the AIQ universal schemas.

### Service compatibility

This adapter targets the collection-management contract validated with NeMo
Retriever commit `edfed55da` plus the TXT/HTML tokenizer landing patch, or a
merged successor containing both. The service must expose the job-scoped
ingestion routes under `/v1/ingest/job`. A 404 or 410 from version-probing job
creation or immediate document-upload routes is reported as an actionable
API-version mismatch. The same status from job polling remains an ordinary
missing or expired resource response.

TXT, HTML, PDF, Office, image, audio, and video support is determined by the
deployed NRL service and its configured dependencies. AIQ does not override
NRL extraction, OCR, tokenization, embedding, or indexing settings. Use a
service image containing the tokenizer fix before validating TXT or HTML.

### Service configuration

Start from [`configs/config_web_nemo_retriever.yml`](../../../../configs/config_web_nemo_retriever.yml).
The required deployment setting is an explicit workspace scope. The table
below lists every NeMo Retriever environment variable exposed by the service
adapter.

```bash
export NRL_BASE_URL=http://127.0.0.1:7670
export NRL_SCOPE=workspace-123
export NRL_API_TOKEN='replace-with-a-secret'  # omit only for an auth-disabled dev deployment
```

| YAML field | Environment variable | Default | Purpose |
|---|---|---:|---|
| `nrl_base_url` | `NRL_BASE_URL` | `http://127.0.0.1:7670` | Public NRL gateway, not a worker or VectorDB pod |
| `nrl_api_token` | `NRL_API_TOKEN` | unset | Optional bearer token stored as a secret |
| `nrl_scope` | `NRL_SCOPE` | required | Logical workspace scope sent on every request |
| `nrl_connect_timeout_s` | `NRL_CONNECT_TIMEOUT_S` | `30` | TCP/TLS connection timeout |
| `nrl_request_timeout_s` | `NRL_REQUEST_TIMEOUT_S` | `300` | Request timeout, including uploads |
| `nrl_max_retries` | `NRL_MAX_RETRIES` | `5` | Retries for reads and explicitly idempotent writes on transport errors, 429, and retryable 5xx |
| `nrl_max_concurrency` | `NRL_MAX_CONCURRENCY` | `8` | Maximum concurrent multipart uploads |
| `nrl_verify_ssl` | `NRL_VERIFY_SSL` | `true` | Verify gateway certificates |
| `nrl_ca_bundle` | `NRL_CA_BUNDLE` | unset | Optional enterprise CA bundle |
| `nrl_collection_ttl_hours` | `NRL_COLLECTION_TTL_HOURS` | `24` | Expiration applied to new NRL collections |

One token and scope are used per AIQ deployment. Per-user NRL credential
forwarding is not supported. Tokens are never logged and physical NRL storage
identifiers are removed from mapped metadata.

### Connect to the service

From the NRL deployment directory, start the stack and confirm its core
services are healthy:

```bash
docker compose up -d
docker compose ps
```

With the default Compose project name `nrl`, the core rows should include:

```text
nrl-embed-1      healthy
nrl-vectordb-1   healthy
nrl-gateway-1    healthy
```

Container-name prefixes can differ when the Compose project name is
overridden. For local Docker, publish the NRL gateway and use its host port:

```bash
curl -fsS http://127.0.0.1:7670/v1/health
export NRL_BASE_URL=http://127.0.0.1:7670
```

For an NRL deployment on a remote development host, create a tunnel from the
AIQ workstation:

```bash
ssh -N -L 7670:127.0.0.1:7670 user@nrl-host
export NRL_BASE_URL=http://127.0.0.1:7670
```

For Kubernetes, route AIQ to the NRL gateway Service or an enterprise ingress.
Do not route AIQ directly to realtime, batch, or VectorDB pods:

```bash
export NRL_BASE_URL=https://nrl-gateway.example.com
export NRL_CA_BUNDLE=/etc/ssl/certs/enterprise-ca.pem
```

### Start AI-Q with the service backend

```bash
./scripts/setup.sh
source .venv/bin/activate

uv run python .agents/skills/aiq-configure-workflow/scripts/validate_config.py \
  configs/config_web_nemo_retriever.yml

./scripts/start_e2e.sh \
  --config_file configs/config_web_nemo_retriever.yml \
  2>&1 | tee /private/tmp/aiq-nrl-e2e.log
```

The existing AIQ collection APIs create scoped NRL collections, upload files,
return the real NRL job ID, poll aggregate status, list stable document IDs,
and delete documents or collections. Upload retries reuse deterministic job
and manifest identifiers, preventing duplicate processing after a lost
response. `attempt_id` is retained only in diagnostic job/file metadata;
`document_id` remains the stable AIQ file identity.

Query results preserve the order returned by NeMo Retriever and expose its native
vector distance as `Chunk.distance`. Lower values are closer; AIQ does not
normalize or re-rank these backend-specific values.

### Service validation

Run the contract tests without a service:

```bash
uv run pytest tests/knowledge_layer_tests/test_nemo_retriever_adapter.py
```

### Service troubleshooting

| Symptom | Meaning and action |
|---|---|
| Health or connection failure | Confirm `NRL_BASE_URL` reaches the public gateway from the AIQ process or pod. |
| HTTP 401 | Configure a valid `NRL_API_TOKEN`, or confirm the dev deployment explicitly allows unauthenticated access. |
| HTTP 403 | The token is not authorized for `NRL_SCOPE`; use the deployment's assigned workspace credential. |
| Collection/document 404 | Confirm the collection, stable document ID, and scope. Cross-scope resources intentionally return 404. |
| Job creation/upload 404/410 | AIQ and the NRL service expose incompatible collection-management API versions; upgrade the NRL chart/image. A polling 404/410 instead means that job is missing or expired. |
| HTTP 409 | An idempotency key or manifest entry was reused with different content, or the resource already exists. |
| Empty retrieval | Confirm ingestion completed, the configured collection matches the upload collection, and NRL returned indexed documents. |
| TXT/HTML ingestion failure | Use the tokenizer landing patch or a merged successor, then rebuild/redeploy the NRL service image. |
| TLS failure | Keep verification enabled and configure `NRL_CA_BUNDLE` for an enterprise CA. |
