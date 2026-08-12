# NeMo Retriever Knowledge Layer Backend

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

## Compatibility

This adapter targets the collection-management contract validated with NeMo
Retriever commit `edfed55da` plus the TXT/HTML tokenizer landing patch, or a
merged successor containing both. The service must expose the job-scoped
ingestion routes under `/v1/ingest/job`. A 404 or 410 from those routes is
reported as an actionable API-version mismatch.

TXT, HTML, PDF, Office, image, audio, and video support is determined by the
deployed NRL service and its configured dependencies. AIQ does not override
NRL extraction, OCR, tokenization, embedding, or indexing settings. Use a
service image containing the tokenizer fix before validating TXT or HTML.

## Configuration

Start from [`configs/config_web_nemo_retriever.yml`](../../../../configs/config_web_nemo_retriever.yml).
The required deployment setting is an explicit workspace scope:

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
| `nrl_max_retries` | `NRL_MAX_RETRIES` | `5` | Retries for transport errors, 429, and retryable 5xx |
| `nrl_max_concurrency` | `NRL_MAX_CONCURRENCY` | `8` | Maximum concurrent multipart uploads |
| `nrl_verify_ssl` | `NRL_VERIFY_SSL` | `true` | Verify gateway certificates |
| `nrl_ca_bundle` | `NRL_CA_BUNDLE` | unset | Optional enterprise CA bundle |
| `nrl_collection_ttl_hours` | `NRL_COLLECTION_TTL_HOURS` | `24` | Expiration applied to new NRL collections |

One token and scope are used per AIQ deployment. Per-user NRL credential
forwarding is not supported. Tokens are never logged and physical NRL storage
identifiers are removed from mapped metadata.

## Connecting to NRL

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

## Start AIQ

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

## Summary reconciliation

Document summaries live in AIQ's summary store, but NeMo Retriever owns document
lifetime, including server-side collection expiration. Collections therefore
disappear without AIQ observing a delete. On startup the ingestor reconciles the
two on a background thread: documents NRL serves gain a summary row, rows NRL no
longer backs are removed, and a collection's summaries are cleared once NRL
positively reports the collection as absent.

Reconciliation never blocks or fails startup, and a transport failure leaves the
store untouched rather than deleting summaries NRL could not confirm. The same
pass warms the document-ID-to-filename mapping that document deletes rely on.

## Validation

Run the contract tests without a service:

```bash
uv run pytest tests/knowledge_layer_tests/test_nemo_retriever_adapter.py
uv run python tests/knowledge_layer_tests/run_adapter_compliance.py \
  --backend nemo_retriever --quick \
  --config '{"base_url":"http://127.0.0.1:7670","scope":"workspace-123"}'
```

Run TXT, HTML, PDF, lifecycle, scope-isolation, durability, and latency checks
against a live landing:

```bash
AIQ_NRL_LIVE_TESTS=1 \
uv run pytest -s tests/knowledge_layer_tests/test_nemo_retriever_live.py
```

Set `NRL_SECOND_SCOPE` and optionally `NRL_SECOND_API_TOKEN` to include the
cross-scope 404 check. The live test reports direct NRL and AIQ-adapter p50,
p95, and p99 query latency.

## Troubleshooting

| Symptom | Meaning and action |
|---|---|
| Health or connection failure | Confirm `NRL_BASE_URL` reaches the public gateway from the AIQ process or pod. |
| HTTP 401 | Configure a valid `NRL_API_TOKEN`, or confirm the dev deployment explicitly allows unauthenticated access. |
| HTTP 403 | The token is not authorized for `NRL_SCOPE`; use the deployment's assigned workspace credential. |
| Collection/document 404 | Confirm the collection, stable document ID, and scope. Cross-scope resources intentionally return 404. |
| Job-route 404/410 | AIQ and the NRL service expose incompatible collection-management API versions; upgrade the NRL chart/image. |
| HTTP 409 | An idempotency key or manifest entry was reused with different content, or the resource already exists. |
| Empty retrieval | Confirm ingestion completed, the configured collection matches the upload collection, and NRL returned indexed documents. |
| TXT/HTML ingestion failure | Use the tokenizer landing patch or a merged successor, then rebuild/redeploy the NRL service image. |
| TLS failure | Keep verification enabled and configure `NRL_CA_BUNDLE` for an enterprise CA. |
