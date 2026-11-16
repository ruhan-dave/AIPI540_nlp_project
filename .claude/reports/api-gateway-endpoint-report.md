# Implementation Report — API Gateway-Generated API Endpoint

**Plan**: `.claude/plans/aws-deployment.md` § *Planned Change: API Gateway-Generated API Endpoint*
**Branch**: `feature/aws-deployment` **Status**: COMPLETE (no infrastructure applied — see *Issues encountered*)

**Commit**: `bad28e0` — *feat: serve the API through an API Gateway HTTP API endpoint*
**Pull request**: https://github.com/ruhan-dave/AIPI540_nlp_project/pull/3 (base `feature/aws-deployment` ← head `feature/api-gateway-endpoint`)

## Summary

The public boundary is now an API Gateway HTTP API's generated `execute-api` HTTPS endpoint instead of a
public ALB with a custom domain. The load balancer became `internal` with a private HTTP listener that only
the VPC link's security group may reach, and Terraform now creates the HTTP API, the VPC link, a
`HTTP_PROXY`/`PRIVATE` integration, `ANY /` + `ANY /{proxy+}` routes and a `$default` stage — so there is no
domain to buy and no ACM certificate to renew.

The two limits the HTTP API imposes are resolved in the service rather than worked around. The 10 MB
payload cap (which cannot be raised) is handled by the plan's preferred option: a presigned S3 upload flow
that keeps the 50 MiB contract, while the multipart route is capped at the gateway-compatible 8 MiB. The 30
second integration timeout is handled by a deadline on the synchronous prediction path: the provider call
gets a 20-second budget and the split runs under a 25-second deadline, and an overrun falls back to the
whole review instead of letting the gateway answer 504.

## Tasks completed

### Terraform — the network boundary

- `infra/terraform/main.tf` (UPDATE) — the ALB is `internal = true` with an HTTP/80 listener replacing the
  public HTTPS/443 listener; the ALB security group accepts HTTP only from the new VPC link security group
  (the `0.0.0.0/0:443` rule is gone); a new `aws_security_group.vpc_link` reaches the ALB over port 80;
  `aws_apigatewayv2_api.http` (generated endpoint enabled), `aws_apigatewayv2_vpc_link.api`,
  `aws_apigatewayv2_integration.api` (`HTTP_PROXY` / `PRIVATE` / `VPC_LINK`, method `ANY`, payload format
  1.0, 29 s timeout), `aws_apigatewayv2_route.root` (`ANY /`), `aws_apigatewayv2_route.proxy`
  (`ANY /{proxy+}`) and `aws_apigatewayv2_stage.default` (`$default`, auto-deploy); the ECS service depends
  on the new listener; the task definitions carry the new environment values.
- `infra/terraform/variables.tf` (UPDATE) — `api_domain_name` and `acm_certificate_arn` are no longer
  inputs (they are kept as validated no-ops so a stale `.tfvars` is rejected loudly rather than silently
  ignored); added `alb_subnet_ids`, `vpc_link_subnet_ids`, `vpc_link_security_group_ids`,
  `max_multipart_upload_bytes`, `upload_url_ttl_seconds`, `predict_timeout_seconds`,
  `predict_deadline_seconds`, `worker_split_timeout_seconds` and `http_api_integration_timeout_ms`, with a
  validation that the prediction deadline is shorter than the integration timeout.
- `infra/terraform/outputs.tf` (UPDATE) — `api_url` is now the API Gateway `api_endpoint`; added
  `http_api_id` and `vpc_link_id`; `alb_dns_name` is documented as internal-only.
- `infra/terraform/terraform.tfvars.example` (UPDATE) — the domain and certificate entries are gone; the
  new subnet inputs are documented.

### Service — the two limits the HTTP API imposes

- `service/settings.py` (UPDATE) — `max_multipart_upload_bytes` (8 MiB), `upload_url_ttl_seconds`,
  `predict_timeout_seconds`, `predict_deadline_seconds`, `worker_split_timeout_seconds`, each validated at
  start-up.
- `service/schemas.py` (UPDATE) — `UploadRequest`, `UploadTicket` and `FinalizeRequest`.
- `service/batch_io.py` (UPDATE) — `presign_object_put` (a key-scoped presigned PUT) and `object_size`
  (the object's size, or `None` when it does not exist).
- `service/server.py` (UPDATE) — `POST /jobs/uploads` issues the presigned URL; `POST /jobs/uploads/
  {upload_id}/finalize` verifies the object, validates its size and CSV shape, creates the job and enqueues
  it; the upload id becomes the job id, so finalizing one upload twice is refused with `409`; the multipart
  route is capped at the gateway-compatible limit; `/predict` runs the split under a hard deadline.
- `service/inference.py` (UPDATE) — `Splitter` takes an explicit `timeout` and passes it to the provider;
  it exposes `timeout` as a property.
- `service/worker.py` (UPDATE) — the worker builds its splitter with the longer, gateway-free budget.
- `scripts/llm_api.py`, `scripts/llm_providers.py` (UPDATE) — `llm_generate` takes a timeout; OpenRouter's
  `timeout` is now the **total** budget for the call, so the one structured-output retry only runs when the
  budget still leaves room for it; Gemini passes the budget to the SDK as `HttpOptions(timeout=...)`.

### Frontend

- `app.py` (UPDATE) — `upload_batch_csv()` asks for a presigned URL, PUTs the CSV straight to S3, then
  finalizes the upload; `submit_batch_run()` uses it.

### Plan and docs

- `.claude/plans/aws-deployment.md` (UPDATE) — the resource table, the "do not provision" list, the
  Terraform input/output list, the Phase 7 and Phase 8 steps, the validation checklist, the teardown
  inventory and the assumptions are reconciled with the new topology; the CSV-upload and prediction-timeout
  subsections record their resolution.
- `.claude/plans/integrate-bert-model-into-app.md` (UPDATE) — the companion plan's batch-client contract now
  describes the presigned upload flow.
- `README.md`, `CLAUDE.md`, `.claude/references/deployment.md` (UPDATE) — the architecture diagram, the API
  route table, the Terraform requirements, the Railway variables, the cost note and the teardown checklist.

## Tests added

80 tests, all passing (was 59; +21).

| File | New cases |
|---|---|
| `tests/test_app_client.py` (NEW) | 5 — the three-step request sequence (ticket → PUT → finalize), the bearer token on every API call, and the three failure paths (refused ticket, failed upload, rejected finalize). The HTTP calls are replaced, so what is under test is the client's contract with the service. |
| `tests/test_server.py` | 12 — the upload ticket's key and presigned URL, finalize creating the job (upload id == job id, one SQS pointer), finalize without an object (`409`), finalize rejecting a CSV without `content` (`422`), finalize over the byte limit (`413`), a non-UUID upload id (`422`), a traversal attempt refused, finalizing twice (`409`), a failed enqueue marked `FAILED`, the multipart byte limit, the documented default limits, `/predict` falling back to the whole review when the split overruns the deadline, and the splitter passing its budget to the provider. |
| `tests/test_batch_io.py` | 3 — `presign_object_put` scoping the URL to one key, `object_size` reporting a size, and `object_size` reporting a missing object as `None`. |
| `tests/test_worker.py` | 1 — the worker's split budget is its own, longer one. |

## Validation results

| Check | Result |
|---|---|
| `pytest tests/ -q` | **80 passed** |
| `pyflakes service/ tests/ scripts/llm_api.py scripts/llm_providers.py` | clean |
| `pyflakes app.py` | 2 warnings, both **pre-existing** (`WORD2VEC_DOWNLOAD_URL`, `calculate_score_nb` imported but unused — unchanged by this work) |
| `terraform -chdir=infra/bootstrap fmt -check -recursive` | pass |
| `terraform -chdir=infra/terraform fmt -check -recursive` | pass |
| `terraform -chdir=infra/bootstrap validate` | pass |
| `terraform -chdir=infra/terraform validate` | pass |
| `terraform plan` (application, read-only, against the real account) | **42 to add, 0 to change, 0 to destroy**; the resource list is exactly the intended set — the HTTP API, its integration, both routes, the stage and the VPC link are in; there is **no** NAT gateway, VPC, VPC endpoint, Cloud Map, WAF, ElastiCache, autoscaling or Container Insights |
| The Streamlit app still boots (`AppTest`) | pass — no exceptions, no `st.error`, all five subheaders render |

**Not run:** `terraform apply`, `terraform destroy`, the AWS push, the Railway deploy, and the live
end-to-end job through a deployed stack. The first two need explicit approval; the rest need AWS and Railway
credentials and a deployed stack.

## Deviations from the plan

1. **The multipart route was kept, with a second, smaller limit.** The plan offered two ways to resolve the
   10 MB payload cap: the presigned upload flow (its stated preference) or lowering the limit. Both are
   implemented. `MAX_UPLOAD_BYTES` (50 MiB) governs the presigned path and the job data as a whole;
   `MAX_MULTIPART_UPLOAD_BYTES` (8 MiB) governs `POST /jobs`, because that route carries the CSV through the
   gateway. Keeping the multipart route means a direct (non-gateway) deployment still accepts a large upload
   on one request, and the frontend has a single path that works in both environments.
2. **The upload id is the job id.** The plan says the API issues the URL for "a server-chosen input key" and
   the client then asks the API to "create the job". Deriving the job id from the upload id makes the finalize
   step idempotent by construction — a second finalize fails the job record's `attribute_not_exists` condition
   and is refused with `409` — without a second table or a second id.
3. **The upload id is validated against a UUID pattern.** The plan did not say this, but the id becomes part
   of an S3 key, so `service/server.py` rejects anything that is not a 32-character hex. A traversal attempt
   cannot reach another object.
4. **The provider's `timeout` became a total budget rather than a per-request one.**
   `scripts/llm_providers.openrouter_generate` may retry once, so a per-request timeout could have taken
   twice the intended budget. `timeout` is now the budget for the whole call, and the retry only runs if the
   budget still leaves room for it (`MIN_RETRY_BUDGET_SECONDS`).
5. **The worker got its own split budget** (`WORKER_SPLIT_TIMEOUT_SECONDS`, 60 s). The plan bounds the
   synchronous path; applying the same 20-second budget to the worker would have turned slow reviews into
   whole-review fallbacks across every batch job, for no benefit — the worker has no gateway in front of it.
6. **`api_domain_name` and `acm_certificate_arn` are deprecated no-op variables rather than deleted.** The
   plan says to remove them as required inputs. They are kept with a validation that rejects a non-null
   value, so an environment's existing `terraform.tfvars` fails with a message that says what to do instead
   of a confusing "value for undeclared variable" warning.
7. **`api_endpoint_enabled` is not in AWS provider 5.100.0.** The attribute was added later; the supported
   spelling is `disable_execute_api_endpoint = false`, which is what the stack uses. The endpoint is enabled
   by default either way.
8. **`overwrite.path` was left unset on the integration.** The `$default` stage's base path mapping is `/`,
   so no stage prefix reaches FastAPI and the default listener forwards the path unchanged. Setting
   `overwrite.path` explicitly was not needed and risked prefixing the path.

## Issues encountered

- **`aws_apigatewayv2_api` has no `api_endpoint_enabled` in provider 5.100.0** (deviation 7). Caught by
  `terraform validate`.
- **The plan runs against a local backend.** The application root's S3 backend config (`backend.hcl`) is
  gitignored and does not exist — the state bucket was never created, because `terraform apply` needs
  explicit approval. The plan was therefore produced from a temporary copy of the root with the backend
  block removed, against the real AWS account and real subnet IDs. `terraform validate` runs against the
  real root.
- **`coalesce` does not fall through on an empty list.** `coalesce([], var.public_subnet_ids)` returns the
  empty list, not the fallback, so the subnet defaults are written as an explicit length check.
- **Nothing was deployed.** The plan requires explicit approval for `terraform apply`, so no AWS or Railway
  resource was created. The Phase 8 checks against a live stack — the generated URL through `/healthz`, an
  authenticated `/predict`, a real CSV job, the DLQ test, the duplicate test and the heartbeat test — remain
  to be run after the stack is applied.
- **The 50 MiB finalize read is not measured.** Finalize reads the uploaded object back to validate it, and
  that read plus the CSV parse happens inside one gateway request. It is bounded by the 50 MiB limit and
  should take a few seconds inside the VPC, but it has not been timed against a real 50 MiB object.
