# Deployment: AWS backend + Railway frontend

The demo is two environments: **AWS hosts the API and the worker**, **Railway hosts the Streamlit
frontend**. AWS resources are Terraform-managed; the Railway service and its Volume are set up by
hand or with the Railway CLI.

This is the reference for the deployment described in `.claude/plans/aws-deployment.md`. The README
has the step-by-step commands; this file has the shape of the system and the checklists.

---

## The boundary

```
Railway (Streamlit)  ──HTTPS + Bearer $API_TOKEN──▶  API Gateway  ──private HTTP──▶  ALB  ──▶  ECS/Fargate API
   (execute-api URL)                                   (VPC link)       (internal)                            │
                                                                                                ECS/Fargate worker
                                                                                                      │
                                                          S3 (inputs + results), DynamoDB (jobs) ◀──────┘
```

- Railway calls the API over **public HTTPS**, through the API Gateway's generated `execute-api`
  endpoint. The endpoint comes with AWS's own certificate: no domain, no ACM certificate.
- The **ALB is internal**. It has no public address and no public ingress rule — only the VPC
  link's security group may reach it. The gateway-to-ALB hop is HTTP inside the VPC; the bearer
  token is encrypted on the leg that leaves the VPC.
- The **API and the worker share one image** and differ only in their command
  (`python -m service.server` vs `python -m service.worker`).
- The frontend **never loads BERT** in the hosted environment. `BERT_INFERENCE_MODE=remote` makes
  the app call `POST /predict` instead.
- Review text lives in S3, never in SQS and never in a log line.

---

## The AWS resource set

`infra/terraform` creates exactly this — nothing more:

| Resource | Terraform name | Notes |
|---|---|---|
| ECR | `aws_ecr_repository.api` | One demo-owned repository. `force_delete = true` so its image manifests go with it. |
| ECS cluster | `aws_ecs_cluster.main` | Container Insights explicitly off. |
| API service | `aws_ecs_service.api` | 1 vCPU / 2 GiB, public IP, in the ALB's target group. |
| Worker service | `aws_ecs_service.worker` | 1 vCPU / 2 GiB, public IP, **no inbound security-group rule**. |
| Task definitions | `aws_ecs_task_definition.{api,worker}` | Same image, separate definitions and commands. |
| SQS | `aws_sqs_queue.jobs` + `.dlq` | Visibility timeout 600 s, `maxReceiveCount = 3`, SSE, TLS-only policy. |
| DynamoDB | `aws_dynamodb_table.jobs` | On-demand, keyed on `job_id`, TTL on `expires_at`. |
| S3 | `aws_s3_bucket.job_data` | Private, unversioned, SSE, 7-day lifecycle, 1-day multipart abort. `force_destroy = false`. |
| Secrets Manager | `aws_secretsmanager_secret.{api,provider}` | Metadata and ARN only; the values are seeded out of band. No recovery window. |
| CloudWatch | `aws_cloudwatch_log_group.backend` | 7-day retention, created before the services start. |
| ALB (internal) | `aws_lb.api`, `aws_lb_listener.http` | `internal = true`, HTTP/80, ingress from the VPC link's security group only. |
| API Gateway | `aws_apigatewayv2_api.http` | HTTP API with the generated `execute-api` endpoint enabled. This is the public URL. |
| VPC link | `aws_apigatewayv2_vpc_link.api` | Reaches the internal ALB. Its ENIs sit in the VPC link's subnets. |
| Integration | `aws_apigatewayv2_integration.api` | `HTTP_PROXY` / `PRIVATE` / `VPC_LINK`, method `ANY`, payload format 1.0, 29 s timeout. |
| Routes + stage | `aws_apigatewayv2_route.{root,proxy}`, `aws_apigatewayv2_stage.default` | `ANY /` and `ANY /{proxy+}`; the `$default` stage adds no path prefix. |
| IAM | execution + task roles | Narrow: ECR pull, logs, two secret ARNs, KMS decrypt, one queue, one table, one bucket. |

**Not created, on purpose:** a VPC, subnets, a NAT gateway, VPC endpoints, WAF,
ElastiCache, autoscaling, Container Insights. Fargate needs a VPC and subnets, so an **existing**
public VPC is an input. The tasks get public IPs for outbound access (ECR, CloudWatch, the LLM
provider); there is no NAT gateway to pay for. Private subnets without NAT or VPC endpoints would
block that egress.

The state bucket lives in `infra/bootstrap` — a separate, versioned, encrypted bucket with public
access blocked, TLS required, `force_destroy = false` and a `prevent_destroy` lifecycle guard.

---

## The API surface

| Route | Auth | Contract |
|---|---|---|
| `GET /healthz` | none | `503 {"status":"loading"}` until the local checkpoint loads, then `200 {"status":"ready"}`. The ALB health check uses it, so traffic only reaches a ready task. |
| `POST /predict` | bearer | `{"text": ..., "split": true}` → `{"provider", "split_fallback", "sub_comments": [{text, topic, topic_conf, sentiment, sentiment_raw}]}`. |
| `POST /jobs/uploads` | bearer | `{"filename": ...}` → `201 {"upload_id", "upload_url", "input_key", "expires_in"}`. A presigned PUT scoped to one key. |
| `POST /jobs/uploads/{upload_id}/finalize` | bearer | `{"options": {...}}` → `202 {"job_id", "status", "total_reviews"}`. Verifies the object, validates the CSV, creates the job, enqueues it. |
| `POST /jobs` | bearer | One `.csv` multipart upload (≤ 8 MiB) + an `options` JSON form field → `202 {"job_id", "status", "total_reviews"}`. |
| `GET /jobs/{job_id}` | bearer | `{"job_id", "status", "completed_reviews", "total_reviews", "error"}`. |
| `GET /jobs/{job_id}/result` | bearer | `{"job_id", "url", "expires_in"}` — a presigned URL, only once the job has completed. |

Status codes that matter:

- `401` — missing or wrong bearer token.
- `409` — the upload never arrived, or it was already finalized.
- `413` — the upload is over its limit (8 MiB on the multipart route, 50 MiB on the presigned route).
- `422` — the CSV has no `content` column, no rows, an empty `content` column, or more than 1,000
  reviews. Also an upload id that is not a UUID.
- `429` — the client is over its rate limit. `Retry-After` says how long to wait.
- `503` — the model is not loaded yet, the service is saturated, or the job could not be queued.
  `Retry-After` is set.

**Why a CSV does not go through `POST /jobs`.** An API Gateway HTTP API caps a request payload at
10 MB and the cap cannot be raised, so the multipart route is capped at 8 MiB. The full 50 MiB
contract lives on the upload flow: the frontend asks for a presigned URL, PUTs the CSV straight to
the private bucket, and finalizes. The frontend holds no AWS credentials — the URL authorises one
PUT to one key and expires in 15 minutes. The upload id becomes the job id, so the same upload
cannot become two jobs.

**`/predict` never fails because the LLM key is missing.** `scripts/llm_api` raises at import when
the active provider's key is unset; the service imports it lazily and turns that into a
whole-review fallback with `provider: "none"` and `split_fallback: true`. The UI shows that
fallback rather than an empty table.

**`/predict` also never overruns the gateway.** The provider call gets a 20-second budget
(retries included) and the whole split runs under a 25-second deadline, both shorter than the
gateway's 29-second integration timeout. When the deadline passes, the split is *abandoned, not
cancelled* — it is already paid for — and the caller gets the same whole-review fallback. The
gateway never has to answer 504 while the service is still working.

**The worker has its own, longer budget** (`WORKER_SPLIT_TIMEOUT_SECONDS`, 60 s). It has no gateway
in front of it and a job takes 17–50 minutes anyway, so applying the API's short budget there
would turn slow reviews into whole-review fallbacks for no reason.

---

## The worker

One SQS message is one job. The message body is `{job_id, input_key, options}` — a pointer, never
a payload.

The loop:

1. **Claim** the job in DynamoDB. Claimable means `PENDING`, `RETRYABLE`, or `RUNNING` with an
   expired lease. A completed or failed job is never claimed. A job with a **live** lease is left
   alone, so a duplicate waits instead of starting a second worker.
2. **Read** the input CSV from S3 in 25-review chunks.
3. **Split** each review with the LLM (one call per review) unless `options.split` is false, which
   treats the whole review as one sub-comment. A per-review split failure falls back to the whole
   review and marks that row `split_fallback`.
4. **Label** the resulting sub-comments with BERT in batches of 64.
5. **Commit** the chunk to `chunks/{job_id}/{chunk_index}.csv` in S3, *then* record the progress in
   DynamoDB. A worker that dies resumes from the last committed chunk and does not pay for the LLM
   calls again.
6. **Assemble** the committed chunks into `results/{job_id}/results.csv`, mark the job `COMPLETED`,
   and delete the message.

**Heartbeats.** The queue's visibility timeout (600 s) is far shorter than a full job: at 1,000
reviews and one sequential LLM call per review, a job takes roughly 17–50 minutes. The worker
therefore renews both the SQS visibility timeout and the DynamoDB lease every 60 seconds, logging
`job_heartbeat`. Without that, a second worker would pick the job up mid-flight.

**Failures.** A failed job is *not* deleted from the queue. SQS redelivers it; after three
attempts the message reaches the dead-letter queue and the job record becomes `FAILED`. A job whose
enqueue failed at the API is marked `FAILED` immediately and the API returns `503` rather than a
success that would never complete.

**The result CSV** preserves the input columns and adds `source_row_index`, `sub_comment_index`,
`sub_comment`, `bert_topic`, `bert_topic_conf`, `bert_sentiment`, `bert_sentiment_raw` and
`split_fallback`. `review_id` is carried through when the input has it, and is not invented when it
does not.

---

## Railway

| Setting | Value |
|---|---|
| Build | `Dockerfile.frontend` |
| Start | `streamlit run app.py --server.address 0.0.0.0 --server.port $PORT` |
| `BERT_INFERENCE_MODE` | `remote` |
| `REVIEW_SERVICE_URL` | the API Gateway endpoint — `terraform output -raw api_url` |
| `API_TOKEN` | the same token as the AWS secret — server-side only |
| Volume | mounted at `/app/models/trained_model` |

The frontend image has **no Torch and no Transformers** and does not contain the checkpoint or the
word2vec binary. The word2vec file (about 1.53 GB compressed) is uploaded once to the Volume:

```bash
railway volume add --service <service-id> --mount-path /app/models/trained_model \
  --project <project-id> --environment <environment-id> --json
railway volume files upload models/trained_model/GoogleNews-vectors-negative300.bin.gz \
  /GoogleNews-vectors-negative300.bin.gz --project <project-id> \
  --environment <environment-id> --service <service-id> --json
```

**Seed the volume before the first Streamlit run.** `load_word2vec()` caches "not available" for
the life of the process if it runs while the file is missing, so a late upload needs a restart.

**Scaling.** The cache is process-local: every replica performs its own cold load and holds its
own ~3.64 GB uncompressed matrix. Budget at least 8 GiB per replica and keep **one** replica until
the Volume's behaviour across replicas has been verified — five 1.53 GB copies would exceed the
stated 5 GB service storage allowance, and the 48 GB per-service ceiling does not raise the 8 GB
per-replica limit.

**Access.** The frontend is public by default. A server-side `API_TOKEN` protects the API but does
not stop a visitor from spending LLM credits through the UI. Restrict access with application
authentication or Railway's access controls.

---

## Cost

Recalculate the AWS and Railway bills **separately** — the old ~$3–4/day estimate covered only the
API and the worker. The Fargate tasks, the ALB, the API Gateway (per request) and its VPC link
(per hour per VPC link), SQS, DynamoDB and S3 are billed separately, and Railway bills for the
frontend, its RAM and its Volume separately. **Do not leave either environment running after the
demo.**

---

## Teardown

`terraform destroy` removes only what is in the selected state. It does **not** remove Railway,
externally managed DNS or certificates, shared networking, or anything created outside Terraform.
Use the exact account, region, workspace, `terraform.tfvars` and backend configuration that were
used to deploy. **Never** use `-auto-approve`, `terraform state rm` or a broad account-wide delete.

### 1. Freeze the demo and approve the deletion

- Disable new submissions from the Railway UI; let in-flight jobs finish (or decide explicitly to
  discard them); download any result CSVs worth keeping.
- Confirm that job inputs, results, job records, queue messages, logs and the demo-only secrets may
  be permanently deleted.
- Record the AWS account, region, workspace, state-bucket name, job-data bucket name and the
  resource names **before** destroying anything.
- Do not target the existing VPC, subnets, Internet Gateway, a shared IAM role, or a user-owned
  domain or certificate.
- Get explicit approval for the reviewed destroy plan **and** for the permanent removal of job data
  and state history.

### 2. Empty the job-data bucket

`force_destroy = false` keeps Terraform from silently erasing review data. After the owner's
approval:

```bash
aws s3 rm "s3://${JOB_DATA_BUCKET}" --recursive --region "${AWS_REGION}"
aws s3api list-multipart-uploads --bucket "${JOB_DATA_BUCKET}" --region "${AWS_REGION}"
```

Abort any incomplete multipart uploads. If versioning was ever enabled outside this plan, remove
every object version and delete marker as well — `aws s3 rm --recursive` alone does not do that.
**Do not delete the bucket yet**; Terraform removes it after the application stack is destroyed.

### 3. Destroy the application stack

```bash
terraform -chdir=infra/terraform plan -destroy -out=demo-destroy.tfplan
terraform -chdir=infra/terraform show -no-color demo-destroy.tfplan
# after a human reviews and approves the saved plan:
terraform -chdir=infra/terraform apply demo-destroy.tfplan
terraform -chdir=infra/terraform state list
```

The plan should contain only this demo's ECS services/tasks/task definitions, the ECR repository,
the ALB/listener/target group/security groups, the API Gateway HTTP API with its routes,
integration and stage, the VPC link, the SQS queue and DLQ, the DynamoDB table, the
**empty** job-data bucket, the log group, the dedicated IAM roles and policies, and the demo-only
secrets.

The ECR repository has `force_delete = true`, so its image manifests go with it. The demo secrets
have **no recovery window**, so their deletion is irreversible — rotate or revoke the API token and
the provider key and remove the Railway `API_TOKEN` variable first. Never apply those immediate
deletion settings to shared or production resources.

Afterwards, wait for AWS's asynchronous cleanup and verify there are no residuals: ECS services or
running tasks, task-definition revisions, task or ALB network interfaces, the HTTP API and its
routes/integration/stage, the VPC link and its network interfaces, ECR images, load
balancers, SQS messages, the DynamoDB table or its backups, job-data objects or multipart uploads,
the log group, the secrets, or the dedicated IAM roles and policies. Remove a DNS record or ACM
certificate only if this demo created it exclusively; leave externally owned ones intact.

The base topology does not create AWS Backup recovery points, DynamoDB point-in-time recovery, S3
replication, ALB access-log buckets or cross-account log exports. If any of those are enabled
later, add their destinations and retained artifacts to this checklist before calling the teardown
complete.

### 4. Remove the state bucket last

The state bucket must outlive the application stack so Terraform can record its destruction. Keep
it if the state history is required. Otherwise, after the application state is empty and no
Terraform operation is running:

```bash
# Comment the backend block out in infra/bootstrap/backend.tf, then:
terraform -chdir=infra/bootstrap init -migrate-state
terraform -chdir=infra/bootstrap state list
```

Verify that the local bootstrap state lists only the state-bucket resources. The versioned bucket
also holds the current and historical application/bootstrap state objects and may hold lockfile
versions. After explicit approval, permanently delete every object version and delete marker and
abort all multipart uploads — an ordinary `aws s3 rm --recursive` is **not** sufficient for a
versioned bucket. Use an audited, version-aware cleanup and a teardown role with only the necessary
version-list/delete and multipart-abort permissions; do not broaden the normal application role.

```bash
aws s3api list-object-versions --bucket "$TF_STATE_BUCKET" --region "$AWS_REGION"
aws s3api list-multipart-uploads --bucket "$TF_STATE_BUCKET" --region "$AWS_REGION"
```

This permanently erases the Terraform state history. Temporarily remove the `prevent_destroy`
guard, keep `force_destroy = false`, and inspect the local-state destroy plan before applying it:

```bash
terraform -chdir=infra/bootstrap plan -destroy -out=state-bucket-destroy.tfplan
terraform -chdir=infra/bootstrap show -no-color state-bucket-destroy.tfplan
# after a human explicitly approves:
terraform -chdir=infra/bootstrap apply state-bucket-destroy.tfplan
terraform -chdir=infra/bootstrap state list
```

**Keep the `prevent_destroy` guard in the bootstrap resource source.** After the bucket is gone,
return the bootstrap root to its initial local-backend state and re-add the S3 backend only after a
new state bucket exists. Retain the backend examples and the Terraform sources.

### 5. Tear down Railway separately

Terraform does not manage Railway. After confirming the local word2vec backup is intact, remove the
Railway service, its Volume and the `API_TOKEN` variable through the CLI or dashboard, and verify
that the service and the Volume are gone.

### 6. Preserve the local assets

Keep the repository source, the Terraform configuration and examples, the Dockerfiles, the
documentation, the BERT checkpoint and the local word2vec file. **Destroying cloud infrastructure
must not delete the local repository** — do not use `git clean -fdx` and do not remove model or
source directories as part of the teardown. The cloud ECR images and the remote job data are gone,
but both can be rebuilt and re-uploaded from the local assets.
