# AIPI540 A2

Analyzing customer reviews represent a significant aspect of every marketing / tech support team. However, they do not necessarily give the most informative info. In particular, there may be a part of the sentence that has a relatively positive sentiment but another part could be relatively negative. Furthermore, there may be cases where multiple topics are included in a particular review. It thus becomes a necessity to map the sentiment and topic to specific parts of the sentence. 

For instance, a review may be "I like how easily this app makes tracking my daily healh habits but not a fan of the increased monthly subscription cost". Obviously, the first part represents a positive review for the app features and the review turns negative for pricing. Traditional sentiment analysis may not be able to capture the nunace of customer reviews well enough. 

How to assign the sentiment score and topics to sentences chunks are the main subjects of exploration for this project. 

## Scripting (Harshitha) 
We first selected app on google play store that are long-standing, and new apps. Newer companies were also represented. Apps with a good amount of were selected. 
A total of 40 apps were made and stored in a json file. 

## Naive approach
Used a naive bayes to predict the sentiment of a whole sentence. Because 

## Classical Machine Learning Approach (Harshitha and Harry) 
Initially, we tried clustering, which was to assign a probability to each different cluster it identified. This unsupervised learning appraoch made sense since we had to make 
A downside to this approach was that we had to manually decide which cluster it was, leading to some subjectivity. It was also unclear how many clusters we would choose. 
However, this approach gave us some ideas on what topics we need to assign. Eventually we settled with the following topics: 
['price', 'function', 'design', 'customer service', 'other', 'quality', 'security', 'overall']

Then, we used a logistic regression model to obtain the coefficients for keys words, as shown below: 

![Final DataFrame for training](screenshots/logistic_table.png)

We also used the Rake model to obtain key words, with built-in extraction function, as shown below: 

![Final DataFrame for training](screenshots/rake_table.png)

We did some cleaning, such as lower casing, removing stop words, removing punctuations, and stemming. 

We then compared each word to the list of positive and negative words obtained from above and generated cosine similarity scores. For each word in a new sentence, we converted it into a word vector with a word2vec model, and made cosine similarity score with both the mean vector representation of both the positive word list and the negative word list (centroid vector). Then, we subtacted the negative similarity score from the positive similarity score to a get a final similarity metric. If this number is greater than a threshold (0.5), it is marked positive, otherwise negative. 

The output looks like:
![Final DataFrame for training](screenshots/rake_output.png)

Finally, with the key words being marked, we then color code the words and made a html file to visually display the sentiment analysis. 

## Deep Learning (Dave and XiaoQuan) 

We designed a prompt template for the LLM to parse the input text and output a JSON format. An example would look like this:

  {{
      "sub_comment": "I love this app",
      "topic": "quality",
      "sentiment": 1
  }},
  {{
      "sub_comment": ", but the price is too high.",
      "topic": "price",
      "sentiment": -1
  }}

Then, a python function is written to process the dataframe into a more legible format, with new columns "sub_comment", "topic", and "sentiment" to represent the corresponding information. 

## Evaluation 
As a quality check, the eval_model.py script evaluates the quality of model performance for multi-label topic classification with per-topic sentiment scores. It returns a python dictionary of evaluation metrics like Topic_Precision, Topic_F1, and Topic_Recall for topic classification and sentiment_MSE for sentiment scoring. A Topic_Sentiment_Coverage metric is also included.

![Final DataFrame for training](screenshots/cleaned.png)

We used BERT model to conduct both inferences on classification (assigning topic) and regression (predicting sentiment score) to each partial sentence. 

With the subsentence, we then added self-attention masks, trauncation, and padding to faciliate the model training. We removed stop words, stemmed words into their root forms, and made lower case for the words before tokenizing the words. BertTokenizerFast is the model for tokenization. 

We then split the data into train, val, and test sets for final evaluation. 

During training, we split the train dataset into batched to be passed into the BERT model, along with the self-attention masks, sentiment scores, topic, and sentiment_loss
For training the multi-classification, we used accuracy because we care equally about all the topics, not one particular class or imbalanced cases. The outputs are topic_logits, which then passes through a softmax function to map into proababilities for each topic class.

For training the scoring, we evaluated with RSME. It represents how accurate the model is at scoring by taking the root of the sum of squared errors. 

The trained checkpoint now ships in `models/saved_bert_model/` and is wired into the app as a fourth
analysis column. It loads **entirely from local files** — `config.json`, `head_config.json`,
`tokenizer.json` and `model.safetensors` — with `local_files_only=True`, so it never touches the
HuggingFace Hub or the network. Its contract:

- **8 topics**, in the data-derived order `customer service, design, function, other, overall,
  price, quality, security` (`sorted(df['topic'].unique())` over the training corpus). This is
  *not* the 6-label vocabulary the split prompt uses — the splitter keeps 6 labels, the BERT head
  has 8.
- **Sentiment is a regression score in [-1, 1]**, regressing the LLM's per-sub-comment
  `sentiment` float, *not* the review `score` column (1–5). Raw head outputs overshoot the range
  on ~18% of the corpus, so they are clamped to [-1, 1] and the unclamped value is kept as
  `sentiment_raw`.

## Front End

A Streamlit front end is made to demo our project details. We make sure to add coloring effect to
the front end design such that we are also capable of making

```bash
streamlit run app.py
```

### Input modes

| Mode | What it does |
|---|---|
| **Manual Review Input** | Analyzes one typed review with a prominent multi-select for Naive Bayes, Logistic Regression, RAKE, and BERT Multi-Task (all selected by default). |
| **Google Play Store App URL** | Fetches the newest English/US reviews for an app and submits BERT jobs to the async review service, one job per batch. |
| **CSV Upload** | Uploads raw reviews and submits BERT jobs the same way. |

### Manual model selection

Choose one or more models from the visible **Models to run** pills in the main panel. All four are
selected by default; only selected columns run. URL and CSV batch jobs always use BERT Multi-Task.

### Model behavior

- **Naive Bayes** and **Logistic Regression** show a review-level `Positive` / `Negative`
  percentage and colour the sentence word by word. Word-level colouring needs the word2vec binary
  (gitignored, about 1.53 GB from Figshare — see the setup below); without it the columns still show the review-level
  scores.
- **RAKE** lists the review's top keyword phrases. It needs no model and no word2vec.
- **BERT Multi-Task** splits the review into sub-comments with the configured LLM provider, labels
  each one with a topic and a sentiment, and shows the per-sub-comment table plus a rollup
  (dominant topic by mean confidence, mean sentiment). The sentiment is a **regression score in
  [-1, 1]**, not a percentage. Without a provider key the app still runs and labels the whole
  review as a single sub-comment, with a visible warning.

### Batch controls (URL and CSV modes)

| Control | Default | Range |
|---|---|---|
| Number of batches | 1 | 1–10 |
| Reviews per batch | 100 | 100–1,000 |

A run is capped at 10 batches / 10,000 reviews. The app shows the available rows, the selected
rows, the jobs it will create and the estimated number of LLM split requests (about one per
selected review) before anything is submitted; if the source has more rows than the selection
covers, you must confirm how many rows will be left unprocessed.

### CSV upload format

The `content` column is **required**; any other columns (for example `score` and `date`) are
preserved in the result. Uploads are limited to 50 MiB.

```csv
content,score,date
"The app is useful, but the subscription is too expensive.",2,2025-01-15
```

A CSV without a `content` column, an empty file, or a file over the size limit is rejected with an
actionable message.

### Results

Submitted batches are polled for status and show per-batch and overall progress. When the batches
finish, the app combines their result CSVs in `source_row_index` order and renders a table of
`source_row_index`, `content`, `sub_comment`, `bert_topic`, `bert_topic_conf`, `bert_sentiment`
and `split_fallback` (preview capped at 1,000 rows, with the full row count shown). The complete
combined annotated CSV is available for download.

Batch mode talks to the review service described in `.claude/plans/aws-deployment.md`; point the
local app at it with `REVIEW_SERVICE_URL` (default `http://localhost:8000`) and `API_TOKEN`. The
Railway frontend uses `BERT_INFERENCE_MODE=remote` and the AWS API's HTTPS URL.

## Install dependencies

```bash
python3.12 -m venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Run these commands from the repository root. The project pins a PyArrow version compatible with NumPy 1.26 and a
scikit-learn version matching the saved model pickles.

## Data Preprocessing

By running the following command, the data will be preprocessed and saved to `models/data/processed_data.csv`.

```bash
python scripts/preprocess_data.py
```

To check the quality of the processed data, run the following command:

```bash
python scripts/check_label_quality.py
```

## Local BERT inference

`models/bert_inference.py` loads the checkpoint from disk with no Hub and no network, and
`scripts/bert_inference.py` annotates the whole corpus and writes an agreement report:

```bash
python scripts/bert_inference.py --batch-size 64
```

This writes `data/processed/bert_sub_comments.csv` (one row per sub-comment) and
`data/processed/bert_agreement_report.txt`. The report measures **agreement with the annotator
that produced the corpus labels**, on the corpus the model was trained on — it is optimistic by
construction and is not accuracy against human ground truth.

## Setup notes

- `OPENROUTER_API_KEY` (or `GEMINI_API_KEY`) in `.env` is needed for the LLM split. The app starts
  without it and falls back to whole-review BERT.
- `models/saved_bert_model/model.safetensors` is **not** in git — it is 438 MB, over GitHub's
  100 MB per-file limit. The rest of the checkpoint directory is tracked; copy the weights in
  separately.
- `models/trained_model/GoogleNews-vectors-negative300.bin.gz` is gitignored and must be
  downloaded by hand (about 1.53 GB; Figshare link in `models/scoring.py`). In production, upload it directly
  to the Railway frontend's persistent Volume using `railway volume files upload`; it is not in the AWS
  API/worker image or AWS job-data S3 bucket. Keep a local backup because Railway Volume-only storage has no
  separate remote recovery copy.

## Deployment

The demo runs as two pieces: **AWS hosts the API and the worker**, **Railway hosts the Streamlit
frontend**. The frontend never loads BERT — in the hosted environment it calls the AWS API over
HTTPS. AWS resources are provisioned with Terraform; the Railway service and its Volume are set up
by hand (or with the Railway CLI).

```
Railway (Streamlit)  ──HTTPS + bearer token──▶  ALB  ──▶  ECS/Fargate API  ──▶  SQS
                                                                                    │
                                                                            ECS/Fargate worker
                                                                                    │
                                                            S3 (job input/result) ◀──┘
```

### The API

| Route | Auth | What it does |
|---|---|---|
| `GET /healthz` | none | `503 {"status":"loading"}` until the local checkpoint loads, then `200 {"status":"ready"}`. |
| `POST /predict` | bearer | One review in, split and labelled sub-comments out. |
| `POST /jobs` | bearer | One CSV upload (≤ 50 MiB, ≤ 1,000 reviews, a `content` column) becomes one job. |
| `GET /jobs/{job_id}` | bearer | Status plus `completed_reviews` and `total_reviews`. |
| `GET /jobs/{job_id}/result` | bearer | A short-lived presigned URL, only after the job completes. |

Every route except `/healthz` needs `Authorization: Bearer $API_TOKEN`. `429` means the client is over
its rate limit; `503` means the model is not ready or the service is saturated.

### Terraform

Two roots: `infra/bootstrap` creates the remote-state bucket, `infra/terraform` creates the
application stack. The bootstrap root starts with local state because the bucket does not exist yet.

```bash
# 1. bootstrap the state bucket (plan, review, then apply — applying needs approval)
terraform -chdir=infra/bootstrap fmt -check -recursive
terraform -chdir=infra/bootstrap init
terraform -chdir=infra/bootstrap validate
terraform -chdir=infra/bootstrap plan -out=state-bucket.tfplan
terraform -chdir=infra/bootstrap show -no-color state-bucket.tfplan
terraform -chdir=infra/bootstrap apply state-bucket.tfplan

# 2. copy the backend configs and migrate the state
cp infra/bootstrap/backend.hcl.example infra/bootstrap/backend.hcl   # then fill it in
cp infra/terraform/backend.hcl.example infra/terraform/backend.hcl     # then fill it in
# uncomment the backend block in infra/bootstrap/backend.tf
terraform -chdir=infra/bootstrap init -migrate-state -backend-config=backend.hcl

# 3. the application stack
terraform -chdir=infra/terraform init -backend-config=backend.hcl
terraform -chdir=infra/terraform validate
terraform -chdir=infra/terraform plan -var='desired_count=0' -out=bootstrap.tfplan
terraform -chdir=infra/terraform show -no-color bootstrap.tfplan
```

`terraform.tfvars` is gitignored; `terraform.tfvars.example` is tracked and holds only non-secret
settings. **Never put `API_TOKEN` or `OPENROUTER_API_KEY` in a `.tfvars` file, in the backend
configuration, or in a Terraform variable** — they live in Secrets Manager and in Railway's sealed
variables.

The stack needs an **existing VPC** with at least two public subnets in different Availability
Zones, a **public DNS name** for the API, and an **ACM certificate** for that name in the same
region. Point the domain at the load balancer's DNS name (`terraform output -raw alb_dns_name`).
If the domain or the certificate is not available, **do not** fall back to plain HTTP — the bearer
token would cross the internet in the clear.

### Applying, in two stages

1. Apply with `desired_count=0`. This creates everything and starts no tasks — ECS must not try to
   pull an image that does not exist yet.
2. Seed the two Secrets Manager secrets with their values, out of band (the AWS console or a
   secret-delivery pipeline). Terraform manages only the secrets' metadata and ARNs.
3. Build and push the image:

   ```bash
   AWS_REGION=$(terraform -chdir=infra/terraform output -raw aws_region)
   API_ECR_URL=$(terraform -chdir=infra/terraform output -raw api_ecr_repository_url)
   export IMAGE_TAG=v1   # the same value as `image_tag` in terraform.tfvars
   aws ecr get-login-password --region "$AWS_REGION" \
     | docker login --username AWS --password-stdin "${API_ECR_URL%%/*}"
   docker build --platform linux/amd64 -f Dockerfile -t "$API_ECR_URL:$IMAGE_TAG" .
   docker push "$API_ECR_URL:$IMAGE_TAG"
   ```

   `--platform linux/amd64` matters: Fargate runs x86-64, and an image built on an Apple Silicon
   machine is arm64.
4. Apply again with `desired_count=1` to start the API and the worker.

### The worker

The worker reads the input CSV in 25-review chunks, calls the LLM splitter once per review, labels
the resulting sub-comments with BERT in batches of 64, and commits each chunk to S3 before
recording the progress. A job that dies resumes from the last committed chunk instead of repeating
the LLM calls it already paid for. The SQS visibility timeout (600 s) and the DynamoDB lease are
renewed every 60 seconds, because one job takes far longer than the visibility timeout: roughly
17–50 minutes for a full 1,000-review job. A job that fails three times lands in the dead-letter
queue and its record is marked `FAILED`.

### Railway (the frontend)

The frontend image is `Dockerfile.frontend`. It installs Streamlit, pandas, PyArrow,
scikit-learn and Gensim — **not** Torch or Transformers, because the hosted frontend calls the
remote API instead of loading BERT.

Railway variables:

| Variable | Value |
|---|---|
| `BERT_INFERENCE_MODE` | `remote` |
| `REVIEW_SERVICE_URL` | `https://<your API domain>` |
| `API_TOKEN` | the same token that is in the AWS secret — server-side only |

The start command is `streamlit run app.py --server.address 0.0.0.0 --server.port $PORT`; Railway
supplies `$PORT` and the public domain.

The GoogleNews word2vec binary is **not** in the image. Attach a Railway Volume at
`/app/models/trained_model` and seed it once:

```bash
railway volume add --service <service-id> --mount-path /app/models/trained_model \
  --project <project-id> --environment <environment-id> --json
railway volume files upload models/trained_model/GoogleNews-vectors-negative300.bin.gz \
  /GoogleNews-vectors-negative300.bin.gz --project <project-id> \
  --environment <environment-id> --service <service-id> --json
```

Seed the volume **before** the first Streamlit run: if `load_word2vec()` runs while the file is
missing, the process caches "not available" and needs a restart. The file is about 1.53 GB
compressed and expands to about 3.64 GB in memory, so budget at least 8 GiB per replica and keep
one replica until the volume's behaviour across replicas has been verified.

### Cost and access

- The AWS API and the worker each run 1 vCPU / 2 GiB on Fargate, plus the ALB, SQS, DynamoDB and
  S3. Railway charges for the frontend and its Volume separately. Recalculate both before
  deploying, and **do not leave either environment running after the demo**.
- **The frontend is public by default.** A server-side `API_TOKEN` protects the API, but it does
  not stop a visitor from spending your LLM credits by triggering jobs through the UI. Restrict
  access with application authentication or Railway's access controls.

### Teardown

`terraform destroy` removes only what is in the selected state — not Railway, not externally
managed DNS or certificates, not shared networking. Use the same account, region, workspace,
`.tfvars` and backend configuration that were used to deploy, and never use `-auto-approve`,
`terraform state rm` or a broad account-wide delete.

The full checklist — freezing the demo, emptying the job-data bucket, destroying the application
stack, removing the state bucket last, and tearing Railway down separately — is in
`.claude/references/deployment.md`.

### Local proof of concept

```bash
docker build -t reviewassist-api:local -f Dockerfile .
docker run --rm -p 8000:8000 \
  -e AWS_REGION=us-east-1 -e JOB_BUCKET=demo \
  -e QUEUE_URL=https://sqs.us-east-1.amazonaws.com/1/demo -e JOBS_TABLE=demo \
  -e API_TOKEN=dev-token reviewassist-api:local
curl -s localhost:8000/healthz
```

The service tests run against moto's in-memory S3, SQS and DynamoDB:

```bash
uv pip install -r requirements-dev.txt
uv run pytest tests/ -q
```

