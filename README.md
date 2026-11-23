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
