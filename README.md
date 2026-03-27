# Phase4_Group4_Project

# Twitter Sentiment Analysis: Apple & Google Products
## Phase 4 Group 4 Project

---

## Summary

Social media platforms generate massive volumes of consumer opinion data every day. For technology companies like Apple and Google, understanding whether public discourse is positive, negative, or neutral is essential for protecting brand reputation, guiding product development, and informing marketing strategy. Manual analysis at this scale is infeasible, creating a clear need for automated sentiment classification systems.

This project develops a supervised Natural Language Processing (NLP) pipeline to classify the sentiment of approximately 9,000 tweets about Apple and Google products. The dataset, sourced from CrowdFlower via data.world, contains tweets human-labeled as positive, negative, or neutral. The class distribution is heavily imbalanced — neutral tweets represent roughly 60% of the data, positive tweets approximately 33%, and negative tweets only about 6%. This imbalance directly shaped our evaluation strategy, leading us to adopt weighted F1-score over accuracy as the primary metric.

Text preprocessing was performed using NLTK, with careful retention of negation terms to preserve sentiment-bearing context. Tweets were vectorized using TF-IDF with sublinear scaling, trigrams, and expanded vocabulary. Class imbalance was addressed through SMOTE oversampling within cross-validation folds to prevent data leakage. We evaluated models iteratively across multiple packages — scikit-learn (Logistic Regression, Naive Bayes, Random Forest, LinearSVC), XGBoost, and spaCy word embeddings — with hyperparameters tuned via GridSearchCV. Model interpretability was assessed using LIME for local explanations and SHAP for global feature importance analysis.

The final tuned model achieved strong weighted F1-score performance on the held-out test set, with the strongest classification on positive and neutral classes. The negative class remains the most challenging due to limited training examples. We discuss stakeholder implications, model limitations including dataset age and sarcasm detection, and recommend future work including transformer-based models and real-time deployment pipelines.


## 1. Introduction: Business Understanding

### The Real-World Problem

In today's digital landscape, companies like Apple and Google receive **thousands of mentions on social media every hour**. Manually reading and categorizing this volume of feedback is impossible. Yet understanding public sentiment is critical — a viral negative tweet about a product defect can escalate into a PR crisis within hours, while positive buzz around a product launch can be amplified for marketing advantage. **Companies need an automated, scalable way to monitor and classify sentiment in real-time.**

### Stakeholders and How They Would Use This Model

**Product Marketing Teams** Monitor sentiment during product launches (e.g., new iPhone, Google Pixel). Identify which features generate positive buzz and which receive criticism. Adjust messaging in real-time.

**Customer Support / CX Teams** Set up alerts for spikes in negative sentiment to proactively address widespread product issues before they escalate. Prioritize support resources.

**Product Development Teams** Analyze negative tweets to identify recurring complaints (e.g., battery life, software bugs). Feed insights into the product roadmap for future iterations.

**Executive Leadership / PR** Track overall brand health over time via sentiment dashboards. Compare sentiment between Apple vs. Google products. Prepare data-driven responses to media inquiries.

### Our Solution
We build an NLP classifier that can automatically categorize a tweet as **positive**, **negative**, or **neutral**  enabling stakeholders to process thousands of tweets per minute and power automated dashboards, alerts, and reports.

## 2. Data Understanding

###  Data Source

The dataset comes from **CrowdFlower** (now Figure Eight / Appen) via [data.world](https://data.world/crowdflower/brands-and-product-emotions). It contains **9,093 tweets** collected during the 2011 South by Southwest (SXSW) conference in Austin, Texas. Each tweet was reviewed by human raters who assessed:

1. **Which brand or product** the tweet is directed at (e.g., iPhone, iPad, Google, Android)
2. **The sentiment** expressed toward that brand/product (Positive, Negative, Neutral, or "I can't tell")

### Why This Dataset Is Suitable

##### Criterion  Assessment 

 **Labeled data**  Human-annotated sentiment labels enable supervised classification — no need for unsupervised labeling heuristics
 
 **Real-world text**  Tweets are authentic social media posts with natural language patterns (abbreviations, hashtags, slang, sarcasm) 
 
 **Multiclass labels**  Positive/Negative/Neutral categories match real business needs for brand monitoring 
 
 **Relevant domain**  Apple and Google are among the world's most discussed brands — findings generalize to tech brand monitoring 
 
 **Manageable size**  ~9,000 tweets is large enough for classical ML models while remaining computationally feasible on local machines 

### Data Limitations (Identified Upfront)

1. **Temporal bias** — All tweets are from SXSW 2011. Language patterns, products (iPhone 4, iPad 1), and slang have changed dramatically. The model may not generalize to modern tweets.
2. **Event-specific context** — SXSW is a tech conference, so the audience skews tech-savvy and enthusiastic. Sentiment distribution may differ from everyday Twitter discourse.
3. **Annotation ambiguity** — The dataset includes an "I can't tell" label, indicating that even human raters found some tweets ambiguous. This sets a ceiling on achievable model accuracy.
4. **No demographic metadata** — We have no information about tweet authors (location, follower count, verified status), which could provide useful context signals.
5. **Class imbalance** — As we'll see below, the neutral class heavily dominates, making minority class (negative) prediction challenging.


## 3. Data Preparation

**Libraries used:**
- `NLTK` — tokenization, stopword list, lemmatization (fine-grained control over preprocessing)
- `re` (regex) — pattern-based text cleaning
- `scikit-learn TfidfVectorizer` — converting text to numerical features

Text data requires specialized preprocessing. Raw tweets contain noise (URLs, @mentions, special characters) that doesn't carry sentiment information and would confuse our model. Our preprocessing pipeline is designed to:

1. **Remove noise** — URLs, @mentions, hashtag symbols, special characters
2. **Normalize text** — lowercase everything for consistency
3. **Tokenize** — split text into individual words
4. **Remove stopwords selectively** — remove common words EXCEPT negation terms ("not", "no", "never", etc.) because these flip sentiment
5. **Lemmatize** — reduce words to base forms ("running" → "run")


### Advanced Data Preparation

Beyond basic text cleaning, we engineer additional features and use unsupervised techniques to better understand and represent our data.

**Advanced techniques:**
1. **Feature Engineering** — Extract metadata from tweets (punctuation, capitalization, lexicon hints)
2. **Unsupervised Clustering (K-Means)** — Explore natural groupings in TF-IDF space
3. **Pipeline Architecture** — Combine TF-IDF + engineered features using `ColumnTransformer` + `Pipeline`


## 4. Modeling: Iterative Approach

Our modeling strategy follows an **iterative, justified progression**:

1. **Baseline** → Binary Logistic Regression (simplest possible model)
2. **Expand scope** → Multiclass classification (add neutral class)
3. **Compare algorithms** → LR vs. Naive Bayes vs. Random Forest (scikit-learn)
4. **Tune best model** → GridSearchCV with Pipeline
5. **Cross-package comparison** → XGBoost (xgboost), spaCy embeddings
6. **Combined features** → TF-IDF + engineered features via ColumnTransformer

Each step is justified by the results of the previous step.

### Baseline: Binary Classification (Positive vs. Negative)

We start with the **simplest credible model**: a Logistic Regression binary classifier using only positive and negative tweets. This serves three purposes:

1. **Validates the pipeline** — confirms preprocessing and vectorization work correctly
2. **Establishes a performance floor** — any multiclass model should at least approach this on the positive/negative distinction
3. **Highest stakeholder value** — positive vs. negative is the most actionable distinction for brand monitoring

### Complete Model Comparison

**Model Selection Rationale:**

| Iteration | Model | Change Made | Justified By |
|---|---|---|---|
| Baseline | Binary LR | Starting point | Simplest model; validates pipeline |
| 1 | Multiclass LR / NB / RF | Expand to 3 classes; compare algorithms | Binary baseline confirmed pipeline works |
| 2 | Tuned LR Pipeline | GridSearchCV on TF-IDF + LR params | LR was best in Iteration 1 |
| 3 | XGBoost | Non-linear model from different package | Test if non-linear interactions improve over LR |
| 4 | spaCy Embeddings | Different text representation | Test if semantic similarity beats exact word matching |
| 5 | TF-IDF + Engineered Features | Add structural features | Test if tweet metadata adds signal beyond words |

**Best model:** Selected based on highest **Weighted F1** on the test set. If multiple models are close, the **simpler model** (Logistic Regression) is preferred for interpretability and deployment.

### Top Features per Sentiment Class

Understanding which words the model associates with each class provides interpretability — crucial for stakeholder trust.

---

## 5. Model Explainability

High-performing models are only useful if stakeholders **trust** them. We use two complementary tools:

1. **LIME** (`lime` package) — Explains individual predictions by perturbing input text
2. **SHAP** (`shap` package) — Provides global and local explanations grounded in game theory

| Tool | Scope | Method | Best For |
|------|-------|--------|----------|
| **LIME** | Local (single prediction) | Perturbation-based | Explaining to non-technical stakeholders |
| **SHAP** | Global + Local | Shapley values | Ensuring model learns real patterns, not artifacts |