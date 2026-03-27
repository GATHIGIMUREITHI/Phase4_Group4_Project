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

## 5. Modeling Approach

Our modeling follows a strict **iterative, evidence-driven progression**. Each new model is justified by the results of the prior model.

### Evaluation Strategy

| Decision | Choice | Rationale |
|---|---|---|
| **Primary metric** | Weighted F1-score | Accuracy is misleading when ~60% of tweets are neutral. A naive "always predict neutral" baseline would score ~60% accuracy. |
| **Secondary metrics** | Macro F1, Accuracy | Macro F1 treats all classes equally (important for minority negative class). Accuracy is included for reference only. |
| **Splitting strategy** | 80/20 stratified train/test split | Preserves class proportions in both sets |
| **Validation** | 5-fold stratified cross-validation | Confirms performance is stable across different data subsets |
| **Class imbalance** | `class_weight='balanced'` + SMOTE | Upweights minority classes in the loss function; SMOTE generates synthetic minority examples |

### Baseline: Binary Classification

**Model:** Logistic Regression (positive vs. negative tweets only)

**Why start here:**
- Simplest credible model — validates that the preprocessing and vectorization pipeline works correctly
- Positive vs. negative is the highest-value distinction for stakeholders
- Establishes a performance floor that any multiclass model should approach

**Result:** Strong performance, confirming that TF-IDF captures clear sentiment words ("love", "great" vs. "hate", "crash"). Pipeline validated.

**Decision → next step:** Pipeline works. Expand to the full 3-class problem.

---

### Iteration 1: Multiclass Algorithm Comparison

**Models compared (all scikit-learn):**

| Algorithm | Why Selected | Strengths | Weaknesses for This Task |
|---|---|---|---|
| Logistic Regression | Strong linear baseline for text | Works well with sparse TF-IDF; interpretable coefficients | Cannot capture non-linear interactions |
| LinearSVC | SVMs excel on high-dimensional sparse data | Maximum-margin classification; often beats LR on text | No probability estimates by default |
| Multinomial Naive Bayes | Classic NLP algorithm | Very fast; good with word counts | Assumes feature independence (violated by bigrams) |
| Random Forest | Non-linear ensemble | Captures feature interactions | Struggles with high-dimensional sparse data |

All models were trained on SMOTE-resampled training data to address class imbalance.

**Key findings:**
- Logistic Regression and LinearSVC lead — confirming the sentiment signal is **largely linear** in TF-IDF space
- Random Forest underperforms — high-dimensional sparse features are not ideal for tree splits
- All models struggle with the negative class due to its small size (~6%)

**Decision → next step:** LR/SVC are best. Tune their hyperparameters along with the TF-IDF vectorizer.

---

### Iteration 2: Hyperparameter Tuning with Pipeline

**Approach:**
- Wrapped TF-IDF + SMOTE + Classifier in an `imblearn.pipeline.Pipeline`
- This ensures TF-IDF vocabulary is learned only from training folds and SMOTE is applied only to training data — **no data leakage**
- Tuned both Logistic Regression and LinearSVC with `GridSearchCV`

**Parameters searched:**

| Parameter | Values Tested | What It Controls |
|---|---|---|
| `tfidf__max_features` | 10000, 15000, 20000 | Vocabulary size — more features capture more signal but risk noise |
| `tfidf__ngram_range` | (1,2), (1,3) | Whether bigrams/trigrams are included |
| `clf__C` | 0.1, 0.5, 1.0, 5.0, 10.0 | Regularization strength — lower C = more regularization |

**Key findings:**
- Tuning improved Weighted F1 over the untuned models
- If bigrams/trigrams were selected, multi-word expressions carry important sentiment signal
- The best C value reveals how much regularization the data needs — higher C means the features are genuinely informative

**Decision → next step:** Optimized within linear scikit-learn models. Test whether non-linear structure exists using XGBoost from a different package.

---

### Iteration 3: XGBoost (Cross-Package)

**Package:** `xgboost` (gradient boosting framework)

**Why try XGBoost:**
- Builds sequential decision tree ensembles where each tree corrects prior errors
- Can capture **non-linear feature interactions** (e.g., "not" + "good" together)
- Built-in L1/L2 regularization
- Represents a fundamentally different algorithmic family than linear models

**Tuning:** GridSearchCV over `n_estimators` (200–500), `max_depth` (4–8), `learning_rate` (0.01–0.1), `subsample`, and `colsample_bytree`, with SMOTE in the pipeline.

**Key findings:**
- If XGBoost matches LR/SVC: the signal is mostly linear → prefer the simpler model (Occam's razor)
- If XGBoost clearly outperforms: non-linear interactions matter → consider for production

**Decision → next step:** All models so far use TF-IDF (bag-of-words). Test whether semantic understanding helps using spaCy embeddings.

---

### Iteration 4: spaCy Word Embeddings

**Package:** `spacy` with `en_core_web_sm` (96-dimensional pre-trained word vectors)

**Why try embeddings:**
- TF-IDF treats every word independently — "good" and "great" are unrelated features
- Word embeddings place semantically similar words close together — "good" ≈ "great" ≈ "excellent"
- Dense, low-dimensional (96d) vs. sparse, high-dimensional (15000d for TF-IDF)
- Tests a **fundamentally different text representation** from a different NLP package

**Key findings:**
- If TF-IDF outperforms embeddings: **exact word presence** matters more than semantic similarity for this dataset — specific words like "love", "crash", "not" are strong, precise signals
- The small `en_core_web_sm` model has limited embedding quality; larger models (`en_core_web_md`, `en_core_web_lg`) would perform better

**Decision → next step:** Test whether combining TF-IDF with engineered structural features improves performance.

---

### Iteration 5: Combined TF-IDF + Engineered Features

**Approach:** `ColumnTransformer` + `Pipeline` combining:
1. TF-IDF features (word content)
2. Engineered features scaled with `StandardScaler`:
   - Exclamation count, question mark count
   - Capitalization ratio, all-caps word count
   - Character length, word count
   - Hashtag/mention counts
   - Positive/negative lexicon word counts

**Why:** Structural patterns in tweets (heavy punctuation, ALL CAPS, dense hashtags) may carry sentiment signal that word frequencies alone miss.

**Key findings:**
- If the combined model improves: structural features carry complementary signal
- If performance is similar: TF-IDF already captures the signal, but engineered features still add interpretability
- The pipeline is **production-ready** — serializable as a single object

---

### Iteration 6: Stacked Word + Character N-grams

**Approach:** Stack two TF-IDF matrices:
- **Word TF-IDF** (1–3 grams, 15000 features)
- **Character TF-IDF** (2–5 char grams, 10000 features, `analyzer='char_wb'`)

**Why character n-grams help:**
- Capture morphological patterns and word fragments
- Robust to misspellings common in tweets ("gr8", "luv", "awsum")
- Provide sub-word information that word-level TF-IDF misses entirely

Trained with LinearSVC + SMOTE on the stacked 25000-feature matrix.

---

### Complete Model Comparison

All models are compared on the same held-out test set using Weighted F1 (primary), Macro F1, and Accuracy:

| Iteration | Model | Package(s) | Representation | Key Change |
|---|---|---|---|---|
| Baseline | Binary LR | scikit-learn | TF-IDF | Validates pipeline |
| 1 | LR / SVC / NB / RF | scikit-learn | TF-IDF + SMOTE | Compare algorithms |
| 2 | Tuned LR or SVC | scikit-learn + imblearn | TF-IDF + SMOTE | Hyperparameter optimization |
| 3 | Tuned XGBoost | xgboost + imblearn | TF-IDF + SMOTE | Non-linear model, different package |
| 4 | LR + spaCy | spacy + scikit-learn | Word embeddings | Different text representation |
| 5 | LR + ColumnTransformer | scikit-learn | TF-IDF + Engineered features | Structural tweet patterns |
| 6 | SVC + Stacked TF-IDF | scikit-learn | Word + Character n-grams | Sub-word information |

The best model is selected based on highest Weighted F1. When models are close, the simpler model is preferred for interpretability and deployment.

---


## 6. Model Explainability

## Model Explainability

We use two complementary explainability tools to ensure stakeholders can **trust and understand** model predictions:

### LIME (Local Interpretable Model-agnostic Explanations)
- **Package:** `lime`
- **Scope:** Explains **individual predictions**
- **Method:** Creates perturbed versions of a tweet (randomly removing words), gets predictions on all versions, and fits a local linear model to identify which words pushed the prediction in each direction
- **Stakeholder value:** *"This tweet was classified as negative because of 'crash', 'dead', and 'not working'."*

### SHAP (SHapley Additive exPlanations)
- **Package:** `shap`
- **Scope:** **Global** feature importance + **local** per-prediction explanations
- **Method:** Uses Shapley values from cooperative game theory to fairly attribute each feature's contribution to the prediction
- **Applied to:** Both the best linear model (`LinearExplainer`) and XGBoost (`TreeExplainer` for exact values)
- **Stakeholder value:** Reveals the top words driving each sentiment class across the entire test set. Comparing SHAP importance across models confirms whether different algorithms learn the same patterns.

### Cross-Model Explainability
Comparing SHAP feature importance between the linear model and XGBoost reveals:
- **Same top features** → sentiment signal is robust and consistent across model types
- **Different top features** → models capture different aspects of the data

---

## Misclassification Analysis

Understanding where the model fails is as important as knowing where it succeeds. The notebook examines:

1. **Error volume:** Total misclassified tweets as a percentage of the test set
2. **Error breakdown:** Which class pairs are most commonly confused (e.g., neutral predicted as positive)
3. **Sample review:** Individual misclassified tweets are displayed to identify qualitative patterns

**Common error patterns:**
- **Neutral ↔ Positive confusion:** Many neutral tweets contain mildly positive language ("just got the new iPad") that doesn't express strong emotion but contains product-positive words
- **Negative class misses:** The small negative training set (~6%) means the model has fewer examples to learn from
- **Sarcasm and irony:** Bag-of-words models cannot detect sarcasm — "Great, my iPhone crashed again" contains positive words but negative sentiment

---

## Cross-Validation

5-fold stratified cross-validation on the training set confirms that model performance is **stable and not an artifact of a lucky train/test split**:
- Fold scores are reported individually
- Mean F1 and standard deviation are computed
- Low standard deviation confirms consistent performance across folds

---


High-performing models are only useful if stakeholders **trust** them. We use two complementary tools:

1. **LIME** (`lime` package) — Explains individual predictions by perturbing input text
2. **SHAP** (`shap` package) — Provides global and local explanations grounded in game theory

| Tool | Scope | Method | Best For |
|------|-------|--------|----------|
| **LIME** | Local (single prediction) | Perturbation-based | Explaining to non-technical stakeholders |
| **SHAP** | Global + Local | Shapley values | Ensuring model learns real patterns, not artifacts |


## 7. Conclusions and Stakeholder Recommendations

### What This Means for Each Stakeholder

| Stakeholder | Implication |
|---|---|
| **Product Marketing** | The model reliably identifies positive tweets for amplification campaigns and detects negative spikes during launches. Borderline neutral tweets may need manual review for high-stakes campaigns. |
| **Customer Support** | Negative tweet detection enables proactive issue response. The model's recall on negative tweets determines how many complaints it catches — stakeholders should monitor this metric. |
| **Product Development** | Top negative-sentiment features from SHAP/LIME directly indicate which product issues are most discussed (e.g., "battery", "crash", "slow"). These can be prioritized in product roadmaps. |
| **Executive / PR** | Sentiment trends can be tracked over time via dashboards powered by this model. The model provides a reliable, data-driven pulse on brand perception. |

### Key Takeaways
1. **Linear models (LR, LinearSVC) perform best** on this dataset — the sentiment signal in tweet text is primarily captured by word presence/absence
2. **SMOTE + class weights** together improve minority class (negative) detection
3. **TF-IDF with sublinear scaling and trigrams** outperforms word embeddings for this task — exact word matching beats semantic similarity when specific product terms are strong signals
4. **The model is interpretable** — LIME and SHAP provide transparent explanations for every prediction, enabling stakeholder trust

---

## Limitations

1. **Dataset age** — All tweets are from SXSW 2011. Language, products (iPhone 4, iPad 1), slang, and social media conventions have changed dramatically. The model must be retrained on current data for production use.
2. **Class imbalance** — Negative tweets represent only ~6% of the data. Even with SMOTE and class weights, the model has fewer real negative examples to learn nuanced patterns from.
3. **Sarcasm and irony** — TF-IDF and linear models treat words independently. "Great, my phone died again" contains positive words but expresses negative sentiment. This is a fundamental limitation of bag-of-words approaches.
4. **Event-specific bias** — SXSW is a tech conference with a tech-savvy, enthusiastic audience. Sentiment distributions in everyday Twitter discourse may differ significantly.
5. **No aspect-level sentiment** — The model classifies overall tweet sentiment, not sentiment toward specific product features (e.g., "love the camera but hate the battery life" is one label).
6. **Generalizability** — Trained exclusively on Apple and Google product tweets. Performance on tweets about other brands, industries, or topics is unknown without further evaluation.
7. **Annotation ceiling** — The original dataset contains "I can't tell" labels, indicating that even human annotators found some tweets ambiguous. This places a natural ceiling on achievable model accuracy.

## Tableau Link

https://public.tableau.com/views/Book1_17611539348850/Dashboard1?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link

## Tableau Image
![Alternative Text](path/to/your/image.png)


