# The Cost of Ignoring Context in Sentiment Classification

An empirical ablation of the **bag-of-words assumption** for sentiment
classification, comparing formal (movie reviews) and informal (tweets) English text.

Research poster project — M.Sc. Natural Language Processing, University of Trier
(NLP module 12401631).
Authors: **Goutam Ravi Dubashi** (1905650), **Jaivik Alpeshbhai Viradiya** (1903410).

## Research questions

- **RQ1** — Does a *supervised* bag-of-words model beat a general-purpose sentiment lexicon?
- **RQ2** — Does adding local context (bigrams, negation marking) improve on unigrams?
- **RQ3** — Does the value of context differ between formal and informal text?
- **RQ4** — Do embeddings trained on the corpus beat sparse TF-IDF features?

## The model ladder

| | Model | Idea |
|---|---|---|
| A | VADER lexicon | unsupervised dictionary baseline |
| B | TF-IDF unigrams + Logistic Regression | supervised bag-of-words |
| C | + bigrams | adds local word-order context |
| D | + negation marking | `_NEG` tag after a negation cue to the clause boundary |
| E | Word2Vec (trained) + Logistic Regression | averaged distributional embeddings |

## Datasets

Both are loaded automatically from NLTK (no manual download).

| Corpus | Docs | Register | Source |
|---|---|---|---|
| `movie_reviews` | 2,000 | formal / long | NLTK; Pang & Lee (2004) |
| `twitter_samples` | 10,000 | informal / short | NLTK |

**Data-integrity note.** `twitter_samples` is distant-labelled by emoticons. Leaving
them in gives a spurious 99.9% accuracy, so the pipeline strips labelling emoticons,
URLs and handles before training (accuracy drops to a realistic ~79.6%).

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
python src/pipeline.py
```

This runs 5-fold stratified cross-validation for all five models on both datasets,
writes `results/results_cv.csv` and `results/results.json`, and regenerates the three
figures in `figures/`. Everything is seeded (`SEED=42`) for exact reproduction.

## Headline results (5-fold CV accuracy)

| Model | Movie Reviews | Tweets |
|---|---|---|
| VADER lexicon | 63.5% | 62.2% |
| TF-IDF unigram | **87.7%** | **79.6%** |
| + bigrams | 87.4% | 79.8% |
| + negation | 87.3% | 79.8% |
| Word2Vec (trained) | 75.1% | 74.6% |

**Finding.** The only large gain comes from learning from data (supervised bag-of-words
beats the lexicon by +24.2 / +17.4 pp). Adding local context changes accuracy by
&lt; 0.5 pp on both domains — yet on movie reviews, 100% of the errors the negation
model fixes contain a negation cue: context helps exactly where linguistics predicts,
but those cases are too rare to move the aggregate. Corpus-trained Word2Vec
underperforms sparse TF-IDF, motivating pretrained/contextual embeddings as future work.

## Repository layout

```
src/pipeline.py     end-to-end experiment (data, models, CV, error analysis, figures)
results/            results_cv.csv, results.json
figures/            fig_accuracy.png, fig_marginal.png, fig_confusion.png
requirements.txt    dependencies
LICENSE             MIT
```
