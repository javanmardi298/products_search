# Product search with hybrid retrieval and reranking (WANDS)

Search for an online home-goods store, evaluated against **human relevance labels**:
about 43k Wayfair products, 480 queries and 233k labels from the
[WANDS](https://github.com/wayfair/WANDS) dataset (Chen et al., ECIR 2022).

The question: which retrieval setup finds the products people judged relevant, and is each
improvement big enough to be real, or just noise?

## Results (160 dev queries)

| Method | nDCG@10 | Recall@10 (Exact) | MRR | judged@10 | ~ms / query |
|---|---|---|---|---|---|
| Dense, name + description (baseline) | 0.700 | 0.275 | 0.518 | 0.842 | 22 |
| BM25, name + description | 0.602 | 0.259 | 0.466 | 0.781 | 41 |
| Dense, + category and features | 0.692 | 0.269 | 0.512 | 0.823 | 23 |
| BM25, + category and features | 0.630 | 0.250 | 0.492 | 0.799 | 52 |
| Hybrid (BM25 + dense, RRF) | 0.706 | 0.268 | 0.543 | 0.853 | 108 |
| **Hybrid + cross-encoder rerank** | **0.736** | **0.311** | **0.560** | 0.868 | 276 |

Gain in nDCG@10 over the baseline, with 95% paired-bootstrap confidence interval (10,000 resamples):

| Method | Gain | 95% CI | Real? |
|---|---|---|---|
| BM25 / basic | -0.098 | [-0.133, -0.065] | clearly worse |
| Dense / rich text | -0.007 | [-0.027, 0.012] | no difference |
| BM25 / rich text | -0.070 | [-0.106, -0.035] | clearly worse |
| Hybrid | +0.007 | [-0.018, 0.031] | no difference |
| **Hybrid + rerank** | **+0.037** | **[0.011, 0.063]** | **yes** |



![ablation](results/ablation_dev.png)



### What I learned
- **Only the reranker gave a real gain** (+0.037 nDCG@10, Recall@10 from 0.275 to 0.311).
  It costs about 12x the latency of dense search (276 ms vs 22 ms per query on a GPU).
- **Adding category and features to the product text did not help.** I expected it to.
  The dense model did not use that extra text better, and BM25 only improved a little.
- **Plain hybrid fusion (RRF) did not beat dense search alone.** The difference
  (+0.007) is inside the confidence interval.
- BM25 alone is much weaker than the embedding model: many queries use different words
  than the product text.

## How it works

| Step | What |
|---|---|
| Data | WANDS product, query and label files. Labels: Exact = 2, Partial = 1, Irrelevant = 0 |
| Product text | basic = name + description; rich = name + category + features + description |
| Dense | `BAAI/bge-small-en-v1.5`, normalized embeddings, cosine similarity |
| BM25 | `rank_bm25` on lowercase tokens without stopwords |
| Hybrid | Reciprocal Rank Fusion (k = 60) of BM25 and dense top-100 |
| Rerank | `cross-encoder/ms-marco-MiniLM-L-6-v2` on the top 50 hybrid results |

## How it is evaluated
- **Human labels only** decide relevance: nDCG@10 (gain 2^label - 1), Recall@10 of Exact products, MRR.
- **judged@10** is reported because WANDS labels are incomplete: unlabelled products
  count as irrelevant, so a method that finds good unlabelled products is under-rated.
- **Fixed split:** 160 dev and 320 test queries in `split.json` (seed 42). All choices
  above were made on dev. **The test queries have not been used yet.**
- **Paired bootstrap** on per-query nDCG for every comparison.

## Limitations
- Dev set only (160 queries); test results are still to come.
- WANDS labels are incomplete (judged@10 is 0.78 to 0.87).
- The reranker was trained on web question answering (MS MARCO), not on product search.
- Latencies were measured on a free Colab T4 GPU, one query at a time.
- Every comparison uses one embedding model and one reranker.


## Run 
1. `01_retrieval_colab.ipynb` in Google Colab,  T4 GPU and run all cells.
2. Locally: `pip install -r requirements.txt` and `python check_common.py`.

WANDS data (MIT license) is downloaded by the code and not stored in this repo.
