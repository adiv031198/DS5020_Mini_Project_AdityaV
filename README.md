# Customers Like You: PCA or SVD for GiftLane

DS5020 Mini-Project · **Aditya Vishwanath** (individual project: Data Engineer and ML Engineer)

A compressed purchase model that powers a "Customers who bought this also bought…" box and a customer-segment map for GiftLane, built on the UCI Online Retail II dataset.

## Success targets (J1)

Agreed on Day 1 from the stakeholder's email, and checked against the final results:

| # | Target | Result | Met? |
|---|---|---|---|
| 1 | Stored model much smaller than the full matrix (under 10 MB) | 3.7 MB of factors vs 161 MB dense matrix | ✅ |
| 2 | Nightly job fast (under 60 seconds) | 3.5 seconds for the full job | ✅ |
| 3 | Recommendations beat the 10-best-sellers baseline on unseen purchases | Hit rate at 10: **40.0%** vs **26.9%** | ✅ |
| 4 | Customer map with 3 to 6 groups marketing can name | 6 named segments | ✅ |

## Data

- **Source:** Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D (CC BY 4.0)
- **File:** `online_retail_II.xlsx`, two sheets (2009–2010: 525,461 rows; 2010–2011: 541,910 rows)
- **Row count loaded:** **1,067,371**, matching the brief
- Raw data is **not** committed to Git. `data/` is in `.gitignore`. Customer IDs never leave `data/` (R4).

## Repository structure

```
build_matrix.py          DE6: rebuilds the whole data pipeline in one command
requirements.txt         exact package versions (R6)
decision_log.md          every decision and its reason (R1)
notebooks/DS5020_MiniProject_AdityaV.ipynb  full analysis, DE1 to ML7, with explanations
reports/                 data profile, cleaning log, matrix report, figures, result tables
memo.pdf                 one-page memo to the stakeholder (J2)
slides.pptx              5-minute presentation (J3)
data/                    raw, interim and processed data (gitignored)
```

## How to run

The project was built and tested on **Google Colab, Python 3.13**.

1. Put `online_retail_II.xlsx` in Google Drive at `MyDrive/DS5020_project/`.
2. Open `notebooks/DS5020_MiniProject_AdityaV.ipynb` in Colab.
3. Run the **Setup** cell. It mounts Drive, enters the repo and pulls the latest version.
4. To rebuild the data from the raw file in one command:
   ```
   python build_matrix.py --raw /content/drive/MyDrive/DS5020_project/online_retail_II.xlsx
   ```
   Options: `--min-customers 10`, `--test-months 2`, `--value log1p|binary|count`, `--refresh-cache`.
5. To run the full analysis, click the final **Git** cell, then **Runtime → Run before**.

To run locally instead: `pip install -r requirements.txt`, place the raw file in `data/raw/`, run `python build_matrix.py`, then run the notebook (skip the Colab Setup cell).

All randomness is seeded (`random_state=42`) and matrix indices are sorted, so results are reproducible. The DE6 cell in the notebook checks that the script and the notebook produce an identical matrix.

## Pipeline summary

| Step | What happens | Key numbers |
|---|---|---|
| DE1 | Load both sheets, cast IDs to text, profile | 1,067,371 rows; 22.77% missing Customer ID |
| DE2 | Clean (strip codes, drop cancellations, guests, non-positive values, non-products, duplicates) | 776,577 rows kept (72.8%) |
| DE3 | Time split: test period from 2011-10-09; targets = new products for returning customers | 1,930 customers, 49,423 targets |
| DE4 | Sparse matrix from training data; products bought by at least 10 customers | 5,473 × 3,687 |
| DE5 | Matrix report | 97.95% sparse; 161.4 MB dense vs 5.0 MB sparse |
| ML2 | Reconstruction error for k = 10 to 200 | 0.790 at k = 50 (log1p); no clear elbow |
| ML3/ML4 | Recommendations and hit rate vs baseline | 40.0% vs 26.9% |
| ML5 | PCA vs SVD comparison | Tie on accuracy; SVD 3× faster |
| ML6 | Customer segments | 6 named segments |
| ML7 | Sensitivity to purchase value | 1/0 beats log1p (0.400 vs 0.378 at k = 50) |

## Final model

**Truncated SVD, k = 50, 1/0 purchase values, dot-product scoring.**

- Customer recommendations score products with the rebuilt matrix (dot product of SVD factors). Cosine similarity ignores popularity and lost to the baseline (23.6%).
- Product-page "also bought" lists use cosine similarity between product factor vectors.

## PCA vs Truncated SVD (ML5)

Same 1/0 matrix, same k = 50.

| Measure | PCA | Truncated SVD |
|---|---|---|
| k used | 50 | 50 |
| Fit time (s) | 3.3 | 1.1 |
| Peak memory during fit (MB)* | 8.3 | 9.7 |
| Size of stored factors (MB) | 3.69 | 3.66 |
| Variance explained / relative reconstruction error | 27.4% variance explained | 0.832 relative error |
| Hit rate at 10 | 0.404 | 0.400 |
| Best-seller baseline hit rate at 10 | 0.269 | 0.269 |
| What component 1 represents | Overall buying volume (corr. 0.98 with basket size) | Overall buying volume (corr. 0.98 with basket size) |

\* Measured with `tracemalloc`, which does not see memory used inside the ARPACK/LAPACK solvers, so both figures are underestimates.

**Conclusion:** both methods recommend equally well. SVD is kept because it is 3× faster, works directly on sparse data, and does not need a dense centred matrix at scoring time.

## Customer segments (ML6)

| Segment | Share | Median products per customer |
|---|---|---|
| Vintage Tea Party | 8% | 35 |
| Kids' Crafts & Games | 19% | 49 |
| Cosy Comfort Gifts | 23% | 23 |
| Party Supplies | 14% | 55 |
| Bags & Storage Bulk Buyers | 15% | 71 |
| Romantic Home Décor | 22% | 38 |

## Known limitations

- **Cold start:** 16% of recent customers are new and cannot be personalised.
- **Rare products:** 9.5% of next purchases are products below the 10-customer filter.
- **Seasonality:** the test period (Oct–Dec) is Christmas ordering season.
- **Overlapping segments:** silhouette score about 0.05.
- **Tuning on the test period:** k and the scoring method were chosen on the test period rather than on a separate validation period.
