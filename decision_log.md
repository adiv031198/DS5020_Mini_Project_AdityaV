# Decision Log (R1)

Every decision in the project, with its reason. Owner for all entries: **Aditya Vishwanath** (individual project).

| # | Step | Decision | Reason / evidence |
|---|---|---|---|
| 1 | Setup | Use Google Colab; raw data in Google Drive, not in Git | Brief says raw data must not be committed; Colab gives a consistent environment |
| 2 | ML1 | Main method: Truncated SVD (decided before fitting) | The matrix is very sparse; SVD works on it directly, while PCA's centring turns zeros into negatives and breaks sparsity |
| 3 | DE1 | Cache the raw Excel file as parquet | Excel takes minutes to load; parquet takes seconds |
| 4 | DE1 | Cast Invoice, StockCode, Description, Country to text | IDs contain letters (C = cancellation, suffix = variant); saving to parquet failed with mixed types |
| 5 | DE1 | Use StockCode, not Description, as the product ID | 5,305 codes but 5,698 descriptions: some codes have several names |
| 6 | DE2 | Strip whitespace from StockCode before cleaning | Found '47503J' and '47503J ' as separate codes for one product; without stripping, its rows were dropped and split |
| 7 | DE2 | Remove cancellations first | So later rules don't double-count them (19,494 rows) |
| 8 | DE2 | Remove lines without Customer ID | Cannot be placed in a customer × product matrix (242,257 rows, 22.77% of raw data) |
| 9 | DE2 | Remove Quantity ≤ 0 | Returns, not purchases. Removed 0 rows: all negatives were already cancellations or guest lines |
| 10 | DE2 | Remove Price ≤ 0 | Accounting adjustments, not purchases (71 rows) |
| 11 | DE2 | Keep only StockCodes of 5 digits + optional letters | Others are postage, fees, discounts, vouchers, test codes (2,917 rows). Verified by reviewing every removed code |
| 12 | DE2 | DCGS codes need no separate rule | 0 DCGS rows have a Customer ID, so they are already removed by decision 8 |
| 13 | DE2 | SP1002 dropped | Ambiguous, but only 3 rows, and it would fail the 10-customer filter anyway |
| 14 | DE2 | Remove exact duplicate rows | Likely double entries (26,055 rows). Result: 776,577 rows, 5,852 customers, 4,619 products |
| 15 | DE3 | Test period = last 2 months (from 2011-10-09) | Required by the brief; time-based so the model never sees the future (R2) |
| 16 | DE3 | Targets = products bought in test but never in training, for returning customers | The brief asks to predict new purchases; re-orders don't need a recommender |
| 17 | DE3 | Drop 378 test-only customers | No history to personalise from (cold start; reported as a risk) |
| 18 | DE3 | Exclude 112 returning customers with no new products | Nothing to predict for them |
| 19 | DE4 | Aggregate quantity per customer–product before transforming | Otherwise repeat purchases add several log values together |
| 20 | DE4 | Stored purchase value: log1p(total quantity) | Limits the effect of huge orders while keeping some volume information |
| 21 | DE4 | Keep products bought by at least 10 customers | Too few buyers to place reliably. 3,687 of 4,529 kept; 90.5% of test targets remain |
| 22 | DE4 | Sort row and column indices | Makes the matrix reproducible |
| 23 | DE4 | Index maps saved inside data/ only | They contain customer IDs (R4) |
| 24 | ML2 | Do not choose k from the error curve alone | Error falls smoothly with no elbow (0.876 at k=10 → 0.639 at k=200); hit rate decides |
| 25 | ML4 | Baseline = 10 best sellers by number of buyers, excluding already-bought items | Required by the brief (R3); excluding owned items keeps the comparison fair |
| 26 | ML4 | Customer scoring uses the dot product of SVD factors, not cosine | Cosine ignores popularity and lost to the baseline (0.236 vs 0.269); dot product scores 0.400 |
| 27 | ML7 | Model converts the matrix to 1/0 values | 1/0 beats log1p at k = 50 (0.400 vs 0.378); the matrix file is unchanged |
| 28 | ML4 | k = 50 | Within noise of k = 100 (0.403; standard error ≈ 0.011) at half the storage |
| 29 | ML3 | Product-page lists use cosine similarity of product factors | Required by the brief; sanity check shows coherent lists (lighting, Regency tea set, cake cases) |
| 30 | ML4 | Nightly job processed in chunks of 1,000 | Avoids ever building the full 161 MB score table; full job runs in 3.5 s |
| 31 | ML5 | PCA scoring adds the mean back after reconstruction | Fair equivalent of SVD's dot-product scoring |
| 32 | ML5 | Keep SVD after comparison | Equal hit rate (0.400 vs 0.404), 3× faster fit, sparse-native, no dense matrix at scoring |
| 33 | ML5 | Memory figures reported with a caveat | tracemalloc misses memory inside ARPACK/LAPACK |
| 34 | ML6 | Normalise customer factors to length 1 before clustering | Component 1 is buying volume; normalising keeps what customers buy, not how much |
| 35 | ML6 | Cluster on all 50 dimensions; 2-D projection for display only | More accurate than clustering on 2 components |
| 36 | ML6 | 6 segments | Silhouette 0.049, tied with 5; 6 gives distinct, nameable groups |
| 37 | ML6 | Name segments by product lift | Without lift every group's top products are the same best sellers |
| 38 | ML6 | Interpret segments as "dominant buying theme" | Low silhouette (≈0.05) shows heavy overlap between groups |
| 39 | DE6 | Package DE1–DE5 as build_matrix.py | One command rebuilds everything (R6); verified identical to the notebook matrix |
| 40 | Limitation | k and scoring chosen on the test period | Time constraints; a separate validation period would be the clean approach |
