"""
build_matrix.py - DE6: rebuild everything from the raw file in one command (R6).

Usage (from the repo root):
    python build_matrix.py --raw /content/drive/MyDrive/DS5020_project/online_retail_II.xlsx

Steps:
    DE1  load both sheets, cast IDs to text, data profile
    DE2  cleaning rules with a cleaning log
    DE3  time-based split (last 2 months = test), held-out new-product targets
    DE4  sparse customer x product matrix from training data only
    DE5  matrix report

Every step is deterministic (sorted indices, no randomness), so rerunning gives the same matrix.
"""
import argparse
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp

ID_COLS = ["Invoice", "StockCode", "Description", "Country"]
PRODUCT_CODE = r"\d{5}[A-Za-z]*"


def parse_args():
    p = argparse.ArgumentParser(description="Rebuild the customer x product matrix from the raw file.")
    p.add_argument("--raw", default="data/raw/online_retail_II.xlsx", help="path to online_retail_II.xlsx")
    p.add_argument("--min-customers", type=int, default=10, help="keep products bought by at least this many customers")
    p.add_argument("--test-months", type=int, default=2, help="length of the test period in months")
    p.add_argument("--value", choices=["log1p", "binary", "count"], default="log1p",
                   help="purchase value stored in the matrix")
    p.add_argument("--refresh-cache", action="store_true", help="re-read the Excel file even if a parquet cache exists")
    return p.parse_args()


def load_raw(raw_path, refresh):
    """DE1: load both sheets (or the parquet cache) and cast ID columns to text."""
    cache = "data/interim/raw.parquet"
    if os.path.exists(cache) and not refresh:
        print(f"[DE1] loading cache {cache}")
        df = pd.read_parquet(cache)
    else:
        print(f"[DE1] reading Excel {raw_path} (a few minutes)")
        sheets = pd.read_excel(raw_path, sheet_name=None)
        for name, s in sheets.items():
            print(f"      sheet {name}: {s.shape}")
        df = pd.concat(sheets.values(), ignore_index=True)
        for col in ID_COLS:
            df[col] = df[col].astype("string")
        df.to_parquet(cache)
    print(f"[DE1] {len(df):,} rows, columns {df.columns.tolist()}")

    profile = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "missing_%": (df.isna().mean() * 100).round(2),
        "unique": df.nunique(),
    })
    profile.to_csv("reports/data_profile.csv")
    return df


def clean(df):
    """DE2: apply the cleaning rules in order and log rows removed by each."""
    df = df.copy()
    df["StockCode"] = df["StockCode"].str.strip()  # '47503J ' vs '47503J'
    log = [("raw data", len(df), 0)]

    def apply_rule(df, keep_mask, name):
        keep_mask = keep_mask.fillna(False)
        removed = int((~keep_mask).sum())
        df = df[keep_mask]
        log.append((name, len(df), removed))
        return df

    df = apply_rule(df, ~df["Invoice"].str.startswith("C"), "remove cancellations (Invoice starts with C)")
    df = apply_rule(df, df["Customer ID"].notna(), "remove lines without Customer ID")
    df = apply_rule(df, df["Quantity"] > 0, "remove non-positive quantity")
    df = apply_rule(df, df["Price"] > 0, "remove non-positive price")
    df = apply_rule(df, df["StockCode"].str.fullmatch(PRODUCT_CODE), "remove non-product codes")
    df = apply_rule(df, ~df.duplicated(), "remove exact duplicate rows")
    df["Customer ID"] = df["Customer ID"].astype(int)

    cleaning_log = pd.DataFrame(log, columns=["step", "rows_after", "rows_removed"])
    cleaning_log.to_csv("reports/cleaning_log.csv", index=False)
    df.to_parquet("data/interim/clean.parquet")
    print("[DE2] cleaning log:")
    print(cleaning_log.to_string(index=False))
    return df


def split(df, test_months):
    """DE3: time-based split; targets = products a returning customer bought in test but never in train."""
    cutoff = df["InvoiceDate"].max() - pd.DateOffset(months=test_months)
    train = df[df["InvoiceDate"] < cutoff]
    test = df[df["InvoiceDate"] >= cutoff]

    train_pairs = train[["Customer ID", "StockCode"]].drop_duplicates()
    test_pairs = test[["Customer ID", "StockCode"]].drop_duplicates()
    test_pairs = test_pairs[test_pairs["Customer ID"].isin(train["Customer ID"])]
    targets = test_pairs.merge(train_pairs, how="left", indicator=True)
    targets = targets[targets["_merge"] == "left_only"].drop(columns="_merge")

    train.to_parquet("data/processed/train.parquet")
    targets.to_parquet("data/processed/test_targets.parquet")
    print(f"[DE3] cutoff {cutoff} | train {len(train):,} rows | test {len(test):,} rows")
    print(f"      test customers {test['Customer ID'].nunique():,}, "
          f"returning {test_pairs['Customer ID'].nunique():,}, "
          f"with new products {targets['Customer ID'].nunique():,}, target pairs {len(targets):,}")
    return train, targets


def build_matrix(train, targets, min_customers, value):
    """DE4: sparse customer x product matrix from training rows only."""
    agg = train.groupby(["Customer ID", "StockCode"], as_index=False)["Quantity"].sum()  # aggregate before the log

    n_buyers = agg.groupby("StockCode")["Customer ID"].nunique()
    keep_products = n_buyers[n_buyers >= min_customers].index
    agg = agg[agg["StockCode"].isin(keep_products)]

    customers = np.sort(agg["Customer ID"].unique())
    products = np.sort(agg["StockCode"].unique())
    cust_idx = {c: i for i, c in enumerate(customers)}
    prod_idx = {p: j for j, p in enumerate(products)}

    rows = agg["Customer ID"].map(cust_idx).values
    cols = agg["StockCode"].map(prod_idx).values
    q = agg["Quantity"].values.astype(float)
    vals = {"log1p": np.log1p(q), "binary": np.ones_like(q), "count": q}[value]
    A = sp.csr_matrix((vals, (rows, cols)), shape=(len(customers), len(products)))

    before = len(targets)
    targets = targets[targets["Customer ID"].isin(cust_idx) & targets["StockCode"].isin(prod_idx)]

    sp.save_npz("data/processed/A_train.npz", A)
    pd.Series(customers, name="Customer ID").to_csv("data/processed/customer_index.csv", index_label="row")
    pd.Series(products, name="StockCode").to_csv("data/processed/product_index.csv", index_label="col")
    targets.to_parquet("data/processed/test_targets_filtered.parquet")

    print(f"[DE4] products {n_buyers.size:,} -> kept {len(products):,} (min {min_customers} customers), "
          f"value = {value}")
    print(f"      targets kept {len(targets):,} of {before:,} ({len(targets) / before:.1%}), "
          f"eval customers {targets['Customer ID'].nunique():,}")
    return A


def matrix_report(A):
    """DE5: shape, non-zeros, sparsity, memory dense vs sparse."""
    n_rows, n_cols = A.shape
    dense_mb = n_rows * n_cols * 8 / 1e6
    sparse_mb = (A.data.nbytes + A.indices.nbytes + A.indptr.nbytes) / 1e6
    report = pd.DataFrame({
        "measure": ["customers (rows)", "products (cols)", "non-zeros", "sparsity %",
                    "dense memory (MB)", "sparse memory (MB)"],
        "value": [n_rows, n_cols, A.nnz, round(100 * (1 - A.nnz / (n_rows * n_cols)), 2),
                  round(dense_mb, 1), round(sparse_mb, 1)],
    })
    report.to_csv("reports/matrix_report.csv", index=False)
    print("[DE5] matrix report:")
    print(report.to_string(index=False))


def main():
    args = parse_args()
    for d in ["data/interim", "data/processed", "reports"]:
        os.makedirs(d, exist_ok=True)

    df = load_raw(args.raw, args.refresh_cache)
    df = clean(df)
    train, targets = split(df, args.test_months)
    A = build_matrix(train, targets, args.min_customers, args.value)
    matrix_report(A)
    print("\nDone. Outputs in data/interim/, data/processed/ and reports/.")


if __name__ == "__main__":
    main()
