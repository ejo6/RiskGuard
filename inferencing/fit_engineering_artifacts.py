import argparse
import os
from pathlib import Path

import pandas as pd

from feature_engineering import FeatureEngineer


def main():
    parser = argparse.ArgumentParser(description="Fit and save feature engineering artifacts from a training CSV.")
    parser.add_argument("train_csv", help="Path to the training transactions CSV (raw, pre-engineering).")
    parser.add_argument("--label-col", action="append", default=["isFraud"], help="Label column(s) to drop while fitting.")
    default_out = Path(__file__).resolve().parent / 'artifacts' / 'engineering_artifacts.json'
    parser.add_argument("--out", default=str(default_out), help="Output JSON path for artifacts.")
    args = parser.parse_args()

    df = pd.read_csv(args.train_csv)

    fe = FeatureEngineer()
    fe.fit(df, label_cols=args.label_col)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fe.save(str(out_path))
    print(f"Saved artifacts to: {out_path}")


if __name__ == "__main__":
    main()
