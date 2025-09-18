import os
import sys
import argparse
from typing import Tuple, List

import numpy as np
import pandas as pd

from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput


ENGINEERED_FILES = {
    "lgbm_top25": "datasets/engineered_training_top25.csv",
    "lgbm_top50": "datasets/engineered_training_top50.csv",
    "lgbm_top100": "datasets/engineered_training_top100.csv",
}

FEATURE_COUNTS = {
    "lgbm_top25": 25,
    "lgbm_top50": 50,
    "lgbm_top100": 100,
}

SAMPLE_TXN_CSV = "Inference/inferencing/sample_transactions.csv"


def load_first_n_txn_ids(path: str, n_rows: int) -> List[str]:
    df = pd.read_csv(path, nrows=n_rows)
    col = None
    for candidate in ["TransactionID", "transaction_id", "id"]:
        if candidate in df.columns:
            col = candidate
            break
    if col is None:
        # Fall back to row indices as identifiers
        return [str(i) for i in range(len(df))]
    return df[col].astype(str).tolist()


def load_engineered_features(path: str, n_features: int, n_rows: int) -> Tuple[np.ndarray, List[str]]:
    if pd is None:
        raise RuntimeError("pandas is required to read CSV files; please install pandas.")

    df = pd.read_csv(path, nrows=n_rows)

    # Keep target if present for optional display, but exclude from features
    target_col = None
    for c in ["isFraud", "target", "label"]:
        if c in df.columns:
            target_col = c
            break

    targets = df[target_col].tolist() if target_col is not None else [None] * len(df)

    if target_col is not None:
        df = df.drop(columns=[target_col])

    # Coerce everything to numeric; non-numeric becomes NaN, then fill with 0
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    if df.shape[1] != n_features:
        # Try to trim or pad to expected feature size to match the model
        if df.shape[1] > n_features:
            df = df.iloc[:, :n_features]
        else:
            # Pad with zeros to the right if we have fewer columns
            pad_count = n_features - df.shape[1]
            for i in range(pad_count):
                df[f"_pad_{i}"] = 0.0

    x = df.to_numpy(dtype=np.float32, copy=False)
    if x.shape[1] != n_features:
        raise ValueError(
            f"Engineered features shape mismatch for {path}: got {x.shape[1]}, expected {n_features}"
        )
    return x, targets


def infer_batch(cli: InferenceServerClient, model: str, x: np.ndarray):
    inp = InferInput("float_input", x.shape, "FP32")
    inp.set_data_from_numpy(x, binary_data=True)

    outputs = [
        InferRequestedOutput("probabilities", binary_data=True),
        InferRequestedOutput("label", binary_data=True),
    ]

    return cli.infer(model_name=model, inputs=[inp], outputs=outputs)


def main():
    parser = argparse.ArgumentParser(description="Run engineered feature inference against Triton.")
    parser.add_argument("-n", "--rows", type=int, default=int(os.environ.get("ROWS", 10)),
                        help="Number of rows to read and infer (default: 10). Can also set ROWS env var.")
    args = parser.parse_args()

    triton_url = os.environ.get("TRITON_URL", "localhost:8000")
    cli = InferenceServerClient(url=triton_url)

    # Load identifiers for friendly output
    try:
        txn_ids = load_first_n_txn_ids(SAMPLE_TXN_CSV, args.rows)
    except Exception as e:
        print(f"Warning: failed to load transaction IDs from {SAMPLE_TXN_CSV}: {e}")
        txn_ids = [str(i) for i in range(args.rows)]

    for model, csv_path in ENGINEERED_FILES.items():
        n_features = FEATURE_COUNTS[model]
        print(f"\nModel: {model} | Features: {n_features} | File: {csv_path}")

        # Optional readiness check
        try:
            if hasattr(cli, "is_model_ready") and not cli.is_model_ready(model):
                print(f"  - Model {model} not ready on {triton_url}")
        except Exception:
            pass

        try:
            x, targets = load_engineered_features(csv_path, n_features, args.rows)
        except Exception as e:
            print(f"  - Failed to load features: {e}")
            continue

        try:
            result = infer_batch(cli, model, x)
        except Exception as e:
            print(f"  - Inference error: {e}")
            continue

        probs = result.as_numpy("probabilities")  # shape [N, 2]
        labels = result.as_numpy("label")
        if labels is None:
            labels = np.argmax(probs, axis=1)
        labels = np.array(labels).reshape(-1)

        # Print compact per-row summary
        print("  idx  txn_id            label   p0       p1       target")
        for i in range(min(len(txn_ids), probs.shape[0])):
            p0, p1 = probs[i, 0], probs[i, 1]
            tid = txn_ids[i] if i < len(txn_ids) else str(i)
            tgt = targets[i] if i < len(targets) else None
            print(f"  {i:>3}  {tid:<16}  {int(labels[i]):>5}  {p0:0.4f}  {p1:0.4f}  {tgt}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    
