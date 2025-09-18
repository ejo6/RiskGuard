import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput

from feature_engineering import FeatureEngineer, load_top_columns_from_csv


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    # In the new layout this file lives at <repo>/inferencing/engineer_and_infer.py
    if here.parent.name == 'inferencing':
        return here.parent.parent  # repo root
    # In the old layout it might live at <repo>/Inference/inferencing/engineer_and_infer.py
    if here.parent.name == 'inferencing' and here.parent.parent.name == 'Inference':
        return here.parent.parent.parent
    # Fallback: two levels up
    return here.parents[2]


def _engineered_top_csv_map() -> dict:
    root = _repo_root()
    candidates = [
        root / 'datasets',              # new structure
        root / 'Inference' / 'datasets' # old structure
    ]
    for base in candidates:
        if (base / 'engineered_training_top25.csv').exists():
            return {
                'lgbm_top25': str(base / 'engineered_training_top25.csv'),
                'lgbm_top50': str(base / 'engineered_training_top50.csv'),
                'lgbm_top100': str(base / 'engineered_training_top100.csv'),
            }
    # default to new structure paths (may be overridden by --top-csv)
    base = _repo_root() / 'datasets'
    return {
        'lgbm_top25': str(base / 'engineered_training_top25.csv'),
        'lgbm_top50': str(base / 'engineered_training_top50.csv'),
        'lgbm_top100': str(base / 'engineered_training_top100.csv'),
    }


def get_top_columns(model: str, override_csv: str = None) -> List[str]:
    csv_map = _engineered_top_csv_map()
    csv_path = override_csv or csv_map.get(model)
    if not csv_path:
        raise ValueError(f"No top-columns CSV known for model {model}. Provide --top-csv.")
    return load_top_columns_from_csv(csv_path)


def infer(cli: InferenceServerClient, model: str, x: np.ndarray):
    inp = InferInput("float_input", x.shape, "FP32")
    inp.set_data_from_numpy(x, binary_data=True)
    outputs = [
        InferRequestedOutput("probabilities", binary_data=True),
        InferRequestedOutput("label", binary_data=True),
    ]
    return cli.infer(model_name=model, inputs=[inp], outputs=outputs)


def main():
    parser = argparse.ArgumentParser(description="Engineer features from raw transactions and run inference on Triton.")
    # Default input attempts new then old structure
    default_input_new = _repo_root() / 'datasets' / 'merged_transactions.csv'
    default_input_old = _repo_root() / 'Inference' / 'datasets' / 'merged_transactions.csv'
    default_input = str(default_input_new if default_input_new.exists() else default_input_old)
    parser.add_argument("--input", default=default_input, help="Raw transactions CSV path.")
    parser.add_argument("-n", "--rows", type=int, default=int(os.environ.get("ROWS", 10)), help="Rows to process.")
    parser.add_argument("--model", choices=["lgbm_top25", "lgbm_top50", "lgbm_top100", "all"], default="all", help="Model to use.")
    parser.add_argument("--top-csv", default=None, help="Optional override CSV to read top-column order from (inferred by model otherwise).")
    default_art_new = _repo_root() / 'inferencing' / 'artifacts' / 'engineering_artifacts.json'
    default_art_old = _repo_root() / 'Inference' / 'inferencing' / 'artifacts' / 'engineering_artifacts.json'
    default_art = str(default_art_new if default_art_new.exists() else default_art_old)
    parser.add_argument("--artifacts", default=default_art, help="Path to saved engineering artifacts JSON.")
    parser.add_argument("--triton-url", default=os.environ.get("TRITON_URL", "localhost:8000"), help="Triton HTTP endpoint host:port")
    args = parser.parse_args()

    # Load raw data
    raw = pd.read_csv(args.input, nrows=args.rows)

    # Load artifacts and engineer features
    fe = FeatureEngineer.load(args.artifacts)

    models = [args.model] if args.model != "all" else ["lgbm_top25", "lgbm_top50", "lgbm_top100"]

    cli = InferenceServerClient(url=args.triton_url)

    # Transaction IDs for printout
    txn_ids = None
    for candidate in ["TransactionID", "transaction_id", "id"]:
        if candidate in raw.columns:
            txn_ids = raw[candidate].astype(str).tolist()
            break
    if txn_ids is None:
        txn_ids = [str(i) for i in range(len(raw))]

    for model in models:
        top_cols = get_top_columns(model, args.top_csv)
        feats_df = fe.transform(raw, top_cols)
        x = feats_df.to_numpy(dtype=np.float32)

        try:
            res = infer(cli, model, x)
        except Exception as e:
            print(f"Model {model} inference error: {e}")
            continue

        probs = res.as_numpy("probabilities")
        labels = res.as_numpy("label")
        if labels is None:
            labels = np.argmax(probs, axis=1)
        labels = np.array(labels).reshape(-1)

        print(f"\nModel: {model}  rows: {len(x)}  features: {x.shape[1]}")
        print("  idx  txn_id            label   p0       p1")
        for i in range(min(len(txn_ids), probs.shape[0])):
            p0, p1 = probs[i, 0], probs[i, 1]
            tid = txn_ids[i]
            print(f"  {i:>3}  {tid:<16}  {int(labels[i]):>5}  {p0:0.4f}  {p1:0.4f}")


if __name__ == "__main__":
    main()
