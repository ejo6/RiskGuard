import json
import math
import os
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

import numpy as np


def _nan_key(x: Any) -> str:
    if x is None:
        return "<NONE>"
    try:
        if isinstance(x, float) and math.isnan(x):
            return "<NaN>"
    except Exception:
        pass
    return str(x)


class FeatureEngineer:
    """
    Encapsulates the feature engineering steps from the engineering notebook
    and provides fit/transform/save/load methods for production use.

    Artifacts stored:
    - freq_maps: value->count for ['card1','addr1','P_emaildomain']
    - card1_stats: per-card1 TransactionAmt mean/std plus global fallbacks
    - top_devices: list of popular DeviceInfo prefixes (after cleaning)
    - label_maps: category->int mapping for selected categorical columns
    """

    def __init__(self, artifacts: Optional[Dict[str, Any]] = None):
        self.artifacts: Dict[str, Any] = artifacts or {}

    # -------------------------------------------
    # Fitting from training data stufff
    # -------------------------------------------
    def fit(self, df: 'pd.DataFrame', label_cols: Optional[List[str]] = None, top_devices_k: int = 20) -> None:

        df = df.copy()
        if label_cols:
            for c in label_cols:
                if c in df.columns:
                    df.drop(columns=[c], inplace=True)

        # Clean up device info to derive top device prefixes
        dev_clean = df.get('DeviceInfo')
        if dev_clean is not None:
            dev_clean = dev_clean.astype(str).str.extract(r'(^[A-Za-z]+)', expand=False).fillna('unknown')
            top_devices = dev_clean.value_counts().index[:top_devices_k].tolist()
        else:
            top_devices = []

        # Frequency maps
        freq_maps: Dict[str, Dict[str, int]] = {}
        for col in ['card1', 'addr1', 'P_emaildomain']:
            if col in df.columns:
                vc = df[col].value_counts(dropna=False)
                freq_maps[col] = { _nan_key(k): int(v) for k, v in vc.items() }

        # Group stats for TransactionAmt by card1
        card1_stats: Dict[str, Dict[str, float]] = {}
        global_mean = float(pd.to_numeric(df.get('TransactionAmt'), errors='coerce').mean()) if 'TransactionAmt' in df.columns else 0.0
        global_std = float(pd.to_numeric(df.get('TransactionAmt'), errors='coerce').std()) if 'TransactionAmt' in df.columns else 0.0
        if 'card1' in df.columns and 'TransactionAmt' in df.columns:
            tmp = df[['card1', 'TransactionAmt']].copy()
            tmp['TransactionAmt'] = pd.to_numeric(tmp['TransactionAmt'], errors='coerce')
            grp = tmp.groupby('card1')['TransactionAmt'].agg(['mean', 'std']).reset_index()
            for _, row in grp.iterrows():
                c = _nan_key(row['card1'])
                mean = float(row['mean']) if not math.isnan(row['mean']) else global_mean
                std = float(row['std']) if not math.isnan(row['std']) else 0.0
                card1_stats[c] = {'mean': mean, 'std': std}

        # Label maps for categorical columns likely to be selected
        label_maps: Dict[str, Dict[str, int]] = {}
        candidate_cats = ['DeviceInfo', 'DeviceType', 'P_emaildomain', 'id_31']
        for col in candidate_cats:
            if col in df.columns:
                series = df[col].astype(str)
                cats = series.dropna().unique().tolist()
                # Assign 1..N; reserve 0 for unknown
                label_maps[col] = { str(v): i+1 for i, v in enumerate(cats) }

        self.artifacts = {
            'freq_maps': freq_maps,
            'card1_stats': card1_stats,
            'global_amt_mean': global_mean,
            'global_amt_std': global_std,
            'top_devices': top_devices,
            'label_maps': label_maps,
            'version': 1,
        }

    # -------------------------------------------
    # Persistence
    # -------------------------------------------
    def save(self, path: str) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.artifacts, f)

    @classmethod
    def load(cls, path: str) -> 'FeatureEngineer':
        with open(path, 'r', encoding='utf-8') as f:
            artifacts = json.load(f)
        return cls(artifacts)

    # -------------------------------------------
    # Transformation
    # -------------------------------------------
    def _map_freq(self, series: 'pd.Series', col: str) -> 'pd.Series':
        fmap = self.artifacts.get('freq_maps', {}).get(col, {})
        return series.map(lambda v: fmap.get(_nan_key(v), 0)).astype(np.float32)

    def _card1_stats(self, series: 'pd.Series', which: str) -> 'pd.Series':
        stats = self.artifacts.get('card1_stats', {})
        gmean = self.artifacts.get('global_amt_mean', 0.0)
        gstd = self.artifacts.get('global_amt_std', 0.0)
        def getter(v):
            d = stats.get(_nan_key(v))
            if d is None:
                return gmean if which == 'mean' else gstd
            return float(d.get(which, gmean if which == 'mean' else gstd))
        return series.map(getter).astype(np.float32)

    def _label_encode(self, series: 'pd.Series', col: str) -> 'pd.Series':
        m = self.artifacts.get('label_maps', {}).get(col, {})
        return series.astype(str).map(lambda v: m.get(str(v), 0)).astype(np.float32)

    def transform(self, raw_df: 'pd.DataFrame', top_columns: List[str]) -> 'pd.DataFrame':
        df = raw_df.copy()

        # Derived time features
        if 'TransactionDT' in df.columns:
            dt = pd.to_numeric(df['TransactionDT'], errors='coerce')
            df['Transaction_day'] = (dt // (3600 * 24)).astype('Int64')
            df['Transaction_hour'] = ((dt // 3600) % 24).astype('Int64')
            df['is_night'] = (df['Transaction_hour'] < 6).astype(np.int32)

        # Log amount
        if 'TransactionAmt' in df.columns:
            df['TransactionAmt_log'] = np.log1p(pd.to_numeric(df['TransactionAmt'], errors='coerce')).astype(np.float32)

        # Email domain features
        if 'P_emaildomain' in df.columns:
            ped = df['P_emaildomain'].astype(str)
            df['P_emaildomain_suffix'] = ped.str.split('.').str[-1]
            df['P_emaildomain_bin'] = ped.isin(['gmail.com', 'yahoo.com', 'hotmail.com']).astype(np.int32)

        # Device features
        if 'DeviceType' in df.columns:
            df['DeviceType'] = df['DeviceType'].fillna('unknown')
        if 'DeviceInfo' in df.columns:
            df['DeviceInfo_clean'] = df['DeviceInfo'].astype(str).str.extract(r'(^[A-Za-z]+)', expand=False).fillna('unknown')
            top_devices = set(self.artifacts.get('top_devices', []))
            if top_devices:
                df['DeviceInfo_clean'] = df['DeviceInfo_clean'].where(df['DeviceInfo_clean'].isin(top_devices), 'Other')

        # Missing flags
        if 'id_02' in df.columns:
            df['id_02_missing'] = df['id_02'].isnull().astype(np.int32)
        if 'dist1' in df.columns:
            df['dist1_missing'] = df['dist1'].isnull().astype(np.int32)

        # Frequency encodings
        if 'card1' in df.columns:
            df['card1_freq'] = self._map_freq(df['card1'], 'card1')
            # Per-card stats
            df['card1_amt_mean_x'] = self._card1_stats(df['card1'], 'mean')
            df['card1_amt_std_x'] = self._card1_stats(df['card1'], 'std')
            # Provide "_y" aliases if requested later
            df['card1_amt_mean_y'] = df['card1_amt_mean_x']
            df['card1_amt_std_y'] = df['card1_amt_std_x']
        if 'addr1' in df.columns:
            df['addr1_freq'] = self._map_freq(df['addr1'], 'addr1')
        if 'P_emaildomain' in df.columns:
            df['P_emaildomain_freq'] = self._map_freq(df['P_emaildomain'], 'P_emaildomain')

        # Label-encode selected categorical columns if present in top_columns
        for col in ['DeviceInfo', 'DeviceType', 'P_emaildomain', 'id_31']:
            if col in df.columns and col in top_columns:
                df[col] = self._label_encode(df[col], col)

        # Assemble the output frame in the exact top_columns order
        out = pd.DataFrame(index=df.index)
        for col in top_columns:
            if col in df.columns:
                series = pd.to_numeric(df[col], errors='coerce')
                out[col] = series.fillna(0.0).astype(np.float32)
            else:
                # Column not produced; fill zeros
                out[col] = np.zeros(len(df), dtype=np.float32)

        return out


def load_top_columns_from_csv(path: str) -> List[str]:
    """Read header of engineered training CSV and return feature column list excluding label columns."""
    # Read only header row via pandas by reading 0 rows
    df = pd.read_csv(path, nrows=0)
    cols = list(df.columns)
    for tgt in ['isFraud', 'target', 'label']:
        if tgt in cols:
            cols.remove(tgt)
    return cols

