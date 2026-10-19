import pandas as pd
import numpy as np

def _is_missing(val) -> bool:
    if pd.isna(val) or val is None:
        return True
    if isinstance(val, str) and val.strip() == "":
        return True
    return False

def audit(df: pd.DataFrame, *, target: str, id_col: str, time_col: str) -> dict:
    for col in [target, id_col, time_col]:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in dataframe")
            
    n_rows = len(df)
    missing = {}
    for col in df.columns:
        miss_count = df[col].apply(_is_missing).sum()
        missing[col] = float(miss_count / n_rows) if n_rows > 0 else 0.0
        
    non_missing_ids = df[id_col][~df[id_col].apply(_is_missing)]
    id_counts = non_missing_ids.value_counts()
    duplicate_ids = sorted(list(set(id_counts[id_counts > 1].index)))
    duplicate_rows = int(df.duplicated().sum())
    
    constant_columns = []
    for col in df.columns:
        if col == target:
            continue
        non_miss_vals = df[col][~df[col].apply(_is_missing)].unique()
        if len(non_miss_vals) <= 1:
            constant_columns.append(col)
            
    target_miss = df[target].apply(_is_missing)
    unlabelled_rows = int(target_miss.sum())
    
    labelled_mask = ~target_miss
    if labelled_mask.sum() == 0:
        positive_rate = 0.0
    else:
        labelled_targets = df.loc[labelled_mask, target]
        pos_count = ((labelled_targets == 1) | (labelled_targets == True)).sum()
        positive_rate = float(pos_count / labelled_mask.sum())
        
    non_miss_times = df[time_col][~df[time_col].apply(_is_missing)]
    bad_timestamps = 0
    parsed_times = []
    for val in non_miss_times:
        parsed = pd.to_datetime(val, errors="coerce")
        if pd.isna(parsed):
            bad_timestamps += 1
        else:
            parsed_times.append(parsed)
            
    if parsed_times:
        time_range = [min(parsed_times).isoformat(), max(parsed_times).isoformat()]
    else:
        time_range = [None, None]
        
    return {
        "n_rows": n_rows,
        "missing": missing,
        "duplicate_ids": duplicate_ids,
        "duplicate_rows": duplicate_rows,
        "constant_columns": sorted(constant_columns),
        "unlabelled_rows": unlabelled_rows,
        "positive_rate": positive_rate,
        "bad_timestamps": bad_timestamps,
        "time_range": time_range,
        "leakage_suspects": []
    }