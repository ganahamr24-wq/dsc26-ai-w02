import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def audit(df: pd.DataFrame, *, target: str, id_col: str, time_col: str) -> dict:
    # 1. Validation for mandatory columns
    for col in [target, id_col, time_col]:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    n_rows = len(df)

    # Helper function to check if a value is missing (NaN, None, empty or whitespace-only string)
    def is_missing_val(val):
        if pd.isna(val):
            return True
        if isinstance(val, str) and val.strip() == "":
            return True
        return False

    # 2. Missingness per column
    missing_mask = df.map(is_missing_val)
    missing = {col: float(missing_mask[col].mean()) for col in df.columns}

    # 3. Duplicate IDs
    non_missing_ids = df[~missing_mask[id_col]][id_col]
    dup_ids = non_missing_ids[non_missing_ids.duplicated()].unique()
    duplicate_ids = sorted(list(dup_ids))

    # 4. Duplicate Rows
    duplicate_rows = int(df.duplicated(keep="first").sum())

    # 5. Constant Columns (ignoring target, ignoring missing values)
    constant_columns = []
    for col in df.columns:
        if col == target:
            continue
        non_missing_vals = df[col][~missing_mask[col]]
        if non_missing_vals.nunique() <= 1:
            constant_columns.append(col)
    constant_columns = sorted(constant_columns)

    # 6. Unlabelled rows & Positive rate
    target_missing = missing_mask[target]
    unlabelled_rows = int(target_missing.sum())

    labelled_df = df[~target_missing]
    if len(labelled_df) == 0:
        positive_rate = 0.0
    else:
        # Convert target to integer/float 1 or 0
        pos_count = (labelled_df[target] == 1).sum()
        positive_rate = float(pos_count / len(labelled_df))

    # 7. Timestamps and Time range
    parsed_times = pd.to_datetime(df[time_col], errors="coerce")
    bad_timestamps = int((missing_mask[time_col] == False) & (parsed_times.isna()))

    valid_times = parsed_times.dropna()
    if len(valid_times) == 0:
        time_range = [None, None]
    else:
        min_time = valid_times.min().isoformat()
        max_time = valid_times.max().isoformat()
        time_range = [min_time, max_time]

    # 8. Leakage Suspects
    leakage_suspects = set()

    # Calculate only on labelled rows
    if len(labelled_df) >= 20:
        labels = labelled_df[target].astype(int)
        unique_labels = labels.unique()

        # Needs both classes present
        if len(unique_labels) == 2:
            cols_to_check = [c for c in df.columns if c not in [target, id_col, time_col]]

            for col in cols_to_check:
                # Rule: A column with a single distinct value (including missing as a value) is never a suspect
                if df[col].nunique(dropna=False) <= 1:
                    continue

                col_labelled = labelled_df[col]
                col_missing = missing_mask.loc[labelled_df.index, col]

                # Check (a): Numeric or boolean value ROC AUC
                is_num_or_bool = pd.api.types.is_numeric_dtype(col_labelled) or pd.api.types.is_bool_dtype(col_labelled)
                if is_num_or_bool:
                    non_missing_mask = ~col_missing
                    val_labelled = col_labelled[non_missing_mask]
                    lbl_for_val = labels[non_missing_mask]

                    if len(val_labelled) >= 20 and len(lbl_for_val.unique()) == 2:
                        try:
                            score_a = roc_auc_score(lbl_for_val, val_labelled.astype(float))
                            if score_a >= 0.98 or score_a <= 0.02:
                                leakage_suspects.add(col)
                        except Exception:
                            pass

                # Check (b): Missingness indicator ROC AUC
                missing_indicator = col_missing.astype(int)
                try:
                    score_b = roc_auc_score(labels, missing_indicator)
                    if score_b >= 0.98 or score_b <= 0.02:
                        leakage_suspects.add(col)
                except Exception:
                    pass

    return {
        "n_rows": n_rows,
        "missing": missing,
        "duplicate_ids": duplicate_ids,
        "duplicate_rows": duplicate_rows,
        "constant_columns": constant_columns,
        "unlabelled_rows": unlabelled_rows,
        "positive_rate": positive_rate,
        "bad_timestamps": bad_timestamps,
        "time_range": time_range,
        "leakage_suspects": sorted(list(leakage_suspects)),
    }