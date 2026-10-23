"""Leakage-aware preprocessing for archive/diabetes.csv.

The source CSV is never overwritten. Outputs are written to outputs/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


INPUT_PATH = Path("archive/diabetes.csv")
OUTPUT_DIR = Path("outputs")
TARGET = "Outcome"

# Values identified as class-conditional median imputations in the supplied CSV.
SUSPECT_VALUES = {
    "Insulin": [102.5, 169.5],
    "SkinThickness": [27, 32],
}
MISSING_FLAGS = {
    "Insulin": "Insulin_missing",  # Kept as requested by the user.
    "SkinThickness": "SkinThick_missing",
}


def iqr_cap(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Cap observed values outside [Q1 - 1.5 IQR, Q3 + 1.5 IQR].

    pandas quantiles skip NaNs, so for the columns reverted to missing the
    fences are fitted only on observed values. NaNs are intentionally kept
    missing here.
    """
    audit = []
    for column in columns:
        q1, q3 = data[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        low_count = (data[column] < lower).sum()
        high_count = (data[column] > upper).sum()
        data[column] = data[column].clip(lower=lower, upper=upper)
        audit.append(
            {
                "variable": column,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_fence": lower,
                "upper_fence": upper,
                "capped_low_count": low_count,
                "capped_high_count": high_count,
                "capped_total": low_count + high_count,
            }
        )
    return pd.DataFrame(audit)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    raw = pd.read_csv(INPUT_PATH)
    processed = raw.copy()

    # 1. Revert suspected label-derived values to missing and save their locations.
    replacement_audit = []
    for column, suspect_values in SUSPECT_VALUES.items():
        mask = processed[column].isin(suspect_values)
        flag = MISSING_FLAGS[column]
        processed[flag] = mask.astype("int8")
        processed.loc[mask, column] = np.nan
        replacement_audit.append(
            {
                "variable": column,
                "suspect_values_reverted_to_missing": ", ".join(map(str, suspect_values)),
                "reverted_count": int(mask.sum()),
                "missing_flag_column": flag,
            }
        )

    # 2. Fit IQR fences on observed values only, then cap their outliers.
    #    Target/indicator columns are never used or modified here.
    feature_columns = [column for column in raw.columns if column != TARGET]
    iqr_audit = iqr_cap(processed, feature_columns)



    # 3. Persist data and transparent audit tables.
    processed.to_csv(OUTPUT_DIR / "diabetes_preprocessed.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(replacement_audit).to_csv(
        OUTPUT_DIR / "diabetes_suspect_value_audit.csv", index=False, encoding="utf-8-sig"
    )
    
    iqr_audit.to_csv(OUTPUT_DIR / "diabetes_postprocess_iqr_audit.csv", index=False, encoding="utf-8-sig")

    # 4. Visual validation.
    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    chart_specs = [("Insulin", "Insulin_missing"), ("SkinThickness", "SkinThick_missing")]
    colors = {0: "#4C78A8", 1: "#E45756"}

    for row, (column, flag) in enumerate(chart_specs):
        before, after = axes[row]
        sns.boxplot(data=raw, x=TARGET, y=column, palette=colors, ax=before)
        before.set_title(f"Before: {column} by outcome")
        before.set_xlabel("Outcome")

        # fitted before imputation, so a newly fitted boxplot IQR is not the
        # criterion used for treatment.
        sns.boxplot(data=processed, x=TARGET, y=column, palette=colors,
                    showfliers=False, ax=after)
        after.set_title(f"After: {column} by outcome (IQR capped, missing kept as NaN)")
        after.set_xlabel("Outcome")

        # Write counts directly on the after chart for the new missingness feature.
        missing_count = int(processed[flag].sum())
        after.text(
            0.5, 0.97, f"{flag}=1: {missing_count} rows",
            transform=after.transAxes, ha="center", va="top", fontsize=10,
        )

    fig.suptitle("Validation of leakage-aware preprocessing", fontsize=16, fontweight="bold")
    fig.savefig(OUTPUT_DIR / "diabetes_preprocessing_validation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for ax, (column, flag) in zip(axes, chart_specs):
        counts = processed[flag].value_counts().sort_index()
        labels = [f"Observed\n({int(counts.get(0, 0))})", f"Missing(NaN)\n({int(counts.get(1, 0))})"]
        ax.bar(labels, [counts.get(0, 0), counts.get(1, 0)], color=["#4C78A8", "#E45756"])
        ax.set_title(f"{column}: missingness indicator")
        ax.set_ylabel("Rows")
    fig.savefig(OUTPUT_DIR / "diabetes_missingness_indicators.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("Completed preprocessing.")
    print(processed[list(SUSPECT_VALUES)].isna().sum())
    print(iqr_audit[["variable", "capped_total"]].to_string(index=False))


if __name__ == "__main__":
    main()
