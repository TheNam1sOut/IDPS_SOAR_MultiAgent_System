import pandas as pd
import numpy as np


def load_dataset(path):
    print("[+] Loading dataset...")
    df = pd.read_csv(path)
    print(f"[+] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns\n")
    return df


def check_basic_info(df):
    print("========== BASIC INFO ==========")
    print(df.info())
    print("\nData types distribution:")
    print(df.dtypes.value_counts())
    print()


def find_duplicate_by_value(df, sample_size=50000):
    print("========== DUPLICATE COLUMNS BY VALUE ==========")
    cols = df.columns
    checked = set()

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c1, c2 = cols[i], cols[j]
            if (c1, c2) in checked:
                continue

            # So sánh trên sample để đỡ tốn RAM
            if df[c1].head(sample_size).equals(df[c2].head(sample_size)):
                print(f" - {c1}  <==>  {c2}")
                checked.add((c1, c2))

    return len(checked)



def find_zero_variance_columns(df):
    print("========== ZERO VARIANCE COLUMNS ==========")
    zero_var_cols = [col for col in df.columns if df[col].nunique() == 1]

    if len(zero_var_cols) == 0:
        print("No zero-variance columns found.\n")
    else:
        print(f"Found {len(zero_var_cols)} zero-variance columns:")
        for col in zero_var_cols:
            print(f" - {col}")
        print()

    return zero_var_cols


def check_nan_values(df):
    print("========== NaN VALUES ==========")
    nan_count = df.isna().sum()
    nan_cols = nan_count[nan_count > 0]

    if nan_cols.empty:
        print("No NaN values found.\n")
    else:
        print("Columns containing NaN:")
        for col, cnt in nan_cols.items():
            print(f" - {col}: {cnt} ({cnt/len(df)*100:.4f}%)")
        print()

    return nan_cols.index.tolist()


def check_infinite_values(df):
    print("========== INFINITE VALUES ==========")
    inf_cols = []

    for col in df.select_dtypes(include=[np.number]).columns:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            inf_cols.append(col)
            print(f" - {col}: {inf_count} ({inf_count/len(df)*100:.4f}%)")

    if not inf_cols:
        print("No infinite values found.")

    print()
    return inf_cols


def feature_scale_analysis(df):
    print("========== FEATURE SCALE ANALYSIS ==========")
    desc = df.describe().T
    desc["range"] = desc["max"] - desc["min"]

    print("Top 10 features with largest value range:")
    print(desc.sort_values("range", ascending=False).head(10))
    print()

    print("Top 10 features with smallest value range:")
    print(desc.sort_values("range", ascending=True).head(10))
    print()


def label_distribution(df, label_col="Label"):
    print("========== LABEL DISTRIBUTION ==========")
    if label_col not in df.columns:
        print("Label column not found.\n")
        return

    counts = df[label_col].value_counts()
    percentages = counts / len(df) * 100

    for label, count in counts.items():
        print(f"Label {label:>5}: {count:8d} ({percentages[label]:6.2f}%)")

    print()



def main():
    DATASET_PATH = "./data/preprocessed/train_processed.csv" 

    df = load_dataset(DATASET_PATH)

    check_basic_info(df)

    dup_cols = find_duplicate_by_value(df)
    zero_var_cols = find_zero_variance_columns(df)

    nan_cols = check_nan_values(df)
    inf_cols = check_infinite_values(df)

    feature_scale_analysis(df)

    label_distribution(df)

    print("========== AUDIT SUMMARY ==========")
    print(f"Duplicate columns     : {dup_cols}")
    print(f"Zero-variance columns : {len(zero_var_cols)}")
    print(f"Columns with NaN      : {len(nan_cols)}")
    print(f"Columns with Infinity : {len(inf_cols)}")
    print("\nAudit completed.")


if __name__ == "__main__":
    main()
