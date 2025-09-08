import os
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

input_csv = '/data5/shuyanz/oppo_data/wide_list_changjiang_zhujiang.csv'
image_dir = '/data5/shuyanz/oppo_data/second_and_third'
output_dir = '/data5/shuyanz/oppo_data/split_dataset'

task_type = 'multilabel'

label_cols = ["bright", "dark", "low_contrast", "high_contrast", "overexposed", "over_suppressed"]

random_state = 42

materialize_images = False


def try_import_multilabel_splitter():
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
        return MultilabelStratifiedShuffleSplit
    except ImportError:
        raise ImportError(
            "Please install the 'iterative-stratification' package to use multilabel stratified splitting.")


def ensure_labels_cols(df, label_cols):
    missing = [c for c in label_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing label columns in CSV: {missing}")
    for c in label_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)
    return df


def multilabel_train_val_test_split(X, Y, test_size=0.15, val_size=0.15, random_state=42):
    msss = try_import_multilabel_splitter()
    msss1 = msss(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(msss1.split(X, Y))

    X_train_val, Y_train_val = X.iloc[train_val_idx], Y.iloc[train_val_idx]
    X_test, Y_test = X.iloc[test_idx], Y.iloc[test_idx]

    msss2 = msss(n_splits=1, test_size=val_size / (1 - test_size), random_state=random_state)
    train_idx, val_idx = next(msss2.split(X_train_val, Y_train_val))

    x_train, y_train = X_train_val.iloc[train_idx], Y_train_val.iloc[train_idx]
    x_val, y_val = X_train_val.iloc[val_idx], Y_train_val.iloc[val_idx]

    return (train_idx, val_idx, test_idx, train_val_idx)


def summarize(df, label_cols, name):
    total = len(df)
    has_issue = (df[label_cols].sum(axis=1) > 0).sum()
    no_issue = (df[label_cols].sum(axis=1) == 0).sum()
    print(f"[{name}] Total: {total}, Has Issue: {has_issue}, No Issue: {no_issue}")
    counts = df[label_cols].astype(bool).sum()
    print(f"[{name}] Label counts:\n{counts}")
    for c in label_cols:
        print(f" - {c:<16}: {counts.get(c, 0)}")


def main():
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_csv)

    df = ensure_labels_cols(df, label_cols)

    X = np.array(len(df)).reshape(-1, 1)
    Y = df[label_cols].values.astype(int)

    result = multilabel_train_val_test_split(df, df[label_cols], test_size=0.15, val_size=0.15,
                                             random_state=random_state)
    train_idx, val_idx, test_idx, train_val_idx = result

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    train_csv = os.path.join(output_dir, 'train_6labels.csv')
    val_csv = os.path.join(output_dir, 'val_6labels.csv')
    test_csv = os.path.join(output_dir, 'test_6labels.csv')

    # df_train.to_csv(train_csv, index=indexFalse)
    # df_val.to_csv(val_csv, index=False)
    # df_test.to_csv(test_csv, index=False)

    summarize(df, label_cols, "Overall")
    summarize(df_train, label_cols, "Train")
    summarize(df_val, label_cols, "Validation")
    summarize(df_test, label_cols, "Test")
    print(f"Train/Val/Test split: {len(df_train)}/{len(df_val)}/{len(df_test)}")


if __name__ == '__main__':
    main()
