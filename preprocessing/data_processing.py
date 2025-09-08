# -*- coding: utf-8 -*-

import os
import re
import glob
import pandas as pd

image_dir = ['/data5/shuyanz/oppo_data/second_round/', '/data5/shuyanz/oppo_data/third_round/']
image_dir = '/data5/shuyanz/oppo_data/second_and_third/'
input_csv = '/data5/shuyanz/oppo_data/SAB类客观评测问题项信息.csv'
out_csv_explored = '/data5/shuyanz/oppo_data/explored_data.csv'
out_csv_grouped = '/data5/shuyanz/oppo_data/grouped_data.csv'
out_csv_explored_subset = '/data5/shuyanz/oppo_data/explored_data_subset.csv'
out_csv_grouped_subset = '/data5/shuyanz/oppo_data/grouped_data_subset.csv'

img_exts = ['.jpg']

explored_cols = ["图片", "问题点", "严重程度"]
explored_cols_out = {"图片": "filename", "问题点": "label", "严重程度": "level"}
problem_types = ["整体偏亮", "整体偏暗", "整体对比度低", "整体对比度高", "高光过曝", "高光压制过度"]
problem_en_types = ["bright", "dark", "low_contrast", "high_contrast", "overexposed", "over_suppressed"]
problem_types_out = {"整体偏亮": "bright", "整体偏暗": "dark", "整体对比度低": "low_contrast",
                     "整体对比度高": "high_contrast", "高光过曝": "overexposed", "高光压制过度": "over_suppressed"}
severity_map = {"无": int(0), "普通": int(1), "严重": int(2), "堵塞": int(3), "阻塞": int(3)}


def pick_up_filename():
    image_basenames = set()
    for ext in img_exts:
        for p in glob.glob(os.path.join(image_dir, f'*{ext}')):
            image_basenames.add(os.path.basename(p))

    df = pd.read_csv(input_csv, encoding='utf-8-sig')
    df["图片"] = df["图片"].astype(str).str.strip()
    df["问题点"] = df["问题点"].astype(str).str.strip()
    df["严重程度"] = df["严重程度"].astype(str).str.strip()

    filtered_df = df[df["图片"].isin(image_basenames)].copy()
    filtered_df_subset = filtered_df[filtered_df["问题点"].isin(problem_types)].copy()
    # translate problem types and severity levels
    filtered_df_subset["问题点"] = filtered_df_subset["问题点"].map(problem_types_out)
    filtered_df_subset["严重程度"] = filtered_df_subset["严重程度"].map(severity_map)

    explored_out = filtered_df[explored_cols]
    explorted_out_subset = filtered_df_subset[explored_cols].rename(columns=explored_cols_out)
    explored_out.sort_values(by=["图片"], inplace=True)
    explorted_out_subset.sort_values(by=["filename"], inplace=True)
    explored_out.to_csv(out_csv_explored, index=True, encoding='utf-8-sig')
    # save explored subset and only English is used in explored subset
    explorted_out_subset.to_csv(out_csv_explored_subset, index=True)

    explored_out["issue_with_severity"] = explored_out.apply(
        lambda r: f"{r['问题点']}({r['严重程度']})" if r['问题点'] else f"({r['严重程度']})", axis=1
    )
    grouped_df = (
        explored_out.groupby("图片", as_index=False)
        .agg(issues_joined=("issue_with_severity", lambda s: "|".join([x for x in s if str(x).strip() != ""])))
    )
    grouped_df.sort_values(by=["图片"], inplace=True)
    grouped_df.to_csv(out_csv_grouped, index=True, encoding='utf-8-sig')

    explorted_out_subset["issue_with_severity"] = explorted_out_subset.apply(
        lambda r: f"{r['label']}({r['level']})" if r['label'] else f"({r['level']})", axis=1
    )
    grouped_df_subset = (
        explorted_out_subset.groupby("filename", as_index=False)
        .agg(issues_joined=("issue_with_severity", lambda s: "|".join([x for x in s if str(x).strip() != ""])))
    )
    grouped_df_subset.sort_values(by=["filename"], inplace=True)
    grouped_df_subset.to_csv(out_csv_grouped_subset, index=True)

    print(f"Total images found: {len(image_names)}")


def unique_filename_list(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    df["level"] = pd.to_numeric(df["level"], errors='coerce').fillna(0).astype(int)
    df = df[df["label"].isin(problem_en_types)].copy()
    wide = (
        df.pivot_table(
            index="filename",
            columns="label",
            values="level",
            aggfunc='max',
            fill_value=0
        )
        .reindex(columns=problem_en_types, fill_value=0)
        .reset_index()
    )
    # wide.to_csv(output_csv, index=False)

    image_basenames = []
    for ext in img_exts:
        image_basenames.extend([os.path.basename(p) for p in glob.glob(os.path.join(image_dir, f'*{ext}'))])
    # pick up image basenames that contains "changjiang" or "zhujiang"
    image_basenames = [name for name in image_basenames if re.search(r'(?i)(changjiang|zhujiang)', name)]
    image_basenames = sorted(set(image_basenames))

    wide_full = pd.DataFrame({'filename': image_basenames})
    wide_full = wide_full.merge(wide, on='filename', how='left')

    for col in problem_en_types:
        if col not in wide_full.columns:
            wide_full[col] = 0
    wide_full[problem_en_types] = wide_full[problem_en_types].fillna(0).astype(int)
    wide_full.to_csv(output_csv, index=False)

    print(f"Unique filenames with issues saved to {output_csv}")


if __name__ == '__main__':
    # pick_up_filename()
    unique_filename_list(out_csv_explored_subset, '/data5/shuyanz/oppo_data/wide_list_changjiang_zhujiang.csv')
