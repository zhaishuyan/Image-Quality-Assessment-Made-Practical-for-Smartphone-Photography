import pandas as pd
import numpy as np
import torch

# 你的标签名（顺序要与 CSV 中一致）
PROBLEM_LABELS = [
    "bright", "dark", "low_contrast", "high_contrast", "overexposed", "over_suppressed",
    "white_balance_red", "white_balance_blue", "white_balance_green",
    "white_balance_yellow", "crushed_shadows", "over_brightened_shadows"
]

def compute_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {csv_path}")

    # 将字符串转为整数list
    df["pred_label"] = df["pred_label"].apply(lambda x: list(map(int, x.split(","))))
    df["gt_label"] = df["gt_label"].apply(lambda x: list(map(int, x.split(","))))

    # 转成Tensor方便计算
    preds = torch.tensor(df["pred_label"].tolist(), dtype=torch.int)
    gts = torch.tensor(df["gt_label"].tolist(), dtype=torch.int)

    num_labels = preds.shape[1]
    acc_dict, miss_dict = {}, {}

    for i in range(num_labels):
        gt_i = gts[:, i]
        pred_i = preds[:, i]

        # ---- 准确率 ----
        acc_i = torch.mean((gt_i == pred_i).float())

        # ---- 漏报率 ----
        severe_mask = (gt_i >= 2)
        if severe_mask.sum() > 0:
            miss_i = torch.sum((pred_i <= 1) & severe_mask).float() / severe_mask.sum().float()
        else:
            miss_i = torch.tensor(0.0)

        acc_dict[PROBLEM_LABELS[i]] = acc_i.item()
        miss_dict[PROBLEM_LABELS[i]] = miss_i.item()

    # 计算平均
    acc_dict["all"] = float(np.mean(list(acc_dict.values())))
    miss_dict["all"] = float(np.mean(list(miss_dict.values())))

    # 输出结果
    print("\n=== 测试集评估结果 ===")
    print("Label\t\tAccuracy\tMissRate")
    for lbl in PROBLEM_LABELS:
        print(f"{lbl:20s}\t{acc_dict[lbl]*100:6.2f}%\t{miss_dict[lbl]*100:6.2f}%")
    print("---------------------------------------------")
    print(f"平均准确率: {acc_dict['all']*100:.2f}%")
    print(f"平均漏报率: {miss_dict['all']*100:.2f}%")

    # 保存为CSV
    df_out = pd.DataFrame({
        "label": list(PROBLEM_LABELS) + ["all"],
        "accuracy": [acc_dict[k] for k in PROBLEM_LABELS] + [acc_dict["all"]],
        "miss_rate": [miss_dict[k] for k in PROBLEM_LABELS] + [miss_dict["all"]],
    })
    out_path = csv_path.replace(".csv", "_metrics.csv")
    df_out.to_csv(out_path, index=False)
    print(f"\n[Saved metrics] → {out_path}")


if __name__ == "__main__":
    # 你可以修改这里为自己的CSV路径
    # csv_path = "output3_whole_12_labels/oppo_test_99.csv"
    csv_path='/data5/shuyanz/oppo_code/IQA_with_HEX_FocalFidelityLoss_PromptLearning/output2_multi_templates/oppo_test_99.csv'
    compute_from_csv(csv_path)
