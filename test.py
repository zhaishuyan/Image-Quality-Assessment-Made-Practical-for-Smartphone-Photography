import torch
import numpy as np
import pandas as pd
import clip
from tqdm import tqdm
from utils import set_dataset, _preprocess2, _preprocess3
from model import ClipPromptClassifier, freeze_text_encoder, build_prompts
from my_loss import pLoss_all_fidelity
from hex_graph import graph_SO_FFSC
import os
from sklearn.metrics import recall_score, f1_score
from utils import find_latest_ckpt, load_checkpoint_if_any

device = 'cuda:4' if torch.cuda.is_available() else 'cpu'

# -------------------- 定义常量 --------------------
ATTRS = ["overall too bright", "overall too dark", "low global contrast", "high global contrast",
         "white balance shifted towards red", "white balance shifted towards blue",
         "white balance shifted towards yellow"]
LEVELS = ["slight", "strong", "extreme"]
TEMPLATES = [
    "{attr}, {level}",
    "overall {attr}, {level}",
    "a photo with {attr}, {level}",
    "image showing {attr}, {level}",
    "highlights are {attr}, {level}",
]
PROBLEM_LABELS = ["bright", "dark", "low_contrast", "high_contrast",
                  "white_balance_red", "white_balance_blue", "white_balance_yellow"]

CLASS_PHRASES = build_prompts(ATTRS, LEVELS, TEMPLATES)

NUM_PROBLEMS = len(PROBLEM_LABELS)
NUM_LEVELS = 4  # 四个等级：0,1,2,3


def do_batch(model, x):
    batch_size = x.size(0)
    num_patch = x.size(1)
    x = x.view(-1, x.size(2), x.size(3), x.size(4))
    logits_per_image = model.forward(x)
    logits_per_image = logits_per_image.view(batch_size, num_patch, -1)
    logits_per_image = logits_per_image.mean(1)
    return logits_per_image



def _inverse_mapping(x):
    B, C = x.shape
    x_matrix = x.reshape(B, C // len(LEVELS), len(LEVELS)).to(torch.int32)
    out = x_matrix.sum(dim=2)
    return out


def compute_num_graph(pred, gt, threshold=0.5):
    y_pred_binary = pred.detach()
    y_pred_binary_ls = torch.zeros_like(y_pred_binary)  # - 1
    y_pred_binary_ls[y_pred_binary >= 0.5] = 1
    pred_mapped = _inverse_mapping(y_pred_binary_ls)
    gt_mapped = _inverse_mapping(gt)
    true_num = (pred_mapped == gt_mapped).to(torch.float32)
    nums = torch.sum(true_num, dim=0)  # List[int]
    return nums, pred_mapped, gt_mapped


def eval(model, criterion, loader, phase, dataset, epoch, output_dir):
    model.eval()

    hits = [0 for _ in range(len(PROBLEM_LABELS))]
    csv_rows = []
    pred_all = []
    gt_all = []

    for step, sample_batched in enumerate(loader, 0):
        x, filenames, all_node = sample_batched['I'], sample_batched['filename'], sample_batched['all_node']
        x = x.to(device)
        all_node_gt = all_node.to(device)
        with torch.no_grad():
            logits_per_image = do_batch(model, x)
            pMargin = criterion.infer(logits_per_image)
        nums, pred_mapped, gt_mapped = compute_num_graph(pMargin, all_node_gt)
        pred_all.append(pred_mapped.detach().cpu())
        gt_all.append(gt_mapped.detach().cpu())

        for i in range(NUM_PROBLEMS):
            hits[i] += nums[i].item()
        pred_list = pred_mapped.detach().cpu().tolist()  # List[List[int]]，每样本长度6
        gt_list = gt_mapped.detach().cpu().tolist()
        for f, p, g in zip(filenames, pred_list, gt_list):
            csv_rows.append({
                "filename": f,
                "pred_label": ",".join(map(str, p)),
                "gt_label": ",".join(map(str, g)),
            })

    pred_all = torch.cat(pred_all, dim=0).numpy()
    gt_all = torch.cat(gt_all, dim=0).numpy()
    labels_list = list(range(0, len(LEVELS) + 1))

    B = len(loader.dataset)
    acc_dict = {PROBLEM_LABELS[i]: hits[i] / B for i in range(NUM_PROBLEMS)}
    acc_dict['all'] = sum(hits) / (B * NUM_PROBLEMS)

    recall_dict = {}
    f1_dict = {}
    for i, lbl in enumerate(PROBLEM_LABELS):
        r = recall_score(
            gt_all[:, i], pred_all[:, i],
            labels=labels_list, average='macro',  # 'macro', 'micro', 'weighted', None
            zero_division=0
        )
        f1 = f1_score(
            gt_all[:, i], pred_all[:, i],
            labels=labels_list, average='macro',  # 'macro', 'micro', 'weighted', None
            zero_division=0
        )
        recall_dict[lbl] = float(r)
        f1_dict[lbl] = float(f1)
    recall_dict['all'] = float(np.mean(list(recall_dict.values())))
    f1_dict['all'] = float(np.mean(list(f1_dict.values())))

    if (epoch + 1) % 10 == 0 and phase in ('val', 'test'):
        out_csv = f"{output_dir}/{dataset}_{phase}_{epoch}.csv"  # 每个 epoch 会覆盖一次；如需按 epoch 存，可自行加上 epoch 编号
        df = pd.DataFrame(csv_rows, columns=["filename", "pred_label", "gt_label"])
        df.to_csv(out_csv, index=False, encoding='utf-8')
        print(f"[Saved] {out_csv} ({len(df)} rows)")

    # ---- 输出信息 ----
    print(f"\n=== {dataset} {phase} finished ===")
    print(f"{'Label':25s} {'Accuracy':>10s} {'Recall':>10s} {'F1':>10s}")
    print("-" * 60)
    for lbl in PROBLEM_LABELS:
        print(f"{lbl:25s} {acc_dict[lbl] * 100:9.2f}% {recall_dict[lbl] * 100:9.2f}% {f1_dict[lbl] * 100:9.2f}%")
    print("-" * 60)
    print(f"{'mean':25s} {acc_dict['all'] * 100:9.2f}% {recall_dict['all'] * 100:9.2f}% {f1_dict['all'] * 100:9.2f}%\n")

    return acc_dict, recall_dict, f1_dict


def main(split_id=0):
    # === 路径配置 ===
    split = 'split' + str(split_id)
    ckpt_path = "/data5/shuyanz/oppo_code/HEXIQA_OPPO/ckpt_01_split2/IQA_HEX-00044.pt"
    root_path = "/data5/shuyanz/oppo_data/OPPO_splits/"+split+"/"
    # oppo_set = "/data5/shuyanz/oppo_data/second_and_third/"
    oppo_set = "/data5/shuyanz/oppo_data/second_and_third"
    # test_csv = os.path.join(root_path, "test_12labels_less0.csv")
    test_csv = os.path.join(root_path, "test.csv")
    output_dir = './output_testing/'
    os.makedirs(output_dir, exist_ok=True)
    print(test_csv)

    preprocess2 = _preprocess2()
    test_loader = set_dataset(test_csv, 50, oppo_set, 8, preprocess2, 5, NUM_PROBLEMS, True)

    # === 加载模型 ===
    clip_model, _ = clip.load("ViT-B/32", device='cpu')
    clip_model.float()
    freeze_text_encoder(clip_model)
    model = ClipPromptClassifier(clip_model, CLASS_PHRASES, n_ctx=4).to(device)
    load_checkpoint_if_any(ckpt_path, model, None, None, device)
    model.eval()

    criterion = pLoss_all_fidelity(hexG=graph_SO_FFSC())

    eval(model, criterion, test_loader, 'test', 'OPPO', 0, output_dir)

if __name__ == "__main__":
    torch.manual_seed(20200626)
    np.random.seed(20200626)
    for split_id in range(10):
        print(f"=== Evaluating split {split_id} ===")
        main(split_id=split_id)