import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sympy.benchmarks.bench_meijerint import alpha
# from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import clip
import random
import time
import scipy.stats
from utils import set_dataset, _preprocess2, _preprocess3, convert_models_to_fp32, save_metrics_csv_and_plot, \
    load_checkpoint_if_any
import torch.nn.functional as F
from itertools import product
import os
import pickle
import logging

from my_loss import pLoss_all_fidelity
from hex_graph import graph_SO_FFSC
from model import ClipPromptClassifier, freeze_text_encoder, build_prompts
from sklearn.metrics import f1_score, recall_score

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
NUM_NODES = len(ATTRS) * len(LEVELS)
NUM_PROBLESMS = len(PROBLEM_LABELS)
CLASS_PHRASES = build_prompts(ATTRS, LEVELS, TEMPLATES)

device = 'cuda:5' if torch.cuda.is_available() else 'cpu'


def main():
    lr_visual = 5e-6
    lr_prompt = 5e-4
    n_ctx = 4
    num_epoch = 50
    batch_size = 100
    num_patch = 3
    num_workers = 8
    alpha = 0.9
    gamma = 1.0
    split = 'split1'
    ckpt_path = 'ckpt_2_alpha' + str(alpha) + '_gamma' + str(gamma) + '_' + split  # path of checkpoint
    output_path = 'output_2_alpha' + str(alpha) + '_gamma' + str(gamma) + '_' + split
    oppo_set = '/data5/shuyanz/oppo_data/second_third_fourth/'  # path of image
    root_path = '/data5/shuyanz/oppo_data/OPPO_splits_3000/' + split  # path of csv file
    oppo_train_csv = os.path.join(root_path, 'train.csv')
    oppo_val_csv = os.path.join(root_path, 'val.csv')
    oppo_test_csv = os.path.join(root_path, 'test.csv')
    os.makedirs(ckpt_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    hist = {
        "train_loss": [],
        "val_acc": {"bright": [], "dark": [], "low_contrast": [], "high_contrast": [],
                    "white_balance_red": [], "white_balance_blue": [],
                    "white_balance_yellow": [], "all": []},
        "test_acc": {"bright": [], "dark": [], "low_contrast": [], "high_contrast": [],
                     "white_balance_red": [], "white_balance_blue": [],
                     "white_balance_yellow": [], "all": []},
        "val_recall": {"bright": [], "dark": [], "low_contrast": [], "high_contrast": [],
                       "white_balance_red": [], "white_balance_blue": [],
                       "white_balance_yellow": [], "all": []},
        "test_recall": {"bright": [], "dark": [], "low_contrast": [], "high_contrast": [],
                        "white_balance_red": [], "white_balance_blue": [],
                        "white_balance_yellow": [], "all": []},
        "val_f1": {"bright": [], "dark": [], "low_contrast": [], "high_contrast": [],
                   "white_balance_red": [], "white_balance_blue": [],
                   "white_balance_yellow": [], "all": []},
        "test_f1": {"bright": [], "dark": [], "low_contrast": [], "high_contrast": [],
                    "white_balance_red": [], "white_balance_blue": [],
                    "white_balance_yellow": [], "all": []}
    }

    clip_model, preprocess = clip.load("ViT-B/32", device='cpu')
    clip_model.float()
    freeze_text_encoder(clip_model)
    model = ClipPromptClassifier(clip_model, CLASS_PHRASES, n_ctx=n_ctx).to(device)

    print("Model prepared")

    prompt_params = [model.text_prompt.ctx]
    visual_params = [p for p in model.clip_model.visual.parameters() if p.requires_grad]
    alpha_params = [model.log_alpha]
    bias_params = [model.bias]

    optimizer = torch.optim.AdamW([
        {'params': prompt_params, 'lr': lr_prompt, 'weight_decay': 0.001},
        {'params': visual_params, 'lr': lr_visual, 'weight_decay': 0.001},
        {'params': alpha_params, 'lr': 5e-5, 'weight_decay': 0.0},
        {'params': bias_params, 'lr': 5e-4, 'weight_decay': 0.0}
    ])

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

    train_loss = []
    best_result = {}
    best_epoch = 0

    start_epoch = load_checkpoint_if_any(ckpt_path, model, optimizer, scheduler, device)

    # -----------------------------------
    # define the loss
    # -----------------------------------
    print("Define loss")
    criterion = pLoss_all_fidelity(hexG=graph_SO_FFSC(), alpha=alpha, gamma=gamma)
    print("Loss defined")
    train_loader = set_dataset(oppo_train_csv, batch_size, oppo_set, num_workers, preprocess3, num_patch, NUM_PROBLESMS,
                               False)
    val_loader = set_dataset(oppo_val_csv, batch_size, oppo_set, num_workers, preprocess2, 9, NUM_PROBLESMS, True)
    test_loader = set_dataset(oppo_test_csv, batch_size, oppo_set, num_workers, preprocess2, 9, NUM_PROBLESMS, True)

    print("Start training!")
    for epoch in range(start_epoch, num_epoch):
        best_result, best_epoch = train_single_epoch(model, best_result, best_epoch, train_loader, val_loader,
                                                     test_loader, epoch, start_epoch, train_loss, ckpt_path, optimizer,
                                                     scheduler,
                                                     criterion, hist, output_path)
        scheduler.step()

    save_metrics_csv_and_plot(hist, output_path, PROBLEM_LABELS)


def do_batch(model, x):
    batch_size = x.size(0)
    num_patch = x.size(1)
    x = x.view(-1, x.size(2), x.size(3), x.size(4))
    logits_per_image = model.forward(x)
    logits_per_image = logits_per_image.view(batch_size, num_patch, -1)
    logits_per_image = logits_per_image.mean(1)
    return logits_per_image


def train_single_epoch(model, best_result, best_epoch, train_loader, val_loader, test_loader, epoch, start_epoch,
                       train_loss,
                       ckpt_path, optimizer, scheduler, criterion, hist, output_path):
    start_time = time.time()
    beta = 0.9
    running_loss = train_loss[-1] if epoch > start_epoch else 0.0
    running_duration = 0.0
    num_steps_per_epoch = 10
    local_counter = epoch * num_steps_per_epoch + 1
    epoch_losses = []

    # model.eval()
    model.train()

    print('Learning rate:')
    print(optimizer.state_dict()['param_groups'][0]['lr'])
    if optimizer.state_dict()['param_groups'][0]['lr'] == 0:
        scheduler.step()
        print(optimizer.state_dict()['param_groups'][0]['lr'])

    num_sample_per_task = []

    for dataset_idx, sample_batched in enumerate(train_loader, 0):
        x, all_node_gt = sample_batched['I'], sample_batched['all_node']
        x = x.to(device)
        all_node_gt = all_node_gt.to(device)
        num_sample_per_task.append(x.size(0))

        optimizer.zero_grad()
        logits_per_image = do_batch(model, x)
        all_loss, pMargin = criterion(logits_per_image, all_node_gt)

        all_loss.backward()
        optimizer.step()

        # statistics
        running_loss = beta * running_loss + (1 - beta) * all_loss.data.item()
        loss_corrected = running_loss / (1 - beta ** local_counter)

        current_time = time.time()
        duration = current_time - start_time
        running_duration = beta * running_duration + (1 - beta) * duration
        duration_corrected = running_duration / (1 - beta ** local_counter)
        examples_per_sec = x.size(0) / duration_corrected
        format_str = ('(E:%d, S:%d ) [Loss = %.4f] (%.1f samples/sec; %.3f '
                      'sec/batch)')
        print(format_str % (epoch, dataset_idx + 1, loss_corrected,
                            examples_per_sec, duration_corrected))

        local_counter += 1
        start_time = time.time()

        train_loss.append(loss_corrected)
        epoch_losses.append(all_loss.data.item())
        # break

    acc_val, recall_val, f1_val = (
        eval(model, criterion, val_loader, phase='val', dataset='oppo', epoch=epoch, output_dir=output_path))
    acc_test, recall_test, f1_test = (
        eval(model, criterion, test_loader, phase='test', dataset='oppo', epoch=epoch, output_dir=output_path))

    if hist is not None and len(epoch_losses) > 0:
        hist["train_loss"].append(float(np.mean(epoch_losses)))

    if hist is not None:
        for k in PROBLEM_LABELS + ["all"]:
            hist["val_acc"][k].append(float(acc_val[k]))
            hist["test_acc"][k].append(float(acc_test[k]))
            hist["val_recall"][k].append(float(recall_val[k]))
            hist["test_recall"][k].append(float(recall_test[k]))
            hist["val_f1"][k].append(float(f1_val[k]))
            hist["test_f1"][k].append(float(f1_test[k]))

    if (epoch + 1) % 1 == 0:
        model_name = '{}-{:0>5d}.pt'.format('IQA_HEX', epoch)
        ckpt_name = os.path.join(ckpt_path, model_name)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, ckpt_name)  # just change to your preferred folder/filename

    return best_result, best_epoch


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

    hits = [0 for _ in range(NUM_PROBLESMS)]
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

        for i in range(NUM_PROBLESMS):
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
    acc_dict = {PROBLEM_LABELS[i]: hits[i] / B for i in range(NUM_PROBLESMS)}
    acc_dict['all'] = sum(hits) / (B * NUM_PROBLESMS)

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

    # return acc_dict


if __name__ == '__main__':
    seed = 20200626
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    preprocess2 = _preprocess2()
    preprocess3 = _preprocess3()

    main()
