# from ImageDataset import ImageDataset
from torch.utils.data import DataLoader
from ImageDataset2 import ImageDataset2
# from ImageDataset import ImageDataset_SPAQ, ImageDataset_TID, ImageDataset_PIPAL, ImageDataset_ava

from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip
from torchvision import transforms

from PIL import Image
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import glob
import torch
from typing import Optional, Tuple


try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def set_dataset(csv_file, bs, data_set, num_workers, preprocess, num_patch, num_problems, test):

    data = ImageDataset2(
        csv_file=csv_file,
        img_dir=data_set,
        num_patch=num_patch,
        test=test,
        num_problems=num_problems,
        preprocess=preprocess)

    if test:
        shuffle = False
    else:
        shuffle = True

    loader = DataLoader(data, batch_size=bs, shuffle=shuffle, pin_memory=True, num_workers=num_workers)

    return loader


class AdaptiveResize(object):
    """Resize the input PIL Image to the given size adaptively.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=InterpolationMode.BILINEAR, image_size=None):
        assert isinstance(size, int)
        self.size = size
        self.interpolation = interpolation
        if image_size is not None:
            self.image_size = image_size
        else:
            self.image_size = None

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        h, w = img.size

        if self.image_size is not None:
            if h < self.image_size or w < self.image_size:
                return transforms.Resize(self.image_size, self.interpolation)(img)

        if h < self.size or w < self.size:
            return img
        else:
            return transforms.Resize(self.size, self.interpolation)(img)


def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _preprocess2():
    return Compose([
        _convert_image_to_rgb,
        # AdaptiveResize(768),
        transforms.Resize([300, 300]),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def _preprocess3():
    return Compose([
        _convert_image_to_rgb,
        # AdaptiveResize(768),
        transforms.Resize([300, 300]),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()


def save_metrics_csv_and_plot(hist, output_path, PROBLEM_LABELS):
    """
    生成：
      - metrics_log.csv（包含 train_loss、各 val/test 准确率+漏报率）
      - train_loss.png
      - val_accuracy.png
      - test_accuracy.png
      - val_recall_rate.png
      - test_recall_rate.png
      - val_f1.png
      - test_f1.png
      - curves_all.png（五联图：train loss / val acc / test acc / val recall / test recall/ val f1 / test f1）
    """
    os.makedirs(output_path, exist_ok=True)
    labels = PROBLEM_LABELS + ["all"]

    # -------- 对齐 epoch 数 --------
    len_loss = len(hist.get("train_loss", []))
    len_val  = min(len(hist["val_acc"].get(k, []))  for k in labels) if hist.get("val_acc") else 0
    len_test = min(len(hist["test_acc"].get(k, [])) for k in labels) if hist.get("test_acc") else 0
    len_val_recall  = min(len(hist["val_recall"].get(k, []))  for k in labels) if hist.get("val_recall") else 0
    len_test_recall = min(len(hist["test_recall"].get(k, [])) for k in labels) if hist.get("test_recall") else 0
    len_val_f1 = min(len(hist["val_f1"].get(k, []))  for k in labels) if hist.get("val_f1") else 0
    len_test_f1 = min(len(hist["test_f1"].get(k, [])) for k in labels) if hist.get("test_f1") else 0

    num_epochs = min(
        len_loss if len_loss > 0 else float('inf'),
        len_val if len_val > 0 else float('inf'),
        len_test if len_test > 0 else float('inf'),
        len_val_recall if len_val_recall > 0 else float('inf'),
        len_test_recall if len_test_recall > 0 else float('inf'),
        len_val_f1 if len_val_f1 > 0 else float('inf'),
        len_test_f1 if len_test_f1 > 0 else float('inf')
    )
    if not np.isfinite(num_epochs) or num_epochs == 0:
        print("[save_metrics] No enough data to plot. Skip.")
        return
    num_epochs = int(num_epochs)
    epochs = list(range(1, num_epochs + 1))

    # -------- 构造 CSV --------
    data = {"epoch": epochs, "train_loss": hist["train_loss"][:num_epochs]}
    for k in labels:
        data[f"val_acc_{k}"]  = hist["val_acc"][k][:num_epochs]
        data[f"test_acc_{k}"] = hist["test_acc"][k][:num_epochs]
        data[f"val_recall_{k}"] = hist["val_recall"][k][:num_epochs]
        data[f"test_recall_{k}"] = hist["test_recall"][k][:num_epochs]
        data[f"val_f1_{k}"] = hist["val_f1"][k][:num_epochs]
        data[f"test_f1_{k}"] = hist["test_f1"][k][:num_epochs]

    df = pd.DataFrame(data)
    csv_path = os.path.join(output_path, "metrics_log.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"[Saved] {csv_path} ({len(df)} rows)")

    # csv_path = os.path.join(output_path, "metrics_log.csv")
    # if not _file_noempty(csv_path):
    #     print(f"[save_metrics] No metrics_log.csv found under {output_path}. Skip.")
    #     return
    # df = pd.read_csv(csv_path, encoding='utf-8-sig')
    # if df.empty:
    #     print(f"[save_metrics] metrics_log.csv is empty. Skip.")
    #     return
    # labels = PROBLEM_LABELS + ["all"]
    # epochs = df["epoch"].tolist()

    # -------- 绘图部分 --------
    def plot_metric(df, metric_prefix, title, ylabel, fname, ylim=(0, 1.0)):
        plt.figure(figsize=(8, 6))
        for k in labels:
            col = f"{metric_prefix}_{k}"
            if col in df:
                plt.plot(epochs, df[col], marker='o', label=k)
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.ylim(*ylim)
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.legend(ncol=4, fontsize=8)
        out_path = os.path.join(output_path, fname)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"[Saved] {out_path}")

    # --- 单图 ---
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, df["train_loss"], marker='o')
    plt.xlabel('Epoch'); plt.ylabel('Training Loss')
    plt.title('Training Loss over Epochs')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "train_loss.png"), dpi=200)
    plt.close()

    # --- Validation/Test Accuracy ---
    plot_metric(df, "val_acc",  "Validation Accuracy over Epochs", "Validation Accuracy", "val_accuracy.png")
    plot_metric(df, "test_acc", "Test Accuracy over Epochs",       "Test Accuracy",       "test_accuracy.png")

    # --- Validation/Test Recall ---
    plot_metric(df, "val_recall", "Validation Recall over Epochs", "Validation Recall", "val_recall.png")
    plot_metric(df, "test_recall","Test Recall over Epochs",       "Test Recall",       "test_recall.png")

    # --- Validation/Test F1 ---
    plot_metric(df, "val_f1", "Validation F1 Score over Epochs", "Validation F1 Score", "val_f1.png")
    plot_metric(df, "test_f1","Test F1 Score over Epochs",       "Test F1 Score",       "test_f1.png")

    # -------- 五联图 --------
    fig, axes = plt.subplots(1, 5, figsize=(28, 5))

    # 1. train loss
    axes[0].plot(epochs, df["train_loss"], marker='o')
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].grid(True, linestyle='--', alpha=0.4)

    # 2. val acc
    for k in labels:
        if f"val_acc_{k}" in df:
            axes[1].plot(epochs, df[f"val_acc_{k}"], marker='o', label=k)
    axes[1].set_title('Validation Accuracy'); axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].set_ylim(0.0, 1.0); axes[1].grid(True, linestyle='--', alpha=0.4)

    # 3. test acc
    for k in labels:
        if f"test_acc_{k}" in df:
            axes[2].plot(epochs, df[f"test_acc_{k}"], marker='o', label=k)
    axes[2].set_title('Test Accuracy'); axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Accuracy')
    axes[2].set_ylim(0.0, 1.0); axes[2].grid(True, linestyle='--', alpha=0.4)

    # 4. val recall
    for k in labels:
        if f"val_recall_{k}" in df:
            axes[3].plot(epochs, df[f"val_recall_{k}"], marker='o', label=k)
    axes[3].set_title('Validation Recall'); axes[3].set_xlabel('Epoch'); axes[3].set_ylabel('Recall')
    axes[3].set_ylim(0.0, 1.0); axes[3].grid(True, linestyle='--', alpha=0.4)

    # 5. test recall
    for k in labels:
        if f"test_recall_{k}" in df:
            axes[4].plot(epochs, df[f"test_recall_{k}"], marker='o', label=k)
    axes[4].set_title('Test Recall'); axes[4].set_xlabel('Epoch'); axes[4].set_ylabel('Recall')
    axes[4].set_ylim(0.0, 1.0); axes[4].grid(True, linestyle='--', alpha=0.4)

    handles, leg_labels = axes[4].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="lower center", ncol=7, frameon=False, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    all_png_path = os.path.join(output_path, "curves_all.png")
    plt.savefig(all_png_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {all_png_path}")


CKPT_RE = re.compile(r"IQA_HEX-(\d{5})\.pt$")

def find_latest_ckpt(ckpt_dir: str) -> Optional[Tuple[str, int]]:
    """返回 (ckpt_path, epoch)，若无则 None。"""
    paths = glob.glob(os.path.join(ckpt_dir, "IQA_HEX-*.pt"))
    best = None
    best_epoch = -1
    for p in paths:
        m = CKPT_RE.search(os.path.basename(p))
        if not m:
            continue
        ep = int(m.group(1))
        if ep > best_epoch:
            best_epoch = ep
            best = p
    if best is None:
        return None
    return best, best_epoch

def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device):
    """有些版本 load_state_dict 后 state 还在 CPU，这里统一搬到目标设备。"""
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def load_checkpoint_if_any(ckpt_dir, model, optimizer, scheduler, device) -> int:
    """
    从 ckpt_dir 恢复，返回 start_epoch（下一轮要跑的 epoch 编号）。
    若没有 ckpt，返回 0。
    """
    found = find_latest_ckpt(ckpt_dir)
    if found is None:
        print(f"[resume] No checkpoints under {ckpt_dir}, start from scratch.")
        return 0

    path, last_epoch = found
    print(f"[resume] Loading latest checkpoint: {path} (epoch {last_epoch})")
    ckpt = torch.load(path, map_location=device)

    # 严格/不严格按需选择：如果你改了模型结构，改成 strict=False
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        move_optimizer_state_to_device(optimizer, device)

    if scheduler is not None:
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        else:
            # 兼容旧版本 ckpt：没有保存 scheduler，就让它从 last_epoch 继续
            scheduler.last_epoch = last_epoch

    # 恢复历史（可选）
    if 'hist' in ckpt:
        print("[resume] History found in ckpt -> restoring hist")
        # 注意：这里直接覆盖外部的 hist 变量，你也可以选择 merge
        # 我们在 main 里会用返回值把它接回去
        global _RESUMED_HIST
        _RESUMED_HIST = ckpt['hist']
    if 'train_loss' in ckpt:
        global _RESUMED_TRAIN_LOSS
        _RESUMED_TRAIN_LOSS = ckpt['train_loss']

    # 下一轮要训练的 epoch
    return last_epoch + 1
