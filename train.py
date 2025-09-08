import torch
import torch.nn as nn
import numpy as np
# from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import clip
import random
import time
import scipy.stats
from utils import set_dataset, _preprocess2, _preprocess3, convert_models_to_fp32
import torch.nn.functional as F
from itertools import product
import os
import pickle
import logging

from my_loss import pLoss_all_fidelity
from hex_graph import graph_SO_FFSC

device = 'cuda:5' if torch.cuda.is_available() else 'cpu'


# class MLP(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(MLP, self).__init__()
#         self.relu = nn.ReLU()
#         self.fc = nn.Linear(input_dim, output_dim, dtype=torch.float32)
#         # Initialize the fc layers.
#         nn.init.kaiming_normal_(self.fc.weight.data)
#         if self.fc.bias is not None:
#             nn.init.constant_(self.fc.bias.data, val=0)
#
#     def forward(self, x):
#         x = self.relu(x)
#         return self.fc(x)

class CLIPWithHead(nn.Module):
    def __init__(self, clip_model, input_dim, output_dim):
        super(CLIPWithHead, self).__init__()
        self.clip_model = clip_model
        feat_dim = input_dim
        self.head = nn.Linear(feat_dim, output_dim)

    def forward(self, x):
        x = self.clip_model.encode_image(x)
        x = self.head(x)
        return x


def main():
    ckpt_path = '/data5/shuyanz/oppo_data/ckpt'  # path of checkpoint
    oppo_set = '/data5/shuyanz/oppo_data/second_and_third/'  # path of image
    root_path = '/data5/shuyanz/oppo_data/'  # path of csv file
    oppo_train_csv = os.path.join(root_path, 'train_6labels.csv')
    oppo_val_csv = os.path.join(root_path, 'val_6labels.csv')
    oppo_test_csv = os.path.join(root_path, 'test_6labels.csv')

    num_node = 18  # number of node

    lr_clip = 1e-6
    lr_head = 1e-3
    num_epoch = 10
    batch_size = 50
    num_patch = 5
    num_workers = 8


    # model加入线性层
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model.float()
    # my_mlp = MLP(512, num_node)
    model = CLIPWithHead(clip_model, 512, num_node)
    for p in model.clip_model.transformer.parameters():
        p.requires_grad = False
    model.to(device)

    optimizer = torch.optim.AdamW([
        # {'params': model.parameters(), 'lr': initial_lr, 'weight_decay': 0.001},
        # {'params': my_mlp.parameters(), 'lr': initial_lr, 'weight_decay': 0.001}
        {'params':[p for p in model.clip_model.parameters() if p.requires_grad], 'lr': lr_clip, 'weight_decay': 0.001},
        {'params': model.head.parameters(), 'lr': lr_head, 'weight_decay': 0.001}
    ])

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

    train_loss = []
    start_epoch = 0

    best_result = {'attr1': 0.0, 'attr2': 0.0, 'attr3': 0.0, 'attr4': 0.0, 'attr5': 0.0, 'attr6': 0.0}
    best_epoch = {'attr1': 0, 'attr2': 0, 'attr3': 0, 'attr4': 0, 'attr5': 0, 'attr6': 0}

    # -----------------------------------
    # define the loss
    # -----------------------------------
    criterion = pLoss_all_fidelity(hexG=graph_SO_FFSC())

    train_loader = set_dataset(oppo_train_csv, batch_size, oppo_set, num_workers, preprocess3, num_patch, False)
    val_loader = set_dataset(oppo_val_csv, batch_size, oppo_set, num_workers, preprocess2, 15, True)
    test_loader = set_dataset(oppo_test_csv, batch_size, oppo_set, num_workers, preprocess2, 15, True)

    # result_pkl = {}
    # 需要改输入
    for epoch in range(0, num_epoch):
        best_result, best_epoch = train_single_epoch(model, best_result, best_epoch, train_loader, val_loader,
                                                     test_loader,
                                                     epoch, train_loss, ckpt_path, optimizer, scheduler, criterion)
        scheduler.step()

        # print('lambda weight:')
        # print(weighting_method.method.lambda_weight[:, epoch])

    # pkl_name = os.path.join(root_path, 'all_results.pkl')
    # with open(pkl_name, 'wb') as f:
    #     pickle.dump(result_pkl, f)

    # lambdas = weighting_method.method.lambda_weight
    # pkl_name = os.path.join(root_path, 'lambdas.pkl')
    # with open(pkl_name, 'wb') as f:
    #     pickle.dump(lambdas, f)


def do_batch(model, x):
    batch_size = x.size(0)
    num_patch = x.size(1)

    x = x.view(-1, x.size(2), x.size(3), x.size(4))

    logits_per_image = model.forward(x)

    logits_per_image = logits_per_image.view(batch_size, num_patch, -1)

    logits_per_image = logits_per_image.mean(1)

    return logits_per_image


def train_single_epoch(model, best_result, best_epoch, train_loader, val_loader, test_loader, epoch, train_loss,
                       ckpt_path, optimizer, scheduler, criterion):
    start_time = time.time()
    beta = 0.9
    running_loss = 0 if epoch == 0 else train_loss[-1]
    running_duration = 0.0
    num_steps_per_epoch = 10
    local_counter = epoch * num_steps_per_epoch + 1

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
        # if device == "cpu":
        #     optimizer.step()
        # else:
        #     convert_models_to_fp32(model)
        #     optimizer.step()
        #     clip.model.convert_weights(model)

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

        # logger.info(format_str % (epoch, dataset_idx + 1, loss_corrected,
        #                           examples_per_sec, duration_corrected))

        local_counter += 1
        start_time = time.time()

        train_loss.append(loss_corrected)

    if (epoch >= 0):
        acc_bright, acc_dark, acc_low_contrast, acc_high_contrast, acc_overexposed, acc_over_suppressed, acc_all = (
            eval(model, criterion, val_loader, phase='val', dataset='oppo'))
        acc_bright1, acc_dark1, acc_low_contrast1, acc_high_contrast1, acc_overexposed1, acc_over_suppressed1, acc_all1 = (
            eval(model, criterion, test_loader, phase='test', dataset='oppo'))

        print_text_val = 'val acc results:' + 'bright:{}, dark:{}, low_contrast:{}, high_contrast:{}, overexposed:{}, over_suppressed:{}, all:{}'.format(
            acc_bright, acc_dark, acc_low_contrast, acc_high_contrast, acc_overexposed, acc_over_suppressed, acc_all)
        # logger.info(print_text_val)
        print(print_text_val)

        print_text_test = ' test acc results:' + 'bright:{}, dark:{}, low_contrast:{}, high_contrast:{}, overexposed:{}, over_suppressed:{}, all:{}'.format(
            acc_bright1, acc_dark1, acc_low_contrast1, acc_high_contrast1, acc_overexposed1, acc_over_suppressed1,
            acc_all1)
        # logger.info(print_text_test)
        print(print_text_test)

        if epoch % 10 == 0:
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
    x_matrix = x.reshape(B, C // 3, 3).to(torch.int32)
    out = x_matrix.sum(dim=2)
    return out


def compute_num_graph(pred, gt, threshold=0.5):
    y_pred_binary = pred.detach()
    y_pred_binary_ls = torch.zeros_like(y_pred_binary)  # - 1
    y_pred_binary_ls[y_pred_binary >= 0.5] = 1

    pred_mapped = _inverse_mapping(y_pred_binary_ls)
    gt_mapped = _inverse_mapping(gt)

    true_num = (pred_mapped == gt_mapped).to(torch.float32)

    return torch.sum(true_num[:, 0]), torch.sum(true_num[:, 1]), torch.sum(
        true_num[:, 2]), torch.sum(true_num[:, 3]), torch.sum(true_num[:, 4]), torch.sum(
        true_num[:, 5])


def eval(model, criterion, loader, phase, dataset):
    model.eval()

    bright_num = 0
    dark_num = 0
    low_contrast_num = 0
    high_contrast_num = 0
    overexposed_num = 0
    over_suppressed_num = 0

    for step, sample_batched in enumerate(loader, 0):
        x, all_node = sample_batched['I'], sample_batched['all_node']

        x = x.to(device)
        all_node_gt = all_node.to(device)
        # all_node_gt = all_node_gt + all_node.cpu().tolist()

        # Calculate features
        with torch.no_grad():
            logits_per_image = do_batch(model, x)

        pMargin = criterion.infer(logits_per_image)
        num1, num2, num3, num4, num5, num6 = compute_num_graph(pMargin, all_node_gt)
        bright_num += num1
        dark_num += num2
        low_contrast_num += num3
        high_contrast_num += num4
        overexposed_num += num5
        over_suppressed_num += num6

    B = len(loader.dataset)
    acc_bright = bright_num.item() / B
    acc_dark = dark_num.item() / B
    acc_low_contrast = low_contrast_num.item() / B
    acc_high_contrast = high_contrast_num.item() / B
    acc_overexposed = overexposed_num.item() / B
    acc_over_suppressed = over_suppressed_num.item() / B
    acc_all = (bright_num + dark_num + low_contrast_num + high_contrast_num + overexposed_num + over_suppressed_num).item() / (
                      B * 6)

    print_text = dataset + ' ' + phase + ' finished'
    print(print_text)

    return acc_bright, acc_dark, acc_low_contrast, acc_high_contrast, acc_overexposed, acc_over_suppressed, acc_all


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
