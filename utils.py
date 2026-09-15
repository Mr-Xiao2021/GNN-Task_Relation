import gc
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import (to_scipy_sparse_matrix, scatter, )
from torchmetrics import AveragePrecision, AUROC
from tqdm.autonotebook import trange
from models.model import LLMModel
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class SentenceEncoder:
    def __init__(self, llm_name, cache_dir="cache_data/model", batch_size=1, multi_gpu=False):
        self.llm_name = llm_name
        self.device, _ = get_available_devices()
        self.batch_size = batch_size
        self.multi_gpu = multi_gpu
        self.model = LLMModel(llm_name, quantization=False, peft=False, cache_dir=cache_dir)
        self.model.to(self.device)

    def encode(self, texts, to_tensor=True):
        all_embeddings = []
        with torch.no_grad():
            for start_index in trange(0, len(texts), self.batch_size, desc="Batches", disable=False, ):
                sentences_batch = texts[start_index: start_index + self.batch_size]
                text_tokens = self.model.tokenizer(sentences_batch, return_tensors="pt", padding="longest", truncation=True,
                                           max_length=500).to(self.device)
                embeddings, _ = self.model.encode(text_tokens, pooling=True)
                embeddings = embeddings.cpu()
                all_embeddings.append(embeddings)
        all_embeddings = torch.cat(all_embeddings, dim=0)
        if not to_tensor:
            all_embeddings = all_embeddings.numpy()

        return all_embeddings

    def flush_model(self):
        # Delete the text encoder before GNN training to release accelerator memory.
        if self.model is not None:
            self.model = None
        gc.collect()
        if hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()


def binary_single_auc_func(func, output, batch):
    output = output.view(-1, batch.num_classes[0])
    score = torch.sigmoid(output)
    # if len(score.unique()) == 1:
    # print(output[:20])
    label = batch.bin_labels[batch.true_nodes_mask]
    # print(score)
    # print(label)
    return func.update(score, label.view(-1, batch.num_classes[0]))


def flat_auc(func, output, batch):
    return func(torch.sigmoid(output).view(-1), batch.bin_labels[batch.true_nodes_mask].view(-1))


def binary_apr_func(func, output, batch):
    output = output.view(-1, batch.num_classes[0])
    score = torch.sigmoid(output)
    label = batch.bin_labels[batch.true_nodes_mask]
    return func.update(score, label.view(len(batch), -1))


def binary_auc_multi_func(func, output, batch):
    output = output.view(-1, batch.num_classes[0])
    score = torch.sigmoid(output)
    label = batch.bin_labels[batch.true_nodes_mask]
    return func.update(score, label.view(-1, batch.num_classes[0]))


def label_apr_func(func, output, batch):
    score = torch.sigmoid(output)
    return func.update(score, batch.y)


def flat_label_func(func, output, batch):
    labels = batch.y.view(-1)
    valid_ind = labels == labels
    return func(output.view(-1)[valid_ind], labels[valid_ind])


def classification_single_func(func, output, batch):
    label = batch.bin_labels[batch.true_nodes_mask].view(-1, batch.num_classes[0])
    output = output.view(-1, batch.num_classes[0])
    return func(output, torch.argmax(label, dim=-1))


class MultiApr(torch.nn.Module):
    def __init__(self, num_labels=1):
        super().__init__()
        self.metrics = torch.nn.ModuleList([AveragePrecision(task="binary") for i in range(num_labels)])

    def update(self, preds, targets):
        for i, met in enumerate(self.metrics):
            pred = preds[:, i]
            target = targets[:, i]
            valid_idx = target == target
            # print(pred[valid_idx])
            # print(target[valid_idx])
            met.update(pred[valid_idx], target[valid_idx].to(torch.long))

    def compute(self):
        full_val = []
        for met in self.metrics:
            try:
                res = met.compute()
                if res == res:
                    full_val.append(res)
            except BaseException:
                pass
        return torch.tensor(full_val).mean()

    def reset(self):
        for met in self.metrics:
            met.reset()


class MultiAuc(torch.nn.Module):
    def __init__(self, num_labels=1):
        super().__init__()
        self.metrics = torch.nn.ModuleList([AUROC(task="binary") for i in range(num_labels)])

    def update(self, preds, targets):
        for i, met in enumerate(self.metrics):
            pred = preds[:, i]
            target = targets[:, i]
            valid_idx = target == target
            # print(pred[valid_idx])
            # print(target[valid_idx])
            met.update(pred[valid_idx], target[valid_idx].to(torch.long))

    def compute(self):
        full_val = []
        for met in self.metrics:
            try:
                res = met.compute()
                if res == res:
                    full_val.append(res)
            except BaseException:
                pass
        return torch.tensor(full_val).mean()

    def reset(self):
        for met in self.metrics:
            met.reset()


def scipy_rwpe(data, walk_length):
    row, col = data.edge_index # row=e, (e=edge_num)
    N = data.num_nodes

    value = data.edge_weight
    if value is None:
        value = torch.ones(data.num_edges, device=row.device)
    value = scatter(value, row, dim_size=N, reduce="sum").clamp(min=1)[row] # 每条边都是同样的转移概率
    value = 1.0 / value
    # 归一化邻接矩阵 P = D⁻¹A 
    adj = to_scipy_sparse_matrix(data.edge_index, edge_attr=value, num_nodes=data.num_nodes)

    out = adj
    pe_list = [out.diagonal()]
    for _ in range(walk_length - 1):
        out = out @ adj # P, P², P³, ..., P¹⁶, for walk_length=16
        pe_list.append(out.diagonal())
    pe = torch.tensor(np.stack(pe_list, axis=-1)) # shape: (e, walk_length)

    return pe


def get_available_devices():
    r"""Get IDs of all available accelerators.

    Returns:
        device (torch.device): Main device (NPU, GPU, or CPU).
        gpu_ids (list): List of available accelerator IDs.
    """
    gpu_ids = []
    if hasattr(torch, "npu") and torch.npu.is_available():
        gpu_ids += [device_id for device_id in range(torch.npu.device_count())]
        device = torch.device(f'npu:{gpu_ids[0]}')
        torch.npu.set_device(device)
    elif torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device(f'cuda:{gpu_ids[0]}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    return device, gpu_ids


def get_label_texts(labels):
    label_texts = [None] * int(len(labels) * 2)
    for entry in labels:
        label_texts[labels[entry][0]] = (
                "prompt node. molecule property description. " + "The molecule is effective to the following assay. " +
                labels[entry][1][0][:-41])
        label_texts[labels[entry][0] + len(labels)] = (
                "prompt node. molecule property description. " + "The molecule is not effective to the following "
                                                                 "assay. " +
                labels[entry][1][0][:-41])
    return label_texts


def set_mask(data, name, index, dtype=torch.bool):
    mask = torch.zeros(data.num_nodes, dtype=dtype)
    mask[index] = True
    setattr(data, name, mask)
