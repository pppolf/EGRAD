import torch.nn.functional as F
import torch
import os
import random
import numpy as np
import logging
from typing import List, Union
import torch_scatter
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.data import Data, HeteroData
from torch_geometric.datasets import Planetoid,Flickr
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree, to_undirected, k_hop_subgraph
from torch_geometric.utils import homophily
from contextlib import contextmanager

try:
    if 'logger' not in globals():
        logging.basicConfig()
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
except NameError:
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

def set_determinism(seed):
    import torch.backends.cudnn as cudnn
    # seed = 1234
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # TODO: Do we need deterministic in cudnn ? Double check
    cudnn.deterministic = True
    cudnn.benchmark = False
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

@contextmanager
def _allow_unsafe_torch_load():
    """仅在受信任的数据文件加载阶段把 torch.load 的 weights_only 关掉。"""
    import torch as _torch
    _orig_load = _torch.load
    def _patched_load(*args, **kwargs):
        kwargs.setdefault('weights_only', False)  # 关键：强制关闭 weights_only
        return _orig_load(*args, **kwargs)
    _torch.load = _patched_load
    try:
        yield
    finally:
        _torch.load = _orig_load

def get_dataset(dataset_name, device):
    logger.info('Loading {} Dataset'.format(dataset_name))

    # ====== Cora / Citeseer / Pubmed ======
    if dataset_name in ['Cora', 'Citeseer', 'Pubmed']:
        dataset = Planetoid(root='data/', name=dataset_name)
        graph = dataset[0].to(device)
        num_classes = int(graph.y.max().item() + 1)
        return graph, num_classes

    # ====== Flickr ======
    if dataset_name == 'Flickr':
        transform = T.Compose([T.NormalizeFeatures()])
        dataset = Flickr(root='data/Flickr/', transform=transform)
        graph = dataset[0].to(device)
        num_classes = int(graph.y.max().item() + 1)
        return graph, num_classes

    # ====== OGBN-Arxiv ======
    if dataset_name == 'ogbn-arxiv':
        logger.info('Loading OGBN-Arxiv via OGB (with safe torch.load override)...')
        # 仅在这一步关闭 weights_only
        with _allow_unsafe_torch_load():
            dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='data/OGB/')
            graph = dataset[0]
            split_idx = dataset.get_idx_split()

        # 构造 mask
        num_nodes = graph.num_nodes
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask   = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask  = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[split_idx['train']] = True
        val_mask[split_idx['valid']]   = True
        test_mask[split_idx['test']]   = True

        graph.train_mask = train_mask.to(device)
        graph.val_mask   = val_mask.to(device)
        graph.test_mask  = test_mask.to(device)

        # 标签维度压缩并放到设备
        graph.y = graph.y.squeeze(1)
        graph = graph.to(device)

        num_classes = int(graph.y.max().item() + 1)
        return graph, num_classes

    raise ValueError(f"Unknown dataset: {dataset_name}")

def get_split(data, device, node_idx=None):
    rs = np.random.RandomState(10)
    if node_idx is None:
        perm = rs.permutation(data.num_nodes)
    else:
        perm = node_idx[rs.permutation(len(node_idx))].cpu().numpy()
    train_number = int(0.2 * len(perm))

    num_classes = int(torch.max(data.y)) + 1
    if num_classes > 10:
        class2num_nodes = [0] * num_classes
        for i in range(num_classes):
            class2num_nodes[i] = train_number // num_classes
        for i in range(train_number % num_classes):
            class2num_nodes[i] += 1

        idx_train = None
        train_mask = torch.zeros_like(data.train_mask).to(device)
        for i in range(num_classes):
            _idx_train = torch.tensor(perm).to(device)
            _idx_train = _idx_train[data.y[perm] == i][:class2num_nodes[i]]
            idx_train = torch.cat([idx_train, _idx_train], dim=0) if idx_train is not None else _idx_train
            train_mask[_idx_train] = True
        idx_train = torch.sort(idx_train)[0]
        data.train_mask = train_mask.clone()
    else:
        idx_train = torch.tensor(sorted(perm[:train_number])).to(device)
        data.train_mask = torch.zeros_like(data.train_mask)
        data.train_mask[idx_train] = True

    val_number = int(0.1 * len(perm))
    idx_val = torch.tensor(sorted(perm[train_number: train_number + val_number])).to(device)
    data.val_mask = torch.zeros_like(data.val_mask)
    data.val_mask[idx_val] = True

    test_number = int(0.2 * len(perm))
    idx_test = torch.tensor(sorted(perm[train_number + val_number: train_number + val_number + test_number])).to(device)
    data.test_mask = torch.zeros_like(data.test_mask)
    data.test_mask[idx_test] = True

    idx_clean_test = idx_test[:int(len(idx_test) / 2)]
    idx_atk = idx_test[int(len(idx_test) / 2):]

    return data, idx_train, idx_val, idx_clean_test, idx_atk

def subgraph(subset, edge_index, edge_attr=None, relabel_nodes: bool = False):
    """Returns the induced subgraph of :obj:`(edge_index, edge_attr)`
    containing the nodes in :obj:`subset`.

    Args:
        subset (LongTensor, BoolTensor or [int]): The nodes to keep.
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`, :class:`Tensor`)
    """

    node_mask = subset
    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    edge_index = edge_index[:, edge_mask]
    edge_attr = edge_attr[edge_mask] if edge_attr is not None else None
    return edge_index, edge_attr, edge_mask

@functional_transform('column_normalize_features')
class ColumnNormalizeFeatures(BaseTransform):
    r"""Column-normalizes the attributes given in :obj:`attrs` to sum-up to one
    (functional name: :obj:`normalize_features`).

    Args:
        attrs (List[str]): The names of attributes to normalize.
            (default: :obj:`["x"]`)
    """
    def __init__(self, attrs: List[str] = ["x"]):
        super(BaseTransform, self).__init__()
        self.attrs = attrs

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        for store in data.stores:
            for key, value in store.items(*self.attrs):
                if value.numel() > 0:
                    value = value - value.min(dim=0, keepdim=True)[0]
                    value.div_(value.max(dim=0, keepdim=True)[0] + 1e-6)
                    store[key] = value
        return data

    def __call__(self, data: Union[Data, HeteroData]) -> Union[Data, HeteroData]:
        return self.forward(data)


def calc_adjusted_homophily(edge_index, labels):

    edge_index = to_undirected(edge_index)

    num_labels = torch.max(labels).item() + 1
    label_degree_cnt = torch.zeros(num_labels, dtype=torch.float)

    node_degrees = degree(edge_index[0])
    label_degree_cnt = torch_scatter.scatter(node_degrees, labels, reduce='sum')

    num_edges = edge_index.shape[1]
    total = torch.sum(label_degree_cnt ** 2) / num_edges ** 2

    edge_hm = homophily(edge_index, labels, method='edge')
    adjusted_homophily = (edge_hm - total) / (1.0 - total)
    return adjusted_homophily

# ----------------------------
# 训练 & 测试函数
# ----------------------------
def train(model, data, idx_train, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[idx_train], data.y[idx_train])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad
def test(model, data, idx_train, idx_val, idx_test):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    acc_train = (pred[idx_train] == data.y[idx_train]).float().mean().item()
    acc_val = (pred[idx_val] == data.y[idx_val]).float().mean().item()
    acc_test = (pred[idx_test] == data.y[idx_test]).float().mean().item()

    return acc_train, acc_val, acc_test

def evaluate_asr(args, backdoor_gen_model, idx_atk, benign_model, poison_x, overall_induct_edge_index):
    device = poison_x.device
    ASR = 0
    for idx in idx_atk:
        # 使用子图编号
        sub_induct_nodeset, sub_induct_edge_index, sub_mapping, sub_edge_mask = k_hop_subgraph(
            node_idx=[idx],
            num_hops=2,
            edge_index=overall_induct_edge_index,
            relabel_nodes=True
        )

        relabeled_node_idx = sub_mapping
        sub_induct_edge_weights = torch.ones([sub_induct_edge_index.shape[1]]).to(device)
        # 注入触发器
        induct_x, induct_edge_index, induct_edge_weights = None, None, None
        if args.attack == 'sba':
            induct_x, induct_edge_index, induct_edge_weights, _, _ = backdoor_gen_model.inject_trigger_rand(
                relabeled_node_idx,
                poison_x[sub_induct_nodeset],
                sub_induct_edge_index
            )
        elif args.attack == 'ugba':
            induct_x, induct_edge_index, induct_edge_weights = backdoor_gen_model.inject_trigger(
                relabeled_node_idx,
                poison_x[sub_induct_nodeset],
                sub_induct_edge_index,
                sub_induct_edge_weights
            )
        elif args.attack == 'gta':
            induct_x, induct_edge_index, induct_edge_weights = backdoor_gen_model.inject_trigger(
                relabeled_node_idx,
                poison_x[sub_induct_nodeset],
                sub_induct_edge_index,
                sub_induct_edge_weights
            )
        else:
            raise ValueError('Attack is not aviliable!')

        # 只保留权重大于0的边
        mask = induct_edge_weights > 0.0
        induct_edge_index = induct_edge_index[:, mask].to(device)
        induct_edge_weights = induct_edge_weights[mask].to(device)

        with torch.no_grad():
            # 攻击成功率
            output = benign_model(induct_x, induct_edge_index, induct_edge_weights)
            train_attach_rate = (output.argmax(dim=1)[relabeled_node_idx] == args.target_class)
            ASR += train_attach_rate

    # 统计最终指标
    ASR = ASR.item() / len(idx_atk)
    return ASR