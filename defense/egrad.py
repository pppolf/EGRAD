import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from torch_geometric.utils import k_hop_subgraph

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
    import os, numpy as np, torch
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

# ---------------------------
# 防御所需函数
# ---------------------------
def edge_importance_grad(model, x, edge_index, train_mask, edge_weight=None, pseudo_y=None):
    model.eval()
    x_req = x.clone().detach().requires_grad_(True)
    out = model(x_req, edge_index, edge_weight=edge_weight)
    if pseudo_y is None:
        pseudo_y = out[train_mask].argmax(dim=1).detach()
    loss = F.cross_entropy(out[train_mask], pseudo_y)
    model.zero_grad()
    loss.backward(retain_graph=True)
    node_score = (x_req.grad.abs() * x_req.abs()).sum(dim=1)
    u, v = edge_index
    e_score = (node_score[u] + node_score[v]) / 2.0
    e_score = (e_score - e_score.min()) / (e_score.max() - e_score.min() + 1e-9)
    return e_score.detach()

def compute_risk_scores(x, edge_index, I_e, alpha=0.5, beta=0.5, lambda1=0.5, lambda2=0.5):
    """
    计算边的异常度和组合风险：
    - A_e: 基于度的 z-score + 特征相似性(1 - cosine)
    - R_e: lambda1 * I_e + lambda2 * A_e
    """
    u, v = edge_index
    num_nodes = x.size(0)
    # 度（无向图）使用 u,v 的计数和
    deg = torch.bincount(u, minlength=num_nodes) + torch.bincount(v, minlength=num_nodes)
    deg_mean = deg.float().mean()
    deg_std = deg.float().std() + 1e-9
    z_uv = ((deg[u].float() + deg[v].float()) - deg_mean) / deg_std

    a = x[u]; b = x[v]
    dot = (a * b).sum(dim=1)
    na = a.norm(dim=1) + 1e-9
    nb = b.norm(dim=1) + 1e-9
    cos = dot / (na * nb)
    A_e = alpha * z_uv + beta * (1 - cos)
    A_e = (A_e - A_e.min()) / (A_e.max() - A_e.min() + 1e-9)

    R_e = lambda1 * I_e + lambda2 * A_e
    R_e = (R_e - R_e.min()) / (R_e.max() - R_e.min() + 1e-9)
    return A_e.detach(), R_e.detach()

def attenuate_edge_weights(R_e, tau=0.5, eta=0.8):
    """Sigmoid 门控的边权衰减: w = 1 - eta * sigmoid(R - tau)"""
    gate = torch.sigmoid(R_e - tau)
    w = 1.0 - eta * gate
    return torch.clamp(w, 0.0, 1.0)


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

def egrad(
        args,
        backdoor_gen_model, 
        benign_model, 
        train_lr, 
        weight_decay, 
        idx_atk,
        idx_clean_test,
        poison_x, 
        poison_edge_index, 
        poison_edge_weights, 
        overall_induct_edge_index,
        overall_induct_edge_weights, 
        train_mask, 
        y_full, 
        def_epochs, alpha, eta, tau, gamma, lambda1, R_scale, recalc_every
    ):
    # set_determinism(args.seed)
    benign_model.train()
    logger.info(f"Init weight mean: {torch.mean(torch.cat([p.flatten() for p in benign_model.parameters()])).item()}")
    
    with torch.no_grad():
        out_tmp = benign_model(poison_x, poison_edge_index, poison_edge_weights)
        pseudo_y_fixed = out_tmp[train_mask].argmax(dim=1).detach()

    I_old = edge_importance_grad(benign_model, poison_x, poison_edge_index,
                            train_mask, edge_weight=poison_edge_weights, pseudo_y=pseudo_y_fixed)
    A_e, R_e = compute_risk_scores(poison_x, poison_edge_index, I_old,
                                alpha=alpha, beta=(1-alpha), lambda1=lambda1, lambda2=(1-lambda1))


    R_scaled = (R_e - R_e.mean()) / (R_e.std() + 1e-9)
    R_scaled = R_scale * R_scaled
    def_edge_weight = attenuate_edge_weights(R_scaled, tau=tau, eta=eta)

    logger.info(f"[Defense] edge_weight stats -> min={def_edge_weight.min().item():.3f}, "
                f"max={def_edge_weight.max().item():.3f}, attenuated={(def_edge_weight<0.99).sum().item()}/{def_edge_weight.numel()}")


    def_opt = torch.optim.Adam(benign_model.parameters(), lr=train_lr, weight_decay=weight_decay)
    I_ref = I_old.clone().detach()
    for ep in range(def_epochs):
        benign_model.train(); def_opt.zero_grad()
        out = benign_model(poison_x, poison_edge_index, def_edge_weight)
        ce = F.cross_entropy(out[train_mask], y_full[train_mask]) 

        if (ep % recalc_every) == 0:
            I_new = edge_importance_grad(benign_model, poison_x, poison_edge_index, train_mask, edge_weight=def_edge_weight, pseudo_y=pseudo_y_fixed)
        else:
            I_new = I_ref
        exp_loss = F.mse_loss(I_new, I_ref)
        loss = ce + gamma * exp_loss
        loss.backward(); def_opt.step()

        if (ep + 1) % 20 == 0 or ep == 0:
            with torch.no_grad():
                acc_tr = (out[train_mask].argmax(dim=1) == y_full[train_mask]).float().mean().item()
            logger.info(f"[Defense] Epoch {ep+1} | loss={loss.item():.4f} | train-acc(real)={acc_tr:.4f}")
    final_clean_acc = benign_model.test(poison_x, overall_induct_edge_index, overall_induct_edge_weights, y_full, idx_clean_test)
    logger.info(f"[After Defense] Clean Test Acc = {final_clean_acc:.4f}")

    ASR_after = evaluate_asr(args, backdoor_gen_model, idx_atk, benign_model, poison_x, overall_induct_edge_index)
    logger.info(f"[After Defense] ASR = {ASR_after:.4f}")