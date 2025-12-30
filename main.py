import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.datasets import Planetoid,Flickr
import logging
from torch_geometric.utils import to_undirected, degree
import torch_geometric.transforms as T
import heuristic_selection as hs
from models.construct import model_construct
from torch_geometric.utils import k_hop_subgraph
from utils import evaluate_asr, subgraph, set_determinism, get_dataset, get_split
import torch.nn.functional as F
import random

from attack.sba import SBA
from attack.ugba import UGBA
from attack.gta import GTA

from defense.egrad import egrad

try:
    if 'logger' not in globals():
        logging.basicConfig()
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
except NameError:
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1027, help='Random seed.')

    # gpu setting
    parser.add_argument('--debug', action='store_true', default=False, help='debug mode')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--device_id', type=int, default=0, help='GPU id')
    
    # model setting
    parser.add_argument('--model', type=str, default='GCN', choices=['GCN','GAT'], help='Model used to attack')
    parser.add_argument('--dataset', type=str, default='Cora', help='Dataset', choices=['Cora','Pubmed','Flickr','ogbn-arxiv'])
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=32, help='Number of hidden units.')
    parser.add_argument('--target_class', type=int, default=0, help='use target class to poision')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--epochs', type=int,  default=200, help='Number of epochs to train benign and backdoor model.')
    parser.add_argument('--train_lr', type=float, default=0.01, help='Initial learning rate.')
    
    # backdoor setting
    parser.add_argument('--attack', type=str, default='sba', choices=['sba', 'ugba', 'gta'], help='selction attack type')
    parser.add_argument('--trigger_size', type=int, default=3, help='tirgger_size')
    parser.add_argument('--vs_ratio', type=float, default=0.005, help='ratio of poisoning nodes relative to the full graph')
    parser.add_argument('--dis_weight', type=float, default=1, help="Weight of cluster distance")
    parser.add_argument('--selection_method', type=str, default='none',
                        choices=['loss', 'conf', 'cluster', 'none', 'cluster_degree', 'clean_label'],
                        help='Method to select idx_attach for training trojan model (none means randomly select)')

    # defense setting
    parser.add_argument('--defense', type=str, default='none', choices=['none', 'egrad'], help='Mode of defense')
    
    # sba setting
    parser.add_argument('--sba_attack_method', type=str, default='Rand_Gene', choices=['Rand_Gene', 'Rand_Samp', 'Basic', 'None'],
                        help='Method to select idx_attach for training trojan model (none means randomly select)')
    parser.add_argument('--sba_trigger_prob', type=float, default=0.5,
                        help="The probability to generate the trigger's edges in random method")
    parser.add_argument('--sba_selection_method', type=str, default='none',
                        choices=['loss', 'conf', 'cluster', 'none', 'cluster_degree'],
                        help='Method to select idx_attach for training trojan model (none means randomly select)')
    
    # ugba setting
    parser.add_argument('--ugba_thrd', type=float, default=0.5)
    parser.add_argument('--ugba_lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--ugba_trojan_epochs', type=int, default=100, help='Number of epochs to train trigger generator.')
    parser.add_argument('--ugba_inner_epochs', type=int, default=3, help='Number of inner')
    parser.add_argument('--ugba_target_loss_weight', type=float, default=1, help="Weight of optimize outer trigger generator")
    parser.add_argument('--ugba_homo_loss_weight', type=float, default=100, help="Weight of optimize similarity loss")
    parser.add_argument('--ugba_homo_boost_thrd', type=float, default=0.8, help="Threshold of increase similarity")
    
    # gta attack
    parser.add_argument('--gta_thrd', type=float, default=0.5)
    parser.add_argument('--gta_lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--gta_trojan_epochs', type=int, default=400, help='Number of epochs to train trigger generator.')
    parser.add_argument('--gta_loss_factor', type=float, default=1, help='loss balance factor')
    parser.add_argument('--gta_selection_method', type=str, default='none',
                        choices=['loss', 'conf', 'cluster', 'none', 'cluster_degree'],
                        help='Method to select idx_attach for training trojan model (none means randomly select)')

    # egrad setting
    parser.add_argument('--egrad_def_epochs', type=int, default=100, help='Number of defensive rounds')
    parser.add_argument('--egrad_alpha', type=float, default=0.5)
    parser.add_argument('--egrad_eta', type=float, default=0.9)
    parser.add_argument('--egrad_tau', type=float, default=0.5)
    parser.add_argument('--egrad_gamma', type=float, default=0.1, help='Explain the consistency loss weight')
    parser.add_argument('--egrad_lambda1', type=float, default=0.5)
    parser.add_argument('--egrad_R_scale', type=int, default=4, help='Scaling coefficient')
    parser.add_argument('--egrad_recalc_every', type=int, default=10, help='The explanation is recalculated every epoch')
    
    args = parser.parse_known_args()[0]
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = ('cuda:{}' if torch.cuda.is_available() and args.cuda else 'cpu').format(args.device_id)

    set_determinism(args.seed)

    # Dataset Settings
    data, num_class = get_dataset(args.dataset, device)

    feat_dim = data.x.shape[1]
    data, idx_train, idx_val, idx_clean_test, idx_atk = get_split(data, device)
    logger.info(f'The number of trigger nodes for test set injection: {len(idx_atk)}, The number of clean test set nodes: {len(idx_clean_test)}')
    data.edge_index = to_undirected(data.edge_index) # Transform into an undirected edge
    train_edge_index, _, edge_mask = subgraph(torch.bitwise_not(data.test_mask),data.edge_index,relabel_nodes=False)
    mask_edge_index = data.edge_index[:,torch.bitwise_not(edge_mask)]

    unlabeled_idx = (torch.bitwise_not(data.test_mask)&torch.bitwise_not(data.train_mask)).nonzero().flatten()
    # num_test_nodes = int(torch.sum(data.test_mask).item())
    num_test_nodes = len(data.test_mask) - data.test_mask.sum()
    size = max(1, int(np.ceil(args.vs_ratio * num_test_nodes.cpu())))

    # idx_attach = obtain_attach_nodes(args=args, node_idxs=unlabeled_idx, size=size)

    idx_attach = None
    if args.selection_method == 'none':
        idx_attach = hs.obtain_attach_nodes(args, unlabeled_idx, size)
    elif args.selection_method == 'cluster':
        idx_attach = hs.cluster_distance_selection(
            args, data, idx_train, idx_val, idx_clean_test, unlabeled_idx, train_edge_index, size, device
        )
        idx_attach = torch.LongTensor(idx_attach).to(device)
    elif args.selection_method == 'cluster_degree':
        if args.dataset == 'Pubmed':
            idx_attach = hs.cluster_degree_selection_separate_fixed(
                args, data, idx_train, idx_val, idx_clean_test, unlabeled_idx, train_edge_index, size, device
            )
        else:
            idx_attach = hs.cluster_degree_selection(
                args, data, idx_train, idx_val, idx_clean_test, unlabeled_idx, train_edge_index, size, device
            )
        idx_attach = torch.LongTensor(idx_attach).to(device)
    elif args.selection_method == 'clean_label':
        idx_attach = hs.obtain_clean_label_attach_nodes(args, unlabeled_idx, data.y, size, args.target_class)

    unlabeled_idx = torch.tensor(list(set(unlabeled_idx.cpu().numpy()) - set(idx_attach.cpu().numpy()))).to(device)
    logger.info(f"Using attach size = {size} (0.5% of test nodes = {num_test_nodes})")

    # Build an attack constructor
    backdoor_gen_model, train_node_idx = None, None
    if args.attack == 'sba':
        backdoor_gen_model = SBA(
            features=data.x,
            seed=args.seed,
            attack_method=args.sba_attack_method,
            trigger_prob=args.sba_trigger_prob,
            trigger_size=args.trigger_size,
            target_class=args.target_class,
            dataset=args.dataset,
            device=device
        )
        poison_x, poison_edge_index, poison_edge_weights, poison_labels, _, _ = backdoor_gen_model.get_poisoned_rand(
            features=data.x,
            edge_index=train_edge_index,
            labels=data.y,
            idx_attach=idx_attach
        )
        train_node_idx = torch.cat([idx_train, idx_attach]).to(device)
    elif args.attack == 'ugba':
        backdoor_gen_model = UGBA(
            seed=args.seed, thrd=args.ugba_thrd, hidden=args.hidden, trojan_epochs=args.ugba_trojan_epochs,
            inner_epochs=args.ugba_inner_epochs, lr=args.ugba_lr, weight_decay=args.weight_decay,
            target_loss_weight=args.ugba_target_loss_weight, homo_loss_weight=args.ugba_homo_loss_weight,
            homo_boost_thrd=args.ugba_homo_boost_thrd, trigger_size=args.trigger_size, target_class=args.target_class, device=device
        )
        backdoor_gen_model.fit(data.x, train_edge_index, None, data.y, idx_train, idx_attach, unlabeled_idx)
        poison_x, poison_edge_index, poison_edge_weights, poison_labels = backdoor_gen_model.get_poisoned(
            features=data.x,
            edge_index=train_edge_index,
            edge_weight=None,
            labels=data.y,
            attach_idx=idx_attach
        )
        train_node_idx = torch.cat([idx_train, idx_attach]).to(device)
    elif args.attack == 'gta':
        backdoor_gen_model = GTA(
            thrd=args.gta_thrd, hidden=args.hidden, trojan_epochs=args.gta_trojan_epochs,
            loss_factor=args.gta_loss_factor, lr=args.gta_lr, weight_decay=args.weight_decay,
            trigger_size=args.trigger_size, target_class=args.target_class, device=device
        )
        backdoor_gen_model.fit(data.x, train_edge_index, None, data.y, idx_train, idx_attach, unlabeled_idx)
        poison_x, poison_edge_index, poison_edge_weights, poison_labels = backdoor_gen_model.get_poisoned(
            data.x, train_edge_index, None, data.y, idx_attach
        )
        train_node_idx = torch.cat([idx_train, idx_attach]).to(device)
    else:
        raise ValueError('invalid attack method!')
    logger.info("Percent of left attach nodes: {:.3f}".format(len(set(train_node_idx.tolist()) & set(idx_attach.tolist())) / len(idx_attach)))

    # begin model
    benign_model = model_construct(
        dataset=args.dataset,
        model_name=args.model,
        feat_dim=feat_dim, 
        num_class=num_class, 
        hidden=args.hidden, 
        dropout=args.dropout, 
        lr=args.train_lr, 
        weight_decay=args.weight_decay,
        device=device
    ).to(device)

    benign_model.fit(poison_x, poison_edge_index, poison_edge_weights, poison_labels, train_node_idx, idx_val, train_iters=args.epochs, verbose=args.debug)
    output = benign_model(poison_x, poison_edge_index, poison_edge_weights)

    if idx_attach is not None:
        train_attach_rate = (output.argmax(dim=1)[idx_attach] == poison_labels[idx_attach]).float().mean()
        logger.info("Target class rate on Vs: {:.4f}".format(train_attach_rate))

    induct_edge_index = torch.cat([poison_edge_index, mask_edge_index], dim=1)
    induct_edge_weights = torch.cat([poison_edge_weights,torch.ones([mask_edge_index.shape[1]],dtype=torch.float,device=device)])
    overall_induct_edge_index, overall_induct_edge_weights = induct_edge_index.clone(), induct_edge_weights.clone()
    clean_acc = benign_model.test(poison_x, induct_edge_index, induct_edge_weights, data.y, idx_clean_test)

    # before defense ASR
    before_asr = evaluate_asr(args, backdoor_gen_model, idx_atk, benign_model, poison_x, overall_induct_edge_index)

    # data proces
    n_poison_nodes = poison_x.size(0)
    n_orig_nodes = data.x.size(0)
    n_new = max(0, n_poison_nodes - n_orig_nodes)
    logger.info(f"Poisoned graph: nodes={n_poison_nodes}, original={n_orig_nodes}, new_nodes={n_new}")
    train_mask = torch.zeros(n_poison_nodes, dtype=torch.bool, device=device)
    train_node_idx = torch.cat([idx_train, idx_attach]).to(device)
    train_mask[train_node_idx] = True
    test_mask = torch.zeros(n_poison_nodes, dtype=torch.bool, device=device)
    test_mask[idx_clean_test] = True
    if n_new > 0:
        pad_y = torch.full((n_new,), fill_value=0, dtype=torch.long, device=device)
        y_full = torch.cat([data.y.to(device), pad_y], dim=0)
    else:
        y_full = data.y.to(device)

    assert y_full.size(0) == n_poison_nodes, "The y_full section does not match the length of poison_x"

    # Pre-defense diagnosis: Clean accuracy & ASR
    logger.info(f"[Before Defense] Clean Test Acc = {clean_acc:.4f}")
    logger.info(f"[Before Defense] ASR = {before_asr:.4f}")

    if args.defense == 'egrad':
        egrad(
            args=args,
            backdoor_gen_model=backdoor_gen_model,
            benign_model=benign_model,
            train_lr=args.train_lr,
            weight_decay=args.weight_decay,
            idx_atk=idx_atk,
            idx_clean_test=idx_clean_test,
            poison_x=poison_x,
            poison_edge_index=poison_edge_index,
            poison_edge_weights=poison_edge_weights,
            overall_induct_edge_index=overall_induct_edge_index,
            overall_induct_edge_weights=overall_induct_edge_weights,
            train_mask=train_mask,
            y_full=y_full,
            def_epochs=args.egrad_def_epochs,
            alpha=args.egrad_alpha,
            eta=args.egrad_eta,
            tau=args.egrad_tau,
            gamma=args.egrad_gamma,
            lambda1=args.egrad_lambda1,
            R_scale=args.egrad_R_scale,
            recalc_every=args.egrad_recalc_every
        )
    else:
        logger.info('No selection defense method!')