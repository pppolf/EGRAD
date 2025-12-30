import torch
import numpy as np
from torch_geometric.utils import erdos_renyi_graph, to_undirected

class SBA:
    """ Backdoor attacks to graph neural networks.
    """

    def __init__(self, features, seed, attack_method, trigger_prob, trigger_size, target_class, dataset, device):
        self.device = device
        self.seed = seed
        self.attack_method = attack_method
        self.trigger_prob = trigger_prob
        self.trigger_size = trigger_size
        self.target_class = target_class
        self.trojan_feat, self.trojan_edge_index = self.gene_trigger(features, dataset)

    def gene_trigger(self, features, dataset):
        trojan_edge_index = erdos_renyi_graph(self.trigger_size, edge_prob=self.trigger_prob).to(self.device)

        rs = np.random.RandomState(self.seed)
        if self.attack_method == 'Rand_Gene':
            print("Rand generate the trigger")
            features = features.cpu().numpy()
            mean = -1.0 + features.mean(axis=0)
            std = features.std(axis=0)

            trojan_feat = []
            for i in range(self.trigger_size):
                trojan_feat.append(torch.tensor(rs.normal(mean, std), dtype=torch.float32, device=self.device))
            trojan_feat = torch.stack(trojan_feat)
        else:
            print("Rand sample the trigger")
            idx = rs.randint(features.shape[0], size=self.trigger_size)
            trojan_feat = features[idx]

        return trojan_feat, trojan_edge_index

    def get_poisoned_rand(self, features, edge_index, labels, idx_attach):
        trojan_labels = labels.clone()
        trojan_labels[idx_attach] = self.target_class
        # inject_trigger_rand now returns trigger edges too
        trojan_features, trojan_edge_index, weights, all_trigger_edges, trigger_edge_list = \
            self.inject_trigger_rand(idx_attach, features, edge_index)

        return trojan_features, trojan_edge_index, weights, trojan_labels, all_trigger_edges, trigger_edge_list

    def inject_trigger_rand(self, idx_attach, features, edge_index):
        features, edge_index = features.clone(), edge_index.clone()

        start = int(features.shape[0])
        new_edges = []  # list of [u,v] pairs (python ints) for all injected edges
        per_attach_edge_list = []  # list of sets per attach (for subgraph-level detection)

        # 1) attach edges: (attach_node, first node of trigger block)
        for i, idx in enumerate(idx_attach):
            attach_u = int(idx.item())
            # nodes in this trigger block will be numbered start + i*trigger_size ... start + i*trigger_size + trigger_size -1
            first_node = start + i * self.trigger_size
            # connect attach to every node in block? your original code connects attach to first node only:
            new_edges.append((attach_u, first_node))

        # 2) internal trigger edges for each attach block (shift trojan_edge_index nodes)
        for i in range(len(idx_attach)):
            base = start + i * self.trigger_size
            tmp_edge_index = self.trojan_edge_index.clone().cpu().numpy()  # shape [2, E_block]
            block_edges = []
            for a, b in zip(tmp_edge_index[0], tmp_edge_index[1]):
                u = int(base + int(a))
                v = int(base + int(b))
                block_edges.append((u, v))
            # add block edges
            new_edges.extend(block_edges)

        # 3) make undirected (i.e., for each (u,v) also add (v,u) conceptually) and deduplicate by normalized pair (min,max)
        normalized = set()
        for (u, v) in new_edges:
            if u <= v:
                normalized.add((u, v))
            else:
                normalized.add((v, u))

        # construct per-attach edge sets too
        for i, idx in enumerate(idx_attach):
            base = start + i * self.trigger_size
            attach_u = int(idx.item())
            s = set()
            # attach edge
            first_node = base
            s.add(tuple(sorted((attach_u, first_node))))
            # internal edges
            tmp_edge_index = self.trojan_edge_index.clone().cpu().numpy()
            for a, b in zip(tmp_edge_index[0], tmp_edge_index[1]):
                u = int(base + int(a)); v = int(base + int(b))
                s.add(tuple(sorted((u, v))))
            per_attach_edge_list.append(s)

        # make tensor of injected edges (unique, undirected, in normalized (min,max) order)
        all_trigger_edges_list = list(normalized)
        if len(all_trigger_edges_list) > 0:
            te_u = torch.tensor([p[0] for p in all_trigger_edges_list], dtype=torch.long, device=self.device)
            te_v = torch.tensor([p[1] for p in all_trigger_edges_list], dtype=torch.long, device=self.device)
            all_trigger_edges = torch.stack([te_u, te_v], dim=0)  # shape [2, M]
        else:
            all_trigger_edges = torch.empty((2,0), dtype=torch.long, device=self.device)

        # 4) create trojan_edge_index as original edges + new (we need original direction as before)
        # Reconstruct new edges as directed pairs before to_undirected to match earlier behavior
        new_edge_u = []
        new_edge_v = []
        for (u, v) in all_trigger_edges_list:
            new_edge_u.append(u); new_edge_v.append(v)
            new_edge_u.append(v); new_edge_v.append(u)  # add reverse to maintain to_undirected behavior
        if len(new_edge_u) > 0:
            new_edge_index = torch.tensor([new_edge_u, new_edge_v], dtype=torch.long, device=self.device)
            trojan_edge_index = torch.cat([edge_index, new_edge_index], dim=1)
        else:
            trojan_edge_index = edge_index.clone()

        # 5) trojan features
        trojan_features = self.trojan_feat.unsqueeze(0).repeat([len(idx_attach), 1, 1]).reshape(self.trigger_size * len(idx_attach), -1)
        trojan_features = torch.cat([features, trojan_features], dim=0)
        weights = torch.ones([trojan_edge_index.shape[1]], dtype=torch.float32, device=self.device)

        # return trojan features/edges/weights and trigger ground truth (both tensor and list-of-sets)
        return trojan_features, trojan_edge_index, weights, all_trigger_edges, per_attach_edge_list
