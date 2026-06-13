"""
siamese_gnn.py
==============
Siamese GAT-GNN for graph similarity and PyG data conversion utility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool


class SiameseGNN(nn.Module):
    """
    Twin-branch Graph Attention Network (GAT) for computing a cosine
    similarity score between two knowledge-graph PyG Data objects.

    Edge weights (from triple_type-based KG construction) are fed into
    the GATConv layers via the edge_dim parameter, allowing the attention
    mechanism to scale by semantic importance.
    """

    def __init__(self, input_dim: int = 768, hidden_dim: int = 256,
                 output_dim: int = 128, heads: int = 4):
        super().__init__()
        # edge_dim=1 → GATConv will concatenate the scalar edge weight to
        # the attention logit, giving richer edge-aware attention.
        self.gat1        = GATConv(input_dim, hidden_dim // heads, heads=heads,
                                   dropout=0.2, concat=True, edge_dim=1)
        self.gat2        = GATConv(hidden_dim, output_dim, heads=1,
                                   dropout=0.2, concat=False, edge_dim=1)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(output_dim)

    def forward_once(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        # Use edge weights if present; fall back to uniform weights otherwise
        edge_attr = getattr(data, 'edge_attr', None)
        if edge_attr is None or edge_attr.shape[0] == 0:
            edge_attr = torch.ones((edge_index.shape[1], 1),
                                   dtype=torch.float32, device=x.device)

        batch = (data.batch if data.batch is not None
                 else torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        x = self.gat1(x, edge_index, edge_attr=edge_attr)
        x = self.layer_norm1(x)
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gat2(x, edge_index, edge_attr=edge_attr)
        x = self.layer_norm2(x)
        x = F.elu(x)
        return global_mean_pool(x, batch)

    def forward(self, data1: Data, data2: Data) -> torch.Tensor:
        emb1 = self.forward_once(data1)
        emb2 = self.forward_once(data2)
        return F.cosine_similarity(emb1, emb2)


def nx_to_pyg_data(nx_graph, node_embeddings_dict: dict) -> Data:
    """
    Convert a NetworkX DiGraph with precomputed node embeddings to a
    PyTorch Geometric Data object.

    Parameters
    ----------
    nx_graph : nx.DiGraph
    node_embeddings_dict : dict
        Mapping from node label (str) to either:
          * a torch.Tensor of shape [768]   (legacy format), or
          * {'emb': Tensor[768], 'node_type': str}  (new format).

    Returns
    -------
    torch_geometric.data.Data  with:
        x           : float32 tensor [N, 768]
        edge_index  : long tensor    [2, E]
        edge_attr   : float32 tensor [E, 1]   (edge weights)
    """
    nodes = list(nx_graph.nodes())
    if not nodes:
        return Data(x=torch.zeros((1, 768)),
                    edge_index=torch.empty((2, 0), dtype=torch.long),
                    edge_attr=torch.empty((0, 1), dtype=torch.float32))

    node_mapping = {node: i for i, node in enumerate(nodes)}
    x = torch.zeros((len(nodes), 768))
    for node, idx in node_mapping.items():
        if node in node_embeddings_dict:
            entry = node_embeddings_dict[node]
            # Accept both old (Tensor) and new (dict with 'emb' key) formats
            if isinstance(entry, dict):
                emb = entry.get('emb', torch.zeros(768))
            else:
                emb = entry
            x[idx] = emb.detach().cpu() if isinstance(emb, torch.Tensor) else torch.tensor(emb)

    edges = list(nx_graph.edges(data=True))
    if edges:
        edge_index = torch.tensor(
            [[node_mapping[u], node_mapping[v]] for u, v, _ in edges],
            dtype=torch.long
        ).t().contiguous()
        # Extract edge weights; default 1.0 if not present
        weights = [d.get('weight', 1.0) for _, _, d in edges]
        edge_attr = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr  = torch.empty((0, 1), dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
