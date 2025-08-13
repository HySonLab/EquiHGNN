import torch
import torch.nn as nn
from ogb.graphproppred.mol_encoder import AtomEncoder
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import to_dense_adj

from equihgnn.common.registry import registry

from .layers.egnn_layer import EGNN_Network


@registry.register_model("egnn")
class EGNN_Net(nn.Module):
    def __init__(
        self, num_target: int = 1, hidden_channels: int = 256, num_layers: int = 2
    ):
        super(EGNN_Net, self).__init__()
        self.hidden_dim = hidden_channels
        self.atom_encoder = AtomEncoder(emb_dim=hidden_channels)

        self.egnn_layer = EGNN_Network(
            dim=self.hidden_dim,
            norm_coors=True,
            valid_radius=5.0,
            num_nearest_neighbors=16,
            depth=num_layers,
        )

        self.node_dec = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.graph_dec = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, num_target),
        )

    def forward(self, data):
        adj_dense = to_dense_adj(data.edge_index)
        adj_dense = torch.block_diag(*[adj for adj in adj_dense])
        adj_bool = adj_dense.bool()

        x = self.atom_encoder(data.x)
        x, _ = self.egnn_layer(
            feats=x.unsqueeze(0),
            coors=data.pos.unsqueeze(0),
            adj_mat=adj_bool.unsqueeze(0),
        )

        x = x.squeeze(0)
        x = self.node_dec(x)
        x = global_add_pool(x, data.batch)

        pred = self.graph_dec(x)

        return pred.squeeze(1)
