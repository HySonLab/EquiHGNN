import torch
import torch.nn as nn
from ogb.graphproppred.mol_encoder import AtomEncoder
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import to_dense_adj

from equihgnn.common.registry import registry

from .layers.equiformer_layer import Equiformer


@registry.register_model("equiformer")
class Equiformer_Net(nn.Module):
    def __init__(
        self, num_target: int = 1, hidden_channels: int = 256, num_layers: int = 1
    ):
        super(Equiformer_Net, self).__init__()
        self.hidden_dim = hidden_channels
        self.atom_encoder = AtomEncoder(emb_dim=hidden_channels)

        self.egnn_layer = Equiformer(
            dim=self.hidden_dim,
            heads=1,
            depth=num_layers,
            dim_head=48,
            num_degrees=2,
            valid_radius=5,
            num_neighbors=16,
            l2_dist_attention=False,
            reduce_dim_out=False,
            attend_self=True,
            linear_out=True,
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
        adj_dense = to_dense_adj(data.edge_index)  # float tensor
        adj_dense = torch.block_diag(*[adj for adj in adj_dense])
        adj_bool = adj_dense.bool()

        x = self.atom_encoder(data.x)
        x = self.egnn_layer(
            inputs=x.unsqueeze(0),
            coors=data.pos.unsqueeze(0),
            adj_mat=adj_bool.unsqueeze(0),
        )
        x = x.type0

        x = x.squeeze(0)
        x = self.node_dec(x)
        x = global_add_pool(x, data.batch)

        pred = self.graph_dec(x)

        return pred.squeeze(1)
