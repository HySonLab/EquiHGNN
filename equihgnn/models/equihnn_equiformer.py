import torch
import torch.nn as nn
from ogb.graphproppred.mol_encoder import AtomEncoder
from torch_geometric.nn import global_add_pool

from equiformer_pytorch import Equiformer

from ..common.registry import registry
from layers.conv import MHNNConv, MHNNSConv
from layers.mlp import MLP


@registry.register_model("equiformer_equihnns")
class EquiformerEquiHNNS(nn.Module):
    def __init__(self, num_target, args):
        """Equivariant Molecular Hypergraph Neural Network,
        (Shared parameters between all message passing layers)

        This is built upon MHNN, but it has been incorporated spherical harmonic
        for transforming 3D coordinate information.

        Args:
            num_target (int): number of output targets
            args (NamedTuple): global args
        """
        super().__init__()

        act = {"Id": nn.Identity(), "relu": nn.ReLU(), "prelu": nn.PReLU()}
        self.act = act[args.activation]
        self.dropout = nn.Dropout(args.dropout)
        self.mlp1_layers = args.MLP1_num_layers
        self.mlp2_layers = args.MLP2_num_layers
        self.mlp3_layers = args.MLP3_num_layers
        self.nlayer = args.All_num_layers

        self.atom_encoder = AtomEncoder(emb_dim=args.MLP_hidden)

        self.equiformer_layer = Equiformer(
            dim=args.MLP_hidden,
            heads=1,
            depth=1,
            dim_head=48,
            num_degrees=2,
            valid_radius=5,
            num_neighbors=16,
            l2_dist_attention=False,
            reduce_dim_out=False,
            attend_self=True,
            linear_out=True,
        )

        self.conv = MHNNSConv(
            args.MLP_hidden,
            mlp1_layers=self.mlp1_layers,
            mlp2_layers=self.mlp2_layers,
            mlp3_layers=self.mlp3_layers,
            aggr=args.aggregate,
            dropout=args.dropout,
            normalization=args.normalization,
        )

        self.mlp_out = MLP(
            in_channels=args.MLP_hidden,
            hidden_channels=args.output_hidden,
            out_channels=num_target,
            num_layers=args.output_num_layers,
            dropout=args.dropout,
            Normalization=args.normalization,
            InputNorm=False,
        )

    def reset_parameters(self):
        self.conv.reset_parameters()
        self.mlp_out.reset_parameters()

    def forward(self, data):
        V, E = data.edge_index0, data.edge_index1

        feats = self.atom_encoder(data.x)
        feats = feats.unsqueeze(0)
        coors = data.pos.unsqueeze(0)
        masks = torch.ones(feats.size()[:2], device=data.x.device).bool()
        x = self.equiformer_layer(feats, coors, masks)
        x = x.type0

        x0 = x
        for i in range(self.nlayer):
            x = self.dropout(x)
            x = self.conv(x, V, E, x0)
            x = self.act(x)
        x = self.dropout(x)
        x = global_add_pool(x, data.batch)
        x = self.mlp_out(x)
        return x.view(-1)
