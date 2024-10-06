import torch
import torch.nn as nn
from ogb.graphproppred.mol_encoder import AtomEncoder
from torch_geometric.nn import global_add_pool

from equihgnn.common.registry import registry
from equihgnn.models.layers.conv import MHNNSConv
from equihgnn.models.layers.mlp import MLP
from equihgnn.models.layers.se3_transformer_layer import SE3Transformer


@registry.register_model("se3_transformer_equihnns")
class SE3TransformerEquiHNNS(nn.Module):
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

        self.se3_transformer_layer = SE3Transformer(
            dim=args.MLP_hidden,
            heads=2,
            depth=2,
            dim_head=32,
            num_degrees=2,
            valid_radius=5,
            num_neighbors=16,
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
        # print(data)

        V, E = data.edge_index0, data.edge_index1
        x = self.atom_encoder(data.x)
        feats = x.unsqueeze(0)
        coors = data.pos.unsqueeze(0)
        masks = torch.ones(feats.size()[:2]).bool().to(x.device)

        x = self.se3_transformer_layer(feats, coors, masks)
        x = x.squeeze(0)

        x0 = x
        for i in range(self.nlayer):
            x = self.dropout(x)
            x = self.conv(x, V, E, x0)
            x = self.act(x)
        x = self.dropout(x)
        x = global_add_pool(x, data.batch)
        x = self.mlp_out(x)
        return x.view(-1)
