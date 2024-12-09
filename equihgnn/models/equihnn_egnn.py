import torch
import torch.nn as nn
from ogb.graphproppred.mol_encoder import AtomEncoder
from torch_geometric.nn import global_add_pool

from egnn_pytorch import EGNN

from ..common.registry import registry
from layers.conv import MHNNConv, MHNNSConv
from layers.mlp import MLP


@registry.register_model("egnn_equihnn")
class EGNNEquiHNN(nn.Module):
    def __init__(self, num_target, args):
        """Equivariant Molecular Hypergraph Neural Network
        (Shared parameters between all message passing layers)

        This is built upon MHNNM, but it has been incorporated spherical harmonic
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
        self.mlp4_layers = args.MLP4_num_layers
        self.nlayer = args.All_num_layers

        self.egnn_layer = EGNN(
            dim=args.MLP_hidden,
            norm_coors=True,
            norm_feats=True,
            valid_radius=5.0,
            num_nearest_neighbors=16,
        )

        self.atom_encoder = AtomEncoder(emb_dim=args.MLP_hidden)
        self.bond_encoder = nn.Embedding(6, args.MLP_hidden)

        self.conv = MHNNConv(
            args.MLP_hidden,
            mlp1_layers=self.mlp1_layers,
            mlp2_layers=self.mlp2_layers,
            mlp3_layers=self.mlp3_layers,
            mlp4_layers=self.mlp4_layers,
            aggr=args.aggregate,
            dropout=args.dropout,
            normalization=args.normalization,
        )

        self.mlp_out = MLP(
            in_channels=args.MLP_hidden * 2,
            hidden_channels=args.output_hidden * 2,
            out_channels=num_target,
            num_layers=args.output_num_layers,
            dropout=args.dropout,
            Normalization=args.normalization,
            InputNorm=False,
        )

    def forward(self, data):
        V, E = data.edge_index0, data.edge_index1
        e_batch = []
        for i in range(data.n_e.shape[0]):
            e_batch += data.n_e[i].item() * [i]
        e_batch = torch.tensor(e_batch, dtype=torch.long, device=data.x.device)
        he_batch = e_batch[data.e_order > 2]

        x = self.atom_encoder(data.x)
        x, _ = self.egnn_layer(x.unsqueeze(0), data.pos.unsqueeze(0))
        x = x.squeeze(0)

        e = self.bond_encoder(data.edge_attr.squeeze())

        for i in range(self.nlayer):
            x, e = self.conv(x, e, V, E)
            if i == self.nlayer - 1:
                # remove relu for the last layer
                x = self.dropout(x)
                e = self.dropout(e)
            else:
                x = self.dropout(self.act(x))
                e = self.dropout(self.act(e))

        x = global_add_pool(x, data.batch)
        e = global_add_pool(e[data.e_order > 2], he_batch)
        out = self.mlp_out(torch.cat((x, e), -1))
        return out.view(-1)


@registry.register_model("egnn_equihnns")
class EGNNEquiHNNS(nn.Module):
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

        self.egnn_layer = EGNN(
            dim=args.MLP_hidden,
            norm_coors=True,
            norm_feats=True,
            valid_radius=5.0,
            num_nearest_neighbors=16,
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
        x = self.atom_encoder(data.x)
        x, _ = self.egnn_layer(x.unsqueeze(0), data.pos.unsqueeze(0))
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


@registry.register_model("egnn_equihnnm")
class EGNNEquiHNNM(nn.Module):
    def __init__(self, num_target, args):
        """Equivariant Molecular Hypergraph Neural Network (MHNN)
        (Multiple message passing layers, no parameters shared between layers)

        This is built upon MHNNM, but it has been incorporated spherical harmonic
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
        self.mlp4_layers = args.MLP4_num_layers
        self.nlayer = args.All_num_layers

        self.egnn_layer = EGNN(
            dim=args.MLP_hidden,
            norm_coors=True,
            norm_feats=True,
            valid_radius=5.0,
            num_nearest_neighbors=16,
        )

        self.atom_encoder = AtomEncoder(emb_dim=args.MLP_hidden)
        self.bond_encoder = nn.Embedding(6, args.MLP_hidden)

        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(self.nlayer):
            self.layers.append(
                MHNNConv(
                    args.MLP_hidden,
                    mlp1_layers=self.mlp1_layers,
                    mlp2_layers=self.mlp2_layers,
                    mlp3_layers=self.mlp3_layers,
                    mlp4_layers=self.mlp4_layers,
                    aggr=args.aggregate,
                    dropout=args.dropout,
                    normalization=args.normalization,
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(args.MLP_hidden))

        self.mlp_out = MLP(
            in_channels=args.MLP_hidden,
            hidden_channels=args.output_hidden,
            out_channels=num_target,
            num_layers=args.output_num_layers,
            dropout=args.dropout,
            Normalization=args.normalization,
            InputNorm=False,
        )

    def forward(self, data):
        V, E = data.edge_index0, data.edge_index1
        e_batch = []
        for i in range(data.n_e.shape[0]):
            e_batch += data.n_e[i].item() * [i]
        e_batch = torch.tensor(e_batch, dtype=torch.long, device=data.x.device)

        x = self.atom_encoder(data.x)

        x, _ = self.egnn_layer(x.unsqueeze(0), data.pos.unsqueeze(0))
        x = x.squeeze(0)

        e = self.bond_encoder(data.edge_attr.squeeze())

        for i, layer in enumerate(self.layers):
            x, e = layer(x, e, V, E)
            x = self.batch_norms[i](x)

            if i == self.nlayer - 1:
                # remove relu for the last layer
                x = self.dropout(x)
                e = self.dropout(e)
            else:
                x = self.dropout(self.act(x))
                e = self.dropout(self.act(e))

        x = global_add_pool(x, data.batch)
        out = self.mlp_out(x)
        return out.view(-1)
