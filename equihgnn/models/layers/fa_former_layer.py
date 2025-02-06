import torch
import torch.nn as nn
from einops import einsum, rearrange
from torch_geometric.utils import to_dense_batch

# https://github.com/Graph-and-Geometric-Learning/Frame-Averaging-Transformer


class FAFormerConfig:
    model_type = "faformer"

    def __init__(
        self,
        dim=3,  # dimension of the coordinates
        d_input=64,
        d_model=64,
        d_edge_model=64,
        n_layers=3,
        n_pos=1_000,  # maximum number of positions
        proj_drop=0.1,
        attn_drop=0.1,
        n_neighbors=16,
        valid_radius=1e6,
        embedding_grad_frac=1.0,
        n_heads=4,
        norm="layer",
        activation="silu",
        **kwargs,
    ):

        self.dim = dim
        self.d_input = d_input
        self.d_model = d_model
        self.d_edge_model = d_edge_model
        self.n_layers = n_layers
        self.n_pos = n_pos
        self.proj_drop = proj_drop
        self.attn_drop = attn_drop
        self.n_neighbors = n_neighbors
        self.valid_radius = valid_radius
        self.embedding_grad_frac = embedding_grad_frac
        self.n_heads = n_heads
        self.norm = norm
        self.activation = activation

        for key, value in kwargs.items():
            setattr(self, key, value)

        self._hidden_dim_check()

    def _hidden_dim_check(self):
        # check if d_model/d_edge_model can be divided by n_heads
        assert (
            self.d_model % self.n_heads == 0
        ), "d_model should be divisible by n_heads"
        assert (
            self.d_edge_model % self.n_heads == 0
        ), "d_edge_model should be divisible by n_heads"


class FrameAveraging(nn.Module):
    def __init__(self, dim=3, backward=False):
        super(FrameAveraging, self).__init__()

        self.dim = dim
        self.n_frames = 2**dim
        self.ops = self.create_ops(dim)  # [2^dim, dim]
        self.backward = backward

    def create_ops(self, dim):
        colon = slice(None)
        accum = []
        directions = torch.tensor([-1, 1])

        for ind in range(dim):
            dim_slice = [None] * dim
            dim_slice[ind] = colon
            accum.append(directions[dim_slice])

        accum = torch.broadcast_tensors(*accum)
        operations = torch.stack(accum, dim=-1)
        operations = rearrange(operations, "... d -> (...) d")
        return operations

    def create_frame(self, X, mask=None):
        assert (
            X.shape[-1] == self.dim
        ), f"expected points of dimension {self.dim}, but received {X.shape[-1]}"

        if mask is None:
            mask = torch.ones(*X.shape[:-1], device=X.device).bool()
        mask = mask.unsqueeze(-1)
        center = (X * mask).sum(dim=1) / mask.sum(dim=1)
        X = X - center.unsqueeze(1) * mask  # [B,N,dim]
        X_ = X.masked_fill(~mask, 0.0)

        C = torch.bmm(X_.transpose(1, 2), X_)  # [B,dim,dim] (Cov)
        if not self.backward:
            C = C.detach()

        _, eigenvectors = torch.linalg.eigh(C, UPLO="U")  # [B,dim,dim]
        F_ops = self.ops.unsqueeze(1).unsqueeze(0).to(
            X.device
        ) * eigenvectors.unsqueeze(
            1
        )  # [1,2^dim,1,dim] x [B,1,dim,dim] -> [B,2^dim,dim,dim]
        h = torch.einsum(
            "boij,bpj->bopi", F_ops.transpose(2, 3), X
        )  # transpose is inverse [B,2^dim,N,dim]

        h = h.view(X.size(0) * self.n_frames, X.size(1), self.dim)
        return h, F_ops.detach(), center

    def invert_frame(self, X, mask, F_ops, center):
        X = torch.einsum("boij,bopj->bopi", F_ops, X)
        X = X.mean(dim=1)  # frame averaging
        X = X + center.unsqueeze(1)
        if mask is None:
            return X
        return X * mask.unsqueeze(-1)


def batched_index_select(values, indices, dim=1):
    value_dims = values.shape[(dim + 1) :]
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
    indices = indices[(..., *((None,) * len(value_dims)))]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expand_len
    return values.gather(dim, indices)


# scale the row vectors of coordinates such that their root-mean-square norm is one
class NormCoordLN(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super(NormCoordLN, self).__init__()
        self.eps = eps

    def _norm_no_nan(self, x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
        """
        L2 norm of tensor clamped above a minimum value `eps`.

        :param sqrt: if `False`, returns the square of the L2 norm
        """
        # clamp is slow
        # out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
        out = torch.sum(torch.square(x), axis, keepdims) + eps
        return torch.sqrt(out) if sqrt else out

    def forward(self, coords):
        # coords: [..., N, 3]
        vn = self._norm_no_nan(coords, axis=-1, keepdims=True, sqrt=False, eps=self.eps)
        nonzero_mask = vn > 2 * self.eps
        vn = torch.sum(vn * nonzero_mask, dim=-2, keepdim=True) / (
            self.eps + torch.sum(nonzero_mask, dim=-2, keepdim=True)
        )
        vn = torch.sqrt(vn + self.eps)
        return nonzero_mask * (coords / vn)


def get_activation(activation="gelu"):
    return {
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "relu": nn.ReLU,
    }[activation]


def MLPWrapper(
    in_features,
    hidden_features,
    out_features,
    activation="gelu",
    norm_layer=None,
    bias=True,
    drop=0.0,
    drop_last=True,
):
    if activation == "swiglu":
        return SwiGLUMLP(
            in_features,
            hidden_features,
            out_features,
            norm_layer,
            bias,
            drop,
            drop_last,
        )
    else:
        return MLP(
            in_features,
            hidden_features,
            out_features,
            get_activation(activation),
            norm_layer,
            bias,
            drop,
            drop_last,
        )


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        drop_last=True,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = (
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        )
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop) if drop_last else nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SwiGLUMLP(nn.Module):
    """MLP w/ GLU style gating
    See: https://arxiv.org/abs/1612.08083, https://arxiv.org/abs/2002.05202
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        norm_layer=None,
        bias=True,
        drop=0.0,
        drop_last=True,
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        assert hidden_features % 2 == 0

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = nn.SiLU()
        self.drop1 = nn.Dropout(drop)
        self.norm = (
            norm_layer(hidden_features // 2)
            if norm_layer is not None
            else nn.Identity()
        )
        self.fc2 = nn.Linear(hidden_features // 2, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop) if drop_last else nn.Identity()

    def init_weights(self):
        # override init of fc1 w/ gate portion set to weight near zero, bias=1
        fc1_mid = self.fc1.bias.shape[0] // 2
        nn.init.ones_(self.fc1.bias[fc1_mid:])
        nn.init.normal_(self.fc1.weight[fc1_mid:], std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x1, x2 = x.chunk(2, dim=-1)
        x = self.act(x1) * x2
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class FAFFN(FrameAveraging):
    def __init__(
        self,
        d_model,
        proj_drop,
        activation,
        mlp_ratio=4.0,
    ):
        super(FAFFN, self).__init__()

        self.W_frame = MLPWrapper(
            3, d_model, d_model, activation, nn.LayerNorm, True, proj_drop
        )
        self.ffn = MLPWrapper(
            d_model * 2,
            int(d_model * mlp_ratio),
            d_model,
            activation,
            nn.LayerNorm,
            True,
            proj_drop,
        )
        self.ln = nn.LayerNorm(d_model)

    def forward(self, token_embs, geo_feats, batch_idx):
        token_embs = self.ln(token_embs)

        batch_token_embs, _ = to_dense_batch(token_embs, batch=batch_idx)
        batch_geo_feats, coord_mask = to_dense_batch(
            geo_feats, batch=batch_idx
        )  # [B, N, 3]

        B, N = batch_token_embs.size(0), batch_token_embs.size(1)

        geo_frame_feats, F_ops, center = self.create_frame(
            batch_geo_feats, coord_mask
        )  # [B*8, N, 3]
        geo_frame_feats = (
            self.W_frame(geo_frame_feats).view(B, 8, N, -1).mean(dim=1)
        )  # [B, N, H]

        h_token_embs = torch.cat(
            [batch_token_embs, geo_frame_feats], dim=-1
        )  # [B, N, d_model]
        return self.ffn(h_token_embs)[coord_mask]


class EdgeModule(FrameAveraging):
    def __init__(
        self,
        d_model,
        d_edge_model,
        proj_drop=0.0,
        activation="gelu",
    ):
        super(EdgeModule, self).__init__()

        self.coord_mlp = MLPWrapper(
            3 + 1, d_edge_model, d_edge_model, activation, nn.LayerNorm, True, proj_drop
        )
        self.edge_mlp = MLPWrapper(
            d_model * 2 + d_edge_model,
            d_model,
            d_model,
            activation,
            nn.LayerNorm,
            True,
            proj_drop,
        )

        self.att_mlp = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())

    def forward(self, token_embs, geo_feats, neighbor_indices, neighbor_masks):
        N, N_neighbors = token_embs.size(0), neighbor_indices.size(-1)

        radial_coords = (
            geo_feats.unsqueeze(dim=1) - geo_feats[neighbor_indices]
        )  # [N, N_neighbors, 3]
        radial_coord_norm = torch.sum(radial_coords**2, dim=-1).unsqueeze(
            -1
        )  # [N, N_neighbors, 1]

        """local frame features"""
        frame_feats, _, _ = self.create_frame(
            radial_coords, neighbor_masks
        )  # [N*8, N_neighbors, 3]
        frame_feats = frame_feats.view(
            N, 8, N_neighbors, -1
        )  # [N, 8, N_neighbors, d_model]

        radial_coord_norm = radial_coord_norm.unsqueeze(dim=1).expand(
            N, 8, N_neighbors, -1
        )
        frame_feats = self.coord_mlp(
            torch.cat([frame_feats, radial_coord_norm], dim=-1)
        ).mean(
            dim=1
        )  # [N, N_neighbors, d_model]

        pair_embs = torch.cat(
            [
                token_embs.unsqueeze(dim=1).expand(N, N_neighbors, -1),
                token_embs[neighbor_indices],
            ],
            dim=-1,
        )  # [N, N_neighbors, d_model*2]
        pair_embs = self.edge_mlp(torch.cat([pair_embs, frame_feats], dim=-1))
        return pair_embs * self.att_mlp(pair_embs)


class MLPAttnEdgeAggregation(FrameAveraging):
    def __init__(
        self,
        d_model,
        d_edge_model,
        n_heads,
        proj_drop=0.0,
        attn_drop=0.0,
        activation="gelu",
    ):
        super(MLPAttnEdgeAggregation, self).__init__()

        self.d_head, self.d_edge_head, self.n_heads = (
            d_model // n_heads,
            d_edge_model // n_heads,
            n_heads,
        )

        self.layernorm_qkv = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 3),
        )
        self.layernorm_qkv_edge = nn.Sequential(
            nn.LayerNorm(d_edge_model),
            nn.Linear(d_edge_model, d_edge_model * 2),
        )

        self.mlp_attn = nn.Linear(self.d_head, 1, bias=False)
        self.edge_attn = nn.Linear(self.d_edge_head, 1, bias=False)
        self.W_output = MLPWrapper(
            d_model + d_edge_model,
            d_model,
            d_model,
            activation,
            nn.LayerNorm,
            True,
            proj_drop,
        )
        self.W_gate = nn.Linear(d_model, 1)
        self.attn_dropout = nn.Dropout(attn_drop)

        if n_heads > 1:
            self.W_frame_agg = nn.Sequential(nn.Linear(n_heads, 1), nn.SiLU())

        self._init_params()

    def _init_params(self):
        nn.init.constant_(self.W_gate.weight, 0.0)
        nn.init.constant_(self.W_gate.bias, 1.0)

    def forward(
        self,
        token_embs,
        geo_feats,
        edge_feats,
        neighbor_indices,
        batch_idx,
        neighbor_masks,
    ):
        # token_embs: [N, -1], geo_feats: [N, 3], edge_feats: [N, N_neighbor, -1]
        # neighbor_indices: [N, N_neighbor], neighbor_masks: [N, N_neighbor]
        residual, n_tokens = token_embs, token_embs.size(0)
        d_head, n_heads = self.d_head, self.n_heads

        """qkv transformation"""
        q_s, k_s, v_s = self.layernorm_qkv(token_embs).chunk(3, dim=-1)
        q_s, k_s, v_s = map(
            lambda x: rearrange(x, "n (h d) -> n h d", h=n_heads), (q_s, k_s, v_s)
        )
        q_edge_s, v_edge_s = self.layernorm_qkv_edge(edge_feats).chunk(2, dim=-1)
        q_edge_s, v_edge_s = map(
            lambda x: rearrange(x, "n m (h d) -> n m h d", h=n_heads),
            (q_edge_s, v_edge_s),
        )
        gate_s = self.W_gate(
            token_embs
        ).sigmoid()  # for residue connection of geometric features

        """attention map"""
        message = (q_s.unsqueeze(dim=1) + k_s[neighbor_indices]).view(
            n_tokens, -1, n_heads, d_head
        )
        attn_map = self.mlp_attn(message).squeeze(dim=-1)
        attn_map = attn_map + self.edge_attn(q_edge_s).squeeze(
            dim=-1
        )  # [N, N_neighbor, n_heads]
        attn_map.masked_fill_(~neighbor_masks.unsqueeze(dim=-1), -1e9)
        attn_map = self.attn_dropout(
            nn.Softmax(dim=-1)(attn_map.transpose(1, 2))
        )  # [N, n_heads, N_neighbor]

        """context aggregation"""
        v_s_neighs = v_s[neighbor_indices].view(
            n_tokens, -1, n_heads, d_head
        )  # [N, N_neighbor, n_heads, D]
        scalar_context = einsum(attn_map, v_s_neighs, "n h m, n m h d -> n h d").view(
            n_tokens, -1
        )  # [N, n_heads*D]
        edge_context = einsum(attn_map, v_edge_s, "n h m, n m h d -> n h d").view(
            n_tokens, -1
        )  # [N, n_heads*D]
        scalar_output = (
            self.W_output(torch.cat([scalar_context, edge_context], dim=-1)) + residual
        )

        if n_heads == 1:
            geo_context = einsum(
                attn_map, geo_feats[neighbor_indices], "n h m, n m d -> n h d"
            ).squeeze(dim=1)
        else:
            # use FA to aggregate the geometric features in different heads to make sure equivariance
            batch_geo_feats, coord_mask = to_dense_batch(
                geo_feats, batch=batch_idx
            )  # [batch_B, batch_N, 3]
            batch_B, batch_N = batch_geo_feats.size(0), batch_geo_feats.size(1)
            geo_frame_feats, F_ops, center = self.create_frame(
                batch_geo_feats, coord_mask
            )  # [batch_B*8, batch_N, 3]

            # flattern the batch dimension for aligning with attention map and neighbor indices
            geo_frame_feats = (
                geo_frame_feats.view(batch_B, 8, batch_N, 3)
                .transpose(2, 1)
                .reshape(batch_B, batch_N, 8 * 3)
            )  # [batch_B, batch_N, 8*3]
            geo_frame_feats_flattern = (
                geo_frame_feats[coord_mask].view(n_tokens, 8, 3).transpose(1, 0)
            )  # [8, N, 3]

            geo_frame_feats_flattern = (
                geo_frame_feats_flattern.unsqueeze(dim=2)
                .expand(-1, -1, n_heads, -1)
                .reshape(8 * n_tokens, n_heads * 3)
            )  # [8*N, n_heads*3]
            neighbor_indices_expand = (
                neighbor_indices.unsqueeze(dim=0)
                .expand(8, -1, -1)
                .reshape(8 * n_tokens, -1)
            )  # [8*N, N_neighbor]
            geo_neighbors = geo_frame_feats_flattern[neighbor_indices_expand].view(
                8, n_tokens, -1, n_heads, 3
            )  # [8, N, N_neighbor, n_heads, 3]
            attn_map_expand = (
                attn_map.unsqueeze(dim=0).expand(8, -1, -1, -1).transpose(3, 2)
            )  # [8, N, n_heads, N_neighbor]

            geo_context = einsum(
                attn_map_expand, geo_neighbors, "f n m h, f n m h d -> f n h d"
            )  # [8, N, n_heads, 3]
            geo_context = (
                self.W_frame_agg(geo_context.transpose(3, 2))
                .squeeze(-1)
                .transpose(1, 0)
                .reshape(n_tokens, -1)
            )  # [N, 8*3]

            batch_geo_context, _ = to_dense_batch(
                geo_context, batch=batch_idx
            )  # [batch_B, batch_N, 8*3]
            batch_geo_context = batch_geo_context.view(
                batch_B, batch_N, 8, 3
            ).transpose(
                2, 1
            )  # [batch_B, 8, batch_N, 3]
            geo_context = self.invert_frame(
                batch_geo_context, coord_mask, F_ops, center
            )  # [batch_B, batch_N, 3]
            geo_context = geo_context[coord_mask]  # [N, 3]

        geo_output = geo_context * gate_s + geo_feats * (1 - gate_s)  # [N, 3]
        return scalar_output, geo_output


class FAFormerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        d_edge_model,
        n_heads,
        proj_drop=0.0,
        attn_drop=0.0,
        activation="gelu",
    ):
        super(FAFormerEncoderLayer, self).__init__()

        self.self_attn = MLPAttnEdgeAggregation(
            d_model, d_edge_model, n_heads, proj_drop, attn_drop, activation
        )
        self.ffn = FAFFN(d_model, proj_drop, activation)
        self.edge_module = EdgeModule(d_model, d_edge_model, proj_drop, activation)

    def forward(
        self,
        token_embs,
        geo_feats,
        edge_feats,
        neighbor_indices,
        batch_idx,
        neighbor_masks,
    ):
        # neighbor_indices: [B, N, N_neighbor], neighbor_masks: [B, N, N_neighbor]
        token_embs, geo_feats = self.self_attn(
            token_embs,
            geo_feats,
            edge_feats,
            neighbor_indices,
            batch_idx,
            neighbor_masks,
        )

        edge_feats = edge_feats + self.edge_module(
            token_embs, geo_feats, neighbor_indices, neighbor_masks
        )

        token_embs = token_embs + self.ffn(token_embs, geo_feats, batch_idx)
        return token_embs, geo_feats, edge_feats


class FAFormer(nn.Module):
    def __init__(self, config):
        super(FAFormer, self).__init__()

        self.input_transform = nn.Linear(config.d_input, config.d_model)
        self.edge_module = EdgeModule(
            config.d_model, config.d_edge_model, config.proj_drop, config.activation
        )
        self.layers = nn.ModuleList(
            [
                FAFormerEncoderLayer(
                    config.d_model,
                    config.d_edge_model,
                    config.n_heads,
                    config.proj_drop,
                    config.attn_drop,
                    config.activation,
                )
                for _ in range(config.n_layers)
            ]
        )
        if config.n_pos is not None:
            self.pos_emb = nn.Embedding(config.n_pos, config.d_model)

        self.dropout_module = nn.Dropout(config.proj_drop)

        self.n_neighbors = config.n_neighbors
        self.valid_radius = config.valid_radius
        self.embedding_grad_frac = config.embedding_grad_frac

    def _build_graph(self, coords, batch_idx, n_neighbors, valid_radius):
        exclude_self_mask = torch.eye(
            coords.shape[0], dtype=torch.bool, device=coords.device
        )  # 1: diagonal elements
        batch_mask = batch_idx.unsqueeze(0) == batch_idx.unsqueeze(
            1
        )  # [N, N], True if the token is in the same batch

        # calculate relative distance
        rel_pos = rearrange(coords, "n d -> n 1 d") - rearrange(coords, "n d -> 1 n d")
        rel_dist = rel_pos.norm(dim=-1).detach()  # [N, N]
        rel_dist.masked_fill_(exclude_self_mask | ~batch_mask, 1e9)

        dist_values, nearest_indices = rel_dist.topk(n_neighbors, dim=-1, largest=False)
        neighbor_mask = (
            dist_values <= valid_radius
        )  # [N, N_neighbors], True if distance is within valid radius
        return nearest_indices, neighbor_mask

    def forward(self, features, coords, z=None):
        B, N = coords.shape[0], coords.shape[1]

        pad_mask = features.sum(dim=-1) == 0

        """flattern the batch dimension for acceleration"""
        batch_idx = (
            torch.arange(B, device=coords.device).unsqueeze(-1).repeat(1, N)[~pad_mask]
        )
        features = features[~pad_mask]  # [-1, 3]
        coords = coords[~pad_mask]  # [-1, 3]

        token_embs = self.input_transform(features)  # [-1, d_model]

        # for learnable positional embedding
        if hasattr(self, "pos_emb"):
            assert z is not None
            token_embs = token_embs + self.pos_emb(z)  # [-1, d_model]

        token_embs = self.dropout_module(token_embs)
        token_embs = (
            self.embedding_grad_frac * token_embs
            + (1 - self.embedding_grad_frac) * token_embs.detach()
        )

        """generate graph"""
        nearest_indices, neighbor_mask = self._build_graph(
            coords, batch_idx, int(min(self.n_neighbors, N)), self.valid_radius
        )

        edge_feats = self.edge_module(
            token_embs, coords, nearest_indices, neighbor_mask
        )
        for i, layer in enumerate(self.layers):
            token_embs, coords, edge_feats = layer(
                token_embs,
                coords,
                edge_feats,
                nearest_indices,
                batch_idx,
                neighbor_mask,
            )

        token_embs, _ = to_dense_batch(token_embs, batch=batch_idx, max_num_nodes=N)
        coords, _ = to_dense_batch(coords, batch=batch_idx, max_num_nodes=N)

        return token_embs, coords
