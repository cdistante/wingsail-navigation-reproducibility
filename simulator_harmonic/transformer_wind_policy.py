import torch
import torch.nn as nn
import torch.nn.functional as F


class WindPolicyTransformer(nn.Module):
    """
    Predict local direction (d_row, d_col) from local planning context.

    Cell-wise inputs:
    - occupancy
    - gamma
    - potential
    - relative row/col to boat
    - relative row/col to goal

    Global inputs:
    - wind sin/cos
    - no-go angle
    """

    def __init__(
        self,
        grid_size: int = 41,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.grid_size = int(grid_size)
        self.num_tokens = self.grid_size * self.grid_size

        # Per-cell features: occ, gamma, pot, rel_boat_r, rel_boat_c, rel_goal_r, rel_goal_c
        self.cell_in_dim = 7

        self.cell_embed = nn.Sequential(
            nn.Linear(self.cell_in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Learn position embedding from normalized (row, col).
        self.pos_embed = nn.Sequential(
            nn.Linear(2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.global_mlp = nn.Sequential(
            nn.Linear(3, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2),
        )

        rr, cc = torch.meshgrid(
            torch.arange(self.grid_size, dtype=torch.float32),
            torch.arange(self.grid_size, dtype=torch.float32),
            indexing="ij",
        )
        rr = (rr / max(1, self.grid_size - 1)) * 2.0 - 1.0
        cc = (cc / max(1, self.grid_size - 1)) * 2.0 - 1.0
        pos = torch.stack([rr, cc], dim=-1).reshape(self.num_tokens, 2)
        self.register_buffer("pos_coords", pos, persistent=False)

    def _build_cell_features(self, occ, gamma, pot, boat_rc, goal_rc):
        # occ/gamma/pot shape: (B, H, W)
        bsz, h, w = occ.shape
        assert h == self.grid_size and w == self.grid_size, "Grid size mismatch."

        rr = torch.arange(h, dtype=occ.dtype, device=occ.device).view(1, h, 1).expand(bsz, h, w)
        cc = torch.arange(w, dtype=occ.dtype, device=occ.device).view(1, 1, w).expand(bsz, h, w)

        br = boat_rc[:, 0].view(-1, 1, 1).to(occ.dtype)
        bc = boat_rc[:, 1].view(-1, 1, 1).to(occ.dtype)
        gr = goal_rc[:, 0].view(-1, 1, 1).to(occ.dtype)
        gc = goal_rc[:, 1].view(-1, 1, 1).to(occ.dtype)

        scale = float(max(1, self.grid_size - 1))
        rel_boat_r = (rr - br) / scale
        rel_boat_c = (cc - bc) / scale
        rel_goal_r = (rr - gr) / scale
        rel_goal_c = (cc - gc) / scale

        feats = torch.stack(
            [
                occ,
                gamma,
                pot,
                rel_boat_r,
                rel_boat_c,
                rel_goal_r,
                rel_goal_c,
            ],
            dim=-1,
        )
        return feats.reshape(bsz, self.num_tokens, self.cell_in_dim)

    def forward(self, occ, gamma, pot, boat_rc, goal_rc, wind_sin_cos, no_go_angle_deg):
        occ = occ.float()
        gamma = gamma.float()
        pot = pot.float()

        x = self._build_cell_features(occ, gamma, pot, boat_rc, goal_rc)
        x = self.cell_embed(x)

        pos = self.pos_embed(self.pos_coords).unsqueeze(0)
        x = x + pos

        x = self.encoder(x)
        pooled = x.mean(dim=1)

        no_go_norm = (no_go_angle_deg.float().view(-1, 1) / 90.0).clamp(0.0, 2.0)
        g = torch.cat([wind_sin_cos.float(), no_go_norm], dim=1)
        g = self.global_mlp(g)

        out = self.head(torch.cat([pooled, g], dim=1))
        return out


def direction_loss(pred, target, alpha_mse: float = 0.7, alpha_cos: float = 0.3):
    pred_n = F.normalize(pred, dim=1, eps=1e-8)
    tgt_n = F.normalize(target, dim=1, eps=1e-8)
    mse = F.mse_loss(pred_n, tgt_n)
    cos = 1.0 - F.cosine_similarity(pred_n, tgt_n, dim=1).mean()
    return alpha_mse * mse + alpha_cos * cos, mse.detach(), cos.detach()
