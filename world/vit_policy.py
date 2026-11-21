"""Small Vision Transformer policy used before the JEPA stage.

The implementation deliberately uses an explicit ViT encoder block rather
than ``nn.TransformerEncoder``.  Its compact patch-token architecture is easy
to export to a NumPy-only inference model, allowing rollout simulations to
remain runnable even where the simulator environment has no PyTorch wheel.
"""

from __future__ import absolute_import, division, print_function

try:
    import torch
    from torch import nn
except ImportError:  # The environmental simulator itself does not require Torch.
    torch = None
    nn = None


if torch is not None:

    class VisionTransformerBlock(nn.Module):
        """Pre-normalised multi-head self-attention and MLP residual block."""

        def __init__(self, embed_dim, num_heads, mlp_ratio=2.0, layer_norm_epsilon=1.0e-5):
            super(VisionTransformerBlock, self).__init__()
            if embed_dim % num_heads != 0:
                raise ValueError("embed_dim must be divisible by num_heads.")
            hidden_dim = int(round(embed_dim * mlp_ratio))
            self.num_heads = int(num_heads)
            self.head_dim = embed_dim // num_heads
            self.scale = self.head_dim ** -0.5
            self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_epsilon)
            self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
            self.projection = nn.Linear(embed_dim, embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_epsilon)
            self.mlp_first = nn.Linear(embed_dim, hidden_dim)
            self.mlp_second = nn.Linear(hidden_dim, embed_dim)

        @staticmethod
        def _gelu(value):
            """Use the tanh GELU approximation reproduced by NumPy inference."""

            return 0.5 * value * (
                1.0 + torch.tanh(0.7978845608028654 * (value + 0.044715 * value.pow(3)))
            )

        def forward(self, tokens):
            batch, token_count, embed_dim = tokens.shape
            residual = tokens
            value = self.norm1(tokens)
            qkv = self.qkv(value).reshape(
                batch, token_count, 3, self.num_heads, self.head_dim
            )
            qkv = qkv.permute(2, 0, 3, 1, 4)
            query, key, value = qkv[0], qkv[1], qkv[2]
            attention = torch.matmul(query, key.transpose(-2, -1)) * self.scale
            attention = torch.softmax(attention, dim=-1)
            value = torch.matmul(attention, value)
            value = value.transpose(1, 2).reshape(batch, token_count, embed_dim)
            tokens = residual + self.projection(value)
            residual = tokens
            value = self.mlp_second(self._gelu(self.mlp_first(self.norm2(tokens))))
            return residual + value


    class EnvironmentalVisionTransformer(nn.Module):
        """ViT grid encoder fused with global environmental and boat features.

        Inputs are a normalised occupancy grid ``[B, C, H, W]`` and a
        normalised global feature vector.  The prediction is an unnormalised
        two-component East/North direction vector; callers normalise it before
        computing angular loss or converting it to a heading.
        """

        def __init__(
            self,
            input_channels,
            grid_size,
            patch_size,
            global_feature_count,
            embed_dim=32,
            num_heads=4,
            num_layers=1,
            mlp_ratio=2.0,
            layer_norm_epsilon=1.0e-5,
        ):
            super(EnvironmentalVisionTransformer, self).__init__()
            if grid_size % patch_size != 0:
                raise ValueError("grid_size must be divisible by patch_size.")
            if input_channels < 1 or global_feature_count < 1:
                raise ValueError("ViT input channel and global feature counts must be positive.")
            self.input_channels = int(input_channels)
            self.grid_size = int(grid_size)
            self.patch_size = int(patch_size)
            self.global_feature_count = int(global_feature_count)
            self.embed_dim = int(embed_dim)
            self.num_heads = int(num_heads)
            self.num_layers = int(num_layers)
            self.mlp_ratio = float(mlp_ratio)
            self.layer_norm_epsilon = float(layer_norm_epsilon)
            self.patches_per_side = self.grid_size // self.patch_size
            self.token_count = self.patches_per_side ** 2
            patch_feature_count = self.input_channels * self.patch_size ** 2
            self.patch_embedding = nn.Linear(patch_feature_count, self.embed_dim)
            self.class_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.position_embedding = nn.Parameter(
                torch.zeros(1, self.token_count + 1, self.embed_dim)
            )
            self.blocks = nn.ModuleList(
                [
                    VisionTransformerBlock(
                        self.embed_dim,
                        self.num_heads,
                        self.mlp_ratio,
                        self.layer_norm_epsilon,
                    )
                    for _ in range(self.num_layers)
                ]
            )
            self.final_norm = nn.LayerNorm(self.embed_dim, eps=self.layer_norm_epsilon)
            self.environment_first = nn.Linear(self.global_feature_count, self.embed_dim)
            self.environment_second = nn.Linear(self.embed_dim, self.embed_dim)
            self.head = nn.Linear(2 * self.embed_dim, 2)
            self._initialise_weights()

        def _initialise_weights(self):
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
            nn.init.normal_(self.class_token, mean=0.0, std=0.02)
            nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

        def config(self):
            """Return architecture data required by the NumPy runtime."""

            return {
                "input_channels": self.input_channels,
                "grid_size": self.grid_size,
                "patch_size": self.patch_size,
                "global_feature_count": self.global_feature_count,
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "num_layers": self.num_layers,
                "mlp_ratio": self.mlp_ratio,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }

        def forward(self, grids, global_features):
            if grids.ndim != 4:
                raise ValueError("grids must have shape [batch, channels, height, width].")
            batch, channels, height, width = grids.shape
            if (
                channels != self.input_channels
                or height != self.grid_size
                or width != self.grid_size
            ):
                raise ValueError("Grid dimensions do not match the trained ViT configuration.")
            if global_features.shape != (batch, self.global_feature_count):
                raise ValueError("Global feature dimensions do not match the trained ViT configuration.")
            patches = grids.reshape(
                batch,
                channels,
                self.patches_per_side,
                self.patch_size,
                self.patches_per_side,
                self.patch_size,
            )
            patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(batch, self.token_count, -1)
            tokens = self.patch_embedding(patches)
            class_token = self.class_token.expand(batch, -1, -1)
            tokens = torch.cat((class_token, tokens), dim=1) + self.position_embedding
            for block in self.blocks:
                tokens = block(tokens)
            grid_embedding = self.final_norm(tokens)[:, 0]
            environmental_embedding = torch.relu(self.environment_first(global_features))
            environmental_embedding = torch.relu(self.environment_second(environmental_embedding))
            return self.head(torch.cat((grid_embedding, environmental_embedding), dim=1))


else:

    class EnvironmentalVisionTransformer(object):
        """Import-safe placeholder that explains the optional training dependency."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "EnvironmentalVisionTransformer requires PyTorch. Use a Python/PyTorch "
                "environment compatible with the optional Vision Transformer trainer."
            )
