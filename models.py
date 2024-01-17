# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import math
import random
from typing import Optional
from contextlib import nullcontext

import torch
import torch.nn as nn
import numpy as np
from jaxtyping import Float, Shaped
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from torch import Tensor
from torch.utils.checkpoint import checkpoint

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob, use_cfg_embedding: bool = True, continuous: bool = False):
        super().__init__()
        self.continuous = continuous
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        if self.continuous:
            self.embedding_projection = nn.Linear(num_classes, hidden_size)
        else:
            self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        if self.continuous:
            labels = labels * (1 - drop_ids[:, None].to(labels))
        else:
            labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_projection(labels) if self.continuous else self.embedding_table(labels)
        return embeddings



#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        use_cfg_embedding: bool = True,
        use_gradient_checkpointing: bool = True,
        is_label_continuous: bool = False,

        # below are Fixed Point-specific arguments.
        fixed_point: bool = False,

        # size
        fixed_point_pre_depth: int = 1, 
        fixed_point_post_depth: int = 1, 

        # iteration counts
        fixed_point_no_grad_min_iters: int = 0, 
        fixed_point_no_grad_max_iters: int = 0,
        fixed_point_with_grad_min_iters: int = 28, 
        fixed_point_with_grad_max_iters: int = 28,

        # solution recycle
        fixed_point_reuse_solution = False,
        
        # pre_post_timestep_conditioning
        fixed_point_pre_post_timestep_conditioning: bool = True,

        # adaptively distributing iterations among timesteps. Currently we only support linear distribution.
        adaptive: bool = False,
        iteration_controller = None
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_gradient_checkpointing = use_gradient_checkpointing

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob, 
            use_cfg_embedding=use_cfg_embedding, continuous=is_label_continuous)
        num_patches = self.x_embedder.num_patches
        
        # Will use fixed sin-cos embedding:
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.register_buffer('pos_embed', torch.zeros(1, num_patches, hidden_size))

        # New: Fixed Point
        self.fixed_point = fixed_point
        if self.fixed_point:
            self.fixed_point_no_grad_min_iters = fixed_point_no_grad_min_iters
            self.fixed_point_no_grad_max_iters = fixed_point_no_grad_max_iters
            self.fixed_point_with_grad_min_iters = fixed_point_with_grad_min_iters
            self.fixed_point_with_grad_max_iters = fixed_point_with_grad_max_iters
            self.fixed_point_pre_post_timestep_conditioning = fixed_point_pre_post_timestep_conditioning
            self.blocks_pre = nn.ModuleList([DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(fixed_point_pre_depth)])
            self.block_pre_projection = nn.Linear(hidden_size, hidden_size)
            self.block_fixed_point_projection_fc1 = nn.Linear(2 * hidden_size, 2 * hidden_size)
            self.block_fixed_point_projection_act = nn.GELU(approximate="tanh")
            self.block_fixed_point_projection_fc2 = nn.Linear(2 * hidden_size, hidden_size)
            self.block_fixed_point = DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            self.blocks_post = nn.ModuleList([DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(fixed_point_post_depth)])
            self.blocks = [*self.blocks_pre, self.block_fixed_point, *self.blocks_post]
            self.fixed_point_reuse_solution = fixed_point_reuse_solution
            self.last_solution = None

            self.adaptive = adaptive
            self.iteration_controller = iteration_controller
        else:
            self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        if self.y_embedder.continuous:
            nn.init.normal_(self.y_embedder.embedding_projection.weight, std=0.02)
            nn.init.constant_(self.y_embedder.embedding_projection.bias, 0)
        else:
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
    
    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs
        return ckpt_forward

    def _forward_dit(self, x, t, y):
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D))
        c = t + y                                # (N, D)
        for block in self.blocks:
            x = checkpoint(self.ckpt_wrapper(block), x, c) if self.use_gradient_checkpointing else block(x, c)  # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x
    
    def _forward_fixed_point_blocks(
        self, x: Float[Tensor, "b t d"], x_input_injection: Float[Tensor, "b t d"], c: Float[Tensor, "b d"], num_iterations: int
    ) -> Float[Tensor, "b t d"]:
        def forward_pass(x):
            x = torch.cat((x, x_input_injection), dim=-1)  # (N, T, D * 2)
            x = self.block_fixed_point_projection_fc1(x)  # (N, T, D * 2)
            x = self.block_fixed_point_projection_act(x)  # (N, T, D * 2)
            x = self.block_fixed_point_projection_fc2(x)  # (N, T, D)
            x = self.block_fixed_point(x, c)  # (N, T, D)
            return x
        
        if self.adaptive:
            num_iterations = self.iteration_controller.get()

        for _ in range(num_iterations):
            x = forward_pass(x)
        return x
    
    def _check_inputs(self, x: Float[Tensor, "b c h w"], t: Shaped[Tensor, "b"], y: Shaped[Tensor, "b"]) -> None:
        if self.fixed_point_reuse_solution:
            if not torch.all(t[0] == t).item():
                raise ValueError(t)

    def _forward_fixed_point(self, x: Float[Tensor, "b c h w"], t: Shaped[Tensor, "b"], y: Shaped[Tensor, "b"]) -> Float[Tensor, "b c h w"]:
        self._check_inputs(x, t, y)
        x: Float[Tensor, "b t d"] = self.x_embedder(x) + self.pos_embed
        t_emb: Float[Tensor, "b d"] = self.t_embedder(t)
        y: Float[Tensor, "b d"] = self.y_embedder(y, self.training)
        c: Float[Tensor, "b d"] = t_emb + y
        c_pre_post_fixed_point: Float[Tensor, "b d"] = (t_emb + y) if self.fixed_point_pre_post_timestep_conditioning else y
        
        # Pre-Fixed Point
        # Note: If using DDP with find_unused_parameters=True, checkpoint causes issues. For more 
        # information, see https://github.com/allenai/longformer/issues/63#issuecomment-648861503
        for block in self.blocks_pre:
            x: Float[Tensor, "b t d"] = checkpoint(self.ckpt_wrapper(block), x, c_pre_post_fixed_point) if self.use_gradient_checkpointing else block(x, c_pre_post_fixed_point)
        condition = x.clone()

        # Whether to reuse the previous solution at the next iteration
        init_solution = self.last_solution if (self.fixed_point_reuse_solution and self.last_solution is not None) else x.clone()

        # Fixed Point (we have condition and init_solution)
        x_input_injection = self.block_pre_projection(condition)

        # NOTE: This section of code should have no_grad, but cannot due to a DDP bug. See
        # https://discuss.pytorch.org/t/does-distributeddataparallel-work-with-torch-no-grad-and-find-unused-parameters-false/122594
        # for more information
        with nullcontext():  # we use x.detach() in place of torch.no_grad due to DDP issue
            num_iterations_no_grad = random.randint(self.fixed_point_no_grad_min_iters, self.fixed_point_no_grad_max_iters)
            x = self._forward_fixed_point_blocks(x=init_solution.detach(), x_input_injection=x_input_injection.detach(), c=c, num_iterations=num_iterations_no_grad)
            x = x.detach()  # no grad
        num_iterations_with_grad = random.randint(self.fixed_point_with_grad_min_iters, self.fixed_point_with_grad_max_iters)
        x = self._forward_fixed_point_blocks(x=x, x_input_injection=x_input_injection, c=c, num_iterations=num_iterations_with_grad)

        # Save solution for reuse at next step
        if self.fixed_point_reuse_solution:
            self.last_solution = x.clone()
        
        # Post-Fixed Point
        for block in self.blocks_post:
            x = checkpoint(self.ckpt_wrapper(block), x, c_pre_post_fixed_point) if self.use_gradient_checkpointing else block(x, c_pre_post_fixed_point)
        
        # Output
        x: Float[Tensor, "b t p2c"] = self.final_layer(x, c_pre_post_fixed_point)  # p2c = patch_size ** 2 * out_channels)
        x: Float[Tensor, "b c h w"] = self.unpatchify(x)
        return x
    
    def reset(self):
        self.last_solution = None
    
    def forward(self, x, t, y):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        if self.fixed_point:
            return self._forward_fixed_point(x, t, y)
        else:
            return self._forward_dit(x, t, y)

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}
