from typing import Tuple, Union

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.autoencoders.vq_model import VQEncoderOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils.accelerate_utils import apply_forward_hook
from einops import rearrange
from timm.models.layers import trunc_normal_

from prismatic.action_vqvae.modeling_enc_dec import (
    ActionVQVaeEncoder, # 确保这个导入存在
    DecoderOutput,
    ActionVQVaeDecoder, # 将 Decoder 也导入进来
)
from prismatic.action_vqvae.residual_vq import ResidualVQ
from prismatic.action_vqvae.st_causal_conv import STCausalConvBlock
from prismatic.action_vqvae.vqvae_utils import get_tensor


class ActionVQVAE(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 256,
        num_residual_layers: int = 4,
        downsample_ratio: int = 4,
        # New parameters for STCausalConvBlock
        kernel_size: int = 3,
        # decoder related
        decoder_in_channels: int = 1,
        decoder_out_channels: int = 1,
        decoder_layers_per_block: Tuple[int, ...] = (3, 3, 3, 3),
        decoder_up_block_types: Tuple[str, ...] = (
            "UpDecoderBlockCausal2D",
            "UpDecoderBlockCausal2D",
            "UpDecoderBlockCausal2D",
            "UpDecoderBlockCausal2D",
        ),
        decoder_block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        decoder_block_dropout: Tuple[int, ...] = (0.0, 0.0, 0.0, 0.0),
        decoder_act_fn: str = "silu",
        decoder_norm_num_groups: int = 32,
        decoder_type: str = "causal_vae_conv",
        vq_embed_dim: int = 128,
        num_vq_embeddings: int = 256,
        device: str = "cuda",
        action_window_size: int = 5,
        vq_groups: int = 4,
    ):
        super().__init__()

        # --- Encoder using STCausalConvBlocks ---
        encoder_layers = [STCausalConvBlock(input_dim, hidden_dim, kernel_size=kernel_size)]
        for _ in range(num_residual_layers):
            encoder_layers.append(STCausalConvBlock(hidden_dim, hidden_dim, kernel_size=kernel_size))
        
        # Downsampling
        for i in range(downsample_ratio // 2):
             encoder_layers.append(STCausalConvBlock(hidden_dim, hidden_dim, kernel_size=kernel_size, dilation=2**(i+1)))
             encoder_layers.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=2, stride=2)) # Simple downsampling

        encoder_layers.append(nn.Conv1d(hidden_dim, vq_embed_dim, 1))
        self.encoder = nn.Sequential(*encoder_layers)

        # --- Decoder (assuming it's defined elsewhere and we keep it) ---
        # This part is simplified. You might need to adjust ActionVQVaeDecoder or replace it as well.
        from prismatic.action_vqvae.modeling_enc_dec import ActionVQVaeDecoder

        # pass init params to Decoder
        self.decoder = ActionVQVaeDecoder(
            in_channels=decoder_in_channels,
            out_channels=decoder_out_channels,
            up_block_types=decoder_up_block_types,
            block_out_channels=decoder_block_out_channels,
            layers_per_block=decoder_layers_per_block,
            norm_num_groups=decoder_norm_num_groups,
            act_fn=decoder_act_fn,
            block_dropout=decoder_block_dropout,
            device=device,
            action_window_size=action_window_size,
        )
        self.vq_embed_dim = vq_embed_dim

        self.vqvae_groups = vq_groups
        self.vqvae_n_embed = 256
        discrete_cfg = {"groups": self.vqvae_groups, "n_embed": self.vqvae_n_embed}

        # self.quantize = VectorQuantizer(num_vq_embeddings, vq_embed_dim, beta=0.25, remap=None, sane_index_shape=False)
        # self.quant_conv= nn.Conv2d(latent_channels, vq_embed_dim, 1)
        self.vq_layer = ResidualVQ(
            dim=self.vq_embed_dim,
            num_quantizers=discrete_cfg["groups"],
            codebook_size=self.vqvae_n_embed,
            kmeans_init=True,
            # sync_codebook = False, # important! if not set, loss will be different when the number of gpus are different
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def preprocess(self, state):
        if not torch.is_tensor(state):
            state = get_tensor(state, self.device)
        if state.ndimension() == 2:
            state = state.unsqueeze(0) # B, T*A -> 1, T*A
        if state.ndimension() == 3 and state.shape[1] != self.config.input_dim:
             state = rearrange(state, "b t a -> b a t")
        return state.to(self.device)

    @apply_forward_hook
    def encode(self, x: torch.Tensor, return_dict: bool = True) -> VQEncoderOutput:
        x = self.preprocess(x)
        h = self.encoder(x)
        h = h.reshape(h.shape[0], -1)
        if not return_dict:
            return (h,)

        return VQEncoderOutput(latents=h)

    @apply_forward_hook
    def decode(
        self,
        h: torch.Tensor,
        robot_type=None,
        frequency=None,
        force_not_quantize: bool = False,
        return_dict: bool = True,
        shape=None,
    ) -> Union[DecoderOutput, torch.Tensor]:
        dec = self.decoder(h, robot_type, frequency)
        return dec
    
    def forward(
        self, sample: torch.Tensor, robot_type=None, frequency=None, return_dict: bool = True
    ) -> Union[DecoderOutput, Tuple[torch.Tensor, ...]]:
        r"""
        Args:
            sample (`torch.Tensor`): Input sample.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.autoencoders.vq_model.VQEncoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.autoencoders.vq_model.VQEncoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.autoencoders.vq_model.VQEncoderOutput`] is returned, otherwise a
                plain `tuple` is returned.
        """
        # The forward pass is simplified to return components needed by the wrapper
        x = self.preprocess(sample)
        
        # Encode
        encoded_latents = self.encoder(x)

        # Quantize
        quantized_latents, vq_codes, commit_loss = self.vq_layer(encoded_latents)

        # Decode
        # The decoder expects a 2D latent vector (B, D). We squeeze the length dimension.
        quantized_latents_for_decoder = quantized_latents.squeeze(-1)
        reconstructed_actions = self.decoder(quantized_latents_for_decoder)

        # The wrapper will now handle loss calculation.
        # We return the reconstructed action and the commitment loss.
        return reconstructed_actions, commit_loss

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value
        

class ActionVQVAEPE(ModelMixin, ConfigMixin):

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        # encoder related parameters
        input_dim: int = 7,
        hidden_dim: int = 256,
        num_residual_layers: int = 4,
        downsample_ratio: int = 4,
        kernel_size: int = 3,
        # decoder related
        decoder_in_channels: int = 1,
        decoder_out_channels: int = 1,
        decoder_layers_per_block: Tuple[int, ...] = (3, 3, 3, 3),
        decoder_up_block_types: Tuple[str, ...] = (
            "UpDecoderBlockCausal2D",
            "UpDecoderBlockCausal2D",
            "UpDecoderBlockCausal2D",
            "UpDecoderBlockCausal2D",
        ),
        decoder_block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        decoder_block_dropout: Tuple[int, ...] = (0.0, 0.0, 0.0, 0.0),
        decoder_act_fn: str = "silu",
        decoder_norm_num_groups: int = 32,
        decoder_type: str = "causal_vae_conv",
        vq_embed_dim: int = 128,
        num_vq_embeddings: int = 256,
        num_frequencies: int = 10,
        min_freq: float = 0.0,
        max_freq: float = 8.0,
        action_dim: int = 7, # Add action_dim to config
        temporal_compression_ratio: int = 5,
        device: str = "cuda",
        use_action_type_pe: bool = False,
        use_time_pe: bool = False,
    ):
        super().__init__()

        # pass init params to Encoder

        self.num_frequencies = num_frequencies
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.temporal_compression_ratio = temporal_compression_ratio
        encoder_in_channels = num_frequencies * 2 + 1

        # assume temporal_compression_ratio == T
        freqs = 2 ** torch.linspace(self.min_freq, self.max_freq, self.num_frequencies)
        time_emb_ = (
            torch.arange(self.temporal_compression_ratio).float() / self.temporal_compression_ratio * 2 * torch.pi
        )
        time_emb = time_emb_[..., None] * freqs
        time_emb = torch.sin(torch.cat([time_emb, time_emb + torch.pi / 2.0], dim=-1))
        time_emb = torch.cat([time_emb, time_emb_[..., None]], dim=-1)
        self.register_buffer("time_emb", time_emb[None, ..., None])

        self.xyz_emb = nn.Parameter(torch.randn(1, encoder_in_channels, 1, 3), requires_grad=True)
        self.euler_emb = nn.Parameter(torch.randn(1, encoder_in_channels, 1, 3), requires_grad=True)
        self.gripper_emb = nn.Parameter(torch.randn(1, encoder_in_channels, 1, 1), requires_grad=True)

        self.use_action_type_pe = use_action_type_pe
        self.use_time_pe = use_time_pe

        # --- Encoder using STCausalConvBlocks for ActionVQVAEPE ---
        # The input dimension is now different due to positional embeddings
        # After reshape, the input channel will be encoder_in_channels * action_dim
        encoder_input_dim = encoder_in_channels * action_dim
        encoder_layers = [STCausalConvBlock(encoder_input_dim, hidden_dim, kernel_size=kernel_size)]
        for _ in range(num_residual_layers):
            encoder_layers.append(STCausalConvBlock(hidden_dim, hidden_dim, kernel_size=kernel_size))
        
        for i in range(downsample_ratio // 2):
             encoder_layers.append(STCausalConvBlock(hidden_dim, hidden_dim, kernel_size=kernel_size, dilation=2**(i+1)))
             encoder_layers.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=2, stride=2))

        encoder_layers.append(nn.Conv1d(hidden_dim, vq_embed_dim, 1))
        self.encoder = nn.Sequential(*encoder_layers)
        # pass init params to Decoder
        self.decoder = ActionVQVaeDecoder(
            in_channels=decoder_in_channels,
            out_channels=decoder_out_channels,
            up_block_types=decoder_up_block_types,
            block_out_channels=decoder_block_out_channels,
            layers_per_block=decoder_layers_per_block,
            norm_num_groups=decoder_norm_num_groups,
            act_fn=decoder_act_fn,
            block_dropout=decoder_block_dropout,
            device=device,
        )
        self.vq_embed_dim = vq_embed_dim

        self.vqvae_groups = 4  # 4
        self.vqvae_n_embed = 256  # 256
        discrete_cfg = {"groups": self.vqvae_groups, "n_embed": self.vqvae_n_embed}

        # self.quantize = VectorQuantizer(num_vq_embeddings, vq_embed_dim, beta=0.25, remap=None, sane_index_shape=False)
        # self.quant_conv= nn.Conv2d(latent_channels, vq_embed_dim, 1)
        self.vq_layer = ResidualVQ(
            dim=self.vq_embed_dim,
            num_quantizers=discrete_cfg["groups"],
            codebook_size=self.vqvae_n_embed,
            kmeans_init=True,
            # sync_codebook = False, # important! if not set, loss will be different when the number of gpus are different
        )

        self.apply(self._init_weights)

        # self.start_event = torch.cuda.Event(enable_timing=True)
        # self.end_event = torch.cuda.Event(enable_timing=True)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def apply_action_type_emb(self, action):
        if action.ndim == 3:
            action = rearrange(action, "b t n -> b 1 t n")
        b, c, t, n = action.shape
        assert n == 7, n
        action = action + torch.cat([self.xyz_emb, self.euler_emb, self.gripper_emb], dim=-1)
        return action

    def apply_time_emb(self, action):
        if action.ndim == 3:
            action = rearrange(action, "b t n -> b 1 t n")
        b, c, t, n = action.shape  # c = 21, t = 8, n = 7
        assert n == 7, n
        action = rearrange(action, "b c t n -> b t c n")
        action = self.time_emb.to(dtype=action.dtype, device=action.device) + action
        return rearrange(action, "b t c n -> b c t n", b=b, t=t, n=n)

    def preprocess(self, action):
        if not torch.is_tensor(action):
            action = get_tensor(action, self.device)
        if action.ndimension() == 2:
            action = action.unsqueeze(0)
        action = action.to(self.device)
        if self.use_action_type_pe:
            action = self.apply_action_type_emb(action)
        if self.use_time_pe:
            action = self.apply_time_emb(action)
        # action = einops.rearrange(action, "N T A -> N (T A)")
        return action  # .to(self.device)

    @apply_forward_hook
    def encode(self, x: torch.Tensor, return_dict: bool = True) -> VQEncoderOutput:
        x = self.preprocess(x)
        h = self.encoder(x)
        # h = self.quant_conv(h)
        h = h.reshape(h.shape[0], -1)
        if not return_dict:
            return (h,)

        return VQEncoderOutput(latents=h)

    @apply_forward_hook
    def decode(
        self,
        h: torch.Tensor,
        robot_type=None,
        frequency=None,
        force_not_quantize: bool = False,
        return_dict: bool = True,
        shape=None,
    ) -> Union[DecoderOutput, torch.Tensor]:
        # h = self.post_quant_conv(h)
        dec = self.decoder(h, robot_type, frequency)

        return dec  # , commit_loss

    def forward(
        self, sample: torch.Tensor, robot_type=None, frequency=None, return_dict: bool = True
    ) -> Union[DecoderOutput, Tuple[torch.Tensor, ...]]:
        r"""
        Args:
            sample (`torch.Tensor`): Input sample.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.autoencoders.vq_model.VQEncoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.autoencoders.vq_model.VQEncoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.autoencoders.vq_model.VQEncoderOutput`] is returned, otherwise a
                plain `tuple` is returned.
        """
        # The forward pass is simplified to return components needed by the wrapper
        x = self.preprocess(sample)
        
        # Reshape the 4D tensor from preprocess to 3D for our STCausalConvBlock encoder
        if x.ndim == 4:
            # (B, C, T, A) -> (B, C*A, T)
            x = rearrange(x, "b c t a -> b (c a) t")

        # Encode
        encoded_latents = self.encoder(x)

        # Permute dimensions for ResidualVQ layer: (B, D, L) -> (B, L, D)
        encoded_latents = encoded_latents.permute(0, 2, 1)

        # Quantize
        quantized_latents, vq_codes, commit_loss = self.vq_layer(encoded_latents)

        # Permute back and reshape for the decoder: (B, L, D) -> (B, D, L) -> (B, D, H, W)
        quantized_latents = quantized_latents.permute(0, 2, 1)
        # Decode
        # The decoder expects a 2D latent vector (B, D). We squeeze the length dimension.
        quantized_latents_for_decoder = quantized_latents.squeeze(-1)
        reconstructed_actions = self.decoder(quantized_latents_for_decoder)

        # The wrapper will now handle loss calculation.
        # We return the reconstructed action and the commitment loss.
        return reconstructed_actions, commit_loss

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value
