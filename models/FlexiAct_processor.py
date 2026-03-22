import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention import Attention
from typing import Optional, List
import numpy as np
import os

class LoRALinearLayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, network_alpha=None, device=None, dtype=None):
        super().__init__()

        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
        # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        self.network_alpha = network_alpha
        self.rank = rank

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        return up_hidden_states.to(orig_dtype)
    

class RefNetLoRAProcessor(nn.Module):
    r"""
    Processor for implementing scaled dot-product attention for the CogVideoX model. It applies a rotary embedding on
    query and key vectors, but does not include spatial normalization.
    """

    def __init__(
            self, 
            dim: int, 
            rank=4, 
            network_alpha=None, 
            lora_weight=1, 
            stage=None,
            # motioninversion
            num_motion_tokens=226,
            is_train=True,
            # attn reweight
            reweight_scale=None,
            vid2embed=1,
            embed2vid=0,
            # visualize
            attn_map_save_path=None,
            cur_layer=0,
            cur_step=0,
            save_step=[10],
            save_layer=[40],
            allow_reweight=True,
        ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CogVideoXAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.lora_weight = lora_weight
        self.lora_q = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.lora_k = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.lora_v = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.lora_proj = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.stage = stage
        self.is_train = is_train
        self.attention_mask = None

        # Attn reweight
        self.reweight_scale = reweight_scale
        self.vid2embed = vid2embed
        self.embed2vid = embed2vid

        # Visualization
        self.attn_map_save_path = attn_map_save_path
        self.cur_layer = cur_layer
        self.cur_step = cur_step
        self.save_step = save_step
        self.save_layer = save_layer

        # Stage 1/2
        if stage is not None:
            self.num_motion_tokens = num_motion_tokens
            self.motion_inversion_tokens = nn.Parameter(torch.zeros(1, num_motion_tokens, 3072))
            nn.init.zeros_(self.motion_inversion_tokens)

        self.allow_reweight = allow_reweight

    def save_attn_map(self, q_vis, k_vis, v_vis, save_step=[10], save_layer=[40]):
        attn_map = torch.zeros_like(v_vis)
        # calculate cur_step and cur_layer
        if not os.path.exists(os.path.join(self.attn_map_save_path, "status.txt")):
            with open(os.path.join(self.attn_map_save_path, "status.txt"), "w") as f:
                f.write(f"0")
                cur_status = 0
        else:
            with open(os.path.join(self.attn_map_save_path, "status.txt"), "r") as f:
                cur_status = int(f.read()) + 1
            # overwrite
            with open(os.path.join(self.attn_map_save_path, "status.txt"), "w") as f:
                f.write(f"{cur_status}")
        cur_step = cur_status // 42
        cur_layer = cur_status % 42

        if cur_step in save_step and cur_layer in save_layer:
            print(f"save attn map at step {cur_step} layer {cur_layer}")
            for i in range(q_vis.shape[1]):
                q_mini = q_vis[:, i:i+1]
                k_mini = k_vis[:, i:i+1]
                attn_map_mini = F.scaled_dot_product_attention(q_mini, k_mini, v_vis, attn_mask=None, dropout_p=0.0, is_causal=False) # [heads, seq_len, seq_len]
                attn_map += attn_map_mini
            save_path = os.path.join(self.attn_map_save_path, f"{cur_step}_{cur_layer}.pt")
            torch.save(attn_map, save_path)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        alow_reweight: bool = True,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)
        if self.stage is not None:
            # FAE
            cat_tokens = self.motion_inversion_tokens.repeat(encoder_hidden_states.size(0), 1, 1)
            encoder_hidden_states = torch.cat([encoder_hidden_states, cat_tokens], dim=1)
            encoder_h_seq_length = encoder_hidden_states.size(1)
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        else:
            # RefAdapter
            encoder_h_seq_length = encoder_hidden_states.size(1)
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states) + self.lora_q(hidden_states) * self.lora_weight
        key = attn.to_k(hidden_states) + self.lora_k(hidden_states) * self.lora_weight
        value = attn.to_v(hidden_states) + self.lora_v(hidden_states) * self.lora_weight

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query[:, :, encoder_h_seq_length:] = apply_rotary_emb(query[:, :, encoder_h_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, encoder_h_seq_length:] = apply_rotary_emb(key[:, :, encoder_h_seq_length:], image_rotary_emb)

        if self.allow_reweight:
            # attn reweight
            if self.reweight_scale is not None:
                q_len = query.shape[-2]
                attention_mask = torch.zeros((q_len, q_len), device=query.device, dtype=query.dtype)
                if self.vid2embed:
                    # enhance vid -> embed
                    attention_mask[encoder_h_seq_length:,text_seq_length:encoder_h_seq_length] += self.reweight_scale
                if self.embed2vid:
                    # enhance embed -> vid
                    attention_mask[text_seq_length:encoder_h_seq_length,encoder_h_seq_length:] += self.reweight_scale

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        if self.attn_map_save_path is not None:
            q_vis = query[1:2].clone().to("cuda")
            k_vis = key[1:2].clone().to("cuda")
            v_vis = torch.eye(q_vis.shape[-2]).to("cuda", dtype=q_vis.dtype)
            v_vis = v_vis.unsqueeze(0).unsqueeze(0)
            q_vis.requires_grad = False
            k_vis.requires_grad = False
            v_vis.requires_grad = False
            self.save_attn_map(q_vis, k_vis, v_vis, save_step=self.save_step, save_layer=self.save_layer)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states) + self.lora_proj(hidden_states) * self.lora_weight
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [encoder_h_seq_length, hidden_states.size(1) - encoder_h_seq_length], dim=1
        )
        encoder_hidden_states = encoder_hidden_states[:, :text_seq_length, :]
        return hidden_states, encoder_hidden_states

    
