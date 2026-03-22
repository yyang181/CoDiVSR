from typing import Any, Dict, Optional, Tuple, Union, List, Callable

import torch, os, math
from torch import nn
from PIL import Image
from tqdm import tqdm

from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXBlock, CogVideoXTransformer3DModel

from diffusers.pipelines.cogvideo.pipeline_cogvideox import CogVideoXPipeline, CogVideoXPipelineOutput
from diffusers.pipelines.cogvideo.pipeline_cogvideox_image2video import CogVideoXImageToVideoPipeline
from diffusers.pipelines.cogvideo.pipeline_cogvideox_video2video import CogVideoXVideoToVideoPipeline
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.pipelines.cogvideo.pipeline_cogvideox import retrieve_timesteps
from transformers import T5EncoderModel, T5Tokenizer
from diffusers.models import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel
from diffusers.schedulers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler
from diffusers.pipelines import DiffusionPipeline   
from diffusers.models.modeling_utils import ModelMixin

### support flexiact refadapter
from models.FlexiAct_processor import RefNetLoRAProcessor
from models.det_processor import SkipConv1dCogVideoXAttnProcessor2_0
from diffusers.models.attention_processor import AttnProcessor2_0 

### support tlc
import numpy as np
from numpy import pi, exp, sqrt

### support low pass filter
from models import lp_utils
from einops import rearrange

### support spatio-temporal resblock
from models.resnet_vividvr import SpatioTemporalResBlock
from diffusers.models.embeddings import CogVideoXPatchEmbed

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

def tensor2latent(t, vae):
    video_length = t.shape[2]
    t = rearrange(t, "b c f h w -> (b f) c h w")
    chunk_size = 1
    latents_list = []
    for ind in range(0,t.shape[0],chunk_size):
        latents_list.append(vae.encode(t[ind:ind+chunk_size]).latent_dist.sample())
    latents = torch.cat(latents_list, dim=0)
    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
    latents = latents * vae.config.scaling_factor
    return latents

def temporal_vae_decode(vae, z, num_f):
    return vae.decode(z/vae.config.scaling_factor, num_frames=num_f).sample

    

def vae_decode_chunk(vae, z, chunk_size=3):
    z = rearrange(z, "b c f h w -> (b f) c h w")
    video = []
    for ind in range(0, z.shape[0], chunk_size):
        num_f = z[ind:ind+chunk_size].shape[0]
        video.append(temporal_vae_decode(vae, z[ind:ind+chunk_size],num_f))
    video = torch.cat(video)
    return video

def fourier_transform(x, balance=None):
    """
    Apply Fourier transform to the input tensor and separate it into low-frequency and high-frequency components.

    Args:
    x (torch.Tensor): Input tensor of shape [batch_size, channels, height, width]
    balance (torch.Tensor or float, optional): Learnable balance parameter for adjusting the cutoff frequency.

    Returns:
    low_freq (torch.Tensor): Low-frequency components (with real and imaginary parts)
    high_freq (torch.Tensor): High-frequency components (with real and imaginary parts)
    """
    # Perform 2D Real Fourier transform (rfft2 only computes positive frequencies)
    x = x.to(torch.float32)
    fft_x = torch.fft.rfft2(x, dim=(-2, -1))
    
    # Calculate magnitude of frequency components
    magnitude = torch.abs(fft_x)

    # Set cutoff based on balance or default to the 80th percentile of the magnitude for low frequency
    if balance is None:
        # Downsample the magnitude to reduce computation for large tensors
        subsample_size = 10000  # Adjust based on available memory and tensor size
        if magnitude.numel() > subsample_size:
            # Randomly select a subset of values to approximate the quantile
            magnitude_sample = magnitude.flatten()[torch.randint(0, magnitude.numel(), (subsample_size,))]
            cutoff = torch.quantile(magnitude_sample, 0.8)  # 80th percentile for low frequency
        else:
            cutoff = torch.quantile(magnitude, 0.8)  # 80th percentile for low frequency
    else:
        # balance is clamped for safety and used to scale the mean-based cutoff
        cutoff = magnitude.mean() * (1 + 10 * balance)

    # Smooth mask using sigmoid to ensure gradients can pass through
    sharpness = 10  # A parameter to control the sharpness of the transition
    low_freq_mask = torch.sigmoid(sharpness * (cutoff - magnitude))
    
    # High-frequency mask can be derived from low-frequency mask (1 - low_freq_mask)
    high_freq_mask = 1 - low_freq_mask
    
    # Separate low and high frequencies using smooth masks
    low_freq = fft_x * low_freq_mask
    high_freq = fft_x * high_freq_mask

    # Return real and imaginary parts separately
    low_freq = torch.stack([low_freq.real, low_freq.imag], dim=-1)
    high_freq = torch.stack([high_freq.real, high_freq.imag], dim=-1)
    
    return low_freq, high_freq


def extract_frequencies(video: torch.Tensor, balance=None):
    """
    Extract high-frequency and low-frequency components of a video using Fourier transform.

    Args:
    video (torch.Tensor): Input video tensor of shape [batch_size, channels, frames, height, width]

    Returns:
    low_freq (torch.Tensor): Low-frequency components of the video
    high_freq (torch.Tensor): High-frequency components of the video
    """
    # batch_size, channels, frames, _, _ = video.shape
    video = rearrange(video, 'b c t h w -> (b t) c h w')  # Reshape for Fourier transform

    # Apply Fourier transform to each frame
    low_freq, high_freq = fourier_transform(video, balance=balance)

    return low_freq, high_freq

# def find_nearest_timestep(sqrt_alpha_bar_vals, scheduler):
#     """
#     参数:
#         sqrt_alpha_bar_vals: 1D tensor, 形状为 [B]
#         scheduler.alphas_cumprod: 1D tensor, 形状为 [T]

#     返回:
#         nearest_timesteps: 1D tensor, 形状为 [B]，每个值是最接近的 timestep 索引
#     """
#     # 暂存输入设备
#     device = sqrt_alpha_bar_vals.device

#     # 放到 CPU 上计算
#     sqrt_alpha_bar_vals_cpu = sqrt_alpha_bar_vals.cpu()
#     sqrt_alpha_bars_cpu = torch.sqrt(scheduler.alphas_cumprod.cpu())  # shape: [T]

#     # 计算每个 batch 元素与所有 timesteps 的差值，输出形状为 [B, T]
#     diffs = torch.abs(sqrt_alpha_bars_cpu[None, :] - sqrt_alpha_bar_vals_cpu[:, None])  # [B, T]

#     # 找出最小差值对应的索引
#     nearest_timesteps = torch.argmin(diffs, dim=1)  # [B]

#     # 放回原设备
#     return nearest_timesteps.to(device)

def find_nearest_timestep(sqrt_alpha_bar_vals, scheduler):
    """
    参数:
        sqrt_alpha_bar_vals: float、标量 Tensor 或 1D Tensor, 形状为 [B]
        scheduler.alphas_cumprod: 1D tensor, 形状为 [T]

    返回:
        nearest_timesteps: 1D tensor, 形状为 [B] 或标量 tensor, 表示最接近的 timestep 索引
    """
    # 放到 CPU 上统一计算
    sqrt_alpha_bars_cpu = torch.sqrt(scheduler.alphas_cumprod.cpu())  # shape: [T]

    if isinstance(sqrt_alpha_bar_vals, torch.Tensor):
        device = sqrt_alpha_bar_vals.device
        sqrt_alpha_bar_vals_cpu = sqrt_alpha_bar_vals.detach().cpu()

        # 如果是 scalar tensor（0-dim 或长度为1），视为单个值
        if sqrt_alpha_bar_vals_cpu.dim() == 0:
            sqrt_alpha_bar_vals_cpu = sqrt_alpha_bar_vals_cpu.unsqueeze(0)

    else:
        # 如果是 float，转换为 tensor
        sqrt_alpha_bar_vals_cpu = torch.tensor([sqrt_alpha_bar_vals], dtype=torch.float32)
        device = 'cpu'
        is_scalar = True

    # 计算差异矩阵 [B, T]
    diffs = torch.abs(sqrt_alpha_bars_cpu[None, :] - sqrt_alpha_bar_vals_cpu[:, None])
    nearest_timesteps = torch.argmin(diffs, dim=1)  # [B]

    return nearest_timesteps.to(device)




def prepare_lp(
    # --- Filter Selection & Strength ---
    lp_filter_type: str,
    lp_resize_factor: float,
    orig_image_latents: torch.Tensor, # Shape [B, F_padded, C, H, W]
) -> torch.Tensor | None:
    # --- Filter in Latent Space ---
    lp_image_latents = lp_utils.apply_low_pass_filter_v1(
        orig_image_latents, # Input has shape [B, F_padded, C, H, W]
        filter_type=lp_filter_type,
        blur_sigma=0.0,
        blur_kernel_size=0.0,
        resize_factor=lp_resize_factor,
    )

    lp_image_latents = lp_image_latents.to(dtype=orig_image_latents.dtype)

    return lp_image_latents

class LocalAttention3D:
    def __init__(self, kernel_size=(3, 128, 128), overlap=(0.5, 0.5, 0.5)):
        super().__init__()
        self.kernel_size = kernel_size
        self.overlap = overlap
        
    def grids(self, x):
        b, c, f, h, w = x.shape
        self.original_size = (b, c, f, h, w)
        kf, kh, kw = self.kernel_size
        
        # 防止kernel超出边界
        
        kf = min(kf, f)
        kh = min(kh, h)
        kw = min(kw, w)
        # print(f"h, w {h, w} self.original_size: {self.original_size} kf: {kf} kh: {kh} kw: {kw}")
        self.tile_weights = self._gaussian_weights(kf, kh, kw)

        # 计算步长
        step_f = kf if f == kf else max(1, int(kf * self.overlap[0]))
        step_h = kh if h == kh else max(1, int(kh * self.overlap[1]))
        step_w = kw if w == kw else max(1, int(kw * self.overlap[2]))
        
        parts = []
        idxes = []
        # print(f"step_f: {step_f} step_h: {step_h} step_w: {step_w}")
        for fi in range(0, f, step_f):
            if fi + kf > f:
                fi = f - kf
            for hi in range(0, h, step_h):
                if hi + kh > h:
                    hi = h - kh
                for wi in range(0, w, step_w):
                    if wi + kw > w:
                        wi = w - kw
                    parts.append(x[:, :, fi:fi + kf, hi:hi + kh, wi:wi + kw])
                    # print(f"shape {x[:, :, fi:fi + kf, hi:hi + kh, wi:wi + kw].shape}")
                    idxes.append({'f': fi, 'h': hi, 'w': wi})

        self.idxes = idxes
        return torch.cat(parts, dim=0)

    def _gaussian_weights(self, tile_depth, tile_height, tile_width):
        var = 0.01
        midpoint_d = (tile_depth - 1) / 2
        midpoint_h = (tile_height - 1) / 2
        midpoint_w = (tile_width - 1) / 2
        
        # 计算各个维度上的权重
        d_probs = [exp(-(d-midpoint_d)*(d-midpoint_d)/(tile_depth*tile_depth)/(2*var)) / sqrt(2*pi*var) for d in range(tile_depth)]
        h_probs = [exp(-(h-midpoint_h)*(h-midpoint_h)/(tile_height*tile_height)/(2*var)) / sqrt(2*pi*var) for h in range(tile_height)]
        w_probs = [exp(-(w-midpoint_w)*(w-midpoint_w)/(tile_width*tile_width)/(2*var)) / sqrt(2*pi*var) for w in range(tile_width)]

        # 生成3D高斯权重
        weights = np.outer(np.outer(d_probs, h_probs).reshape(-1), w_probs).reshape(tile_depth, tile_height, tile_width)
        return torch.tensor(weights, device=torch.device('cuda')).unsqueeze(0).repeat(16, 1, 1, 1)

    def grids_inverse(self, outs):
        preds = torch.zeros(self.original_size).to(outs.device)
        b, c, f, h, w = self.original_size
        count_mt = torch.zeros((b, 16, f, h, w)).to(outs.device)
        kf, kh, kw = self.kernel_size

        for cnt, each_idx in enumerate(self.idxes):
            fi = each_idx['f']
            hi = each_idx['h']
            wi = each_idx['w']
            preds[:, :, fi:fi + kf, hi:hi + kh, wi:wi + kw] += outs[cnt, :, :, :, :] * self.tile_weights
            count_mt[:, :, fi:fi + kf, hi:hi + kh, wi:wi + kw] += self.tile_weights

        del outs
        torch.cuda.empty_cache()
        return preds / count_mt

    def forward(self, x):
        qkv = self.grids(x)
        out = self.grids_inverse(qkv)
        return out
    

class CogVideoXTransformer3DModelVividVR(CogVideoXTransformer3DModel, ModelMixin):
    """
    Add tracking maps to the CogVideoX transformer model.

    Parameters:
        num_tracking_blocks (`int`, defaults to `18`):
            The number of tracking blocks to use. Must be less than or equal to num_layers.
    """

    def __init__(
        self,
        num_attention_heads: int = 30,
        attention_head_dim: int = 64,
        in_channels: int = 16,
        out_channels: Optional[int] = 16,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        time_embed_dim: int = 512,
        text_embed_dim: int = 4096,
        num_layers: int = 30,
        dropout: float = 0.0,
        attention_bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 49,
        patch_size: int = 2,
        patch_size_t: Optional[int] = None,
        temporal_compression_ratio: int = 4,
        max_text_seq_length: int = 226,
        activation_fn: str = "gelu-approximate",
        timestep_activation_fn: str = "silu",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        use_rotary_positional_embeddings: bool = False,
        use_learned_positional_embeddings: bool = False,
        patch_bias: bool = True,
        **kwargs
    ):
        super().__init__(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            flip_sin_to_cos=flip_sin_to_cos,
            freq_shift=freq_shift,
            time_embed_dim=time_embed_dim,
            text_embed_dim=text_embed_dim,
            num_layers=num_layers,
            dropout=dropout,
            attention_bias=attention_bias,
            sample_width=sample_width,
            sample_height=sample_height,
            sample_frames=sample_frames,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            temporal_compression_ratio=temporal_compression_ratio,
            max_text_seq_length=max_text_seq_length,
            activation_fn=activation_fn,
            timestep_activation_fn=timestep_activation_fn,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            spatial_interpolation_scale=spatial_interpolation_scale,
            temporal_interpolation_scale=temporal_interpolation_scale,
            use_rotary_positional_embeddings=use_rotary_positional_embeddings,
            use_learned_positional_embeddings=use_learned_positional_embeddings,
            patch_bias=patch_bias,
            **kwargs
        )

        inner_dim = num_attention_heads * attention_head_dim

        ### Initialize tracking blocks
        # self.num_tracking_blocks = num_tracking_blocks

        # # Ensure num_tracking_blocks is not greater than num_layers
        # if num_tracking_blocks > num_layers:
        #     raise ValueError("num_tracking_blocks must be less than or equal to num_layers")

        # # Create linear layers for combining hidden states and tracking maps
        # self.combine_linears = nn.ModuleList(
        #     [nn.Linear(inner_dim, inner_dim, device="cpu") for _ in range(num_tracking_blocks)]
        # )

        # # Initialize weights of combine_linears to zero
        # for linear in self.combine_linears:
        #     linear.weight.data.zero_()
        #     linear.bias.data.zero_()

        # # Create transformer blocks for processing tracking maps
        # self.transformer_blocks_copy = nn.ModuleList(
        #     [
        #         CogVideoXBlock(
        #             dim=inner_dim,
        #             num_attention_heads=self.config.num_attention_heads,
        #             attention_head_dim=self.config.attention_head_dim,
        #             time_embed_dim=self.config.time_embed_dim,
        #             dropout=self.config.dropout,
        #             activation_fn=self.config.activation_fn,
        #             attention_bias=self.config.attention_bias,
        #             norm_elementwise_affine=self.config.norm_elementwise_affine,
        #             norm_eps=self.config.norm_eps,
        #         ).to_empty(device="cpu")
        #         for _ in range(num_tracking_blocks)
        #     ]
        # )


        ### Initialize control feature projector and patch embedding
        self.control_feat_proj = nn.ModuleList([
            SpatioTemporalResBlock(in_channels, 320, time_embed_dim, merge_strategy="learned", groups=16),
            SpatioTemporalResBlock(320, 320, time_embed_dim, merge_strategy="learned", groups=32),
            SpatioTemporalResBlock(320, in_channels, time_embed_dim, merge_strategy="learned", groups=16)
        ])
        
        # Unfreeze parameters that need to be trained
        for proj in self.control_feat_proj:
            for param in proj.parameters():
                param.requires_grad = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        ofs: Optional[Union[int, float, torch.LongTensor]] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        args: Optional[Dict[str, Any]] = None,
    ):
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_frames, channels, height, width = hidden_states.shape

        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        if self.ofs_embedding is not None:
            ofs_emb = self.ofs_proj(ofs)
            ofs_emb = ofs_emb.to(dtype=hidden_states.dtype)
            ofs_emb = self.ofs_embedding(ofs_emb)
            emb = emb + ofs_emb


        # 2. Patch embedding
        ### add control feature projector
        if 1:
            control_states = hidden_states
            B, F, C, H, W = control_states.shape
            # print(f"control_states: {control_states.shape}, emb: {emb.shape}") # control_states: torch.Size([4, 14, 32, 60, 90]), emb: torch.Size([4, 512]) 
            control_states = rearrange(control_states, "B F C H W -> (B F) C H W")
            res_emb = emb.repeat(F, 1)
            # print(f"control_states: {control_states.shape}, res_emb: {res_emb.shape}") # control_states: torch.Size([32, 16, 128, 128]), res_emb: torch.Size([32, 512])
            # print(f"control_states: {control_states.shape}, res_emb: {res_emb.shape}") # control_states: torch.Size([56, 32, 60, 90]), res_emb: torch.Size([224, 512]) 
            # print(control_states.dtype, res_emb.dtype) # torch.bfloat16 torch.bfloat16
            for module in self.control_feat_proj:
                control_states = module(control_states, res_emb, torch.ones((F), device=control_states.device))
            control_states = rearrange(control_states, "(B F) C H W -> B F C H W", B=B, F=F)
            control_states = hidden_states + control_states
            control_states = self.patch_embed(encoder_hidden_states, control_states)
            control_states = self.embedding_dropout(control_states)
            hidden_states = control_states
        else:
            hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
            hidden_states = self.embedding_dropout(hidden_states)


        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        # 3. Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    emb,
                    image_rotary_emb,
                    attention_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                    attention_kwargs=attention_kwargs,
                )

        hidden_states = self.norm_final(hidden_states)

        # 4. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        p = self.config.patch_size
        p_t = self.config.patch_size_t

        if p_t is None:
            output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
            output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
        else:
            output = hidden_states.reshape(
                batch_size, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
            )
            output = output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        # 把你需要的参数传给 kwargs，让 ModelMixin 在构建模型时传给 __init__, 注意需要再init添加这些参数，否则传递参数时会被忽略
        # load_refadapter: bool = False, load_skipconv1d: bool = False, 
        # kwargs["load_refadapter"] = load_refadapter
        # kwargs["load_skipconv1d"] = load_skipconv1d
        load_refadapter = kwargs.get("load_refadapter", False)
        load_skipconv1d = kwargs.get("load_skipconv1d", False)

        try:
            model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
            print("✅ Loaded DiffusionAsShader checkpoint directly.")
            
            if 0:
                assert not (load_refadapter and load_skipconv1d), "Cannot load both RefAdapter and SkipConv1D at the same time. Please choose one."
                if load_refadapter:
                    print("🔄 DiffusionAsShader: Loading RefAdapter processor from FlexiAct...")
                    # 复制当前 attn_processors
                    flexiact_procs = dict(model.attn_processors)  # or deepcopy if needed
                    modified_count = 0
                    for name, module in model.attn_processors.items():
                        # print(name) # 42 transformer_blocks and 18 transformer_blocks_copy
                        # print(f"{name}: {type(module)}") # transformer_blocks.18.attn1.processor: <class 'diffusers.models.attention_processor.CogVideoXAttnProcessor2_0'>
                        ### only add lora to base transformer_blocks
                        if 'transformer_blocks_copy' not in name: 
                            modified_count += 1
                            flexiact_procs[name] = RefNetLoRAProcessor(
                                dim=3072, 
                                rank=64, 
                                network_alpha=32, 
                                lora_weight=0.7,
                                stage=None,
                                allow_reweight=False,
                            )
                    flexiact_procs = flexiact_procs.copy() # make a copy to protect the original dict. transformer.set_attn_processor would pop out contents in dict.
                    print(f"✅ DiffusionAsShader: Loaded RefNetLoRA processor with {len(flexiact_procs)} processors, {modified_count} processors modified, first 5 and last 5 keys are: {list(flexiact_procs.keys())[:5]} ... {list(flexiact_procs.keys())[-5:]}")
                    model.set_attn_processor(flexiact_procs)
                    # assert 0

                    print("🔄 DiffusionAsShader: Loading RefAdapter state dict from FlexiAct...")
                    refadapter_ckpt_path = os.path.abspath('checkpoints/refnetlora_step40000_model.pt')
                    refadapter_state_dict = torch.load(refadapter_ckpt_path)
                    load_result = model.load_state_dict(refadapter_state_dict, strict=False)
                    loaded_keys = [k for k in refadapter_state_dict.keys() if k not in load_result.unexpected_keys]
                    print(f"✅ DiffusionAsShader: Loaded RefAdapter from: {refadapter_ckpt_path}")
                    print(f"🔑 DiffusionAsShader: Total keys in refadapter_ckpt_path: {len(refadapter_state_dict)}")
                    print(f"🟢 DiffusionAsShader: Loaded keys into model: {len(loaded_keys)}")
                    # for key in loaded_keys:
                    #     print(f"    [LOADED] {key}")
                    # if load_result.missing_keys:
                    #     print(f"⚠️  Missing keys in checkpoint (not found in model):")
                    #     for k in load_result.missing_keys:
                    #         print(f"    [MISSING] {k}")
                    if load_result.unexpected_keys:
                        print(f"⚠️ DiffusionAsShader: Unexpected keys in checkpoint (model has no matching param):")
                        for k in load_result.unexpected_keys:
                            print(f" DiffusionAsShader: [UNEXPECTED] {k}")
                    assert len(load_result.unexpected_keys) == 0, "Unexpected keys found in the checkpoint. Please check the model architecture and the checkpoint."
                    assert len(loaded_keys) == len(refadapter_state_dict), "Not all keys were loaded from the checkpoint. Please check the model architecture and the checkpoint."
                
                if load_skipconv1d:
                    print("🔄 DiffusionAsShader: Loading SkipConv1D processor from DeT...")
                    # 复制当前 attn_processors
                    height = model.config.sample_height // model.config.patch_size
                    width = model.config.sample_width // model.config.patch_size
                    frames = model.config.sample_frames // model.config.temporal_compression_ratio + 1
                    dim = model.config.num_attention_heads * model.config.attention_head_dim

                    # print(f"height: {height}, width: {width}, frames: {frames}, dim: {dim}");assert 0 # height: 30, width: 45, frames: 13, dim: 3072 
                    det_processors = {}
                    # for name, module in model.attn_processors.items():
                        # print(name) # 42 transformer_blocks and 18 transformer_blocks_copy
                        # print(f"{name}: {type(module)}") # transformer_blocks.18.attn1.processor: <class 'diffusers.models.attention_processor.CogVideoXAttnProcessor2_0'>
                    modified_count = 0
                    for key, value in model.attn_processors.items():
                        if 'transformer_blocks_copy' in key: 
                            modified_count += 1
                            det_processors[key] = SkipConv1dCogVideoXAttnProcessor2_0(
                                height=height,
                                width=width,
                                frames=frames,
                                dim=dim,
                                rank=128,
                                kernel_size=3,
                                module_type='conv1d',
                            ).to(dtype=model.dtype)
                            # print(f"✅ CogVideoXTransformer3DModelTracking __from_pretrained__ Exception: Loaded SkipConv1D processor for {key} with type {type(det_processors[key])}")
                        else:
                            det_processors[key] = value
                            # print(f"✅ CogVideoXTransformer3DModelTracking __from_pretrained__ Exception: Loaded SkipConv1D processor for {key} with type {type(det_processors[key])}")

                    det_processors = det_processors.copy() # make a copy to protect the original dict. transformer.set_attn_processor would pop out contents in dict.
                    print(f"✅ DiffusionAsShader: Loaded SkipConv1D processor with {len(det_processors)} processors, {modified_count} processors modified")
                    model.set_attn_processor(det_processors)


            for param in model.parameters():
                param.requires_grad = False
            
            for linear in model.combine_linears:
                for param in linear.parameters():
                    param.requires_grad = True
                
            for block in model.transformer_blocks_copy:
                for param in block.parameters():
                    param.requires_grad = True
                
            for param in model.initial_combine_linear.parameters():
                param.requires_grad = True
            
            return model
        
        except Exception as e:
            print(f"Failed to load as DiffusionAsShader: {e}")
            print("Attempting to load as CogVideoXTransformer3DModel and convert...")

            base_model = CogVideoXTransformer3DModel.from_pretrained(pretrained_model_name_or_path, **kwargs)
            
            config = dict(base_model.config)
            config["num_tracking_blocks"] = kwargs.pop("num_tracking_blocks", 18)
            
            model = cls(**config)
            model.load_state_dict(base_model.state_dict(), strict=False)

            model.initial_combine_linear.weight.data.zero_()
            model.initial_combine_linear.bias.data.zero_()
            
            for linear in model.combine_linears:
                linear.weight.data.zero_()
                linear.bias.data.zero_()
            
            for i in range(model.num_tracking_blocks):
                model.transformer_blocks_copy[i].load_state_dict(model.transformer_blocks[i].state_dict())
            
            ### add RefAdapter loading logic
            if 1:
                assert not (load_refadapter and load_skipconv1d), "Cannot load both RefAdapter and SkipConv1D at the same time. Please choose one."
                if load_refadapter:
                    print("🔄 CogVideoXTransformer3DModelTracking __from_pretrained__: Loading RefAdapter processor  from FlexiAct...")
                    # 复制当前 attn_processors
                    flexiact_procs = dict(model.attn_processors)  # or deepcopy if needed
                    modified_count = 0
                    for name, module in model.attn_processors.items():
                        # print(name) # 42 transformer_blocks and 18 transformer_blocks_copy
                        # print(f"{name}: {type(module)}") # transformer_blocks.18.attn1.processor: <class 'diffusers.models.attention_processor.CogVideoXAttnProcessor2_0'>
                        ### only add lora to base transformer_blocks
                        if 'transformer_blocks_copy' not in name: 
                            modified_count += 1
                            flexiact_procs[name] = RefNetLoRAProcessor(
                                dim=3072, 
                                rank=64, 
                                network_alpha=32, 
                                lora_weight=0.7,
                                stage=None,
                                allow_reweight=False,
                            )
                    flexiact_procs = flexiact_procs.copy() # make a copy to protect the original dict. transformer.set_attn_processor would pop out contents in dict.
                    print(f"✅ CogVideoXTransformer3DModelTracking __from_pretrained__ Exception: Loaded RefNetLoRA processor with {len(flexiact_procs)} processors, {modified_count} processors modified, first 5 and last 5 keys are: {list(flexiact_procs.keys())[:5]} ... {list(flexiact_procs.keys())[-5:]}")
                    model.set_attn_processor(flexiact_procs)
                    # assert 0

                    print("🔄 CogVideoXTransformer3DModelTracking __from_pretrained__ Exception: Loading RefAdapter state dict from FlexiAct...")
                    refadapter_ckpt_path = os.path.abspath('checkpoints/refnetlora_step40000_model.pt')
                    refadapter_state_dict = torch.load(refadapter_ckpt_path)
                    load_result = model.load_state_dict(refadapter_state_dict, strict=False)
                    loaded_keys = [k for k in refadapter_state_dict.keys() if k not in load_result.unexpected_keys]
                    print(f"✅ CogVideoXTransformer3DModelTracking __from_pretrained__ Exception: Loaded RefAdapter from: {refadapter_ckpt_path}")
                    print(f"🔑 CogVideoXTransformer3DModelTracking __from_pretrained__ Exception: Total keys in refadapter_ckpt_path: {len(refadapter_state_dict)}")
                    print(f"🟢 CogVideoXTransformer3DModelTracking __from_pretrained__ Exception: Loaded keys into model: {len(loaded_keys)}")
                    # for key in loaded_keys:
                    #     print(f"    [LOADED] {key}")
                    # if load_result.missing_keys:
                    #     print(f"⚠️  Missing keys in checkpoint (not found in model):")
                    #     for k in load_result.missing_keys:
                    #         print(f"    [MISSING] {k}")
                    if load_result.unexpected_keys:
                        print(f"⚠️ CogVideoXTransformer3DModelTracking __from_pretrained__ Exception: Unexpected keys in checkpoint (model has no matching param):")
                        for k in load_result.unexpected_keys:
                            print(f"    [UNEXPECTED] {k}")
                    assert len(load_result.unexpected_keys) == 0, "Unexpected keys found in the checkpoint. Please check the model architecture and the checkpoint."
                    assert len(loaded_keys) == len(refadapter_state_dict), "Not all keys were loaded from the checkpoint. Please check the model architecture and the checkpoint."


                if load_skipconv1d:
                    print("🔄 CogVideoXTransformer3DModelTracking __from_pretrained__ Exception: Loading SkipConv1D processor from DeT...")
                    # 复制当前 attn_processors
                    height = model.config.sample_height // model.config.patch_size
                    width = model.config.sample_width // model.config.patch_size
                    frames = model.config.sample_frames // model.config.temporal_compression_ratio + 1
                    dim = model.config.num_attention_heads * model.config.attention_head_dim

                    # print(f"height: {height}, width: {width}, frames: {frames}, dim: {dim}");assert 0 # height: 30, width: 45, frames: 13, dim: 3072 
                    det_processors = {}
                    # for name, module in model.attn_processors.items():
                        # print(name) # 42 transformer_blocks and 18 transformer_blocks_copy
                        # print(f"{name}: {type(module)}") # transformer_blocks.18.attn1.processor: <class 'diffusers.models.attention_processor.CogVideoXAttnProcessor2_0'>
                    modified_count = 0
                    for key, value in model.attn_processors.items():
                        if 'transformer_blocks_copy' in key: 
                            modified_count += 1
                            det_processors[key] = SkipConv1dCogVideoXAttnProcessor2_0(
                                height=height,
                                width=width,
                                frames=frames,
                                dim=dim,
                                rank=128,
                                kernel_size=3,
                                module_type='conv1d',
                            ).to(dtype=model.dtype)
                            # print(f"✅ CogVideoXTransformer3DModelTracking __from_pretrained__ Exception: Loaded SkipConv1D processor for {key} with type {type(det_processors[key])}")
                        else:
                            det_processors[key] = value
                            # print(f"✅ CogVideoXTransformer3DModelTracking __from_pretrained__ Exception: Loaded SkipConv1D processor for {key} with type {type(det_processors[key])}")

                    det_processors = det_processors.copy() # make a copy to protect the original dict. transformer.set_attn_processor would pop out contents in dict.
                    print(f"✅ CogVideoXTransformer3DModelTracking __from_pretrained__ Exception: Loaded SkipConv1D processor with {len(det_processors)} processors, {modified_count} processors modified")
                    model.set_attn_processor(det_processors)


            for param in model.parameters():
                param.requires_grad = False
            
            for linear in model.combine_linears:
                for param in linear.parameters():
                    param.requires_grad = True
                
            for block in model.transformer_blocks_copy:
                for param in block.parameters():
                    param.requires_grad = True
                
            for param in model.initial_combine_linear.parameters():
                param.requires_grad = True
            
            return model

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        save_function: Optional[Callable] = None,
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        max_shard_size: Union[int, str] = "5GB",
        push_to_hub: bool = False,
        **kwargs,
    ):
        super().save_pretrained(
            save_directory,
            is_main_process=is_main_process,
            save_function=save_function,
            safe_serialization=safe_serialization,
            variant=variant,
            max_shard_size=max_shard_size,
            push_to_hub=push_to_hub,
            **kwargs,
        )
        
        if is_main_process:
            config_dict = dict(self.config)
            config_dict.pop("_name_or_path", None)
            config_dict.pop("_use_default_values", None)
            config_dict["_class_name"] = "CogVideoXTransformer3DModelTracking"
            config_dict["num_tracking_blocks"] = self.num_tracking_blocks
            config_dict["load_refadapter"] = self.load_refadapter
            config_dict["load_skipconv1d"] = self.load_skipconv1d

            os.makedirs(save_directory, exist_ok=True)
            with open(os.path.join(save_directory, "config.json"), "w", encoding="utf-8") as f:
                import json
                json.dump(config_dict, f, indent=2)

 
class CogVideoXTransformer3DModelMidResidual(CogVideoXTransformer3DModel, ModelMixin):
    """
    Add tracking maps to the CogVideoX transformer model.

    Parameters:
        num_tracking_blocks (`int`, defaults to `18`):
            The number of tracking blocks to use. Must be less than or equal to num_layers.
    """

    def __init__(
        self,
        num_attention_heads: int = 30,
        attention_head_dim: int = 64,
        in_channels: int = 16,
        out_channels: Optional[int] = 16,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        time_embed_dim: int = 512,
        text_embed_dim: int = 4096,
        num_layers: int = 30,
        dropout: float = 0.0,
        attention_bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 49,
        patch_size: int = 2,
        patch_size_t: Optional[int] = None,
        temporal_compression_ratio: int = 4,
        max_text_seq_length: int = 226,
        activation_fn: str = "gelu-approximate",
        timestep_activation_fn: str = "silu",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        use_rotary_positional_embeddings: bool = False,
        use_learned_positional_embeddings: bool = False,
        patch_bias: bool = True,
        **kwargs
    ):
        super().__init__(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            flip_sin_to_cos=flip_sin_to_cos,
            freq_shift=freq_shift,
            time_embed_dim=time_embed_dim,
            text_embed_dim=text_embed_dim,
            num_layers=num_layers,
            dropout=dropout,
            attention_bias=attention_bias,
            sample_width=sample_width,
            sample_height=sample_height,
            sample_frames=sample_frames,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            temporal_compression_ratio=temporal_compression_ratio,
            max_text_seq_length=max_text_seq_length,
            activation_fn=activation_fn,
            timestep_activation_fn=timestep_activation_fn,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            spatial_interpolation_scale=spatial_interpolation_scale,
            temporal_interpolation_scale=temporal_interpolation_scale,
            use_rotary_positional_embeddings=use_rotary_positional_embeddings,
            use_learned_positional_embeddings=use_learned_positional_embeddings,
            patch_bias=patch_bias,
            **kwargs
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        ofs: Optional[Union[int, float, torch.LongTensor]] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        args: Optional[Dict[str, Any]] = None,
    ):
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_frames, channels, height, width = hidden_states.shape

        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        if self.ofs_embedding is not None:
            ofs_emb = self.ofs_proj(ofs)
            ofs_emb = ofs_emb.to(dtype=hidden_states.dtype)
            ofs_emb = self.ofs_embedding(ofs_emb)
            emb = emb + ofs_emb


        # 2. Patch embedding
        ### add control feature projector
        if 0:
            control_states = hidden_states
            B, F, C, H, W = control_states.shape
            # print(f"control_states: {control_states.shape}, emb: {emb.shape}") # control_states: torch.Size([4, 14, 32, 60, 90]), emb: torch.Size([4, 512]) 
            control_states = rearrange(control_states, "B F C H W -> (B F) C H W")
            res_emb = emb.repeat(F, 1)
            # print(f"control_states: {control_states.shape}, res_emb: {res_emb.shape}") # control_states: torch.Size([32, 16, 128, 128]), res_emb: torch.Size([32, 512])
            # print(f"control_states: {control_states.shape}, res_emb: {res_emb.shape}") # control_states: torch.Size([56, 32, 60, 90]), res_emb: torch.Size([224, 512]) 
            # print(control_states.dtype, res_emb.dtype) # torch.bfloat16 torch.bfloat16
            for module in self.control_feat_proj:
                control_states = module(control_states, res_emb, torch.ones((F), device=control_states.device))
            control_states = rearrange(control_states, "(B F) C H W -> B F C H W", B=B, F=F)
            control_states = hidden_states + control_states
            control_states = self.patch_embed(encoder_hidden_states, control_states)
            control_states = self.embedding_dropout(control_states)
            hidden_states = control_states
        else:
            hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
            hidden_states = self.embedding_dropout(hidden_states)


        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        # 3. Transformer blocks
        residual_from_21 = None

        # 3. Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    emb,
                    image_rotary_emb,
                    attention_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                    attention_kwargs=attention_kwargs,
                )

            # 保存第21个block的输出
            if i == 20:  
                residual_from_21 = hidden_states

            # 在第42个block加上 residual
            if i == 41 and residual_from_21 is not None:  
                hidden_states = hidden_states + residual_from_21


        hidden_states = self.norm_final(hidden_states)

        # 4. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        p = self.config.patch_size
        p_t = self.config.patch_size_t

        if p_t is None:
            output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
            output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
        else:
            output = hidden_states.reshape(
                batch_size, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
            )
            output = output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)


class CogVideoXTransformer3DModelUnetResidual(CogVideoXTransformer3DModel, ModelMixin):
    """
    Add tracking maps to the CogVideoX transformer model.

    Parameters:
        num_tracking_blocks (`int`, defaults to `18`):
            The number of tracking blocks to use. Must be less than or equal to num_layers.
    """

    def __init__(
        self,
        num_attention_heads: int = 30,
        attention_head_dim: int = 64,
        in_channels: int = 16,
        out_channels: Optional[int] = 16,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        time_embed_dim: int = 512,
        text_embed_dim: int = 4096,
        num_layers: int = 30,
        dropout: float = 0.0,
        attention_bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 49,
        patch_size: int = 2,
        patch_size_t: Optional[int] = None,
        temporal_compression_ratio: int = 4,
        max_text_seq_length: int = 226,
        activation_fn: str = "gelu-approximate",
        timestep_activation_fn: str = "silu",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        use_rotary_positional_embeddings: bool = False,
        use_learned_positional_embeddings: bool = False,
        patch_bias: bool = True,
        **kwargs
    ):
        super().__init__(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            flip_sin_to_cos=flip_sin_to_cos,
            freq_shift=freq_shift,
            time_embed_dim=time_embed_dim,
            text_embed_dim=text_embed_dim,
            num_layers=num_layers,
            dropout=dropout,
            attention_bias=attention_bias,
            sample_width=sample_width,
            sample_height=sample_height,
            sample_frames=sample_frames,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            temporal_compression_ratio=temporal_compression_ratio,
            max_text_seq_length=max_text_seq_length,
            activation_fn=activation_fn,
            timestep_activation_fn=timestep_activation_fn,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            spatial_interpolation_scale=spatial_interpolation_scale,
            temporal_interpolation_scale=temporal_interpolation_scale,
            use_rotary_positional_embeddings=use_rotary_positional_embeddings,
            use_learned_positional_embeddings=use_learned_positional_embeddings,
            patch_bias=patch_bias,
            **kwargs
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        ofs: Optional[Union[int, float, torch.LongTensor]] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        args: Optional[Dict[str, Any]] = None,
    ):
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_frames, channels, height, width = hidden_states.shape

        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        if self.ofs_embedding is not None:
            ofs_emb = self.ofs_proj(ofs)
            ofs_emb = ofs_emb.to(dtype=hidden_states.dtype)
            ofs_emb = self.ofs_embedding(ofs_emb)
            emb = emb + ofs_emb


        # 2. Patch embedding
        ### add control feature projector
        if 0:
            control_states = hidden_states
            B, F, C, H, W = control_states.shape
            # print(f"control_states: {control_states.shape}, emb: {emb.shape}") # control_states: torch.Size([4, 14, 32, 60, 90]), emb: torch.Size([4, 512]) 
            control_states = rearrange(control_states, "B F C H W -> (B F) C H W")
            res_emb = emb.repeat(F, 1)
            # print(f"control_states: {control_states.shape}, res_emb: {res_emb.shape}") # control_states: torch.Size([32, 16, 128, 128]), res_emb: torch.Size([32, 512])
            # print(f"control_states: {control_states.shape}, res_emb: {res_emb.shape}") # control_states: torch.Size([56, 32, 60, 90]), res_emb: torch.Size([224, 512]) 
            # print(control_states.dtype, res_emb.dtype) # torch.bfloat16 torch.bfloat16
            for module in self.control_feat_proj:
                control_states = module(control_states, res_emb, torch.ones((F), device=control_states.device))
            control_states = rearrange(control_states, "(B F) C H W -> B F C H W", B=B, F=F)
            control_states = hidden_states + control_states
            control_states = self.patch_embed(encoder_hidden_states, control_states)
            control_states = self.embedding_dropout(control_states)
            hidden_states = control_states
        else:
            hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
            hidden_states = self.embedding_dropout(hidden_states)


        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        # 3. Transformer blocks
        residual_dict = {}  # 存储对称 residual

        num_blocks = len(self.transformer_blocks)

        # 3. Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    emb,
                    image_rotary_emb,
                    attention_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                    attention_kwargs=attention_kwargs,
                )

            # 目标对称 block index
            mirror_idx = num_blocks - 1 - i

            # 如果我在前半段，则保存下来
            if i < mirror_idx:
                residual_dict[i] = hidden_states

            # 如果我在后半段，找到对应的前半 residual，加上去
            elif i > mirror_idx and mirror_idx in residual_dict:
                hidden_states = hidden_states + residual_dict[mirror_idx]


        hidden_states = self.norm_final(hidden_states)

        # 4. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        p = self.config.patch_size
        p_t = self.config.patch_size_t

        if p_t is None:
            output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
            output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
        else:
            output = hidden_states.reshape(
                batch_size, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
            )
            output = output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
