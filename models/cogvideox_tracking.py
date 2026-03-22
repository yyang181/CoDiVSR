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
from models.det_processor import SkipConv1dCogVideoXAttnProcessor2_0, MotionResidualCogVideoXAttnProcessor2_0
from diffusers.models.attention_processor import AttnProcessor2_0 

### support tlc
import numpy as np
from numpy import pi, exp, sqrt

### support low pass filter
from models import lp_utils
from einops import rearrange

### support control feature projector, spatio-temporal resblock
from types import MethodType
from models.resnet_vividvr import SpatioTemporalResBlock
from models.cogvideox_vividvr import CogVideoXTransformer3DModelVividVR, CogVideoXTransformer3DModelMidResidual, CogVideoXTransformer3DModelUnetResidual

### support vectorized timestep adaptation
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.attention_processor import AttentionProcessor, CogVideoXAttnProcessor2_0, FusedCogVideoXAttnProcessor2_0
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import CogVideoXPatchEmbed, TimestepEmbedding, Timesteps
from diffusers.models.normalization import AdaLayerNorm, CogVideoXLayerNormZero
from diffusers.utils.torch_utils import maybe_allow_in_graph
# from models.cogvideox_tracking import CogVideoXTransformer3DModel_VTA
from models.embeddings import TimestepEmbedding_VTA
from models.normalization import AdaLayerNorm_VTA, CogVideoXLayerNormZero_VTA

### support multistep inference 
from diffusers.utils.torch_utils import randn_tensor

### support discriminator
from contextlib import nullcontext
from models.D import ImageConvNextDiscriminator
from safetensors.torch import load_file


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

def transformer_load_vta(transformer, args):
    assert getattr(args, "enable_vta", False), "You must set --enable_vta to True to enable vectorized timestep adaptation."
    print("🔄 DiffusionAsShader: Loading vectorized timestep adaptation.")

    # 替换 forward
    transformer.forward = MethodType(CogVideoXTransformer3DModel_VTA.forward, transformer)
    
    return transformer

def transformer_load_cfp(transformer, args):
    assert getattr(args, "enable_cfp", False), "You must set --enable_cfp to True to load control feature projector from Vivid-VR."
    print("🔄 DiffusionAsShader: Loading control feature projector from Vivid-VR...")

    # 确定模型当前的 dtype
    model_dtype = next(transformer.parameters()).dtype

    in_channels = transformer.config.in_channels
    time_embed_dim = transformer.config.time_embed_dim

    ### Initialize control feature projector and patch embedding
    # transformer.control_feat_proj = nn.ModuleList([
    #     SpatioTemporalResBlock(in_channels, 320, time_embed_dim, merge_strategy="learned", groups=16),
    #     SpatioTemporalResBlock(320, 320, time_embed_dim, merge_strategy="learned", groups=32),
    #     SpatioTemporalResBlock(320, in_channels, time_embed_dim, merge_strategy="learned", groups=16)
    # ])
    transformer.control_feat_proj = nn.ModuleList([
        zero_module(SpatioTemporalResBlock(in_channels, 320, time_embed_dim, merge_strategy="learned", groups=16)),
        zero_module(SpatioTemporalResBlock(320, 320, time_embed_dim, merge_strategy="learned", groups=32)),
        zero_module(SpatioTemporalResBlock(320, in_channels, time_embed_dim, merge_strategy="learned", groups=16))
    ])

    # Unfreeze parameters that need to be trained
    for proj in transformer.control_feat_proj:
        for param in proj.parameters():
            param.requires_grad = True

    # 替换 forward
    transformer.forward = MethodType(CogVideoXTransformer3DModelVividVR.forward, transformer)

    # 如果需要，可以在这里统一转换整个模型的 dtype
    transformer = transformer.to(model_dtype)
    
    return transformer, list(transformer.state_dict().keys())


def transformer_enable_midresidual(transformer, args):
    assert getattr(args, "enable_midresidual", False), "You must set --enable_midresidual to True to enable mid residual."
    print("🔄 DiffusionAsShader: Loading mid residual forward...")

    # 确定模型当前的 dtype
    model_dtype = next(transformer.parameters()).dtype

    # 替换 forward
    transformer.forward = MethodType(CogVideoXTransformer3DModelMidResidual.forward, transformer)

    # 如果需要，可以在这里统一转换整个模型的 dtype
    transformer = transformer.to(model_dtype)
    
    return transformer, list(transformer.state_dict().keys())


def transformer_enable_unetresidual(transformer, args):
    assert getattr(args, "enable_unetresidual", False), "You must set --enable_unetresidual to True to enable unet residual."
    print("🔄 DiffusionAsShader: Loading unet residual forward...")

    # 确定模型当前的 dtype
    model_dtype = next(transformer.parameters()).dtype

    # 替换 forward
    transformer.forward = MethodType(CogVideoXTransformer3DModelUnetResidual.forward, transformer)

    # 如果需要，可以在这里统一转换整个模型的 dtype
    transformer = transformer.to(model_dtype)
    
    return transformer, list(transformer.state_dict().keys())
    
def transformer_load_discriminator(transformer, args, accelerator, logging_):
    assert getattr(args, "add_gan_loss", False), "You must set --add_gan_loss to True to load discriminator."

    pretrained_discriminator_ckpt_path = os.path.abspath('./checkpoints/HYPIR_sd2_D.safetensors')

    accelerator.print("🔄 DiffusionAsShader: Loading discriminator")

    # 确定模型当前的 dtype
    model_dtype = next(transformer.parameters()).dtype

    class SuppressLogging:
        def __init__(self, level=logging_.CRITICAL):
            self.level = level
            self.original_level = logging_.getLogger().level

        def __enter__(self):
            logging_.getLogger().setLevel(self.level)

        def __exit__(self, exc_type, exc_val, exc_tb):
            logging_.getLogger().setLevel(self.original_level)

    # Suppress logs from open-clip
    ctx = (
        nullcontext()
        if accelerator.is_local_main_process
        else SuppressLogging(logging_.WARNING)
    )
    with ctx:
        discriminator = ImageConvNextDiscriminator(precision="bf16").to(device=accelerator.device)

    discriminator.decoder.load_state_dict(load_file(pretrained_discriminator_ckpt_path))
    accelerator.print(f"✅ DiffusionAsShader: Loaded discriminator from {pretrained_discriminator_ckpt_path}")
    
    discriminator.eval().requires_grad_(False)

    # 如果需要，可以在这里统一转换整个模型的 dtype
    discriminator = discriminator.to(model_dtype)
    
    return discriminator, list(discriminator.state_dict().keys())

def transformer_enable_motion(transformer, args, NUM_MODIFIED_BLOCKS=42):
    ### load enable_motion processor 
    assert getattr(args, "enable_motion", False), "You must set --enable_motion to True to load enable_motion processor ."
    assert not getattr(args, "load_skipconv1d", False), "You must set --load_skipconv1d to False."

    print("🔄 DiffusionAsShader: Loading enable_motion processor ...")
    attn_proc_height = 60 // transformer.config.patch_size if not getattr(args, "add_gan_loss", False) else 40 // transformer.config.patch_size 
    attn_proc_width = 90 // transformer.config.patch_size if not getattr(args, "add_gan_loss", False) else 60 // transformer.config.patch_size 
    attn_proc_frames = 14 // transformer.config.patch_size_t if not getattr(args, "add_gan_loss", False) else 26 // transformer.config.patch_size_t # latent temporal is: 5+1=6 or 13+1=14  2 for baseline_v0_sft_lowpass_skipconv1d_v2_ResumeLowpassSkip8406_motion_ganloss_datav1s2
    attn_proc_dim = transformer.config.num_attention_heads * transformer.config.attention_head_dim
    
    # print(f"height: {height}, width: {width}, frames: {frames}, dim: {dim}");assert 0 # height: 30, width: 45, frames: 13, dim: 3072 
    det_processors = {}
    modified_count = 0
    for det_key, value in transformer.attn_processors.items():
        block_idx = int(det_key.split(".")[1])
        if block_idx in range(0, NUM_MODIFIED_BLOCKS):
            modified_count += 1
            det_processors[det_key] = MotionResidualCogVideoXAttnProcessor2_0(
                height=attn_proc_height,
                width=attn_proc_width,
                frames=attn_proc_frames,
                dim=attn_proc_dim,
                rank=128,
                kernel_size=3,
                module_type='conv1d_motion_residual',
                FLAG_SKIPCONV1D_v2=False,
            ).to(dtype=transformer.dtype)

            for det_processors_param in det_processors[det_key].parameters():
                det_processors_param.requires_grad_(True)
        else:
            det_processors[det_key] = value

    det_processors = det_processors.copy() # make a copy to protect the original dict. transformer.set_attn_processor would pop out contents in dict.
    num_det_processors = len(det_processors)
    transformer.set_attn_processor(det_processors)
    model_keys_after_skipconv1d = list(transformer.state_dict().keys())
    print(f"✅ DiffusionAsShader: Loaded enable_motion processor with {num_det_processors} processors, {modified_count} processors modified.")
    return transformer, model_keys_after_skipconv1d


def transformer_load_skipconv1d(transformer, args, NUM_MODIFIED_BLOCKS=42):
    ### load SkipConv1D processor 
    assert getattr(args, "load_skipconv1d", False), "You must set --load_skipconv1d to True to load SkipConv1D processor ."

    print("🔄 DiffusionAsShader: Loading SkipConv1D processor ...")
    # 复制当前 attn_processors
    # if getattr(args, "enable_sft", False) and getattr(args, "enable_cogvideox1dot5", False):
    #     attn_proc_height = 60 // transformer.config.patch_size
    #     attn_proc_width = 90 // transformer.config.patch_size
    #     attn_proc_frames = 14 // transformer.config.patch_size_t
    #     attn_proc_dim = transformer.config.num_attention_heads * transformer.config.attention_head_dim
    # else:
    #     attn_proc_height = transformer.config.sample_height // transformer.config.patch_size
    #     attn_proc_width = transformer.config.sample_width // transformer.config.patch_size
    #     attn_proc_frames = transformer.config.sample_frames // transformer.config.temporal_compression_ratio + 1
    #     attn_proc_dim = transformer.config.num_attention_heads * transformer.config.attention_head_dim
    attn_proc_height = 60 // transformer.config.patch_size if not getattr(args, "add_gan_loss", False) else 60 // transformer.config.patch_size 
    attn_proc_width = 90 // transformer.config.patch_size if not getattr(args, "add_gan_loss", False) else 90 // transformer.config.patch_size 
    attn_proc_frames = 14 // transformer.config.patch_size_t if not getattr(args, "add_gan_loss", False) else 4 // transformer.config.patch_size_t # latent temporal is: 5+1=6 or 13+1=14  2 for baseline_v0_sft_lowpass_skipconv1d_v2_ResumeLowpassSkip8406_motion_ganloss_datav1s2
    attn_proc_dim = transformer.config.num_attention_heads * transformer.config.attention_head_dim
    
    if getattr(args, "use_data_pt", None) == './exp/save_data_pt_v1_stage2':
        # 320x480x25 -> 40x60x7
        attn_proc_height = 40 // transformer.config.patch_size
        attn_proc_width = 60 // transformer.config.patch_size
        attn_proc_frames = 8 // transformer.config.patch_size_t
        
    # print(f"height: {height}, width: {width}, frames: {frames}, dim: {dim}");assert 0 # height: 30, width: 45, frames: 13, dim: 3072 
    det_processors = {}
    # for name, module in model.attn_processors.items():
        # print(name) # 42 transformer_blocks and 18 transformer_blocks_copy
        # print(f"{name}: {type(module)}") # transformer_blocks.18.attn1.processor: <class 'diffusers.models.attention_processor.CogVideoXAttnProcessor2_0'>
    modified_count = 0
    for det_key, value in transformer.attn_processors.items():
        block_idx = int(det_key.split(".")[1])
        if block_idx in range(0, NUM_MODIFIED_BLOCKS):
            modified_count += 1
            det_processors[det_key] = SkipConv1dCogVideoXAttnProcessor2_0(
                height=attn_proc_height,
                width=attn_proc_width,
                frames=attn_proc_frames,
                dim=attn_proc_dim,
                rank=128,
                kernel_size=3,
                module_type='conv1d',
                FLAG_SKIPCONV1D_v2 = not getattr(args, "load_skipconv1d_add_at_end", False),
            ).to(dtype=transformer.dtype)

            for det_processors_param in det_processors[det_key].parameters():
                det_processors_param.requires_grad_(True)
            # print(f"✅ CogVideoXTransformer3DModelTracking __from_pretrained__ Exception: Loaded SkipConv1D processor for {key} with type {type(det_processors[key])}")
        else:
            det_processors[det_key] = value

    det_processors = det_processors.copy() # make a copy to protect the original dict. transformer.set_attn_processor would pop out contents in dict.
    num_det_processors = len(det_processors)
    transformer.set_attn_processor(det_processors)
    model_keys_after_skipconv1d = list(transformer.state_dict().keys())
    print(f"✅ DiffusionAsShader: Loaded SkipConv1D processor with {num_det_processors} processors, {modified_count} processors modified.")
    return transformer, model_keys_after_skipconv1d


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
    

class CogVideoXTransformer3DModelTracking(CogVideoXTransformer3DModel, ModelMixin):
    """
    Add tracking maps to the CogVideoX transformer model.

    Parameters:
        num_tracking_blocks (`int`, defaults to `18`):
            The number of tracking blocks to use. Must be less than or equal to num_layers.
    """

    def __init__(
        self,
        num_tracking_blocks: Optional[int] = 18,
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
        load_refadapter: bool = False,
        load_skipconv1d: bool = False,
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
            **kwargs
        )

        inner_dim = num_attention_heads * attention_head_dim
        self.num_tracking_blocks = num_tracking_blocks

        # Ensure num_tracking_blocks is not greater than num_layers
        if num_tracking_blocks > num_layers:
            raise ValueError("num_tracking_blocks must be less than or equal to num_layers")

        # Create linear layers for combining hidden states and tracking maps
        self.combine_linears = nn.ModuleList(
            [nn.Linear(inner_dim, inner_dim, device="cpu") for _ in range(num_tracking_blocks)]
        )

        # Initialize weights of combine_linears to zero
        for linear in self.combine_linears:
            linear.weight.data.zero_()
            linear.bias.data.zero_()

        # Create transformer blocks for processing tracking maps
        self.transformer_blocks_copy = nn.ModuleList(
            [
                CogVideoXBlock(
                    dim=inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    time_embed_dim=self.config.time_embed_dim,
                    dropout=self.config.dropout,
                    activation_fn=self.config.activation_fn,
                    attention_bias=self.config.attention_bias,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                ).to_empty(device="cpu")
                for _ in range(num_tracking_blocks)
            ]
        )


        load_refadapter = kwargs.get("load_refadapter", False) or load_refadapter
        load_skipconv1d = kwargs.get("load_skipconv1d", False) or load_skipconv1d
        self.load_refadapter = load_refadapter
        self.load_skipconv1d = load_skipconv1d
        print('🔄🔄🔄', self.load_refadapter, self.load_skipconv1d)
        ### init load_refadapter and load_skipconv1d
        assert not (load_refadapter and load_skipconv1d), "Cannot load both RefAdapter and SkipConv1D at the same time. Please choose one."
        if load_refadapter:
            print("🔄 CogVideoXTransformer3DModelTracking __init__: Loading RefAdapter processor from FlexiAct...")
            # 复制当前 attn_processors
            flexiact_procs = dict(self.attn_processors)  # or deepcopy if needed
            modified_count = 0
            for name, module in self.attn_processors.items():
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
            print(f"✅ CogVideoXTransformer3DModelTracking __init__: Loaded RefNetLoRA processor with {len(flexiact_procs)} processors, {modified_count} processors modified, first 5 and last 5 keys are: {list(flexiact_procs.keys())[:5]} ... {list(flexiact_procs.keys())[-5:]}")
            self.set_attn_processor(flexiact_procs)


        if load_skipconv1d:
            print("🔄 CogVideoXTransformer3DModelTracking __init__: Loading SkipConv1D processor ...")
            # 复制当前 attn_processors
            height = self.config.sample_height // self.config.patch_size
            width = self.config.sample_width // self.config.patch_size
            frames = self.config.sample_frames // self.config.temporal_compression_ratio + 1
            dim = self.config.num_attention_heads * self.config.attention_head_dim

            # print(f"height: {height}, width: {width}, frames: {frames}, dim: {dim}");assert 0 # height: 30, width: 45, frames: 13, dim: 3072 
            det_processors = {}
            # for name, module in model.attn_processors.items():
                # print(name) # 42 transformer_blocks and 18 transformer_blocks_copy
                # print(f"{name}: {type(module)}") # transformer_blocks.18.attn1.processor: <class 'diffusers.models.attention_processor.CogVideoXAttnProcessor2_0'>
            modified_count = 0
            for key, value in self.attn_processors.items():
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
                    ).to(dtype=self.dtype)
                    # print(f"✅ CogVideoXTransformer3DModelTracking __init__: Loaded SkipConv1D processor for {key} with type {type(det_processors[key])}")
                else:
                    det_processors[key] = value
                    # print(f"✅ CogVideoXTransformer3DModelTracking __init__: Loaded SkipConv1D processor for {key} with type {type(det_processors[key])}")

            det_processors = det_processors.copy() # make a copy to protect the original dict. transformer.set_attn_processor would pop out contents in dict.
            print(f"✅ CogVideoXTransformer3DModelTracking __init__: Loaded SkipConv1D processor with {len(det_processors)} processors, {modified_count} processors modified")
            self.set_attn_processor(det_processors)


        # For initial combination of hidden states and tracking maps
        self.initial_combine_linear = nn.Linear(inner_dim, inner_dim, device="cpu")
        self.initial_combine_linear.weight.data.zero_()
        self.initial_combine_linear.bias.data.zero_()

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
        
        # Unfreeze parameters that need to be trained
        for linear in self.combine_linears:
            for param in linear.parameters():
                param.requires_grad = True
        
        for block in self.transformer_blocks_copy:
            for param in block.parameters():
                param.requires_grad = True
        
        for param in self.initial_combine_linear.parameters():
            param.requires_grad = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        tracking_maps: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
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

        # 2. Patch embedding
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)

        # Process tracking maps
        prompt_embed = encoder_hidden_states.clone()
        tracking_maps_hidden_states = self.patch_embed(prompt_embed, tracking_maps)
        tracking_maps_hidden_states = self.embedding_dropout(tracking_maps_hidden_states)
        del prompt_embed

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]
        tracking_maps = tracking_maps_hidden_states[:, text_seq_length:]

        # Combine hidden states and tracking maps initially
        combined = hidden_states + tracking_maps
        tracking_maps = self.initial_combine_linear(combined)

        # Process transformer blocks
        for i in range(len(self.transformer_blocks)):
            if self.training and self.gradient_checkpointing:
                # Gradient checkpointing logic for hidden states
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.transformer_blocks[i]),
                    hidden_states,
                    encoder_hidden_states,
                    emb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states = self.transformer_blocks[i](
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                )
            
            if i < len(self.transformer_blocks_copy):
                if self.training and self.gradient_checkpointing:
                    # Gradient checkpointing logic for tracking maps
                    tracking_maps, _ = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(self.transformer_blocks_copy[i]),
                        tracking_maps,
                        encoder_hidden_states,
                        emb,
                        image_rotary_emb,
                        **ckpt_kwargs,
                    )
                else:
                    tracking_maps, _ = self.transformer_blocks_copy[i](
                        hidden_states=tracking_maps,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=emb,
                        image_rotary_emb=image_rotary_emb,
                    )
                
                # Combine hidden states and tracking maps
                tracking_maps = self.combine_linears[i](tracking_maps)


                ### add controlnext normalization at only first block
                if getattr(args, "controlnext_normalization_at_start", False) and i == 0:
                    token_height = self.config.sample_height // self.config.patch_size
                    token_width = self.config.sample_width // self.config.patch_size
                    token_frames = self.config.sample_frames // self.config.temporal_compression_ratio + 1
                    token_dim = self.config.num_attention_heads * self.config.attention_head_dim
                
                    # print(batch_size, num_frames, channels, height, width, token_frames, token_dim, token_height, token_width, hidden_states.shape, tracking_maps.shape, encoder_hidden_states.shape)
                    # assert 0 #  13 32 60 90 13 3072 30 45 torch.Size([2, 17550, 3072]) torch.Size([2, 17550, 3072]) torch.Size([2, 226, 3072]) 

                    ### reshape hidden_states and tracking_maps back to 5D tensors
                    # hidden_states_5d = hidden_states.reshape(-1, token_frames, token_dim, token_height, token_width)
                    hidden_states_5d = hidden_states.view(batch_size, token_frames, token_height * token_width, token_dim)
                    hidden_states_5d = hidden_states_5d.transpose(2, 3)
                    hidden_states_5d = hidden_states_5d.view(batch_size, token_frames, token_dim, token_height, token_width)
                    hidden_states_5d = hidden_states_5d.flatten(0, 1)  # Flatten batch and frames
                    # conditional_controls = tracking_maps.reshape(-1, token_frames, token_dim, token_height, token_width)
                    conditional_controls = tracking_maps.view(batch_size, token_frames, token_height * token_width, token_dim)
                    conditional_controls = conditional_controls.transpose(2, 3)
                    conditional_controls = conditional_controls.view(batch_size, token_frames, token_dim, token_height, token_width)
                    conditional_controls = conditional_controls.flatten(0, 1)  # Flatten batch and frames
                    # print(f"hidden_states_5d: {hidden_states_5d.shape}, conditional_controls: {conditional_controls.shape}, hidden_states: {hidden_states.shape}, tracking_maps: {tracking_maps.shape}")
                    # hidden_states_5d: torch.Size([52, 3072, 30, 45]), conditional_controls: torch.Size([52, 3072, 30, 45]), hidden_states: torch.Size([4, 17550, 3072]), tracking_maps: torch.Size([4, 17550, 3072])

                    ### calculate mean and std for normalization
                    mean_hidden_states, std_hidden_states = torch.mean(hidden_states_5d, dim=(1, 2, 3), keepdim=True), torch.std(hidden_states_5d, dim=(1, 2, 3), keepdim=True)
                    mean_tracking_maps, std_tracking_maps = torch.mean(conditional_controls, dim=(1, 2, 3), keepdim=True), torch.std(conditional_controls, dim=(1, 2, 3), keepdim=True)
                    conditional_controls = (conditional_controls - mean_tracking_maps) * (std_hidden_states / (std_tracking_maps + 1e-5)) + mean_hidden_states
                    conditional_controls = torch.nn.functional.adaptive_avg_pool2d(conditional_controls, hidden_states_5d.shape[-2:])

                    ### reshape conditional_controls back to 3D tensor for transformer attention
                    tracking_maps = conditional_controls.view(batch_size, token_frames, *conditional_controls.shape[1:]) # BT C H W -> B T C H W
                    tracking_maps = tracking_maps.flatten(3).transpose(2, 3)  # [batch, num_frames, height x width, channels]
                    # print(f"1 conditional_controls: {conditional_controls.shape}, tracking_maps: {tracking_maps.shape}, hidden_states: {hidden_states.shape}")
                    # 1 conditional_controls: torch.Size([52, 3072, 30, 45]), tracking_maps: torch.Size([4, 13, 1350, 3072]), hidden_states: torch.Size([4, 17550, 3072]) 
                    tracking_maps = tracking_maps.flatten(1, 2)  # [batch, num_frames x height x width, channels]
                    # print(f"2 conditional_controls: {conditional_controls.shape}, tracking_maps: {tracking_maps.shape}, hidden_states: {hidden_states.shape}")
                    # 2 conditional_controls: torch.Size([52, 3072, 30, 45]), tracking_maps: torch.Size([4, 17550, 3072]), hidden_states: torch.Size([4, 17550, 3072])


                    # # the normalization is applied to the  tracking maps
                    # conditional_controls = tracking_maps.flatten(0, 1)
                    # mean_hidden_states, std_hidden_states = torch.mean(hidden_states.flatten(0, 1), dim=(1, 2, 3), keepdim=True), torch.std(hidden_states.flatten(0, 1), dim=(1, 2, 3), keepdim=True)
                    # mean_tracking_maps, std_tracking_maps = torch.mean(tracking_maps.flatten(0, 1), dim=(1, 2, 3), keepdim=True), torch.std(tracking_maps.flatten(0, 1), dim=(1, 2, 3), keepdim=True)
                    # conditional_controls = (conditional_controls - mean_tracking_maps) * (std_hidden_states / (std_tracking_maps + 1e-5)) + mean_hidden_states
                    # conditional_controls = torch.nn.functional.adaptive_avg_pool2d(conditional_controls, hidden_states.shape[-2:])
                    # tracking_maps = conditional_controls.reshape(batch_size, num_frames, channels, height, width)
                    
                    # print(f"conditional_controls: {conditional_controls.shape}, tracking_maps: {tracking_maps.shape}, hidden_states: {hidden_states.shape}");assert 0
                    # # conditional_controls: torch.Size([26, 32, 60, 90]), tracking_maps: torch.Size([2, 13, 32, 60, 90]), hidden_states: torch.Size([2, 13, 32, 60, 90])

                    #  0.2: This superparameter is used to adjust the control level: increasing this value will strengthen the control level.
                    # sample = sample + conditional_controls * scale * 0.2


                hidden_states = hidden_states + tracking_maps * getattr(args, "control_scale", 1.0) # default to 1.0 if not set
                

        if not self.config.use_rotary_positional_embeddings:
            # CogVideoX-2B
            hidden_states = self.norm_final(hidden_states)
        else:
            # CogVideoX-5B
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            hidden_states = self.norm_final(hidden_states)
            hidden_states = hidden_states[:, text_seq_length:]

        # 4. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        # Note: we use `-1` instead of `channels`:
        #   - It is okay to `channels` use for CogVideoX-2b and CogVideoX-5b (number of input channels is equal to output channels)
        #   - However, for CogVideoX-5b-I2V also takes concatenated input image latents (number of input channels is twice the output channels)
        p = self.config.patch_size
        output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
        output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)

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
                    print("🔄 DiffusionAsShader: Loading SkipConv1D processor ...")
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
                    print("🔄 CogVideoXTransformer3DModelTracking __from_pretrained__ Exception: Loading SkipConv1D processor ...")
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

class CogVideoXPipelineTracking(CogVideoXPipeline, DiffusionPipeline):

    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        vae: AutoencoderKLCogVideoX,
        transformer: CogVideoXTransformer3DModelTracking,
        scheduler: Union[CogVideoXDDIMScheduler, CogVideoXDPMScheduler],
    ):
        super().__init__(tokenizer, text_encoder, vae, transformer, scheduler)
        
        if not isinstance(self.transformer, CogVideoXTransformer3DModelTracking):
            raise ValueError("The transformer in this pipeline must be of type CogVideoXTransformer3DModelTracking")

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 480,
        width: int = 720,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 226,
        tracking_maps: Optional[torch.Tensor] = None,
    ) -> Union[CogVideoXPipelineOutput, Tuple]:
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        num_videos_per_prompt = 1

        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            callback_on_step_end_tensor_inputs,
            prompt_embeds,
            negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        do_classifier_free_guidance = guidance_scale > 1.0

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt,
            do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)

        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )

        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            old_pred_original_sample = None
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                tracking_maps_latent = torch.cat([tracking_maps] * 2) if do_classifier_free_guidance else tracking_maps
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    image_rotary_emb=image_rotary_emb,
                    attention_kwargs=attention_kwargs,
                    tracking_maps=tracking_maps_latent,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred.float()

                if use_dynamic_cfg:
                    self._guidance_scale = 1 + guidance_scale * (
                        (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                    )
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                else:
                    latents, old_pred_original_sample = self.scheduler.step(
                        noise_pred,
                        old_pred_original_sample,
                        t,
                        timesteps[i - 1] if i > 0 else None,
                        latents,
                        **extra_step_kwargs,
                        return_dict=False,
                    )
                latents = latents.to(prompt_embeds.dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if not output_type == "latent":
            video = self.decode_latents(latents)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)
        return CogVideoXPipelineOutput(frames=video)

class CogVideoXImageToVideoPipelineTracking(CogVideoXImageToVideoPipeline, DiffusionPipeline):

    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        vae: AutoencoderKLCogVideoX,
        transformer: CogVideoXTransformer3DModelTracking,
        scheduler: Union[CogVideoXDDIMScheduler, CogVideoXDPMScheduler],
    ):
        super().__init__(tokenizer, text_encoder, vae, transformer, scheduler)
        
        if not isinstance(self.transformer, CogVideoXTransformer3DModelTracking) and not isinstance(self.transformer, CogVideoXTransformer3DModel):
            raise ValueError("The transformer in this pipeline must be of type CogVideoXTransformer3DModelTracking or CogVideoXTransformer3DModel")
            
        # 打印transformer blocks的数量
        print(f"Number of transformer blocks: {len(self.transformer.transformer_blocks)}")
        print(f"Number of tracking transformer blocks: {len(self.transformer.transformer_blocks_copy)}") if hasattr(self.transformer, 'transformer_blocks_copy') else print("No tracking transformer blocks found.")
        # self.transformer = torch.compile(self.transformer)

    def prepare_latents(
        self,
        batch_size: int = 1,
        num_channels_latents: int = 16,
        num_frames: int = 13,
        height: int = 60,
        width: int = 90,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        num_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_frames,
            num_channels_latents,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )

        ### replace the vae_scale_factor_spatial with the actual spatial scale factor
        self.transformer.config.sample_width = width // self.vae_scale_factor_spatial
        self.transformer.config.sample_height = height // self.vae_scale_factor_spatial
        self.transformer.config.sample_frames = num_frames

        # For CogVideoX1.5, the latent should add 1 for padding (Not use)
        if self.transformer.config.patch_size_t is not None:
            shape = shape[:1] + (shape[1] + shape[1] % self.transformer.config.patch_size_t,) + shape[2:]

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents
    
    @torch.no_grad()
    def __call__(
        self,
        video: Optional[torch.Tensor] = None,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 226,
        args: Optional[Dict[str, Any]] = None,
    ) -> Union[CogVideoXPipelineOutput, Tuple]:
        # Most of the implementation remains the same as the parent class
        # We will modify the parts that need to handle tracking_maps

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        height = height or self.transformer.config.sample_height * self.vae_scale_factor_spatial
        width = width or self.transformer.config.sample_width * self.vae_scale_factor_spatial
        num_frames = num_frames or self.transformer.config.sample_frames

        num_videos_per_prompt = 1
       
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            image=video,
            prompt=prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        # # 1. Check inputs and set default values
        # self.check_inputs(
        #     image,
        #     prompt,
        #     height,
        #     width,
        #     negative_prompt,
        #     callback_on_step_end_tensor_inputs,
        #     prompt_embeds,
        #     negative_prompt_embeds,
        # )
        self._guidance_scale = guidance_scale
        self._current_timestep = None
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        if getattr(args, "enable_cogvideox1dot5", False):
            do_classifier_free_guidance = False

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            del negative_prompt_embeds



        if getattr(args, "enable_cogvideox1dot5", False):
            # 4. Prepare timesteps
            num_inference_steps = num_inference_steps if getattr(args, "enable_pipeline_forward", False) else 1

            if getattr(args, "enable_dynanoise", False):
                ### Sample a random timestep for each image
                avg_score = torch.tensor(args.tmp_clipiqa_score, dtype=tracking_maps.dtype, device=tracking_maps.device)

                if 0:
                    sr_noise_step = find_nearest_timestep(avg_score, self.scheduler)
                    # print(clipiqa_score, sr_noise_step); assert 0 # tensor([0.3457, 0.1904, 0.2168, 0.3281], device='cuda:1', dtype=torch.bfloat16) tensor([609, 741, 716, 623], device='cuda:1')
                    timesteps = sr_noise_step.to(device=tracking_maps.device, dtype=torch.long)  # shape: [B]
                else:
                    sr_noise_step = 399
                    timesteps = torch.full(
                        (batch_size,),
                        fill_value=sr_noise_step,
                        dtype=torch.long,
                        device=tracking_maps.device,
                    )

            elif getattr(args, "enable_pipeline_forward", False):
                # 4. Prepare timesteps
                timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)

            else:
                ### Sample a random timestep for each image
                sr_noise_step = 399
                timesteps = torch.full(
                    (batch_size,),
                    fill_value=sr_noise_step,
                    dtype=torch.long,
                    device=tracking_maps.device,
                )

            # timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
            self._num_timesteps = len(timesteps)


            # 5. Prepare latents
            latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

            # For CogVideoX 1.5, the latent frames should be padded to make it divisible by patch_size_t
            patch_size_t = self.transformer.config.patch_size_t
            additional_frames = 0
            if patch_size_t is not None and latent_frames % patch_size_t != 0:
                additional_frames = patch_size_t - latent_frames % patch_size_t
                # print(additional_frames, latent_frames, patch_size_t) # 1 13 2
                num_frames += additional_frames * self.vae_scale_factor_temporal
                # print(num_frames);assert 0 # 109

        else:
            # 4. Prepare timesteps
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
            self._num_timesteps = len(timesteps)

        video = video.to(device=device, dtype=prompt_embeds.dtype)
        video_latent_dist = self.vae.encode(video).latent_dist
        video_latent = video_latent_dist.sample() * self.vae.config.scaling_factor
        video_latent = video_latent.to(device=device, dtype=prompt_embeds.dtype)

        latent_channels = self.transformer.config.in_channels // 2
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        del video
        # print(num_frames, latents.shape, video_latent.shape, num_frames, height, width);assert 0 
        # 49 torch.Size([1, 13, 16, 60, 90]) torch.Size([1, 13, 16, 60, 90])  
        # 109 torch.Size([1, 28, 16, 120, 160]) torch.Size([1, 16, 27, 120, 160])

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Create rotary embeds if required
        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )

        # 8. Create ofs embeds if required
        ofs_emb = None if self.transformer.config.ofs_embed_dim is None else latents.new_full((1,), fill_value=2.0)

        if getattr(args, "enable_cogvideox1dot5", False):
            ### handel patch_size_t for CogVideoX 1.5
            patch_size_t = self.transformer.config.patch_size_t
            if patch_size_t is not None:
                ncopy = video_latent.shape[2] % patch_size_t
                # Copy the first frame ncopy times to match patch_size_t
                first_frame = video_latent[:, :, :1, :, :]  # Get first frame [B, C, 1, H, W]
                video_latent = torch.cat([first_frame.repeat(1, 1, ncopy, 1, 1), video_latent], dim=2)
                # print(f"CogVideoX 1.5: video_latent shape after patch_size_t adjustment: {video_latent.shape}") # [B, C, F + ncopy, H, W]

                assert video_latent.shape[2] % patch_size_t == 0
                video_latent = video_latent.permute(0, 2, 1, 3, 4).contiguous() # [B, C, F, H, W] -> [B, F, C, H, W]

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        if getattr(args, "enable_cogvideox1dot5", False) and not getattr(args, "enable_pipeline_forward", False):

            if getattr(args, "enable_dynanoise", False):
                # add noise to tracking_maps for more details generation
                noisy_video_latent = self.scheduler.add_noise(tracking_maps, latents, timesteps)

            else:
                noisy_video_latent = latents
            # noisy_video_latent = tracking_maps

            ### use_low_pass_guidance
            if getattr(args, "use_low_pass_guidance", False):
                # Low-pass version input
                # Timestep scheduled low-pass filter strength ([0, 1] range)
                lp_strength = 1.0
                lp_filter_type = 'down_up'
                lp_filter_in_latent = True
                lp_resize_factor = 0.25 # 0.25
                modulated_lp_resize_factor = lp_resize_factor

                # low-pass filter
                # latent B F C H W
                lp_tracking_maps = prepare_lp(
                    # --- Filter Selection & Strength (Modulated) ---
                    lp_filter_type=lp_filter_type,
                    lp_resize_factor=modulated_lp_resize_factor,
                    orig_image_latents=video_latent, # Shape [B, F_padded, C, H, W]
                )
            else:
                lp_tracking_maps = video_latent

            # latent_model_input = torch.cat([noisy_video_latent, tracking_maps], dim=2)
            latent_model_input = torch.cat([noisy_video_latent, lp_tracking_maps], dim=2)

            # Predict noise
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                predicted_noise = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timesteps,
                    ofs=ofs_emb,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
                )[0]
            # print(predicted_noise.shape, latent_model_input.shape, timesteps.shape) # torch.Size([1, 14, 16, 60, 90]) torch.Size([1, 14, 32, 60, 90]) torch.Size([1])
            latents = self.scheduler.get_velocity(
                predicted_noise, noisy_video_latent, timesteps
            )

        elif getattr(args, "enable_cogvideox1dot5", False) and getattr(args, "enable_pipeline_forward", False):
            # 8. Denoising loop

            ### fix the attn_processors height, width, frames, dim for SkipConv1dCogVideoXAttnProcessor2_0
            if getattr(args, "load_skipconv1d", False):
                # modify attn_processors self.height
                def set_spatial_temporal_params(model, height, width, frames, dim):
                    modified_count = 0
                    for name, module in model.attn_processors.items():
                        if isinstance(module, SkipConv1dCogVideoXAttnProcessor2_0):
                            modified_count += 1
                            module.height = height
                            module.width = width
                            module.frames = frames
                            module.dim = dim
                            # print(f"Set SkipConv1dCogVideoXAttnProcessor2_0 params: height={height}, width={width}, frames={frames}, dim={dim}")
                    print(f"✅ Set spatial-temporal params for {modified_count} SkipConv1dCogVideoXAttnProcessor2_0 processors with height={height}, width={width}, frames={frames}, dim={dim}")

                # print(latent.shape) # torch.Size([1, 28, 16, 120, 160]) B T C H W ;assert 0  input size: 206438400 = 14(T) * 60(H) * 80(W) * 3072(dim)
                attn_processors_height = latents.shape[3] // self.transformer.config.patch_size
                attn_processors_width = latents.shape[4] // self.transformer.config.patch_size
                attn_processors_frames = latents.shape[1] // self.transformer.config.patch_size_t
                attn_processors_dim = self.transformer.config.num_attention_heads * self.transformer.config.attention_head_dim
                set_spatial_temporal_params(self.transformer, height=attn_processors_height, width=attn_processors_width, frames=attn_processors_frames, dim=attn_processors_dim)

            ### use_low_pass_guidance
            if getattr(args, "use_low_pass_guidance", False):
                # Low-pass version input
                # Timestep scheduled low-pass filter strength ([0, 1] range)
                lp_strength = 1.0
                lp_filter_type = 'down_up'
                lp_filter_in_latent = True
                lp_resize_factor = 0.25 # 0.25
                modulated_lp_resize_factor = lp_resize_factor

                # low-pass filter
                # latent B F C H W
                lp_tracking_maps = prepare_lp(
                    # --- Filter Selection & Strength (Modulated) ---
                    lp_filter_type=lp_filter_type,
                    lp_resize_factor=modulated_lp_resize_factor,
                    orig_image_latents=video_latent, # Shape [B, F_padded, C, H, W]
                )
            else:
                lp_tracking_maps = video_latent


            with self.progress_bar(total=num_inference_steps) as progress_bar:
                old_pred_original_sample = None
                for i, t in enumerate(timesteps):
                    if self.interrupt:
                        continue

                    self._current_timestep = t
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    latent_image_input = torch.cat([lp_tracking_maps] * 2) if do_classifier_free_guidance else lp_tracking_maps
                    
                    ### support enable lora
                    # print(latent_model_input.shape, latent_image_input.shape);assert 0
                    # torch.Size([1, 28, 16, 120, 160]) torch.Size([1, 28, 16, 120, 160])  
                    latent_model_input = torch.cat([latent_model_input, torch.cat([latent_image_input] * 2) if do_classifier_free_guidance else latent_image_input], dim=2)
                    del latent_image_input

                    tracking_maps_input = None

                    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                    timestep = t.expand(latent_model_input.shape[0])

                    # Predict noise
                    self.transformer.to(dtype=latent_model_input.dtype)
                    # predict noise model_output
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        encoder_hidden_states=prompt_embeds,
                        timestep=timestep,
                        ofs=ofs_emb,
                        image_rotary_emb=image_rotary_emb,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
                    del latent_model_input           
                    noise_pred = noise_pred.float()

                    # perform guidance
                    if use_dynamic_cfg:
                        self._guidance_scale = 1 + guidance_scale * (
                            (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                        )
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                        del noise_pred_uncond, noise_pred_text

                    # compute the previous noisy sample x_t -> x_t-1
                    if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                        latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                    else:
                        latents, old_pred_original_sample = self.scheduler.step(
                            noise_pred,
                            old_pred_original_sample,
                            t,
                            timesteps[i - 1] if i > 0 else None,
                            latents,
                            **extra_step_kwargs,
                            return_dict=False,
                        )
                    del noise_pred
                    latents = latents.to(prompt_embeds.dtype)

                    # call the callback, if provided
                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                        negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()

        else:
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                old_pred_original_sample = None
                for i, t in enumerate(timesteps):
                    if self.interrupt:
                        continue

                    self._current_timestep = t
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    latent_image_input = torch.cat([image_latents] * 2) if do_classifier_free_guidance else image_latents

                    ### support non-reference image input
                    if getattr(args, "non_reference", False) or getattr(args, "noised_image_dropout", 0.05) == 1.0:
                        # NOTE: noised_image_dropout is set to 1.0, which means no image input is used.
                        latent_image_input = torch.zeros_like(latent_image_input)
                    
                    ### support enable lora
                    # print(latent_model_input.shape, latent_image_input.shape, tracking_image_latents.shape, tracking_maps.shape if tracking_maps is not None else None);assert 0
                    # torch.Size([2, 14, 16, 60, 90]) torch.Size([2, 14, 16, 60, 90]) torch.Size([1, 14, 16, 60, 90]) torch.Size([1, 13, 16, 60, 90])
                    latent_model_input = torch.cat([latent_model_input, latent_image_input], dim=2) if not getattr(args, "enable_lora", False) and not getattr(args, "enable_sft", False) else torch.cat([latent_model_input, torch.cat([tracking_maps] * 2) if do_classifier_free_guidance else tracking_maps], dim=2)
                    del latent_image_input

                    # Handle tracking maps
                    if tracking_maps is not None and not getattr(args, "enable_lora", False) and not getattr(args, "enable_sft", False):
                        latents_tracking_image = torch.cat([tracking_image_latents] * 2) if do_classifier_free_guidance else tracking_image_latents
                        tracking_maps_input = torch.cat([tracking_maps] * 2) if do_classifier_free_guidance else tracking_maps

                        ### support non-reference image input
                        if getattr(args, "non_reference", False):
                            latents_tracking_image = torch.zeros_like(latents_tracking_image)

                        tracking_maps_input = torch.cat([tracking_maps_input, latents_tracking_image], dim=2)
                        del latents_tracking_image
                    else:
                        tracking_maps_input = None

                    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                    timestep = t.expand(latent_model_input.shape[0])

                    # Predict noise
                    self.transformer.to(dtype=latent_model_input.dtype)
                    if tracking_maps_input is None:
                        # predict noise model_output
                        noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            encoder_hidden_states=prompt_embeds,
                            timestep=timestep,
                            ofs=ofs_emb,
                            image_rotary_emb=image_rotary_emb,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]

                    else:
                        # predict noise model_output
                        noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            encoder_hidden_states=prompt_embeds,
                            timestep=timestep,
                            ofs=ofs_emb,
                            image_rotary_emb=image_rotary_emb,
                            attention_kwargs=attention_kwargs,
                            tracking_maps=tracking_maps_input,
                            return_dict=False,
                            args=args,
                        )[0]

                    del latent_model_input           
                    if tracking_maps_input is not None:
                        del tracking_maps_input
                    noise_pred = noise_pred.float()

                    # perform guidance
                    if use_dynamic_cfg:
                        self._guidance_scale = 1 + guidance_scale * (
                            (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                        )
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                        del noise_pred_uncond, noise_pred_text

                    # compute the previous noisy sample x_t -> x_t-1
                    if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                        latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                    else:
                        latents, old_pred_original_sample = self.scheduler.step(
                            noise_pred,
                            old_pred_original_sample,
                            t,
                            timesteps[i - 1] if i > 0 else None,
                            latents,
                            **extra_step_kwargs,
                            return_dict=False,
                        )
                    del noise_pred
                    latents = latents.to(prompt_embeds.dtype)

                    # call the callback, if provided
                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                        negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()

        # 9. Post-processing
        if not output_type == "latent":
            if getattr(args, "enable_cogvideox1dot5", False):
                # Discard any padding frames that were added for CogVideoX 1.5
                latents = latents[:, additional_frames:]
            video = self.decode_latents(latents)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return CogVideoXPipelineOutput(frames=video)


    @torch.no_grad()
    def inference_tlc(
        self,
        image: Union[torch.Tensor, Image.Image],
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 226,
        tracking_maps: Optional[torch.Tensor] = None,
        tracking_image: Optional[torch.Tensor] = None,
        args: Optional[Dict[str, Any]] = None,
    ) -> Union[CogVideoXPipelineOutput, Tuple]:
        # Most of the implementation remains the same as the parent class
        # We will modify the parts that need to handle tracking_maps

        # ### suppress torch dynamo errors
        # import torch._dynamo
        # torch._dynamo.config.suppress_errors = True

        # 1. Check inputs and set default values
        self.check_inputs(
            image,
            prompt,
            height,
            width,
            negative_prompt,
            callback_on_step_end_tensor_inputs,
            prompt_embeds,
            negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            del negative_prompt_embeds

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)

        # 5. Prepare latents
        image = self.video_processor.preprocess(image, height=height, width=width).to(
            device, dtype=prompt_embeds.dtype
        )

        tracking_image = self.video_processor.preprocess(tracking_image, height=height, width=width).to(
            device, dtype=prompt_embeds.dtype
        )
        if self.transformer.config.in_channels != 16:
            latent_channels = self.transformer.config.in_channels // 2
        else:
            latent_channels = self.transformer.config.in_channels
        latents, image_latents = self.prepare_latents(
            image,
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        del image
        
        _, tracking_image_latents = self.prepare_latents(
            tracking_image,
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents=None,
        )
        del tracking_image

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Create rotary embeds if required
        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )
        # print(f"image_rotary_emb type: {type(image_rotary_emb)}, {image_rotary_emb[0].shape}, {image_rotary_emb[1].shape}")
        # assert 0 # image_rotary_emb type: <class 'tuple'>, torch.Size([349740, 64]), torch.Size([349740, 64])

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        ### do tlc init 
        if getattr(args, "test_tlc", False):
            self.tlc_tracking_maps = LocalAttention3D(args.tlc_kernel, (0.875, 0.875, 0.875))
            self.tlc_latents = LocalAttention3D(args.tlc_kernel, (0.875, 0.875, 0.875))
            self.tlc_image_latents = LocalAttention3D(args.tlc_kernel, (0.875, 0.875, 0.875))
            self.tlc_tracking_image_latents = LocalAttention3D(args.tlc_kernel, (0.875, 0.875, 0.875))
            # self.tlc_i

            print(f"Before tlc init, Tracking maps shape: {tracking_maps.shape}, latents shape: {latents.shape}, image_latents shape: {image_latents.shape}, tracking_image_latents shape: {tracking_image_latents.shape}")
            # Tracking maps shape: torch.Size([1, 58, 16, 135, 180]), latents shape: torch.Size([1, 58, 16, 135, 180]), image_latents shape: torch.Size([1, 58, 16, 135, 180]), tracking_image_latents shape: torch.Size([1, 58, 16, 135, 180])  
            tracking_maps = tracking_maps.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
            latents = latents.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
            image_latents = image_latents.permute(0, 2, 1, 3, 4)
            tracking_image_latents = tracking_image_latents.permute(0, 2, 1, 3, 4)

            latents = self.tlc_latents.grids(latents)
            image_latents = self.tlc_image_latents.grids(image_latents)
            tracking_image_latents = self.tlc_tracking_image_latents.grids(tracking_image_latents)
            tracking_maps = self.tlc_tracking_maps.grids(tracking_maps)

            tracking_maps = tracking_maps.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
            latents = latents.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
            image_latents = image_latents.permute(0, 2, 1, 3, 4)
            tracking_image_latents = tracking_image_latents.permute(0, 2, 1, 3, 4)
            print(f"After tlc init, Tracking maps shape: {tracking_maps.shape}, latents shape: {latents.shape}, image_latents shape: {image_latents.shape}, tracking_image_latents shape: {tracking_image_latents.shape}")
            tlc_batchs = tracking_image_latents.shape[0]
            # After tlc init, Tracking maps shape: torch.Size([24, 13, 16, 120, 160]), latents shape: torch.Size([24, 13, 16, 120, 160]), image_latents shape: torch.Size([24, 13, 16, 120, 160]), tracking_image_latents shape: torch.Size([24, 13, 16, 120, 160])

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            old_pred_original_sample = None
            for i, t in enumerate(timesteps):

                ### support tlc
                if getattr(args, "test_tlc", False) and i >= 1:
                    latents = self.tlc_latents.grids(latents).to(dtype=latents.dtype)

                concat_grid = []
                    
                if self.interrupt:
                    continue

                for tlc_batch in range(tlc_batchs):
                    print(f"Processing tlc_batch: {tlc_batch}/{tlc_batchs}, i: {i}, t: {t.item()}")
                    latents_tlc = latents[tlc_batch,:,:,:,:].unsqueeze(0)
                    image_latents_tlc = image_latents[tlc_batch,:,:,:,:].unsqueeze(0)
                    tracking_image_latents_tlc = tracking_image_latents[tlc_batch,:,:,:,:].unsqueeze(0)
                    tracking_maps_tlc = tracking_maps[tlc_batch,:,:,:,:].unsqueeze(0)


                    latent_model_input = torch.cat([latents_tlc] * 2) if do_classifier_free_guidance else latents_tlc
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    latent_image_input = torch.cat([image_latents_tlc] * 2) if do_classifier_free_guidance else image_latents_tlc

                    ### support non-reference image input
                    if getattr(args, "non_reference", False):
                        latent_image_input = torch.zeros_like(latent_image_input)
                    
                    latent_model_input = torch.cat([latent_model_input, latent_image_input], dim=2)
                    del latent_image_input

                    # Handle tracking maps
                    if tracking_maps_tlc is not None:
                        latents_tracking_image = torch.cat([tracking_image_latents_tlc] * 2) if do_classifier_free_guidance else tracking_image_latents_tlc
                        tracking_maps_input = torch.cat([tracking_maps_tlc] * 2) if do_classifier_free_guidance else tracking_maps_tlc

                        ### support non-reference image input
                        if getattr(args, "non_reference", False):
                            latents_tracking_image = torch.zeros_like(latents_tracking_image)

                        tracking_maps_input = torch.cat([tracking_maps_input, latents_tracking_image], dim=2)
                        del latents_tracking_image
                    else:
                        tracking_maps_input = None

                    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                    timestep = t.expand(latent_model_input.shape[0])

                    # Predict noise
                    self.transformer.to(dtype=latent_model_input.dtype)
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        encoder_hidden_states=prompt_embeds,
                        timestep=timestep,
                        image_rotary_emb=image_rotary_emb,
                        attention_kwargs=attention_kwargs,
                        tracking_maps=tracking_maps_input,
                        return_dict=False,
                    )[0]
                    del latent_model_input
                    if tracking_maps_input is not None:
                        del tracking_maps_input
                    noise_pred = noise_pred.float()

                    # perform guidance
                    if use_dynamic_cfg:
                        self._guidance_scale = 1 + guidance_scale * (
                            (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                        )
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                        del noise_pred_uncond, noise_pred_text

                    # compute the previous noisy sample x_t -> x_t-1
                    if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                        latents_tlc = self.scheduler.step(noise_pred, t, latents_tlc, **extra_step_kwargs, return_dict=False)[0]
                    else:
                        latents_tlc, old_pred_original_sample = self.scheduler.step(
                            noise_pred,
                            old_pred_original_sample,
                            t,
                            timesteps[i - 1] if i > 0 else None,
                            latents_tlc,
                            **extra_step_kwargs,
                            return_dict=False,
                        )
                    del noise_pred

                    concat_grid.append(latents_tlc)
                    latents_tlc = latents_tlc.to(prompt_embeds.dtype)

                    # call the callback, if provided
                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                        negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()

                latents = self.tlc_latents.grids_inverse(torch.cat(concat_grid, dim=0)).to(latents_tlc.dtype)

        # 9. Post-processing
        if not output_type == "latent":
            video = self.decode_latents(latents)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return CogVideoXPipelineOutput(frames=video)
    

class CogVideoXVideoToVideoPipelineTracking(CogVideoXVideoToVideoPipeline, DiffusionPipeline):

    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        vae: AutoencoderKLCogVideoX,
        transformer: CogVideoXTransformer3DModelTracking,
        scheduler: Union[CogVideoXDDIMScheduler, CogVideoXDPMScheduler],
    ):
        super().__init__(tokenizer, text_encoder, vae, transformer, scheduler)
        
        if not isinstance(self.transformer, CogVideoXTransformer3DModelTracking):
            raise ValueError("The transformer in this pipeline must be of type CogVideoXTransformer3DModelTracking")
            
    @torch.no_grad()
    def __call__(
        self,
        video: List[Image.Image] = None,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 480,
        width: int = 720,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        strength: float = 0.8,
        guidance_scale: float = 6,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 226,
        tracking_maps: Optional[torch.Tensor] = None,
    ) -> Union[CogVideoXPipelineOutput, Tuple]:

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        num_videos_per_prompt = 1

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            strength=strength,
            negative_prompt=negative_prompt,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            video=video,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 2. Default call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt,
            do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, timesteps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_videos_per_prompt)
        self._num_timesteps = len(timesteps)

        # 5. Prepare latents
        if latents is None:
            video = self.video_processor.preprocess_video(video, height=height, width=width)
            video = video.to(device=device, dtype=prompt_embeds.dtype)

        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            video,
            batch_size * num_videos_per_prompt,
            latent_channels,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
            latent_timestep,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Create rotary embeds if required
        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            # for DPM-solver++
            old_pred_original_sample = None
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                tracking_maps_input = torch.cat([tracking_maps] * 2) if do_classifier_free_guidance else tracking_maps
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                # predict noise model_output
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    image_rotary_emb=image_rotary_emb,
                    attention_kwargs=attention_kwargs,
                    tracking_maps=tracking_maps_input,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred.float()

                # perform guidance
                if use_dynamic_cfg:
                    self._guidance_scale = 1 + guidance_scale * (
                        (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                    )
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                else:
                    latents, old_pred_original_sample = self.scheduler.step(
                        noise_pred,
                        old_pred_original_sample,
                        t,
                        timesteps[i - 1] if i > 0 else None,
                        latents,
                        **extra_step_kwargs,
                        return_dict=False,
                    )
                latents = latents.to(prompt_embeds.dtype)

                # call the callback, if provided
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if not output_type == "latent":
            video = self.decode_latents(latents)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return CogVideoXPipelineOutput(frames=video)

@maybe_allow_in_graph
class CogVideoXBlock_VTA(nn.Module):
    r"""
    Transformer block used in [CogVideoX](https://github.com/THUDM/CogVideo) model.

    Parameters:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`):
            The number of channels in each head.
        time_embed_dim (`int`):
            The number of channels in timestep embedding.
        dropout (`float`, defaults to `0.0`):
            The dropout probability to use.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to be used in feed-forward.
        attention_bias (`bool`, defaults to `False`):
            Whether or not to use bias in attention projection layers.
        qk_norm (`bool`, defaults to `True`):
            Whether or not to use normalization after query and key projections in Attention.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_eps (`float`, defaults to `1e-5`):
            Epsilon value for normalization layers.
        final_dropout (`bool` defaults to `False`):
            Whether to apply a final dropout after the last feed-forward layer.
        ff_inner_dim (`int`, *optional*, defaults to `None`):
            Custom hidden dimension of Feed-forward layer. If not provided, `4 * dim` is used.
        ff_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Feed-forward layer.
        attention_out_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Attention output projection layer.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = False,
        qk_norm: bool = True,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = True,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()

        # 1. Self Attention
        self.norm1 = CogVideoXLayerNormZero_VTA(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.attn1 = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor=CogVideoXAttnProcessor2_0(),
        )

        # 2. Feed Forward
        self.norm2 = CogVideoXLayerNormZero_VTA(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)
        attention_kwargs = attention_kwargs or {}

        # norm & modulate
        
        # print(hidden_states.shape, encoder_hidden_states.shape, temb.shape)
        # torch.Size([4, 9450, 3072]) torch.Size([4, 226, 3072]) torch.Size([4, 14, 512])

        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
            hidden_states, encoder_hidden_states, temb
        )

        # attention
        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **attention_kwargs,
        )

        hidden_states = hidden_states + gate_msa * attn_hidden_states
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
            hidden_states, encoder_hidden_states, temb
        )

        # feed-forward
        norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
        encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]

        return hidden_states, encoder_hidden_states


class CogVideoXTransformer3DModel_VTA(ModelMixin, ConfigMixin, PeftAdapterMixin, CacheMixin):
    """
    A Transformer model for video-like data in [CogVideoX](https://github.com/THUDM/CogVideo).

    Parameters:
        num_attention_heads (`int`, defaults to `30`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, defaults to `64`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `16`):
            The number of channels in the output.
        flip_sin_to_cos (`bool`, defaults to `True`):
            Whether to flip the sin to cos in the time embedding.
        time_embed_dim (`int`, defaults to `512`):
            Output dimension of timestep embeddings.
        ofs_embed_dim (`int`, defaults to `512`):
            Output dimension of "ofs" embeddings used in CogVideoX-5b-I2B in version 1.5
        text_embed_dim (`int`, defaults to `4096`):
            Input dimension of text embeddings from the text encoder.
        num_layers (`int`, defaults to `30`):
            The number of layers of Transformer blocks to use.
        dropout (`float`, defaults to `0.0`):
            The dropout probability to use.
        attention_bias (`bool`, defaults to `True`):
            Whether to use bias in the attention projection layers.
        sample_width (`int`, defaults to `90`):
            The width of the input latents.
        sample_height (`int`, defaults to `60`):
            The height of the input latents.
        sample_frames (`int`, defaults to `49`):
            The number of frames in the input latents. Note that this parameter was incorrectly initialized to 49
            instead of 13 because CogVideoX processed 13 latent frames at once in its default and recommended settings,
            but cannot be changed to the correct value to ensure backwards compatibility. To create a transformer with
            K latent frames, the correct value to pass here would be: ((K - 1) * temporal_compression_ratio + 1).
        patch_size (`int`, defaults to `2`):
            The size of the patches to use in the patch embedding layer.
        temporal_compression_ratio (`int`, defaults to `4`):
            The compression ratio across the temporal dimension. See documentation for `sample_frames`.
        max_text_seq_length (`int`, defaults to `226`):
            The maximum sequence length of the input text embeddings.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to use in feed-forward.
        timestep_activation_fn (`str`, defaults to `"silu"`):
            Activation function to use when generating the timestep embeddings.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether to use elementwise affine in normalization layers.
        norm_eps (`float`, defaults to `1e-5`):
            The epsilon value to use in normalization layers.
        spatial_interpolation_scale (`float`, defaults to `1.875`):
            Scaling factor to apply in 3D positional embeddings across spatial dimensions.
        temporal_interpolation_scale (`float`, defaults to `1.0`):
            Scaling factor to apply in 3D positional embeddings across temporal dimensions.
    """

    _skip_layerwise_casting_patterns = ["patch_embed", "norm"]
    _supports_gradient_checkpointing = True
    _no_split_modules = ["CogVideoXBlock", "CogVideoXPatchEmbed"]

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 30,
        attention_head_dim: int = 64,
        in_channels: int = 16,
        out_channels: Optional[int] = 16,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        time_embed_dim: int = 512,
        ofs_embed_dim: Optional[int] = None,
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
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim

        if not use_rotary_positional_embeddings and use_learned_positional_embeddings:
            raise ValueError(
                "There are no CogVideoX checkpoints available with disable rotary embeddings and learned positional "
                "embeddings. If you're using a custom model and/or believe this should be supported, please open an "
                "issue at https://github.com/huggingface/diffusers/issues."
            )

        # 1. Patch embedding
        self.patch_embed = CogVideoXPatchEmbed(
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            in_channels=in_channels,
            embed_dim=inner_dim,
            text_embed_dim=text_embed_dim,
            bias=patch_bias,
            sample_width=sample_width,
            sample_height=sample_height,
            sample_frames=sample_frames,
            temporal_compression_ratio=temporal_compression_ratio,
            max_text_seq_length=max_text_seq_length,
            spatial_interpolation_scale=spatial_interpolation_scale,
            temporal_interpolation_scale=temporal_interpolation_scale,
            use_positional_embeddings=not use_rotary_positional_embeddings,
            use_learned_positional_embeddings=use_learned_positional_embeddings,
        )
        self.embedding_dropout = nn.Dropout(dropout)

        # 2. Time embeddings and ofs embedding(Only CogVideoX1.5-5B I2V have)

        self.time_proj = Timesteps(inner_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding_VTA(inner_dim, time_embed_dim, timestep_activation_fn)

        self.ofs_proj = None
        self.ofs_embedding = None
        if ofs_embed_dim:
            self.ofs_proj = Timesteps(ofs_embed_dim, flip_sin_to_cos, freq_shift)
            self.ofs_embedding = TimestepEmbedding_VTA(
                ofs_embed_dim, ofs_embed_dim, timestep_activation_fn
            )  # same as time embeddings, for ofs

        # 3. Define spatio-temporal transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                CogVideoXBlock_VTA(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_final = nn.LayerNorm(inner_dim, norm_eps, norm_elementwise_affine)

        # 4. Output blocks
        self.norm_out = AdaLayerNorm_VTA(
            embedding_dim=time_embed_dim,
            output_dim=2 * inner_dim,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            chunk_dim=1,
        )

        if patch_size_t is None:
            # For CogVideox 1.0
            output_dim = patch_size * patch_size * out_channels
        else:
            # For CogVideoX 1.5
            output_dim = patch_size * patch_size * patch_size_t * out_channels

        self.proj_out = nn.Linear(inner_dim, output_dim)

        self.gradient_checkpointing = False

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections with FusedAttnProcessor2_0->FusedCogVideoXAttnProcessor2_0
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        self.set_attn_processor(FusedCogVideoXAttnProcessor2_0())

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

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

        # print(timesteps.shape, ofs.shape if ofs is not None else "No ofs provided", hidden_states.shape); assert 0
        # torch.Size([4, 14]) torch.Size([1] torch.Size([4, 7]) torch.Size([1]) torch.Size([4, 14, 32, 60, 90]))  
        
        if timesteps.dim() == 2:
            # 2D 输入，比如 [B, F]
            B_timestep, F_timestep = timesteps.shape
            timesteps_flat = timesteps.reshape(B_timestep * F_timestep)  # 拉平成一维向量，shape (B*F,)
            t_emb_flat = self.time_proj(timesteps_flat)  # 输出 (B*F, embedding_dim)
            # 再reshape回 (B, F, embedding_dim) 方便后续操作
            t_emb = t_emb_flat.reshape(B_timestep, F_timestep, -1)

            # print(f"t_emb shape: {t_emb.shape}, hidden_states shape: {hidden_states.shape}, encoder_hidden_states shape: {encoder_hidden_states.shape}")
            # t_emb shape: torch.Size([4, 14, 3072]), hidden_states shape: torch.Size([4, 14, 32, 60, 90]), encoder_hidden_states shape: torch.Size([4, 226, 4096])

            # timesteps does not contain any weights and will always return f32 tensors
            # but time_embedding might actually be running in fp16. so we need to cast here.
            # there might be better ways to encapsulate this.
            t_emb = t_emb.to(dtype=hidden_states.dtype)
            emb = self.time_embedding(t_emb, timestep_cond)

            # print(f"emb shape: {emb.shape}, hidden_states shape: {hidden_states.shape}, encoder_hidden_states shape: {encoder_hidden_states.shape}")
            # emb shape: torch.Size([4, 14, 512]), hidden_states shape: torch.Size([4, 14, 32, 60, 90]), encoder_hidden_states shape: torch.Size([4, 226, 4096])

            if self.ofs_embedding is not None:
                ofs_emb = self.ofs_proj(ofs)
                ofs_emb = ofs_emb.to(dtype=hidden_states.dtype)
                ofs_emb = self.ofs_embedding(ofs_emb)
                # print(f"ofs_emb shape: {ofs_emb.shape}, emb shape: {emb.shape}")
                # ofs_emb shape: torch.Size([1, 512]), emb shape: torch.Size([4, 14, 512])
                # ofs_emb is broadcasted to match the shape
                
                # 先把 ofs_emb 变成 [1, 1, 512]
                ofs_emb_expanded = ofs_emb.unsqueeze(1)  # shape: [1, 1, 512]
                # 复制数据到 [4, 14, 512]
                ofs_emb_repeated = ofs_emb_expanded.repeat(emb.shape[0], emb.shape[1], 1)

                emb = emb + ofs_emb_repeated
                # print(f"emb shape after adding ofs_emb: {emb.shape}")
                # emb shape after adding ofs_emb: torch.Size([4, 14, 512])

        else:
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
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        # print(f"hidden_states shape after patch embedding: {hidden_states.shape}, encoder_hidden_states shape: {encoder_hidden_states.shape}")
        # hidden_states shape after patch embedding: torch.Size([4, 9676, 3072]), encoder_hidden_states shape: torch.Size([4, 226, 4096]) 
        hidden_states = self.embedding_dropout(hidden_states)

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        # print(f"hidden_states shape after patch embedding: {hidden_states.shape}, encoder_hidden_states shape: {encoder_hidden_states.shape}");assert 0
        # hidden_states shape after patch embedding: torch.Size([4, 9450, 3072]), encoder_hidden_states shape: torch.Size([4, 226, 3072]) 

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