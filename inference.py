from pathlib import Path
import argparse
import logging

import torch
from torchvision import transforms
from torchvision.io import write_video
from tqdm import tqdm

from diffusers import (
    CogVideoXDPMScheduler,
    CogVideoXPipeline,
)

from transformers import set_seed
from typing import Dict, Tuple
from diffusers.models.embeddings import get_3d_rotary_pos_embed

import json
import os
import cv2
from PIL import Image

from pathlib import Path
import imageio.v3 as iio
import glob

from datetime import datetime
import gc

# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip

decord.bridge.set_bridge("torch")

logging.basicConfig(level=logging.INFO)

from transformers import T5EncoderModel, T5Tokenizer
from diffusers import AutoencoderKLCogVideoX, CogVideoXDDIMScheduler, CogVideoXTransformer3DModel, CogVideoXImageToVideoPipeline, CogVideoXPipeline
from models.cogvideox_tracking import CogVideoXImageToVideoPipelineTracking
from safetensors.torch import safe_open, load_file

### support DeT attnprocessors
from models.det_processor import SkipConv1dCogVideoXAttnProcessor2_0, MotionResidualCogVideoXAttnProcessor2_0
from models.cogvideox_tracking import transformer_load_skipconv1d, transformer_enable_motion

### support low pass filter
from models.cogvideox_tracking import prepare_lp

### support dynamic noise
from models.cogvideox_tracking import find_nearest_timestep

### support control feature projector 
from models.cogvideox_tracking import transformer_load_cfp, transformer_enable_midresidual, transformer_enable_unetresidual

import contextlib

### support multistep inference 
from diffusers.pipelines.cogvideo.pipeline_cogvideox import retrieve_timesteps
from diffusers.utils.torch_utils import randn_tensor

# 0 ~ 1
to_tensor = transforms.ToTensor()
video_exts = ['.mp4', '.avi', '.mov', '.mkv']
fr_metrics = ['psnr', 'ssim', 'lpips', 'dists']

@contextlib.contextmanager
def use_custom_attention(attn_type):
    original_fn = F.scaled_dot_product_attention
    if attn_type == 'sage':
        F.scaled_dot_product_attention = sageattn
        print("[Context] Using SAGE Attention")
    elif attn_type == 'fa3':
        from sageattention.fa3_wrapper import fa3
        F.scaled_dot_product_attention = fa3
        print("[Context] Using FA3 Attention")
    elif attn_type == 'fa3_fp8':
        from sageattention.fa3_wrapper import fa3_fp8
        F.scaled_dot_product_attention = fa3_fp8
        print("[Context] Using FA3 FP8 Attention")

    try:
        yield  # 进入上下文
    finally:
        F.scaled_dot_product_attention = original_fn
        print("[Context] Restored default Attention")


def no_grad(func):
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return wrapper


def is_video_file(filename):
    return any(filename.lower().endswith(ext) for ext in video_exts)


def read_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(to_tensor(Image.fromarray(rgb)))
    cap.release()
    return torch.stack(frames)


def read_image_folder(folder_path):
    image_files = sorted([
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    frames = [to_tensor(Image.open(p).convert("RGB")) for p in image_files]
    return torch.stack(frames)


def load_sequence(path):
    # return a tensor of shape [F, C, H, W] // 0, 1
    if os.path.isdir(path):
        return read_image_folder(path)
    elif os.path.isfile(path):
        if is_video_file(path):
            return read_video_frames(path)
        elif path.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Treat image as a single-frame video
            img = to_tensor(Image.open(path).convert("RGB"))
            return img.unsqueeze(0)  # [1, C, H, W]
    raise ValueError(f"Unsupported input: {path}")

@no_grad
def compute_metrics(pred_frames, gt_frames, metrics_model, metric_accumulator, file_name):

    print(f"\n\n[{file_name}] Metrics:", end=" ")
    for name, model in metrics_model.items():
        scores = []
        for i in range(pred_frames.shape[0]):
            pred = pred_frames[i].unsqueeze(0)
            if gt_frames != None:
                gt = gt_frames[i].unsqueeze(0)
            if name in fr_metrics:
                score = model(pred, gt).item()
            else:
                score = model(pred).item()
            scores.append(score)
        val = sum(scores) / len(scores)
        metric_accumulator[name].append(val)
        print(f"{name.upper()}={val:.4f}", end="  ")
    print()


def save_frames_as_png(video, output_dir, fps=8):
    """
    Save video frames as PNG sequence.

    Args:
        video (torch.Tensor): shape [B, C, F, H, W], float in [0, 1]
        output_dir (str): directory to save PNG files
        fps (int): kept for API compatibility
    """
    video = video[0]  # Remove batch dimension
    video = video.permute(1, 2, 3, 0)  # [F, H, W, C]

    os.makedirs(output_dir, exist_ok=True)
    frames = (video * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    
    for i, frame in enumerate(frames):
        filename = os.path.join(output_dir, f"{i:03d}.png")
        Image.fromarray(frame).save(filename)


def save_video_with_imageio_lossless(video, output_path, fps=8):
    """
    Save a video tensor to .mkv using imageio.v3.imwrite with ffmpeg backend.

    Args:
        video (torch.Tensor): shape [B, C, F, H, W], float in [0, 1]
        output_path (str): where to save the .mkv file
        fps (int): frames per second
    """
    video = video[0]
    video = video.permute(1, 2, 3, 0)

    frames = (video * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()

    iio.imwrite(
        output_path,
        frames,
        fps=fps,
        codec='libx264rgb',
        pixelformat='rgb24',
        macro_block_size=None,
        ffmpeg_params=['-crf', '0'],
    )


def save_video_with_imageio(video, output_path, fps=8, format='yuv444p'):
    """
    Save a video tensor to .mp4 using imageio.v3.imwrite with ffmpeg backend.

    Args:
        video (torch.Tensor): shape [B, C, F, H, W], float in [0, 1]
        output_path (str): where to save the .mp4 file
        fps (int): frames per second
    """
    video = video[0]
    video = video.permute(1, 2, 3, 0)

    frames = (video * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()

    if format == 'yuv444p':
        iio.imwrite(
            output_path,
            frames,
            fps=fps,
            codec='libx264',
            pixelformat='yuv444p',
            macro_block_size=None,
            ffmpeg_params=['-crf', '0'],
        )
    else:
        iio.imwrite(
            output_path,
            frames,
            fps=fps,
            codec='libx264',
            pixelformat='yuv420p',
            macro_block_size=None,
            ffmpeg_params=['-crf', '10'],
        )


def preprocess_video_match(
    video_path: Path | str,
    is_match: bool = False,
) -> torch.Tensor:
    """
    Loads a single video.

    Args:
        video_path: Path to the video file.
    Returns:
        A torch.Tensor with shape [F, C, H, W] where:
          F = number of frames
          C = number of channels (3 for RGB)
          H = height
          W = width
    """
    if isinstance(video_path, str):
        video_path = Path(video_path)
    video_reader = decord.VideoReader(uri=video_path.as_posix())
    original_fps = video_reader.get_avg_fps()
    # print(f"Video FPS: {ori_fps}")

    video_num_frames = len(video_reader)
    frames = video_reader.get_batch(list(range(video_num_frames)))
    F, H, W, C = frames.shape
    original_shape = (F, H, W, C)
    
    pad_f = 0
    pad_h = 0
    pad_w = 0

    if is_match:
        remainder = (F - 1) % 8
        if remainder != 0:
            last_frame = frames[-1:]
            pad_f = 8 - remainder
            repeated_frames = last_frame.repeat(pad_f, 1, 1, 1)
            frames = torch.cat([frames, repeated_frames], dim=0)

        pad_h = (16 - H % 16) % 16
        pad_w = (16 - W % 16) % 16
        if pad_h > 0 or pad_w > 0:
            # pad = (w_left, w_right, h_top, h_bottom)
            frames = torch.nn.functional.pad(frames, pad=(0, 0, 0, pad_w, 0, pad_h))  # pad right and bottom

    # to F, C, H, W
    return frames.float().permute(0, 3, 1, 2).contiguous(), pad_f, pad_h, pad_w, original_shape, original_fps


def remove_padding_and_extra_frames(video, pad_F, pad_H, pad_W):
    if pad_F > 0:
        video = video[:, :, :-pad_F, :, :]
    if pad_H > 0:
        video = video[:, :, :, :-pad_H, :]
    if pad_W > 0:
        video = video[:, :, :, :, :-pad_W]
    
    return video


def make_temporal_chunks(F, chunk_len, overlap_t=8):
    """
    Args:
        F: total number of frames
        chunk_len: int, chunk length in time (excluding overlap)
        overlap: int, number of overlapping frames between chunks
    Returns:
        time_chunks: List of (start_t, end_t) tuples
    """
    if chunk_len == 0:
        return [(0, F)]

    effective_stride = chunk_len - overlap_t
    if effective_stride <= 0:
        raise ValueError("chunk_len must be greater than overlap")

    chunk_starts = list(range(0, F - overlap_t, effective_stride))
    if chunk_starts[-1] + chunk_len < F:
        chunk_starts.append(F - chunk_len)

    time_chunks = []
    for i, t_start in enumerate(chunk_starts):
        t_end = min(t_start + chunk_len, F)
        time_chunks.append((t_start, t_end))

    if len(time_chunks) >= 2 and time_chunks[-1][1] - time_chunks[-1][0] < chunk_len:
        last = time_chunks.pop()
        prev_start, _ = time_chunks[-1]
        time_chunks[-1] = (prev_start, last[1])

    return time_chunks


def make_spatial_tiles(H, W, tile_size_hw, overlap_hw=(32, 32)):
    """
    Args:
        H, W: height and width of the frame
        tile_size_hw: Tuple (tile_height, tile_width)
        overlap_hw: Tuple (overlap_height, overlap_width)
    Returns:
        spatial_tiles: List of (start_h, end_h, start_w, end_w) tuples
    """
    tile_height, tile_width = tile_size_hw
    overlap_h, overlap_w = overlap_hw

    if tile_height == 0 or tile_width == 0:
        return [(0, H, 0, W)]

    tile_stride_h = tile_height - overlap_h
    tile_stride_w = tile_width - overlap_w

    if tile_stride_h <= 0 or tile_stride_w <= 0:
        raise ValueError("Tile size must be greater than overlap")

    h_tiles = list(range(0, H - overlap_h, tile_stride_h))
    if not h_tiles or h_tiles[-1] + tile_height < H:
        h_tiles.append(H - tile_height)
    
     # Merge last row if needed
    if len(h_tiles) >= 2 and h_tiles[-1] + tile_height > H:
        h_tiles.pop()

    w_tiles = list(range(0, W - overlap_w, tile_stride_w))
    if not w_tiles or w_tiles[-1] + tile_width < W:
        w_tiles.append(W - tile_width)
    
    # Merge last column if needed
    if len(w_tiles) >= 2 and w_tiles[-1] + tile_width > W:
        w_tiles.pop()

    spatial_tiles = []
    for h_start in h_tiles:
        h_end = min(h_start + tile_height, H)
        if h_end + tile_stride_h > H:
            h_end = H
        for w_start in w_tiles:
            w_end = min(w_start + tile_width, W)
            if w_end + tile_stride_w > W:
                w_end = W
            spatial_tiles.append((h_start, h_end, w_start, w_end))
    return spatial_tiles


def get_valid_tile_region(t_start, t_end, h_start, h_end, w_start, w_end,
                          video_shape, overlap_t, overlap_h, overlap_w):
    _, _, F, H, W = video_shape

    t_len = t_end - t_start
    h_len = h_end - h_start
    w_len = w_end - w_start

    valid_t_start = 0 if t_start == 0 else overlap_t // 2
    valid_t_end = t_len if t_end == F else t_len - overlap_t // 2
    valid_h_start = 0 if h_start == 0 else overlap_h // 2
    valid_h_end = h_len if h_end == H else h_len - overlap_h // 2
    valid_w_start = 0 if w_start == 0 else overlap_w // 2
    valid_w_end = w_len if w_end == W else w_len - overlap_w // 2

    out_t_start = t_start + valid_t_start
    out_t_end = t_start + valid_t_end
    out_h_start = h_start + valid_h_start
    out_h_end = h_start + valid_h_end
    out_w_start = w_start + valid_w_start
    out_w_end = w_start + valid_w_end

    return {
        "valid_t_start": valid_t_start, "valid_t_end": valid_t_end,
        "valid_h_start": valid_h_start, "valid_h_end": valid_h_end,
        "valid_w_start": valid_w_start, "valid_w_end": valid_w_end,
        "out_t_start": out_t_start, "out_t_end": out_t_end,
        "out_h_start": out_h_start, "out_h_end": out_h_end,
        "out_w_start": out_w_start, "out_w_end": out_w_end,
    }


def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    num_frames: int,
    transformer_config: Dict,
    vae_scale_factor_spatial: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:

    grid_height = height // (vae_scale_factor_spatial * transformer_config.patch_size)
    grid_width = width // (vae_scale_factor_spatial * transformer_config.patch_size)

    if transformer_config.patch_size_t is None:
        base_num_frames = num_frames
    else:
        base_num_frames = (
            num_frames + transformer_config.patch_size_t - 1
        ) // transformer_config.patch_size_t
    freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
        embed_dim=transformer_config.attention_head_dim,
        crops_coords=None,
        grid_size=(grid_height, grid_width),
        temporal_size=base_num_frames,
        grid_type="slice",
        max_size=(grid_height, grid_width),
        device=device,
    )

    return freqs_cos, freqs_sin
    
@no_grad
def process_video(
    pipe: CogVideoXPipeline,
    video: torch.Tensor,
    prompt: str = '',
    noise_step: int = 0,
    sr_noise_step: int = 399,
    args: Dict = None,
):
    # SR the video frames based on the prompt.
    # `num_frames` is the Number of frames to generate.

    ### check the number of frames of video as the last chunk may be %4 != 1 
    FLAG_CHUNK_PADDING = True
    if FLAG_CHUNK_PADDING == True and video.shape[2] % 4 != 1 and args.chunk_len > 0:
        print(f"Warning: The number of frames {video.shape[2]} is not compatible with chunk padding. It should be %4 == 1.")
        # Pad the video to make the number of frames % 4 == 1
        chunk_F = video.shape[2]
        chunk_remainder = (chunk_F - 1) % 4
        if chunk_remainder != 0:
            last_frame = video[:,:,-1:,:,:]
            pad_f = 4 - chunk_remainder
            repeated_frames = last_frame.repeat(1, 1, pad_f, 1, 1)
            video = torch.cat([video, repeated_frames], dim=2)
            assert video.shape[2] % 4 == 1, f"After padding, the number of frames {video.shape[2]} is still not compatible with chunk padding. It should be %4 == 1."

    if getattr(args, "enable_dynanoise", False):
        if 0:
            # video.shape B C F H W
            # 初始化 clipiqa
            clipiqa_metric = pyiqa.create_metric('clipiqa', device=pipe.vae.device)

            # print(f"Tracking frames shape: {tracking_frames.shape}, torch.min: {torch.min(tracking_frames)}, torch.max: {torch.max(tracking_frames)}, dtype: {tracking_frames.dtype}, device: {tracking_frames.device}")
            # assert 0 # Tracking frames shape: torch.Size([49, 3, 480, 720]), torch.min: -1.0, torch.max: 1.0, dtype: torch.float32, device: cpu
            
            # Tensor inputs, img_tensor_x/y: (N, 3, H, W), RGB, 0 ~ 1
            video_clipiqa_input = video.permute(0, 2, 1, 3, 4)[0]  # to [B, F, C, H, W]
            assert video.shape[0] == 1, "CLIP-IQA only supports batch size of 1 for now."

            FLAG_ENABLE_CLIPIQA_CHUNKS = True
            if FLAG_ENABLE_CLIPIQA_CHUNKS:
                clipiqa_chunk_len = 16
                weighted_score_sum = 0.0
                total_frame_count = 0

                chunk_score = []
                if video_clipiqa_input.shape[0] > clipiqa_chunk_len:
                    # Split into chunks of clipiqa_chunk_len frames
                    for _i in range(0, video_clipiqa_input.shape[0], clipiqa_chunk_len):
                        chunk = video_clipiqa_input[_i:_i + clipiqa_chunk_len]
                        # Normalize to [0, 1]
                        chunk = (chunk + 1.0) / 2.0
                        current_scores = clipiqa_metric(chunk)
                        current_scores = current_scores.squeeze()  # shape: [N]
                        weighted_score_sum += current_scores.sum().item()
                        total_frame_count += current_scores.numel()
                    clipiqa_score = weighted_score_sum / total_frame_count

            else:
                video_clipiqa_input = (video_clipiqa_input + 1.0) / 2.0
                clipiqa_score = clipiqa_metric(video_clipiqa_input).mean().item()

            print(f"Average CLIP-IQA score for video: {clipiqa_score:.4f}")
            # args.tmp_clipiqa_score = avg_score

            # ✅ 主动释放资源
            del clipiqa_metric, video_clipiqa_input
            gc.collect()
            torch.cuda.empty_cache()  # 如果 device 是 'cuda'
    
    # Add fixed noise for dynanoise v3
    if getattr(args, "enable_dynanoise", False):
        if 0:
            image_noise_sigma = torch.normal(
                mean=-3.0, std=0.5,
                size=(video.size(0),),
                device=video.device, dtype=video.dtype
            )
            image_noise_sigma = torch.exp(image_noise_sigma)
            video = video + torch.randn_like(video) * image_noise_sigma[:, None, None, None, None]
        
    # Encode video
    # video = video.to(pipe.vae.device, dtype=pipe.vae.dtype)
    video = video.to(pipe._execution_device, dtype=pipe.vae.dtype) # support cpu offload

    latent_dist = pipe.vae.encode(video).latent_dist
    latent = latent_dist.sample() * pipe.vae.config.scaling_factor

    patch_size_t = pipe.transformer.config.patch_size_t
    if patch_size_t is not None:
        ncopy = latent.shape[2] % patch_size_t
        # Copy the first frame ncopy times to match patch_size_t
        first_frame = latent[:, :, :1, :, :]  # Get first frame [B, C, 1, H, W]
        latent = torch.cat([first_frame.repeat(1, 1, ncopy, 1, 1), latent], dim=2)

        assert latent.shape[2] % patch_size_t == 0

    batch_size, num_channels, num_frames, height, width = latent.shape

    # Get prompt embeddings
    prompt_token_ids = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.transformer.config.max_text_seq_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    prompt_token_ids = prompt_token_ids.input_ids
    prompt_embedding = pipe.text_encoder(
        prompt_token_ids.to(latent.device)
    )[0]
    _, seq_len, _ = prompt_embedding.shape
    prompt_embedding = prompt_embedding.view(batch_size, seq_len, -1).to(dtype=latent.dtype)

    latent = latent.permute(0, 2, 1, 3, 4) # [B, F, C, H, W]

    # Add noise to latent (Select)
    if noise_step != 0:
        noise = torch.randn_like(latent)
        add_timesteps = torch.full(
            (batch_size,),
            fill_value=noise_step,
            dtype=torch.long,
            device=latent.device,
        )
        latent = pipe.scheduler.add_noise(latent, noise, add_timesteps)
    
    # 8. Create ofs embeds if required
    ofs_emb = None if pipe.transformer.config.ofs_embed_dim is None else latent.new_full((1,), fill_value=2.0)

    if getattr(args, "enable_dynanoise", False):

        if 0:
            avg_score = torch.tensor(clipiqa_score, dtype=latent.dtype, device=latent.device)
            sr_noise_step = find_nearest_timestep(avg_score, pipe.scheduler)
            # print(clipiqa_score, sr_noise_step); assert 0 # tensor([0.3457, 0.1904, 0.2168, 0.3281], device='cuda:1', dtype=torch.bfloat16) tensor([609, 741, 716, 623], device='cuda:1')
            timesteps = sr_noise_step.to(device=latent.device, dtype=torch.long)  # shape: [B]

            # add noise to latent
            noise_clipiqascore = torch.randn_like(latent)
            latent = pipe.scheduler.add_noise(latent, noise_clipiqascore, timesteps)

        else:
            timesteps = torch.full(
                (batch_size,),
                fill_value=sr_noise_step,
                dtype=torch.long,
                device=latent.device,
            )

            # v4
            # v4 use tracking_maps as noisy_video_latents
            add_noise_step = args.add_noise_step
            add_timesteps = torch.full(
                (batch_size,),
                fill_value=add_noise_step,
                dtype=torch.long,
                device=latent.device,
            )
            
            noise = torch.randn_like(latent)
            noisy_video_latents = pipe.scheduler.add_noise(latent, noise, add_timesteps)
            # timestep: 399, sqrt_alpha_prod: tensor([0.6273], dtype=torch.float64), sqrt_one_minus_alpha_prod: tensor([0.7788], dtype=torch.float64)


    elif getattr(args, "enable_multistep", False):
        # 4. Prepare timesteps
        num_inference_steps = args.num_inference_steps
        timesteps = None
        timesteps, num_inference_steps = retrieve_timesteps(pipe.scheduler, num_inference_steps, latent.device, timesteps)
        # print(timesteps) 
        # tensor([999, 979, 959, 939, 919, 899, 879, 859, 839, 819, 799, 779, 759, 739,
        # 719, 699, 679, 659, 639, 619, 599, 579, 559, 539, 519, 499, 479, 459,
        # 439, 419, 399, 379, 359, 339, 319, 299, 279, 259, 239, 219, 199, 179,
        # 159, 139, 119,  99,  79,  59,  39,  19], device='cuda:0')

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * pipe.scheduler.order, 0)

    else:
        timesteps = torch.full(
            (batch_size,),
            fill_value=sr_noise_step,
            dtype=torch.long,
            device=latent.device,
        )
        noisy_video_latents = latent

    # Prepare rotary embeds
    vae_scale_factor_spatial = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
    transformer_config = pipe.transformer.config
    rotary_emb = (
        prepare_rotary_positional_embeddings(
            height=height * vae_scale_factor_spatial,
            width=width * vae_scale_factor_spatial,
            num_frames=num_frames,
            transformer_config=transformer_config,
            vae_scale_factor_spatial=vae_scale_factor_spatial,
            device=latent.device,
        )
        if pipe.transformer.config.use_rotary_positional_embeddings
        else None
    )

    if getattr(args, "load_skipconv1d", False) or getattr(args, "enable_motion", False):
        assert not (getattr(args, "load_skipconv1d", False) and getattr(args, "enable_motion", False)), "You cannot enable both --load_skipconv1d and --enable_motion. Please choose one."
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
            print(f"✅ Set spatial-temporal params for {modified_count} SkipConv1dCogVideoXAttnProcessor2_0 processors with height={height}, width={width}, frames={frames}, dim={dim}") if modified_count > 0 else None

            modified_count_motion = 0
            for name, module in model.attn_processors.items():
                if isinstance(module, MotionResidualCogVideoXAttnProcessor2_0):
                    modified_count_motion += 1
                    module.height = height
                    module.width = width
                    module.frames = frames
                    module.dim = dim
            print(f"✅ Set spatial-temporal params for {modified_count_motion} MotionResidualCogVideoXAttnProcessor2_0 processors with height={height}, width={width}, frames={frames}, dim={dim}") if modified_count_motion > 0 else None

        # print(latent.shape) # torch.Size([1, 28, 16, 120, 160]) B T C H W ;assert 0  input size: 206438400 = 14(T) * 60(H) * 80(W) * 3072(dim)
        attn_processors_height = latent.shape[3] // transformer.config.patch_size
        attn_processors_width = latent.shape[4] // transformer.config.patch_size
        attn_processors_frames = latent.shape[1] // transformer.config.patch_size_t
        attn_processors_dim = transformer.config.num_attention_heads * transformer.config.attention_head_dim
        set_spatial_temporal_params(transformer, height=attn_processors_height, width=attn_processors_width, frames=attn_processors_frames, dim=attn_processors_dim)


    assert not (getattr(args, "use_low_pass_guidance", False) and getattr(args, "enable_alg", False)), "You cannot enable both --use_low_pass_guidance and --enable_alg. Please choose one. Because they are valid for muti-step and single-step respectively."
    
    if getattr(args, "enable_multistep", False):
        do_classifier_free_guidance = False
        eta = 0.0
        device = pipe._execution_device
        generator = torch.Generator().manual_seed(args.seed) if args.seed is not None else None
        extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta)

        ### generate noisy latents
        # latents = torch.randn_like(latent)
        # print(batch_size, num_channels, num_frames, height, width);assert 0 # 1 16 28 240 320
        shape = (
            batch_size,
            num_frames,
            num_channels,
            height,
            width,
        )
        latents = randn_tensor(shape, generator=generator, device=device, dtype=prompt_embedding.dtype)
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * pipe.scheduler.init_noise_sigma
        # latents = latent


        with pipe.progress_bar(total=num_inference_steps) as progress_bar:
            old_pred_original_sample = None
            for i, t in enumerate(timesteps):
                pipe._current_timestep = t

                # Low-pass version input
                if getattr(args, "enable_alg", False):
                    from models import lp_utils

                    lp_strength_schedule_type = "interval" # Scheduling type for low-pass filtering strength. Options: {"none", "linear", "interval", "exponential"}

                    # --- Constant Interval Scheduling Params for LP Strength ---
                    schedule_interval_start_time = 0.0
                    schedule_interval_end_time = 0.04

                    # --- Linear Scheduling Params for LP Strength ---
                    schedule_linear_start_weight = None
                    schedule_linear_end_weight = None
                    schedule_linear_end_time = None

                    # --- Exponential Scheduling Params for LP Strength ---
                    schedule_exp_decay_rate = 10.0

                    # Timestep scheduled low-pass filter strength ([0, 1] range)
                    lp_strength = lp_utils.get_lp_strength(
                        step_index=i,
                        total_steps=num_inference_steps,
                        lp_strength_schedule_type=lp_strength_schedule_type,
                        schedule_interval_start_time=schedule_interval_start_time,
                        schedule_interval_end_time=schedule_interval_end_time,
                        schedule_linear_start_weight=schedule_linear_start_weight,
                        schedule_linear_end_weight=schedule_linear_end_weight,
                        schedule_linear_end_time=schedule_linear_end_time,
                        schedule_exp_decay_rate=schedule_exp_decay_rate,
                    )

                    # Timestep scheduled low-pass filter strength ([0, 1] range)
                    # lp_strength = 1.0
                    lp_filter_type = 'down_up'
                    lp_filter_in_latent = True
                    lp_resize_factor = 0.25 # 0.25
                    # modulated_lp_resize_factor = lp_resize_factor
                    modulated_lp_resize_factor = 1.0 - (1.0 - lp_resize_factor) * lp_strength

                    print(f"Step {i}, LP Strength: {lp_strength}, LP Resize Factor: {modulated_lp_resize_factor}") if modulated_lp_resize_factor != 1.0 else None
                    # Step 0, LP Strength: 1.0
                    # Step 1, LP Strength: 1.0
                    # Step 2, LP Strength: 0.0

                    # low-pass filter
                    # latent B F C H W
                    lp_latent = prepare_lp(
                        # --- Filter Selection & Strength (Modulated) ---
                        lp_filter_type=lp_filter_type,
                        lp_resize_factor=modulated_lp_resize_factor,
                        orig_image_latents=latent, # Shape [B, F_padded, C, H, W]
                    )
                else:
                    lp_latent = latent

                # latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = torch.cat([latents + latent] * 2) if do_classifier_free_guidance else latents + latent
                latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

                latent_image_input = torch.cat([lp_latent] * 2) if do_classifier_free_guidance else lp_latent
                
                # print(latent_model_input.shape, latent_image_input.shape, noise.shape);assert 0
                # torch.Size([2, 14, 16, 60, 90]) torch.Size([2, 14, 16, 60, 90]) torch.Size([1, 14, 16, 60, 90]) torch.Size([1, 13, 16, 60, 90])
                # torch.Size([1, 28, 16, 120, 160]) torch.Size([1, 28, 16, 120, 160]) torch.Size([1, 28, 16, 120, 160]) 
                latent_model_input = torch.cat([latent_model_input, latent_image_input], dim=2)
                del latent_image_input

                tracking_maps_input = None

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                # Predict noise
                pipe.transformer.to(dtype=latent_model_input.dtype)
                # predict noise model_output
                noise_pred = pipe.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embedding,
                    timestep=timestep,
                    ofs=ofs_emb,
                    image_rotary_emb=rotary_emb,
                    attention_kwargs=None,
                    return_dict=False,
                )[0]

                del latent_model_input           
                if tracking_maps_input is not None:
                    del tracking_maps_input
                noise_pred = noise_pred.float()

                # compute the previous noisy sample x_t -> x_t-1
                if not isinstance(pipe.scheduler, CogVideoXDPMScheduler):
                    latents = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                else:
                    latents, old_pred_original_sample = pipe.scheduler.step(
                        noise_pred,
                        old_pred_original_sample,
                        t,
                        timesteps[i - 1] if i > 0 else None,
                        latents,
                        **extra_step_kwargs,
                        return_dict=False,
                    )
                del noise_pred
                latents = latents.to(prompt_embedding.dtype)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0):
                    progress_bar.update()

        latent_generate = latents

    else:

        # Low-pass version input
        if getattr(args, "use_low_pass_guidance", False):
            # Timestep scheduled low-pass filter strength ([0, 1] range)
            lp_strength = 1.0
            lp_filter_type = 'down_up'
            lp_filter_in_latent = True
            lp_resize_factor = 0.25 # 0.25
            modulated_lp_resize_factor = lp_resize_factor

            # low-pass filter
            # latent B F C H W
            lp_latent = prepare_lp(
                # --- Filter Selection & Strength (Modulated) ---
                lp_filter_type=lp_filter_type,
                lp_resize_factor=modulated_lp_resize_factor,
                orig_image_latents=latent, # Shape [B, F_padded, C, H, W]
            )

            if getattr(args, "use_low_pass_guidance_latent", False):
                # low-pass filter in latent space
                latent = lp_latent
                
        else:
            lp_latent = latent


        ### single step denoising loop
        ### handle i2v model input, i.e., concat input channels. ofs_emb is not None only in I2V models.
        # print(latent.shape);assert 0 # torch.Size([1, 28, 16, 120, 160])
        if getattr(args, "enable_t2v", False):
            latent_model_input = noisy_video_latents if ofs_emb is not None else latent
        else:
            latent_model_input = torch.cat([noisy_video_latents, lp_latent], dim=2) if ofs_emb is not None else latent
                
        # Predict noise
        if getattr(args, "attention_type", None) is not None:
            with use_custom_attention(args.attention_type):
                predicted_noise = pipe.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embedding,
                    timestep=timesteps,
                    ofs=ofs_emb,
                    image_rotary_emb=rotary_emb,
                    return_dict=False,
                )[0]
        else:
            predicted_noise = pipe.transformer(
                hidden_states=latent_model_input,
                encoder_hidden_states=prompt_embedding,
                timestep=timesteps,
                ofs=ofs_emb,
                image_rotary_emb=rotary_emb,
                return_dict=False,
            )[0]
        
        ### support I2V model by check if ofs_emb is None
        # print(predicted_noise.shape, latent.shape, latent[:, :, :latent.shape[2] // 2, :, :].shape, ofs_emb is None)
        # latent_generate = pipe.scheduler.get_velocity(
        #     predicted_noise, latent if ofs_emb is None else latent[:, :, :latent.shape[2] // 2, :, :], timesteps
        # )
        latent_generate = pipe.scheduler.get_velocity(
            predicted_noise, latent, timesteps
        )

    # generate video
    if patch_size_t is not None and ncopy > 0:
        latent_generate = latent_generate[:, ncopy:, :, :, :]

    # [B, C, F, H, W]
    video_generate = pipe.decode_latents(latent_generate)
    video_generate = (video_generate * 0.5 + 0.5).clamp(0.0, 1.0)
    
    if FLAG_CHUNK_PADDING and args.chunk_len > 0:
        # check if chunk_remainder in local variables
        if 'chunk_remainder' in locals():
            assert chunk_remainder != 0, "chunk_remainder should not be 0 when FLAG_CHUNK_PADDING is True"
            # Remove the last frames that were added for padding
            video_generate = video_generate[:, :, :-pad_f, :, :]

    return video_generate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VSR using DOVE")

    parser.add_argument("--input_dir", type=str)

    parser.add_argument("--input_json", type=str, default=None)

    parser.add_argument("--gt_dir", type=str, default=None)

    parser.add_argument("--eval_metrics", type=str, default='') # 'psnr,ssim,lpips,dists,clipiqa,musiq,maniqa,niqe'

    parser.add_argument("--model_path", type=str)

    parser.add_argument("--lora_path", type=str, default=None, help="The path of the LoRA weights to be used")

    parser.add_argument("--output_path", type=str, default="./results", help="The path save generated video")

    parser.add_argument("--fps", type=int, default=0, help="The frames per second for the generated video")

    parser.add_argument("--dtype", type=str, default="bfloat16", help="The data type for computation")

    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")

    parser.add_argument("--upscale_mode", type=str, default="bilinear")

    parser.add_argument("--upscale", type=int, default=4)

    parser.add_argument("--noise_step", type=int, default=0)

    parser.add_argument("--sr_noise_step", type=int, default=399)

    parser.add_argument("--add_noise_step", type=int, default=399)

    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps")

    parser.add_argument("--is_cpu_offload", action="store_true", help="Enable CPU offload for the model")

    parser.add_argument("--is_vae_st", action="store_true", help="Enable VAE slicing and tiling")

    parser.add_argument("--png_save", action="store_true", help="Save output as PNG sequence")

    parser.add_argument("--save_format", type=str, default="yuv444p", help="Save output as PNG sequence")

    # Crop and Tiling Parameters
    parser.add_argument("--tile_size_hw", type=int, nargs=2, default=(0, 0), help="Tile size for spatial tiling (height, width)")

    parser.add_argument("--overlap_hw", type=int, nargs=2, default=(32, 32))

    parser.add_argument("--chunk_len", type=int, default=0, help="Chunk length for temporal chunking")

    parser.add_argument("--overlap_t", type=int, default=8)

    parser.add_argument("--load_skipconv1d",action="store_true",default=False,help=("Whether or not to load skipconv1d for better motion control, see https://github.com/Shi-qingyu/DeT/blob/main/train_cogvideox.py"),)

    parser.add_argument("--use_low_pass_guidance",action="store_true",default=False,help=("Whether or not use low pass filter, see https://github.com/choi403/ALG/blob/main/pipeline_cogvideox_image2video_lowpass.py"),)

    parser.add_argument("--use_low_pass_guidance_latent",action="store_true",default=False,help=("Whether or not use low pass filter, see https://github.com/choi403/ALG/blob/main/pipeline_cogvideox_image2video_lowpass.py"),)

    parser.add_argument("--enable_alg",action="store_true",default=False,help=("Whether or not use low pass filter, see https://github.com/choi403/ALG/blob/main/pipeline_cogvideox_image2video_lowpass.py"),)

    parser.add_argument("--enable_dynanoise",action="store_true",default=False,help=("Whether or not use clipiqa score to modify noise strength, see https://github.com/chaofengc/IQA-PyTorch"),)

    parser.add_argument("--enable_cfp",action="store_true",default=False,help=("Whether or not use control feature projector, see Vivid-VR https://github.com/csbhr/Vivid-VR/tree/main"),)

    parser.add_argument("--enable_vta",action="store_true",default=False,help=("Whether or not use vectorized timestep adaptation, see https://github.com/Yaofang-Liu/Pusa-VidGen/blob/main/PusaV1/examples/pusavideo/train_wan_pusa.py"),)

    parser.add_argument("--enable_multistep",action="store_true",default=False,help=("Whether or not use multi-step training/inference"),)

    parser.add_argument("--enable_motion",action="store_true",default=False,help=("Whether or not use motion residual to enhance motion information"),)
    
    parser.add_argument("--enable_pipeline_forward",action="store_true",default=False,help=("Whether or not use pipeline forward to process videos"),)

    parser.add_argument("--enable_frame_by_frame_vae",action="store_true",default=False,help=("Whether or not use frame by frame vae to process videos"),)

    parser.add_argument("--enable_midresidual",action="store_true",default=False,help=("Whether or not use DiT block 21 direct connect to 42 for residual learning"),)
    
    parser.add_argument("--enable_unetresidual",action="store_true",default=False,help=("Whether or not use DiT block 0-21 direct connect to 22-42 for residual learning"),)

    parser.add_argument("--enable_cogvideox1dot5", default=True,help=("Whether or not use multi-step training/inference"),)

    parser.add_argument("--enable_t2v",action="store_true",default=False,help=("Whether or not to use text-to-video model, ./checkpoints/CogVideoX1.5-5B"),) #

    parser.add_argument(
        "--load_skipconv1d_add_at_end",
        action="store_true",
        default=False,
        help=(
            "Whether or not to load skipconv1d for better motion control, see https://github.com/Shi-qingyu/DeT/blob/main/train_cogvideox.py"
        ),
    )
    
    parser.add_argument('--compile', action='store_true', help='Compile the model')
    
    parser.add_argument('--attention_type', type=str, default=None, choices=['sdpa', 'sage', 'fa3', 'fa3_fp8', None], help='Attention type')

    args = parser.parse_args()

    if args.attention_type is not None:
        try:
            from sageattention import sageattn
            import torch.nn.functional as F
        except ImportError:
            print("SAGE Attention not found. Please install the sageattention package.")
            args.attention_type = None

        if args.attention_type == 'sage':
            print("Using SAGE Attention")
        elif args.attention_type == 'fa3':
            print("Using FA3 Attention")
        elif args.attention_type == 'fa3_fp8':
            print("Using FA3 FP8 Attention")

    if args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    elif args.dtype == "float32":
        dtype = torch.float32
    else:
        raise ValueError("Invalid dtype. Choose from 'float16', 'bfloat16', or 'float32'.")
    
    if args.chunk_len > 0:
        print(f"Chunking video into {args.chunk_len} frames with {args.overlap_t} overlap")
        overlap_t = args.overlap_t
    else:
        overlap_t = 0
    if args.tile_size_hw != (0, 0):
        print(f"Tiling video into {args.tile_size_hw} frames with {args.overlap_hw} overlap")
        overlap_hw = args.overlap_hw
    else:
        overlap_hw = (0, 0)
    
    # Set seed
    set_seed(args.seed)

    if args.input_json is not None and args.input_json.endswith('.json'):
        with open(args.input_json, 'r') as f:
            video_prompt_dict = json.load(f)
    elif args.input_json is not None and args.input_json.endswith('.csv'):
        import pandas as pd
        df = pd.read_csv(args.input_json)
        df['basename'] = df['path'].apply(os.path.basename)
        video_prompt_dict = dict(zip(df['basename'], df['text']))
        # video_prompt_dict = dict(zip(os.path.basename(df['path']), df['text']))
        print(f"Loaded {len(video_prompt_dict)} prompts from CSV file.")
        # print(f"Example: {list(video_prompt_dict.items())[:5]}");assert 0
    else:
        video_prompt_dict = {}
    
    # Get all video files from input directory
    video_files = []
    for ext in video_exts:
        video_files.extend(glob.glob(os.path.join(args.input_dir, f'*{ext}')))
    video_files = sorted(video_files)  # Sort files for consistent ordering

    if not video_files:
        raise ValueError(f"No video files found in {args.input_dir}")

    new_output_dir = os.path.join(args.output_path)
    args.output_path = new_output_dir
    print(f"Output directory: {new_output_dir}")


    os.makedirs(args.output_path, exist_ok=True)
    
    # 1.  Load the pre-trained CogVideoX pipeline with the specified precision (bfloat16).
    # add device_map="balanced" in the from_pretrained function and remove the enable_model_cpu_offload()
    # function to use Multi GPUs.

    # pipe = CogVideoXPipeline.from_pretrained(args.model_path, torch_dtype=dtype)
    pretrained_congvideox_5b_i2v_path = "checkpoints/CogVideoX1.5-5B-I2V" if not getattr(args, "enable_t2v", False) else "checkpoints/CogVideoX1.5-5B"
    vae = AutoencoderKLCogVideoX.from_pretrained(pretrained_congvideox_5b_i2v_path, subfolder="vae", torch_dtype=torch.bfloat16)
    text_encoder = T5EncoderModel.from_pretrained(pretrained_congvideox_5b_i2v_path, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    tokenizer = T5Tokenizer.from_pretrained(pretrained_congvideox_5b_i2v_path, subfolder="tokenizer", torch_dtype=torch.bfloat16)

    if getattr(args, "load_skipconv1d", False) or getattr(args, "enable_cfp", False) or getattr(args, "enable_motion", False):
        # init from pretrained model
        transformer = CogVideoXTransformer3DModel.from_pretrained(pretrained_congvideox_5b_i2v_path, subfolder="transformer", torch_dtype=torch.bfloat16)

        ### init empty skipconv1d parameters
        if getattr(args, "load_skipconv1d", False):
            FLAG_SKIPCONV1D_v2 = True
            transformer, model_keys_after_skipconv1d = transformer_load_skipconv1d(transformer, args, NUM_MODIFIED_BLOCKS=42 if FLAG_SKIPCONV1D_v2 else 16)

        ### init empty motion residual parameters
        if getattr(args, "enable_motion", False):
            transformer, model_keys_after_motion = transformer_enable_motion(transformer, args, NUM_MODIFIED_BLOCKS=42)

        ### init empty control feature projector parameters
        if getattr(args, "enable_cfp", False):
            transformer, model_keys_after_cfp = transformer_load_cfp(transformer, args)

        ### init empty midresidual parameters
        if getattr(args, "enable_midresidual", False):
            transformer, model_keys_after_midresidual = transformer_enable_midresidual(transformer, args)

        ### init empty unetresidual parameters
        if getattr(args, "enable_unetresidual", False):
            transformer, model_keys_after_unetresidual = transformer_enable_unetresidual(transformer, args)

        ### load transformer from resume checkpoints
        _safetensor_path = os.path.join(args.model_path, "transformer", "diffusion_pytorch_model.safetensors")
        _state_dict = load_file(_safetensor_path)
        # missing_keys, unexpected_keys = transformer.load_state_dict(_state_dict) # , strict=False
        missing_keys, unexpected_keys = transformer.load_state_dict(_state_dict, strict=False) # , strict=False
        if missing_keys or unexpected_keys:
            print(f"❌ Loading SFT weights from state_dict led to missing keys not found in the model: {missing_keys}.")
            print(f"❌ Loading SFT weights from state_dict led to unexpected keys not found in the model: {unexpected_keys}.")
        else:
            print(f"✅ SFT weights loaded from {args.model_path} with {len(_state_dict)} keys")

        del _safetensor_path, _state_dict

    else:
        transformer = CogVideoXTransformer3DModel.from_pretrained(args.model_path, subfolder="transformer", torch_dtype=torch.bfloat16)

    scheduler = CogVideoXDDIMScheduler.from_pretrained(pretrained_congvideox_5b_i2v_path, subfolder="scheduler")

    if getattr(args, "enable_pipeline_forward", False):
        pipe = CogVideoXImageToVideoPipelineTracking(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )

    elif getattr(args, "enable_t2v", False):
        pipe = CogVideoXPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )
                
    else:
        pipe = CogVideoXImageToVideoPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )
    
    # If you're using with lora, add this code
    if args.lora_path:
        print(f"Loading LoRA weights from {args.lora_path}")
        pipe.load_lora_weights(
            args.lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="test_1"
        )
        pipe.fuse_lora(components=["transformer"], lora_scale=1.0) # lora_scale = lora_alpha / rank

    # 2. Set Scheduler.
    # Can be changed to `CogVideoXDPMScheduler` or `CogVideoXDDIMScheduler`.
    # We recommend using `CogVideoXDDIMScheduler` for CogVideoX-2B.
    # using `CogVideoXDPMScheduler` for CogVideoX-5B / CogVideoX-5B-I2V.

    # pipe.scheduler = CogVideoXDDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.scheduler = CogVideoXDPMScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )

    # 3. Enable CPU offload for the model.
    # turn off if you have multiple GPUs or enough GPU memory(such as H100) and it will cost less time in inference
    # and enable to("cuda")

    if args.is_cpu_offload:
        # pipe.enable_model_cpu_offload()
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to("cuda")
    
    if args.compile:
        pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")

    if args.is_vae_st:
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
    
    # pipe.transformer.eval()
    # torch.set_grad_enabled(False)

    # 4. Set the metircs
    if args.eval_metrics != '':
        metrics_list = [m.strip().lower() for m in args.eval_metrics.split(',')]
        metrics_models = {}
        for name in metrics_list:
            try:
                metrics_models[name] = pyiqa.create_metric(name).to(pipe.device).eval()
            except Exception as e:
                print(f"Failed to initialize metric '{name}': {e}")
        metric_accumulator = {name: [] for name in metrics_list}
    else:
        metrics_models = None
        metric_accumulator = None
    
    video_count = 0
    for video_path in tqdm(video_files, desc="Processing videos"):
        video_count += 1
        # if video_count != 44:
        #     continue
        video_name = os.path.basename(video_path)
        
        # # only process specified videos
        # current_video_name_list = ["受降00_00-00_08.mp4", "受降00_54-01_05.mp4"]
        # if video_name not in current_video_name_list:
        #     print(f"Skipping {video_name}")
        #     continue

        prompt = video_prompt_dict.get(video_name, "")

        ### skipping exiting videos
        if 1:
            file_name = os.path.basename(video_path)            
            suffix_use_low_pass_guidance = '_lowpass' if getattr(args, "use_low_pass_guidance", False) else ''
            suffix_use_low_pass_guidance_latent = '_lowpass_latent' if getattr(args, "use_low_pass_guidance_latent", False) else ''
            suffix_enable_dynanoise = '_dynanoise' if getattr(args, "enable_dynanoise", False) else ''
            suffix_enable_skipconv1d = '_skipconv1d' if getattr(args, "load_skipconv1d", False) else ''
            suffix_enable_motion = '_motion' if getattr(args, "enable_motion", False) else ''
            suffix_enable_cfp = '_cfp' if getattr(args, "enable_cfp", False) else ''
            suffix_enable_pipeline_forward = '_pipeline_forward' if getattr(args, "enable_pipeline_forward", False) else ''
            suffix_num_inference_steps = f"_steps{args.num_inference_steps}" if getattr(args, "enable_multistep", False) else '' 
            suffix_enable_alg = '_alg' if getattr(args, "enable_alg", False) else ''
            suffix_enable_midresidual = '_midresidual' if getattr(args, "enable_midresidual", False) else ''
            suffix_enable_unetresidual = '_unetresidual' if getattr(args, "enable_unetresidual", False) else ''
            suffix = f"{suffix_enable_skipconv1d}{suffix_use_low_pass_guidance}{suffix_use_low_pass_guidance_latent}{suffix_enable_dynanoise}{suffix_enable_cfp}{suffix_enable_pipeline_forward}{suffix_num_inference_steps}{suffix_enable_alg}{suffix_enable_motion}{suffix_enable_midresidual}{suffix_enable_unetresidual}"
            file_name_ = os.path.splitext(file_name)[0]
            output_path = os.path.join(args.output_path, f"{file_name_}{suffix}.mp4")

            if os.path.exists(output_path.replace('.mkv', '.mp4')):
                print(f"Output video {output_path.replace('.mkv', '.mp4')} exists, skipping")
                continue

        if os.path.exists(video_path):
            # Read video
            # [F, C, H, W]
            video, pad_f, pad_h, pad_w, original_shape, original_fps = preprocess_video_match(video_path, is_match=True)
            H_, W_ = video.shape[2], video.shape[3]
            video = torch.nn.functional.interpolate(video, size=(H_*args.upscale, W_*args.upscale), mode=args.upscale_mode, align_corners=False)
            __frame_transform = transforms.Compose(
                [transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)] # -1, 1
            )
            video = torch.stack([__frame_transform(f) for f in video], dim=0)
            video = video.unsqueeze(0)
            # [B, C, F, H, W]
            video = video.permute(0, 2, 1, 3, 4).contiguous()

            _B, _C, _F, _H, _W = video.shape
            time_chunks = make_temporal_chunks(_F, args.chunk_len, overlap_t)
            spatial_tiles = make_spatial_tiles(_H, _W, args.tile_size_hw, overlap_hw)

            if args.chunk_len > 0:
                output_video = torch.zeros_like(video, dtype=dtype, device='cuda')
                write_count = torch.zeros_like(video, dtype=torch.uint8, device='cuda')  # 若最大写入次数<255
            else:
                output_video = torch.zeros_like(video)
                write_count = torch.zeros_like(video, dtype=torch.int)

            print(f"Process video: {video_name} | Prompt: {prompt} | Frame: {_F} (ori: {original_shape[0]}; pad: {pad_f}) | Target Resolution: {_H}, {_W} (ori: {original_shape[1]*args.upscale}, {original_shape[2]*args.upscale}; pad: {pad_h}, {pad_w}) | Chunk Num: {len(time_chunks)*len(spatial_tiles)}")

            idx_time_chunks = 0
            for t_start, t_end in time_chunks:
                idx_time_chunks += 1
                idx_spatial_tiles = 0
                for h_start, h_end, w_start, w_end in spatial_tiles:
                    idx_spatial_tiles += 1
                    video_chunk = video[:, :, t_start:t_end, h_start:h_end, w_start:w_end]

                    if args.chunk_len > 0:
                        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print(f"video_chunk: {idx_time_chunks}.{idx_spatial_tiles}/{len(time_chunks)*len(spatial_tiles)} {video_chunk.shape} | t: {t_start}:{t_end} | h: {h_start}:{h_end} | w: {w_start}:{w_end} | time: {current_time}")
                    # print(f"video_chunk: {video_chunk.shape} | t: {t_start}:{t_end} | h: {h_start}:{h_end} | w: {w_start}:{w_end}")

                    if getattr(args, "enable_pipeline_forward", False):

                        ### check the number of frames of video as the last chunk may be %4 != 1 
                        FLAG_CHUNK_PADDING = True
                        if FLAG_CHUNK_PADDING == True and video_chunk.shape[2] % 4 != 1:
                            print(f"Warning: The number of frames {video_chunk.shape[2]} is not compatible with chunk padding. It should be %4 == 1.")
                            # Pad the video to make the number of frames % 4 == 1
                            chunk_F = video_chunk.shape[2]
                            chunk_remainder = (chunk_F - 1) % 4
                            if chunk_remainder != 0:
                                last_frame = video_chunk[:,:,-1:,:,:]
                                pad_f = 4 - chunk_remainder
                                repeated_frames = last_frame.repeat(1, 1, pad_f, 1, 1)
                                video_chunk = torch.cat([video_chunk, repeated_frames], dim=2)
                                assert video_chunk.shape[2] % 4 == 1, f"After padding, the number of frames {video_chunk.shape[2]} is still not compatible with chunk padding. It should be %4 == 1."
                                
                        batch_size, num_channels, num_frames, height, width = video_chunk.shape
                        # print(f"video_chunk after padding: {video_chunk.shape}, batch_size: {batch_size}, num_channels: {num_channels}, num_frames: {num_frames}, height: {height}, width: {width}")
                        # video_chunk after padding: torch.Size([1, 3, 105, 960, 1280]), batch_size: 1, num_channels: 3, num_frames: 105, height: 960, width: 1280

                        # [B, C, F, H, W]
                        _video_generate = pipe(
                            prompt=prompt,
                            negative_prompt="The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion.",
                            video=video_chunk,
                            num_videos_per_prompt=1,
                            num_inference_steps=args.num_inference_steps,
                            num_frames=num_frames,
                            use_dynamic_cfg=True,
                            guidance_scale=6.0,
                            generator=torch.Generator().manual_seed(args.seed),
                            height=height,
                            width=width,
                            output_type="pt",
                            args=args,  # Pass args if needed
                        ).frames[0]

                        _video_generate = _video_generate.unsqueeze(0)  # Ensure batch dimension is present  B T C H W torch.Size([1, 105, 3, 960, 1280])
                        _video_generate = _video_generate.permute(0, 2, 1, 3, 4).contiguous() # [B, C, F, H, W] torch.Size([1, 3, 105, 960, 1280])

                        if FLAG_CHUNK_PADDING:
                            # check if chunk_remainder in local variables
                            if 'chunk_remainder' in locals():
                                assert chunk_remainder != 0, "chunk_remainder should not be 0 when FLAG_CHUNK_PADDING is True"
                                # Remove the last frames that were added for padding
                                _video_generate = _video_generate[:, :, :-pad_f, :, :]

                        print(f"video_chunk: {idx_time_chunks}.{idx_spatial_tiles}/{len(time_chunks)*len(spatial_tiles)} {_video_generate.shape} | t: {t_start}:{t_end} | h: {h_start}:{h_end} | w: {w_start}:{w_end}")
                    else:
                        # [B, C, F, H, W]
                        _video_generate = process_video(
                            pipe=pipe,
                            video=video_chunk,
                            prompt=prompt,
                            noise_step=args.noise_step,
                            sr_noise_step=args.sr_noise_step,
                            args=args,
                        )    

                    region = get_valid_tile_region(
                        t_start, t_end, h_start, h_end, w_start, w_end,
                        video_shape=video.shape,
                        overlap_t=overlap_t,
                        overlap_h=overlap_hw[0],
                        overlap_w=overlap_hw[1],
                    )

                    print(region)
                    output_video[:, :, region["out_t_start"]:region["out_t_end"],
                                    region["out_h_start"]:region["out_h_end"],
                                    region["out_w_start"]:region["out_w_end"]] = \
                    _video_generate[:, :, region["valid_t_start"]:region["valid_t_end"],
                                    region["valid_h_start"]:region["valid_h_end"],
                                    region["valid_w_start"]:region["valid_w_end"]]
                    write_count[:, :, region["out_t_start"]:region["out_t_end"],
                                    region["out_h_start"]:region["out_h_end"],
                                    region["out_w_start"]:region["out_w_end"]] += 1
            
            video_generate = output_video

            if (write_count == 0).any():
                print("Error: Lack of write in region !!!")
                exit()
            if (write_count > 1).any():
                print("Error: Write count > 1 in region !!!")
                exit()

            # video_generate = remove_padding_and_extra_frames(video_generate, pad_f, pad_h*4, pad_w*4)
            video_generate = remove_padding_and_extra_frames(video_generate, pad_f, pad_h*args.upscale, pad_w*args.upscale)
            
            # file_name = os.path.basename(video_path)
            # # output_path = os.path.join(args.output_path, file_name)
            # suffix_use_low_pass_guidance = '_lowpass' if getattr(args, "use_low_pass_guidance", False) else ''
            # suffix_use_low_pass_guidance_latent = '_lowpass_latent' if getattr(args, "use_low_pass_guidance_latent", False) else ''
            # suffix_enable_dynanoise = '_dynanoise' if getattr(args, "enable_dynanoise", False) else ''
            # suffix_enable_skipconv1d = '_skipconv1d' if getattr(args, "load_skipconv1d", False) else ''
            # suffix = f"{suffix_enable_skipconv1d}{suffix_use_low_pass_guidance}{suffix_use_low_pass_guidance_latent}{suffix_enable_dynanoise}"
            # file_name_ = os.path.splitext(file_name)[0]
            # output_path = os.path.join(args.output_path, f"{file_name_}{suffix}.mp4")

            if metrics_models is not None:
                #  [1, C, F, H, W] -> [F, C, H, W]
                pred_frames = video_generate[0]
                pred_frames = pred_frames.permute(1, 0, 2, 3).contiguous()
                if args.gt_dir is not None:
                    gt_frames = load_sequence(os.path.join(args.gt_dir, file_name))
                else:
                    gt_frames = None
                compute_metrics(pred_frames, gt_frames, metrics_models, metric_accumulator, file_name)

            if args.png_save:
                # Save as PNG sequence
                output_dir = output_path.rsplit('.', 1)[0]  # Remove extension
                save_frames_as_png(video_generate, output_dir, fps=args.fps)
            else:
                output_path = output_path.replace('.mkv', '.mp4')
                save_video_with_imageio(video_generate, output_path, fps=args.fps, format=args.save_format) if args.fps != 0 else save_video_with_imageio(video_generate, output_path, fps=original_fps, format=args.save_format)

            # clear GPU memory and released variables
            # clear GPU memory and released variables
            try:
                del video, output_video, write_count, video_generate
                del _video_generate, video_chunk, pred_frames, gt_frames
            except:
                pass
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        else:
            print(f"Warning: {video_name} not found in {args.input_dir}")

    if metrics_models is not None:
        print("\n=== Overall Average Metrics ===")
        count = len(next(iter(metric_accumulator.values())))
        overall_avg = {metric: 0 for metric in metrics_list}
        out_name = 'metrics_'
        for metric in metrics_list:
            out_name += f"{metric}_"
            scores = metric_accumulator[metric]
            if scores:
                avg = sum(scores) / len(scores)
                overall_avg[metric] = avg
                print(f"{metric.upper()}: {avg:.4f}")

        out_name = out_name.rstrip('_') + '.json'
        out_path = os.path.join(args.output_path, out_name)
        output = {
            "per_sample": metric_accumulator,
            "average": overall_avg,
            "count": count
        }
        with open(out_path, 'w') as f:
            json.dump(output, f, indent=2)

    print("All videos processed.")
