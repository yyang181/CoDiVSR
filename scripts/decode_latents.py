import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import transforms
from diffusers import AutoencoderKLCogVideoX, CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video
import pyiqa
import csv
from filelock import FileLock
from torch.multiprocessing import Process, set_start_method

from pathlib import Path
import decord  # isort:skip
decord.bridge.set_bridge("torch")

import torch.multiprocessing as mp
import gc
from functools import partial

from einops import rearrange
import random

def no_grad(func):
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return wrapper

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

# ====== 子进程函数 ======
def process(rank, device_id, paths, lock_path, processing_dict_keys, csv_path, save_dir):
    # 初始化 pipe
    vae_path = "checkpoints/CogVideoX1.5-5B-I2V"
    vae = AutoencoderKLCogVideoX.from_pretrained(vae_path, subfolder="vae", torch_dtype=torch.bfloat16)
    pipe = CogVideoXImageToVideoPipeline(
        vae=vae,
        text_encoder=None,
        tokenizer=None,
        transformer=None,
        scheduler=None,
    )
    device = torch.device(f"cuda:{device_id}")
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    pipe.to(device)
    pipe.vae.eval()

    # 初始化 clipiqa
    clipiqa_metric = pyiqa.create_metric('clipiqa', device=device)
    to_tensor = transforms.ToTensor()

    for path in tqdm(paths, desc=f"GPU-{device_id}"):
        name = os.path.splitext(os.path.basename(path))[0]
        latent_dict = torch.load(path, map_location=device)

        for key in latent_dict:
            if key not in processing_dict_keys:
                continue

            latents = latent_dict[key]  # [B, T, C, H, W]
            with torch.no_grad():
                video = pipe.decode_latents(latents)
                videos = pipe.video_processor.postprocess_video(video=video, output_type="pil")

                for b, frames in enumerate(videos):
                    video_name = f"{key}_{b}.mp4"
                    video_folder = os.path.join(save_dir, name)
                    video_path = os.path.join(video_folder, video_name)
                    os.makedirs(video_folder, exist_ok=True)

                    # 计算 CLIP-IQA
                    scores = []
                    for frame in frames:
                        frame_tensor = to_tensor(frame).unsqueeze(0).to(device)
                        score = clipiqa_metric(frame_tensor).item()
                        scores.append(score)
                    avg_score = sum(scores) / len(scores)

                    export_to_video(frames, video_path, fps=25)

                    # 写入 CSV（加锁）
                    with FileLock(lock_path):
                        with open(csv_path, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([video_path, avg_score])

                    print(f"[GPU-{device_id}] Saved: {video_path}, CLIP-IQA: {avg_score:.4f}")

@no_grad
def calculate_clip_iqa():
    # ====== 设置基础路径与配置 ======
    load_dir = '/opt/data/private/yyx/code/DiffusionAsShader/exp/save_data_pt_v1'
    save_dir = '/opt/data/private/yyx/code/DiffusionAsShader/exp/save_data_pt_v1_decoded_videos'
    csv_path = os.path.join(save_dir, "clipiqa_scores.csv")
    lock_path = csv_path + ".lock"
    processing_dict_keys = ['tracking_maps']  # 需要处理的 latent key

    os.makedirs(save_dir, exist_ok=True)

    # ====== 预加载路径列表 ======
    all_pth_paths = []
    for root, _, files in os.walk(load_dir):
        for file in files:
            if file.endswith('.pth'):
                all_pth_paths.append(os.path.join(root, file))
    all_pth_paths.sort()

    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise RuntimeError("至少需要2张GPU才能运行此脚本。")

    # 平分任务
    chunk_size = (len(all_pth_paths) + world_size - 1) // world_size
    path_chunks = [all_pth_paths[i * chunk_size:(i + 1) * chunk_size] for i in range(world_size)]

    # 如果 CSV 不存在，则写入表头
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['video_path', 'clipiqa_score'])

    # 启动多 GPU 进程
    processes = []
    for rank in range(world_size):
        p = Process(target=process, args=(rank, rank, path_chunks[rank], lock_path, processing_dict_keys, csv_path, save_dir))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

@no_grad
def add_noise_to_latents():
    """向 latents 添加噪声"""
    load_dir = '/opt/data/private/yyx/code/DiffusionAsShader/exp/save_data_pt_v1'
    mp4_dir = '/opt/data/private/yyx/code/DiffusionAsShader/exp/save_data_pt_v1_decoded_videos'
    save_dir = '/opt/data/private/yyx/code/DiffusionAsShader/exp/save_data_pt_v2'
    mp4_save_dir = '/opt/data/private/yyx/code/DiffusionAsShader/exp/save_data_pt_v2_decoded_videos'
    processing_dict_keys = ['tracking_maps']  # 需要处理的 latent key

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(mp4_save_dir, exist_ok=True)

    # ====== 预加载路径列表 ======
    all_pth_paths = []
    for root, _, files in os.walk(load_dir):
        for file in files:
            if file.endswith('.pth'):
                all_pth_paths.append(os.path.join(root, file))
    all_pth_paths.sort()
    print(f"Found {len(all_pth_paths)} .pth files.", all_pth_paths[0], all_pth_paths[-1])

    all_mp4_paths = [p.replace(load_dir, mp4_dir).replace('.pth', '') for p in all_pth_paths]

    # 初始化 pipe
    vae_path = "checkpoints/CogVideoX1.5-5B-I2V"
    vae = AutoencoderKLCogVideoX.from_pretrained(vae_path, subfolder="vae", torch_dtype=torch.bfloat16)
    pipe = CogVideoXImageToVideoPipeline(
        vae=vae,
        text_encoder=None,
        tokenizer=None,
        transformer=None,
        scheduler=None,
    )
    device_id = 0
    device = torch.device(f"cuda:{device_id}")
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    pipe.to(device)
    pipe.vae.eval()

    for path, mp4_dir in tqdm(zip(all_pth_paths, all_mp4_paths), desc=f"GPU-{device_id}"):
        name = os.path.splitext(os.path.basename(path))[0]
        latent_dict = torch.load(path, map_location=device)

        # get all mp4 files from mp4_dir
        video_files = [os.path.join(mp4_dir,f) for f in os.listdir(mp4_dir) if f.endswith('.mp4') and not f.startswith('.')]
        video_files = sorted(video_files)

        # [F, C, H, W]
        batch_videos = []
        for video_file in video_files:
            video, pad_f, pad_h, pad_w, original_shape, original_fps = preprocess_video_match(video_file, is_match=True)
            # print(f"Original shape: {original_shape}, FPS: {original_fps}, Pad F: {pad_f}, Pad H: {pad_h}, Pad W: {pad_w}") # Video shape after transform: torch.Size([1, 3, 49, 480, 720]), torch.Min: -1.0, torch.Max: 1.0
            __frame_transform = transforms.Compose(
                [transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)] # -1, 1
            )
            video = torch.stack([__frame_transform(f) for f in video], dim=0)
            video = video.unsqueeze(0)
            # [B, C, F, H, W]
            video = video.permute(0, 2, 1, 3, 4).contiguous()
            # print(f"Video shape after transform: {video.shape}, torch.Min: {torch.min(video)}, torch.Max: {torch.max(video)}")
            batch_videos.append(video)
        
        batch_videos = torch.cat(batch_videos, dim=0)  # [B, C, F, H, W]
        batch_videos = batch_videos.to(pipe.vae.device, dtype=pipe.vae.dtype)
        # print(f"Batch videos shape: {batch_videos.shape}, torch.Min: {torch.min(batch_videos)}, torch.Max: {torch.max(batch_videos)}")
        # Batch videos shape: torch.Size([4, 3, 49, 480, 720]), torch.Min: -1.0, torch.Max: 1.0

        # Add noise to latents
        image_noise_sigma = torch.normal(
            mean=-3.0, std=0.5, size=(batch_videos.size(0),), device=pipe.vae.device, dtype=pipe.vae.dtype
        )
        image_noise_sigma = torch.exp(image_noise_sigma)
        noisy_images = batch_videos + torch.randn_like(batch_videos) * image_noise_sigma[:, None, None, None, None]
        
        # Encode video
        latent_dist = pipe.vae.encode(noisy_images).latent_dist
        latent = latent_dist.sample() * pipe.vae.config.scaling_factor # [B, C, F, H, W]

        # 1. save noised latent to latent_dict
        noised_latent_key = f"{processing_dict_keys[0]}_noised"
        if noised_latent_key not in latent_dict:
            latent_dict[noised_latent_key] = latent
        else:
            raise ValueError(f"Key {noised_latent_key} already exists in latent_dict. Please choose a different key.")
        save_path = os.path.join(save_dir, f"{name}.pth")
        torch.save(latent_dict, save_path)
        print(f"Saved step {name} data to {save_path}")


        latent_generate = latent.permute(0, 2, 1, 3, 4) # [B, F, C, H, W]

        # Decode latents
        decoded_video = pipe.decode_latents(latent_generate)
        decoded_video_pil = pipe.video_processor.postprocess_video(video=decoded_video, output_type="pil")

        for b, frames in enumerate(decoded_video_pil):
            video_name = f"{processing_dict_keys[0]}_{b}.mp4"
            video_path = os.path.join(mp4_save_dir, name, video_name)
            os.makedirs(os.path.dirname(video_path), exist_ok=True)

            export_to_video(frames, video_path, fps=25)

            print(f"Saved video: {video_path}")
        assert 0


def process_partition(device_id, pth_paths_partition, mp4_paths_partition):
    processing_dict_keys = ['tracking_maps']
    vae_path = "checkpoints/CogVideoX1.5-5B-I2V"
    save_dir = '/opt/data/private/yyx/code/DiffusionAsShader/exp/save_data_pt_v2'
    mp4_save_dir = '/opt/data/private/yyx/code/DiffusionAsShader/exp/save_data_pt_v2_decoded_videos'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(mp4_save_dir, exist_ok=True)

    with torch.no_grad():
        device = torch.device(f"cuda:{device_id}")
        vae = AutoencoderKLCogVideoX.from_pretrained(vae_path, subfolder="vae", torch_dtype=torch.bfloat16)
        pipe = CogVideoXImageToVideoPipeline(
            vae=vae,
            text_encoder=None,
            tokenizer=None,
            transformer=None,
            scheduler=None,
        )
        pipe.to(device)
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
        pipe.vae.eval()

        for path, mp4_dir in tqdm(zip(pth_paths_partition, mp4_paths_partition), desc=f"GPU-{device_id}", total=len(pth_paths_partition)):
            name = os.path.splitext(os.path.basename(path))[0]
            latent_dict = torch.load(path, map_location=device)

            video_files = [os.path.join(mp4_dir, f) for f in os.listdir(mp4_dir)
                        if f.endswith('.mp4') and not f.startswith('.')]
            video_files = sorted(video_files)

            batch_videos = []
            for video_file in video_files:
                video, pad_f, pad_h, pad_w, original_shape, original_fps = preprocess_video_match(video_file, is_match=True)
                __frame_transform = transforms.Compose([
                    transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)
                ])
                video = torch.stack([__frame_transform(f) for f in video], dim=0).unsqueeze(0)
                video = video.permute(0, 2, 1, 3, 4).contiguous()
                batch_videos.append(video)

            batch_videos = torch.cat(batch_videos, dim=0).to(device=device, dtype=pipe.vae.dtype)

            # Add fixed noise to latents
            image_noise_sigma = torch.normal(
                mean=-3.0, std=0.5,
                size=(batch_videos.size(0),),
                device=device, dtype=pipe.vae.dtype
            )
            image_noise_sigma = torch.exp(image_noise_sigma)
            noisy_images = batch_videos + torch.randn_like(batch_videos) * image_noise_sigma[:, None, None, None, None]

            # Encode the video
            latent_dist = pipe.vae.encode(noisy_images).latent_dist
            latent = latent_dist.sample() * pipe.vae.config.scaling_factor
            latent = latent.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
            latent = latent.to(memory_format=torch.contiguous_format, dtype=torch.bfloat16)

            # 1. save noised latent to latent_dict
            noised_latent_key = f"{processing_dict_keys[0]}_noised"
            if noised_latent_key in latent_dict:
                raise ValueError(f"{noised_latent_key} already exists.")
            latent_dict[noised_latent_key] = latent
            save_path = os.path.join(save_dir, f"{name}.pth")
            torch.save(latent_dict, save_path)
            print(f"Saved step {name} data to {save_path}")

            # Decode the latents
            latent_generate = latent
            decoded_video = pipe.decode_latents(latent_generate)
            decoded_video_pil = pipe.video_processor.postprocess_video(video=decoded_video, output_type="pil")

            # 2. save decoded videos
            for b, frames in enumerate(decoded_video_pil):
                video_name = f"{processing_dict_keys[0]}_{b}.mp4"
                video_path = os.path.join(mp4_save_dir, name, video_name)
                os.makedirs(os.path.dirname(video_path), exist_ok=True)
                export_to_video(frames, video_path, fps=25)
                print(f"Saved video: {video_path}")

            # 显存清理
            del batch_videos, noisy_images, latent_dist, latent, decoded_video, decoded_video_pil
            torch.cuda.empty_cache()
            gc.collect()

def add_noise_to_latents_multigpu():
    load_dir = '/opt/data/private/yyx/code/DiffusionAsShader/exp/save_data_pt_v1'
    mp4_dir = '/opt/data/private/yyx/code/DiffusionAsShader/exp/save_data_pt_v1_decoded_videos'

    all_pth_paths = []
    for root, _, files in os.walk(load_dir):
        for file in files:
            if file.endswith('.pth'):
                all_pth_paths.append(os.path.join(root, file))
    all_pth_paths.sort()

    all_mp4_paths = [p.replace(load_dir, mp4_dir).replace('.pth', '') for p in all_pth_paths]
    assert len(all_pth_paths) == len(all_mp4_paths)

    # Split data in 2 chunks
    mid = len(all_pth_paths) // 2
    partition_0 = (all_pth_paths[:mid], all_mp4_paths[:mid])
    partition_1 = (all_pth_paths[mid:], all_mp4_paths[mid:])

    # 启动两个进程，分别绑定 GPU 0 和 GPU 1
    mp.set_start_method('spawn', force=True)
    procs = []
    for gpu_id, (pths, mp4s) in enumerate([partition_0, partition_1]):
        p = mp.Process(target=process_partition, args=(gpu_id, pths, mp4s))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

def random_temporal_crop(x, num_frames=5, min_stride=1, max_stride=1):
    """
    从 x 的时间维度中抽取 5 帧：
    - 起始帧随机
    - 间隔 stride 随机 ∈ [min_stride, max_stride]
    - 固定顺序，等间隔采样
    - 不越界
    """
    B, C, F, H, W = x.shape  # F = 49
    valid_strides = []

    # Step 1: 找出所有合法的 stride（能抽出 5 帧且不越界）
    for stride in range(min_stride, max_stride + 1):
        required_min_frames = (num_frames - 1) * stride + 1  # 最后一帧索引 = 4*stride
        if required_min_frames <= F:  # 即 4*stride <= 48
            valid_strides.append(stride)

    if not valid_strides:
        raise ValueError(f"无法在 {F} 帧内以 stride {min_stride}~{max_stride} 抽取 {num_frames} 帧")

    # Step 2: 随机选择一个合法的 stride
    stride = random.choice(valid_strides)

    # Step 3: 计算最大允许的起始帧
    max_start = F - 1 - (num_frames - 1) * stride  # 48 - 4*stride
    start = random.randint(0, max_start)

    # Step 4: 生成等间隔索引
    indices = [start + i * stride for i in range(num_frames)]

    # Step 5: 索引张量
    x_sampled = x[:, :, indices, :, :]  # [B, C, 5, H, W]

    return x_sampled, indices, stride

def single_tensor_handel(batch_videos, height=320, width=480, num_frames=25):
    ### resize batch_videos to 320x640
    # Step 1: 使用 rearrange 将 batch 和 frames 合并到 batch 维度，保留 (C, H, W)
    # 每一帧作为一个 2D 图像处理
    if 1:
        # resize 
        tmp_B, tmp_C, tmp_F, tmp_H, tmp_W = batch_videos.shape
        batch_videos_flat = rearrange(batch_videos, 'b c f h w -> (b f) c h w')  # [4*49, 3, 480, 720] = [196, 3, 480, 720]
        # Step 2: 使用 interpolate 缩放空间尺寸 (H, W)
        batch_videos_resized = F.interpolate(
            batch_videos_flat,
            size=(height, width),
            mode='bilinear',
            align_corners=False,
            antialias=True  # 推荐开启（尤其缩小图像时）
        )
        # Step 3: 恢复原始结构：先拆分 batch 和 frames
        batch_videos_out = rearrange(batch_videos_resized, '(b f) c new_h new_w -> b c f new_h new_w', b=tmp_B, f=tmp_F, c=tmp_C, new_h=height, new_w=width)

        final_batch_videos_out, final_indices, final_stride = random_temporal_crop(batch_videos_out, num_frames)
    else:
        # center crop
        batch_videos_out = transforms.functional.center_crop(batch_videos, output_size=(height, width))

        final_batch_videos_out, final_indices, final_stride = random_temporal_crop(batch_videos_out, num_frames)

    # print(batch_videos_out.shape, final_batch_videos_out.shape, final_indices);assert 0  
    # torch.Size([4, 3, 49, 320, 640]) torch.Size([4, 3, 25, 320, 640]) [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
    return final_batch_videos_out, final_indices, final_stride

def process_resize2DOVE(device_id, pth_paths_partition, mp4_paths_partition, batch_vae=True):
    processing_dict_keys = ['tracking_maps', 'video_latents']
    vae_path = "checkpoints/CogVideoX1.5-5B-I2V"
    save_dir = '/opt/data/private/yyx/code/DiffusionAsShader/exp/save_data_pt_v1_stage2'
    mp4_save_dir = '/opt/data/private/yyx/code/DiffusionAsShader/exp/save_data_pt_v1_stage2_decoded_videos'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(mp4_save_dir, exist_ok=True)

    with torch.no_grad():
        device = torch.device(f"cuda:{device_id}")
        vae = AutoencoderKLCogVideoX.from_pretrained(vae_path, subfolder="vae", torch_dtype=torch.bfloat16)
        pipe = CogVideoXImageToVideoPipeline(
            vae=vae,
            text_encoder=None,
            tokenizer=None,
            transformer=None,
            scheduler=None,
        )
        pipe.to(device)
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
        pipe.vae.eval()

        for path, mp4_dir in tqdm(zip(pth_paths_partition, mp4_paths_partition), desc=f"GPU-{device_id}", total=len(pth_paths_partition)):
            name = os.path.splitext(os.path.basename(path))[0]
            latent_dict = torch.load(path, map_location=device)

            if 0:
                for key in latent_dict:
                    if key not in processing_dict_keys:
                        continue

                    latents = latent_dict[key]  # [B, T, C, H, W]

                    # Decode the latents
                    latent_generate = latents
                    decoded_video = pipe.decode_latents(latent_generate)
                    decoded_video_pil = pipe.video_processor.postprocess_video(video=decoded_video, output_type="pil")

                    # 2. save decoded videos
                    for b, frames in enumerate(decoded_video_pil):
                        video_name = f"{key}_{b}.mp4"
                        video_path = os.path.join(mp4_save_dir, name, video_name)
                        os.makedirs(os.path.dirname(video_path), exist_ok=True)
                        export_to_video(frames, video_path, fps=25)
                        print(f"Saved video: {video_path}")
                assert 0

            for key in latent_dict:
                if key not in processing_dict_keys:
                    continue

                latents = latent_dict[key]  # [B, T, C, H, W]

                if 1:
                    ### input is latent
                    with torch.no_grad():
                        video = pipe.decode_latents(latents)
                        batch_videos = pipe.video_processor.postprocess_video(video=video, output_type="pt")
                        batch_videos = (batch_videos - 0.5) * 2.0
                        batch_videos = rearrange(batch_videos, 'b f c h w -> b c f h w')
                        # print(batch_videos.shape, torch.min(batch_videos), torch.max(batch_videos));assert 0
                        # torch.Size([4, 3, 49, 480, 720]) tensor(-1., device='cuda:1', dtype=torch.bfloat16) tensor(1., device='cuda:1', dtype=torch.bfloat16)
                else:
                    ### input is mp4
                    video_files = [os.path.join(mp4_dir, f) for f in os.listdir(mp4_dir)
                                if f.endswith('.mp4') and not f.startswith('.')]
                    video_files = sorted(video_files)

                    batch_videos = []
                    for video_file in video_files:
                        video, pad_f, pad_h, pad_w, original_shape, original_fps = preprocess_video_match(video_file, is_match=True)
                        __frame_transform = transforms.Compose([
                            transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)
                        ])
                        video = torch.stack([__frame_transform(f) for f in video], dim=0).unsqueeze(0)
                        video = video.permute(0, 2, 1, 3, 4).contiguous()
                        batch_videos.append(video)

                    batch_videos = torch.cat(batch_videos, dim=0).to(device=device, dtype=pipe.vae.dtype)

                ### Add fixed noise to latents
                # image_noise_sigma = torch.normal(
                #     mean=-3.0, std=0.5,
                #     size=(batch_videos.size(0),),
                #     device=device, dtype=pipe.vae.dtype
                # )
                # image_noise_sigma = torch.exp(image_noise_sigma)
                # noisy_images = batch_videos + torch.randn_like(batch_videos) * image_noise_sigma[:, None, None, None, None]


                ### get T H W = 25, 320, 640
                # print(batch_videos.shape, torch.min(batch_videos), torch.max(batch_videos));assert 0 
                # torch.Size([4, 3, 49, 480, 720]) tensor(-1., device='cuda:1', dtype=torch.bfloat16) tensor(1., device='cuda:1', dtype=torch.bfloat16)
                batch_videos_processed, batch_videos_processed_indices, batch_videos_processed_stride = single_tensor_handel(batch_videos)
                print(batch_videos_processed_indices, batch_videos_processed_stride)
                # print(batch_videos_processed.shape);assert 0 # torch.Size([4, 3, 25, 320, 640])


                # Encode the video
                if batch_vae:
                    latent_dist = pipe.vae.encode(batch_videos_processed).latent_dist
                    latent = latent_dist.sample() * pipe.vae.config.scaling_factor
                    latent = latent.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
                    latent = latent.to(memory_format=torch.contiguous_format, dtype=torch.bfloat16)
                else:
                    # encoding each frame independently
                    B, C, F, H, W = batch_videos_processed.shape
                    batch_videos_reshaped = rearrange(batch_videos_processed, 'b c f h w -> (b f) c h w')  # [B*F, C, H, W]
                    batch_videos_reshaped = batch_videos_reshaped.unsqueeze(2)  # [B*F, 1, H, W], vae input
                    latent_dist = pipe.vae.encode(batch_videos_reshaped).latent_dist
                    latent = latent_dist.sample() * pipe.vae.config.scaling_factor  # [B*F, C, H', W']
                    latent = latent.squeeze(2)  # [B*F, C, H', W']
                    _, latent_C, latent_H, latent_W = latent.shape
                    latent = rearrange(latent, '(b f) c h w -> b f c h w', b=B, f=F, c=latent_C, h=latent_H, w=latent_W)
                    latent = latent.to(memory_format=torch.contiguous_format, dtype=torch.bfloat16)
                # print(batch_videos_processed.shape, latent.shape, torch.min(latent), torch.max(latent));assert 0

                # 1. save latent to latent_dict
                latent_dict[key] = latent

                if key == 'video_latents':
                    # 同时保存 decoded video
                    B, T, C, H, W = latent.shape
                    # latent_reshaped = rearrange(latent, 'b f c h w -> (b f) c h w')  # [B*T, C, H, W]
                    # latent_reshaped = latent_reshaped.unsqueeze(2)  # [B*T, 1, H, W]
                    # latent_reshaped = latent_reshaped / pipe.vae.config.scaling_factor
                    # decoded_video = pipe.vae.decode(latent_reshaped).sample
                    # decoded_video = decoded_video.squeeze(2)  # [B*T, C, H, W]
                    
                    latent_reshaped = rearrange(latent, 'b f c h w -> (b f) c h w')  # [B*T, C, H, W]
                    latent_reshaped = latent_reshaped.unsqueeze(1)  # [B*T, 1, H, W]
                    decoded_video = pipe.decode_latents(latent_reshaped)
                    decoded_video = decoded_video.squeeze(2)  # [B*T, C, H, W]
                    _, decoded_C, decoded_H, decoded_W = decoded_video.shape
                    decoded_video = rearrange(decoded_video, '(b f) c h w -> b c f h w', b=B, f=T, c=decoded_C, h=decoded_H, w=decoded_W)

            ### save gt pixels to latent_dict
            latent_dict['video_latents_decoded'] = decoded_video

            save_path = os.path.join(save_dir, f"{name}.pth")
            torch.save(latent_dict, save_path)
            print(f"Saved step {name} data to {save_path}")

            for key in latent_dict:
                if key not in processing_dict_keys:
                    continue

                latents = latent_dict[key]  # [B, T, C, H, W]

                # Decode the latents
                if batch_vae:
                    latent_generate = latents
                    decoded_video = pipe.decode_latents(latent_generate)
                    decoded_video_pil = pipe.video_processor.postprocess_video(video=decoded_video, output_type="pil")
                else:
                    B, T, C, H, W = latents.shape
                    latents_reshaped = rearrange(latents, 'b f c h w -> (b f) c h w')  # [B*T, C, H, W]
                    latents_reshaped = latents_reshaped.unsqueeze(1)  # [B*T, 1, H, W]
                    decoded_video = pipe.decode_latents(latents_reshaped)
                    decoded_video = decoded_video.squeeze(2)  # [B*T, C, H, W]
                    _, decoded_C, decoded_H, decoded_W = decoded_video.shape
                    decoded_video = rearrange(decoded_video, '(b f) c h w -> b c f h w', b=B, f=T, c=decoded_C, h=decoded_H, w=decoded_W)
                    decoded_video_pil = pipe.video_processor.postprocess_video(video=decoded_video, output_type="pil")

                # 2. save decoded videos
                for b, frames in enumerate(decoded_video_pil):
                    video_name = f"{key}_{b}.mp4"
                    video_path = os.path.join(mp4_save_dir, name, video_name)
                    os.makedirs(os.path.dirname(video_path), exist_ok=True)
                    export_to_video(frames, video_path, fps=25)
                    print(f"Saved video: {video_path}")

            # 显存清理
            del batch_videos, latent_dist, latent, decoded_video, decoded_video_pil
            torch.cuda.empty_cache()
            gc.collect()


def resize2DOVE_multigpu():
    load_dir = '/opt/data/private/yyx/code/DiffusionAsShader/exp/save_data_pt_v1'
    mp4_dir = '/opt/data/private/yyx/code/DiffusionAsShader/exp/save_data_pt_v1_decoded_videos'

    all_pth_paths = []
    for root, _, files in os.walk(load_dir):
        for file in files:
            if file.endswith('.pth'):
                all_pth_paths.append(os.path.join(root, file))
    all_pth_paths.sort()

    all_mp4_paths = [p.replace(load_dir, mp4_dir).replace('.pth', '') for p in all_pth_paths]
    assert len(all_pth_paths) == len(all_mp4_paths)

    # Split data in 2 chunks
    mid = len(all_pth_paths) // 2
    partition_0 = (all_pth_paths[:mid], all_mp4_paths[:mid])
    partition_1 = (all_pth_paths[mid:], all_mp4_paths[mid:])

    # 启动两个进程，分别绑定 GPU 0 和 GPU 1
    mp.set_start_method('spawn', force=True)
    procs = []
    for gpu_id, (pths, mp4s) in enumerate([partition_0, partition_1]):
        p = mp.Process(target=process_resize2DOVE, args=(gpu_id, pths, mp4s))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

# ====== 主函数：分配任务并启动多进程 ======
if __name__ == "__main__":
    # calculate_clip_iqa() # v1 -> v1_decoded_videos
    # add_noise_to_latents()  # v1 -> v2
    # add_noise_to_latents_multigpu()  # v1 -> v2, 多 GPU 版本
    resize2DOVE_multigpu()  # v1 -> v1_stage2, 多 GPU 版本




# import os
# import torch
# import argparse
# from tqdm import tqdm
# from torchvision import transforms
# from diffusers import AutoencoderKLCogVideoX, CogVideoXImageToVideoPipeline
# from diffusers.utils import export_to_video
# import pyiqa
# from PIL import Image
# import imageio

# # === 命令行参数 ===
# parser = argparse.ArgumentParser()
# parser.add_argument('--local_rank', type=int, default=0)   # ⭐ 当前 GPU 编号
# parser.add_argument('--world_size', type=int, default=1)   # ⭐ 总 GPU 数
# args = parser.parse_args()

# # === 设备分配 ===
# device = torch.device(f'cuda:{args.local_rank}' if torch.cuda.is_available() else 'cpu')
# torch.cuda.set_device(device)

# # === 加载 VAE pipeline ===
# pretrained_congvideox_5b_i2v_path = "checkpoints/CogVideoX1.5-5B-I2V"
# vae = AutoencoderKLCogVideoX.from_pretrained(pretrained_congvideox_5b_i2v_path, subfolder="vae", torch_dtype=torch.bfloat16)
# pipe = CogVideoXImageToVideoPipeline(
#     vae=vae,
#     text_encoder=None,
#     tokenizer=None,
#     transformer=None,
#     scheduler=None,
# )
# pipe.vae.enable_tiling()
# pipe.vae.enable_slicing()
# pipe.to(device)
# pipe.vae.eval()

# # === CLIP-IQA 初始化 ===
# clipiqa_metric = pyiqa.create_metric('clipiqa', device=device)
# to_tensor = transforms.ToTensor()

# # === 配置路径 ===
# load_dir = '/opt/data/private/yyx/code/DiffusionAsShader/exp/save_data_pt_v1'
# save_dir = '/opt/data/private/yyx/code/DiffusionAsShader/exp/save_data_pt_v1_decoded_videos'
# processing_dict_keys = ['tracking_maps']

# os.makedirs(save_dir, exist_ok=True)

# # === 获取所有 .pth 文件 ===
# all_pth_paths = []
# for root, _, files in os.walk(load_dir):
#     for file in files:
#         if file.endswith('.pth'):
#             all_pth_paths.append(os.path.join(root, file))
# all_pth_paths.sort()

# # === 分配当前 GPU 负责的部分 === ⭐
# pth_paths = all_pth_paths[args.local_rank::args.world_size]
# print(f"[GPU {args.local_rank}] Processing {len(pth_paths)} files out of {len(all_pth_paths)}.")

# # === 解码主循环 ===
# for path in tqdm(pth_paths, desc=f"GPU {args.local_rank}"):
#     name = os.path.splitext(os.path.basename(path))[0]
#     latent_dict = torch.load(path, map_location=device)
#     print(f"Processing {name}... keys: {list(latent_dict.keys())}")

#     for key in latent_dict.keys():
#         if key not in processing_dict_keys:
#             continue

#         latents = latent_dict[key]  # shape: [B, T, C, H, W]
#         print(f"  Key: {key}, Shape: {latents.shape}, Min: {latents.min():.4f}, Max: {latents.max():.4f}")

#         with torch.no_grad():
#             video = pipe.decode_latents(latents)
#             videos = pipe.video_processor.postprocess_video(video=video, output_type="pil")

#             for b, frames in enumerate(videos):
#                 video_name = f"{key}_{b}.mp4"
#                 video_path = os.path.join(save_dir, name, video_name)
#                 os.makedirs(os.path.dirname(video_path), exist_ok=True)

#                 # === CLIP-IQA ===
#                 scores = []
#                 for frame in frames:
#                     frame_tensor = to_tensor(frame).unsqueeze(0).to(device)
#                     with torch.no_grad():
#                         score = clipiqa_metric(frame_tensor).item()
#                     scores.append(score)
                
#                 avg_score = sum(scores) / len(scores)
#                 print(f"    → CLIP-IQA score: {avg_score:.4f}")

#                 export_to_video(frames, video_path, fps=25)
#                 print(f"Saved video: {video_path}")


# import os
# import torch
# from tqdm import tqdm
# from torchvision.utils import save_image
# import imageio

# from diffusers import (
#     CogVideoXDPMScheduler,
#     CogVideoXPipeline,
# )

# from diffusers import AutoencoderKLCogVideoX, CogVideoXDDIMScheduler, CogVideoXTransformer3DModel, CogVideoXImageToVideoPipeline
# from diffusers.utils import export_to_video

# import pyiqa
# from torchvision import transforms

# # load pipeline
# pretrained_congvideox_5b_i2v_path = "checkpoints/CogVideoX1.5-5B-I2V"
# vae = AutoencoderKLCogVideoX.from_pretrained(pretrained_congvideox_5b_i2v_path, subfolder="vae", torch_dtype=torch.bfloat16)
# pipe = CogVideoXImageToVideoPipeline(
#     vae=vae,
#     text_encoder=None,
#     tokenizer=None,
#     transformer=None,
#     scheduler=None,
# )
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# pipe.vae.enable_tiling()
# pipe.vae.enable_slicing()
# pipe.to(device)  # ✅ 现在可以调用
# pipe.vae.eval()

# # 初始化 clipiqa 指标
# clipiqa_metric = pyiqa.create_metric('clipiqa', device=device)
# to_tensor = transforms.ToTensor()

# # === 配置 ===
# load_dir = '/opt/data/private/yyx/code/DiffusionAsShader/exp/save_data_pt_v1'  # 替换为你实际的路径
# save_dir = '/opt/data/private/yyx/code/DiffusionAsShader/exp/save_data_pt_v1_decoded_videos'

# processing_dict_keys = ['tracking_maps']  # 替换为你实际的键名

# os.makedirs(save_dir, exist_ok=True)
# # === 查找所有.pth文件 ===
# pth_paths = []
# for root, _, files in os.walk(load_dir):
#     for file in files:
#         if file.endswith('.pth'):
#             pth_paths.append(os.path.join(root, file))

# pth_paths.sort()  # 按照文件名排序
# print(f'Found {len(pth_paths)} .pth files.')

# # === 逐个解码 ===
# for path in tqdm(pth_paths):
#     name = os.path.splitext(os.path.basename(path))[0]
#     latent_dict = torch.load(path, map_location=device)  # latent shape: [T, C, H, W]
#     print(f"Processing {name}... keys in latent_dict: {list(latent_dict.keys())}")
#     for key in latent_dict.keys():
#         if key not in processing_dict_keys:
#             # print(f"Skipping key: {key}")
#             continue
#         print(f"  Key: {key}, Shape: {latent_dict[key].shape}, torch.Min: {torch.min(latent_dict[key])}, torch.Max: {torch.max(latent_dict[key])}")

#         with torch.no_grad():
#             latents = latent_dict[key]  # shape: [B, T, C, H, W]
#             video = pipe.decode_latents(latents)
#             videos = pipe.video_processor.postprocess_video(video=video, output_type="pil")

#             for b, frames in enumerate(videos):
#                 video_name = f"{key}_{b}.mp4"
#                 video_path = os.path.join(save_dir, name, video_name)
#                 os.makedirs(os.path.dirname(video_path), exist_ok=True)

#                 # === 计算 CLIP-IQA: 每帧计算然后平均 ===
#                 scores = []
#                 for frame in frames:
#                     frame_tensor = to_tensor(frame).unsqueeze(0).to(device)  # [1, 3, H, W]
#                     with torch.no_grad():
#                         score = clipiqa_metric(frame_tensor).item()
#                     scores.append(score)
                
#                 avg_score = sum(scores) / len(scores)
#                 print(f"    → CLIP-IQA score: {avg_score:.4f}")

#                 export_to_video(frames, video_path, fps=25)

#                 print(f"Saved video: {video_path}")



