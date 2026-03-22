import os
import numpy as np
from PIL import Image
from decord import VideoReader
from decord.bridge import set_bridge
from diffusers.utils import export_to_video

set_bridge("native")

# ---------- 基础判断函数 ----------

def is_image_file(filename):
    return filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))

def is_video_file(filename):
    return filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))

# ---------- 图像保存 ----------

def save_image(np_img, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img = np.clip(np_img * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(save_path)

# ---------- 图像处理模式 ----------

def adaptive_crop(img: Image.Image, target_w: int = 720, target_h: int = 480) -> np.ndarray:
    """自适应中心裁剪图像到指定宽高比例"""
    img = img.resize((img.width * 4, img.height * 4), Image.BICUBIC)
    input_ratio = img.width / img.height
    target_ratio = target_w / target_h

    if input_ratio > target_ratio:
        new_h = target_h
        new_w = int(target_h * input_ratio)
    else:
        new_w = target_w
        new_h = int(target_w / input_ratio)

    img = img.resize((new_w, new_h), Image.BICUBIC)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    img = img.crop((left, top, left + target_w, top + target_h))
    return np.array(img).astype(np.float32) / 255.0

def down_up_sample(img: Image.Image, scale: int = 4) -> np.ndarray:
    """图像先下采样再上采样"""
    w, h = img.size
    img_down = img.resize((w // scale, h // scale), Image.BICUBIC)
    img_up = img_down.resize((w, h), Image.BICUBIC)
    return np.array(img_up).astype(np.float32) / 255.0

def add_gaussian_noise(img_np: np.ndarray, sigma: float = 0.05) -> np.ndarray:
    """给图像添加高斯噪声，img_np 为 float32 格式 [0,1]"""
    noise = np.random.normal(0, sigma, img_np.shape).astype(np.float32)
    return np.clip(img_np + noise, 0.0, 1.0)

# ---------- 视频处理 ----------

def process_video(input_path: str, output_path: str, mode: str = "adaptive_crop", scale: int = 4, noise_sigma: float = 0.05):
    vr = VideoReader(input_path)
    fps = int(vr.get_avg_fps())
    frames = []

    for frame in vr:
        img = Image.fromarray(frame.asnumpy())
        if mode == "adaptive_crop":
            img_np = adaptive_crop(img)
        elif mode == "down_up_sample":
            img_np = down_up_sample(img, scale)
        elif mode == "add_noise":
            img_np = np.array(img).astype(np.float32) / 255.0
            img_np = add_gaussian_noise(img_np, sigma=noise_sigma)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        frames.append(img_np)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    export_to_video(np.stack(frames), output_video_path=output_path, fps=fps)
    print(f"🎬 Saved video: {output_path}")

# ---------- 图像处理 ----------

def process_image(input_path: str, output_path: str, mode: str = "adaptive_crop", scale: int = 4, noise_sigma: float = 0.05):
    img = Image.open(input_path).convert("RGB")
    if mode == "adaptive_crop":
        img_np = adaptive_crop(img)
    elif mode == "down_up_sample":
        img_np = down_up_sample(img, scale)
    elif mode == "add_noise":
        img_np = np.array(img).astype(np.float32) / 255.0
        img_np = add_gaussian_noise(img_np, sigma=noise_sigma)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    save_image(img_np, output_path)
    print(f"🖼️ Saved image: {output_path}")

# ---------- 输入路径自动分派 ----------

def process_input(input_path: str, output_root: str, mode: str = "adaptive_crop", scale: int = 4, noise_sigma: float = 0.05):
    os.makedirs(output_root, exist_ok=True)

    if os.path.isdir(input_path):
        all_files = sorted([
            os.path.join(dp, f) for dp, _, fn in os.walk(input_path)
            for f in fn if is_image_file(f) or is_video_file(f)
        ])
        for file in all_files:
            rel_path = os.path.relpath(file, input_path)
            if is_image_file(file):
                out_path = os.path.join(output_root, os.path.splitext(rel_path)[0] + ".png")
                process_image(file, out_path, mode, scale, noise_sigma)
            elif is_video_file(file):
                out_path = os.path.join(output_root, rel_path)
                process_video(file, out_path, mode, scale, noise_sigma)
    else:
        # 单文件处理
        if is_image_file(input_path):
            filename = os.path.splitext(os.path.basename(input_path))[0] + ".png"
            process_image(input_path, os.path.join(output_root, filename), mode, scale, noise_sigma)
        elif is_video_file(input_path):
            process_video(input_path, os.path.join(output_root, os.path.basename(input_path)), mode, scale, noise_sigma)
        else:
            print(f"❌ Unsupported file: {input_path}")

# ---------- 主函数入口 ----------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Media Preprocessing Tool")
    parser.add_argument("--input", type=str, required=True, help="Input file or folder")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--mode", type=str, choices=["adaptive_crop", "down_up_sample", "add_noise"], default="adaptive_crop")
    parser.add_argument("--scale", type=int, default=4, help="Scale factor for down-up sampling")
    parser.add_argument("--noise_sigma", type=float, default=0.05, help="Sigma of Gaussian noise to add in 'add_noise' mode")
    args = parser.parse_args()

    process_input(args.input, args.output, args.mode, args.scale, args.noise_sigma)





# import os
# import decord
# import numpy as np
# from PIL import Image
# from decord import VideoReader
# from diffusers.utils import export_to_video

# decord.bridge.set_bridge('native')


# def process_frame_pil(img: Image.Image, target_w: int = 720, target_h: int = 480) -> np.ndarray:
#     img = img.resize((img.width * 4, img.height * 4), Image.BICUBIC)
#     input_ratio = img.width / img.height
#     target_ratio = target_w / target_h

#     if input_ratio > target_ratio:
#         new_h = target_h
#         new_w = int(target_h * input_ratio)
#     else:
#         new_w = target_w
#         new_h = int(target_w / input_ratio)

#     img = img.resize((new_w, new_h), Image.BICUBIC)
#     left = (new_w - target_w) // 2
#     top = (new_h - target_h) // 2
#     img = img.crop((left, top, left + target_w, top + target_h))

#     img_array = np.array(img).astype(np.float32) / 255.0
#     return img_array


# def process_video_adaptive_crop(input_path: str, output_path: str):
#     vr = VideoReader(input_path)
#     fps = vr.get_avg_fps()

#     processed_frames = []
#     for i, frame in enumerate(vr):
#         img = Image.fromarray(frame.asnumpy())
#         img_array = process_frame_pil(img)
#         processed_frames.append(img_array)

#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     export_to_video(np.stack(processed_frames), output_video_path=output_path, fps=int(fps))
#     print(f"✅ Cropped video saved to: {output_path}")


# def process_image_adaptive_crop(input_path: str, output_path: str):
#     img = Image.open(input_path).convert("RGB")
#     img_array = process_frame_pil(img)
#     img_out = (img_array * 255).clip(0, 255).astype(np.uint8)
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     Image.fromarray(img_out).save(output_path)
#     print(f"🖼️ Cropped image saved to: {output_path}")


# def process_video_down_up_sample(input_path: str, output_path: str, scale: int = 4):
#     vr = VideoReader(input_path)
#     fps = vr.get_avg_fps()
#     processed_frames = []

#     for frame in vr:
#         img_np = frame.asnumpy()
#         img = Image.fromarray(img_np)
#         orig_w, orig_h = img.width, img.height
#         img_down = img.resize((orig_w // scale, orig_h // scale), Image.BICUBIC)
#         img_up = img_down.resize((orig_w, orig_h), Image.BICUBIC)
#         img_array = np.array(img_up).astype(np.float32) / 255.0
#         processed_frames.append(img_array)

#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     export_to_video(np.stack(processed_frames), output_video_path=output_path, fps=int(fps))
#     print(f"📉🔼 Down-up sampled video saved to: {output_path}")


# def process_image_down_up_sample(input_path: str, output_path: str):
#     img = Image.open(input_path).convert("RGB")
#     orig_w, orig_h = img.width, img.height
#     img_down = img.resize((orig_w // 4, orig_h // 4), Image.BICUBIC)
#     img_up = img_down.resize((orig_w, orig_h), Image.BICUBIC)
#     img_out = np.array(img_up).astype(np.uint8)
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     Image.fromarray(img_out).save(output_path)
#     print(f"🖼️ Down-up sampled image saved to: {output_path}")


# def is_image_file(filename):
#     return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))


# def is_video_file(filename):
#     return filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))


# def process_input(input_path: str, output_root: str, mode: str = "adaptive_crop"):
#     assert mode in ["adaptive_crop", "down_up_sample"], f"Unsupported mode: {mode}"

#     if os.path.isdir(input_path):
#         files = sorted([
#             f for f in os.listdir(input_path)
#             if is_image_file(f) or is_video_file(f)
#         ])

#         if all(is_video_file(f) for f in files):
#             # 🟡 文件夹中全部是视频
#             for file in files:
#                 input_video_path = os.path.join(input_path, file)
#                 output_video_path = os.path.join(output_root, file)

#                 if mode == "adaptive_crop":
#                     process_video_adaptive_crop(input_video_path, output_video_path)
#                 elif mode == "down_up_sample":
#                     process_video_down_up_sample(input_video_path, output_video_path, scale=2)

#         else:
#             # 🟢 混合图片或子目录情况
#             for root, _, files in os.walk(input_path):
#                 for file in sorted(files):
#                     if is_image_file(file):
#                         input_img_path = os.path.join(root, file)
#                         rel_path = os.path.relpath(input_img_path, input_path)
#                         output_img_path = os.path.join(output_root, os.path.splitext(rel_path)[0] + ".png")

#                         if mode == "adaptive_crop":
#                             process_image_adaptive_crop(input_img_path, output_img_path)
#                         elif mode == "down_up_sample":
#                             process_image_down_up_sample(input_img_path, output_img_path)

#     elif is_video_file(input_path):
#         filename = os.path.basename(input_path)
#         output_path = os.path.join(output_root, filename)

#         if mode == "adaptive_crop":
#             process_video_adaptive_crop(input_path, output_path)
#         elif mode == "down_up_sample":
#             process_video_down_up_sample(input_path, output_path)

#     elif is_image_file(input_path):
#         filename = os.path.basename(input_path)
#         output_path = os.path.join(output_root, os.path.splitext(filename)[0] + ".png")

#         if mode == "adaptive_crop":
#             process_image_adaptive_crop(input_path, output_path)
#         elif mode == "down_up_sample":
#             process_image_down_up_sample(input_path, output_path)

#     else:
#         print(f"❌ Unsupported file type: {input_path}")


# # ====== 主函数入口 ======
# if __name__ == "__main__":
#     # 例子 1：中心裁剪
#     # # 示例用法
#     # # video input 
#     # # input_dir = "/opt/data/private/yyx/data/VSR_benchmark/input_video/VideoLQ"
#     # # output_dir = "/opt/data/private/yyx/data/VSR_benchmark/input_video_480p/VideoLQ_center_crop_480p"

#     # # image input
#     # input_dir = "/opt/data/private/yyx/data/VSR_benchmark/results/SUPIR"
#     # output_dir = "/opt/data/private/yyx/data/VSR_benchmark/results/SUPIR_center_crop_480p"
#     # process_input("/path/to/input", "/path/to/output/crop_480p", mode="adaptive_crop")

#     # 例子 2：下采样再上采样

#     # process_input("/opt/data/private/yyx/data/VSR_benchmark/input_video/VideoLQ", "/opt/data/private/yyx/data/VSR_benchmark/input_video/VideoLQ_downupx4", mode="down_up_sample")
#     process_input("/opt/data/private/yyx/data/VSR_benchmark/input_video/VideoLQ", "/opt/data/private/yyx/data/VSR_benchmark/input_video/VideoLQ_downupx2", mode="down_up_sample")

