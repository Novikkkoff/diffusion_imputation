import argparse
import json
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import soundfile as sf
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from dataset_packetloss import get_packet_loss_dataloaders
from main_model import CSDI_Physio
from utils import recon_ri_to_wav


DEFAULT_CLEAN_DIR = "/home/ubuntu/projects/ia.novikov/CSDIweather/CSDI/train/clean_signals"
DEFAULT_LOSSY_DIR = "/home/ubuntu/projects/ia.novikov/CSDIweather/CSDI/train/lossy_signals"
DEFAULT_TRACES_DIR = "/home/ubuntu/projects/ia.novikov/CSDIweather/CSDI/train/traces"

SR = 16000
N_FFT = 512
HOP_LENGTH = 256
WIN_LENGTH = 512
L_FRAMES = 625
TARGET_DIM = N_FFT // 2 + 1


def parse_steps(text, max_step):
    if text.strip().lower() == "auto":
        raw = [max_step, int(max_step * 0.75), int(max_step * 0.5), int(max_step * 0.25), 10, 0]
    else:
        raw = [int(x.strip()) for x in text.split(",") if x.strip()]
    return sorted({min(max(s, 0), max_step) for s in raw}, reverse=True)


def ri_to_db(ri, eps=1e-6):
    mag = torch.sqrt(ri[0] ** 2 + ri[1] ** 2 + eps)
    return 20.0 * torch.log10(mag + eps).detach().cpu().numpy()


def save_spec(path, ri, title, vmin=None, vmax=None, mask=None):
    spec = ri_to_db(ri)
    if vmin is None:
        vmin = float(np.percentile(spec, 1))
    if vmax is None:
        vmax = float(np.percentile(spec, 99))

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    img = ax.imshow(
        spec,
        aspect="auto",
        origin="lower",
        extent=[0, spec.shape[1] * HOP_LENGTH / SR, 0, SR / 2000],
        vmin=vmin,
        vmax=vmax,
        cmap="magma",
        interpolation="nearest",
    )
    if mask is not None:
        hole_map = (1.0 - mask[0]).detach().cpu().numpy()
        overlay = np.zeros((*hole_map.shape, 4), dtype=np.float32)
        overlay[..., 2] = 1.0
        overlay[..., 3] = hole_map * 0.35
        ax.imshow(
            overlay,
            aspect="auto",
            origin="lower",
            extent=[0, spec.shape[1] * HOP_LENGTH / SR, 0, SR / 2000],
            interpolation="nearest",
        )
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (kHz)")
    fig.colorbar(img, ax=ax, pad=0.01, format="%.0f dB")
    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def save_triplet(path, clean_ri, lossy_ri, recon_ri, mask, title):
    clean_db = ri_to_db(clean_ri)
    lossy_db = ri_to_db(lossy_ri)
    recon_db = ri_to_db(recon_ri)
    vmin = float(np.percentile(clean_db, 1))
    vmax = float(np.percentile(clean_db, 99))
    t_max = clean_db.shape[1] * HOP_LENGTH / SR

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, sharey=True)
    fig.suptitle(title, fontsize=12)
    for ax, name, spec, show_mask in zip(
        axes,
        ["clean target", "lossy input", "current reconstruction"],
        [clean_db, lossy_db, recon_db],
        [False, True, False],
    ):
        img = ax.imshow(
            spec,
            aspect="auto",
            origin="lower",
            extent=[0, t_max, 0, SR / 2000],
            vmin=vmin,
            vmax=vmax,
            cmap="magma",
            interpolation="nearest",
        )
        if show_mask:
            hole_map = (1.0 - mask[0]).detach().cpu().numpy()
            overlay = np.zeros((*hole_map.shape, 4), dtype=np.float32)
            overlay[..., 2] = 1.0
            overlay[..., 3] = hole_map * 0.35
            ax.imshow(
                overlay,
                aspect="auto",
                origin="lower",
                extent=[0, t_max, 0, SR / 2000],
                interpolation="nearest",
            )
        ax.set_title(name)
        ax.set_ylabel("Frequency (kHz)")
        fig.colorbar(img, ax=ax, pad=0.01, format="%.0f dB")
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def save_holes_triplet(path, clean_ri, lossy_ri, recon_ri, mask, title):
    hole = 1.0 - mask
    save_triplet(
        path,
        clean_ri * hole,
        lossy_ri * hole,
        recon_ri * hole,
        mask,
        title,
    )


def save_animation(path_base, frame_items, clean_ri, lossy_ri, mask, title_prefix, fps=8):
    if not frame_items:
        return

    clean_db = ri_to_db(clean_ri)
    lossy_db = ri_to_db(lossy_ri)
    vmin = float(np.percentile(clean_db, 1))
    vmax = float(np.percentile(clean_db, 99))
    t_max = clean_db.shape[1] * HOP_LENGTH / SR
    hole_map = (1.0 - mask[0]).detach().cpu().numpy()

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, sharey=True)
    fig.suptitle(title_prefix, fontsize=12)

    specs = [clean_db, lossy_db, ri_to_db(frame_items[0][1])]
    titles = ["clean target", "lossy input", ""]
    images = []
    for ax, spec, title in zip(axes, specs, titles):
        img = ax.imshow(
            spec,
            aspect="auto",
            origin="lower",
            extent=[0, t_max, 0, SR / 2000],
            vmin=vmin,
            vmax=vmax,
            cmap="magma",
            interpolation="nearest",
            animated=True,
        )
        images.append(img)
        ax.set_title(title)
        ax.set_ylabel("Frequency (kHz)")

    overlay = np.zeros((*hole_map.shape, 4), dtype=np.float32)
    overlay[..., 2] = 1.0
    overlay[..., 3] = hole_map * 0.35
    axes[1].imshow(
        overlay,
        aspect="auto",
        origin="lower",
        extent=[0, t_max, 0, SR / 2000],
        interpolation="nearest",
    )
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()

    def update(frame_idx):
        step, ri = frame_items[frame_idx]
        images[2].set_data(ri_to_db(ri))
        axes[2].set_title(f"current reconstruction / state, t={step}")
        fig.suptitle(f"{title_prefix}  frame {frame_idx + 1}/{len(frame_items)}", fontsize=12)
        return images

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(frame_items),
        interval=1000 / max(fps, 1),
        blit=False,
    )

    mp4_path = f"{path_base}.mp4"
    gif_path = f"{path_base}.gif"
    try:
        writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
        anim.save(mp4_path, writer=writer, dpi=120)
        print(f"Saved video: {mp4_path}")
    except Exception as e:
        print(f"MP4 save failed ({e}); saving GIF instead")
        writer = animation.PillowWriter(fps=fps)
        anim.save(gif_path, writer=writer, dpi=95)
        print(f"Saved GIF: {gif_path}")
    plt.close(fig)


def save_holes_animation(path_base, frame_items, clean_ri, lossy_ri, mask, title_prefix, fps=8):
    hole = 1.0 - mask
    hole_frames = [(step, ri * hole) for step, ri in frame_items]
    save_animation(
        path_base,
        hole_frames,
        clean_ri * hole,
        lossy_ri * hole,
        mask,
        title_prefix,
        fps=fps,
    )


def load_config(path):
    candidate = os.path.join("config", path)
    if os.path.exists(candidate):
        path = candidate
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_loader(args, config):
    train_loader, valid_loader, test_loader = get_packet_loss_dataloaders(
        clean_dir=args.clean_dir,
        lossy_dir=args.lossy_dir,
        traces_dir=args.traces_dir,
        batch_size=1,
        seed=args.seed,
        n_folds=5,
        fold_idx=args.nfold,
        missing_ratio=args.missing_ratio,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        L=L_FRAMES,
        return_clean=True,
    )
    source = {"train": train_loader, "valid": valid_loader, "test": test_loader}[args.split]
    source.dataset.missing_ratio = args.missing_ratio

    stds = []
    for i, batch in enumerate(train_loader):
        stds.append(batch["clean_ri"].std().item())
        if i >= 50:
            break
    amp_norm = float(np.mean(stds))
    for loader in (train_loader, valid_loader, test_loader):
        loader.dataset.amp_norm = amp_norm

    return DataLoader(
        Subset(source.dataset, [args.index]),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    ), amp_norm


def find_local_index_by_item_id(dataset, item_id):
    item_id = os.path.splitext(os.path.basename(item_id))[0]
    for local_idx, global_idx in enumerate(dataset.use_index_list):
        if dataset.ids[int(global_idx)] == item_id:
            return local_idx
    return None


def build_loader_for_item_id(args, config):
    train_loader, valid_loader, test_loader = get_packet_loss_dataloaders(
        clean_dir=args.clean_dir,
        lossy_dir=args.lossy_dir,
        traces_dir=args.traces_dir,
        batch_size=1,
        seed=args.seed,
        n_folds=5,
        fold_idx=args.nfold,
        missing_ratio=args.missing_ratio,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        L=L_FRAMES,
        return_clean=True,
    )
    loaders = {"train": train_loader, "valid": valid_loader, "test": test_loader}
    source = loaders[args.split]
    source.dataset.missing_ratio = args.missing_ratio

    local_idx = find_local_index_by_item_id(source.dataset, args.item_id)
    if local_idx is None:
        available = []
        for name, loader in loaders.items():
            idx = find_local_index_by_item_id(loader.dataset, args.item_id)
            if idx is not None:
                available.append(f"{name}: local_index={idx}")
        hint = "; ".join(available) if available else "not found in train/valid/test for this fold/seed"
        raise ValueError(f"item_id={args.item_id!r} not found in split={args.split}. Available: {hint}")

    stds = []
    for i, batch in enumerate(train_loader):
        stds.append(batch["clean_ri"].std().item())
        if i >= 50:
            break
    amp_norm = float(np.mean(stds))
    for loader in (train_loader, valid_loader, test_loader):
        loader.dataset.amp_norm = amp_norm

    return DataLoader(
        Subset(source.dataset, [local_idx]),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    ), amp_norm, local_idx


def reverse_with_frames(model, observed_data, cond_mask, side_info, keep_steps):
    B, _, _, _ = observed_data.shape
    current_sample = torch.randn_like(observed_data)
    frames = {}

    with torch.no_grad():
        for t in range(model.num_steps - 1, -1, -1):
            if model.is_unconditional:
                diff_input = cond_mask * observed_data + (1.0 - cond_mask) * current_sample
            else:
                cond_obs = cond_mask * observed_data
                diff_input = torch.cat([cond_obs, current_sample], dim=1)

            predicted = model.diffmodel(
                diff_input,
                side_info,
                torch.full((B,), t, device=model.device, dtype=torch.long),
            )
            coeff1 = 1.0 / model.alpha_hat[t] ** 0.5
            coeff2 = (1.0 - model.alpha_hat[t]) / (1.0 - model.alpha[t]) ** 0.5
            current_sample = coeff1 * (current_sample - coeff2 * predicted)
            current_sample = cond_mask * observed_data + (1.0 - cond_mask) * current_sample

            if t in keep_steps:
                frames[t] = current_sample.detach().cpu()

    return current_sample.detach(), frames


def reverse_all_frames(model, observed_data, cond_mask, side_info, stride=1):
    B, _, _, _ = observed_data.shape
    current_sample = torch.randn_like(observed_data)
    frames = []

    with torch.no_grad():
        for t in range(model.num_steps - 1, -1, -1):
            if model.is_unconditional:
                diff_input = cond_mask * observed_data + (1.0 - cond_mask) * current_sample
            else:
                cond_obs = cond_mask * observed_data
                diff_input = torch.cat([cond_obs, current_sample], dim=1)

            predicted = model.diffmodel(
                diff_input,
                side_info,
                torch.full((B,), t, device=model.device, dtype=torch.long),
            )
            coeff1 = 1.0 / model.alpha_hat[t] ** 0.5
            coeff2 = (1.0 - model.alpha_hat[t]) / (1.0 - model.alpha[t]) ** 0.5
            current_sample = coeff1 * (current_sample - coeff2 * predicted)
            current_sample = cond_mask * observed_data + (1.0 - cond_mask) * current_sample

            if t % stride == 0 or t == model.num_steps - 1:
                frames.append((t, current_sample[0].detach().cpu()))

    return current_sample.detach(), frames


def main():
    parser = argparse.ArgumentParser(description="Visualize forward and reverse diffusion for one audio item.")
    parser.add_argument("--config", default="base.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", default="", help="Path to model .pth or save folder name")
    parser.add_argument("--split", choices=["train", "valid", "test"], default="valid")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--item_id", default="", help="Item id, e.g. wav filename without .wav; overrides --index")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--nfold", type=int, default=0)
    parser.add_argument("--missing_ratio", type=float, default=0.0)
    parser.add_argument("--steps", default="auto", help="Reverse steps to save, e.g. auto or 99,75,50,25,10,0")
    parser.add_argument("--make_video", action="store_true", help="Save forward/reverse/inference animations")
    parser.add_argument("--video_stride", type=int, default=2, help="Save every N diffusion steps in videos")
    parser.add_argument("--video_fps", type=int, default=8)
    parser.add_argument("--outdir", default="")
    parser.add_argument("--clean_dir", default=DEFAULT_CLEAN_DIR)
    parser.add_argument("--lossy_dir", default=DEFAULT_LOSSY_DIR)
    parser.add_argument("--traces_dir", default=DEFAULT_TRACES_DIR)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    config = load_config(args.config)
    config["model"]["is_unconditional"] = False
    config["diffusion"]["max_seq_len_time"] = L_FRAMES
    config["diffusion"]["max_seq_len_freq"] = TARGET_DIM

    visual_name = args.item_id or str(args.index)
    outdir = args.outdir or os.path.join("save", f"diffusion_visual_{args.split}_{visual_name}")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "visual_config.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "config": config}, f, indent=2)

    if args.item_id:
        loader, amp_norm, local_idx = build_loader_for_item_id(args, config)
        print(f"Selected item_id={args.item_id} from split={args.split}, local_index={local_idx}")
    else:
        loader, amp_norm = build_loader(args, config)
    batch = next(iter(loader))

    model = CSDI_Physio(config, args.device, target_dim=TARGET_DIM).to(args.device)
    if args.checkpoint:
        ckpt = args.checkpoint
        if os.path.isdir(ckpt):
            ckpt = os.path.join(ckpt, "model.pth")
        elif not os.path.exists(ckpt):
            ckpt = os.path.join("save", ckpt, "model.pth")
        print(f"Loading checkpoint: {ckpt}")
        model.load_state_dict(torch.load(ckpt, map_location=args.device))
    model.eval()

    observed_data, clean_data, observed_mask, observed_tp, _, _ = model.process_data(batch)
    cond_mask = observed_mask
    side_info = model.get_side_info(observed_tp, cond_mask)
    keep_steps = parse_steps(args.steps, model.num_steps - 1)

    item_id = batch.get("item_id", ["item_0"])[0]
    print(f"Item: {item_id}")
    print(f"amp_norm: {amp_norm:.4f}")
    print(f"hole ratio: {(1.0 - observed_mask).mean().item():.4f}")
    print(f"Saving visualizations to: {outdir}")

    clean_ri = clean_data[0].detach().cpu()
    lossy_ri = observed_data[0].detach().cpu()
    mask = observed_mask[0].detach().cpu()
    clean_db = ri_to_db(clean_ri)
    vmin = float(np.percentile(clean_db, 1))
    vmax = float(np.percentile(clean_db, 99))

    save_spec(os.path.join(outdir, "00_clean_target.png"), clean_ri, f"{item_id} clean target", vmin, vmax)
    save_spec(os.path.join(outdir, "01_lossy_input_masked.png"), lossy_ri, f"{item_id} lossy input", vmin, vmax, mask)

    if "clean_wav" in batch:
        sf.write(os.path.join(outdir, f"{item_id}_clean.wav"), batch["clean_wav"][0].cpu().numpy(), SR)
    if "lossy_wav" in batch:
        sf.write(os.path.join(outdir, f"{item_id}_lossy.wav"), batch["lossy_wav"][0].cpu().numpy(), SR)

    forward_dir = os.path.join(outdir, "forward_model_input_noisy_everywhere")
    forward_composite_dir = os.path.join(outdir, "forward_inpaint_composite_view")
    reverse_dir = os.path.join(outdir, "reverse_denoise")
    holes_forward_dir = os.path.join(outdir, "forward_model_input_holes_only")
    holes_forward_composite_dir = os.path.join(outdir, "forward_inpaint_composite_holes_only")
    holes_reverse_dir = os.path.join(outdir, "reverse_denoise_holes_only")
    os.makedirs(forward_dir, exist_ok=True)
    os.makedirs(forward_composite_dir, exist_ok=True)
    os.makedirs(reverse_dir, exist_ok=True)
    os.makedirs(holes_forward_dir, exist_ok=True)
    os.makedirs(holes_forward_composite_dir, exist_ok=True)
    os.makedirs(holes_reverse_dir, exist_ok=True)

    fixed_noise = torch.randn_like(clean_data)
    forward_video_frames = []
    forward_composite_video_frames = []
    for t in sorted(keep_steps):
        tt = torch.full((clean_data.shape[0],), t, device=model.device, dtype=torch.long)
        alpha_t = model.alpha_torch[tt]
        x_t = alpha_t.sqrt() * clean_data + (1.0 - alpha_t).sqrt() * fixed_noise
        noisy_model_input = x_t
        noisy_composite_view = cond_mask * observed_data + (1.0 - cond_mask) * x_t
        save_triplet(
            os.path.join(forward_dir, f"forward_model_input_t{t:03d}.png"),
            clean_ri,
            lossy_ri,
            noisy_model_input[0].detach().cpu(),
            mask,
            f"Forward diffusion model input: noisy clean everywhere, t={t}",
        )
        save_triplet(
            os.path.join(forward_composite_dir, f"forward_composite_t{t:03d}.png"),
            clean_ri,
            lossy_ri,
            noisy_composite_view[0].detach().cpu(),
            mask,
            f"Forward diffusion composite view: known observed + noisy holes, t={t}",
        )
        save_holes_triplet(
            os.path.join(holes_forward_dir, f"forward_model_input_holes_t{t:03d}.png"),
            clean_ri,
            lossy_ri,
            noisy_model_input[0].detach().cpu(),
            mask,
            f"Forward diffusion model input holes only: t={t}",
        )
        save_holes_triplet(
            os.path.join(holes_forward_composite_dir, f"forward_composite_holes_t{t:03d}.png"),
            clean_ri,
            lossy_ri,
            noisy_composite_view[0].detach().cpu(),
            mask,
            f"Forward diffusion composite holes only: t={t}",
        )

    if args.make_video:
        for t in range(0, model.num_steps, max(args.video_stride, 1)):
            tt = torch.full((clean_data.shape[0],), t, device=model.device, dtype=torch.long)
            alpha_t = model.alpha_torch[tt]
            x_t = alpha_t.sqrt() * clean_data + (1.0 - alpha_t).sqrt() * fixed_noise
            noisy_composite_view = cond_mask * observed_data + (1.0 - cond_mask) * x_t
            forward_video_frames.append((t, x_t[0].detach().cpu()))
            forward_composite_video_frames.append((t, noisy_composite_view[0].detach().cpu()))
        if forward_video_frames[-1][0] != model.num_steps - 1:
            t = model.num_steps - 1
            tt = torch.full((clean_data.shape[0],), t, device=model.device, dtype=torch.long)
            alpha_t = model.alpha_torch[tt]
            x_t = alpha_t.sqrt() * clean_data + (1.0 - alpha_t).sqrt() * fixed_noise
            noisy_composite_view = cond_mask * observed_data + (1.0 - cond_mask) * x_t
            forward_video_frames.append((t, x_t[0].detach().cpu()))
            forward_composite_video_frames.append((t, noisy_composite_view[0].detach().cpu()))
        save_animation(
            os.path.join(outdir, "video_forward_model_input_noisy_everywhere"),
            forward_video_frames,
            clean_ri,
            lossy_ri,
            mask,
            f"Forward diffusion model input / noisy clean everywhere: {item_id}",
            fps=args.video_fps,
        )
        save_holes_animation(
            os.path.join(outdir, "video_forward_model_input_holes_only"),
            forward_video_frames,
            clean_ri,
            lossy_ri,
            mask,
            f"Forward diffusion model input holes only: {item_id}",
            fps=args.video_fps,
        )
        save_animation(
            os.path.join(outdir, "video_forward_inpaint_composite_view"),
            forward_composite_video_frames,
            clean_ri,
            lossy_ri,
            mask,
            f"Forward diffusion composite view / known observed + noisy holes: {item_id}",
            fps=args.video_fps,
        )
        save_holes_animation(
            os.path.join(outdir, "video_forward_inpaint_composite_holes_only"),
            forward_composite_video_frames,
            clean_ri,
            lossy_ri,
            mask,
            f"Forward diffusion composite holes only: {item_id}",
            fps=args.video_fps,
        )

    if args.make_video:
        final_sample, reverse_video_frames = reverse_all_frames(
            model,
            observed_data,
            cond_mask,
            side_info,
            stride=max(args.video_stride, 1),
        )
        frames = {t: ri.unsqueeze(0) for t, ri in reverse_video_frames}
        save_animation(
            os.path.join(outdir, "video_reverse_denoise"),
            reverse_video_frames,
            clean_ri,
            lossy_ri,
            mask,
            f"Reverse diffusion / denoising: {item_id}",
            fps=args.video_fps,
        )
        save_holes_animation(
            os.path.join(outdir, "video_reverse_denoise_holes_only"),
            reverse_video_frames,
            clean_ri,
            lossy_ri,
            mask,
            f"Reverse diffusion holes only: {item_id}",
            fps=args.video_fps,
        )
        inference_frames = reverse_video_frames
        save_animation(
            os.path.join(outdir, "video_inference_progress"),
            inference_frames,
            clean_ri,
            lossy_ri,
            mask,
            f"Inference progress from noise to reconstruction: {item_id}",
            fps=args.video_fps,
        )
        save_holes_animation(
            os.path.join(outdir, "video_inference_progress_holes_only"),
            inference_frames,
            clean_ri,
            lossy_ri,
            mask,
            f"Inference progress holes only: {item_id}",
            fps=args.video_fps,
        )
    else:
        final_sample, frames = reverse_with_frames(model, observed_data, cond_mask, side_info, set(keep_steps))

    for t in keep_steps:
        if t not in frames:
            continue
        save_triplet(
            os.path.join(reverse_dir, f"reverse_t{t:03d}.png"),
            clean_ri,
            lossy_ri,
            frames[t][0],
            mask,
            f"Reverse denoising: after step t={t}",
        )
        save_holes_triplet(
            os.path.join(holes_reverse_dir, f"reverse_holes_t{t:03d}.png"),
            clean_ri,
            lossy_ri,
            frames[t][0],
            mask,
            f"Reverse denoising holes only: after step t={t}",
        )

    alpha = float(batch["amp_alpha"][0].item()) if "amp_alpha" in batch else 0.5
    beta = float(batch["amp_beta"][0].item()) if "amp_beta" in batch else 3.6454
    norm = float(batch["amp_norm"][0].item()) if "amp_norm" in batch else 1.0
    recon_wav = recon_ri_to_wav(
        observed_data[0].detach().cpu(),
        final_sample[0].detach().cpu(),
        mask,
        alpha=alpha,
        beta=beta,
        norm=norm,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        match_hole_energy=True,
    )
    sf.write(os.path.join(outdir, f"{item_id}_recon.wav"), recon_wav.numpy(), SR)

    final_ri = observed_data[0].detach().cpu() * mask + final_sample[0].detach().cpu() * (1.0 - mask)
    hole = 1.0 - mask
    if hole.sum() > 0 and mask.sum() > 0:
        known_vals = observed_data[0].detach().cpu()[mask.expand_as(final_ri) > 0]
        hole_vals = final_ri[hole.expand_as(final_ri) > 0]
        scale = (known_vals.std() / (hole_vals.std() + 1e-8)).clamp(0.25, 2.0)
        final_ri = observed_data[0].detach().cpu() * mask + final_ri * hole * scale
        print(f"final hole energy scale: {scale.item():.4f}")

    save_triplet(
        os.path.join(outdir, "99_final_reconstruction.png"),
        clean_ri,
        lossy_ri,
        final_ri,
        mask,
        f"Final reconstruction: {item_id}",
    )
    save_holes_triplet(
        os.path.join(outdir, "99_final_reconstruction_holes_only.png"),
        clean_ri,
        lossy_ri,
        final_ri,
        mask,
        f"Final reconstruction holes only: {item_id}",
    )


if __name__ == "__main__":
    main()
