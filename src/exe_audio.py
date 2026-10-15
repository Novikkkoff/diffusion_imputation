import argparse
import torch
import datetime
import json
import yaml
import os
import numpy as np
from main_model import CSDI_Physio
from dataset_packetloss import get_packet_loss_dataloaders
from utils import train, evaluate, evaluate_one_item_metrics, print_item_metrics_table


parser = argparse.ArgumentParser(description="CSDI Audio Inpainting")
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--testmissingratio", type=float, default=0.0)
parser.add_argument("--nfold", type=int, default=0, help="0-4")
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=1)
parser.add_argument("--limit_train", type=int, default=0, help="Use only the first N training items; 0 means all")
parser.add_argument("--limit_valid", type=int, default=10, help="Use only the first N validation items for checks; 0 means all")
parser.add_argument("--overfit_index", type=int, default=-1, help="If >=0, train/evaluate on this single train item")
parser.add_argument("--item_id", type=str, default="", help="Train/evaluate on item id(s), comma-separated; names are wav filenames without .wav")

args = parser.parse_args()

path = "config/" + args.config
if not os.path.exists(path):
    path = args.config

with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
config["model"]["test_missing_ratio"] = args.testmissingratio

print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = f"./save/audio_fold{args.nfold}_{current_time}/"
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

CLEAN_DIR  = "/home/ubuntu/projects/ia.novikov/CSDIweather/CSDI/train/clean_signals"
LOSSY_DIR  = "/home/ubuntu/projects/ia.novikov/CSDIweather/CSDI/train/lossy_signals"
TRACES_DIR = "/home/ubuntu/projects/ia.novikov/CSDIweather/CSDI/train/traces"

SR         = 16000
N_FFT      = 512
HOP_LENGTH = 256   # hop = win/2 = 256  -> COLA условие выполнено для Hann окна
WIN_LENGTH = 512   # COLA: hop <= win//2  =>  256 <= 256  ✓
                   # hop=320 нарушал COLA -> ISTFT падал с "window overlap add min: 1"

# С center=True: N = ceil(T / hop) = ceil(160000 / 256) = 625
L_FRAMES   = 625
TARGET_DIM = N_FFT // 2 + 1  # = 257

config["diffusion"]["max_seq_len_time"] = L_FRAMES  # точно совпадает с center=True STFT
config["diffusion"]["max_seq_len_freq"] = TARGET_DIM

train_loader, valid_loader, test_loader = get_packet_loss_dataloaders(
    clean_dir=CLEAN_DIR,
    lossy_dir=LOSSY_DIR,
    traces_dir=TRACES_DIR,
    batch_size=config["train"]["batch_size"],
    seed=args.seed,
    n_folds=5,
    fold_idx=args.nfold,
    missing_ratio=config["model"].get("train_missing_ratio", 0.0),
    sr=SR,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    win_length=WIN_LENGTH,
    L=L_FRAMES,
    return_clean=True,
)
from torch.utils.data import Subset, DataLoader

def make_subset_loader(base_loader, indices, shuffle=False):
    return DataLoader(
        Subset(base_loader.dataset, indices),
        batch_size=base_loader.batch_size,
        shuffle=shuffle,
        num_workers=base_loader.num_workers,
        pin_memory=True,
        drop_last=False,
    )

def find_local_index_by_item_id(dataset, item_id):
    item_id = os.path.splitext(os.path.basename(item_id))[0]
    for local_idx, global_idx in enumerate(dataset.use_index_list):
        if dataset.ids[int(global_idx)] == item_id:
            return local_idx
    return None

def parse_item_ids(text):
    return [
        os.path.splitext(os.path.basename(x.strip()))[0]
        for x in text.split(",")
        if x.strip()
    ]

train_ds = train_loader.dataset
valid_ds = valid_loader.dataset
train_ds.missing_ratio = config["model"].get("train_missing_ratio", 0.0)
valid_ds.missing_ratio = 0.0

if args.item_id:
    item_ids = parse_item_ids(args.item_id)
    train_hits = []
    valid_hits = []
    missing_ids = []

    for item_id in item_ids:
        train_idx = find_local_index_by_item_id(train_ds, item_id)
        valid_idx = find_local_index_by_item_id(valid_ds, item_id)
        if train_idx is None and valid_idx is None:
            missing_ids.append(item_id)
            continue
        if train_idx is not None:
            train_hits.append((item_id, train_idx))
        if valid_idx is not None:
            valid_hits.append((item_id, valid_idx))

    if missing_ids:
        raise ValueError(f"item_id(s) not found in train or valid split: {missing_ids}")

    if len(train_hits) == len(item_ids):
        selected_loader = train_loader
        selected_indices = [idx for _, idx in train_hits]
        selected_split = "train"
    elif len(valid_hits) == len(item_ids):
        selected_loader = valid_loader
        selected_indices = [idx for _, idx in valid_hits]
        selected_split = "valid"
    else:
        raise ValueError(
            "Selected item_ids are split across train/valid. "
            f"train={train_hits}, valid={valid_hits}. "
            "Use item ids from one split, or change seed/nfold."
        )

    active_train_loader = make_subset_loader(selected_loader, selected_indices, shuffle=True)
    active_valid_loader = make_subset_loader(selected_loader, selected_indices, shuffle=False)
    print(f"Training mode: selected item_ids={item_ids}")
    print(f"Selected split: {selected_split}")
    print(f"Selected local indices: {selected_indices}")
elif args.overfit_index >= 0:
    train_indices = [args.overfit_index]
    valid_indices = [args.overfit_index]
    active_train_loader = make_subset_loader(train_loader, train_indices, shuffle=False)
    active_valid_loader = make_subset_loader(train_loader, valid_indices, shuffle=False)
    print(f"Training mode: overfit one train item index={args.overfit_index}")
else:
    n_train = len(train_ds) if args.limit_train <= 0 else min(args.limit_train, len(train_ds))
    n_valid = len(valid_ds) if args.limit_valid <= 0 else min(args.limit_valid, len(valid_ds))
    active_train_loader = make_subset_loader(train_loader, list(range(n_train)), shuffle=True)
    # active_valid_loader = make_subset_loader(valid_loader, list(range(n_valid)), shuffle=False)
    active_valid_loader = active_train_loader
    print(f"Training mode: train_items={n_train}, valid_items={n_valid}")

batch = next(iter(active_train_loader))
print("hole ratio:", (1.0 - batch["observed_mask"]).mean().item())
print("holes count:", (1.0 - batch["observed_mask"]).sum().item())
model = CSDI_Physio(config, args.device, target_dim=TARGET_DIM).to(args.device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(model.alpha[-1])
print((1 - model.alpha[-1]) ** 0.5)
print(model.alpha[-1] ** 0.5)
if args.modelfolder != "":
    ckpt_path = f"./save/{args.modelfolder}/model_epoch_300.pth"
    print(f"Loading pretrained weights from {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=args.device))

# Вычисляем реальный std по нескольким батчам
stds = []
for i, batch in enumerate(train_loader):
    ri = batch["clean_ri"]  # (B, 2, F, L) — уже после _compress_ri
    stds.append(ri.std().item())
    if i >= 50:
        break
real_std = float(np.mean(stds))
train_loader.dataset.amp_norm=real_std
valid_loader.dataset.amp_norm=real_std
test_loader.dataset.amp_norm=real_std
print(f"Real compressed RI std: {real_std:.4f}")
# Затем выставь amp_norm = real_std в датасете и пересоздай лоадеры
# Sanity check: compress -> decompress должен давать оригинал
# Берём сырой STFT (до компрессии), а не batch["clean_ri"] который уже сжат
ds = train_loader.dataset
_test_wav = ds._load_wav_mono(
    os.path.join(ds.clean_dir, f"{ds.ids[0]}.wav")
)
_test_wav, _, _ = ds._crop_pad_triplet(
    _test_wav, _test_wav, torch.ones(_test_wav.numel()), ds.wav_len
)
ri_raw = ds._stft_ri(_test_wav)        # (2, F, N) — сырой STFT
ri_comp = ds._compress_ri(ri_raw)      # сжимаем один раз
ri_back = ds._decompress_ri(ri_comp)   # обратно
err = (ri_raw - ri_back).abs().max().item()
print(f"Round-trip max error: {err:.6f}  (должно быть < 1e-4)")
print(f"Compressed RI range: [{ri_comp.min():.4f}, {ri_comp.max():.4f}]")
print(f"Compressed RI std:   {ri_comp.std():.4f}  (хотим ~1.0)")
train(
    model,
    config["train"],
    active_train_loader,
    valid_loader=active_valid_loader,
    foldername=foldername,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    win_length=WIN_LENGTH,
    sr=SR,
)
# --- DEBUG: one-step x0_hat reconstruction ---
import matplotlib.pyplot as plt

def save_debug_spec(lossy_ri, clean_ri, recon_ri, mask, save_path, sr=16000, hop_length=256):
    def ri_to_db(ri, eps=1e-6):
        mag = torch.sqrt(ri[0] ** 2 + ri[1] ** 2 + eps)
        return 20.0 * torch.log10(mag + eps).detach().cpu().numpy()

    spec_lossy = ri_to_db(lossy_ri)
    spec_clean = ri_to_db(clean_ri)
    spec_recon = ri_to_db(recon_ri)

    F, L = spec_clean.shape
    t_max = L * hop_length / sr
    vmin = spec_clean.min()
    vmax = spec_clean.max()

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, sharey=True)

    for ax, title, spec in zip(
        axes,
        ["Lossy", "Clean", "x0_hat direct recon"],
        [spec_lossy, spec_clean, spec_recon],
    ):
        im = ax.imshow(
            spec,
            aspect="auto",
            origin="lower",
            extent=[0, t_max, 0, sr / 2 / 1000],
            vmin=vmin,
            vmax=vmax,
            cmap="magma",
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_ylabel("Frequency (kHz)")
        fig.colorbar(im, ax=ax, pad=0.01)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


model.eval()
batch = next(iter(active_valid_loader))

# важно: перенести batch на device делает process_data внутри модели,
# если у тебя process_data уже это делает. Если нет — надо вручную.
with torch.no_grad():
    for t_value in [5, 10, 25, 50, 75]:
        recon_direct, x0_hat, clean_data, observed_data, cond_mask = model.debug_x0_hat(
            batch, t_value=t_value
        )

        debug_dir = os.path.join(foldername, "debug_x0_hat")
        os.makedirs(debug_dir, exist_ok=True)

        for i in range(min(4, recon_direct.shape[0])):
            item_id = batch.get("item_id", [f"item_{i}"])[i]
            save_debug_spec(
                lossy_ri=observed_data[i].cpu(),
                clean_ri=clean_data[i].cpu(),
                recon_ri=recon_direct[i].cpu(),
                mask=cond_mask[i].cpu(),
                save_path=os.path.join(debug_dir, f"{item_id}_t{t_value}_x0hat.png"),
                sr=SR,
                hop_length=HOP_LENGTH,
            )
evaluate(
    model,
    active_valid_loader,
    nsample=args.nsample,
    scaler=1,
    foldername=foldername,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    win_length=WIN_LENGTH,
    sr=SR,
)

metric_batch = next(iter(active_valid_loader))
for i in range(3):
    metric_rows = evaluate_one_item_metrics(
        model,
        metric_batch,
        item_index=i,
        nsample=args.nsample,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        sr=SR,
    )
    print_item_metrics_table(metric_rows)
