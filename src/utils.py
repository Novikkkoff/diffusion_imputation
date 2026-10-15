import os
import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import itertools

try:
    from torchmetrics.audio import DeepNoiseSuppressionMeanOpinionScore
except ImportError:
    try:
        from torchmetrics.audio.dnsmos import DeepNoiseSuppressionMeanOpinionScore
    except ImportError:
        DeepNoiseSuppressionMeanOpinionScore = None

try:
    from pesq import pesq as pesq_fn
    HAS_PESQ = True
except ImportError:
    HAS_PESQ = False


# ---------------------------------------------------------------------------
# ISTFT из Re/Im тензора
# ---------------------------------------------------------------------------

def ri_to_wav(
    ri: torch.Tensor,
    n_fft: int = 512,
    hop_length: int = 256,
    win_length: int = 512,
    length_samples: int = None,
) -> torch.Tensor:
    """
    ri: (2, F, L) — Re/Im спектрограмма
    Возвращает: (T,) float32 waveform
    """
    X = torch.complex(ri[0], ri[1])
    window = torch.hann_window(win_length, device=X.device)
    return torch.istft(
        X,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        length=length_samples,
    )


# ---------------------------------------------------------------------------
# Метрики
# ---------------------------------------------------------------------------

def si_sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """est, ref: (T,). Возвращает scalar (dB)."""
    est = est - est.mean()
    ref = ref - ref.mean()
    scale    = torch.dot(est, ref) / (torch.dot(ref, ref) + eps)
    s_target = scale * ref
    e_noise  = est - s_target
    ratio = (torch.dot(s_target, s_target) + eps) / (torch.dot(e_noise, e_noise) + eps)
    return 10.0 * torch.log10(ratio)


def pesq_wb_16k(ref_wav: torch.Tensor, deg_wav: torch.Tensor, sr: int = 16000) -> float:
    """Возвращает float PESQ или np.nan если нельзя посчитать."""
    if not HAS_PESQ:
        return np.nan
    T = min(ref_wav.numel(), deg_wav.numel())
    if T < sr // 4:
        return np.nan
    ref = ref_wav[:T].detach().cpu().numpy().astype(np.float32)
    deg = deg_wav[:T].detach().cpu().numpy().astype(np.float32)
    if not (np.isfinite(ref).all() and np.isfinite(deg).all()):
        return np.nan
    ref = np.clip(ref, -1.0, 1.0)
    deg = np.clip(deg, -1.0, 1.0)
    if np.sqrt(np.mean(ref ** 2)) < 1e-4 or np.sqrt(np.mean(deg ** 2)) < 1e-4:
        return np.nan
    try:
        return float(pesq_fn(sr, ref, deg, "wb"))
    except Exception:
        return np.nan


def get_dnsmos_metric(device=None):
    if DeepNoiseSuppressionMeanOpinionScore is None:
        return None
    return DeepNoiseSuppressionMeanOpinionScore(
        fs=16000, personalized=False, device=device or "cpu",
    )


def dnsmos_score_one(metric, wav: torch.Tensor) -> float:
    x = wav.detach().float().clamp(-1.0, 1.0)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    out = metric(x)
    if isinstance(out, dict):
        for k in ("ovrl", "OVRL", "overall", "mos"):
            if k in out:
                v = out[k]
                if isinstance(v, torch.Tensor):
                    return float(v.detach().cpu().reshape(-1)[0].item())
                return float(v)
        first = next(iter(out.values()))
        if isinstance(first, torch.Tensor):
            return float(first.detach().cpu().reshape(-1)[0].item())
        return float(first)
    if isinstance(out, torch.Tensor):
        out = out.detach().cpu().reshape(-1)
        return float(out[2].item()) if out.numel() >= 3 else float(out[0].item())
    return float(out)


def metric_row_against_clean(
    clean_wav: torch.Tensor,
    wav: torch.Tensor,
    sr: int = 16000,
    dnsmos_metric=None,
) -> dict:
    """Compute SI-SDR/PESQ against clean and DNSMOS for one waveform."""
    T = min(clean_wav.numel(), wav.numel())
    clean_ = clean_wav[:T].detach().cpu().float()
    wav_ = wav[:T].detach().cpu().float()

    out = {
        "si_sdr": np.nan,
        "pesq": np.nan,
        "dnsmos": np.nan,
    }

    if torch.isfinite(wav_).all() and wav_.abs().max() >= 1e-6:
        out["si_sdr"] = float(si_sdr(wav_, clean_).item())
        out["pesq"] = float(pesq_wb_16k(clean_, wav_, sr=sr))

    if dnsmos_metric is not None:
        try:
            out["dnsmos"] = float(dnsmos_score_one(dnsmos_metric, wav_))
        except Exception:
            out["dnsmos"] = np.nan

    return out


def evaluate_one_item_metrics(
    model,
    batch,
    item_index: int = 0,
    nsample: int = 1,
    n_fft: int = 512,
    hop_length: int = 256,
    win_length: int = 512,
    sr: int = 16000,
    dnsmos_metric=None,
) -> dict:
    """
    Return metrics for clean/lossy/recon for one item from a batch.

    SI-SDR and PESQ are computed against clean_wav.
    DNSMOS is computed independently for each waveform.
    """
    if "clean_wav" not in batch or "lossy_wav" not in batch:
        raise ValueError("Batch must contain clean_wav and lossy_wav. Create dataset with return_wav=True.")

    if dnsmos_metric is None:
        try:
            dnsmos_metric = get_dnsmos_metric(device="cpu")
        except Exception as e:
            print(f"DNSMOS init failed: {e}")
            dnsmos_metric = None

    model.eval()
    with torch.no_grad():
        samples, observed_data, _, observed_mask, _ = model.evaluate(batch, nsample=nsample)
        imputed = samples.median(dim=1).values

    i = item_index
    alpha = float(batch["amp_alpha"][i].item()) if "amp_alpha" in batch else 0.5
    beta = float(batch["amp_beta"][i].item()) if "amp_beta" in batch else 3.6454
    norm = float(batch["amp_norm"][i].item()) if "amp_norm" in batch else 1.0

    recon_wav = recon_ri_to_wav(
        observed_data[i].cpu(),
        imputed[i].cpu(),
        observed_mask[i].cpu(),
        alpha=alpha,
        beta=beta,
        norm=norm,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    )

    clean_wav = batch["clean_wav"][i].cpu()
    lossy_wav = batch["lossy_wav"][i].cpu()
    item_id = batch.get("item_id", [f"item_{i}"])[i]

    rows = {
        "clean": metric_row_against_clean(clean_wav, clean_wav, sr=sr, dnsmos_metric=dnsmos_metric),
        "lossy": metric_row_against_clean(clean_wav, lossy_wav, sr=sr, dnsmos_metric=dnsmos_metric),
        "recon": metric_row_against_clean(clean_wav, recon_wav, sr=sr, dnsmos_metric=dnsmos_metric),
    }

    rows["delta_recon_minus_lossy"] = {
        key: rows["recon"][key] - rows["lossy"][key]
        for key in ("si_sdr", "pesq", "dnsmos")
    }
    rows["item_id"] = item_id
    return rows


def print_item_metrics_table(metrics: dict):
    item_id = metrics.get("item_id", "")
    if item_id:
        print(f"\nMetrics for item: {item_id}")
    print(f"{'signal':<24} {'SI-SDR':>10} {'PESQ':>10} {'DNSMOS':>10}")
    for name in ("clean", "lossy", "recon", "delta_recon_minus_lossy"):
        row = metrics[name]
        print(
            f"{name:<24} "
            f"{row['si_sdr']:>10.3f} "
            f"{row['pesq']:>10.3f} "
            f"{row['dnsmos']:>10.3f}"
        )

def decompress_ri(ri_tilde, alpha=0.5, beta=3.6454, eps=1e-8, norm=1):
    re = ri_tilde[0] * norm
    im = ri_tilde[1] * norm
    mag_tilde = torch.sqrt(re**2 + im**2 + eps)
    inv_scale = (beta ** (-1.0 / alpha)) * torch.pow(mag_tilde, (1.0 / alpha) - 1.0)
    return torch.stack([re * inv_scale, im * inv_scale], dim=0)


# ---------------------------------------------------------------------------
# Восстановление одного Re/Im тензора → wav  (с декомпрессией)
# ---------------------------------------------------------------------------

def recon_ri_to_wav(
    orig_ri_norm: torch.Tensor,   # (2, F, N) lossy нормированный
    imputed_ri: torch.Tensor,     # (2, F, N) восстановленный нормированный
    mask: torch.Tensor,           # (1, F, N) 1=known 0=hole
    alpha: float,
    beta: float,
    norm: float,
    n_fft: int,
    hop_length: int,
    win_length: int,
    match_hole_energy: bool = False,
    min_scale: float = 0.25,
    max_scale: float = 2.0,
    ) -> torch.Tensor:
    """Склеивает known (из orig) и holes (из imputed), декомпрессирует, делает ISTFT."""
    orig_ri_norm = orig_ri_norm.cpu()
    imputed_ri = imputed_ri.cpu()
    mask = mask.cpu()
    hole = 1.0 - mask
    if match_hole_energy and hole.sum() > 0 and mask.sum() > 0:
        known_vals = orig_ri_norm[mask.expand_as(orig_ri_norm) > 0]
        hole_vals = imputed_ri[hole.expand_as(imputed_ri) > 0]
        known_std = known_vals.std()
        hole_std = hole_vals.std()
        scale = (known_std / (hole_std + 1e-8)).clamp(min_scale, max_scale)
        imputed_ri = imputed_ri * scale
    final_norm = orig_ri_norm * mask + imputed_ri * hole
    final_ri = decompress_ri(final_norm.cpu(), alpha=alpha, beta=beta, norm=norm)

    return ri_to_wav(final_ri, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


# ---------------------------------------------------------------------------
# Визуализация спектрограмм
# ---------------------------------------------------------------------------

def _save_spectrogram_figure(
    lossy_ri: torch.Tensor,
    clean_ri: torch.Tensor,
    recon_ri: torch.Tensor,
    mask: torch.Tensor,
    save_path: str,
    sr: int = 16000,
    hop_length: int = 256,
    item_id: str = "",
):
    """
    lossy_ri, clean_ri, recon_ri: (2, F, L) нормированные Re/Im
    mask: (1, F, L) 1=known 0=hole
    """
    def ri_to_db(ri, eps=1e-6):
        mag = torch.sqrt(ri[0] ** 2 + ri[1] ** 2 + eps)
        return 20.0 * torch.log10(mag + eps).cpu().numpy()

    spec_lossy = ri_to_db(lossy_ri)
    spec_clean = ri_to_db(clean_ri)
    spec_recon = ri_to_db(recon_ri)
    hole_map   = (1.0 - mask[0]).cpu().numpy()

    F, L   = spec_clean.shape
    t_max  = L * hop_length / sr
    vmin   = spec_clean.min()
    vmax   = spec_clean.max()

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, sharey=True)
    fig.suptitle(f"Spectrograms — {item_id}", fontsize=11)

    for ax, title, spec, mark_holes in zip(
        axes,
        ["Lossy (input)", "Clean (target)", "Reconstructed"],
        [spec_lossy, spec_clean, spec_recon],
        [True, False, False],
    ):
        img = ax.imshow(
            spec, aspect="auto", origin="lower",
            extent=[0, t_max, 0, sr / 2 / 1000],
            vmin=vmin, vmax=vmax, cmap="magma", interpolation="nearest",
        )
        ax.set_ylabel("Frequency (kHz)", fontsize=9)
        ax.set_title(title, fontsize=10, pad=3)
        if mark_holes:
            hole_rgba = np.zeros((*hole_map.shape, 4))
            hole_rgba[..., 2] = 1.0
            hole_rgba[..., 3] = hole_map * 0.45
            ax.imshow(
                hole_rgba, aspect="auto", origin="lower",
                extent=[0, t_max, 0, sr / 2 / 1000], interpolation="nearest",
            )
            patch = mpatches.Patch(color=(0, 0, 1, 0.45), label="packet loss")
            ax.legend(handles=[patch], fontsize=8, loc="upper right")
        fig.colorbar(img, ax=ax, pad=0.01, format="%.0f dB", aspect=25)

    axes[-1].set_xlabel("Time (s)", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Быстрая проверка качества восстановления прямо во время обучения
# ---------------------------------------------------------------------------

def quick_audio_check(
    model,
    valid_loader,
    epoch_no: int,
    foldername: str,
    n_fft: int = 512,
    hop_length: int = 256,
    win_length: int = 512,
    sr: int = 16000,
    n_samples: int = 2,       # сколько файлов из батча проверять
):
    """
    Запускает inference на первых n_samples файлах из одного валидационного батча.
    Сохраняет спектрограмму PNG и wav, печатает SI-SDR/PESQ.

    Цель: понять за 1-2 минуты идёт ли обучение в правильном направлении,
    не дожидаясь полного evaluate().
    """
    check_dir = os.path.join(foldername, "quick_checks")
    os.makedirs(check_dir, exist_ok=True)

    # берём всегда первый батч (один и тот же для сравнения между эпохами)
    batch = next(iter(valid_loader))

    model.eval()
    with torch.no_grad():
        samples, observed_data, target_mask, observed_mask, _ = model.evaluate(batch, nsample=1)

    # samples: (B, 1, 2, F, N) → берём первые n_samples
    imputed = samples[:n_samples, 0]          # (n_samples, 2, F, N)
    obs     = observed_data[:n_samples]       # (n_samples, 2, F, N)
    mask    = observed_mask[:n_samples]       # (n_samples, 1, F, N)

    results = []
    for i in range(min(n_samples, imputed.shape[0])):
        item_id = batch.get("item_id", [f"item_{i}"])[i]
        alpha = float(batch["amp_alpha"][i].item()) if "amp_alpha" in batch else 0.5
        beta  = float(batch["amp_beta"][i].item())  if "amp_beta"  in batch else 3.6454
        norm  = float(batch["amp_norm"][i].item())  if "amp_norm"  in batch else 1.0

        recon_wav = recon_ri_to_wav(
            obs[i].cpu(), imputed[i].cpu(), mask[i].cpu(),
            alpha=alpha, beta=beta, norm=norm,
            n_fft=n_fft, hop_length=hop_length, win_length=win_length, match_hole_energy=True
        )

        # метрики (только если есть clean_wav)
        si = np.nan
        pq = np.nan
        if "clean_wav" in batch:
            clean_wav = batch["clean_wav"][i].cpu()
            T = min(recon_wav.numel(), clean_wav.numel())
            recon_ = recon_wav[:T]
            clean_ = clean_wav[:T]

            if torch.isfinite(recon_).all() and recon_.abs().max() > 1e-6:
                si = si_sdr(recon_, clean_).item()
                pq = pesq_wb_16k(clean_, recon_, sr=sr)

                # сохраняем wav
                wav_path = os.path.join(check_dir, f"ep{epoch_no:04d}_{item_id}_recon.wav")
                sf.write(wav_path, recon_.numpy(), sr)

        # спектрограмма
        if "clean_ri" in batch:
            spec_path = os.path.join(check_dir, f"ep{epoch_no:04d}_{item_id}_spec.png")
            try:
                _save_spectrogram_figure(
                    lossy_ri=obs[i].cpu(),
                    clean_ri=batch["clean_ri"][i].cpu(),
                    recon_ri=(obs[i] * mask[i] + imputed[i] * (1 - mask[i])).cpu(),
                    mask=mask[i].cpu(),
                    save_path=spec_path,
                    sr=sr, hop_length=hop_length, item_id=f"ep{epoch_no} {item_id}",
                )
            except Exception as e:
                print(f"  [quick_check] spec save failed: {e}")

        results.append((item_id, si, pq))
        pesq_str = f"{pq:.3f}" if np.isfinite(pq) else "n/a"
        si_str   = f"{si:.2f} dB" if np.isfinite(si) else "n/a (NaN — модель ещё не учится)"
        print(f"  [{item_id}] SI-SDR={si_str}  PESQ={pesq_str}")

    return results


# ---------------------------------------------------------------------------
# TRAIN
# ---------------------------------------------------------------------------

def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval: int = 100,
    audio_check_interval: int = 100,    # каждые N эпох делать quick_audio_check
    foldername: str = "",
    n_fft: int = 512,
    hop_length: int = 256,
    win_length: int = 512,
    sr: int = 16000,
):
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)

    if foldername:
        os.makedirs(foldername, exist_ok=True)
        output_path = os.path.join(foldername, "model.pth")
        plot_path   = os.path.join(foldername, "loss_curve.png")
        hist_path   = os.path.join(foldername, "loss_history.npz")
    else:
        output_path = plot_path = hist_path = None

    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    train_loss_history  = []
    valid_loss_history  = []
    valid_epoch_history = []
    best_valid_loss     = float("inf")

    itr_per_epoch = int(config.get("itr_per_epoch", len(train_loader)))

    for epoch_no in range(config["epochs"]):

        # ---- TRAIN ----
        model.train()
        avg_loss    = 0.0
        nan_batches = 0
        grad_norms  = []
        train_iter = iter(train_loader)
        with tqdm(range(itr_per_epoch), mininterval=5.0, maxinterval=50.0, desc=f"Epoch {epoch_no}") as it:
            # for batch_no, train_batch in enumerate(it, start=1):
            for step_no in it:
                batch_no = step_no + 1
                try:
                    train_batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    train_batch = next(train_iter)
                optimizer.zero_grad(set_to_none=True)
                loss = model(train_batch)

                # --- NaN/Inf guard ---
                if not torch.isfinite(loss):
                    nan_batches += 1
                    if nan_batches <= 3:
                        print(f"\n  WARNING: loss={loss.item()} at epoch {epoch_no} batch {batch_no}")
                    optimizer.zero_grad(set_to_none=True)
                    # if batch_no >= itr_per_epoch:
                    #     break
                    continue

                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                grad_norms.append(grad_norm.item())
                optimizer.step()

                avg_loss += loss.item()
                it.set_postfix(
                    loss=f"{avg_loss / batch_no:.4f}",
                    grad=f"{grad_norm:.3f}",
                    refresh=False,
                )
                # if batch_no >= itr_per_epoch:
                #     break

        # lr_scheduler.step()
        train_epoch_loss = avg_loss / max(batch_no - nan_batches, 1)
        train_loss_history.append(train_epoch_loss)

        avg_grad = float(np.mean(grad_norms)) if grad_norms else 0.0
        nan_info = f"  NaN batches: {nan_batches}" if nan_batches > 0 else ""
        print(f"Epoch {epoch_no:4d} | train_loss={train_epoch_loss:.5f} | "
              f"grad_norm={avg_grad:.4f} | lr={optimizer.param_groups[0]['lr']:.2e}{nan_info}")

        # ---- VALID LOSS ----
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0.0
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, desc="Valid") as it:
                    for v_no, valid_batch in enumerate(it, start=1):
                        vloss = model(valid_batch, is_train=0)
                        if torch.isfinite(vloss):
                            avg_loss_valid += vloss.item()
                        if v_no >= 50:   # 50 батчей × 10 шагов = 500 форвард-пассов
                            break

            valid_epoch_loss = avg_loss_valid / max(v_no, 1)
            valid_loss_history.append(valid_epoch_loss)
            valid_epoch_history.append(epoch_no + 1)

            if valid_epoch_loss < best_valid_loss:
                best_valid_loss = valid_epoch_loss
                print(f"  best valid loss: {best_valid_loss:.6f} → сохранён _best.pth")
                if output_path:
                    torch.save(model.state_dict(), output_path.replace(".pth", "_best.pth"))

        # ---- QUICK AUDIO CHECK ----
        if (
            valid_loader is not None
            and foldername
            and (epoch_no + 1) % audio_check_interval == 0
        ):
            print(f"\n--- Quick audio check (epoch {epoch_no + 1}) ---")
            try:
                quick_audio_check(
                    model, train_loader, epoch_no + 1, foldername,
                    n_fft=n_fft, hop_length=hop_length, win_length=win_length, sr=sr,
                    n_samples=2,
                )
                quick_audio_check(
                    model, valid_loader, epoch_no + 1, foldername,
                    n_fft=n_fft, hop_length=hop_length, win_length=win_length, sr=sr,
                    n_samples=2,
                )
            except Exception as e:
                print(f"  quick_check failed: {e}")
            print()

        # ---- CHECKPOINT каждые 10 эпох ----
        if (epoch_no + 1) % 10 == 0 and foldername:
            ckpt_path = os.path.join(foldername, f"model_epoch_{epoch_no+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  checkpoint saved: {ckpt_path}")

        # ---- LOSS CURVE ----
        if plot_path and len(train_loss_history) > 0:
            _save_loss_plot(
                train_loss_history, valid_loss_history, valid_epoch_history, plot_path
            )
            if hist_path:
                np.savez(
                    hist_path,
                    train_loss=np.array(train_loss_history, dtype=np.float32),
                    valid_loss=np.array(valid_loss_history, dtype=np.float32),
                    valid_epochs=np.array(valid_epoch_history, dtype=np.int32),
                )

    if output_path:
        torch.save(model.state_dict(), output_path)
        print(f"Final model saved: {output_path}")


def _save_loss_plot(train_hist, valid_hist, valid_epochs, path):
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(1, len(train_hist) + 1), train_hist, label="train", linewidth=1.5)
    if valid_hist:
        plt.plot(valid_epochs, valid_hist, "o-", label="valid", linewidth=1.5, markersize=4)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()


# ---------------------------------------------------------------------------
# EVALUATE (полный прогон на тест-сете)
# ---------------------------------------------------------------------------

def evaluate(
    model,
    test_loader,
    nsample: int = 1,
    scaler: float = 1.0,
    foldername: str = "",
    n_fft: int = 512,
    hop_length: int = 256,
    win_length: int = 512,
    sr: int = 16000,
):
    """
    Полный inference на test_loader.
    Считает SI-SDR, PESQ(recon), PESQ(lossy), DNSMOS.
    Сохраняет wav + PNG спектрограммы в foldername/recon_wavs/.
    """
    assert hop_length <= win_length // 2, (
        f"COLA violation: hop={hop_length} > win//2={win_length//2}"
    )

    model.eval()

    si_sdr_sum       = 0.0
    pesq_recon_sum   = 0.0
    pesq_lossy_sum   = 0.0
    pesq_recon_n     = 0      # отдельные счётчики — PESQ может быть NaN для части файлов
    pesq_lossy_n     = 0
    n_audio          = 0

    dnsmos_clean_sum = 0.0
    dnsmos_lossy_sum = 0.0
    dnsmos_recon_sum = 0.0
    dnsmos_n         = 0

    dnsmos_metric = None
    try:
        dnsmos_metric = get_dnsmos_metric(device="cpu")
    except Exception as e:
        print(f"DNSMOS init failed: {e}")

    recon_dir = os.path.join(foldername, "recon_wavs") if foldername else None
    if recon_dir:
        os.makedirs(recon_dir, exist_ok=True)

    with torch.no_grad():
        with tqdm(test_loader, mininterval=5.0, desc="Evaluate") as it:
            for batch_no, test_batch in enumerate(itertools.islice(it, 100), start=1):
                output = model.evaluate(test_batch, nsample)
                samples, observed_data, target_mask, observed_mask, _ = output

                # медиана по nsample — устойчивее среднего для диффузии
                imputed = samples.median(dim=1).values  # (B, 2, F, N)
                B = imputed.shape[0]

                for i in range(B):
                    item_id = test_batch.get("item_id", [f"item_{batch_no}_{i}"])[i]
                    alpha = float(test_batch["amp_alpha"][i].item()) if "amp_alpha" in test_batch else 0.5
                    beta  = float(test_batch["amp_beta"][i].item())  if "amp_beta"  in test_batch else 3.6454
                    norm  = float(test_batch["amp_norm"][i].item())  if "amp_norm"  in test_batch else 1.0

                    mask_i     = observed_mask[i]    # (1, F, N)
                    orig_ri    = observed_data[i]    # (2, F, N)
                    imputed_ri = imputed[i]          # (2, F, N)

                    recon_wav = recon_ri_to_wav(
                        orig_ri.cpu(), imputed_ri.cpu(), mask_i.cpu(),
                        alpha=alpha, beta=beta, norm=norm,
                        n_fft=n_fft, hop_length=hop_length, win_length=win_length, match_hole_energy=True
                    )

                    if "clean_wav" not in test_batch:
                        continue

                    clean_wav = test_batch["clean_wav"][i].cpu()
                    T = min(recon_wav.numel(), clean_wav.numel())
                    recon_ = recon_wav[:T]
                    clean_ = clean_wav[:T]

                    # пропускаем NaN/silent реконструкции
                    if not torch.isfinite(recon_).all() or recon_.abs().max() < 1e-6:
                        continue

                    si = si_sdr(recon_, clean_).item()
                    si_sdr_sum += si
                    n_audio += 1

                    pq_recon = pesq_wb_16k(clean_, recon_, sr=sr)
                    if np.isfinite(pq_recon):
                        pesq_recon_sum += pq_recon
                        pesq_recon_n   += 1

                    if "lossy_wav" in test_batch:
                        lossy_wav = test_batch["lossy_wav"][i].cpu()
                        T2 = min(lossy_wav.numel(), T)
                        pq_lossy = pesq_wb_16k(clean_[:T2], lossy_wav[:T2], sr=sr)
                        if np.isfinite(pq_lossy):
                            pesq_lossy_sum += pq_lossy
                            pesq_lossy_n   += 1

                    if dnsmos_metric is not None:
                        try:
                            dnsmos_recon_sum += dnsmos_score_one(dnsmos_metric, recon_)
                            dnsmos_clean_sum += dnsmos_score_one(dnsmos_metric, clean_)
                            if "lossy_wav" in test_batch:
                                lossy_w = test_batch["lossy_wav"][i].cpu()
                                dnsmos_lossy_sum += dnsmos_score_one(dnsmos_metric, lossy_w[:T])
                            dnsmos_n += 1
                        except Exception:
                            pass

                    if recon_dir:
                        wav_path  = os.path.join(recon_dir, f"{item_id}_recon.wav")
                        spec_path = os.path.join(recon_dir, f"{item_id}_spec.png")
                        sf.write(wav_path, recon_.numpy(), sr)
                        if "clean_ri" in test_batch:
                            final_ri_norm = orig_ri * mask_i + imputed_ri * (1.0 - mask_i)
                            try:
                                _save_spectrogram_figure(
                                    lossy_ri=orig_ri.cpu(),
                                    clean_ri=test_batch["clean_ri"][i].cpu(),
                                    recon_ri=final_ri_norm.cpu(),
                                    mask=mask_i.cpu(),
                                    save_path=spec_path,
                                    sr=sr, hop_length=hop_length, item_id=item_id,
                                )
                            except Exception:
                                pass

                it.set_postfix(
                    si_sdr=f"{si_sdr_sum / max(n_audio, 1):.2f}",
                    pesq=f"{pesq_recon_sum / max(pesq_recon_n, 1):.3f}",
                )

    # ---- итоговый отчёт ----
    print(f"\n=== Evaluate: {n_audio} файлов ===")
    if n_audio > 0:
        print(f"  SI-SDR avg:      {si_sdr_sum / n_audio:.3f} dB")
    if pesq_recon_n > 0:
        print(f"  PESQ(recon) avg: {pesq_recon_sum / pesq_recon_n:.3f}  (n={pesq_recon_n})")
    if pesq_lossy_n > 0:
        print(f"  PESQ(lossy) avg: {pesq_lossy_sum / pesq_lossy_n:.3f}  (n={pesq_lossy_n})")
    if pesq_recon_n > 0 and pesq_lossy_n > 0:
        delta = pesq_recon_sum / pesq_recon_n - pesq_lossy_sum / pesq_lossy_n
        print(f"  ΔPESQ:           {delta:+.3f}")
    if dnsmos_n > 0:
        print(f"  DNSMOS(clean):   {dnsmos_clean_sum / dnsmos_n:.3f}")
        print(f"  DNSMOS(lossy):   {dnsmos_lossy_sum / dnsmos_n:.3f}")
        print(f"  DNSMOS(recon):   {dnsmos_recon_sum / dnsmos_n:.3f}")
        print(f"  ΔDNSMOS:         {(dnsmos_recon_sum - dnsmos_lossy_sum) / dnsmos_n:+.3f}")

    return {
        "si_sdr": si_sdr_sum / max(n_audio, 1),
        "pesq_recon": pesq_recon_sum / max(pesq_recon_n, 1),
        "pesq_lossy": pesq_lossy_sum / max(pesq_lossy_n, 1),
        "dnsmos_recon": dnsmos_recon_sum / max(dnsmos_n, 1),
        "n_audio": n_audio,
    }
