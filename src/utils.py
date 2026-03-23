import os
import pickle
import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import soundfile as sf
import matplotlib.pyplot as plt

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
    hop_length: int = 160,
    win_length: int = 512,
    sr: int = 16000,
    length_samples: int = None,
) -> torch.Tensor:
    """
    ri : (2, F, L) или (F, L) complex  — Re/Im спектрограмма
         F = n_fft//2 + 1 = 257
    Возвращает: (T,) float32 waveform
    """
    if ri.shape[0] == 2:
        # (2, F, L) -> complex (F, L)
        X = torch.complex(ri[0], ri[1])
    else:
        X = ri  # уже комплексный

    window = torch.hann_window(win_length, device=X.device)
    y = torch.istft(
        X,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=False,
        length=length_samples,
    )
    return y


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

    min_rms = 1e-4
    if np.sqrt(np.mean(ref ** 2)) < min_rms or np.sqrt(np.mean(deg ** 2)) < min_rms:
        return np.nan

    try:
        return float(pesq_fn(sr, ref, deg, "wb"))
    except Exception as e:
        print(f"⚠️ PESQ failed: {e}")
        return np.nan


def get_dnsmos_metric(device=None):
    if DeepNoiseSuppressionMeanOpinionScore is None:
        return None
    return DeepNoiseSuppressionMeanOpinionScore(
        fs=16000,
        personalized=False,
        device=device or "cpu",
    )


def dnsmos_score_one(metric, wav: torch.Tensor) -> float:
    x = wav.detach().float().clamp(-1.0, 1.0)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    out = metric(x)
    if isinstance(out, torch.Tensor):
        return float(out.squeeze().item())
    if isinstance(out, dict):
        for k in ("ovrl", "OVRL", "overall", "mos"):
            if k in out:
                v = out[k]
                return float(v.squeeze().item() if isinstance(v, torch.Tensor) else v)
        first = next(iter(out.values()))
        return float(first.squeeze().item() if isinstance(first, torch.Tensor) else first)
    return float(out)


# ---------------------------------------------------------------------------
# CRPS (оставляем для совместимости, хотя для аудио менее важно)
# ---------------------------------------------------------------------------

def quantile_loss(target, forecast, q: float, eval_points) -> torch.Tensor:
    return 2.0 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast).float() - q))
    )


def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))


def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):
    target   = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler
    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    crps = 0.0
    for q in quantiles:
        q_pred = torch.cat(
            [torch.quantile(forecast[j: j + 1], q, dim=1) for j in range(len(forecast))], dim=0
        )
        crps += quantile_loss(target, q_pred, q, eval_points) / denom
    return crps.item() / len(quantiles)


# ---------------------------------------------------------------------------
# TRAIN
# ---------------------------------------------------------------------------

def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval: int = 20,
    foldername: str = "",
    plot_interval: int = 1,
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

    itr_per_epoch = int(min(config["itr_per_epoch"], len(train_loader)))

    for epoch_no in range(config["epochs"]):

        # ---- TRAIN ----
        model.train()
        avg_loss = 0.0
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0, desc=f"Epoch {epoch_no}") as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad(set_to_none=True)
                loss = model(train_batch)
                loss.backward()
                # gradient clipping для стабильности
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                avg_loss += loss.item()
                it.set_postfix(
                    avg_epoch_loss=avg_loss / batch_no,
                    epoch=epoch_no,
                    refresh=False,
                )
                if batch_no >= itr_per_epoch:
                    break

        lr_scheduler.step()
        train_epoch_loss = avg_loss / max(batch_no, 1)
        train_loss_history.append(train_epoch_loss)

        # ---- VALID ----
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0.0
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0, desc="Valid") as it:
                    for v_no, valid_batch in enumerate(it, start=1):
                        vloss = model(valid_batch, is_train=0)
                        avg_loss_valid += vloss.item()
                        it.set_postfix(
                            valid_avg_epoch_loss=avg_loss_valid / v_no,
                            epoch=epoch_no,
                            refresh=False,
                        )

            valid_epoch_loss = avg_loss_valid / max(v_no, 1)
            valid_loss_history.append(valid_epoch_loss)
            valid_epoch_history.append(epoch_no + 1)

            # ИСПРАВЛЕНО: было `<` вместо `>`
            if valid_epoch_loss < best_valid_loss:
                best_valid_loss = valid_epoch_loss
                print(f"\n✅ best valid loss updated: {best_valid_loss:.6f} at epoch {epoch_no+1}")
                if output_path:
                    torch.save(model.state_dict(), output_path.replace(".pth", "_best.pth"))

        # ---- CHECKPOINT каждые 5 эпох ----
        if (epoch_no + 1) % 5 == 0 and foldername:
            ckpt_path = os.path.join(foldername, f"model_epoch_{epoch_no+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"✅ Checkpoint saved: {ckpt_path}")

        # ---- PLOT ----
        if plot_path and ((epoch_no + 1) % plot_interval == 0 or epoch_no == config["epochs"] - 1):
            plt.figure()
            plt.plot(np.arange(1, len(train_loss_history) + 1), train_loss_history, label="train")
            if valid_loss_history:
                plt.plot(valid_epoch_history, valid_loss_history, label="valid")
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.title("Training / Validation Loss")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()

            if hist_path:
                np.savez(
                    hist_path,
                    train_loss=np.array(train_loss_history, dtype=np.float32),
                    valid_loss=np.array(valid_loss_history, dtype=np.float32),
                    valid_epochs=np.array(valid_epoch_history, dtype=np.int32),
                )

    # финальное сохранение
    if output_path:
        torch.save(model.state_dict(), output_path)
        print(f"✅ Final model saved: {output_path}")


# ---------------------------------------------------------------------------
# EVALUATE
# ---------------------------------------------------------------------------

def evaluate(model, test_loader, nsample: int = 1, scaler: float = 1.0, foldername: str = ""):
    """
    Запускает inference на test_loader, считает SI-SDR, PESQ, DNSMOS.
    Сохраняет реконструированные wav-файлы в foldername/recon_wavs/.

    Предполагаем, что батч содержит:
        observed_data : (B, 2, F, N)
        observed_mask : (B, 1, F, N)
        clean_ri      : (B, 2, F, N)
        clean_wav     : (B, T)
        lossy_wav     : (B, T)   — опционально
        item_id       : list[str]
    """
    model.eval()

    si_sdr_sum       = 0.0
    pesq_recon_sum   = 0.0
    pesq_lossy_sum   = 0.0
    n_audio          = 0

    dnsmos_clean_sum = 0.0
    dnsmos_lossy_sum = 0.0
    dnsmos_recon_sum = 0.0
    dnsmos_n         = 0

    # DNSMOS инициализируем один раз
    dnsmos_metric = None
    try:
        dnsmos_metric = get_dnsmos_metric(device="cpu")
    except Exception as e:
        print(f"⚠️ DNSMOS init failed: {e}")

    recon_dir = os.path.join(foldername, "recon_wavs") if foldername else None
    if recon_dir:
        os.makedirs(recon_dir, exist_ok=True)

    # STFT параметры — должны совпадать с датасетом
    N_FFT       = 512
    HOP_LENGTH  = 160
    WIN_LENGTH  = 512
    SR          = 16000

    with torch.no_grad():
        with tqdm(test_loader, mininterval=5.0, desc="Evaluate") as it:
            for batch_no, test_batch in enumerate(it, start=1):

                output = model.evaluate(test_batch, nsample)
                # samples: (B, nsample, 2, F, L)
                # observed_data, target_mask, observed_mask: (B, 2/1, F, L)
                samples, observed_data, target_mask, observed_mask, _ = output

                # медиана по сэмплам: (B, 2, F, L)
                # samples.median по dim=1 не работает на комплексных -> берём median по каждому каналу
                imputed = samples.median(dim=1).values  # (B, 2, F, L)

                B = imputed.shape[0]
                device = imputed.device

                for i in range(B):
                    # Собираем финальный Re/Im: в known-местах берём оригинал, в дырках — imputed
                    mask_i     = observed_mask[i]     # (1, F, L)
                    orig_ri    = observed_data[i]     # (2, F, L)
                    imputed_ri = imputed[i]            # (2, F, L)

                    final_ri = orig_ri * mask_i + imputed_ri * (1.0 - mask_i)  # (2, F, L)

                    # Реконструируем wav
                    try:
                        recon_wav = ri_to_wav(
                            final_ri.cpu(),
                            n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, sr=SR,
                        )
                    except Exception as e:
                        print(f"⚠️ ISTFT failed for item {i}: {e}")
                        continue

                    # Clean wav из батча
                    if "clean_wav" not in test_batch:
                        continue
                    clean_wav = test_batch["clean_wav"][i].to(device)

                    T = min(recon_wav.numel(), clean_wav.numel())
                    recon_ = recon_wav[:T]
                    clean_ = clean_wav[:T]

                    # SI-SDR
                    si = si_sdr(recon_, clean_).item()
                    si_sdr_sum += si

                    # PESQ recon
                    pq_recon = pesq_wb_16k(clean_, recon_, sr=SR)
                    if np.isfinite(pq_recon):
                        pesq_recon_sum += pq_recon

                    # PESQ lossy (если есть)
                    if "lossy_wav" in test_batch:
                        lossy_wav = test_batch["lossy_wav"][i].to(device)
                        T2 = min(lossy_wav.numel(), T)
                        pq_lossy = pesq_wb_16k(clean_[:T2], lossy_wav[:T2], sr=SR)
                        if np.isfinite(pq_lossy):
                            pesq_lossy_sum += pq_lossy

                    n_audio += 1

                    # DNSMOS
                    if dnsmos_metric is not None:
                        try:
                            dnsmos_recon_sum += dnsmos_score_one(dnsmos_metric, recon_)
                            dnsmos_clean_sum += dnsmos_score_one(dnsmos_metric, clean_)
                            if "lossy_wav" in test_batch:
                                lossy_w = test_batch["lossy_wav"][i].to(device)
                                dnsmos_lossy_sum += dnsmos_score_one(dnsmos_metric, lossy_w[:T])
                            dnsmos_n += 1
                        except Exception as e:
                            print(f"⚠️ DNSMOS failed for item {i}: {e}")

                    # Сохраняем wav
                    if recon_dir:
                        item_id  = test_batch.get("item_id", [f"item_{batch_no}_{i}"])[i]
                        wav_path = os.path.join(recon_dir, f"{item_id}_recon.wav")
                        sf.write(wav_path, recon_.cpu().numpy(), SR)

                it.set_postfix(
                    si_sdr=si_sdr_sum / max(n_audio, 1),
                    pesq=pesq_recon_sum / max(n_audio, 1),
                    refresh=True,
                )

    # ---- Финальные метрики ----
    if n_audio > 0:
        print(f"\n=== Результаты на {n_audio} сэмплах ===")
        print(f"SI-SDR avg:         {si_sdr_sum    / n_audio:.4f} dB")
        print(f"PESQ(recon) avg:    {pesq_recon_sum / n_audio:.4f}")
        print(f"PESQ(lossy) avg:    {pesq_lossy_sum / n_audio:.4f}")
        print(f"ΔPESQ:              {(pesq_recon_sum - pesq_lossy_sum) / n_audio:.4f}")

    if dnsmos_n > 0:
        print(f"DNSMOS(clean) avg:  {dnsmos_clean_sum / dnsmos_n:.4f}")
        print(f"DNSMOS(lossy) avg:  {dnsmos_lossy_sum / dnsmos_n:.4f}")
        print(f"DNSMOS(recon) avg:  {dnsmos_recon_sum / dnsmos_n:.4f}")
        print(f"ΔDNSMOS:            {(dnsmos_recon_sum - dnsmos_lossy_sum) / dnsmos_n:.4f}")
