import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
import os
import os
import torch
import soundfile as sf
import os
import numpy as np
import matplotlib.pyplot as plt
try:
    from torchmetrics.audio import DeepNoiseSuppressionMeanOpinionScore
except ImportError:
    try:
        from torchmetrics.audio.dnsmos import DeepNoiseSuppressionMeanOpinionScore
    except ImportError:
        DeepNoiseSuppressionMeanOpinionScore = None
def maglog_phase_to_wav(
    maglog_LK: torch.Tensor,  # (L,K)
    phase_LK: torch.Tensor,   # (L,K)
    n_fft=512, hop_length=320, win_length=512,
    sr=16000,
    length_samples=None,
):
    # (L,K) -> (K,L)
    maglog = maglog_LK.transpose(0, 1)
    phase  = phase_LK.transpose(0, 1)

    mag = torch.expm1(maglog).clamp(min=0.0)       # обратно из log1p
    X = torch.polar(mag, phase)                    # complex (K,L)

    # istft ждёт (n_fft//2+1, L) => добавим последний бин до 257
    if X.shape[0] == 256:
        last = torch.zeros(1, X.shape[1], dtype=X.dtype, device=X.device)
        X = torch.cat([X, last], dim=0)            # (257,L)

    window = torch.hann_window(win_length, device=X.device)

    y = torch.istft(
        X, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        window=window, center=True, length=length_samples
    )
    return y

def save_recon_wavs_from_batch(
    batch,
    imputed_LK: torch.Tensor,         # (B,L,K) (median)
    out_dir: str,
    sr=16000,
    n_fft=512, hop_length=320, win_length=512,
):
    os.makedirs(out_dir, exist_ok=True)

    lossy_maglog_raw = batch["lossy_maglog_raw"].to(imputed_LK.device)  # (B,L,K)
    lossy_phase      = batch["lossy_phase"].to(imputed_LK.device)       # (B,L,K)
    observed_mask    = batch["observed_mask"].to(imputed_LK.device)     # (B,L,K)

    # item_id может быть списком строк (если default collate) — обработаем аккуратно
    item_ids = batch.get("item_id", None)

    for i in range(imputed_LK.shape[0]):
        final_maglog = lossy_maglog_raw[i] * observed_mask[i] + imputed_LK[i] * (1.0 - observed_mask[i])

        y = maglog_phase_to_wav(
            final_maglog, lossy_phase[i],
            n_fft=n_fft, hop_length=hop_length, win_length=win_length, sr=sr
        )

        # имя
        if item_ids is None:
            name = f"item_{i}"
        else:
            # item_ids может быть list[str] или tuple[str]
            name = str(item_ids[i])

        path = os.path.join(out_dir, f"{name}_recon.wav")
        sf.write(path, y.detach().cpu().numpy(), sr)

def get_dnsmos_metric(device=None):
    """
    DNSMOS (P.808-like overall MOS). Для 16 kHz.
    Возвращает объект torchmetrics.
    """
    # personalized=False -> стандартный режим
    metric = DeepNoiseSuppressionMeanOpinionScore(
        fs=16000,
        personalized=False,
        device=device if device is not None else "cpu",
    )
    return metric

def dnsmos_score_one(metric, wav: torch.Tensor) -> float:
    """
    wav: (T,) torch float tensor, 16 kHz
    Возвращает scalar float (OVRL MOS)
    """
    # DNSMOS ожидает батч: (B, T)
    x = wav.detach().float()

    # нормализация/клиппинг (не обязательно, но полезно)
    x = torch.clamp(x, -1.0, 1.0)

    if x.ndim == 1:
        x = x.unsqueeze(0)  # (1, T)

    # torchmetrics обычно возвращает tensor scalar
    out = metric(x)

    # Некоторые реализации могут вернуть dict/tuple — обработаем мягко
    if isinstance(out, torch.Tensor):
        return float(out.squeeze().detach().cpu().item())
    elif isinstance(out, dict):
        # если вдруг вернёт несколько ключей, пробуем OVRL
        for k in ["ovrl", "OVRL", "overall", "mos"]:
            if k in out:
                v = out[k]
                return float(v.squeeze().detach().cpu().item() if isinstance(v, torch.Tensor) else v)
        # fallback: первый элемент
        first_val = next(iter(out.values()))
        return float(first_val.squeeze().detach().cpu().item() if isinstance(first_val, torch.Tensor) else first_val)
    else:
        return float(out)
    
def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=20,
    foldername="",
    plot_interval=1,          # как часто обновлять график (в эпохах)
):
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)

    if foldername != "":
        os.makedirs(foldername, exist_ok=True)
        output_path = os.path.join(foldername, "model.pth")
        plot_path = os.path.join(foldername, "loss_curve.png")
        hist_path = os.path.join(foldername, "loss_history.npz")
    else:
        plot_path = None
        hist_path = None

    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    train_loss_history = []
    valid_loss_history = []  # будет иметь элементы только на valid_epoch_interval
    valid_epoch_history = []

    best_valid_loss = 1e10

    # сколько итераций реально делать за эпоху
    itr_per_epoch = int(min(config["itr_per_epoch"], len(train_loader)))

    for epoch_no in range(config["epochs"]):
        # ---------------- TRAIN ----------------
        model.train()
        avg_loss = 0.0
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad(set_to_none=True)
                loss = model(train_batch)
                loss.backward()
                optimizer.step()

                avg_loss += loss.item()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )

                if batch_no >= itr_per_epoch:
                    break

        lr_scheduler.step()

        train_epoch_loss = avg_loss / max(batch_no, 1)
        train_loss_history.append(train_epoch_loss)

        # ---------------- VALID ----------------
        valid_epoch_loss = None
        if valid_loader is not None and ((epoch_no + 1) % valid_epoch_interval == 0):
            model.eval()
            avg_loss_valid = 0.0
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for v_no, valid_batch in enumerate(it, start=1):
                        vloss = model(valid_batch, is_train=0)
                        avg_loss_valid += vloss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / v_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )

            valid_epoch_loss = avg_loss_valid / max(v_no, 1)
            valid_loss_history.append(valid_epoch_loss)
            valid_epoch_history.append(epoch_no + 1)

            if best_valid_loss < valid_epoch_loss:
                best_valid_loss = valid_epoch_loss
                print(f"\n✅ best valid loss updated to {best_valid_loss:.6f} at epoch {epoch_no+1}")

        # ---------------- SAVE MODEL ----------------
        if (epoch_no + 1) % 5 == 0 and foldername != "":
            output_path_epoch = os.path.join(foldername, f"model_epoch_{epoch_no+1}.pth")
            torch.save(model.state_dict(), output_path_epoch)
            print(f"✅ Модель сохранена: {output_path_epoch}")

        # ---------------- PLOT LOSS ----------------
        if plot_path is not None and ((epoch_no + 1) % plot_interval == 0 or epoch_no == config["epochs"] - 1):
            plt.figure()
            plt.plot(np.arange(1, len(train_loss_history) + 1), train_loss_history, label="train")

            if len(valid_loss_history) > 0:
                plt.plot(valid_epoch_history, valid_loss_history, label="valid")
                plt.xlabel("epoch")
                plt.ylabel("loss")
                plt.title("Training / Validation Loss")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(plot_path)
                plt.close()

                # сохраним историю тоже
                np.savez(
                    hist_path,
                    train_loss=np.array(train_loss_history, dtype=np.float32),
                    valid_loss=np.array(valid_loss_history, dtype=np.float32),
                    valid_epochs=np.array(valid_epoch_history, dtype=np.int32),
                )

        if foldername != "":
            torch.save(model.state_dict(), output_path)
            print(f"✅ Final model saved: {output_path}")
            if plot_path is not None:
                print(f"📈 Loss plot saved: {plot_path}")

def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )


def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))


def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):

    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)

def calc_quantile_CRPS_sum(target, forecast, eval_points, mean_scaler, scaler):

    eval_points = eval_points.mean(-1)
    target = target * scaler + mean_scaler
    target = target.sum(-1)
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = torch.quantile(forecast.sum(-1),quantiles[i],dim=1)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)

def evaluate(model, test_loader, nsample=100, scaler=1, mean_scaler=0, foldername=""):
    import torch
    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0
        si_sdr_sum = 0.0
        pesq_sum = 0.0
        n_audio = 0
        dnsmos_clean_sum = 0.0
        dnsmos_lossy_sum = 0.0
        dnsmos_recon_sum = 0.0
        dnsmos_n = 0
        pesq_lossy_sum = 0.0
        pesq_recon_sum = 0.0
        pesq_recon_count = 0
        pesq_lossy_count = 0
        # инициализация DNSMOS 1 раз
        dnsmos_metric = None
        try:
            dnsmos_metric = get_dnsmos_metric(device="cpu")  # DNSMOS обычно ок на CPU
        except Exception as e:
            print(f"⚠️ DNSMOS init failed: {e}")
            dnsmos_metric = None
        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []

        max_batches = 2
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                if batch_no > max_batches:
                    break

                output = model.evaluate(test_batch, nsample)
                samples, c_target, eval_points, observed_points, observed_time = output

                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)   # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)

                samples_median = samples.median(dim=1) # values: (B,L,K)
                # imputed median: (B,L,K)
                imputed_LK = samples_median.values

                # recon wav для каждого i в батче
                for i in range(imputed_LK.shape[0]):
                    final_maglog = (
                        test_batch["lossy_maglog_raw"][i].to(imputed_LK.device) * test_batch["observed_mask"][i].to(imputed_LK.device)
                        + imputed_LK[i] * (1.0 - test_batch["observed_mask"][i].to(imputed_LK.device))
                    )

                    recon = maglog_phase_to_wav(
                        final_maglog,                         # (L,K)
                        test_batch["lossy_phase"][i].to(imputed_LK.device),
                        n_fft=512, hop_length=320, win_length=512, sr=16000
                    )  # (T,)

                    clean = test_batch["clean_wav"][i].to(recon.device)  # (T_clean,)

                    # обрезаем до одной длины
                    T = min(recon.numel(), clean.numel())
                    recon_ = recon[:T]
                    clean_ = clean[:T]

                    # SI-SDR
                    si = si_sdr(recon_, clean_).item()

                    # PESQ
                    def has_voice_activity(audio, threshold=0.01):
                        """Проверяет, есть ли речь в сигнале"""
                        energy = torch.mean(audio ** 2)
                        return energy > threshold

                    # В цикле оценки:
                    if has_voice_activity(clean_) and has_voice_activity(recon_):
                        pq = pesq_wb_16k(clean_, recon_, sr=16000)
                    else:
                        pq = 0.0  # или пропускаем
                        print(f"⚠️ Low energy detected, skipping PESQ")
                    if np.isfinite(pq):
                        pesq_recon_sum += pq
                        pesq_recon_count += 1
                    si_sdr_sum += si
                    pesq_sum += pq
                    n_audio += 1

                    # --- DNSMOS ---
                    if dnsmos_metric is not None:
                        try:
                            # clean
                            d_clean = dnsmos_score_one(dnsmos_metric, clean_)

                            # lossy: если в batch есть raw lossy_wav
                            if "lossy_wav" in test_batch:
                                lossy = test_batch["lossy_wav"][i].to(recon.device)
                                T_lossy = min(lossy.numel(), clean_.numel())
                                lossy_ = lossy[:T_lossy]
                                # чтобы длины совпали для сравнения с clean_/recon_, можно привести к T
                                T2 = min(lossy_.numel(), T)
                                lossy_ = lossy_[:T2]
                                clean_for_lossy = clean_[:T2]
                                recon_for_dnsmos = recon_[:T2]
                            else:
                                # fallback: собираем lossy из maglog+phase
                                lossy_rebuilt = maglog_phase_to_wav(
                                    test_batch["lossy_maglog_raw"][i].to(imputed_LK.device),
                                    test_batch["lossy_phase"][i].to(imputed_LK.device),
                                    n_fft=512, hop_length=320, win_length=512, sr=16000
                                )
                                T2 = min(lossy_rebuilt.numel(), clean_.numel())
                                lossy_ = lossy_rebuilt[:T2]
                                clean_for_lossy = clean_[:T2]
                                recon_for_dnsmos = recon_[:T2]

                            d_lossy = dnsmos_score_one(dnsmos_metric, lossy_)
                            d_recon = dnsmos_score_one(dnsmos_metric, recon_for_dnsmos)

                            dnsmos_clean_sum += d_clean
                            dnsmos_lossy_sum += d_lossy
                            dnsmos_recon_sum += d_recon
                            dnsmos_n += 1

                        except Exception as e:
                            # не валим весь evaluate из-за DNSMOS
                            print(f"⚠️ DNSMOS failed for item {i}: {e}")

                        # lossy baseline (если нет lossy_wav, используй lossy_rebuilt)
                        if "lossy_wav" in test_batch:
                            lossy = test_batch["lossy_wav"][i].to(recon.device)
                            T2 = min(lossy.numel(), clean.numel())
                            lossy_ = lossy[:T2]
                            clean_lossy_ = clean[:T2]
                        else:
                            lossy_rebuilt = maglog_phase_to_wav(
                                test_batch["lossy_maglog_raw"][i].to(imputed_LK.device),
                                test_batch["lossy_phase"][i].to(imputed_LK.device),
                                n_fft=512, hop_length=320, win_length=512, sr=16000
                            )
                            T2 = min(lossy_rebuilt.numel(), clean.numel())
                            lossy_ = lossy_rebuilt[:T2]
                            clean_lossy_ = clean[:T2]

                        pq_lossy = pesq_wb_16k(clean_lossy_, lossy_, sr=16000)
                        pesq_lossy_sum += pq_lossy
                # ===================== NEW: save wav recon =====================
                if foldername != "" and "lossy_phase" in test_batch and "lossy_maglog_raw" in test_batch:
                    recon_dir = os.path.join(foldername, "recon_wavs")
                    save_recon_wavs_from_batch(
                        test_batch,
                        samples_median.values,       # (B,L,K)
                        out_dir=recon_dir,
                        sr=16000,
                        n_fft=512,
                        hop_length=320,
                        win_length=512,
                    )
                # ===============================================================

                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)

                mse_current = (((samples_median.values - c_target) * eval_points) **2) * (scaler ** 2)
                mae_current = (torch.abs((samples_median.values - c_target) * eval_points)) * scaler

                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                evalpoints_total += eval_points.sum().item()

                it.set_postfix(
                    ordered_dict={
                        "rmse_total": np.sqrt(mse_total / evalpoints_total),
                        "mae_total": mae_total / evalpoints_total,
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )
        print('si_sdr_sum', si_sdr_sum, 'pesq_sum', pesq_sum, 'n_audio', n_audio)
        with open(foldername + "/generated_outputs_nsample" + str(nsample) + ".pk", "wb") as f:
            all_target = torch.cat(all_target, dim=0)
            all_evalpoint = torch.cat(all_evalpoint, dim=0)
            all_observed_point = torch.cat(all_observed_point, dim=0)
            all_observed_time = torch.cat(all_observed_time, dim=0)
            all_generated_samples = torch.cat(all_generated_samples, dim=0)

            pickle.dump(
                [
                    all_generated_samples,
                    all_target,
                    all_evalpoint,
                    all_observed_point,
                    all_observed_time,
                    scaler,
                    mean_scaler,
                ],
                f,
            )

        CRPS = calc_quantile_CRPS(all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler)
        CRPS_sum = calc_quantile_CRPS_sum(all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler)
        with open(foldername + "/result_nsample" + str(nsample) + ".pk", "wb") as f:
                    pickle.dump(
                        [
                            np.sqrt(mse_total / evalpoints_total),
                            mae_total / evalpoints_total,
                            CRPS,
                        ],
                        f,
                    )
                    print("RMSE:", np.sqrt(mse_total / evalpoints_total))
                    print("MAE:", mae_total / evalpoints_total)
                    print("CRPS:", CRPS)
                    print("CRPS_sum:", CRPS_sum)
        if n_audio > 0:
            print(f"SI-SDR avg: {si_sdr_sum / n_audio:.4f} dB")
            print(f"PESQ avg:   {pesq_sum / n_audio:.4f}")

        if dnsmos_n > 0:
            print(f"DNSMOS(clean) avg: {dnsmos_clean_sum / dnsmos_n:.4f}")
            print(f"DNSMOS(lossy) avg: {dnsmos_lossy_sum / dnsmos_n:.4f}")
            print(f"DNSMOS(recon) avg: {dnsmos_recon_sum / dnsmos_n:.4f}")
            print(f"ΔDNSMOS (recon-lossy): {(dnsmos_recon_sum - dnsmos_lossy_sum) / dnsmos_n:.4f}")

        print(f"PESQ(lossy) avg: {pesq_lossy_sum / n_audio:.4f}")
        print(f"PESQ(recon) avg: {pesq_sum / n_audio:.4f}")
        print(f"ΔPESQ: {(pesq_sum - pesq_lossy_sum) / n_audio:.4f}")

        import torch

def si_sdr(est: torch.Tensor, ref: torch.Tensor, eps=1e-8) -> torch.Tensor:
    """
    est, ref: (T,) float tensors
    returns scalar tensor (dB)
    """
    est = est - est.mean()
    ref = ref - ref.mean()

    # projection of est onto ref
    scale = torch.dot(est, ref) / (torch.dot(ref, ref) + eps)
    s_target = scale * ref
    e_noise = est - s_target

    ratio = (torch.dot(s_target, s_target) + eps) / (torch.dot(e_noise, e_noise) + eps)
    return 10 * torch.log10(ratio)

import numpy as np
from pesq import pesq

import numpy as np
from pesq import pesq

def pesq_wb_16k(ref_wav: torch.Tensor, deg_wav: torch.Tensor, sr=16000, min_rms=1e-4):
    """
    Возвращает float или np.nan, если PESQ нельзя посчитать.
    """
    T = min(ref_wav.numel(), deg_wav.numel())
    if T < sr // 4:  # меньше 250 мс — PESQ часто нестабилен
        print("⚠️ Too short for PESQ")
        return np.nan

    ref = ref_wav[:T].detach().cpu().numpy().astype(np.float32)
    deg = deg_wav[:T].detach().cpu().numpy().astype(np.float32)

    # NaN/Inf check
    if not np.isfinite(ref).all() or not np.isfinite(deg).all():
        print("⚠️ NaN/Inf detected, skipping PESQ")
        return np.nan

    # clip
    ref = np.clip(ref, -1.0, 1.0)
    deg = np.clip(deg, -1.0, 1.0)

    # RMS / energy check
    ref_rms = np.sqrt(np.mean(ref**2) + 1e-12)
    deg_rms = np.sqrt(np.mean(deg**  2) + 1e-12)

    if ref_rms < min_rms or deg_rms < min_rms:
        print(f"⚠️ Low energy detected, skipping PESQ (ref_rms={ref_rms:.2e}, deg_rms={deg_rms:.2e})")
        return np.nan

    try:
        return float(pesq(sr, ref, deg, "wb"))
    except Exception as e:
        print(f"⚠️ PESQ failed: {type(e).name}: {e}")
        return np.nan