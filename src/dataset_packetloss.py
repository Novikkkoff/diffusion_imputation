import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import random
class PacketLossDataset(Dataset):
    """
    Возвращает:
      observed_data: (L, K) float32  — lossy maglog с зануленными пропусками
      observed_mask: (L, K) float32  — 1 где наблюдается, 0 где потеря
      gt_mask:       (L, K) float32  — по умолчанию = observed_mask
      timepoints:    (L,)   int64    — 0..L-1
      lossy_maglog_raw: (L,K) float32 — lossy maglog ДО зануления
      lossy_phase:      (L,K) float32 — фаза lossy STFT
      item_id: string
      (опц) clean_data: (L, K) float32 — clean maglog
    """
    def __init__(
        self,
        clean_dir: str,
        lossy_dir: str,
        traces_dir: str,
        use_index_list=None,
        sr: int = 16000,
        n_fft: int = 512,
        hop_length: int = 320,
        win_length: int = 512,
        n_freq: int = 256,
        L: int = 500,
        missing_ratio: float = 0.0,
        seed: int = 0,
        return_clean: bool = False,
        return_wav:bool = False
    ):
        self.clean_dir = clean_dir
        self.lossy_dir = lossy_dir
        self.traces_dir = traces_dir

        self.sr = sr
        self.n_fft = n_fft
        self.hop = hop_length
        self.win = win_length
        self.K = n_freq
        self.L = L
        self.missing_ratio = float(missing_ratio)
        self.rng = np.random.default_rng(seed)
        self.return_clean = return_clean
        self.return_wav = return_wav
        self.wav_len = 160000
        # id по lossy_dir
        ids = sorted([os.path.splitext(f)[0] for f in os.listdir(lossy_dir) if f.endswith(".wav")])

        # оставляем только те, где есть все три файла
        valid_ids = []
        for _id in ids:
            c = os.path.join(clean_dir, f"{_id}.wav")
            l = os.path.join(lossy_dir, f"{_id}.wav")
            t = os.path.join(traces_dir, f"{_id}_is_lost.txt")
            if os.path.exists(c) and os.path.exists(l) and os.path.exists(t):
                valid_ids.append(_id)

        if len(valid_ids) == 0:
            raise ValueError("No valid triplets (clean/lossy/trace) found. Check paths & naming.")

        self.ids = valid_ids

        if use_index_list is None:
            self.use_index_list = np.arange(len(self.ids))
        else:
            self.use_index_list = np.array(use_index_list, dtype=int)

        self.window = torch.hann_window(self.win)  # CPU окно

    def __len__(self):
        return len(self.use_index_list)

    def _load_wav_mono(self, path: str) -> torch.Tensor:
        y, file_sr = sf.read(path)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if file_sr != self.sr:
            raise ValueError(f"Resample needed for {path}: file_sr={file_sr}, expected {self.sr}")
        return torch.tensor(y, dtype=torch.float32)


    # ----------------------------
    # 1) STFT -> (Re, Im) тензор
    # ----------------------------
    def _stft_ri(self, y: torch.Tensor) -> torch.Tensor:
        """
        y: (T,) float32 torch
        return: X_ri (2, F, N) float32 torch, где F = n_fft//2 + 1
        """
        if y.dim() != 1:
            y = y.view(-1)

        X = torch.stft(
            y,
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.win,
            window=self.window.to(y.device),
            center=False,
            return_complex=True,
        )  # (F, N), complex

        # НЕ режем частоты: оставляем все F = n_fft//2 + 1 (для 512 это 257)
        X_ri = torch.stack([X.real, X.imag], dim=0)  # (2, F, N)
        return X_ri.float()


    # ----------------------------------------------------
    # 2) Маска "по времени (семплы)" -> "по кадрам STFT"
    # ----------------------------------------------------
    def _time_mask_to_frame_mask(self, time_mask: torch.Tensor, num_frames: int) -> torch.Tensor:
        """
        time_mask: (T,) 1=известно, 0=дырка (в семплах)
        num_frames: N (число кадров STFT)
        return: M (1, F, N) float32, одинаковая по всем частотам
        """

        T = time_mask.numel()
        win = self.win
        hop = self.hop

        # порог: если более X% окна попало в дырку -> кадр считаем неизвестным
        # для речи с 0.1с дырками хорошо работает 0.3..0.5
        lost_frac_thr = getattr(self, "lost_frac_thr", 0.15)

        frame_mask = torch.ones((num_frames,), dtype=torch.float32)

        for n in range(num_frames):
            start = n * hop
            end = start + win
            if start >= T:
                frame_mask[n] = 1.0
                continue
            if end > T:
                # если окно выходит за пределы (на хвосте) — добиваем "известными"
                seg = F.pad(time_mask[start:], (0, end - T), value=1.0)
            else:
                seg = time_mask[start:end]

            # доля потерянных семплов в окне
            lost_frac = 1.0 - seg.mean()
            frame_mask[n] = 0.0 if lost_frac > lost_frac_thr else 1.0

        # расширяем по частоте: (1, F, N)
        F_bins = self.n_fft // 2 + 1
        M = frame_mask.view(1, 1, num_frames).expand(1, F_bins, num_frames).contiguous()
        return M


    # ----------------------------------------------------
    # 3) Загрузка trace -> time_mask (семплы)
    # ----------------------------------------------------
    def _load_trace_time_mask(self, trace_path: str, T: int) -> torch.Tensor:
        """
        Возвращает time_mask длины T: 1=известно, 0=дырка.

        Поддерживаем 2 популярных формата trace:
        файл содержит 0/1 по семплам (или по отсчётам) -> обрежем/допаддим до T
        """
        txt = open(trace_path, "r", encoding="utf-8").read().strip()
        if len(txt) == 0:
            return torch.ones((T,), dtype=torch.float32)
        tokens = txt.replace(",", " ").split()
        is_binary_vector = all(t in ("0", "1", "0.0", "1.0") for t in tokens) and len(tokens) > 10
        if is_binary_vector:
            arr = torch.tensor([float(t) for t in tokens], dtype=torch.float32)
            arr = 1.0 - arr
            expansion_factor = T // len(arr)
            new_arr = arr.repeat_interleave(expansion_factor)
            new_arr = new_arr[:T]
            return new_arr
        else:
            return torch.ones((T,), dtype=torch.float32)

    # ----------------------------------------------------
    # 4) Синхронный crop/pad clean/lossy/mask до 10 сек
    # ----------------------------------------------------
    def _crop_pad_triplet(self, clean_wav, lossy_wav, time_mask, target_T: int):
        """
        Делает одинаковый кроп/паддинг для clean/lossy/mask.
        Паддинг wav -> 0.0
        Паддинг mask -> 1.0 (известно), чтобы модель не "дорисовывала" пустоту
        """
        clean_wav = clean_wav.view(-1)
        lossy_wav = lossy_wav.view(-1)
        time_mask = time_mask.view(-1)

        T = min(clean_wav.numel(), lossy_wav.numel(), time_mask.numel())
        clean_wav = clean_wav[:T]
        lossy_wav = lossy_wav[:T]
        time_mask = time_mask[:T]
        print('clean_wav.numel(), lossy_wav.numel(), time_mask.numel()', clean_wav.numel(), lossy_wav.numel(), time_mask.numel(), target_T)
        if T >= target_T:
            # простой рандомный кроп
            start = random.randint(0, T - target_T)
            clean_wav = clean_wav[start:start + target_T]
            lossy_wav = lossy_wav[start:start + target_T]
            time_mask = time_mask[start:start + target_T]
        else:
            pad = target_T - T
            clean_wav = F.pad(clean_wav, (0, pad), value=0.0)
            lossy_wav = F.pad(lossy_wav, (0, pad), value=0.0)
            time_mask = F.pad(time_mask.float(), (0, pad), value=1.0)

        return clean_wav, lossy_wav, time_mask


    # ----------------------------------------------------
    # 5) Правильный __getitem__ под inpainting diffusion
    # ----------------------------------------------------
    def __getitem__(self, idx):
        file_idx = self.use_index_list[idx]
        _id = self.ids[file_idx]

        clean_path = os.path.join(self.clean_dir, f"{_id}.wav")
        lossy_path = os.path.join(self.lossy_dir, f"{_id}.wav")
        trace_path = os.path.join(self.traces_dir, f"{_id}_is_lost.txt")

        # wav
        y_lossy = self._load_wav_mono(lossy_path)  # torch (T,)
        if y_lossy.dim() != 1:
            y_lossy = y_lossy.view(-1)

        # для обучения таргет лучше брать clean (если он есть)
        y_clean = self._load_wav_mono(clean_path)
        if y_clean.dim() != 1:
            y_clean = y_clean.view(-1)

        # выравниваем длины (чтобы маска/стфт совпали)
        T = min(y_lossy.numel(), y_clean.numel())
        y_lossy = y_lossy[:T]
        y_clean = y_clean[:T]

        # time-mask (семплы): 1 known, 0 hole (специально инвертируем значения для модели)
        time_mask = self._load_trace_time_mask(trace_path, T)  # (T,)
        print('len(time_mask), len(y_lossy)',len(time_mask), len(y_lossy))
        # фиксируем длительность ~10 секунд
        target_T = self.L * self.hop

        y_clean, y_lossy, time_mask = self._crop_pad_triplet(y_clean, y_lossy, time_mask, target_T)

        # STFT (Re/Im)
        X_lossy = self._stft_ri(y_lossy)  # (2, F, N) 2-комплексные параметры, F-частоты, N-временная ось(10 порезанных секунд аудио на куски)
        X_clean = self._stft_ri(y_clean)  # (2, F, N)

        # фиксируем число кадров по времени (self.L) — как и раньше, только теперь N
        N = X_clean.shape[-1]
        print('N', N, clean_path)
        # if N < self.L:
        #     padN = self.L - N
        #     X_lossy = F.pad(X_lossy, (0, padN))  # pad по последней оси
        #     X_clean = F.pad(X_clean, (0, padN))
        #     N = self.L
        # else:
        #     X_lossy = X_lossy[..., :self.L]
        #     X_clean = X_clean[..., :self.L]
        #     N = self.L

        # маска по кадрам STFT (1, F, N)
        M = self._time_mask_to_frame_mask(time_mask, num_frames=N)
        
        # observed = известная часть lossy (чтобы обнулить все как надо после преобразования Фурье)
        observed_data = X_lossy * M  # (2,F,N) * (1,F,N)
        print('observed_data.shape', observed_data.shape)
        out = {
            "observed_data": observed_data.float(),  # (2, F, L)
            "observed_mask": M.float(),              # (1, F, L) 1 known / 0 hole
            "gt_mask": M.float(),                    # без missing_ratio — одинаковая
            "timepoints": torch.arange(N, dtype=torch.long),

            "clean_ri": X_clean.float(),             # таргет
            "item_id": _id,
        }

        out["clean_wav"] = y_clean
        out["lossy_wav"] = y_lossy
        out["time_mask"] = time_mask.float()

        return out
    

def get_packet_loss_dataloaders(
    clean_dir: str,
    lossy_dir: str,
    traces_dir: str,
    trace_info_path: str = None,  # можно не использовать, если имена 1-в-1
    batch_size: int = 8,
    seed: int = 0,
    n_folds: int = 5,
    fold_idx: int = 0,
    missing_ratio: float = 0.0,
    sr: int = 16000,
    n_fft: int = 512,
    hop_length: int = 320,
    win_length: int = 512,
    K: int = 256,
    L: int = 500,
    num_workers: int = 2,
    return_clean: bool = False,
):
    # создаем полный датасет, чтобы узнать длину/индексы
    full = PacketLossDataset(
        clean_dir=clean_dir,
        lossy_dir=lossy_dir,
        traces_dir=traces_dir,
        use_index_list=None,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_freq=K,
        L=L,
        missing_ratio=0.0,      # split без доп. маскирования
        seed=seed,
        return_clean=return_clean,
    )

    n = len(full)
    ind = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(ind)

    fold_size = n // n_folds
    start = fold_idx * fold_size
    end = n if fold_idx == (n_folds - 1) else (fold_idx + 1) * fold_size

    test_index = ind[start:end]
    remain = np.delete(ind, np.arange(start, end))

    n_train = int(len(remain) * 0.7)
    train_index = remain[:n_train]
    valid_index = remain[n_train:]

    train_ds = PacketLossDataset(
        clean_dir, lossy_dir, traces_dir,
        use_index_list=train_index,
        sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        n_freq=K, L=L,
        missing_ratio=missing_ratio,  # тут можно включать доп. скрытие
        seed=seed,
        return_clean=return_clean,
        return_wav = False
    )
    valid_ds = PacketLossDataset(
        clean_dir, lossy_dir, traces_dir,
        use_index_list=valid_index,
        sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        n_freq=K, L=L,
        missing_ratio=0.1,       # обычно валид без доп. скрытия
        seed=seed + 1,
        return_clean=return_clean,
        return_wav = True
    )
    test_ds = PacketLossDataset(
        clean_dir, lossy_dir, traces_dir,
        use_index_list=test_index,
        sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        n_freq=K, L=L,
        missing_ratio=0.1,
        seed=seed + 2,
        return_clean=return_clean,
        return_wav = True
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, drop_last=False)

    print(f"[PacketLoss] total={n} train={len(train_ds)} valid={len(valid_ds)} test={len(test_ds)}")
    return train_loader, valid_loader, test_loader