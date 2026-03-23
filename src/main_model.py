import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diff_models import diff_CSDI


class CSDI_base(nn.Module):
    """
    Базовый класс диффузионной модели для inpainting.

    Формат данных (после process_data):
        observed_data : (B, 2, F, L)   — Re/Im спектрограмма lossy, дырки зануленны
        observed_mask : (B, 1, F, L)   — 1 known / 0 hole
        cond_mask     : (B, 1, F, L)   — то же или подмаска для обучения
        observed_tp   : (B, L)         — временные позиции кадров 0..L-1

    Внутри модели:
        K = F (число частотных бинов, 257)
        2 комплексных канала разворачиваются в inputdim для diff_CSDI
    """

    def __init__(self, target_dim, config, device):
        super().__init__()
        self.device = device
        # target_dim = число частотных бинов F (257 для n_fft=512)
        self.target_dim = target_dim

        self.emb_time_dim    = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy  = config["model"]["target_strategy"]

        # side_info dim: time_emb + feature_emb [+ 1 для conditional mask]
        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if not self.is_unconditional:
            self.emb_total_dim += 1  # канал маски

        # эмбеддинг по частотной оси (F бинов)
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim,
            embedding_dim=self.emb_feature_dim,
        )

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        # unconditional: 2 канала (Re/Im)
        # conditional:   4 канала (Re_cond, Im_cond, Re_noisy, Im_noisy)
        input_dim = 2 if self.is_unconditional else 4
        self.diffmodel = diff_CSDI(config_diff, input_dim)

        # параметры шумового расписания
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = (
                np.linspace(
                    config_diff["beta_start"] ** 0.5,
                    config_diff["beta_end"] ** 0.5,
                    self.num_steps,
                )
                ** 2
            )
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )
        else:
            raise ValueError(f"Unknown schedule: {config_diff['schedule']}")

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        # (num_steps, 1, 1)  — для broadcast по (B, K, L)
        self.alpha_torch = (
            torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)
        )

    # ------------------------------------------------------------------
    # Вспомогательные методы для масок
    # ------------------------------------------------------------------

    def get_randmask(self, observed_mask):
        """Случайное маскирование части known-значений."""
        # observed_mask: (B, 1, F, L)
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        B = observed_mask.shape[0]
        rand_flat = rand_for_mask.reshape(B, -1)
        for i in range(B):
            sample_ratio = np.random.rand()
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            if num_masked > 0:
                topk_idx = rand_flat[i].topk(num_masked).indices
                rand_flat[i][topk_idx] = -1
        cond_mask = (rand_flat > 0).reshape(observed_mask.shape).float()
        return cond_mask

    def get_hist_mask(self, observed_mask, for_pattern_mask=None):
        """Историческое маскирование (берём паттерн из другого сэмпла)."""
        if for_pattern_mask is None:
            for_pattern_mask = observed_mask
        if self.target_strategy == "mix":
            rand_mask = self.get_randmask(observed_mask)
        cond_mask = observed_mask.clone()
        for i in range(len(cond_mask)):
            if self.target_strategy == "mix" and np.random.rand() > 0.5:
                cond_mask[i] = rand_mask[i]
            else:
                cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1]
        return cond_mask

    # ------------------------------------------------------------------
    # Side information
    # ------------------------------------------------------------------

    def time_embedding(self, pos, d_model=128):
        """
        pos: (B, L) float
        return: (B, L, d_model)
        """
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model, device=self.device)
        position = pos.unsqueeze(2)  # (B, L, 1)
        div_term = 1.0 / torch.pow(
            10000.0,
            torch.arange(0, d_model, 2, device=self.device) / d_model,
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_side_info(self, observed_tp, cond_mask):
        """
        observed_tp : (B, L)
        cond_mask   : (B, 1, F, L)

        Возвращает side_info: (B, side_dim, F, L)
        """
        B, _, F, L = cond_mask.shape

        # временной эмбеддинг: (B, L, emb_time) -> разворачиваем по F
        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B, L, emb_t)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, F, -1)        # (B, L, F, emb_t)

        # частотный эмбеддинг: (F, emb_f) -> разворачиваем по B, L
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim, device=self.device)
        )  # (F, emb_f)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)  # (B, L, F, emb_f)

        # конкатенируем: (B, L, F, emb_total)
        side_info = torch.cat([time_embed, feature_embed], dim=-1)
        # -> (B, emb_total, F, L)
        side_info = side_info.permute(0, 3, 2, 1)

        # нормализация для стабильности обучения
        side_info = (side_info - side_info.mean()) / (side_info.std() + 1e-8)

        if not self.is_unconditional:
            # cond_mask уже (B, 1, F, L) — добавляем как дополнительный канал
            side_info = torch.cat([side_info, cond_mask], dim=1)

        return side_info

    # ------------------------------------------------------------------
    # Формирование входа для диффузионной модели
    # ------------------------------------------------------------------

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        """
        noisy_data    : (B, 2, F, L)
        observed_data : (B, 2, F, L)
        cond_mask     : (B, 1, F, L)

        Возвращает:
            unconditional -> (B, 2, F, L)
            conditional   -> (B, 4, F, L)  [cond_Re, cond_Im, noisy_Re, noisy_Im]
        """
        if self.is_unconditional:
            return noisy_data  # (B, 2, F, L)
        else:
            # маскируем known-часть из observed и hole-часть из noisy
            cond_obs    = cond_mask * observed_data          # (B, 2, F, L)
            noisy_target = (1 - cond_mask) * noisy_data     # (B, 2, F, L)
            return torch.cat([cond_obs, noisy_target], dim=1)  # (B, 4, F, L)

    # ------------------------------------------------------------------
    # Потери
    # ------------------------------------------------------------------

    def calc_loss(self, observed_data, cond_mask, observed_mask, side_info, is_train, set_t=-1):
        """
        observed_data : (B, 2, F, L)
        cond_mask     : (B, 1, F, L)
        observed_mask : (B, 1, F, L)
        """
        B = observed_data.shape[0]

        if is_train != 1:
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)

        current_alpha = self.alpha_torch[t]  # (B, 1, 1) -> broadcast на (B, 2, F, L)
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise

        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)

        # diffmodel возвращает (B, 2, F, L) — предсказанный шум по обоим каналам
        predicted = self.diffmodel(total_input, side_info, t)

        # считаем loss только на «дырках» (target_mask = 1 где нужно восстановить)
        target_mask = observed_mask - cond_mask  # (B, 1, F, L), broadcast на (B, 2, F, L)
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum() * 2  # *2 т.к. Re+Im
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        return loss

    def calc_loss_valid(self, observed_data, cond_mask, observed_mask, side_info, is_train):
        loss_sum = 0.0
        for t in range(self.num_steps):
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, side_info, is_train, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    # ------------------------------------------------------------------
    # Сэмплирование (обратная диффузия)
    # ------------------------------------------------------------------

    def impute(self, observed_data, cond_mask, side_info, n_samples):
        """
        observed_data : (B, 2, F, L)
        cond_mask     : (B, 1, F, L)
        Возвращает   : (B, n_samples, 2, F, L)
        """
        B, C, F, L = observed_data.shape
        imputed_samples = torch.zeros(B, n_samples, C, F, L, device=self.device)

        for i in range(n_samples):
            if self.is_unconditional:
                # зашумляем известную часть по расписанию
                noisy_obs = observed_data.clone()
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = (
                        self.alpha_hat[t] ** 0.5 * noisy_obs
                        + self.beta[t] ** 0.5 * noise
                    )
                    noisy_cond_history.append(noisy_obs * cond_mask)

            current_sample = torch.randn_like(observed_data)  # (B, 2, F, L)

            for t in range(self.num_steps - 1, -1, -1):
                if self.is_unconditional:
                    diff_input = (
                        cond_mask * noisy_cond_history[t]
                        + (1.0 - cond_mask) * current_sample
                    )
                else:
                    cond_obs     = cond_mask * observed_data
                    noisy_target = (1 - cond_mask) * current_sample
                    diff_input   = torch.cat([cond_obs, noisy_target], dim=1)  # (B, 4, F, L)

                predicted = self.diffmodel(
                    diff_input, side_info, torch.tensor([t], device=self.device)
                )  # (B, 2, F, L)

                coeff1 = 1.0 / self.alpha_hat[t] ** 0.5
                coeff2 = (1.0 - self.alpha_hat[t]) / (1.0 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    current_sample = current_sample + sigma * torch.randn_like(current_sample)

            imputed_samples[:, i] = current_sample.detach()

        return imputed_samples  # (B, n_samples, 2, F, L)

    # ------------------------------------------------------------------
    # forward / evaluate
    # ------------------------------------------------------------------

    def forward(self, batch, is_train=1):
        observed_data, observed_mask, observed_tp, gt_mask, for_pattern_mask = self.process_data(batch)

        if is_train == 0:
            cond_mask = gt_mask
        elif self.target_strategy != "random":
            cond_mask = self.get_hist_mask(observed_mask, for_pattern_mask=for_pattern_mask)
        else:
            cond_mask = self.get_randmask(observed_mask)

        side_info = self.get_side_info(observed_tp, cond_mask)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid
        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)

    def evaluate(self, batch, n_samples):
        """
        Возвращает:
            samples        : (B, n_samples, 2, F, L)
            observed_data  : (B, 2, F, L)
            target_mask    : (B, 1, F, L)   — дырки для оценки
            observed_mask  : (B, 1, F, L)
            observed_tp    : (B, L)
        """
        observed_data, observed_mask, observed_tp, gt_mask, _ = self.process_data(batch)

        with torch.no_grad():
            cond_mask   = observed_mask
            target_mask = 1.0 - observed_mask
            side_info   = self.get_side_info(observed_tp, cond_mask)
            samples     = self.impute(observed_data, cond_mask, side_info, n_samples)

        return samples, observed_data, target_mask, observed_mask, observed_tp


class CSDI_Physio(CSDI_base):
    """
    Конкретная реализация для audio inpainting (Re/Im спектрограмма).

    Батч из dataset_packetloss содержит:
        observed_data : (B, 2, F, N)  — Re/Im lossy STFT, дырки зануленны
        observed_mask : (B, 1, F, N)  — 1 known / 0 hole
        gt_mask       : (B, 1, F, N)
        timepoints    : (B, N)        — 0..N-1
    """

    def __init__(self, config, device, target_dim=257):
        # target_dim = число частотных бинов (257 для n_fft=512)
        super().__init__(target_dim, config, device)

    def process_data(self, batch):
        # (B, 2, F, N)
        observed_data = batch["observed_data"].to(self.device).float()
        # (B, 1, F, N)
        observed_mask = batch["observed_mask"].to(self.device).float()
        gt_mask       = batch["gt_mask"].to(self.device).float()
        # (B, N)
        observed_tp = batch["timepoints"].to(self.device).float()

        # Если timepoints пришёл как (N,) без батча — expand
        if observed_tp.dim() == 1:
            observed_tp = observed_tp.unsqueeze(0).expand(observed_data.shape[0], -1)

        for_pattern_mask = observed_mask

        return observed_data, observed_mask, observed_tp, gt_mask, for_pattern_mask
