import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diff_models import diff_CSDI


class CSDI_base(nn.Module):
    """
    Base class for diffusion-based inpainting of complex spectrograms.

    Data format (after process_data):
        observed_data : (B, 2, F, L)  -- Re/Im lossy spectrogram, holes zeroed out
        observed_mask : (B, 1, F, L)  -- 1=known, 0=hole
        cond_mask     : (B, 1, F, L)  -- same or sub-mask used during training
        observed_tp   : (B, L)        -- frame positions 0..L-1

    Mask convention (matches dataset_packetloss.py):
        txt file:  1=packet lost, 0=packet received
        after inversion in _load_trace_time_mask:  1=known, 0=hole
        M from _time_mask_to_frame_mask:           1=known, 0=hole
        observed_data = X_lossy * M  (holes are zeroed)
    """

    def __init__(self, target_dim, config, device):
        super().__init__()
        self.device = device
        self.target_dim = target_dim  # number of frequency bins F (257 for n_fft=512)

        self.emb_time_dim     = config["model"]["timeemb"]
        self.emb_feature_dim  = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy  = config["model"]["target_strategy"]

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if not self.is_unconditional:
            self.emb_total_dim += 1  # mask channel in side_info

        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim,
            embedding_dim=self.emb_feature_dim,
        )

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        # unconditional: 2 channels (Re, Im)
        # conditional:   4 channels (Re_cond, Im_cond, Re_noisy, Im_noisy)
        input_dim = 2 if self.is_unconditional else 4
        self.diffmodel = diff_CSDI(config_diff, input_dim)

        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = (
                np.linspace(
                    config_diff["beta_start"] ** 0.5,
                    config_diff["beta_end"] ** 0.5,
                    self.num_steps,
                ) ** 2
            )
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )
        else:
            raise ValueError(f"Unknown schedule: {config_diff['schedule']}")

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        # shape (num_steps, 1, 1) for broadcasting over (B, F, L)
        self.alpha_torch = (
            torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)
        )

    # ------------------------------------------------------------------
    # Mask helpers
    # ------------------------------------------------------------------

    def get_randmask(self, observed_mask):
        """Randomly hide a fraction of known positions."""
        # observed_mask: (B, 1, F, L),  1=known 0=hole
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
        """Historical masking: borrow hole pattern from another sample in batch."""
        if for_pattern_mask is None:
            for_pattern_mask = observed_mask
        if self.target_strategy == "mix":
            rand_mask = self.get_randmask(observed_mask)
        cond_mask = observed_mask.clone()
        for i in range(len(cond_mask)):
            if self.target_strategy == "mix" and np.random.rand() > 0.5:
                cond_mask[i] = rand_mask[i]
            else:
                # i-1 wraps to last element when i=0, which is fine
                cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1]
        return cond_mask

    # ------------------------------------------------------------------
    # Side information
    # ------------------------------------------------------------------

    def time_embedding(self, pos, d_model=128):
        """
        pos: (B, L) float
        returns: (B, L, d_model)
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
        returns side_info: (B, side_dim, F, L)
        """
        B, _, F, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)    # (B, L, emb_t)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, F, -1)          # (B, L, F, emb_t)

        feature_embed = self.embed_layer(
            torch.arange(self.target_dim, device=self.device)
        )  # (F, emb_f)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)  # (B, L, F, emb_f)

        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B, L, F, emb_total)
        side_info = side_info.permute(0, 3, 2, 1)                   # (B, emb_total, F, L)
        side_info = (side_info - side_info.mean()) / (side_info.std() + 1e-8)

        if not self.is_unconditional:
            # cond_mask: (B, 1, F, L) -- 1=known, 0=hole
            # model sees which positions are conditioned on
            side_info = torch.cat([side_info, cond_mask], dim=1)    # (B, emb_total+1, F, L)

        return side_info

    # ------------------------------------------------------------------
    # Diffmodel input assembly
    # ------------------------------------------------------------------

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        """
        noisy_data    : (B, 2, F, L)  -- x_t (noisy version of clean target)
        observed_data : (B, 2, F, L)  -- lossy Re/Im, holes already zeroed by dataset
        cond_mask     : (B, 1, F, L)  -- 1=known, 0=hole

        For conditional mode we give the model two paired channel groups:
            channels 0-1: cond_obs     = known part of lossy (holes stay 0)
            channels 2-3: noisy_target = x_t everywhere
        The model learns to predict noise at ALL positions, but loss is
        computed only at hole positions (target_mask in calc_loss).
        """
        if self.is_unconditional:
            return noisy_data  # (B, 2, F, L)
        else:
            cond_obs     = cond_mask * observed_data   # (B, 2, F, L)  holes=0
            noisy_target = noisy_data                  # (B, 2, F, L)  full noisy
            return torch.cat([cond_obs, noisy_target], dim=1)  # (B, 4, F, L)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def calc_loss(self, observed_data, clean_data, cond_mask, observed_mask, side_info, is_train, set_t=-1):
        """
        observed_data : (B, 2, F, L)  — lossy Re/Im, дырки зануленны — используется как условие
        clean_data    : (B, 2, F, L)  — чистый Re/Im — на нём строится диффузия
        cond_mask     : (B, 1, F, L)  — 1=known, 0=hole (подаётся в модель как контекст)
        observed_mask : (B, 1, F, L)  — 1=known, 0=real hole в данных

        Логика:
          - Зашумляем clean_data (а не lossy), потому что именно его хотим восстановить
          - В каналах 0-1 (cond) модель видит lossy в known-позициях (из observed_data)
          - В каналах 2-3 (noisy) модель видит x_t — зашумлённый clean
          - Loss = MSE(predicted_noise, true_noise) только на дырках (target_mask)
        """
        B = clean_data.shape[0]

        if is_train != 1:
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)

        current_alpha = self.alpha_torch[t]  # (B, 1, 1)

        # Зашумляем чистый сигнал
        noise      = torch.randn_like(clean_data)
        noisy_clean = current_alpha ** 0.5 * clean_data + (1.0 - current_alpha) ** 0.5 * noise

        # Вход в диффузионную модель: lossy-условие + зашумлённый чистый
        total_input = self.set_input_to_diffmodel(noisy_clean, observed_data, cond_mask)
        predicted   = self.diffmodel(total_input, side_info, t)  # (B, 2, F, L)

        # Loss только на реальных дырках: observed_mask=1 но cond_mask=0
        target_mask = (observed_mask - cond_mask).expand_as(noise)  # (B, 2, F, L)
        residual    = (noise - predicted) * target_mask
        num_eval    = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        return loss

    def calc_loss_valid(self, observed_data, clean_data, cond_mask, observed_mask, side_info, is_train):
        loss_sum = 0.0
        for t in range(self.num_steps):
            loss = self.calc_loss(
                observed_data, clean_data, cond_mask, observed_mask, side_info, is_train, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    # ------------------------------------------------------------------
    # Reverse diffusion sampling
    # ------------------------------------------------------------------

    def impute(self, observed_data, cond_mask, side_info, n_samples):
        """
        observed_data : (B, 2, F, L)  -- lossy Re/Im, holes zeroed
        cond_mask     : (B, 1, F, L)  -- 1=known, 0=hole
        returns       : (B, n_samples, 2, F, L)

        At each denoising step the known positions are kept from observed_data
        and only the hole positions are updated by the diffusion model,
        i.e. we do conditional sampling via replacement.
        """
        B, C, F, L = observed_data.shape
        imputed_samples = torch.zeros(B, n_samples, C, F, L, device=self.device)

        for i in range(n_samples):
            if self.is_unconditional:
                noisy_obs = observed_data.clone()
                noisy_cond_history = []
                for t_step in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = (
                        self.alpha_hat[t_step] ** 0.5 * noisy_obs
                        + self.beta[t_step] ** 0.5 * noise
                    )
                    noisy_cond_history.append(noisy_obs * cond_mask)

            # start from pure noise
            current_sample = torch.randn_like(observed_data)  # (B, 2, F, L)

            for t in range(self.num_steps - 1, -1, -1):
                if self.is_unconditional:
                    diff_input = (
                        cond_mask * noisy_cond_history[t]
                        + (1.0 - cond_mask) * current_sample
                    )
                else:
                    cond_obs     = cond_mask * observed_data
                    noisy_target = current_sample
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

                # replacement trick: paste known values back at every step
                # this enforces the conditioning and prevents drift
                current_sample = (
                    cond_mask * observed_data
                    + (1.0 - cond_mask) * current_sample
                )

            imputed_samples[:, i] = current_sample.detach()

        return imputed_samples  # (B, n_samples, 2, F, L)

    # ------------------------------------------------------------------
    # forward / evaluate
    # ------------------------------------------------------------------

    def forward(self, batch, is_train=1):
        observed_data, clean_data, observed_mask, observed_tp, gt_mask, for_pattern_mask = self.process_data(batch)

        if is_train == 0:
            cond_mask = gt_mask
        elif self.target_strategy != "random":
            cond_mask = self.get_hist_mask(observed_mask, for_pattern_mask=for_pattern_mask)
        else:
            cond_mask = self.get_randmask(observed_mask)

        side_info = self.get_side_info(observed_tp, cond_mask)
        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid
        return loss_func(observed_data, clean_data, cond_mask, observed_mask, side_info, is_train)

    def evaluate(self, batch, n_samples):
        """
        returns:
            samples       : (B, n_samples, 2, F, L)
            observed_data : (B, 2, F, L)
            target_mask   : (B, 1, F, L)  -- holes (0 in observed_mask)
            observed_mask : (B, 1, F, L)
            observed_tp   : (B, L)
        """
        observed_data, clean_data, observed_mask, observed_tp, gt_mask, _ = self.process_data(batch)

        with torch.no_grad():
            cond_mask   = observed_mask          # условие = все известные позиции
            target_mask = 1.0 - observed_mask    # 1 в дырках
            side_info   = self.get_side_info(observed_tp, cond_mask)
            samples     = self.impute(observed_data, cond_mask, side_info, n_samples)

        return samples, observed_data, target_mask, observed_mask, observed_tp


class CSDI_Physio(CSDI_base):
    """
    Concrete subclass for audio packet-loss inpainting.

    Batch from dataset_packetloss:
        observed_data : (B, 2, F, N)  -- Re/Im lossy STFT, дырки зануленны (маска уже применена)
        observed_mask : (B, 1, F, N)  -- 1=known, 0=hole (1 там где данные есть)
        gt_mask       : (B, 1, F, N)  -- то же
        timepoints    : (N,) or (B, N)
        clean_ri      : (B, 2, F, N)  -- чистая Re/Im спектрограмма (цель диффузии)
        clean_wav     : (B, T)
        lossy_wav     : (B, T)
    """

    def __init__(self, config, device, target_dim=257):
        super().__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()   # (B, 2, F, N)
        clean_data    = batch["clean_ri"].to(self.device).float()         # (B, 2, F, N)
        observed_mask = batch["observed_mask"].to(self.device).float()   # (B, 1, F, N)
        gt_mask       = batch["gt_mask"].to(self.device).float()          # (B, 1, F, N)
        observed_tp   = batch["timepoints"].to(self.device).float()       # (N,) or (B, N)

        if observed_tp.dim() == 1:
            observed_tp = observed_tp.unsqueeze(0).expand(observed_data.shape[0], -1)

        for_pattern_mask = observed_mask
        return observed_data, clean_data, observed_mask, observed_tp, gt_mask, for_pattern_mask
