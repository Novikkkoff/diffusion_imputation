import numpy as np
import torch
import torch.nn as nn
from diff_models import diff_CSDI
import torch.nn.functional as F

class CSDI_base(nn.Module):
    def __init__(self, target_dim, config, device):
        super().__init__()
        self.device = device
        self.target_dim = target_dim

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]
        self.debug_quality_check = True  # Включить проверку качества
        self.debug_visualize = True      # Включить визуализацию

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if self.is_unconditional == False:
            self.emb_total_dim += 1  # for conditional mask
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        input_dim = 1 if self.is_unconditional == True else 2
        self.diffmodel = diff_CSDI(config_diff, input_dim)

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)
        self.debug_compare = False  
    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_randmask(self, observed_mask):
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
        for i in range(len(observed_mask)):
            sample_ratio = np.random.rand()  # missing ratio
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask

    def get_hist_mask(self, observed_mask, for_pattern_mask=None):
        if for_pattern_mask is None:
            for_pattern_mask = observed_mask
        if self.target_strategy == "mix":
            rand_mask = self.get_randmask(observed_mask)

        cond_mask = observed_mask.clone()
        for i in range(len(cond_mask)):
            mask_choice = np.random.rand()
            if self.target_strategy == "mix" and mask_choice > 0.5:
                cond_mask[i] = rand_mask[i]
            else:  # draw another sample for histmask (i-1 corresponds to another sample)
                cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1] 
        return cond_mask

    def get_test_pattern_mask(self, observed_mask, test_pattern_mask):
        return observed_mask * test_pattern_mask


    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape
        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        
        side_info = torch.cat([time_embed, feature_embed], dim=-1)
        side_info = side_info.permute(0, 3, 2, 1)
        
        # ДОБАВЬТЕ НОРМАЛИЗАЦИЮ:
        side_info = (side_info - side_info.mean()) / (side_info.std() + 1e-8)
        
        if self.is_unconditional == False:
            side_mask = cond_mask.unsqueeze(1)
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def calc_loss_valid(
        self, observed_data, cond_mask, observed_mask, side_info, is_train
    ):
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, side_info, is_train, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(
        self, observed_data, cond_mask, observed_mask, side_info, is_train, set_t=-1
    ):
        B, K, L = observed_data.shape
        # print(observed_data.shape)
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise

        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)

        predicted = self.diffmodel(total_input, side_info, t)  # (B,K,L)

        target_mask = observed_mask - cond_mask
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_unconditional == True:
            total_input = noisy_data.unsqueeze(1)  # (B,1,K,L)
        else:
            cond_obs = (cond_mask * observed_data).unsqueeze(1)
            noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
            total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)

        return total_input

    def impute(self, observed_data, cond_mask, side_info, n_samples):
        B, K, L = observed_data.shape
        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        # ==================== ПРЕДВАРИТЕЛЬНЫЕ ПРОВЕРКИ ====================
        print(f"=== ДИАГНОСТИКА IMPUTE ===")
        print(f"Размеры: B={B}, K={K}, L={L}, n_samples={n_samples}")
        print(f"num_steps: {self.num_steps}")
        print(f"is_unconditional: {self.is_unconditional}")
        
        # Проверка маски
        mask_coverage = cond_mask.float().mean().item()
        print(f"Заполненность маски: {mask_coverage:.3f} ({mask_coverage*100:.1f}%)")
        print(f"Уникальные значения маски: {torch.unique(cond_mask)}")
        
        # Проверка входных данных
        print(f"observed_data: min={observed_data.min():.3f}, max={observed_data.max():.3f}, mean={observed_data.mean():.3f}")
        print(f"side_info shape: {side_info.shape if side_info is not None else 'None'}")
        
        # Проверка расписания шума
        if hasattr(self, 'alpha_hat') and self.alpha_hat is not None:
            print(f"alpha_hat[:3]: {self.alpha_hat[:3] if len(self.alpha_hat) > 3 else self.alpha_hat}")
            print(f"alpha_hat[-3:]: {self.alpha_hat[-3:] if len(self.alpha_hat) > 3 else self.alpha_hat}")
        
        # ==================== ТЕСТ С ПОЛНЫМ ЗАШУМЛЕНИЕМ ====================
        if mask_coverage < 0.9:  # Если маскирование меньше 90%, проверим полное
            print("\n--- ТЕСТ: Полное маскирование ---")
            full_mask = torch.zeros_like(cond_mask)
            test_full_coverage = full_mask.float().mean().item()
            print(f"Тестовая маска заполненность: {test_full_coverage:.3f}")

        # ==================== ОСНОВНОЙ ЦИКЛ ====================
        for i in range(n_samples):
            print(f"\n--- Сэмпл {i+1}/{n_samples} ---")
            
            # generate noisy observation for unconditional model
            if self.is_unconditional == True:
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)

            current_sample = torch.randn_like(observed_data)
            print(f"Инициализация current_sample: min={current_sample.min():.3f}, max={current_sample.max():.3f}")

            for t in range(self.num_steps - 1, -1, -1):
                print(f"  Шаг {t}: ", end="")
                
                if self.is_unconditional == True:
                    diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                    diff_input = diff_input.unsqueeze(1)  # (B,1,K,L)
                    print(f"unconditional, diff_input shape: {diff_input.shape}")
                else:
                    cond_obs = (cond_mask * observed_data).unsqueeze(1)
                    noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                    
                    # Диагностика каналов
                    cond_nonzero = (cond_obs[0] != 0).float().mean().item()
                    target_nonzero = (noisy_target[0] != 0).float().mean().item()
                    print(f"conditional, cond_obs заполненность: {cond_nonzero:.3f}, target заполненность: {target_nonzero:.3f}")
                    
                    diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
                    print(f"diff_input shape: {diff_input.shape}")

                # Проверка diff_input перед моделью
                print(f"    diff_input: min={diff_input.min():.3f}, max={diff_input.max():.3f}, mean={diff_input.mean():.3f}")
                

                predicted = self.diffmodel(diff_input, side_info, torch.tensor([t]).to(self.device))
                print(f"    predicted: min={predicted.min():.3f}, max={predicted.max():.3f}, norm={predicted.norm():.3f}")

                # Сохраняем предыдущее состояние для отслеживания изменений
                prev_sample = current_sample.clone()
                
                # Вычисление коэффициентов
                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                print(f"    coeff1={coeff1:.3f}, coeff2={coeff2:.3f}")
                
                # Обновление current_sample
                current_sample = coeff1 * (current_sample - coeff2 * predicted)
                
                # Отслеживание изменений
                sample_change = (current_sample - prev_sample).abs().mean().item()
                print(f"    изменение current_sample: {sample_change:.6f}")

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = ((1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]) ** 0.5
                    print(f"    добавление шума: sigma={sigma:.6f}")
                    current_sample += sigma * noise

                # Проверка на NaN/Inf
                if torch.isnan(current_sample).any() or torch.isinf(current_sample).any():
                    print(f"    ⚠️ ОШИБКА: NaN/Inf в current_sample на шаге {t}!")
                    break

            # Финальная диагностика сэмпла
            print(f"Финальный current_sample: min={current_sample.min():.3f}, max={current_sample.max():.3f}")
            
            imputed_samples[:, i] = current_sample.detach()
            
            # Сравнение с оригиналом (если есть доступ к нему)
            if hasattr(self, 'debug_compare') and self.debug_compare:
                original_norm = observed_data.norm().item()
                imputed_norm = current_sample.norm().item()
                similarity = F.cosine_similarity(observed_data.flatten(), current_sample.flatten(), dim=0).item()
                print(f"Сравнение: original_norm={original_norm:.3f}, imputed_norm={imputed_norm:.3f}, similarity={similarity:.3f}")

        # ==================== ФИНАЛЬНАЯ СТАТИСТИКА ====================
        print(f"\n=== ФИНАЛЬНАЯ СТАТИСТИКА ===")
        print(f"imputed_samples: min={imputed_samples.min():.3f}, max={imputed_samples.max():.3f}")
        print(f"Среднее по сэмплам: {imputed_samples.mean(dim=1).norm().item():.3f}")
        

        return imputed_samples

    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            _,
        ) = self.process_data(batch)
        if is_train == 0:
            cond_mask = gt_mask
        elif self.target_strategy != "random":
            cond_mask = self.get_hist_mask(
                observed_mask, for_pattern_mask=for_pattern_mask
            )
        else:
            cond_mask = self.get_randmask(observed_mask)
        # print('observed_tp.shape',observed_tp.shape)
        # print('cond_mask.shape',cond_mask.shape)
        side_info = self.get_side_info(observed_tp, cond_mask)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid
        # print(observed_data.shape)
        # print(cond_mask.shape)
        # print(observed_mask.shape)
        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)

    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            cut_length,
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = observed_mask
            target_mask = 1.0 - observed_mask
            side_info = self.get_side_info(observed_tp, cond_mask)
            violations = (gt_mask[observed_mask == 0] != 0)
            samples = self.impute(observed_data, cond_mask, side_info, n_samples)

            for i in range(len(cut_length)):  # to avoid double evaluation
                target_mask[i, ..., 0 : cut_length[i].item()] = 0
        return samples, observed_data, target_mask, observed_mask, observed_tp



class CSDI_Physio(CSDI_base):
    def __init__(self, config, device, target_dim=256):
        super(CSDI_Physio, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )

