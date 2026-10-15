import torch
import torch.nn as nn
import torch.nn.functional as FF
import math
from linear_attention_transformer import LinearAttentionTransformer  # type: ignore


# ---------------------------------------------------------------------------
# Утилиты для трансформеров
# ---------------------------------------------------------------------------

def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=channels * 4, activation="gelu",
        batch_first=False,  # ожидает (L, B, C)
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def get_linear_trans(heads=8, layers=1, channels=64, max_seq_len=512):
    return LinearAttentionTransformer(
        dim=channels,
        depth=layers,
        heads=heads,
        max_seq_len=max_seq_len,
        n_local_attn_heads=0,
        local_attn_window_size=0,
    )


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


# ---------------------------------------------------------------------------
# Диффузионный эмбеддинг (sinusoidal + 2-слойный MLP)
# ---------------------------------------------------------------------------

class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim // 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]  # (B, embedding_dim)
        x = FF.silu(self.projection1(x))
        x = FF.silu(self.projection2(x))
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)                          # (T, 1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1, dim)
        table = steps * frequencies                                            # (T, dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)        # (T, dim*2)
        return table


# ---------------------------------------------------------------------------
# Основная диффузионная модель
# ---------------------------------------------------------------------------

class diff_CSDI(nn.Module):
    """
    Диффузионная U-Net-подобная модель для inpainting комплексной спектрограммы.

    Вход x имеет форму (B, inputdim, F, L), где:
        inputdim = 2  (unconditional: Re, Im)
        inputdim = 4  (conditional: Re_cond, Im_cond, Re_noisy, Im_noisy)
        F = число частотных бинов (K)
        L = число временных кадров

    Выход: (B, 2, F, L) — предсказанный шум (Re и Im).
    """

    def __init__(self, config, inputdim=4):
        super().__init__()
        self.channels = config["channels"]

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )

        # Проецируем inputdim каналов -> channels
        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)

        # Выходной путь: channels -> channels -> 2 (Re и Im шум)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 2, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    is_linear=config["is_linear"],
                    # max_seq_len_time = L кадров, max_seq_len_freq = F бинов
                    # берём из конфига, дефолты с запасом если не указаны
                    max_seq_len_time=config.get("max_seq_len_time", 512),
                    max_seq_len_freq=config.get("max_seq_len_freq", 257),
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x, cond_info, diffusion_step):
        """
        x            : (B, inputdim, F, L)
        cond_info    : (B, side_dim, F, L)
        diffusion_step: (B,) long

        Возвращает: (B, 2, F, L)
        """
        B, inputdim, F, L = x.shape

        # Flatten F×L для Conv1d
        x = x.reshape(B, inputdim, F * L)
        x = FF.silu(self.input_projection(x))          # SiLU сохраняет отрицательные Re/Im значения
        x = x.reshape(B, self.channels, F, L)

        diffusion_emb = self.diffusion_embedding(diffusion_step)  # (B, diff_emb_dim)

        skip_sum = []
        for layer in self.residual_layers:
            x, skip = layer(x, cond_info, diffusion_emb)
            skip_sum.append(skip)

        # Суммируем skip-connections
        x = torch.stack(skip_sum, dim=0).sum(dim=0) / math.sqrt(len(self.residual_layers))
        # (B, channels, F, L)

        x = x.reshape(B, self.channels, F * L)
        x = FF.relu(self.output_projection1(x))        # (B, channels, F*L)
        x = self.output_projection2(x)                 # (B, 2, F*L)
        x = x.reshape(B, 2, F, L)
        return x


# ---------------------------------------------------------------------------
# Residual block с self-attention по времени и по частоте
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads,
                 is_linear=False, max_seq_len_time=500, max_seq_len_freq=257):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection      = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection       = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection    = Conv1d_with_init(channels, 2 * channels, 1)

        self.is_linear = is_linear
        if is_linear:
            # time_layer обрабатывает последовательности длиной L (временные кадры)
            # feature_layer обрабатывает последовательности длиной F (частоты)
            self.time_layer    = get_linear_trans(heads=nheads, layers=1, channels=channels,
                                                  max_seq_len=max_seq_len_time)
            self.feature_layer = get_linear_trans(heads=nheads, layers=1, channels=channels,
                                                  max_seq_len=max_seq_len_freq)
        else:
            self.time_layer    = get_torch_trans(heads=nheads, layers=1, channels=channels)
            self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

    def forward_time(self, y, base_shape):
        """Self-attention вдоль временной оси L."""
        B, channel, F, L = base_shape
        if L == 1:
            return y

        # (B, channel, F*L) -> (B*F, channel, L) -> attention -> обратно
        y = y.reshape(B, channel, F, L).permute(0, 2, 1, 3).reshape(B * F, channel, L)

        if self.is_linear:
            # LinearAttentionTransformer: (B*F, L, channel)
            y = self.time_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            # nn.TransformerEncoder: (L, B*F, channel)
            y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)

        y = y.reshape(B, F, channel, L).permute(0, 2, 1, 3).reshape(B, channel, F * L)
        return y

    def forward_feature(self, y, base_shape):
        """Self-attention вдоль частотной оси F."""
        B, channel, F, L = base_shape
        if F == 1:
            return y

        # (B, channel, F*L) -> (B*L, channel, F) -> attention -> обратно
        y = y.reshape(B, channel, F, L).permute(0, 3, 1, 2).reshape(B * L, channel, F)

        if self.is_linear:
            y = self.feature_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)

        y = y.reshape(B, L, channel, F).permute(0, 2, 3, 1).reshape(B, channel, F * L)
        return y

    def forward(self, x, cond_info, diffusion_emb):
        """
        x           : (B, channel, F, L)
        cond_info   : (B, side_dim, F, L)
        diffusion_emb: (B, diff_emb_dim)

        Возвращает: (x_out, skip) — оба (B, channel, F, L)
        """
        B, channel, F, L = x.shape
        base_shape = x.shape

        x_flat = x.reshape(B, channel, F * L)  # (B, channel, F*L)

        # добавляем diffusion embedding
        diff_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B, channel, 1)
        y = x_flat + diff_emb

        # self-attention по времени и частоте
        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)      # (B, channel, F*L)

        y = self.mid_projection(y)                   # (B, 2*channel, F*L)

        # добавляем side/conditional информацию
        cond_flat = cond_info.reshape(B, cond_info.shape[1], F * L)
        cond_proj = self.cond_projection(cond_flat)   # (B, 2*channel, F*L)
        y = y + cond_proj

        # gated activation (WaveNet-style)
        gate, filt = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filt)   # (B, channel, F*L)

        y = self.output_projection(y)                 # (B, 2*channel, F*L)
        residual, skip = torch.chunk(y, 2, dim=1)

        # восстанавливаем форму
        x_out    = (x_flat + residual).reshape(base_shape)
        skip_out = skip.reshape(base_shape)

        return x_out / math.sqrt(2.0), skip_out
