import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class MaskedLanguageModelDataset(Dataset):
    def __init__(
        self,
        observations: np.array,
        max_seq_length: int,
        padding_torken=None,
        gen_random_mask: bool = True,
        forward_label: np.array = None,
        n_period: int = 60,
    ):
        self.observations = observations[:, :-1]
        self.predefined_mask = observations[:, -1].astype(int)
        self.max_seq_length = max_seq_length
        self.gen_random_mask = gen_random_mask
        self.padding_torken = padding_torken
        self.forward_label = forward_label
        self.min_samples = n_period + int(n_period * 0.5)
        self.n_period = n_period
        assert (
            self.min_samples < max_seq_length
        ), "min_samples must be less than max_seq_length"

    def __len__(self):
        return len(self.observations) - self.max_seq_length

    def __getitem__(self, idx):
        rnd_seq = np.random.randint(self.min_samples, self.max_seq_length)

        obs = self.observations[idx : idx + rnd_seq]
        predefined_mask = self.predefined_mask[idx : idx + rnd_seq]
        fwd = self.forward_label[idx : idx + rnd_seq]
        dec_obs = self.observations[idx : idx + rnd_seq]

        last_fwd = fwd[-1]
        fwd = fwd[: -self.n_period]
        fwd = np.concatenate([fwd, last_fwd[None, :]], axis=0)

        last_dec_obs = dec_obs[-1]
        dec_obs = dec_obs[: -self.n_period]
        dec_obs = np.concatenate([dec_obs, last_dec_obs[None, :]], axis=0)
        length_dec = fwd.shape[0]

        # 배치 수 만큼 패딩
        fwd = np.concatenate(
            [fwd, np.zeros([self.max_seq_length - fwd.shape[0], fwd.shape[1]])],
            axis=0,
        )
        dec_obs = np.concatenate(
            [
                dec_obs,
                np.zeros([self.max_seq_length - dec_obs.shape[0], dec_obs.shape[1]]),
            ],
            axis=0,
        )
        # observable한 정보 모두 활용 - 시간별로 가리면서 하지 않음
        dec_obs_mask = [1] * length_dec + [0] * (self.max_seq_length - length_dec)
        dec_obs_pad_mask = [0] * length_dec + [1] * (self.max_seq_length - length_dec)

        assert fwd.shape[0] == dec_obs.shape[0], "length error"

        seq_length, num_features = obs.shape
        padding_length = self.max_seq_length - seq_length

        # Apply padding if sequence is shorter than max_seq_length
        if padding_length > 0:
            padding_torken = self.padding_torken
            padding = np.array([padding_torken] * padding_length)[:, None] * np.array(
                [padding_torken] * num_features
            )
            obs = np.concatenate((obs, padding))

        num_predefined_mask = len(predefined_mask) - predefined_mask.sum()
        # mask = predefined_mask.tolist() + [0] * padding_length

        src_mask = predefined_mask.tolist() + [0] * padding_length
        pad_mask = [0] * len(predefined_mask) + [1] * padding_length

        # add random mask to predefined_mask
        if self.gen_random_mask:
            num_masked_tokens = min(
                seq_length, max(int(0.15 * seq_length) - num_predefined_mask, 1)
            )
            masked_indices = torch.randperm(seq_length)[:num_masked_tokens]

            # Mask the index to be hide
            src_mask = [
                0 if i in masked_indices else src_mask[i]
                for i in range(self.max_seq_length)
            ]

        # 차원 확장 및 Fill
        src_mask = np.tile(src_mask, (obs.shape[1], 1)).T
        # pad_mask = np.tile(pad_mask, (obs.shape[1], 1)).T.astype(bool)
        dec_obs_mask = np.tile(dec_obs_mask, (dec_obs.shape[1], 1)).T
        # dec_obs_pad_mask = np.tile(dec_obs_pad_mask, (dec_obs.shape[1], 1)).T.astype(
        #     bool
        # )

        # masked_obs = [
        #     np.zeros(num_features) if mask[i] else obs[i]
        #     for i in range(self.max_seq_length)
        # ]

        # Convert to tensors
        obs = torch.tensor(obs, dtype=torch.float32)
        src_mask = torch.tensor(src_mask, dtype=torch.bool)
        pad_mask = torch.tensor(pad_mask, dtype=torch.bool)

        # mask = torch.tensor(mask, dtype=torch.bool)
        # list to tensor 속도 저하
        # masked_obs = torch.tensor(masked_obs, dtype=torch.float32)
        # masked_obs = torch.tensor(
        #     np.concatenate(masked_obs, axis=0).reshape(obs.shape), dtype=torch.float32
        # )
        fwd = torch.tensor(fwd, dtype=torch.float32)
        dec_obs = torch.tensor(dec_obs, dtype=torch.float32)
        dec_obs_mask = torch.tensor(dec_obs_mask, dtype=torch.bool)
        dec_obs_pad_mask = torch.tensor(dec_obs_pad_mask, dtype=torch.bool)

        # return masked_obs, mask, obs, fwd, dec_obs
        return obs, src_mask, pad_mask, fwd, dec_obs, dec_obs_mask, dec_obs_pad_mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length, enable_concept):
        super(PositionalEncoding, self).__init__()
        self.enable_concept = enable_concept
        self.dropout = nn.Dropout(p=0.1)
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        if self.enable_concept:
            x = (
                x
                + self.pe[: x.size(0), :]
                + x.mean(dim=0, keepdim=True)
                + x.std(dim=0, keepdim=True)
            )
        else:
            x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class MaskedLanguageModel(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        max_seq_length: int,
        num_features: int,
        enable_concept: bool = False,
        dec_num_features: int = None,
    ):
        super(MaskedLanguageModel, self).__init__()
        self.max_seq_length = max_seq_length
        latent_size = int(hidden_size / 4)  # 8 or 4

        if dec_num_features is None:  # 오류 방어 코드 - 실제 디코더는 학습 되지 않음
            dec_num_features = num_features

        self.embedding = nn.Linear(num_features, hidden_size)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=hidden_size * 4,
                dropout=0.1,
                activation="gelu",
            ),
            num_layers=6,
        )

        self.decoder_embedding = nn.Linear(num_features, hidden_size)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=hidden_size * 4,
                dropout=0.1,
                activation="gelu",
            ),
            num_layers=6,
        )

        self.fc_variance = nn.Linear(hidden_size, latent_size)
        self.fc_reparameterize = nn.Linear(latent_size, hidden_size)

        self.fc = nn.Linear(hidden_size, num_features)
        self.fc_up = nn.Linear(hidden_size, 1)
        self.fc_down = nn.Linear(hidden_size, 1)
        self.fc_std = nn.Linear(hidden_size, 1)
        self.pos_encoder = PositionalEncoding(
            hidden_size, max_seq_length, enable_concept
        )

    # padding_ignore 사용 안함 - 학습에 영향을 줄 수 있을 것 같음
    def padding_ignore(self, x):
        idxs = torch.isnan(x) | torch.isinf(x) | (torch.isinf(x) & (x.sign() < 0))
        x.masked_fill_(idxs, float(0))
        return x

    def reparameterize(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z

    def forward(
        self,
        obs: torch.Tensor,
        src_mask: torch.Tensor,
        pad_mask: torch.Tensor,
        domain: str = "context",
        dec_obs: torch.Tensor = None,
        dec_src_mask: torch.Tensor = None,
        dec_pad_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        obs = self.embedding(obs)
        obs = obs.permute(1, 0, 2)  # batch sizes 는 고정인 반면, sequeuce는 다양 함 (계산의 효율성)
        # positional encoding
        obs = self.pos_encoder(obs)
        # output = self.transformer_encoder(
        #     obs, src_mask=src_mask, src_key_padding_mask=pad_mask
        # )
        output = self.transformer_encoder(
            obs, mask=src_mask, src_key_padding_mask=pad_mask
        )

        output = nn.AdaptiveAvgPool1d(output.size(0))(output.permute(1, 2, 0))
        output = output.permute(2, 0, 1)

        mean = self.fc_variance(output)
        log_var = self.fc_variance(output)
        output = self.reparameterize(mean, log_var)
        output = self.fc_reparameterize(output)

        if domain == "band_prediction":  # 디코더 학습 및 추론
            dec_obs = self.decoder_embedding(dec_obs)
            dec_obs = dec_obs.permute(1, 0, 2)
            dec_obs = self.pos_encoder(dec_obs)
            decoder_output = self.transformer_decoder(
                dec_obs,
                output,
                tgt_mask=dec_src_mask,
                tgt_key_padding_mask=dec_pad_mask,
            )
            decoder_output = decoder_output.permute(1, 0, 2)

            output_vector_up = self.fc_up(decoder_output).permute(1, 0, 2)
            output_vector_down = self.fc_down(decoder_output).permute(1, 0, 2)
            output_vector_std = self.fc_std(decoder_output).permute(1, 0, 2)
            return output_vector_up, output_vector_down, output_vector_std

        if domain == "context":  # 인코더 학습 및 추론
            output_vector = self.fc(output)
            output_vector = output_vector.permute(1, 0, 2)
            return output_vector, mean, log_var
