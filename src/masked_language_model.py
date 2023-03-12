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
    ):
        self.observations = observations[:, :-1]
        self.predefined_mask = observations[:, -1].astype(int)
        self.max_seq_length = max_seq_length
        self.gen_random_mask = gen_random_mask
        self.padding_torken = padding_torken

    def __len__(self):
        return len(self.observations) - self.max_seq_length

    def __getitem__(self, idx):
        rnd_seq = np.random.randint(60, self.max_seq_length)

        obs = self.observations[idx : idx + rnd_seq]
        predefined_mask = self.predefined_mask[idx : idx + rnd_seq]

        seq_length, num_features = obs.shape
        padding_length = self.max_seq_length - seq_length

        # Apply padding if sequence is shorter than max_seq_length
        if padding_length > 0:
            padding_torken = self.padding_torken
            padding = np.array([padding_torken] * padding_length)[:, None] * np.array(
                [padding_torken] * num_features
            )
            obs = np.concatenate((obs, padding))

        # Randomly mask a portion of the input sequence
        num_predefined_mask = predefined_mask.sum()

        mask = predefined_mask.tolist() + [0] * padding_length

        # add random mask to predefined_mask
        if self.gen_random_mask:
            num_masked_tokens = min(
                seq_length, max(int(0.15 * seq_length) - num_predefined_mask, 1)
            )
            masked_indices = torch.randperm(seq_length)[:num_masked_tokens]

            # Mask the index to be hide
            mask = [
                1 if i in masked_indices else mask[i]
                for i in range(self.max_seq_length)
            ]

        masked_obs = [
            np.zeros(num_features) if mask[i] else obs[i]
            for i in range(self.max_seq_length)
        ]

        # Convert to tensors
        obs = torch.tensor(obs, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.bool)
        # list to tensor 속도 저하
        # masked_obs = torch.tensor(masked_obs, dtype=torch.float32)
        masked_obs = torch.tensor(
            np.concatenate(masked_obs, axis=0).reshape(obs.shape), dtype=torch.float32
        )

        # mask: Mask the index to be hide (1 means to be hide)
        return masked_obs, mask, obs


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
    ):
        super(MaskedLanguageModel, self).__init__()
        self.max_seq_length = max_seq_length
        latent_size = int(hidden_size / 4)  # 8 or 4

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
        self.fc_variance = nn.Linear(hidden_size, latent_size)
        self.fc_reparameterize = nn.Linear(latent_size, hidden_size)

        self.fc = nn.Linear(hidden_size, num_features)
        self.fc_density = nn.Linear(hidden_size, 1)
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
        masked_obs: torch.Tensor,
        domain: str = "context",
    ) -> torch.Tensor:
        obs = self.embedding(masked_obs)
        obs = obs.permute(1, 0, 2)  # batch sizes 는 고정인 반면, sequeuce는 다양 함 (계산의 효율성)
        # positional encoding
        obs = self.pos_encoder(obs)
        output = self.transformer_encoder(obs)

        if domain == "band_prediction":
            output = self.fc(output)
            output = output.permute(1, 0, 2)
        if domain == "context":
            output = nn.AdaptiveAvgPool1d(output.size(0))(output.permute(1, 2, 0))
            output = output.permute(2, 0, 1)

            mean = self.fc_variance(output)
            log_var = self.fc_variance(output)
            output = self.reparameterize(mean, log_var)
            output = self.fc_reparameterize(output)

            output_vector = self.fc(output)
            output_density = self.fc_density(output)

            output_vector = output_vector.permute(1, 0, 2)
            output_density = output_density.permute(1, 0, 2)

        return output_vector, output_density, mean, log_var
