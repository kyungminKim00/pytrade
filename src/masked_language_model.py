import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class MaskedLanguageModelDataset(Dataset):
    def __init__(
        self, observations: np.array, max_seq_length: int, gen_random_mask: bool = True
    ):
        self.observations = observations[:, :-1]
        self.predefined_mask = observations[:, -1].astype(int)
        self.max_seq_length = max_seq_length
        self.gen_random_mask = gen_random_mask

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
            padding = np.array([0.0] * padding_length)[:, None] * np.array(
                [0.0] * num_features
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

        return masked_obs, mask, obs


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
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
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class MaskedLanguageModel(nn.Module):
    def __init__(self, hidden_size: int, max_seq_length: int, num_features: int):
        super(MaskedLanguageModel, self).__init__()
        self.max_seq_length = max_seq_length

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
        self.fc = nn.Linear(hidden_size, num_features)
        self.pos_encoder = PositionalEncoding(hidden_size, max_seq_length)

    def forward(self, masked_obs: torch.Tensor) -> torch.Tensor:
        obs = self.embedding(masked_obs)
        obs = obs.permute(1, 0, 2)  # batch sizes 는 고정인 반면, sequeuce는 다양 함 (계산의 효율성)
        obs = self.pos_encoder(obs)

        # 이미 인풋 처리 되어 있는 경우 임
        # mask = mask.unsqueeze(0).repeat(self.max_seq_length, 1, 1)
        # output = self.transformer_encoder(obs, mask)

        output = self.transformer_encoder(obs)
        output = self.fc(output)
        output = output.permute(1, 0, 2)

        return output
