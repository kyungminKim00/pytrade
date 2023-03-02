import math
from typing import List

import numpy as np
import ray
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from cross_correlation import CrossCorrelation

ray.init()


class MaskedLanguageModelDataset(Dataset):
    def __init__(
        self, observations: np.array, predefined_mask: np.array, max_seq_length: int
    ):
        self.observations = observations
        self.predefined_mask = predefined_mask
        self.max_seq_length = max_seq_length

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
        num_predefined_mask = seq_length - predefined_mask.sum()

        mask = predefined_mask.tolist() + [0] * padding_length
        num_masked_tokens = min(
            seq_length, max(int(0.15 * seq_length) - num_predefined_mask, 1)
        )
        masked_indices = torch.randperm(seq_length)[:num_masked_tokens]
        mask = [
            0 if i in masked_indices else mask[i] for i in range(self.max_seq_length)
        ]
        masked_obs = [
            obs[i] if mask[i] else np.zeros(num_features)
            for i in range(self.max_seq_length)
        ]

        # Convert to tensors
        obs = torch.tensor(obs, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.bool)
        masked_obs = torch.tensor(masked_obs, dtype=torch.float32)

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
            ),
            num_layers=6,
        )
        self.fc = nn.Linear(hidden_size, num_features)
        self.pos_encoder = PositionalEncoding(hidden_size, max_seq_length)

    def forward(self, masked_obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        obs = self.embedding(masked_obs)
        obs = obs.permute(1, 0, 2)
        obs = self.pos_encoder(obs)

        mask = mask.unsqueeze(0).repeat(self.max_seq_length, 1, 1)
        output = self.transformer_encoder(obs, mask)
        output = self.fc(output)
        output = output.permute(1, 0, 2)

        return output


def train(model, train_dataloader, val_dataloader, num_epochs, lr, weight_decay):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for masked_obs, mask, obs in train_dataloader:
            masked_obs, mask, obs = (
                masked_obs.to(device),
                mask.to(device),
                obs.to(device),
            )
            optimizer.zero_grad()
            output = model(masked_obs, mask)
            # Calculate loss only for masked tokens
            loss = criterion(output[mask], obs[mask])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dataloader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for masked_obs, mask, obs in val_dataloader:
                masked_obs, mask, obs = (
                    masked_obs.to(device),
                    mask.to(device),
                    obs.to(device),
                )
                output = model(masked_obs, mask)
                # Calculate loss only for masked tokens
                loss = criterion(output[mask], obs[mask])
                val_loss += loss.item()
        val_loss /= len(val_dataloader)

        print(f"Epoch {epoch+1}: train loss {train_loss:.4f} | val loss {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")


cc = CrossCorrelation(
    mv_bin=20,  # bin size for moving variance
    correation_bin=60,  # bin size to calculate cross correlation
    x_file_name="./src/local_data/raw/x_toys.csv",
    y_file_name="./src/local_data/raw/y_toys.csv",
    debug=False,
)

# Split data into train and validation sets
train_size = int(len(cc.observatoins) * 0.8)
train_observations = cc.observatoins[:train_size]
train_predefined_mask = cc.predefined_mask[:train_size]
val_observations = cc.observatoins[train_size:]
val_predefined_mask = cc.predefined_mask[train_size:]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_hidden_size = 256
_max_seq_length = 120
_num_features = cc.observatoins.shape[1]
_lr = 0.001
_batch_size = 128
_num_epochs = 10
_weight_decay = 0.0001

# Create data loaders
_train_dataset = MaskedLanguageModelDataset(
    train_observations, train_predefined_mask, _max_seq_length
)
_train_dataloader = DataLoader(_train_dataset, batch_size=_batch_size, shuffle=True)
_val_dataset = MaskedLanguageModelDataset(
    val_observations, val_predefined_mask, _max_seq_length
)
_val_dataloader = DataLoader(_val_dataset, batch_size=_batch_size, shuffle=False)


_model = MaskedLanguageModel(_hidden_size, _max_seq_length, _num_features)
_model.to(device)
train(_model, _train_dataloader, _val_dataloader, _num_epochs, _lr, _weight_decay)
