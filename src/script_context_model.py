import ray
import torch
import torch.nn as nn
import torch.optim as optim
from joblib import dump, load
from torch.utils.data import DataLoader
from tqdm import tqdm

from cross_correlation import CrossCorrelation
from masked_language_model import MaskedLanguageModel, MaskedLanguageModelDataset
from util import print_c

print("Ray initialized already" if ray.is_initialized() else ray.init())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, train_dataloader, val_dataloader, num_epochs, lr, weight_decay):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    best_val_loss = float("inf")

    for epoch in tqdm(range(num_epochs)):
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
            torch.save(model.state_dict(), "./src/assets/context_model.pt")


tv_ratio = 0.8
# 특징 추출
cc = CrossCorrelation(
    mv_bin=20,  # bin size for moving variance
    correation_bin=60,  # bin size to calculate cross correlation
    x_file_name="./src/local_data/raw/x_toys.csv",
    y_file_name="./src/local_data/raw/y_toys.csv",
    debug=False,
    enable_PCA=True,
    ratio=tv_ratio,  # enable_PCA=True 일 때 PCA 모형 구축 용, 차원 축소된 observation 은 전체 샘플 다 포함 함
)
dump(cc, "./src/local_data/assets/crosscorrelation.pkl")
cc = load("./src/local_data/assets/crosscorrelation.pkl")
data = cc.observatoins_merge_idx

# train configuration
max_seq_length = 120
batch_size = 32
hidden_size = 256
num_features = data.shape[1] - 1
epochs = 10
lr = 0.001

# Split data into train and validation sets
train_size = int(len(data) * tv_ratio)
train_observations = data[:train_size]
val_observations = data[train_size:]

train(
    model=MaskedLanguageModel(hidden_size, max_seq_length, num_features).to(device),
    train_dataloader=DataLoader(
        MaskedLanguageModelDataset(train_observations, max_seq_length),
        batch_size=batch_size,
        shuffle=True,
    ),
    val_dataloader=DataLoader(
        MaskedLanguageModelDataset(val_observations, max_seq_length),
        batch_size=batch_size,
        shuffle=False,
    ),
    num_epochs=epochs,
    lr=lr,
    weight_decay=0.0001,
)
