import numpy as np
import plotly.express as px
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


def to_result(res, label=0):
    src = res.detach().cpu().numpy()
    return np.concatenate(
        (
            src,
            (np.ones([src.shape[0], 1]) * label).astype(np.int32),
        ),
        axis=1,
    )


def plot_res(plot_data, mode, epoch):
    assert plot_data.shape[1] == 3, "2D 만 지원, PCA 수행 해야 함"
    fig = px.scatter(
        plot_data,
        x=plot_data[:, 0],
        y=plot_data[:, 1],
        color=plot_data[:, 2],
        title="2D visualization for cross-correlation (blue: real yellow: predict)",
    )
    fig.write_image(f"./src/local_data/assets/plot_check/cc_2D_{mode}_{epoch}.png")


def train(
    model,
    train_dataloader,
    val_dataloader,
    num_epochs,
    lr,
    weight_decay,
    plot_train=False,
    plot_val=False,
):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        plot_data = []
        for masked_obs, mask, obs in train_dataloader:
            masked_obs, mask, obs = (
                masked_obs.to(device),
                mask.to(device),
                obs.to(device),
            )
            if mask.sum() > 0:
                optimizer.zero_grad()
                output = model(masked_obs)
                # Calculate loss only for masked tokens
                loss = criterion(output[mask], obs[mask])
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                if plot_train:
                    plot_data.append(to_result(output[mask], label=1))
                    plot_data.append(to_result(obs[mask], label=0))
        train_loss /= len(train_dataloader)
        if plot_train:
            plot_data = np.concatenate(plot_data, axis=0)
            plot_res(plot_data, "train", epoch)

        model.eval()
        val_loss = 0
        plot_data = []
        with torch.no_grad():
            for masked_obs, mask, obs in val_dataloader:
                masked_obs, mask, obs = (
                    masked_obs.to(device),
                    mask.to(device),
                    obs.to(device),
                )
                if mask.sum() > 0:
                    output = model(masked_obs)
                    # Calculate loss only for masked tokens
                    loss = criterion(output[mask], obs[mask])
                    val_loss += loss.item()

                    if plot_val:
                        plot_data.append(to_result(output[mask], label=1))
                        plot_data.append(to_result(obs[mask], label=0))
        val_loss /= len(val_dataloader)
        if plot_val:
            plot_data = np.concatenate(plot_data, axis=0)
            plot_res(plot_data, "val", epoch)

        print(
            f"Epoch {epoch+1}: train loss {train_loss*10000:.4f} | val loss {val_loss*10000:.4f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "./src/assets/context_model.pt")


tv_ratio = 0.8
# # 특징 추출
# cc = CrossCorrelation(
#     mv_bin=20,  # bin size for moving variance
#     correation_bin=60,  # bin size to calculate cross correlation
#     x_file_name="./src/local_data/raw/x_toys.csv",
#     y_file_name="./src/local_data/raw/y_toys.csv",
#     debug=False,
#     n_components_pca=2,
#     ratio=tv_ratio,  # n_components_pca is not None 일 때 PCA 학습 샘풀, 차원 축소된 observation 은 전체 샘플 다 포함 함
#     alpha=1.5,
# )
# dump(cc, "./src/local_data/assets/crosscorrelation.pkl")
cc = load("./src/local_data/assets/crosscorrelation.pkl")
data = cc.observatoins_merge_idx

# train configuration
max_seq_length = 120
batch_size = 16
hidden_size = 256
num_features = data.shape[1] - 1
epochs = 1500
step_size = 0.0005

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
        MaskedLanguageModelDataset(
            val_observations, max_seq_length, gen_random_mask=False
        ),
        batch_size=batch_size,
        shuffle=False,
    ),
    num_epochs=epochs,
    lr=step_size,
    weight_decay=0.0001,
    plot_train=True,
    plot_val=True,
)
