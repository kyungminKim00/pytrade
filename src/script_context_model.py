import os

import imageio
import numpy as np
import plotly.express as px
import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from joblib import dump, load
from plotly.subplots import make_subplots
from torch.utils.data import DataLoader

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


def plot_res(plot_data_train, plot_data_val, epoch, loss_type):
    fig = []
    for plot_data in [plot_data_train, plot_data_val]:
        fig.append(
            px.scatter(
                plot_data,
                x=plot_data[:, 0],
                y=plot_data[:, 1],
                color=plot_data[:, -1],
                title="2D visualization for cross-correlation (blue: real yellow: predict)",
            )
        )

    # Create subplots with 1 row and 2 columns
    fig_sub = make_subplots(rows=1, cols=2)

    # Add the two figures to the subplot
    fig_sub.add_trace(fig[0].data[0], row=1, col=1)
    fig_sub.add_trace(fig[1].data[0], row=1, col=2)

    # Update the subplot layout
    fig_sub.update_layout(
        title="2D visualization for cross-correlation (blue: real yellow: predict)"
    )

    fig_sub.write_image(
        f"./src/local_data/assets/plot_check/{loss_type}/cc_2D_{loss_type}_{epoch}.png"
    )


def plot_animate(loss_type):
    base_dir = "./src/local_data/assets/plot_check"
    files = []
    for fn in os.listdir(base_dir):
        if f"cc_2D_{loss_type}" in fn:
            files.append(f"{base_dir}/{fn}")

    for file in files:
        if len(files) > 0:
            images = [imageio.imread(fn) for fn in file]
            imageio.mimsave(
                f"{base_dir}/{loss_type}/animated_image_{loss_type}.gif",
                images,
                duration=1,
                loop=1,
            )


def earth_mover(y_pred, y_true):
    return torch.mean(
        torch.square(
            torch.mean(
                torch.square(
                    torch.cumsum(y_true, dim=-1) - torch.cumsum(y_pred, dim=-1)
                ),
                dim=-1,
            )
        )
    )


def train(
    model,
    train_dataloader,
    val_dataloader,
    num_epochs,
    lr,
    weight_decay,
    plot_train=False,
    plot_val=False,
    loss_type="mse",
    domain="context",
):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = loss_dict[loss_type]
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        tmp_data, plot_data_train, plot_data_val = [], [], []
        for masked_obs, mask, obs in train_dataloader:
            masked_obs, mask, obs = (
                masked_obs.to(device),
                mask.to(device),
                obs.to(device),
            )
            if mask.sum() > 0:
                optimizer.zero_grad()
                output = model(masked_obs, domain)
                # Calculate loss only for masked tokens
                loss = criterion(output[mask], obs[mask])
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                if plot_train:
                    tmp_data.append(to_result(output[mask], label=1))
                    tmp_data.append(to_result(obs[mask], label=0))
        train_loss /= len(train_dataloader)
        if plot_train:
            plot_data_train = np.concatenate(tmp_data, axis=0)

        model.eval()
        val_loss = 0
        dist = 0
        tmp_data = []
        with torch.no_grad():
            for masked_obs, mask, obs in val_dataloader:
                masked_obs, mask, obs = (
                    masked_obs.to(device),
                    mask.to(device),
                    obs.to(device),
                )
                if mask.sum() > 0:
                    output = model(masked_obs, domain)
                    loss = criterion(output[mask], obs[mask])
                    val_loss += loss.item()
                    dist += torch.abs(output[mask] - obs[mask]).sum()

                    if plot_val:
                        tmp_data.append(to_result(output[mask], label=1))
                        tmp_data.append(to_result(obs[mask], label=0))
        val_loss /= len(val_dataloader)
        dist /= len(val_dataloader)

        # terminal process
        if plot_val:
            plot_data_val = np.concatenate(tmp_data, axis=0)
            plot_res(plot_data_train, plot_data_val, epoch, loss_type)
        print(
            f"[{loss_type}] Epoch {epoch+1}: train loss {train_loss*10000:.4f} | val loss {val_loss*10000:.4f} | dist {dist:.4f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"./src/assets/{loss_type}_context_model.pt")


loss_dict = {
    "SmoothL1Loss": nn.SmoothL1Loss(),
    "mse": nn.MSELoss(),
    "earth_mover": earth_mover,
    "L1Loss": nn.L1Loss(),
}
tv_ratio = 0.8
# 특징 추출
cc = CrossCorrelation(
    mv_bin=20,  # bin size for moving variance
    correation_bin=60,  # bin size to calculate cross correlation
    x_file_name="./src/local_data/raw/x_toys.csv",
    y_file_name="./src/local_data/raw/y_toys.csv",
    debug=False,
    n_components_pca=10,
    ratio=tv_ratio,  # n_components_pca is not None 일 때 PCA 학습 샘풀, 차원 축소된 observation 은 전체 샘플 다 포함 함
    alpha=1.5,
)
dump(cc, "./src/local_data/assets/crosscorrelation.pkl")
cc = load("./src/local_data/assets/crosscorrelation.pkl")
data = cc.observatoins_merge_idx

# train configuration
max_seq_length = 120
batch_size = 4
hidden_size = 256
num_features = data.shape[1] - 1
epochs = 100
step_size = 5e-5

# Split data into train and validation sets
train_size = int(len(data) * tv_ratio)
train_observations = data[:train_size]
val_observations = data[train_size:]

# model training
for k, v in loss_dict.items():
    if k == "earth_mover":
        train(
            model=MaskedLanguageModel(hidden_size, max_seq_length, num_features).to(
                device
            ),
            train_dataloader=DataLoader(
                MaskedLanguageModelDataset(
                    train_observations,
                    max_seq_length,
                ),
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
            weight_decay=1e-4,
            plot_train=True,
            plot_val=True,
            loss_type=k,
            domain="context",
        )

        # plot result
        plot_animate(loss_type=k)
