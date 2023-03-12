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
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader

from cross_correlation import CrossCorrelation
from masked_language_model import MaskedLanguageModel, MaskedLanguageModelDataset
from util import print_c, remove_files

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
            files.append(f"{base_dir}/{loss_type}/{fn}")

    for file in files:
        if len(files) > 0:
            images = [imageio.imread(fn) for fn in file]
            imageio.mimsave(
                f"{base_dir}/{loss_type}/animated_image_{loss_type}.gif",
                images,
                duration=1,
                loop=1,
            )


def Earthmover(y_pred, y_true):
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


def KL_divergence(mu, log_var):
    var = torch.exp(log_var)
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - var, dim=1).mean()
    return kl_loss


def cal_density(data):
    # 2 components 만 활용 - 시각화 해석에 용이
    diff = data.unsqueeze(1)[:, :, :2] - data.unsqueeze(0)[:, :, :2]
    return (
        torch.sqrt(torch.sum(diff**2, axis=-1)).sum(axis=0)[:, None] / data.shape[0]
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
    weight_vars=None,
):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = loss_dict[loss_type]
    best_val_density = float("inf")

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
                output, output_density, mean, log_var = model(masked_obs, domain)
                # Calculate loss only for masked tokens
                loss = criterion(output[mask], obs[mask]) + KL_divergence(mean, log_var)
                # loss2 = torch.abs(output_density[mask] - cal_density(obs[mask])).mean()

                optimizer.zero_grad()
                # loss.backward(retain_graph=True)
                # loss2.backward()
                loss.backward()
                optimizer.step()

                # train_loss += loss.item() + loss2.item()
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
        density_output, density_obs = [], []
        with torch.no_grad():
            for masked_obs, mask, obs in val_dataloader:
                masked_obs, mask, obs = (
                    masked_obs.to(device),
                    mask.to(device),
                    obs.to(device),
                )
                if mask.sum() > 0:
                    (
                        output,
                        output_density,
                        mean,
                        log_var,
                    ) = model(masked_obs, domain)
                    # loss = (
                    #     criterion(output[mask], obs[mask])
                    #     + KL_divergence(mean, log_var)
                    #     + torch.abs(
                    #         output_density[mask] - cal_density(obs[mask])
                    #     ).mean()
                    # )
                    loss = criterion(output[mask], obs[mask]) + KL_divergence(
                        mean, log_var
                    )
                    val_loss += loss.item()

                    dist += torch.abs(output[mask] - obs[mask]).sum()
                    density_output.append(output[mask])
                    density_obs.append(obs[mask])

                    if plot_val:
                        tmp_data.append(to_result(output[mask], label=1))
                        tmp_data.append(to_result(obs[mask], label=0))
        val_loss /= len(val_dataloader)
        dist /= len(val_dataloader)

        density = []
        if len(density_output) > 0:
            density_output = torch.cat(density_output, dim=0)
            density_obs = torch.cat(density_obs, dim=0)

            for d_data in [density_output, density_obs]:
                density.append(cal_density(d_data).mean())
            density = torch.abs(density[1] - density[0])

        # terminal process
        if plot_val:
            plot_data_val = np.concatenate(tmp_data, axis=0)
            plot_res(plot_data_train, plot_data_val, epoch, loss_type)

        sys_out_print = f"[{loss_type}] Epoch {epoch}: train loss {train_loss*10000:.4f} | val loss {val_loss*10000:.4f} | dist {dist:.4f} | density {density:.4f}"
        print(sys_out_print)
        with open(
            f"./src/local_data/assets/plot_check/{loss_type}/log.txt",
            "a",
            encoding="utf-8",
        ) as log:
            print(sys_out_print, file=log)

        # Save best model
        if density < best_val_density:
            best_val_density = density
            remove_files("./src/assets", loss_type, density)
            torch.save(
                model.state_dict(),
                f"./src/assets/{loss_type}_{epoch}_{density:.4f}_context_model.pt",
            )


loss_dict = {
    "SmoothL1Loss": nn.SmoothL1Loss(),
    "mse": nn.MSELoss(),
    "Earthmover": Earthmover,
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
    data_tranform={
        "n_components": 4,
        "method": "UMAP",
    },  # None: 데이터변환 수행안함, n_components: np.inf 는 전체 차원
    ratio=tv_ratio,  # data_tranform is not None 일 때 PCA 학습 샘플, 차원 축소된 observation 은 전체 샘플 다 포함 함
    alpha=1.5,
)
dump(cc, "./src/local_data/assets/crosscorrelation.pkl")
cc = load("./src/local_data/assets/crosscorrelation.pkl")
data = cc.observatoins_merge_idx
weight_vars = cc.weight_variables
padding_torken = cc.padding_torken

# train configuration
max_seq_length = 120
batch_size = 4
# latent size = hidden_size / 4 를 사용하고 있음. Hidden을 너무 적게 유지 할 수 없게 됨 나중에 데이터에 맞게 변경
hidden_size = 32  # 1:2 or 1:4 (표현력에 강화), 2:1, 1:1 (특징추출을 변경을 적게 가함)
num_features = data.shape[1] - 1
epochs = 150
step_size = 4e-5
enable_concept = True
print(f"num_features {num_features}")

# Split data into train and validation sets
offset = int(len(data) * tv_ratio)
dist = int((len(data) - offset) * 0.5)
train_observations = data[:offset]
val_observations = data[offset : offset + dist]
test_observations = data[offset + dist :]
# offset = int(len(data) * tv_ratio)
# train_observations = data[:offset]
# val_observations = data[offset:]

# 데이터 크기 출력
for k, it in [
    ("Train", train_observations),
    ("Val", val_observations),
    ("Test", test_observations),
]:
    print(f"{k}: {len(it)}")


# context model training
for k, v in loss_dict.items():
    train(
        model=MaskedLanguageModel(
            hidden_size, max_seq_length, num_features, enable_concept
        ).to(device),
        train_dataloader=DataLoader(
            MaskedLanguageModelDataset(
                train_observations,
                max_seq_length,
                padding_torken=padding_torken,
                gen_random_mask=True,  # Fix
            ),
            batch_size=batch_size,
            shuffle=True,
        ),
        val_dataloader=DataLoader(
            MaskedLanguageModelDataset(
                val_observations,
                max_seq_length,
                padding_torken=padding_torken,
                gen_random_mask=False,  # Fix
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
        weight_vars=weight_vars,
    )

    # plot result
    plot_animate(loss_type=k)


# context model tune - skip
# few-shot learning for prediction
similar_samples = {}
for k, v in loss_dict.items():
    loss_type = k
    model_file = f"./src/assets/{loss_type}_context_model.pt"
