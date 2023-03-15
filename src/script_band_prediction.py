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
    domain="band_prediction",
    weight_vars=None,
):
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    criterion = loss_dict[loss_type]
    best_val = float("inf")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        tmp_data, plot_data_train, plot_data_val = [], [], []
        for (
            obs,
            src_mask,
            pad_mask,
            fwd,
            dec_obs,
            dec_src_mask,
            dec_pad_mask,
        ) in train_dataloader:
            fwd_mask = fwd != np.inf
            up_mask = fwd_mask[:, :, 0]
            down_mask = fwd_mask[:, :, 1]
            std_mask = fwd_mask[:, :, 2]
            obs, src_mask, pad_mask, fwd, dec_obs, dec_obs_mask, dec_obs_pad_mask = (
                obs.to(device),
                src_mask.to(device),
                pad_mask.to(device),
                fwd.to(device),
                dec_obs.to(device),
                dec_obs_mask.to(device),
                dec_obs_pad_mask.to(device),
            )
            output_vector_up, output_vector_down, output_vector_std = model(
                obs, src_mask, pad_mask, domain, dec_obs, dec_src_mask, dec_pad_mask
            )
            loss = (
                criterion(output_vector_up[up_mask], fwd[up_mask, 0])
                + criterion(output_vector_down[down_mask], fwd[down_mask, 1])
                + criterion(output_vector_std[std_mask], fwd[std_mask, 2])
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if plot_train:
                tmp_data.append(to_result(output_vector_up, label=1))
                tmp_data.append(to_result(output_vector_down, label=2))
                tmp_data.append(
                    to_result(torch.min(fwd[:, :2], dim=1).values[:, None], label=0)
                )
        train_loss /= len(train_dataloader)
        if plot_train:
            plot_data_train = np.concatenate(tmp_data, axis=0)

        model.eval()
        val_loss = 0
        tmp_data = []
        with torch.no_grad():
            for (
                obs,
                src_mask,
                pad_mask,
                fwd,
                dec_obs,
                dec_src_mask,
                dec_pad_mask,
            ) in val_dataloader:
                fwd_mask = fwd != np.inf
                up_mask = fwd_mask[:, :, 0]
                down_mask = fwd_mask[:, :, 1]
                std_mask = fwd_mask[:, :, 2]
                (
                    obs,
                    src_mask,
                    pad_mask,
                    fwd,
                    dec_obs,
                    dec_obs_mask,
                    dec_obs_pad_mask,
                ) = (
                    obs.to(device),
                    src_mask.to(device),
                    pad_mask.to(device),
                    fwd.to(device),
                    dec_obs.to(device),
                    dec_obs_mask.to(device),
                    dec_obs_pad_mask.to(device),
                )
                output_vector_up, output_vector_down, output_vector_std = model(
                    obs, src_mask, pad_mask, domain, dec_obs, dec_src_mask, dec_pad_mask
                )

                loss = (
                    criterion(output_vector_up[up_mask], fwd[up_mask, 0])
                    + criterion(output_vector_down[down_mask], fwd[down_mask, 1])
                    + criterion(output_vector_std[std_mask], fwd[std_mask, 2])
                )
                val_loss += loss.item()

                if plot_val:
                    tmp_data.append(to_result(output_vector_up, label=1))
                    tmp_data.append(to_result(output_vector_down, label=2))
                    tmp_data.append(to_result(torch.min(fwd[:, :2], dim=1), label=0))
        val_loss /= len(val_dataloader)

        # terminal process
        if plot_val:
            plot_data_val = np.concatenate(tmp_data, axis=0)
            plot_res(plot_data_train, plot_data_val, epoch, loss_type)

        sys_out_print = f"[{loss_type}] Epoch {epoch}: train loss {train_loss*10000:.4f} | val loss {val_loss*10000:.4f}"
        print(sys_out_print)
        with open(
            f"./src/local_data/assets/plot_check/{loss_type}/log.txt",
            "a",
            encoding="utf-8",
        ) as log:
            print(sys_out_print, file=log)

        # Save best model
        if val_loss < best_val:
            remove_files("./src/local_data/assets", loss_type, best_val)
            best_val = val_loss
            torch.save(
                model.state_dict(),
                f"./src/local_data/assets/{loss_type}_{epoch}_{best_val:.4f}_context_model.pt",
            )


loss_dict = {
    "SmoothL1Loss": nn.SmoothL1Loss(),
    "mse": nn.MSELoss(),
    "Earthmover": Earthmover,
    "L1Loss": nn.L1Loss(),
}
tv_ratio = 0.8
# # 특징 추출
# cc = CrossCorrelation(
#     mv_bin=20,  # bin size for moving variance
#     correation_bin=60,  # bin size to calculate cross correlation
#     x_file_name="./src/local_data/raw/x_toys.csv",
#     y_file_name="./src/local_data/raw/y_toys.csv",
#     debug=False,
#     data_tranform={
#         "n_components": 4,
#         "method": "UMAP",
#     },  # None: 데이터변환 수행안함, n_components: np.inf 는 전체 차원
#     ratio=tv_ratio,  # data_tranform is not None 일 때 PCA 학습 샘플, 차원 축소된 observation 은 전체 샘플 다 포함 함
#     alpha=1.5,
#     enable_predefined_mask=False,
# )
# dump(cc, "./src/local_data/assets/crosscorrelation.pkl")
cc = load("./src/local_data/assets/crosscorrelation.pkl")
data = cc.observatoins_merge_idx
weight_vars = cc.weight_variables
padding_torken = cc.padding_torken
forward_label = cc.forward_label


# train configuration
max_seq_length = 120
batch_size = 4
# latent size = hidden_size / 4 를 사용하고 있음. Hidden을 너무 적게 유지 할 수 없게 됨 나중에 데이터에 맞게 변경
hidden_size = 32  # 1:2 or 1:4 (표현력에 강화), 2:1, 1:1 (특징추출을 변경을 적게 가함)
num_features = data.shape[1] - 1
epochs = 150
step_size = 4e-5
enable_concept = False  # Fix 실험값 변경하지 말기
print(f"num_features {num_features}")

# Split data into train and validation sets
offset = int(len(data) * tv_ratio)
dist = int((len(data) - offset) * 0.5)
train_observations, train_label = data[:offset], forward_label[:offset]
val_observations, val_label = (
    data[offset : offset + dist],
    forward_label[offset : offset + dist],
)
test_observations, test_label = data[offset + dist :], forward_label[offset + dist :]

# 데이터 크기 출력
for k, it in [
    ("Train", train_observations),
    ("Val", val_observations),
    ("Test", test_observations),
]:
    print(f"{k}: {len(it)}")

# find context model
model_files = []
for k, v in loss_dict.items():
    loss_type = k
    torket = f"{loss_type}_"
    for fn in os.listdir("./src/local_data/assets"):
        if torket in fn:
            model_files.append(f"./src/local_data/assets/{fn}")
model_files = list(set(model_files))


# prediciton model 은 mse 고정
for model_fn in model_files:
    model = MaskedLanguageModel(
        hidden_size, max_seq_length, num_features, enable_concept
    )
    # model.load_state_dict(torch.load(model_fn))
    # model = model.to(device)

    # # 학습 파라미터 설정
    # for name, param in model.named_parameters():
    #     print(name)
    #     if "fc_up" not in name and "fc_down" not in name and "fc_std" not in name:
    #         param.requires_grad = False
    #     else:
    #         param.requires_grad = True

    train(
        model=model,
        train_dataloader=DataLoader(
            MaskedLanguageModelDataset(
                train_observations,
                max_seq_length,
                padding_torken=padding_torken,
                gen_random_mask=False,  # Fix
                forward_label=train_label,
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
                forward_label=val_label,
            ),
            batch_size=batch_size,
            shuffle=False,
        ),
        num_epochs=epochs,
        lr=step_size,
        weight_decay=1e-4,
        plot_train=True,
        plot_val=True,
        loss_type="mse",
        domain="band_prediction",
        weight_vars=weight_vars,
    )

    # plot result
    plot_animate(loss_type=loss_dict["mse"])
