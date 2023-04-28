import json
import os

import imageio
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from joblib import dump, load
from plotly.subplots import make_subplots
from scipy.stats import wasserstein_distance
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from masked_language_model import MaskedLanguageModel, MaskedLanguageModelDataset
from util import print_c, remove_files

c_distance = wasserstein_distance

# 모듈 정보
with open("./src/context_prediction.json", "r", encoding="utf-8") as fp:
    env_dict = json.load(fp)


def gather_data(
    predict_up_down, real_rtn, real_std, real_idx, ctx_vector_0, ctx_vector_1
):
    def is_any(_tensor):
        return (
            torch.concat(_tensor, axis=0).detach().cpu().numpy()
            if len(_tensor) > 0
            else []
        )

    return {
        "predict_up_down": is_any(predict_up_down),
        "real_rtn": is_any(real_rtn),
        "real_std": is_any(real_std),
        "real_idx": is_any(real_idx),
        "ctx_vector_0": is_any(ctx_vector_0),
        "ctx_vector_1": is_any(ctx_vector_1),
    }


def plot_res(plot_data_train, plot_data_infer, epoch, top_n=10):
    """Data Retrive"""
    n_try = epoch
    real_rtn = plot_data_infer["real_rtn"]
    predict_up_down = plot_data_infer["predict_up_down"]
    # predict_std = plot_data_infer["predict_std"]
    real_idx = plot_data_infer["real_idx"]
    real_std = plot_data_infer["real_std"]

    y = np.where(real_rtn > 0, 1, 0)
    y_hat = np.where(predict_up_down > 0, 1, 0)

    accuracy = accuracy_score(y, y_hat)
    MAE = np.mean(np.abs(real_rtn - predict_up_down))
    print(f"[infer] Accuracy:{accuracy:.3} MAE:{MAE:.3}")

    trn_up_down = np.where(plot_data_train["real_rtn"] > 0, 1, 0)
    condition_up = trn_up_down == 1
    condition_down = ~condition_up
    condition_up_idx = np.argwhere(condition_up).squeeze()
    condition_down_idx = np.argwhere(condition_down).squeeze()

    for latent in ["ctx_vector_0", "ctx_vector_1"]:
        ctx_res = {}
        real_ctx = plot_data_train[latent]
        prd_ctx = plot_data_infer[latent]
        for infer_idx, _ in enumerate(y_hat):
            """Gather Data"""
            prd_ctx_ins = prd_ctx[infer_idx]
            real_rtn_ins = real_rtn[infer_idx]
            real_std_ins = real_std[infer_idx]
            predict_up_down_ins = predict_up_down[infer_idx]
            # predict_std_ins = predict_std[infer_idx]
            real_idx_ins = real_idx[infer_idx]

            if predict_up_down_ins:
                search_idx = condition_up_idx
            else:
                search_idx = condition_down_idx

            score = [
                (idx, c_distance(prd_ctx_ins, real_ctx[idx]))
                for _, idx in enumerate(search_idx)
            ]
            sorted_score = sorted(score, key=lambda x: x[1])
            retrive_idx = np.array(sorted_score[:top_n], dtype=int)[:, 0]

            mu = plot_data_train["real_rtn"][retrive_idx]
            std = plot_data_train["real_std"][retrive_idx]
            ctx_res.update(
                {
                    infer_idx: {
                        "bias": predict_up_down_ins,
                        # "bias_std": predict_std_ins,
                        # "fewshot_bias": np.mean(mu[:3]),
                        # "fewshot_std": np.mean(std[:3]),
                        "fewshot_bias": np.mean(mu),
                        "fewshot_std": np.mean(std),
                        "real_rtn": real_rtn_ins,
                        "real_std": real_std_ins,
                        "real_idx": real_idx_ins,
                    }
                }
            )

    """plot data
    """
    data = pd.DataFrame.from_dict(ctx_res).T
    data.to_csv(f"{env_dict['plot_check_dir']}/img/{latent}_{n_try}.csv")
    
    MAE_2 = np.mean(np.abs(data["real_rtn"] - data["fewshot_bias"]))
    
    fig_sub = make_subplots(rows=2, cols=1)
    colors = px.colors.qualitative.T10
    for i, item in enumerate(
        [
            "bias",
            "fewshot_bias",
            "real_rtn"
        ]
    ):
        scatter = go.Scatter(
            x=list(data.index)
            y=data[item],
            line=dict(color=colors[i]),
            legendgroup=item,
            name=item,
        )
        fig_sub.add_trace(scatter, row=1, col=1)
        
        for i, item in enumerate(
            [
                # "bias_std",
                "fewshot_std"
            ]
        ):
            scatter = go.Scatter(
                x=list(data.index),
                y=data[item],
                line=dict(color=colors[i]),
                legendgroup=item,
                name=item,
            )
            fig_sub.add_trace(scatter, row=2, col=1)
        fig_sub.update_layout(
            title=f"accuracy_score: {accuracy:.3}, MAE:{MAE:.3}, MAE2:{MAE2:.3}, fewshot_std:{data['fewshot_std'].mean():.3}"
        )
        fig_sub.write_image(
            f"{env_dict['plot_check_dir']}/img/final_{latent}_{n_try}.png"
        )

def Earthmover(y_pred, y_true):
    return torch.mean(
        torch.square(torch.cumsum(y_true, dim=-1) - torch.cumsum(y_pred, dim=-1))
    )


def rc_from_model(
    model,
    obs,
    src_mask,
    pad_mask,
    dec_obs,
    dec_src_mask,
    dec_pad_mask,
    fwd,
    domain,
):
    (
        obs,
        src_mask,
        pad_mask,
        fwd,
        dec_obs,
        dec_src_mask,
        dec_pad_mask,
    ) = (
        obs.to(device),
        src_mask.to(device),
        pad_mask.to(device),
        fwd.to(device),
        dec_obs.to(device),
        dec_src_mask.to(device),
        dec_pad_mask.to(device),
    )
    (
        output_vector_up_down,
        encoder_output,
        decoder_output,
    ) = model(obs, src_mask, pad_mask, domain, dec_obs, dec_src_mask, dec_pad_mask)

    return output_vector_up_down, encoder_output, decoder_output


def label_mask(fwd):
    fwd_mask = fwd != np.inf
    return fwd_mask[:,:,-1]


def retrive_context(
    dataloader,
    model,
    domain,
    predict_up_down,
    ctx_vector_0,
    ctx_vector_1,
    real_rtn,
    real_std,
    real_idx,
):
    for (
        obs,
        src_mask,
        pad_mask,
        fwd,
        dec_obs,
        dec_src_mask,
        dec_pad_mask,
    ) in dataloader:
        std_mask = label_mask(fwd)
        (
            output_vector_up_down,
            encoder_output,
            decoder_output,
        ) = rc_from_model(
            model,
            obs,
            src_mask,
            pad_mask,
            dec_obs,
            dec_src_mask,
            dec_pad_mask,
            fwd,
            domain,
        )
        r_rtn, _ = torch.min(fwd[:,:,:2], dim=-1)
        r_rtn = r_rtn[std_mask]
        r_idx = fwd[:,:,:2][std_mask]
        r_std = fwd[:,:,-1][std_mask]
        
        output_vector_up_down = output_vector_up_down.sequence(-1)[std_mask]
        
        predict_up_down.append(output_vector_up_down[-1][None])
        real_rtn.append(real_rtn[-1][None])
        real_std.append(real_std[-1][None])
        real_idx.append(real_idx[-1][None])
        ctx_vector_0.append(encoder_output[:, output_vector_up_down.shape[0]-1, :])
        ctx_vector_1.append(decoder_output[:, output_vector_up_down.shape[0]-1, :])
    
    return (
        predict_up_down,
        ctx_vector_0,
        ctx_vector_1,
        real_rtn,
        real_std,
        real_idx,
    )


def run_model(model, domain, _dataloader):
    model.eval()
    (
        predict_up_down,
        ctx_vector_0,
        ctx_vector_1,
        real_rtn,
        real_std,
        real_idx,
    ) = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    with torch.no_grad():
        (
            predict_up_down,
            ctx_vector_0,
            ctx_vector_1,
            real_rtn,
            real_std,
            real_idx,
        ) = retrive_context(
            _dataloader,
            model,
            domain,
            predict_up_down,
            ctx_vector_0,
            ctx_vector_1,
            real_rtn,
            real_std,
            real_idx,
        )
    plot_data = gather_data(
        predict_up_down,
        real_rtn,real_std,real_idx,ctx_vector_0,ctx_vector_1,
    )
    return plot_data


def retrive(
    model,
    src_dataloader,
    trg_dataloader,
    sampleing_iters,
    domain="band_prediction",
    top_n=10,
):
    plot_data_train = run_model(model, domain, src_dataloader)
    for epoch in range(sampleing_iters):
        plot_data_infer = run_model(model, domain, trg_dataloader)
        plot_res(plot_data_train, plot_data_infer, epoch, top_n)


if __name__ == "__main__":
    print("Ray initialized already" if ray.is_initialized() else ray.init())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # loss_dict = {
    #     "SmoothL1Loss": nn.SmoothL1Loss(),
    #     "mse": nn.MSELoss(),
    #     "Earthmover": Earthmover,
    #     "L1Loss": nn.L1Loss(),
    # }
    tv_ratio = env_dict["tv_ratio"]

    print_c("Load dataset")
    cc = load(f"{env_dict['crosscorrelation']}")
    data = cc.observatoins_merge_idx
    w_vars = cc.weight_variables
    padding_torken = cc.padding_torken
    forward_label = cc.forward_label

    # train configuration
    max_seq_length = env_dict["max_seq_length"]
    batch_size = env_dict["batch_size"]
    # latent size = hidden_size / 4 를 사용하고 있음. Hidden을 너무 적게 유지 할 수 없게 됨 나중에 데이터에 맞게 변경
    hidden_size = env_dict[
        "hidden_size"
    ]  # 1:2 or 1:4 (표현력에 강화), 2:1, 1:1 (특징추출을 변경을 적게 가함)
    step_size = env_dict["step_size"]  # 03.23 58 에폭까지 변화가 없음. lr 줄여봄 (변화없음) -> lr 키워봄
    enable_concept = env_dict["enable_concept"]  # Fix 실험값 변경하지 말기

    num_features = data.shape[1] - 1
    epochs = env_dict["prediction_epochs"]
    print(f"num_features {num_features}")

    # Split data into train and validation sets
    offset = int(len(data) * tv_ratio)
    dist = int((len(data) - offset) * env_dict["val_infer_ratio"])
    train_observations, train_label = data[:offset], forward_label[:offset]
    val_observations, val_label = (
        data[offset : offset + dist],
        forward_label[offset : offset + dist],
    )
    infer_observations, infer_label = (
        data[offset + dist :],
        forward_label[offset + dist :],
    )

    # 데이터 크기 출력
    for k, it in [
        ("Train", train_observations),
        ("Val", val_observations),
        ("Test", infer_observations),
    ]:
        print(f"{k}: {len(it)}")

    model_bias = MaskedLanguageModel(
        hidden_size, max_seq_length, num_features, enable_concept
    )
    model_bias.load_state_dict(torch.load(env_dict['bias_model']))
    model_bias = model_bias.to(device)
    print_c(f"Load model: {env_dict['bias_model']}")

    for name, param in model_bias.named_parameters():
        param.requires_grad = False
    
    retrive(
        model=model_bias,
        src_dataloader=DataLoader(
            MaskedLanguageModelDataset(
                train_observations,
                max_seq_length,
                padding_torken=padding_torken,
                gen_random_mask=False,
                forward_label=train_label,
                mode="decode",
            ),
            batch_size=1,
            shuffle=False,
        ),
        sampleing_iters=epochs,
        domain="band_prediction",
        top_n=env_dict["top_n"],
    )
    
    
    
    # find context model
    model_files = []
    for key, v in loss_dict.items():
        torket = f"{key}_"
        for fn in os.listdir(env_dict["assets_dir"]):
            if torket in fn:
                model_files.append(f"{env_dict['assets_dir']}/{fn}")
    model_files = list(set(model_files))

    # prediciton model 은 mse 고정
    for model_fn in model_files:
        model_mlm = MaskedLanguageModel(
            hidden_size, max_seq_length, num_features, enable_concept
        )
        model_mlm.load_state_dict(torch.load(model_fn))
        model_mlm = model_mlm.to(device)

        # 학습 파라미터 설정
        for name, param in model_mlm.named_parameters():
            if (
                "fc_up" not in name
                and "fc_down" not in name
                and "fc_std" not in name
                and "fc_up_down" not in name
                and "up_down_attn" not in name
                and "fc_up_brg" not in name
                and "fc_down_brg" not in name
                and "decoder_embedding" not in name
                and "transformer_decoder" not in name
            ):
                param.requires_grad = False
            else:
                param.requires_grad = True

        train(
            model=model_mlm,
            train_dataloader=DataLoader(
                MaskedLanguageModelDataset(
                    train_observations,
                    max_seq_length,
                    padding_torken=padding_torken,
                    gen_random_mask=False,  # Fix
                    forward_label=train_label,
                    n_period=env_dict["n_period"],
                    mode="decode",
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
                    n_period=env_dict["n_period"],
                    mode="decode",
                ),
                batch_size=1,
                shuffle=False,
            ),
            infer_dataloader=DataLoader(
                MaskedLanguageModelDataset(
                    infer_observations,
                    max_seq_length,
                    padding_torken=padding_torken,
                    gen_random_mask=False,  # Fix
                    forward_label=infer_label,
                    mode="decode",
                ),
                batch_size=1,
                shuffle=False,
            ),
            num_epochs=epochs,
            lr=step_size,
            weight_decay=1e-4,
            plot_train=True,
            plot_val=True,
            loss_type="mse",
            domain="band_prediction",
            weight_vars=w_vars,
        )

        # # plot result
        # plot_animate(loss_type=loss_dict["mse"])
