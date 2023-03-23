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
        mode: str = "encode",
    ):
        self.observations = observations[:, :-1]
        self.predefined_mask = observations[:, -1].astype(int)
        self.max_seq_length = max_seq_length
        self.gen_random_mask = gen_random_mask
        self.padding_torken = padding_torken
        self.forward_label = forward_label
        # 수정 된 내용 - 03.20
        self.min_samples = n_period + int(n_period * 0.5)
        self.mode = mode
        self.n_period = n_period
        assert (
            self.min_samples < max_seq_length
        ), "min_samples must be less than max_seq_length"

    def __len__(self):
        return len(self.observations) - self.max_seq_length

    def __getitem__(self, idx):
        idx = idx + self.max_seq_length

        padding_torken = self.padding_torken
        rnd_seq = np.random.randint(self.min_samples, self.max_seq_length)

        obs = self.observations[idx - rnd_seq : idx]
        predefined_mask = self.predefined_mask[idx - rnd_seq : idx]
        fwd = self.forward_label[idx - rnd_seq : idx]
        dec_obs = self.observations[idx - rnd_seq : idx]

        last_fwd = fwd[-1]
        fwd = fwd[: -self.n_period]
        fwd = np.concatenate([fwd, last_fwd[None, :]], axis=0)

        last_dec_obs = dec_obs[-1]
        dec_obs = dec_obs[: -self.n_period]
        dec_obs = np.concatenate([dec_obs, last_dec_obs[None, :]], axis=0)
        length_dec = fwd.shape[0]

        # 배치 수 만큼 패딩
        fwd = np.concatenate(
            [
                fwd,
                np.inf * np.ones([self.max_seq_length - fwd.shape[0], fwd.shape[1]]),
            ],
            axis=0,
        )
        dec_obs = np.concatenate(
            [
                dec_obs,
                padding_torken
                * np.ones([self.max_seq_length - dec_obs.shape[0], dec_obs.shape[1]]),
            ],
            axis=0,
        )
        # observable한 정보 모두 활용 - 시간별로 가리면서 하지 않음
        dec_obs_mask = [1] * length_dec + [1] * (self.max_seq_length - length_dec)
        dec_obs_pad_mask = [0] * length_dec + [1] * (self.max_seq_length - length_dec)

        assert fwd.shape[0] == dec_obs.shape[0], "length error"

        seq_length, num_features = obs.shape
        padding_length = self.max_seq_length - seq_length

        # Apply padding if sequence is shorter than max_seq_length
        if padding_length > 0:
            padding = np.array([padding_torken] * padding_length)[:, None] * np.array(
                [padding_torken] * num_features
            )
            obs = np.concatenate((obs, padding))

        num_predefined_mask = len(predefined_mask) - predefined_mask.sum()
        # mask = predefined_mask.tolist() + [0] * padding_length

        if self.mode == "encode":
            src_mask = (
                predefined_mask.tolist() + [1] * padding_length
            )  # padding values 가 이미 설정 되어 있으므로 1
        else:
            src_mask = [1] * len(predefined_mask) + [1] * padding_length
            assert not self.gen_random_mask, "src_mask contains 1 only on a decode mode"
        pad_mask = [0] * len(predefined_mask) + [1] * padding_length

        # add random mask to predefined_mask
        if self.gen_random_mask:
            num_target_mask = int(0.15 * seq_length)
            if num_target_mask > num_predefined_mask:
                num_masked_tokens = min(
                    seq_length, max(num_target_mask - num_predefined_mask, 1)
                )
                masked_indices = torch.randperm(seq_length)[:num_masked_tokens]
                mask_val = 0
            else:
                num_masked_tokens = num_predefined_mask - num_target_mask
                aa = np.argwhere(np.array(src_mask) == 0).squeeze()
                np.random.shuffle(aa)
                masked_indices = torch.tensor(aa)[:num_masked_tokens]
                mask_val = 1

            # Mask the index to be hide
            src_mask = [
                mask_val if i in masked_indices else src_mask[i]
                for i in range(self.max_seq_length)
            ]

        # # 차원 확장 및 Fill - 차원 확장 & 속성확장 & Fill 코드 지우지는 말기
        # src_mask = np.tile(src_mask, (obs.shape[1], 1)).T
        # dec_obs_mask = np.tile(dec_obs_mask, (dec_obs.shape[1], 1)).T

        masked_obs = [
            np.zeros(num_features) if src_mask[i] == 0 else obs[i]
            for i in range(self.max_seq_length)
        ]  # blank만 0으로 채워진 상태, 패딩값은 예약값 할당

        # Convert to tensors
        obs = torch.tensor(obs, dtype=torch.float32)  # 패딩값만 예약값 할당, 나머지는 원래값 할당
        src_mask = torch.tensor(src_mask, dtype=torch.bool)  # blank 만 0으로 할당
        pad_mask = torch.tensor(pad_mask, dtype=torch.bool)  # padding 만 0으로 할당

        # 나중에 코드 지우기
        # mask = torch.tensor(mask, dtype=torch.bool)

        masked_obs = torch.tensor(
            np.concatenate(masked_obs, axis=0).reshape(obs.shape), dtype=torch.float32
        )

        fwd = torch.tensor(fwd, dtype=torch.float32)
        dec_obs = torch.tensor(dec_obs, dtype=torch.float32)
        dec_obs_mask = torch.tensor(dec_obs_mask, dtype=torch.bool)
        dec_obs_pad_mask = torch.tensor(dec_obs_pad_mask, dtype=torch.bool)

        # return masked_obs, mask, obs, fwd, dec_obs
        if self.mode == "encode":
            return (
                obs,
                masked_obs,
                src_mask,
                pad_mask,
                fwd,
                dec_obs,
                dec_obs_mask,
                dec_obs_pad_mask,
            )
        return (
            obs,
            src_mask,
            pad_mask,
            fwd,
            dec_obs,
            dec_obs_mask,
            dec_obs_pad_mask,
        )


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
        if self.enable_concept:  # 학습 안됨
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
        # self.embedding = nn.Linear(num_features, hidden_size, bias=False)  # src_padd_mask 가 동작하면 굳이 False 줄 필요 없음.
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

        self.fc_up_brg = nn.Linear(hidden_size, hidden_size)
        self.fc_up = nn.Linear(hidden_size, 1)
        self.fc_down_brg = nn.Linear(hidden_size, hidden_size)
        self.fc_down = nn.Linear(hidden_size, 1)

        self.bn_latent = nn.BatchNorm1d(num_features=hidden_size)

        # option 1 - concat 후 어텐션, 판별
        # self.up_down_attn = nn.MultiheadAttention(
        #     hidden_size * 2, num_heads=2, dropout=0.1, batch_first=True
        # )
        # self.fc_up_down = nn.Linear(hidden_size * 2, 1)
        # self.fc_std = nn.Linear(hidden_size * 2, 1)

        # option 2 - concat 후, 압축, 어텐션, 판별
        # self.mp1d = nn.MaxPool1d(2, stride=2)
        # self.up_down_attn = nn.MultiheadAttention(
        #     hidden_size, num_heads=2, dropout=0.1, batch_first=True
        # )
        # self.fc_up_down = nn.Linear(hidden_size, 1)
        # self.fc_std = nn.Linear(hidden_size, 1)

        # option 3 - concat 후 압축, 판별
        self.mp1d = nn.MaxPool1d(2, stride=2)
        self.fc_up_down = nn.Linear(hidden_size, 1)
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

    def generate_src_mask(
        self,
        src_mask: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        # 나중에 코드 지우기
        # src_mask = src_mask[:, :, -1]
        src_mask = ~src_mask
        src_pad_mask = src_mask | pad_mask
        return src_pad_mask

    def forward(
        self,
        masked_obs: torch.Tensor = None,
        src_mask: torch.Tensor = None,
        pad_mask: torch.Tensor = None,
        domain: str = "context",
        dec_obs: torch.Tensor = None,
        dec_src_mask: torch.Tensor = None,
        dec_pad_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        emd = self.embedding(masked_obs)
        emd = emd.permute(1, 0, 2)  # batch sizes 는 고정인 반면, sequeuce는 다양 함 (계산의 효율성)
        # positional encoding
        emd = self.pos_encoder(emd)

        # # # # transformer encoder의 구조를 그대로 사용 할 수 없음
        # # # # transform encoder 의 인코더 src_mask는 미래의 값을 볼 것인가 아닌가에 초첨이 맞추어져 있음.
        src_pad_mask = self.generate_src_mask(src_mask, pad_mask)
        output = self.transformer_encoder(emd, src_key_padding_mask=src_pad_mask)
        # output = self.transformer_encoder(emd, src_key_padding_mask=pad_mask)  # it works for

        # output = self.transformer_encoder(emd)

        output = nn.AdaptiveAvgPool1d(output.size(0))(output.permute(1, 2, 0))
        output = output.permute(2, 0, 1)

        mean = self.fc_variance(output)
        log_var = self.fc_variance(output)
        z = self.reparameterize(mean, log_var)
        output = self.fc_reparameterize(z)

        seq_length, batch_size, feature = output.size()
        output = output.view(batch_size * seq_length, feature)
        output = self.bn_latent(output)
        output = output.view(seq_length, batch_size, feature)

        if domain == "band_prediction":  # 디코더 학습 및 추론
            dec_obs = self.decoder_embedding(dec_obs)
            dec_obs = dec_obs.permute(1, 0, 2)
            dec_obs = self.pos_encoder(dec_obs)
            decoder_output = self.transformer_decoder(
                dec_obs,
                output,
                tgt_key_padding_mask=dec_pad_mask,
            )
            decoder_output = decoder_output.permute(1, 0, 2)

            fc_up_brg = self.fc_up_brg(decoder_output)
            output_vector_up = self.fc_up(fc_up_brg)
            fc_down_brg = self.fc_down_brg(decoder_output)
            output_vector_down = self.fc_down(fc_up_brg)

            # option 1 - concat 후 어텐션, 판별
            # cc = torch.concat([fc_up_brg, fc_down_brg], dim=-1)
            # up_down_attn, _ = self.up_down_attn(cc, cc, cc)
            # output_vector_up_down = self.fc_up_down(up_down_attn)
            # output_vector_std = self.fc_std(up_down_attn)

            # option 2 - concat 후, 압축, 어텐션, 판별
            # cc = torch.concat([fc_up_brg, fc_down_brg], dim=-1)
            # cc = self.mp1d(cc)
            # up_down_attn, _ = self.up_down_attn(cc, cc, cc)
            # output_vector_up_down = self.fc_up_down(up_down_attn)
            # output_vector_std = self.fc_std(up_down_attn)

            # option 3 - concat 후 압축, 판별
            cc = torch.concat([fc_up_brg, fc_down_brg], dim=-1)
            cc = self.mp1d(cc)
            output_vector_up_down = self.fc_up_down(cc)
            output_vector_std = self.fc_std(cc)

            # option 4 - decoder_output 을 공용벡터로 그냥 up, down, up_down, std 한꺼번에 구하기
            # 코드 구현 전, 꼭 테스트 해보기 - (hiddens, 4) -> splite (hiddens, 1) X 4

            return (
                output_vector_up,
                output_vector_down,
                output_vector_std,
                output_vector_up_down,
            )

        if domain == "context":  # 인코더 학습 및 추론
            output_vector = self.fc(output.permute(1, 0, 2))
            return (
                output_vector,
                mean,
                log_var,
                z,
            )
