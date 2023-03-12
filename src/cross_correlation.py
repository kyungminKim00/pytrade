import bottleneck as bn
import numpy as np
import pandas as pd
import plotly.express as px
import ray
import umap
from sklearn.decomposition import PCA

from util import print_c


@ray.remote(num_cpus=4)
def calc_corr(x_col, X, Y, correation_bin):
    corr = np.zeros(X.shape[0])
    for k in range(correation_bin, X.shape[0]):
        x_w = X[k - correation_bin : k, x_col]
        y_w = Y[k - correation_bin : k]
        # Calculate the normalized cross-correlation using the NCC formula
        numerator = np.sum((x_w - np.mean(x_w)) * (y_w - np.mean(y_w)))
        denominator = np.sqrt(
            np.sum((x_w - np.mean(x_w)) ** 2) * np.sum((y_w - np.mean(y_w)) ** 2)
        )
        corr[k] = numerator / denominator
    return corr[correation_bin:]


class CrossCorrelation:
    def __init__(
        self,
        mv_bin,
        correation_bin,
        x_file_name,
        y_file_name,
        debug,
        data_tranform=None,
        ratio=0.8,
        alpha=2,
    ):
        # missing values
        self.X = self.fill_data(x_file_name)
        self.Y = self.fill_data(y_file_name)
        self.var_desc = pd.read_csv("./src/local_data/raw/var_description.csv")

        # common datetime
        common_df = self.X.join(self.Y, how="inner")
        self.X = common_df.iloc[:, :-1].values
        self.Y = common_df.iloc[:, -1].values

        # column names and index information
        self.col2idx = list(
            zip(list(common_df.columns), np.arange(len(common_df.columns)))
        )
        # option. 컬럼 이름 full name 으로 표시
        self.col2idx_fulname = []
        for _, it in enumerate(self.col2idx):
            name = self.var_desc.query(f"varcode==@it[0]")["name"].to_list()
            self.col2idx_fulname.append((name[0], it[1]))
        self.col2idx = self.col2idx_fulname

        self.x_col2idx = {it[0]: it[1] for it in self.col2idx[:-1]}
        self.y_col2idx = {self.col2idx[-1][0]: self.col2idx[-1][1]}
        self.x_idx2col = dict(zip(self.x_col2idx.values(), self.x_col2idx.keys()))
        self.y_idx2col = dict(zip(self.y_col2idx.values(), self.y_col2idx.keys()))

        self.X_ma = bn.move_mean(self.X, axis=0, window=mv_bin, min_count=1)
        self.Y_ma = bn.move_mean(self.Y, axis=0, window=mv_bin, min_count=1)

        self.num_vars = self.X.shape[1]
        self.mv_bin = mv_bin
        self.correation_bin = correation_bin

        """다음 코드는 인덱스 정렬이 포함 되어 있음. 코드 위치 변경시 자세히 살펴 봐야 함
        """
        # Obervations
        self.observatoins, self.Y = self.ncc_windowed()

        # mendatory mask
        self.forward_returns, self.observatoins = self.get_forward_returns(
            self.Y, n_periods=60
        )
        self.predefined_mask = self.std_idx(alpha=alpha)
        self.num_sample = self.observatoins.shape[0]
        self.weight_variables = np.ones(self.observatoins.shape[1])

        self.reducer = None
        if data_tranform is not None:
            if data_tranform["n_components"] == np.inf:
                data_tranform["n_components"] = self.observatoins.shape[1]

            self.observatoins, self.weight_variables, self.reducer = self.reduction_dim(
                n_components=data_tranform["n_components"],
                ratio=ratio,
                method=data_tranform["method"],
            )

        # 예약어 np.nanmax(self.observatoins) * 2 + np.nanstd(self.observatoins) * 10
        # 동적 할당 절대 안 됨. 문제의 소지가 많음
        self.padding_torken = 0.00779

        assert (
            np.sum(self.observatoins == self.padding_torken) == 0
        ), "A padding token is a reserved word and must have a unique value"
        assert (
            self.observatoins.shape[0] == self.forward_returns.shape[0]
        ), "observatoins, forward_returns and predefined_mask must have the same length"

        # 인덱스 합치기 (observatoins, mask)
        self.observatoins_merge_idx = np.zeros(self.observatoins.shape[0])
        self.observatoins_merge_idx[self.predefined_mask] = 1
        self.observatoins_merge_idx = self.observatoins_merge_idx[:, None]
        self.observatoins_merge_idx = np.concatenate(
            (self.observatoins, self.observatoins_merge_idx), axis=1
        )

        # summary statistics
        print_c(
            f"observatoins_merge_idx shape: {self.observatoins_merge_idx.shape} \
            nums of masked samples: {self.observatoins_merge_idx[:, -1].sum()} \
            ratio of masked samples: {self.observatoins_merge_idx[:, -1].sum() / self.observatoins_merge_idx.shape[0]}"
        )

        # validation data & data visualization
        if debug:
            assert (
                data_tranform is None
            ), "데이터 탐색용 함수, data_tranform is None으로 두고 데이터 탐색"

            for col in common_df.columns:
                fig = px.line(common_df, x=common_df.index, y=col)
                fig.write_image(
                    f"./src/local_data/assets/plot_check/cross_correlation_{col}.png"
                )
            self.plot_histogram()
            self.plot_pca()

    def reduction_dim(self, n_components, ratio, method):
        train_samples = self.observatoins[: int(self.observatoins.shape[0] * ratio)]
        if method == "PCA":
            pca = PCA(n_components=n_components)
            reducer = pca.fit(train_samples)

            transformed_data = reducer.transform(self.observatoins)
            explained_variance_ratio = reducer.explained_variance_ratio_
        elif method == "UMAP":
            umap_i = umap.UMAP(n_components=n_components)
            reducer = umap_i.fit(train_samples)

            transformed_data = reducer.transform(self.observatoins)
            variances = np.var(transformed_data, axis=0)
            explained_variance_ratio = variances / variances.sum()
        else:
            assert False, "Not implemented method"

        print_c(
            f"[n_components={n_components}] \n\
                        explained_variance_ratio: {explained_variance_ratio} \n\
                            explained_variance:{sum(explained_variance_ratio)}"
        )
        assert (
            explained_variance_ratio.sum() > 0.95
        ), "[at cross_cprreation.py] increase n_components"

        return transformed_data, np.array(explained_variance_ratio), reducer

    def std_idx(self, alpha=2):
        _std = self.forward_returns.std() * alpha
        _t_idx = np.argwhere(np.abs(self.forward_returns) > _std)
        return _t_idx.reshape(-1)

    def fill_data(self, fn):
        df = pd.read_csv(fn)
        df.set_index("Date", inplace=True)
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        return df

    def ncc_windowed(self):
        futures = []
        for i in range(self.num_vars):
            futures.append(
                calc_corr.remote(i, self.X_ma, self.Y_ma, self.correation_bin)
            )
        results = np.array(ray.get(futures)).T
        # # padding 제거
        # zeros = np.all(results == 0, axis=1)
        # results = results[~zeros]

        self.Y = self.Y[self.correation_bin :]

        return (results / (np.nanstd(self.X, axis=0) * np.nanstd(self.Y))), self.Y

    def plot_histogram(self):
        for idx in range(self.observatoins.shape[1]):
            data = self.observatoins[:, idx]
            fig = px.histogram(data, title=f"{self.x_idx2col[idx]}")
            fig.write_image(
                f"./src/local_data/assets/plot_check/histogram_{self.x_idx2col[idx]}.png"
            )

    def get_forward_returns(self, Y, n_periods=60):
        _forward_returns = np.zeros_like(Y)
        for i in range(len(Y) - n_periods):
            _forward_returns[i] = (Y[i + n_periods] - Y[i]) / Y[i]

        # align index
        _forward_returns = _forward_returns[:-n_periods]
        self.observatoins = self.observatoins[:-n_periods, :]

        return _forward_returns, self.observatoins

    def plot_pca(self):
        # perform PCA on the observations
        pca = PCA(n_components=2)
        obs_pca = pca.fit_transform(self.observatoins)

        # align data
        data = obs_pca
        p1_data = obs_pca[:, 0]
        p2_data = obs_pca[:, 1]

        # ad-hoc for analysis
        for alpha in [1, 1.5, 2, 2.5, 3, 3.5, 4]:
            target_idx = self.std_idx(alpha)

            fig = px.scatter(
                data,
                x=p1_data[target_idx],
                y=p2_data[target_idx],
                color=self.forward_returns[target_idx],
                title=f"label std: {alpha}",
            )
            fig.write_image(f"./src/local_data/assets/plot_check/pca_2D_{alpha}.png")
