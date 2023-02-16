import modin.pandas as pd
import numpy as np
from joblib import dump, load
from sklearn.preprocessing import KBinsDiscretizer


class QuantileDiscretizer:
    def __init__(self, df: pd.DataFrame):

        drop_column = [c for c in df.columns if not "spd" in c]
        df = df.drop(columns=drop_column)

        self.mean = df.mean(axis=0)
        self.std = df.std(axis=0, ddof=1)

        # Clip outliers using mean and standard deviation
        # self.clipped_vectors = df.clip(
        #     self.mean - 3 * self.std, self.mean + 3 * self.std, axis=1
        # )

        self.clipped_vectors = df.clip(
            self.mean - 0.5 * self.std, self.mean + 0.5 * self.std, axis=1
        )

        # Scott's rule: Determine the bin size
        n_bins = 3.5 * self.std * np.power(df.shape[0], -1 / 3)

        df.to_csv("./source.csv")
        self.mean.to_csv("./mean.csv")
        self.std.to_csv("./std.csv")
        self.clipped_vectors.to_csv("./std.csv")
        n_bins.to_csv("./n_bins.csv")

        # # KBinsDiscretizer 을 특징별로 두면 코드가 너무 복잡 해짐
        # self.n_bins = np.max(n_bins)

    def discretizer_learn_save(self, obj_fn: str):
        discretizer = {
            "model": KBinsDiscretizer(
                n_bins=self.n_bins, encode="ordinal", strategy="uniform"
            ).fit(self.clipped_vectors),
            "mean": self.mean,
            "std": self.std,
            "n_bins": self.n_bins,
        }
        dump(discretizer, obj_fn)

    # def quantized_vectors(self, vectors):
    #     return self.discretizer.transform(vectors)


# # Convert the quantized vectors into decimal values
# @ray.remote
# def decimal_conversion(vector, n_bins):
#     decimal = 0
#     for i, value in enumerate(vector):
#         decimal += value * (n_bins**i)
#     return decimal


# def quantized_vectors(vectors):
#     return self.discretizer.transform(vectors)
