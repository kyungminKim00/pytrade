import joblib
import modin.pandas as pd
import numpy as np
import ray
from sklearn.preprocessing import KBinsDiscretizer


class QuantileDiscretizer:
    def __init__(self, df: pd.DataFrame):
        self.mean = df.mean(axis=0)
        self.std = df.std(axis=0, ddof=1)

        # Clip outliers using mean and standard deviation
        clipped_vectors = df.clip(self.mean - 3 * self.std, self.mean + 3 * self.std)
        # Scott's rule: Determine the bin size
        n_bins = 3.5 * self.std * np.power(len(df.shape[0]), -1 / 3)

        # KBinsDiscretizer 을 특징별로 두면 코드가 너무 복잡 해짐
        self.n_bins = np.max(n_bins)

        self.discretizer = KBinsDiscretizer(
            n_bins=self.n_bins, encode="ordinal", strategy="uniform"
        ).fit(clipped_vectors)

    def quantized_vectors(self, vectors):
        return self.discretizer.transform(vectors)


# # Convert the quantized vectors into decimal values
# @ray.remote
# def decimal_conversion(vector, n_bins):
#     decimal = 0
#     for i, value in enumerate(vector):
#         decimal += value * (n_bins**i)
#     return decimal


def decimal_conversion(vector, n_bins):
    decimal = 0
    for i, value in enumerate(vector):
        decimal += value * (n_bins**i)
    return decimal


# convert the quantized vectors into decimal values
def convert_to_index(df: pd.DataFrame, fit_discretizer: bool = False) -> pd.Series:
    if fit_discretizer:
        joblib.dump(QuantileDiscretizer(df), "./assets/discretizer.pkl")

    qd = joblib.load("./assets/discretizer.pkl")
    obj_dict = joblib.load("./assets/obj_dict.pkl")

    clipped_vectors = df.clip(qd.mean - 3 * qd.std, qd.mean + 3 * qd.std)

    # # Convert the quantized vectors into decimal values
    # decimal_vectors = ray.get(
    #     [
    #         str(decimal_conversion.remote(vector, qd.n_bins))
    #         for vector in qd.quantized_vectors(clipped_vectors)
    #     ]
    # )
    # Convert the quantized vectors into decimal values
    decimal_vectors = [
        str(decimal_conversion(vector, qd.n_bins))
        for vector in qd.quantized_vectors(clipped_vectors)
    ]

    def id_from_dict(pttn):
        max_number = len(obj_dict)
        key = "_".join(pttn)
        try:
            p_id = obj_dict[key]
        except KeyError:  # add new pattern
            obj_dict[key] = max_number + 1
            p_id = max_number + 1

            joblib.dump(obj_dict, "./assets/obj_dict.pkl")
        return p_id

    pattern_id = list(map(id_from_dict, decimal_vectors))

    return pd.Series(pattern_id, index=df.index)
