import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
import modin.pandas as pd


class QuantileDiscretizer:
    def __init__(self, df: pd.DataFrame):
        self.mean = df.mean(axis=0)
        self.std = df.std(axis=0, ddof=1)

        # Clip outliers using mean and standard deviation
        clipped_vectors = df.clip(self.mean - 3 * self.std, self.mean + 3 * self.std)
        # Scott's rule: Determine the bin size
        n_bins = 3.5 * self.std * np.power(len(df.shape[0]), -1 / 3)
        # add: check later if we need to do this
        self.n_bins = np.max(n_bins)

        self.discretizer = KBinsDiscretizer(
            n_bins=self.n_bins, encode="ordinal", strategy="uniform"
        ).fit(clipped_vectors)

    def quantized_vectors(self, vectors):
        return self.discretizer.transform(vectors)
