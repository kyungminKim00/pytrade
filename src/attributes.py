import modin.pandas as pd


def spread(
    df: pd.DataFrame, final_price: str = None, init_price: str = None
) -> pd.Series:
    return pd.Series(
        (df[final_price] - df[init_price]) / df[init_price],
        index=df.index,
    )


def diff(
    df: pd.DataFrame, final_price: str = None, init_price: str = None
) -> pd.Series:
    return pd.Series(
        (df[final_price] - df[init_price]),
        index=df.index,
    )


def conver_to_index():
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import KBinsDiscretizer

    # Generate 30 random vectors with 5 dimensions
    vectors = np.random.rand(30, 5)

    # Clip outliers using mean and standard deviation
    mean = np.mean(vectors, axis=0)
    std = np.std(vectors, axis=0)
    clipped_vectors = np.clip(vectors, mean - 3 * std, mean + 3 * std)

    # Quantize each dimension into 8 categories using KBinsDiscretizer
    discretizer = KBinsDiscretizer(n_bins=8, encode="ordinal", strategy="uniform")
    quantized_vectors = discretizer.fit_transform(clipped_vectors)

    # Convert the quantized vectors into decimal values
    def decimal_conversion(vector):
        decimal = 0
        for i, value in enumerate(vector):
            decimal += value * (8**i)
        return decimal

    decimal_vectors = [decimal_conversion(vector) for vector in quantized_vectors]
