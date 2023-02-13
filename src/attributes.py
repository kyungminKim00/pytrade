# import numpy as np
import modin.pandas as pd
import ray


@ray.remote
def spread_parallel(analyse_data, f_prc, i_prc):
    return spread(analyse_data, f_prc, i_prc)


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
