import numpy as np
import modin.pandas as pd
import ray
import joblib
from quantile_discretizer import QuantileDiscretizer


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


# Convert the quantized vectors into decimal values
@ray.remote
def decimal_conversion(vector, n_bins):
    decimal = 0
    for i, value in enumerate(vector):
        decimal += value * (n_bins**i)
    return decimal


def conver_to_index(df: pd.DataFrame, fit_discretizer: bool = False) -> pd.Series:
    if fit_discretizer:
        joblib.dump(QuantileDiscretizer(df), "./discretizer.pkl")

    qd = joblib.load("./discretizer.pkl")
    clipped_vectors = df.clip(qd.mean - 3 * qd.std, qd.mean + 3 * qd.std)

    # Convert the quantized vectors into decimal values
    decimal_vectors = ray.get(
        [
            decimal_conversion.remote(vector, qd.n_bins)
            for vector in qd.quantized_vectors(clipped_vectors)
        ]
    )
    return pd.Series(decimal_vectors, index=df.index)


def invarient_risk_allocation():
    import matplotlib.pyplot as plt
    import scipy.optimize as opt

    # Load the asset data and calculate returns
    returns = pd.read_csv("asset_returns.csv", index_col=0)
    returns = returns.pct_change().dropna()

    # Define the optimization function
    def optimal_allocation(weights, returns):
        portfolio_return = np.sum(returns.mean() * weights) * 252
        portfolio_volatility = np.sqrt(
            np.dot(weights.T, np.dot(returns.cov() * 252, weights))
        )
        return portfolio_volatility

    # Minimize the optimization function to find the optimal allocation
    bounds = [(0, 1), (0, 1), (0, 1)]
    initial_guess = [0.33, 0.33, 0.33]
    cons = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    optimal_result = opt.minimize(
        fun=optimal_allocation,
        x0=initial_guess,
        bounds=bounds,
        constraints=cons,
        args=(returns,),
        method="SLSQP",
        options={"disp": True},
    )

    # Extract the optimal weights and calculate the corresponding portfolio statistics
    optimal_weights = optimal_result.x
    portfolio_return = np.sum(returns.mean() * optimal_weights) * 252
    portfolio_volatility = np.sqrt(
        np.dot(optimal_weights.T, np.dot(returns.cov() * 252, optimal_weights))
    )

    # Plot the efficient frontier
    frontier_y = np.linspace(0, 0.15, 100)
    frontier_volatility = []
    for possible_return in frontier_y:
        cons = (
            {"type": "eq", "fun": lambda x: np.sum(x) - 1},
            {
                "type": "eq",
                "fun": lambda x: np.sum(returns.mean() * x) * 252 - possible_return,
            },
        )
        result = opt.minimize(
            fun=optimal_allocation,
            x0=initial_guess,
            bounds=bounds,
            constraints=cons,
            args=(returns,),
            method="SLSQP",
            options={"disp": True},
        )
        frontier_volatility.append(result["fun"])

    plt.plot(frontier_volatility, frontier_y, "b-", label="Efficient Frontier")
    plt.scatter(
        portfolio_volatility,
        portfolio_return,
        c="r",
        marker="o",
        label="Optimal Portfolio",
    )
    plt.xlabel("Volatility")
    plt.ylabel("Return")
    plt.legend(loc="upper right")
    plt.show()
