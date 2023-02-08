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

    """Freedman-Diaconis rule and Scott's rule (pros and cons)

    Freedman-Diaconis rule
    Pros:
    The Freedman-Diaconis rule is less sensitive to outliers in the data, making it well-suited for datasets with large amounts of skewness or variability.
    It is relatively easy to compute, requiring only the interquartile range and the number of data points.
    Cons of Freedman-Diaconis rule:
    
    Cons:
    The Freedman-Diaconis rule can sometimes result in fewer bins than necessary, causing the histogram to be less detailed or informative.
    It is not as commonly used as the Scott's rule and therefore may be less well-known or familiar to practitioners.
    
    Scott's rule
    Pros:
    The Scott's rule is widely used and well-known in the data science community, making it easy to implement and interpret.
    It tends to produce a good balance between detail and simplicity in histograms, providing enough information to understand the data while avoiding over-complexity.
    Cons of Scott's rule:

    Cons:
    The Scott's rule can sometimes result in more bins than necessary, causing the histogram to be overly detailed or difficult to interpret.
    It can be sensitive to outliers in the data, leading to histograms that do not accurately represent the distribution of the data.
    """
    # Quantize each dimension into categories using KBinsDiscretizer with Freedman-Diaconis rule
    iqr = np.subtract(*np.percentile(clipped_vectors, [75, 25], axis=0))
    bin_width = 2 * iqr / np.cbrt(len(clipped_vectors))
    n_bins = np.ceil(
        (clipped_vectors.max(axis=0) - clipped_vectors.min(axis=0)) / bin_width
    )

    # Determine the bin size using Scott's rule
    n_bins = 3.5 * std * np.power(len(clipped_vectors), -1 / 3)

    discretizer = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="uniform")
    quantized_vectors = discretizer.fit_transform(clipped_vectors)

    # Convert the quantized vectors into decimal values
    def decimal_conversion(vector):
        decimal = 0
        for i, value in enumerate(vector):
            decimal += value * (n_bins**i)
        return decimal

    decimal_vectors = [decimal_conversion(vector) for vector in quantized_vectors]


def invarient_risk_allocation():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
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
