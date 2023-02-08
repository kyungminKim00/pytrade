import matplotlib.pyplot as plt
import modin.pandas as pd
import scipy.optimize as opt


def invarient_risk_allocation():

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
