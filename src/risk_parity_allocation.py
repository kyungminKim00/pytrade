import numpy as np
import pandas as pd
from scipy.optimize import minimize

# 가정: returns는 자산 수익률 데이터프레임입니다.
returns = pd.read_csv("asset_returns.csv", index_col=0)


# 최적화 목표 함수: 각 자산의 리스크 기여도 차이의 제곱합을 최소화
def objective(weights, cov_matrix):
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    marginal_risk_contribution = np.dot(cov_matrix, weights) / portfolio_volatility
    risk_contribution = weights * marginal_risk_contribution
    return np.sum((risk_contribution - risk_contribution.mean()) ** 2)


# 제약 조건: 자산 가중치 합계는 1
cons = {"type": "eq", "fun": lambda x: np.sum(x) - 1}

# 가중치의 범위: 각 자산의 가중치는 0과 1 사이
bounds = tuple((0, 1) for asset in range(returns.shape[1]))

# 초기 가중치: 균등 가중치로 시작
init_weights = np.array([1 / returns.shape[1]] * returns.shape[1])

# 공분산 행렬 계산
cov_matrix = returns.cov()

# 최적화 실행
optimal_weights = minimize(
    objective,
    init_weights,
    args=(cov_matrix,),
    method="SLSQP",
    bounds=bounds,
    constraints=cons,
)

print("Optimal weights:", optimal_weights.x)
