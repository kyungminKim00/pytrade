import matplotlib.pyplot as plt  # 데이터를 시각화하기 위한 라이브러리
import numpy as np  # 수치 연산을 위한 라이브러리
import pandas as pd  # 데이터 처리를 위한 라이브러리
import scipy.optimize as opt  # 최적화를 위한 라이브러리
import yfinance as yf
from scipy.stats import norm


# [Markowitz, H. M. (1952). Portfolio selection. The journal of finance, 7(1), 77-91.]
# mean_varience_allocation 함수를 최적화하기 위한 제약 조건 함수를 정의합니다.
def mean_varience_allocation(weights, returns):
    portfolio_return = np.sum(returns.mean() * weights) * 252  # 연간 포트폴리오 수익률
    portfolio_volatility = np.sqrt(
        np.dot(weights.T, np.dot(returns.cov() * 252, weights))
    )  # 연간 포트폴리오 변동성

    return portfolio_volatility


# Var 최대화를 위한 목적 함수를 정의합니다.
def var_allocation(weights, returns, confidence_level=0.05):
    portfolio_return = np.sum(returns.mean() * weights) * 252  # 연간 포트폴리오 수익률
    portfolio_volatility = np.sqrt(
        np.dot(weights.T, np.dot(returns.cov() * 252, weights))
    )  # 연간 포트폴리오 변동성

    # 추가: Value at Risk (VaR) 최대화를 위한 조건
    z_score = norm.ppf(confidence_level)
    portfolio_var = portfolio_volatility * z_score

    # 포트폴리오의 VaR을 최대화하는 목표 함수 계산
    objective_function = -portfolio_var

    # return portfolio_volatility
    return objective_function


# 종목 코드 설정
symbols = {
    "브랜트유": "BZ=F",
    "나스닥": "^IXIC",
    "10년만기미국채선물": "ZN=F",
    "달러인덱스": "DX-Y.NYB",
    "VIX_INX": "^VIX",
}

data = yf.download(list(symbols.values()), period="max")["Close"]  # 종가 데이터 다운로드
data = data.dropna()  # null 값 제거
data.index = pd.to_datetime(data.index).strftime("%Y-%m-%d")  # 인덱스를 트레이딩 날짜로 설정

optimal_allocation = mean_varience_allocation


def invarient_risk_allocation():
    # 자산 수익률 데이터를 불러오고, 퍼센트 변화율을 계산한 뒤, 결측값을 제거합니다.
    returns = data
    returns = returns.pct_change().dropna()
    num_variables = len(returns.columns)  # 자산의 수

    # 최적화 함수를 최소화하여 최적의 자산 배분을 찾습니다.
    bounds = [(0, 1)] * num_variables  # 각 자산의 가중치 범위는 0과 1 사이입니다.
    initial_guess = [np.ceil(1 / num_variables)] * num_variables  # 초기 가중치 추정치
    cons = {"type": "eq", "fun": lambda x: np.sum(x) - 1}  # 모든 가중치의 합이 1이 되도록 하는 제약조건
    optimal_result = opt.minimize(
        fun=optimal_allocation,
        x0=initial_guess,
        bounds=bounds,
        constraints=cons,
        args=(returns,),
        method="SLSQP",  # 제약조건이 있는 최적화 문제를 풀기 위한 알고리즘
        options={"disp": True},
    )

    # 최적의 가중치를 추출하고, 해당 가중치에 대한 포트폴리오 통계를 계산합니다.
    optimal_weights = optimal_result.x
    portfolio_return = np.sum(returns.mean() * optimal_weights) * 252
    portfolio_volatility = np.sqrt(
        np.dot(optimal_weights.T, np.dot(returns.cov() * 252, optimal_weights))
    )

    # 효율적 투자선을 그립니다.
    frontier_y = np.linspace(0, 0.3, 100)  # 가능한 수익률 범위 100%까지
    frontier_volatility = []  # 각 수익률에 대한 최소 변동성을 저장할 리스트
    frontier_dict = {}
    for possible_return in frontier_y:
        cons = (
            {"type": "eq", "fun": lambda x: np.sum(x) - 1},  # 모든 가중치의 합이 1이 되도록 하는 제약조건
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
            method="SLSQP",  # 제약조건이 있는 최적화 문제를 풀기 위한 알고리즘
            options={"disp": True},
        )
        frontier_volatility.append(result["fun"])  # 해당 수익률에 대한 최소 변동성을 저장합니다.
        frontier_dict[possible_return] = result.x

    # 효율적 투자선과 최적 포트폴리오를 그립니다.
    plt.plot(
        frontier_volatility, frontier_y, "b-", label="Efficient Frontier"
    )  # 효율적 투자선 그리기
    plt.title(f"{list(data.columns)}: {optimal_weights}")
    plt.scatter(
        portfolio_volatility,
        portfolio_return,
        c="r",
        marker="o",
        label="Optimal Portfolio",  # 최적 포트폴리오 표시하기
    )
    plt.xlabel("Volatility")  # x축 레이블 설정
    plt.ylabel("Return")  # y축 레이블 설정
    plt.legend(loc="upper right")  # 범례 위치 설정
    plt.savefig(
        "./src/local_data/assets/efficient_frontier.jpeg", format="jpeg"
    )  # JPEG 파일로 저장
    print(frontier_dict)


invarient_risk_allocation()
