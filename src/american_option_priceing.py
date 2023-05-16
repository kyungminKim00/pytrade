def binomial_tree(option_type, S, X, T, r, sigma, N):
    """
    바이너리 트리 모델을 사용하여 유러피안 옵션의 가치를 계산합니다.

    Parameters:
        option_type (str): 콜 옵션인 경우 "call", 풋 옵션인 경우 "put"
        S (float): 현재 주가
        X (float): 행사가격
        T (float): 잔여 만기 기간 (연 단위로 표시된 기간)
        r (float): 무위험 이자율
        sigma (float): 주가 변동성
        N (int): 시간 이산화를 위한 트리의 단계 수

    Returns:
        float: 옵션의 가치
    """
    dt = T / N  # 단계당 시간 간격
    u = np.exp(sigma * np.sqrt(dt))  # 상승 비율
    d = 1 / u  # 하락 비율
    p = (np.exp(r * dt) - d) / (u - d)  # 상승 확률

    # 주가 그리드 생성
    stock_prices = np.zeros((N + 1, N + 1))
    stock_prices[0, 0] = S
    for i in range(1, N + 1):
        stock_prices[i, 0] = stock_prices[i - 1, 0] * u
        for j in range(1, i + 1):
            stock_prices[i, j] = stock_prices[i - 1, j - 1] * d

    # 가치 그리드 생성
    option_values = np.zeros((N + 1, N + 1))
    for j in range(N + 1):
        if option_type == "call":
            option_values[N, j] = max(stock_prices[N, j] - X, 0)
        elif option_type == "put":
            option_values[N, j] = max(X - stock_prices[N, j], 0)

    # 가치 역산
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            option_values[i, j] = np.exp(-r * dt) * (
                p * option_values[i + 1, j] + (1 - p) * option_values[i + 1, j + 1]
            )

    return option_values[0, 0]


# 옵션 가치 계산 예시
call_price = binomial_tree("call", 100, 110, 1, 0.05, 0.2, 100)
put_price = binomial_tree("put", 100, 110, 1, 0.05, 0.2, 100)

print("콜 옵션의 가치:", call_price)
print("풋 옵션의 가치:", put_price)
