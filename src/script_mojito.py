import mojito
import pprint
import json
import pandas as pd


if __name__ == "__main__":
    # 모듈 정보
    with open("./src/mojito.json", "r", encoding="utf-8") as fp:
        env_dict = json.load(fp)

    broker = mojito.KoreaInvestment(
        api_key=env_dict["v_key"],
        api_secret=env_dict["v_secret"],
        acc_no=env_dict["v_acc_no"],
        mock=env_dict["v_mock"],
        exchange=env_dict["exchange"],
    )

    resp = broker.fetch_price("IONQ")
    pprint.pprint(resp)

    balance = broker.fetch_present_balance()
    print(balance)

    symbols = broker.fetch_symbols()

    # result = broker.fetch_oversea_day_night()
    # pprint.pprint(result)

    # minute1_ohlcv = broker.fetch_today_1m_ohlcv("005930")
    # pprint.pprint(minute1_ohlcv)

    # broker = KoreaInvestment(key, secret, exchange="나스닥")
    # import pprint
    # resp = broker.fetch_price("005930")
    # pprint.pprint(resp)
    #
    # b = broker.fetch_balance("63398082")
    # pprint.pprint(b)
    #
    # resp = broker.create_market_buy_order("63398082", "005930", 10)
    # pprint.pprint(resp)
    #
    # resp = broker.cancel_order("63398082", "91252", "0000117057", "00", 60000, 5, "Y")
    # print(resp)
    #
    # resp = broker.create_limit_buy_order("63398082", "TQQQ", 35, 1)
    # print(resp)

    # 실시간주식 체결가
    # broker_ws = KoreaInvestmentWS(
    #   key, secret, ["H0STCNT0", "H0STASP0"], ["005930", "000660"], user_id="idjhh82")
    # broker_ws.start()
    # while True:
    #    data_ = broker_ws.get()
    #    if data_[0] == '체결':
    #        print(data_[1])
    #    elif data_[0] == '호가':
    #        print(data_[1])
    #    elif data_[0] == '체잔':
    #        print(data_[1])

    # 실시간주식호가
    # broker_ws = KoreaInvestmentWS(key, secret, "H0STASP0", "005930")
    # broker_ws.start()
    # for i in range(3):
    #    data = broker_ws.get()
    #    print(data)
    #
    # 실시간주식체결통보
    # broker_ws = KoreaInvestmentWS(key, secret, "H0STCNI0", "user_id")
    # broker_ws.start()
    # for i in range(3):
    #    data = broker_ws.get()
    #    print(data)

    # import pprint
    # broker = KoreaInvestment(key, secret, exchange="나스닥")
    # resp_ohlcv = broker.fetch_ohlcv("TSLA", '1d', to="")
    # print(len(resp_ohlcv['output2']))
    # pprint.pprint(resp_ohlcv['output2'][0])
    # pprint.pprint(resp_ohlcv['output2'][-1])
