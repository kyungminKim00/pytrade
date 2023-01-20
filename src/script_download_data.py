import pandas as pd
import yfinance as yf

if __name__ == "__main__":
    # Get the data
    """
    Ticker
    QQQ: NASDAQ
    ^GSPC: S&P500
    ^KS11: KOSPI
    ^TNX: US government 10
    ^IXIC: NASDAQ
    ^GDAXI: DAX
    ^HSI: HANG SENG INDEX
    ^N225: Nikkei 225
    gc=F: gold future
    SI=F: silver futures
    HG=F: copper futures
    PL=F: platinum futures
    CL=F: crude oil wti futures
    ZS=F: soybean futures
    QA=F: crude oil brent futures

    여기 부터는 내일 정리
    RB=F gasoline futures
    NG=F natural gas futures
    ZC=F corn futures
    ZW=F wheat futures
    DX=F dollar index futures

    """

    # # KOSPI 시간봉
    # ticker = "^KS11"
    # data = yf.download(tickers=ticker, period="max", interval="1h")
    # pd.DataFrame(data).to_csv("./src/local_data/raw/kospi_1h.csv")

    # Dax 일봉
    ticker = "^GDAXI"
    data = yf.download(tickers=ticker, start="2010-01-01", interval="1d")
    pd.DataFrame(data).to_csv("./src/local_data/raw/dax_td1.csv")

    # # Demo Data 일봉 / close
    # ticker = ["^GSPC", "^KS11", "^TNX", "^IXIC", "^GDAXI", "^HSI", "^N225", "gc=F", "SI=F", "HG=F", "PL=F", "CL=F", "ZS=F", "QA=F", "RB=F", "NG=F", "ZC=F", "ZW=F", "DX=F"]
    # data = yf.download(tickers=ticker, start="1998-05-01", interval="1d")
    # data = data["Close"]
    # pd.DataFrame(data).to_csv(f"./src/local_data/raw/tt/demo_td1.csv")
