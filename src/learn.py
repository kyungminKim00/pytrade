import pandas as pd
import yfinance as yf

if __name__ == "__main__":
    # Get the data
    """
    Ticker
    ^GSPC: S&P500
    ^KS11: KOSPI
    QQQ: NASDAQ
    ^GDAXI: DAX
    """

    ticker = "^GDAXI"
    ticker = "QQQ"

    # KOSPI   시간봉
    # data = yf.download(tickers=ticker, period="max", interval="1h")
    # pd.DataFrame(data).to_csv("./dataset/raw/KS11_1h.csv")

    # Dax 일봉
    data = yf.download(tickers=ticker, start="2014-01-01", interval="1d")
    pd.DataFrame(data).to_csv("./dataset/raw/dax_td1.csv")
