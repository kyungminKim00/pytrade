import datetime
import threading
import time

import schedule
import yfinance as yf
from pandas_datareader import data as pdr

kospi = "^KS11"
nasdaq = "^IXIC"

yf.pdr_override()


# 0.5 sec cut, retry
# every min try
def getData():
    time_start = time.time()

    data = pdr.get_data_yahoo(nasdaq, period="2m", interval="1m")
    # max 391 length

    now = datetime.datetime.now()
    now_time = now.strftime("%Y-%m-%d %H:%M:%S")

    print(now_time)
    print("running time :", time.time() - time_start)
    print("data")
    print(data)
    print("\n")


if __name__ == "__main__":
    schedule.every(60).seconds.do(getData)
    while True:
        schedule.run_pending()
        time.sleep(1)


# data = pdr.get_data_yahoo(nasdaq,period="1w", interval = "1m")
# data.to_csv('nasdaq_1w_1m_data.csv')

# """
# time_start = time.time()
# ndq = yf.Ticker(nasdaq)
# data = ndq.history(period="2m", interval = "1m")
# print(time.time()-time_start)
# print(data)
# """


# """
# import matplotlib.pyplot as pyplot
# pyplot.plot(data['Close'])
# pyplot.show()
# """
