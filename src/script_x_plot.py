import random
from multiprocessing import Manager
from time import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import ray
from joblib import dump, load
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_reader import DataReader
from preprocess import SequentialDataSet
from quantile_discretizer import QuantileDiscretizer
from util import print_c, print_flush

candle_size = (1, 3, 5, 15, 60)
w_size = (9, 50, 100)
# w_size = (9, 50)
alpha = 3.5


# 전처리 완료 데이터
offset = 35000  # small data or operating data
# offset = None  # practical data

sequential_data = SequentialDataSet(
    raw_filename_min="./src/local_data/raw/dax_tm3.csv",
    pivot_filename_day="./src/local_data/intermediate/dax_intermediate_pivots.csv",
    candle_size=candle_size,
    w_size=w_size,
    debug=False,
    offset=offset,
)
dump(sequential_data, "./src/local_data/assets/sequential_data.pkl")

# 전처리 완료 데이터 로드
processed_data = load("./src/local_data/assets/sequential_data.pkl")

# 변수 설정
x_real = [c for c in processed_data.train_data.columns if "feature" in c]
y_real = ["y_rtn_close"]

# 이산화 모듈 저장
qd = QuantileDiscretizer(processed_data.train_data, x_real, alpha=alpha)
qd.discretizer_learn_save("./src/local_data/assets/discretizer.pkl")

# 이산화 모형 로드
discretizer = load("./src/local_data/assets/discretizer.pkl")

for col in discretizer["vectors"].columns:
    fig = px.line(
        discretizer["vectors"][col].values,
        title=f"mean:{discretizer['mean'][col]} std:{discretizer['std'][col]}",
    )
    fig.write_image(f"./src/{col}.jpg")

    fig = px.histogram(
        discretizer["vectors"][col].values,
        title=f"mean:{discretizer['mean'][col]} std:{discretizer['std'][col]}",
    )
    fig.write_image(f"./src/{col}_hist.jpg")
