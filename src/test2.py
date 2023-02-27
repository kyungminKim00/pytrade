import os
import random
from multiprocessing import Manager
from time import time
from typing import Any, Dict, List, Optional, Tuple, Union

import bottleneck as bn
import numpy as np
import pandas as pd
import plotly.express as px
import ray
import torch
from joblib import dump, load
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_reader import DataReader
from preprocess import SequentialDataSet
from quantile_discretizer import QuantileDiscretizer
from util import batch_idx, my_json, print_c, print_flush

"""Parameter Configuration
"""
if torch.cuda.is_available():
    _device = "cuda:0"
else:
    _device = "cpu"

candle_size = (1, 3, 5, 15, 60)
w_size = (9, 50, 100)
alpha = 3.5

sup_inf = {
    "feature_cont_close_candle3_ma9": {"sup": 0.003, "inf": -0.003},
    "feature_cont_close_candle3_ma50": {"sup": 0.004, "inf": -0.004},
    "feature_cont_close_candle3_ma100": {"sup": 0.005, "inf": -0.005},
    "feature_cont_close_candle5_ma9": {"sup": 0.003, "inf": -0.003},
    "feature_cont_close_candle5_ma50": {"sup": 0.005, "inf": -0.005},
    "feature_cont_close_candle5_ma100": {"sup": 0.005, "inf": -0.005},
    "feature_cont_close_candle15_ma9": {"sup": 0.0024, "inf": -0.004},
    "feature_cont_close_candle15_ma50": {"sup": 0.005, "inf": -0.005},
    "feature_cont_close_candle15_ma100": {"sup": 0.008, "inf": -0.008},
    "feature_cont_close_candle60_ma9": {"sup": 0.005, "inf": -0.005},
    "feature_cont_close_candle60_ma50": {"sup": 0.001, "inf": -0.0015},
    "feature_cont_close_candle60_ma100": {"sup": 0.0015, "inf": -0.0015},
    "feature_binary_close_10_lower_maginot": {"sup": 0.03, "inf": -0.025},
    "feature_binary_close_10_upper_maginot": {"sup": 0.03, "inf": -0.03},
    "feature_binary_close_20_lower_maginot": {"sup": 0.03, "inf": -0.03},
    "feature_binary_close_20_upper_maginot": {"sup": 0.25, "inf": -0.04},
    "feature_binary_close_50_lower_maginot": {"sup": 0.06, "inf": -0.25},
    "feature_binary_close_50_upper_maginot": {"sup": 0.01, "inf": -0.04},
    "feature_binary_close_3mins_high": {"sup": np.inf, "inf": -0.0017},
    "feature_binary_close_3mins_low": {"sup": 0.002, "inf": -np.inf},
    "feature_binary_close_3mins_open": {"sup": 0.0015, "inf": -0.0015},
    "feature_binary_close_5mins_high": {"sup": np.inf, "inf": -0.0017},
    "feature_binary_close_5mins_low": {"sup": 0.0022, "inf": -np.inf},
    "feature_binary_close_5mins_open": {"sup": 0.002, "inf": -0.002},
    "feature_binary_close_15mins_high": {"sup": np.inf, "inf": -0.0025},
    "feature_binary_close_15mins_low": {"sup": 0.0022, "inf": -np.inf},
    "feature_binary_close_15mins_open": {"sup": 0.0022, "inf": -0.0022},
    "feature_binary_close_60mins_high": {"sup": np.inf, "inf": -0.0035},
    "feature_binary_close_60mins_low": {"sup": 0.005, "inf": -np.inf},
    "feature_binary_close_60mins_open": {"sup": 0.004, "inf": -0.005},
}

# # 전처리 완료 데이터
# offset = 35000  # small data or operating data
# offset = None  # practical data

# print_c("SequentialDataSet 생성")
# sequential_data = SequentialDataSet(
#     raw_filename_min="./src/local_data/raw/dax_tm3.csv",
#     pivot_filename_day="./src/local_data/intermediate/dax_intermediate_pivots.csv",
#     candle_size=candle_size,
#     w_size=w_size,
#     debug=False,
#     offset=offset,
# )
# dump(sequential_data, "./src/local_data/assets/sequential_data.pkl")

# 전처리 완료 데이터 로드
processed_data = load("./src/local_data/assets/sequential_data.pkl")

# 변수 설정
x_real = [c for c in processed_data.train_data.columns if "feature" in c]
y_real = ["y_rtn_close"]

# # todo: optimize
# # supremum, infimum 탐색
# for col, sup_inf_dct in sup_inf.items():
#     data = processed_data.train_data[col]
#     max_val = data.max()
#     min_val = data.min()
#     mean_val = data.mean()

#     sub_range = np.arange(mean_val, max_val, (max_val - mean_val) / 100).tolist()
#     inf_range = np.arange(min_val, mean_val, (mean_val - min_val) / 100).tolist()

#     if sup_inf_dct["sup"] != np.inf:  # supremum 탐색
#         res = np.array(
#             [
#                 (some_val, np.where(data < some_val, 1, 0).sum())
#                 for some_val in sub_range
#             ]
#         )
#         move_std = bn.move_std(res[:, 1], window=50, axis=0)
#         std_diff = np.diff(move_std)
#         some_value = None

#         sup_inf[col]["sup"] = some_value
#     elif sup_inf_dct["inf"] != -np.inf:  # infimum 탐색
#         res = np.array(
#             [
#                 (some_val, np.where(data > some_val, 1, 0).sum())
#                 for some_val in inf_range
#             ]
#         )
#         move_std = bn.move_std(res[:, 1], window=50, axis=0)
#         std_diff = np.diff(move_std)
#         some_value = None

#         sup_inf[col]["inf"] = some_value


# # naive estimator - method 1
# print("naive estimator - method 1")
# action_table = pd.DataFrame()
# for i, col in enumerate(x_real):
#     action_table[f"F{2*i}"] = processed_data.train_data[col] > sup_inf[col]["sup"]
#     action_table[f"F{(2*i)+1}"] = processed_data.train_data[col] < sup_inf[col]["inf"]
# action_table.replace(True, 1, inplace=True)
# action_table.replace(False, 0, inplace=True)
# action_table["y_rtn_close"] = processed_data.train_data["y_rtn_close"]
# action_table.to_csv("./src/local_data/assets/action_table.csv")


# naive estimator - method 2
print_c("naive estimator - method 2")
action_table = pd.DataFrame()
action_table.index = processed_data.train_data.index
for _, col in enumerate(x_real):
    aa = np.where(processed_data.train_data[col] > sup_inf[col]["sup"], 1, 0)
    bb = np.where(processed_data.train_data[col] < sup_inf[col]["inf"], -1, 0)
    action_table[col] = aa + bb
action_table["y_rtn_close"] = processed_data.train_data["y_rtn_close"]
# 후 처리
action_table[1:] = action_table[1:].replace(0, np.nan)
action_table.fillna(method="ffill", inplace=True)
action_table.to_csv("./src/local_data/assets/action_table.csv")


# simulate vote - with ray 시간 오래 걸림
# # (조합이 많아서 시간이 오래 걸리는 거면 데이터 사이즈 60분봉으로 줄여도 별 차이 없을 것 같음)
# @ray.remote(num_cpus=4)
# def simulation_exhaussted(batch_i, num_estimators, obj_ref, y_rtn_close_ref, path):
#     if not os.path.isfile(path):
#         my_json.dump({}, path)
#     data = my_json.load(path)

#     for idx in batch_i:
#         binaryNum = format(idx, "b")
#         code = [int(digit) for digit in binaryNum]
#         mask = [0] * (num_estimators - len(code)) + code

#         t_data = obj_ref * mask
#         # t_data[1:] = t_data[1:].replace(0, np.nan)
#         # t_data = np.array(t_data.fillna(method="ffill"))

#         decision = t_data.sum(axis=1)
#         decision = np.where(decision > 0, 1, 0)
#         rtn_buy = (decision * y_rtn_close_ref).sum()

#         decision = t_data.sum(axis=1)
#         decision = np.where(decision < 0, -1, 0)
#         rtn_sell = (decision * y_rtn_close_ref).sum()

#         rtn = rtn_buy + rtn_sell

#         # print(f"idx: {idx}")
#         if rtn > 0.33:
#             # 메모리 overflow -> 결과 산출 할 때마다 파일로 저장 해 두기
#             print(f"idx: {idx}, rtn: {rtn} mask: {mask}")
#             data[idx] = {"rtn": rtn, "mask": mask}

#     my_json.dump(data, path)

#     return f"save result to {path}"


@ray.remote(num_cpus=4)
def simulation_exhaussted(batch_i, num_estimators, obj_ref, y_rtn_close_ref, path):
    def int_code(str_code):
        return [int(digit) for digit in str_code]

    data = {}
    # if not os.path.isfile(path):
    #     data = {}
    #     my_json.dump(data, path)
    # else:
    #     data = my_json.load(path)

    if os.path.isfile(path):
        data = my_json.load(path)

    binaryNum = [format(idx, "b") for idx in batch_i]
    mask = [[0] * (num_estimators - len(code)) + int_code(code) for code in binaryNum]

    # load to tensor or cuda tensor
    t_data = obj_ref * np.expand_dims(mask, axis=1)

    decision = t_data.sum(axis=2)
    decision = np.where(decision > 0, 1, 0)
    rtn_buy = (decision * y_rtn_close_ref).sum(-1)

    decision = t_data.sum(axis=2)
    decision = np.where(decision < 0, -1, 0)
    rtn_sell = (decision * y_rtn_close_ref).sum(-1)

    rtn = rtn_buy + rtn_sell

    conditions = rtn > 0.29

    if conditions.any():
        idxs = np.array(batch_i)[conditions]
        rtns = np.array(rtn)[conditions]
        masks = np.array(mask)[conditions]

        retirive_masks = {
            str(i): {"rtn": r, "mask": m.tolist()}
            for i, r, m in list(zip(idxs, rtns, masks))
        }

        # print(retirive_masks)
        data.update(retirive_masks)
    else:
        # print_c("조건을 만족하는 아이템 없음")
        pass

    my_json.dump(data, path)
    return f"save result to {path}"


print_c("simulation_exhaussted - mp")
obj_ref = ray.put(np.expand_dims(np.array(action_table.iloc[:, :-1]), axis=0))
y_rtn_close_ref = ray.put(np.expand_dims(action_table["y_rtn_close"], axis=0))

# obj_ref = np.expand_dims(np.array(action_table.iloc[:, :-1]), axis=0)
# y_rtn_close_ref = np.expand_dims(action_table["y_rtn_close"], axis=0)

path = "./src/local_data/assets/mask_result"
num_estimators = action_table.shape[1] - 1
start_idx, end_idx = 40030, 2**num_estimators
end_idx = start_idx + 5000
batch = 250

s_time = time()
res = ray.get(
    [
        simulation_exhaussted.remote(
            idx_estimator,
            num_estimators,
            obj_ref,
            y_rtn_close_ref,
            f"{path}_{idx_estimator[0]}.json",
        )
        # for idx_estimator in range(start_idx, 2**num_estimators)
        for idx_estimator in batch_idx(start_idx, end_idx, batch)
    ]
)
e_time = time()
print_c(f"simulation_exhaussted - mp - {e_time -s_time}s")
assert False, "simulation_exhaussted"
# res = [
#     simulation_exhaussted(
#         idx_estimator,
#         num_estimators,
#         obj_ref,
#         y_rtn_close_ref,
#         f"{path}_{idx_estimator[0]}.json",
#     )
#     # for idx_estimator in range(start_idx, 2**num_estimators)
#     for idx_estimator in batch_idx(start_idx, end_idx, batch)
# ]


print_c("collect score")
# load result history
res = None
dir_name = "./src/local_data/assets/"
for fn in os.listdir(dir_name):
    if ".json" in fn:
        tmp = my_json.load(f"{dir_name}{fn}")
        if res is not None:
            res.update(tmp)
        else:
            res = tmp

selected_mask, best_mask = [], []
best_score = 0
for k, v in res:
    if k > 0.31:
        selected_mask.append(v)
        if k > best_score:
            best_score = k
            best_mask = v
print(f"best_score: {best_score} best_mask: {best_mask}")

assert False, "action_table"

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
    fig.write_image(f"./src/{col}.jpg", scale=0.5)

    fig = px.histogram(
        discretizer["vectors"][col].values,
        title=f"mean:{discretizer['mean'][col]} std:{discretizer['std'][col]}",
    )
    fig.write_image(f"./src/{col}_hist.jpg", scale=0.5)

assert False, "ddd"

# DataReader configure
manager = Manager()
shared_dict = manager.dict()
shared_dict.update(load("./src/local_data/assets/pattern_dict.pkl"))
train_dataset = DataReader(
    df=processed_data.train_data,
    sequence_length=None,
    custom_index=processed_data.train_idx,
    discretizer=discretizer,
    known_real=x_real,
    unknown_real=y_real,
    pattern_dict=shared_dict,
)

batch_size = 100
dataset = train_dataset
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

progress_bar = tqdm(dataloader)
total_sample_about = processed_data.train_data.shape[0]
r_num_new_pattern = 0
for i, (x, x_date, num_new_pattern) in enumerate(progress_bar):
    r_num_new_pattern = num_new_pattern[-1].cpu()
    progress_bar.set_postfix(
        pattern=r_num_new_pattern,
        explain_pattern=(1 - (r_num_new_pattern / total_sample_about)),
    )

# 패턴 커버리지 분석
print_c(f"bins: {discretizer.n_bins} ")
print_c(f"mean: {discretizer.mean} ")
print_c(f"std: {discretizer.std} ")
print_c(f"max: {discretizer.max} ")
print_c(f"min: {discretizer.min} ")
print_c(f"candle_size: {candle_size}")
print_c(f"w_size: {w_size}")
print_c(f"alpha: {alpha}")
print_c(
    f"새롭게 찾은 패턴의수({r_num_new_pattern:,}) \
        샘플수 determineable_samples ({len(train_dataset):,}) \
        전체샘플수({total_sample_about:,})"
)
print_c(f"Score:{(1 - (r_num_new_pattern / total_sample_about))}")

assert False, "Done"


# 새롭게 추가된 패턴 저장
dump(dataset.pattern_dict, "./src/local_data/assets/pattern_dict.pkl")
print(f"dataset.pattern_dict: {dataset.pattern_dict}")


"""[Q-Learning]
"""

# Define the Q-Table
num_states = 10
num_actions = 2
q_table = np.zeros((num_states, num_actions))

# Define the learning rate and discount factor
learning_rate = 0.1
discount_factor = 0.9

# Define the exploration rate and decay rate
exploration_rate = 1.0
min_exploration_rate = 0.01
decay_rate = 0.001

# Define the rewards and transition function
rewards = np.array(
    [
        [0, 1],
        [5, 10],
        [0, 1],
        [5, 10],
        [0, 1],
        [5, 10],
        [0, 1],
        [5, 10],
        [0, 1],
        [10, 20],
    ]
)
transition = np.array(
    [[1, 2], [3, 4], [5, 6], [7, 8], [9, 0], [1, 2], [3, 4], [5, 6], [7, 8], [9, 0]]
)

# Define the Q-Learning algorithm
for episode in range(1000):
    state = np.random.randint(0, num_states)
    done = False
    while not done:
        # Select an action using the exploration rate
        if np.random.uniform(0, 1) < exploration_rate:
            action = np.random.randint(0, num_actions)
        else:
            action = np.argmax(q_table[state, :])

        # Get the next state and reward
        next_state = transition[state, action]
        reward = rewards[state, action]

        # Update the Q-Table
        q_table[state, action] = (1 - learning_rate) * q_table[
            state, action
        ] + learning_rate * (reward + discount_factor * np.max(q_table[next_state, :]))

        # Update the state and exploration rate
        state = next_state
        exploration_rate = min_exploration_rate + (1 - min_exploration_rate) * np.exp(
            -decay_rate * episode
        )

        # Check if the episode is done
        if state == 0:
            done = True

# Test the Q-Learning algorithm
state = 0
done = False
while not done:
    action = np.argmax(q_table[state, :])
    next_state = transition[state, action]
    reward = rewards[state, action]
    print("State: ", state, " Action: ", action, " Reward: ", reward)
    state = next_state
    if state == 0:
        done = True
