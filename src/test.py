import numpy as np
import pandas as pd
import torch

from util import batch_idx, my_json, print_c

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

# # 전처리 완료 데이터 로드
# processed_data = load("./src/local_data/assets/sequential_data.pkl")

# # 변수 설정
# x_real = [c for c in processed_data.train_data.columns if "feature" in c]
# y_real = ["y_rtn_close"]

if torch.cuda.is_available():
    _device = "cuda:0"
else:
    _device = "cpu"


def simulation_exhaussted(batch_i, num_estimators, obj_ref, y_rtn_close_ref, path):
    def int_code(str_code):
        return [int(digit) for digit in str_code]

    data = {}
    # if not os.path.isfile(path):
    #     my_json.dump({}, path)
    # data = my_json.load(path)

    binaryNum = [format(idx, "b") for idx in batch_i]
    mask = [[0] * (num_estimators - len(code)) + int_code(code) for code in binaryNum]

    # load to tensor or cuda tensor
    y_rtn = torch.from_numpy(np.expand_dims(y_rtn_close_ref, axis=0)).to(device=_device)
    t_data = (
        torch.from_numpy(np.expand_dims(obj_ref, axis=0))
        * torch.tensor(np.expand_dims(mask, axis=1))
    ).to(device=_device)

    decision = t_data.sum(axis=2)
    decision = torch.where(decision > 0, 1, 0)
    rtn_buy = (decision * y_rtn).sum(-1)

    decision = t_data.sum(axis=2)
    decision = torch.where(decision < 0, -1, 0)
    rtn_sell = (decision * y_rtn).sum(-1)

    # rtn = (rtn_buy + rtn_sell).cpu().detach().numpy()
    rtn = (rtn_buy + rtn_sell).cpu()

    conditions = rtn > 0.33
    conditions = rtn > 0
    # conditions = conditions.cpu()
    if conditions.any():
        idxs = np.array(batch_i)[conditions]
        rtns = np.array(rtn)[conditions]
        masks = np.array(mask)[conditions]

        retirive_masks = {
            i: {"rtn": r, "mask": m.tolist()}
            for i, r, m in list(zip(idxs, rtns, masks))
        }

        print(retirive_masks)
        data.update(retirive_masks)
    else:
        print_c("조건을 만족하는 아이템 없음")


action_table = pd.read_csv("./src/local_data/assets/action_table.csv")
action_table.set_index("datetime", inplace=True)

print_c("simulation_exhaussted - mp")
obj_ref = np.array(action_table.iloc[:, :-1])
y_rtn_close_ref = action_table["y_rtn_close"]

path = "./src/local_data/assets/mask_result"
num_estimators = action_table.shape[1] - 1
start_idx, end_idx = 1, 2**num_estimators
batch = 10
res = [
    simulation_exhaussted(
        idx_estimator,
        num_estimators,
        obj_ref,
        y_rtn_close_ref,
        f"{path}_{idx_estimator[0]}.json",
    )
    # for idx_estimator in range(start_idx, 2**num_estimators)
    for idx_estimator in batch_idx(start_idx, end_idx, batch)
]
