import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.preprocessing import KBinsDiscretizer

# df = pd.DataFrame(
#     data=np.random.randint(0, 10, size=[15, 6]),
#     columns=["a", "b", "c", "d", "e", "f"],
#     index=np.arange(15),
# )
# df.index.name = "ttt"
# df.to_csv("./test.csv")

obj_fn = "./src/local_data/assets/discretizer.pkl"
df = pd.read_csv("./test.csv")
df.set_index("ttt", inplace=True)

# test 1
mean = df.mean(axis=0)
std = df.std(axis=0, ddof=1)

lower = (mean - 3 * std).astype(int)
upper = (mean + 3 * std).astype(int)
clipped_vectors = df.clip(lower, upper, axis=1)
n_bins = (3.5 * std * np.power(df.shape[0], -1 / 3)).astype(int)

discretizer = {
    "model": KBinsDiscretizer(
        n_bins=n_bins.values, encode="ordinal", strategy="uniform"
    ).fit(clipped_vectors.to_numpy()),
    "mean": mean.values,
    "std": std.values,
    "n_bins": n_bins.values,
    "lower": lower.values,
    "upper": upper.values,
    "valide_key": list(clipped_vectors.columns),
}
print(discretizer["model"].transform(clipped_vectors))
dump(discretizer, obj_fn)
# assert False, "discretizer TEST1 DONE"

# test 2
obj_fn = "./src/local_data/assets/discretizer.pkl"
discretizer = load(obj_fn)
lower = discretizer["lower"]
upper = discretizer["upper"]
clipped_vectors = df.clip(lower, upper, axis=1)
print(discretizer["model"].transform(clipped_vectors))
# assert False, "discretizer TEST2 DONE"
