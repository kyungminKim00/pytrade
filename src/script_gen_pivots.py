import pandas as pd
from gen_pivots import Aggregator

if __name__ == "__main__":
    aggregator = Aggregator(
        data=pd.read_csv(
            "./dataset/raw/dax_td1.csv", sep=",", dtype={"date": object, "time": object}
        )
    )
    aggregator.data.to_csv(
        "./dataset/intermediate/dax_intermediate_pivots.csv", index=None
    )
