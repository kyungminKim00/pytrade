import pandas as pd
from gen_pivots import Aggregator

if __name__ == "__main__":
    aggregator = Aggregator(
        data=pd.read_csv(
            "./src/local_data/raw/dax_td1.csv", sep=",", dtype={"date": object, "time": object}
        )
    )
    aggregator.data.to_csv(
        "./src/local_data/intermediate/dax_intermediate_pivots.csv", index=None
    )
