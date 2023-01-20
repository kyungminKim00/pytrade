import pandas as pd
from gen_pivots import Aggregator

if __name__ == "__main__":
    aggregator = Aggregator(
        data=pd.read_csv(
            "./local_daat/raw/dax_td1.csv", sep=",", dtype={"date": object, "time": object}
        )
    )
    aggregator.data.to_csv(
        "./local_daat/intermediate/dax_intermediate_pivots.csv", index=None
    )
