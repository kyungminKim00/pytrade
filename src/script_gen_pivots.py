import pandas as pd
import ray

from gen_pivots import Aggregator

if __name__ == "__main__":
    """_summary_
    피봇 라인을 생성 하는 모듈 (마지노선 설정)
    마지노선은 일간 데이터로 부터 산출 함

    """
    aggregator = Aggregator(
        data=pd.read_csv(
            "./src/local_data/raw/dax_td1.csv",
            sep=",",
            dtype={"date": object, "time": object},
        )
    )
    aggregator.data.to_csv(
        "./src/local_data/intermediate/dax_intermediate_pivots.csv", index=None
    )
