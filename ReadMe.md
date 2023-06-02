# script_candle_stick.py
- 각 캔들의 패턴의 5일 후 상승 하락의 확률을 알려 주는 스크립트
input: ./src/candle_stick.json
output: ./src/local_data/assets/plot_check/chart/*.png

# script_context_model.py
- 시장 데이터 인코더

# script_context_retrive.py
- 시장 데이터 디코더

# script_report.py
- 시장 인코더-디코더 결과 분석 모듈

# script_distribute_action_simulation_best.py
- Feature 데이터 전역탐색하여 시뮬레이션 수익률을 비트하는 Feature 탐색

# script_explain_variables.py
- 분석 필요

# script_gen_pivots.py
- 일간 데이터 기분 마지노선 생성 모듈
input:
output: ./src/local_data/intermediate/dax_intermediate_pivots.csv

# script_learn.py
-
input: ./src/local_data/raw/dax_tm3.csv, ./src/local_data/intermediate/dax_intermediate_pivots.csv