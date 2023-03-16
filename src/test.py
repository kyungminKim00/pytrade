import os
from datetime import datetime

import imageio
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import ray
import yfinance as yf
from plotly.subplots import make_subplots

from util import print_c

open_data = [33.0, 33.3, 33.5, 33.0, 34.1]
high_data = [33.1, 33.3, 33.6, 33.2, 34.8]
low_data = [32.7, 32.7, 32.8, 32.6, 32.8]
close_data = [33.0, 32.9, 33.3, 33.1, 33.1]
dates = [
    datetime(year=2013, month=10, day=10),
    datetime(year=2013, month=11, day=10),
    datetime(year=2013, month=12, day=10),
    datetime(year=2014, month=1, day=10),
    datetime(year=2014, month=2, day=10),
]

trace = go.Candlestick(
    x=dates, open=open_data, high=high_data, low=low_data, close=close_data
)
fig = go.Figure([trace])
fig.add_vrect(
    x0=datetime(year=2013, month=10, day=10),
    x1=datetime(year=2013, month=12, day=10),
    row="all",
    col=1,
    annotation_text="decline",
    annotation_position="top left",
    fillcolor="green",
    opacity=0.25,
    line_width=0,
)

cs = fig.data[0]

# Set line and fill colors
cs.increasing.fillcolor = "#3D9970"
cs.increasing.line.color = "#3D9970"
cs.decreasing.fillcolor = "#FF4136"
cs.decreasing.line.color = "#FF4136"

fig.show()
