from collections import namedtuple

import torch

# constants

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# data

Memory = namedtuple(
    "Memory", ["state", "action", "action_log_prob", "reward", "done", "value"]
)
AuxMemory = namedtuple("Memory", ["state", "target_value", "old_values"])
