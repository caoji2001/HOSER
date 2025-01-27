from types import SimpleNamespace
import math
import random
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def create_nested_namespace(data):
    if isinstance(data, dict):
        return SimpleNamespace(**{k: create_nested_namespace(v) for k, v in data.items()})
    return data

def get_angle(lat1, lon1, lat2, lon2):
    dy = lat2 - lat1
    dx = math.cos(math.pi / 180 * lat1) * (lon2 - lon1)
    angle = math.atan2(dy, dx)
    if angle < 0:
        angle += math.pi * 2

    return angle
