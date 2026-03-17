import argparse
from datetime import datetime
import json
import os
import random
import torch

import numpy as np


def set_seed(seed):
    """
    Set fixed random seed
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def time_elapsed(start_time, end_time, message="Elapsed Time: "):
    """
    Return the timing result string
    """

    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{message}{int(hours):02}:{int(minutes):02}:{int(seconds):02}"


def cur_time_str():
    now = datetime.now()
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_time


def load_args_from_json(file_path):
    """
    Read json file and return args tuple
    """

    # Read JSON file
    with open(file_path, "r") as f:
        args_dict = json.load(f)

    # Load arguments into argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for key, value in args_dict.items():
        if isinstance(value, list):
            parser.add_argument(f"--{key}", type=str, nargs="+", default=value)
        else:
            parser.add_argument(f"--{key}", type=type(value), default=value)

    args = parser.parse_args([])
    return args


def rank_print(accelerator, text):
    """
    Multi-process printing
    """

    print(f"[{cur_time_str()}][rank {accelerator.local_process_index}]:", text)


def rank_main_print(accelerator, text):
    """
    Only the main process of each node prints
    """

    if accelerator.is_local_main_process:
        print(f"[{cur_time_str()}][rank 0 for all]:", text)