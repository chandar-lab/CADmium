import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append("..")

# Adding Python Path
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
    wait,
)

from rich import print
from tqdm import tqdm
from CadSeqProc.utility.decorator import measure_performance
from CadSeqProc.utility.logger import CLGLogger
import torch
from loguru import logger
import numpy as np
from CadSeqProc.utility.macro import *
from CadSeqProc.utility.utils import get_files_scan, ensure_dir
import argparse
import multiprocessing
import json

import warnings

warnings.filterwarnings("ignore")

multiprocessing.set_start_method("forkserver", force=True)

clglogger = CLGLogger().configure_logger().logger

unique_model_hash = []


# ---------------------------------------------------------------------------- #
#           Create a nested Dictionary with all the annotation paths           #
# ---------------------------------------------------------------------------- #


@measure_performance
def main():
    """
    Create a nested Dictionary with all the annotation paths
    """
    parser = argparse.ArgumentParser(
        description="Generate Split JSON for Training, Test, and Validation"
    )
    parser.add_argument(
        "-p", "--mapper_path", help="Input Directory for DeepCAD JSON dataset", type=str
    )
    parser.add_argument("-o", "--output_dir", type=str)

    args = parser.parse_args()

    seq_prompt_pairs = []
    with open(args.mapper_path, "r") as f:
        mapper_data = json.load(f)

    # TODO: Work on this part
    for key, val in mapper_data.items():
        for sub_key, sub_val in val.items():

            uid = f"{key}/{sub_key}"

            if "vlm_annotation" in sub_val:
                num_prompts = len(val["llm_annotation"])
                if num_prompts > 0:
                    all_keys = [sub_val['']] * num_prompts
                    result = list(zip(all_keys, val["llm_annotation"]))
                    seq_prompt_pairs.extend(result)
            else:
                continue


if __name__ == "__main__":
    main()
