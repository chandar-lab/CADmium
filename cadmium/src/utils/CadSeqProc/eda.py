import os, sys

sys.path.append(os.path.dirname(__file__))
sys.path.append("..")

import json
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from CadSeqProc.utility.utils import get_files_scan
from CadSeqProc.cad_sequence import CADSequence
from CadSeqProc.utility.logger import CLGLogger
from CadSeqProc.utility.decorator import measure_performance
from tqdm import tqdm
import argparse

clg_logger = CLGLogger().configure_logger().logger


# ---------------------------------------------------------------------------- #
#                 Exploratory Data Analysis for DeepCAD Dataset                #
# ---------------------------------------------------------------------------- #

# NOTE: This code is used to perform exploratory data analysis on the DeepCAD dataset. 
# It returds a dataframe with number of sketches, faces, loops, curves and extrusions 
# for each CAD object in the dataset.

def process_one(json_path):

    with open(json_path, "r") as f:
        data = json.load(f)

    uid = "/".join(json_path.split("/")[-2:]).strip(".json")

    try:
        cad_obj = CADSequence.from_dict(data)
    except Exception as e:
        clg_logger.info(f"Json: {uid}. Error: {e}")
        return None
    new_df = pd.DataFrame(
        {
            "uid": uid,
            "sketch": [len(cad_obj.sketch_seq)],
            "face": [len(cad_obj.all_faces)],
            "loop": [len(cad_obj.all_loops)],
            "curve": [len(cad_obj.all_curves)],
            "extrusion": [len(cad_obj.extrude_seq)],
        }
    )

    return new_df

@measure_performance
def process_json(args):
    all_json_files = get_files_scan(args.json_dir, max_workers=args.max_workers)

    df = pd.DataFrame()

    # ----------------------- Process the json Sequentially ---------------------- #
    # for json_path in tqdm(all_json_files):
    #     new_df = process_one(json_path)
    #     if new_df is not None:
    #         df = pd.concat([df, new_df], ignore_index=True)

    # ---------------- Process the Json using Parallel processing ---------------- #
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(process_one, json_path)
            for json_path in tqdm(all_json_files, desc="Submitting Tasks")
        ]
        for future in tqdm(as_completed(futures), desc="Processing Files"):
            val = future.result()  # complexity is number of curves
            if val is not None:
                df = pd.concat([df, val], ignore_index=True)

    # ----------------------- Save the results ---------------------- #
    df.to_csv(os.path.join(args.output_dir, "analysis.csv"), index=False)

    return df


def main():
    parser = argparse.ArgumentParser(description="Exploratory Data Analysis")
    parser.add_argument("--json_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_workers", type=int, default=8)
    args = parser.parse_args()

    process_json(args)


if __name__ == "__main__":
    main()
