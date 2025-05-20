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
import pandas as pd
import time
import traceback
from rich import print
from tqdm import tqdm
from CadSeqProc.utility.decorator import measure_performance
from CadSeqProc.utility.logger import CLGLogger
import torch
from loguru import logger
import numpy as np
from CadSeqProc.utility.macro import *
from CadSeqProc.utility.utils import (
    generate_attention_mask,
    ensure_dir,
    hash_map,
    get_files_scan,
)
from cad_sequence import CADSequence
import argparse
import multiprocessing
import json
from CadSeqProc.cad_sequence import CADSequence
import warnings
import shutil

warnings.filterwarnings("ignore")

multiprocessing.set_start_method("forkserver", force=True)

clglogger = CLGLogger().configure_logger().logger

unique_model_hash = []


# ---------------------------------------------------------------------------- #
#                       DeepCAD Json to CAD-SIGNet Vector                      #
# ---------------------------------------------------------------------------- #

# This code is used to convert the DeepCAD Json dataset to CAD-SIGNet vector representation.
# Required for Training the Text2CAD Transformer

@measure_performance
def main():
    """
    Parse Json into sketch and extrusion sequence tokens
    """
    parser = argparse.ArgumentParser(
        description="Creating Sketch and Extrusion Sequence"
    )
    parser.add_argument(
        "-p", "--input_dir", help="Input Directory for DeepCAD Json dataset", type=str
    )
    parser.add_argument("--split_json", help="Train-test-validation split", type=str, default="")
    parser.add_argument("-o", "--output_dir", type=str)
    parser.add_argument(
        "--dataset",
        type=str,
        default="deepcad",
        choices=["deepcad", "fusion360", "cad_parser"],
    )
    parser.add_argument("--bit", type=int, default=N_BIT)
    parser.add_argument(
        "--max_cad_seq_len",
        type=int,
        default=MAX_CAD_SEQUENCE_LENGTH,
        help="Maximum length of cad sequence",
    )
    parser.add_argument("--max_workers", type=int, default=32)
    parser.add_argument(
        "--padding",
        action="store_true",
        help="Add padding in the vector for same token length",
    )
    parser.add_argument("--deduplicate", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    # print(args)
    if args.verbose:
        clglogger.info(f"Running Task with {args.max_workers} workers.")

    if args.dataset=="deepcad":
        process_deepcad(args, clglogger)
    
    elif args.dataset=="fusion360":
        process_fusion360(args, clglogger)
    
    elif args.dataset=="cad_parser":
        #process_cad_parser(args, clglogger)
        pass
    
    else:
        raise ValueError(f"Invalid dataset name {args.dataset}.")



def process_deepcad(args, clglogger):
    # Get Json Path
    with open(args.split_json, "r") as f:
        data = json.load(f)

    uidlist = (
        data["train"][:82000]
        + data["train"][84000:]
        + data["test"]
        + data["validation"]
    )  # Possible corruption in data causing ProcessPool to fail

    all_json_files = [os.path.join(args.input_dir, uid + ".json") for uid in uidlist]

    # ------------------------------------------- Method 1 ------------------------------------------- #

    process_all_jsons(all_json_files, args, clglogger)

    # ------------------------------------------- Method 2 ------------------------------------------- #

    # NOTE: METHOD 2 - Sometimes the processpoolexecutor or threadpoolexecutor will fail.
    # As a final choice, run the following then
    extra_json_files = [
        os.path.join(args.input_dir, uid + ".json")
        for uid in data["train"][82000:84000]
    ]

    if args.verbose:
        clglogger.info(f"Preprocessing {len(extra_json_files)} using Method 2")
    for json_path in tqdm(extra_json_files):
        try:
            process_json(json_path, args)
        except:
            pass

    clglogger.success(f"Task Complete")



def process_fusion360(args, clglogger):
    all_files = get_files_scan(args.input_dir,max_workers=args.max_workers)
    all_json_files = [file for file in all_files if file.endswith(".json")]
    process_all_jsons(all_json_files, args, clglogger)



def process_all_jsons(all_json_files, args, clglogger):
    DUPLICATE_MODEL = 0
    executor = ProcessPoolExecutor(max_workers=args.max_workers)
    duplicate_uid = []

    if args.verbose:
        clglogger.info(f"Found {len(all_json_files)} files.")
        clglogger.info(f"Saving Sequence in {args.output_dir}")
        if args.deduplicate:
            clglogger.warning(f"Deduplicate is on. Some duplicate models won't be saved.")

    # NOTE: METHOD 1 - Run the following for faster data processing
    # If it fails, resume from the next iteration

    # Submit tasks to the executor
    futures = [
        executor.submit(process_json, json_path, args)
        for json_path in tqdm(all_json_files, desc="Submitting Tasks")
    ]
    unique_uid = dict()
    for future in tqdm(as_completed(futures), desc="Processing Files", total=len(futures)):
        val, uid, complexity = future.result()  # complexity is number of curves
        DUPLICATE_MODEL += val
        if val == 1:
            duplicate_uid.append(uid)
        if val == 0:
            unique_uid[uid] = complexity

    # sorted_uid = sorted(unique_uid.keys(), key=lambda k: unique_uid[k]) # Sorted according to the number of curves

    complexity_df = pd.DataFrame(
        {"uid": unique_uid.keys(), "complexity": unique_uid.values()}
    )
    complexity_df.to_csv(f"complexity.csv", index=False)

    with open(f"duplicate_uid.txt", "w") as f:
        for item in duplicate_uid:
            f.write(str(item) + "\n")

    # with open(f"sorted_uid_train_test_val.json", "w") as f:
    #     json.dump({args.subset: sorted_uid}, f)

    if args.verbose:
        clglogger.info(f"Total Number of Models {len(all_json_files)}")
        clglogger.info(
            f"Total Number of Invalid Models {DUPLICATE_MODEL} and percentage {DUPLICATE_MODEL/len(all_json_files)}"
        )
        clglogger.info(f"Total Number of Unique Models {len(unique_uid)}")


@logger.catch()
def process_json(json_path, args):
    """
    Processes a JSON file and converts it to a vector representations of sketch and extrusion.

    Args:
        json_path (str): The path to the JSON file.
        bit (int): The bit depth of the vector.
        output_dir (str): The output directory.

    Returns:
        int: The number of sketches in the JSON file.
        int: The number of extrusions in the JSON file.

    """
    try:
        
        if args.dataset=="deepcad":
            uid = "/".join(json_path.strip(".json").split("/")[-2:])  # 0003/00003121
            
        elif args.dataset=="fusion360":
            uid = "/".join(json_path.split("/")[-4:-2])
        name = uid.split("/")[-1]  # 00003121

        # Open the JSON file.
        with open(json_path, "r") as f:
            data = json.load(f)

        # Reading From JSON -> Normalize -> Numericalize -> To Vector Representation
        cad_obj, cad_vec, flag_vec, index_vec = CADSequence.json_to_vec(
            data=data,
            bit=args.bit,
            padding=args.padding,
            max_cad_seq_len=MAX_CAD_SEQUENCE_LENGTH,
        )

        # Check for duplication
        to_save = True
        # Perform hashing for unique models
        if args.deduplicate:
            global unique_model_hash

            param = cad_vec[torch.where(cad_vec >= len(END_TOKEN))[0]].tolist()
            hash_vec = hash_map(param)
            if hash_vec in unique_model_hash:
                to_save = False

        cad_seq_dict = {
            "vec": {
                "cad_vec": cad_vec,
                "flag_vec": flag_vec,
                "index_vec": index_vec,
            },
            "mask_cad_dict": {
                "attn_mask": generate_attention_mask(cad_vec.shape[0] - 1),
                "key_padding_mask": cad_vec == END_TOKEN.index("PADDING"),
            },
        }

        # If save
        if to_save:
            # Save the data in .pth format
            output_dir = os.path.join(args.output_dir, uid, "seq")
            # print(output_dir)
            ensure_dir(output_dir)
            torch.save(cad_seq_dict, os.path.join(output_dir, name + ".pth"))
            if args.verbose:
                clglogger.success(f"Saved in {os.path.join(output_dir, name + '.pth')}")

            return 0, uid, len(cad_obj.all_curves)
        else:
            if args.verbose:
                clglogger.warning(f"Skipping {json_path} because of duplication.")
            return 1, uid, 0
    except Exception as e:
        # print(traceback.print_exc())
        if args.verbose:
            clglogger.error(f"Problem with json path {json_path} with error {e}")
        return 1, uid, 0


if __name__ == "__main__":
    main()
