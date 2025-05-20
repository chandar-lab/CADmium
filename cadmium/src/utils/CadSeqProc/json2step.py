import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append("..")

# Adding Python Path
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor

from tqdm import tqdm
from CadSeqProc.utility.decorator import measure_performance
from CadSeqProc.utility.logger import CLGLogger
from CadSeqProc.utility.macro import *
from CadSeqProc.utility.utils import get_files_scan
import argparse
import multiprocessing
import json
from CadSeqProc.cad_sequence import CADSequence
import warnings
import gc

warnings.filterwarnings("ignore")

multiprocessing.set_start_method("forkserver", force=True)
clglogger = CLGLogger().configure_logger().logger

# ---------------------------------------------------------------------------- #
#                           DeepCAD Json to Brep/Mesh                          #
# ---------------------------------------------------------------------------- #


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
    parser.add_argument(
        "--split_json", help="Input Directory for DeepCAD split json", type=str
    )
    parser.add_argument("-o", "--output_dir", type=str)
    parser.add_argument(
        "--dataset",
        type=str,
        default="deepcad",
        choices=["deepcad", "fusion360", "cad_parser"],
    )
    parser.add_argument("--bit", type=int, default=N_BIT)
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--save_type", type=str, default="step")
    args = parser.parse_args()

    # print(args)
    clglogger.info(f"Running Task with {args.max_workers} workers.")

    if args.dataset == "deepcad":
        process_deepcad(args, clglogger)
    elif args.dataset == "fusion360":
        process_fusion360(args, clglogger)

    # all_json_files = sorted(get_files_scan(args.input_dir, max_workers=args.max_workers))


def process_fusion360(args, clglogger):
    """
    Processes the Fusion360 dataset

    Args:
        args (argparse.Namespace): The arguments.
    """
    all_files = get_files_scan(args.input_dir, max_workers=args.max_workers)
    all_json_files = [
        file
        for file in all_files
        if file.endswith(".json") and file.split("/")[-2] == "json"
    ]
    clglogger.info(
        f"Preprocessing {len(all_json_files)} Fusion360 dataset using Method 1."
    )
    process_all_jsons(all_json_files, args, clglogger)
    clglogger.success(f"Task Complete")


def process_deepcad(args, clglogger):
    """
    Processes the DeepCAD dataset

    Args:
        args (argparse.Namespace): The arguments.
    """
    with open(args.split_json, "r") as f:
        data = json.load(f)

    all_json_files = (
        data["train"][:82000]
        + data["train"][84000:]
        + data["test"]
        + data["validation"]
    )

    # --------------------------------- Method 1 --------------------------------- #
    process_all_jsons(all_json_files, args, clglogger)

    extra_json_files = [
        os.path.join(args.input_dir, uid + ".json")
        for uid in data["train"][82000:84000]
    ]

    # --------------------------------- Method 2 --------------------------------- #
    clglogger.info(f"Preprocessing {len(extra_json_files)} using Method 2")
    for json_path in tqdm(all_json_files):
        try:
            process_json(json_path, args)
        except:
            pass

    clglogger.success(f"Task Complete")


def process_all_jsons(all_json_files, args, clglogger):
    """
    Processes all the JSON files in the list and saves the CAD models

    Args:
        all_json_files (list): A list of JSON files.
    """
    # Create a ProcessPoolExecutor
    executor = ThreadPoolExecutor(max_workers=args.max_workers)

    # Submit tasks to the executor
    futures = [
        executor.submit(process_json, json_path, args)
        for json_path in tqdm(all_json_files, desc="Submitting Tasks")
    ]

    # Wait for the tasks to complete
    for future in tqdm(as_completed(futures), desc="Processing Files"):
        future.result()
    
    clglogger.success(f"Method 1 Complete")


def process_json(json_path, args):
    """
    Processes a JSON file and saves the whole CAD model as well as intermediate ones

    Args:
        json_path (str): The path to the JSON file.
    """
    try:
        if args.dataset == "deepcad":
            uid = "/".join(json_path.strip(".json").split("/")[-2:])  # 0003/00003121

        elif args.dataset == "fusion360":
            uid = "/".join(json_path.split("/")[-4:-2])
        name = uid.split("/")[-1]  # 00003121

        # Open the JSON file.
        with open(json_path, "r") as f:
            data = json.load(f)

        # cad_seq = CADSequence.json_to_NormalizedCAD(data=data, bit=args.bit)
        cad_seq = CADSequence.from_dict(all_stat=data)

        # ------------------------- Save the final cad Model ------------------------- #
        cad_seq.save_stp(
            filename=name + "_final",
            output_folder=os.path.join(args.output_dir, uid, args.save_type),
            type=args.save_type,
        )

        # ------------------------ Save the intermediate models ----------------------- #
        num_intermediate_model = len(cad_seq.sketch_seq)
        if num_intermediate_model > 1:
            # -------------------------------- Separate -------------------------------- #

            for i in range(num_intermediate_model):
                new_cad_seq = CADSequence(
                    sketch_seq=[cad_seq.sketch_seq[i]],
                    extrude_seq=[cad_seq.extrude_seq[i]],
                )

                # Make the operation as NewBodyOperation to create a solid body
                new_cad_seq.extrude_seq[0].metadata["boolean"] = 0
                new_cad_seq.save_stp(
                    filename=name + f"_intermediate_{i+1}",
                    output_folder=os.path.join(args.output_dir, uid, args.save_type),
                    type=args.save_type,
                )
                del new_cad_seq
        
        gc.collect()
       
    except Exception as e:
        pass
        clglogger.error(f"Problem processing {json_path}. Error: {e}")


if __name__ == "__main__":
    main()
