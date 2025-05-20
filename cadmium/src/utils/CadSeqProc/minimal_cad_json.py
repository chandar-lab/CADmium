import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append("..")

# Adding Python Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from CadSeqProc.utility.decorator import measure_performance
from CadSeqProc.utility.logger import CLGLogger
from CadSeqProc.utility.macro import *
from CadSeqProc.utility.utils import ensure_dir, get_files_scan
from cad_sequence import CADSequence
import argparse
import traceback
# import multiprocessing
import json
from CadSeqProc.cad_sequence import CADSequence
import warnings

warnings.filterwarnings("ignore")
# multiprocessing.set_start_method("forkserver", force=True)
clglogger = CLGLogger().configure_logger().logger


# ---------------------------------------------------------------------------- #
#                         DeepCAD Json to Minimal Json                         #
# ---------------------------------------------------------------------------- #


@measure_performance
def main():
    """
    Parse DeepCAD Json into more human-readable json format
    """
    parser = argparse.ArgumentParser(
        description="Creating Sketch and Extrusion Sequence"
    )
    parser.add_argument(
        "-p", "--input_dir", help="Input Directory for DeepCAD Json dataset", type=str
    )
    parser.add_argument("--split_json", help="Train-test-validation split", type=str, default="")
    parser.add_argument("-o", "--output_dir", type=str)
    parser.add_argument("--bit", type=int, default=N_BIT)
    parser.add_argument(
        "--dataset",
        type=str,
        default="deepcad",
        choices=["deepcad", "fusion360", "cad_parser"],
    )
    parser.add_argument("--max_workers", type=int, default=32)

    args = parser.parse_args()
    # print(args)
    clglogger.info(f"Running Task with {args.max_workers} workers.")

    if args.dataset == "deepcad":
        process_deepcad(args, clglogger)
    elif args.dataset == "fusion360":
        process_fusion360(args, clglogger)

    clglogger.success(f"Task Complete")


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


def process_deepcad(args, clglogger):
    """
    Processes the DeepCAD dataset

    Args:
        args (argparse.Namespace): The arguments.
    """
    with open(args.split_json, "r") as f:
        data = json.load(f)

    all_uids = (
        data["train"][:82000]
        + data["train"][84000:]
        + data["test"]
        + data["validation"]
    )

    # --------------------------------- Method 1 --------------------------------- #
    all_json_files=[os.path.join(args.input_dir, uid + ".json") for uid in all_uids]
    process_all_jsons(all_json_files, args, clglogger)

    extra_json_files = [
        os.path.join(args.input_dir, uid + ".json")
        for uid in data["train"][82000:84000]
    ]

    # --------------------------------- Method 2 --------------------------------- #
    clglogger.info(f"Preprocessing {len(extra_json_files)} using Method 2")
    for json_path in tqdm(extra_json_files):
        try:
            process_json(json_path, args)
        except:
            pass



def process_all_jsons(all_json_files, args, clglogger):
    clglogger.info(f"Found {len(all_json_files)} files.")
    clglogger.info(f"Saving Sequence in {args.output_dir}.")

    # --------------------------------- Method 1 --------------------------------- #

    # NOTE: METHOD 1 - Run the following for faster data processing

    executor = ThreadPoolExecutor(max_workers=args.max_workers)
    # If it fails, resume from the next iteration

    # Submit tasks to the executor
    futures = [
        executor.submit(process_json, json_path, args)
        for json_path in tqdm(all_json_files, desc="Submitting Tasks")
    ]

    for future in tqdm(as_completed(futures), desc="Processing Files", total=len(futures)):
        val, _ = future.result()  # complexity is number of curves


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

        if args.dataset == "deepcad":
            uid = "/".join(json_path.strip(".json").split("/")[-2:])  # 0003/00003121

        elif args.dataset == "fusion360":
            uid = "/".join(json_path.split("/")[-4:-2])
        name = uid.split("/")[-1]  # 00003121

        # Open the JSON file.
        with open(json_path, "r") as f:
            data = json.load(f)

        # Reading From JSON -> Normalize -> Numericalize -> To Vector Representation
        cad_seq = CADSequence.json_to_NormalizedCAD(data=data, bit=8)
        cad_metadata = cad_seq._json()

        # Save the data in .pth format
        output_dir = os.path.join(
            args.output_dir, uid, "minimal_json"
        )  # Output Directory
        ensure_dir(output_dir)
        output_name = os.path.join(output_dir, name + ".json")
        # clglogger.debug(f"Saving to {output_name}")

        if os.path.exists(output_name):
            os.remove(output_name)

        with open(output_name, "w") as f:
            json.dump(cad_metadata, f, indent=5)

        return 0, uid

    except Exception as e:
        # print(traceback.print_exc())
        clglogger.error(f"Problem with json path {json_path} with error {e}")
        return 1, uid


if __name__ == "__main__":
    main()
