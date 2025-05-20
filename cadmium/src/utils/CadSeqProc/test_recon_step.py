from cad_sequence import CADSequence
import os
import argparse
import glob
import pickle
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# ---------------------------------------------------------------------------- #
#                Generate Step file from Predicted CAD Sequence                #
# ---------------------------------------------------------------------------- #


def process_uid(uid, data, output_dir):
    output_folder = os.path.join(output_dir, uid, "step")
    correct_keys = 0
    wrong_keys = 0
    count_saved = 0
    sample_id = uid.split("/")[-1]  # 00000007
    for i in range(1, 5):
        # check if the key exists
        if "level_" + str(i) not in data[uid]:
            wrong_keys += 1
        else:
            correct_keys += 1
        try:
            CADSequence.from_vec(
                data[uid]["level_" + str(i)]["pred_cad_vec"][0], 2, 8, True
            ).save_stp(
                filename=f"{sample_id}_final_level_" + str(i),
                output_folder=output_folder,
                type="step",
            )
            count_saved += 1
        except Exception as e:
            # print(f"Error in {uid} level {i}")
            # print(e)
            continue

    return count_saved, correct_keys, wrong_keys


def save_step(input_path, output_dir, max_workers):
    with open(input_path, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} records from {input_path}")

    uids = list(data.keys())
    print(f"Loaded {len(uids)} uids")

    total_count_saved = 0
    total_correct_keys = 0
    total_wrong_keys = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_uid, uid, data, output_dir): uid for uid in uids
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing files"
        ):
            count_saved, correct_keys, wrong_keys = future.result()
            total_count_saved += count_saved
            total_correct_keys += correct_keys
            total_wrong_keys += wrong_keys

    print(
        f"{total_count_saved} step files saved in {output_dir} folder out of total {len(uids) * 4} samples"
    )
    print(f"Correct keys: {total_correct_keys}")
    print(f"Wrong keys: {total_wrong_keys}")
    print(f"Total keys: {total_correct_keys + total_wrong_keys}")
    print(
        f"Invalidity Ratio = {1-total_count_saved / (total_correct_keys + total_wrong_keys)}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument(
        "--input_path", help="Predicted CAD Sequence in pkl format", required=True
    )
    parser.add_argument("--output_dir", help="Output dir", required=True)
    parser.add_argument("--max_workers", help="Number of workers", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    save_step(
        input_path=args.input_path,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
    )
