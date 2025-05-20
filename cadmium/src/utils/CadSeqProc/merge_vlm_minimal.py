import os
import json
import argparse
from tqdm import tqdm
from glob import glob
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

 # ---------------------------------------------------------------------------- #
 #                            Only for Text2CAD v1.1                            #
 # ---------------------------------------------------------------------------- #


def extract_shape_info(input_string):
    # Function to extract content between tags
    def extract_content(tag, text):
        pattern = f"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None

    # Extracting values
    name = extract_content("NAME", input_string)
    description = extract_content("DESCRIPTION", input_string)
    keywords = extract_content("KEYWORDS", input_string)

    # Converting keywords to a list
    if keywords:
        keywords = [keyword.strip() for keyword in keywords.split(",")]

    # Creating the dictionary
    return {
        "name": "" if name is None else name,
        "description": "" if description is None else description,
        "keywords": "" if keywords is None else keywords,
    }


def process_single(root_dir, uid):
    root_id, sample_id = uid.split("/")

    with open(
        os.path.join(root_dir, uid, "minimal_json", f"{sample_id}.json"), "r"
    ) as f:
        data = json.load(f)

    all_vlm_annotations = glob(
        os.path.join(root_dir, uid, "qwen2_vlm_annotation/*_*.json")
    )
    all_vlm_annot_dict = {}
    for vlm_annot in all_vlm_annotations:
        file_name = os.path.basename(vlm_annot)
        key_name = (
            "final"
            if "final" in file_name
            else "part_" + file_name.split("_")[-1].strip(".json")
        )

        with open(vlm_annot, "r") as f:
            vlm_data = json.load(f)
            all_vlm_annot_dict[key_name] = extract_shape_info(vlm_data)

    data["final_name"] = all_vlm_annot_dict["final"]["name"]
    data["final_shape"] = all_vlm_annot_dict["final"]["description"]
    for key, val in data["parts"].items():
        if key not in all_vlm_annot_dict:
            annot_key = "final"
        else:
            annot_key = key
        val["description"]["name"] = all_vlm_annot_dict[annot_key]["name"]
        val["description"]["shape"] = all_vlm_annot_dict[annot_key]["description"]

    return data


def process_uid(uid, root_dir, output_dir):
    try:
        root_id, sample_id = uid.split("/")
        merged_metadata = process_single(root_dir, uid)
        output_path = os.path.join(
            output_dir, uid, "minimal_json", f"{sample_id}_merged_vlm.json"
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(merged_metadata, f, indent=4)
    except Exception as e:
        return f"Error in processing {uid}: {e}"
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", type=str
    )
    parser.add_argument(
        "--split_json",
        type=str,
    )
    parser.add_argument(
        "--output_dir", type=str
    )
    parser.add_argument("--max_workers", type=int, default=8)
    
    args = parser.parse_args()

    with open(args.split_json, "r") as f:
        split_json_data = json.load(f)

    all_uids = (
        split_json_data["train"]
        + split_json_data["test"]
        + split_json_data["validation"]
    )

    errors = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(process_uid, uid, args.input_dir, args.output_dir): uid
            for uid in all_uids
        }

        for future in tqdm(as_completed(futures), desc="Processing", total=len(all_uids)):
            error = future.result()
            if error:
                errors.append(error)

    if errors:
        print("Some errors occurred during processing:")
        for error in errors:
            print(error)


if __name__ == "__main__":
    main()
