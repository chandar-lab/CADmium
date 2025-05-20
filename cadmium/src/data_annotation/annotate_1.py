import os
import time
import json
import pandas as pd
import openai
from dotenv import dotenv_values
from concurrent.futures import ThreadPoolExecutor, as_completed
from omegaconf import DictConfig
import hydra
from tqdm import tqdm
import base64


# Note: prompt_template, sections, views, img_path_template, and json_path_template
# are assumed to be defined elsewhere or in the Hydra config.


def process_one(uid_full: str, config: DictConfig, client, model_id: str) -> str:
    """
    Process a single UID: check for existing output, load inputs, call API, and write result.
    """
    # Derive section and file paths
    uid = uid_full.split("/")      # ['0044','00448252']
    uid_section = uid_to_section(uid[0])
    output_path = os.path.join(
        config.data.output_dir,
        uid[0], uid[1],
        "gpt_annotation.txt"
    )
    # Skip if already annotated
    if os.path.exists(output_path):
        return f"skipped {uid_full}"

    # Load JSON description
    desc_path = json_path_template.format(
        uid_start=uid[0], uid_end=uid[1]
    )
    with open(desc_path, 'r') as f:
        j = json.load(f)
    # Remove unwanted fields
    j.pop('final_name', None)
    j.pop('final_shape', None)
    json_desc = str(j)

    # Prepare images + prompt
    content = []
    for view in views:
        img_path = img_path_template.format(
            uid_section_start=uid_section[0], uid_section_end=uid_section[1],
            uid_start=uid[0], uid_end=uid[1], view=view
        )
        b64 = encode_image(img_path)
        content.append({
            "type": "input_image",
            "image_url": f"data:image/png;base64,{b64}"
        })
    content.append({
        "type": "input_text",
        "text": prompt_template.format(json_desc=json_desc)
    })

    # Call OpenAI with retry on rate limits
    while True:
        try:
            resp = client.responses.create(
                model=model_id,
                input=[{"role": "user", "content": content}]
            )
            break
        except Exception:
            print("Rate limit exceeded, retrying in 5s ...")
            time.sleep(5)

    # Write output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(resp.output_text)

    return f"done {uid_full}"


@hydra.main(version_base=None, config_path="../../config/", config_name="annotate")
def main(config: DictConfig) -> None:
    # Load API key
    api_keys = dotenv_values(config.api_keys_path)
    client = openai.OpenAI(api_key=api_keys['OPENAI_API_KEY'])
    model_id = config.model.model_id

    # Read UIDs
    df = pd.read_csv(config.data.original_annotations).reset_index(drop=True)

    uids = df['uid'].to_numpy()

    # Compute split indices
    n_splits = config.n_splits
    idx_split = config.idx_split
    split_size = (len(uids) + n_splits - 1) // n_splits
    start = idx_split * split_size
    end = min((idx_split + 1) * split_size, len(uids))
    split_uids = uids[start:end]

    # Use a thread pool to overlap API calls and enable resumability
    max_workers = 80 # n_cpus * 10

    print(f"Processing {len(split_uids)} UIDs in split {idx_split+1}/{n_splits} with {max_workers} workers.")
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                process_one,
                uid,
                config,
                client,
                model_id
            ): uid
            for uid in split_uids
        }
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc=f"Split {idx_split+1}/{n_splits}"):
            result = future.result()
            tqdm.write(result)


###############################
###          UTILS          ###
###############################


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

prompt_template = """You are an expert mechanical engineer tasked with creating clear, precise instructions for a text-to-CAD generator.

I have a set of 9 multi-view images displaying a 3D model, as well as a JSON file describing the exact CAD operations used to construct the object.

This is the json file:
```json
{json_desc}
```

Create a single, comprehensive text description of this 3D object that:
- Describes all geometrical features accurately based on the operations and dimensions
- Uses natural language as if a human designer were explaining how to model this object
- Is written in second-person as instructions for a text-to-CAD system
- Includes all critical dimensions and geometric relationships (note that you don't need to specify the unit of measurement for lengths)
- Avoids redundancy while ensuring completeness
- Focuses on the design intent and functional geometry
- Answer only with the description. No introductory phrases, titles, commentary, summaries or conclusions

Your description should be concise but complete, capturing every important geometric feature without unnecessary repetition.
"""

sections = [["0000", "0005"],] + [[f"{i:04d}", f"{i+4:04d}"] for i in range(6, 95, 5)] + [["0096", "0099"],]
views = ["000", "001", "002", "003", "004", "005", "006", "007", "bottom", "top"]
img_path_template = "/home/mila/b/baldelld/scratch/LLM4CAD/cadmium/data/text2cad_v1.1/rgb_images/rgb_images_1_{uid_section_start}_{uid_section_end}/{uid_start}/{uid_end}/mvi_rgb_blender/{uid_end}_final/{uid_end}_final_{view}.png"
json_path_template = "/home/mila/b/baldelld/scratch/LLM4CAD/cadmium/data/text2cad_v1.1/jsons/{uid_start}/{uid_end}/minimal_json/{uid_end}.json"


def uid_to_section(uid):
    """
    Given a uid, return the section it belongs to.
    """
    uid = int(uid)
    
    for sec in sections:
        if uid <= int(sec[1]):
            return sec


if __name__ == "__main__":
    main()
