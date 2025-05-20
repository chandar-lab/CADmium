from tqdm import tqdm
import pandas as pd
import os
from omegaconf import DictConfig
import hydra
from dotenv import dotenv_values
import openai
import base64
import time

##############################
###          MAIN          ###
##############################

@hydra.main(version_base=None, config_path="../../config/", config_name="annotate")
def main(config: DictConfig) -> None:
    import json

    n_splits = getattr(config, "n_splits", 1)
    idx_split = getattr(config, "idx_split", 0)

    api_keys = dotenv_values(config.api_keys_path)
    OPENAI_API_KEY = api_keys.get('OPENAI_API_KEY')

    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    model_id = config.model.model_id

    df = pd.read_csv(config.data.original_annotations).reset_index(drop=True)

    print("df shape:", df.shape)

    output_dir = config.data.output_dir
    os.makedirs(output_dir, exist_ok=True)

    uids = df['uid'].to_numpy()
    # Split the uids list
    split_size = (len(uids) + n_splits - 1) // n_splits
    start = idx_split * split_size
    end = min((idx_split + 1) * split_size, len(uids))
    split_uids = uids[start:end]

    pbar = tqdm(split_uids, total=len(split_uids))
    pbar.set_description(f"Split {idx_split+1}/{n_splits}")

    for uid_full in pbar:
        uid = uid_full.split("/") # 0044, 00448252
        uid_section = uid_to_section(uid[0])

        output_path = os.path.join(output_dir, uid[0], uid[1], "gpt_annotation.txt")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if os.path.exists(output_path):
            pbar.set_postfix_str(f"File exists, skipping {uid[0]}_{uid[1]}")
            continue

        json_desc_path = json_path_template.format(
            uid_start=uid[0],
            uid_end=uid[1],
        )
        json_desc = json.loads(open(json_desc_path).read())
        json_desc.pop('final_name')
        json_desc.pop('final_shape')
        json_desc = str(json_desc)

        img_paths = [img_path_template.format(
                uid_section_start=uid_section[0],
                uid_section_end=uid_section[1],
                uid_start=uid[0],
                uid_end=uid[1],
                view=view
            ) for view in views]

        prompt = prompt_template.format(
                json_desc=json_desc
            )

        content = [
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{encode_image(img_path)}"
                } for img_path in img_paths
            ] + [
                {
                    "type": "input_text",
                    "text": prompt
                }
            ]

        generated = False
        while not generated:
            try:
                response = client.responses.create(
                    model=model_id,
                    input=[{"role": "user", "content": content}]
                )
                generated = True
            except Exception as e:
                pbar.set_postfix_str("Rate limit error, retrying in 5s")
                print(e)
                time.sleep(5)
                continue

        with open(output_path, "w") as f:
            f.write(response.output_text)



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


`