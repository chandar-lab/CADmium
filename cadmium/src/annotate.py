from tqdm import tqdm
import pandas as pd
from datasets import Dataset, DatasetDict
import os
from omegaconf import DictConfig
import hydra
from dotenv import dotenv_values
import openai
import base64
import time
import json
from cadmium.src.utils.prompts import ANNOTATION_PROMPT_TEMPLATE

sections = [["0000", "0005"],] + [[f"{i:04d}", f"{i+4:04d}"] for i in range(6, 95, 5)] + [["0096", "0099"],]
views = ["000", "001", "002", "003", "004", "005", "006", "007", "bottom", "top"]
img_path_template = "data/text2cad_v1.1/rgb_images/rgb_images_1_{uid_section_start}_{uid_section_end}/{uid_start}/{uid_end}/mvi_rgb_blender/{uid_end}_final/{uid_end}_final_{view}.png"
json_path_template = "data/text2cad_v1.1/jsons/{uid_start}/{uid_end}/minimal_json/{uid_end}.json"

@hydra.main(version_base=None, config_path="../../config/", config_name="annotate")
def main(config: DictConfig) -> None:

    n_splits = getattr(config, "n_splits", 1)
    idx_split = getattr(config, "idx_split", 0)

    api_keys = dotenv_values(config.api_keys_path)
    OPENAI_API_KEY = api_keys.get('OPENAI_API_KEY')

    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    model_id = config.model.model_id

    df = pd.read_csv(config.data.original_annotations).reset_index(drop=True)
    uids = df['uid'].to_numpy()

    output_dir = config.data.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Split the uids list
    split_size = (len(uids) + n_splits - 1) // n_splits
    start = idx_split * split_size
    end = min((idx_split + 1) * split_size, len(uids))
    split_uids = uids[start:end]

    pbar = tqdm(split_uids, total=len(split_uids))
    pbar.set_description(f"Split {idx_split+1}/{n_splits}")

    # ------- Generate annotations -------

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

        prompt = ANNOTATION_PROMPT_TEMPLATE.format(
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

    # ------- Gather and save as HF dataset -------

    output_dir = config.data.output_dir
    annotated_df = []
    for uid_start in tqdm(os.listdir(output_dir)):
        for uid_end in os.listdir(os.path.join(output_dir, uid_start)):
            if os.path.exists(os.path.join(output_dir, uid_start, uid_end, 'gpt_annotation.txt')):
                annotated_df.append({
                    'uid': uid_start + '/' + uid_end, 
                    'annotation': open(os.path.join(output_dir, uid_start, uid_end, 'gpt_annotation.txt')).read()
                })
    annotated_df = pd.DataFrame(annotated_df)

    json_descs = []
    for uid in tqdm(uids):
        uid_start, uid_end = uid.split('/')
        desc_path = json_path_template.format(uid_start=uid_start, uid_end=uid_end)
        if not os.path.exists(desc_path):
            print(f"File {desc_path} does not exist")
            continue
        with open(desc_path, 'r') as f:
            j = json.load(f)
        # Remove unwanted fields
        j.pop('final_name', None)
        j.pop('final_shape', None)
        json_desc = json.dumps(j)
        json_descs.append({
            'uid': uid,
            'json_desc': json_desc
        })
    json_descs = pd.DataFrame(json_descs)
    df = annotated_df.merge(json_descs, on='uid', how='left')

    splits_path = os.path.join(config.data.base_dir, 'train_test_val.json')
    splits = json.load(open(splits_path, 'r'))

    df_train = df[df['uid'].isin(splits['train'])]
    df_val = df[df['uid'].isin(splits['validation'])]
    df_test = df[df['uid'].isin(splits['test'])]

    train_dataset = Dataset.from_pandas(df_train, preserve_index=False)
    val_dataset = Dataset.from_pandas(df_val, preserve_index=False)
    test_dataset = Dataset.from_pandas(df_test, preserve_index=False)

    dataset = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })

    dataset.save_to_disk(config.data.base_dir.output_ds_path)




def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

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
