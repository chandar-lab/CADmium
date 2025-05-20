from omegaconf import OmegaConf
import pandas as pd
import os
import numpy as np
from dotenv import dotenv_values
import openai
import json
import base64
from tqdm import tqdm

def random_reshuffle_string(string):
    words = string.split()
    np.random.shuffle(words)
    return ' '.join(words)

def process_batch(examples, tokenizer, system_message, max_length=4096):
    """
    Processes a batch of samples for autoregressive fine tuning in a chat format.
    
    For each sample in the batch, the conversation is built as:
      [SYSTEM]   <system_message>
      [USER]     <prompt>
      [ASSISTANT] <target_text>
    
    The entire conversation is tokenized in one pass. Then, for each sample,
    we search for the assistant marker token sequence (e.g. "<|im_start|>assistant\n")
    and mask all tokens up to and including that marker (set them to -100) so that only
    the assistant's answer contributes to the loss.
    
    Args:
        examples (dict): A dictionary with keys "uid", "prompt", and "json_desc".
                         Each value is a list of items.
        tokenizer: A Hugging Face tokenizer that implements apply_chat_template.
        system_message (str): The system message text.
        max_length (int): Maximum length for the entire tokenized sequence.
    
    Returns:
        dict: A dictionary with keys "uid", "input_ids", "attention_mask", "labels".
              The values are lists, one per sample.
    """
    
    # 1. Process the json_desc field for each sample.
    target_texts = []
    for json_desc in examples["json_desc"]:
        target_text = json_desc
        target_texts.append(target_text)
    
    # 2. Build the conversation messages and full text for each sample.
    messages = []
    for prompt, target_text in zip(examples["annotation"], target_texts): ## annotation = prompt for Text2CAD data
        message = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": random_reshuffle_string(prompt)},
            {"role": "assistant", "content": target_text},
        ]
        messages.append(message)
    full_texts = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    # 3. Tokenize the entire batch of conversations.
    full_tokens = tokenizer(
        full_texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )
    
    input_ids = full_tokens["input_ids"]
    attention_mask = full_tokens["attention_mask"]
    
    # 4. Determine the assistant marker and its token ids.
    marker_text = "<|im_start|>assistant\n"
    marker_ids = tokenizer(marker_text, add_special_tokens=False)["input_ids"]
    
    # 5. Create labels by copying input_ids and masking tokens before the assistant answer.
    labels = input_ids.clone()
    # Convert input_ids to a list of lists for easier per-sample processing.
    input_ids_list = input_ids.tolist()
    for i, ids in enumerate(input_ids_list):
        idx = find_sublist(ids, marker_ids)
        if idx == -1:
            mask_boundary = 0  # If marker not found, do not mask any tokens.
        else:
            # Mask tokens up to (and including) the marker.
            mask_boundary = idx + len(marker_ids)
        labels[i, :mask_boundary] = -100

    # 6. Return the processed batch.
    return {
        "uid": examples["uid"],
        "prompt": full_texts,
        "input_ids": input_ids.tolist(),
        "attention_mask": attention_mask.tolist(),
        "labels": labels.tolist(),
    }

def find_sublist(sequence, sub):
    sub_len = len(sub)
    for i in range(len(sequence) - sub_len + 1):
        if sequence[i:i+sub_len] == sub:
            return i
    return -1


config = OmegaConf.load("../../config/annotate_cadmium.yaml")
annotated_df = pd.read_csv('/home/mila/b/baldelld/scratch/LLM4CAD/cadmium/data/text2cad_v1.1/text2cad_v1.1.csv')

json_path_template = "/home/mila/b/baldelld/scratch/LLM4CAD/cadmium/data/text2cad_v1.1/jsons/{uid_start}/{uid_end}/minimal_json/{uid_end}.json"

uids = annotated_df['uid'].tolist()
json_descs = []
for uid in tqdm(uids):
    uid_start, uid_end = uid.split('/')
    desc_path = json_path_template.format(uid_start=uid_start, uid_end=uid_end)
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

annotated_abstract = annotated_df[['uid', 'abstract']]
print('Yeah')
annotated_intermediate = annotated_df[['uid', 'intermediate']]
annotated_expert = annotated_df[['uid', 'expert']]
annotated_beginner = annotated_df[['uid', 'beginner']]

annotated_abstract.rename(columns={'abstract': 'annotation'}, inplace=True)
annotated_intermediate.rename(columns={'intermediate': 'annotation'}, inplace=True)
annotated_expert.rename(columns={'expert': 'annotation'}, inplace=True)
annotated_beginner.rename(columns={'beginner': 'annotation'}, inplace=True)

print('Uhuuhyeah')

annotated_abstract['uid'] = annotated_abstract['uid'].apply(lambda x: x + '_abstract')
print('Uuhuuhyeah')
annotated_intermediate['uid'] = annotated_intermediate['uid'].apply(lambda x: x + '_intermediate')
annotated_expert['uid'] = annotated_expert['uid'].apply(lambda x: x + '_expert')
annotated_beginner['uid'] = annotated_beginner['uid'].apply(lambda x: x + '_beginner')

df = pd.concat([annotated_abstract, annotated_intermediate, annotated_expert, annotated_beginner], axis=0)

json_desc_beginner = json_descs.copy()
json_desc_beginner['uid'] = json_desc_beginner['uid'].apply(lambda x: x + '_beginner')
json_desc_intermediate = json_descs.copy()
json_desc_intermediate['uid'] = json_desc_intermediate['uid'].apply(lambda x: x + '_intermediate')
json_desc_expert = json_descs.copy()
json_desc_expert['uid'] = json_desc_expert['uid'].apply(lambda x: x + '_expert')
json_desc_abstract = json_descs.copy()
json_desc_abstract['uid'] = json_desc_abstract['uid'].apply(lambda x: x + '_abstract')

json_descs = pd.concat([json_desc_beginner, json_desc_intermediate, json_desc_expert, json_desc_abstract], axis=0)

df = df.merge(json_descs, on='uid', how='left')
print('Merged')

splits_path = '/home/mila/b/baldelld/scratch/LLM4CAD/cadmium/data/text2cad_v1.1/train_test_val.json'
splits = json.load(open(splits_path, 'r'))

splits['train'] = [uid + f'{level}' for uid in splits['train'] for level in ['_beginner', '_intermediate', '_expert', '_abstract']]
splits['validation'] = [uid + f'{level}' for uid in splits['validation'] for level in ['_beginner', '_intermediate', '_expert', '_abstract']]
splits['test'] = [uid + f'{level}' for uid in splits['test'] for level in ['_beginner', '_intermediate', '_expert', '_abstract']]

df_train = df[df['uid'].isin(splits['train'])]
df_val = df[df['uid'].isin(splits['validation'])]
df_test = df[df['uid'].isin(splits['test'])]

print('Splitted')


from datasets import Dataset, DatasetDict

train_dataset = Dataset.from_pandas(df_train, preserve_index=False)
val_dataset = Dataset.from_pandas(df_val, preserve_index=False)
test_dataset = Dataset.from_pandas(df_test, preserve_index=False)

dataset = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
    "test": test_dataset
})

print('Dataset created')
print(dataset)

from datasets import load_dataset
from cadmium.utils.prompts import SYSTEM_MESSAGE, SYSTEM_MESSAGES
from cadmium.utils.utils import process_batch
from transformers import AutoTokenizer
from omegaconf import OmegaConf

config = OmegaConf.load("../../config/process_data.yaml")

SYSTEM_MESSAGE = SYSTEM_MESSAGES['schema_imperative_noindent']
tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)

dataset['train'] = dataset['train'].shuffle(seed=42).select(range(20000))
dataset['train'] = dataset['train'].filter(lambda x: x['annotation'] is not None)

dataset['validation'] = dataset['validation'].shuffle(seed=42).select(range(10))
dataset['validation'] = dataset['validation'].filter(lambda x: x['annotation'] is not None)
dataset['test'] = dataset['test'].shuffle(seed=42).select(range(10))
dataset['test'] = dataset['test'].filter(lambda x: x['annotation'] is not None)
processed_dataset = dataset.map(
    lambda x: process_batch(x, tokenizer, SYSTEM_MESSAGE, config.data.max_length),
    batched=True,
    remove_columns=dataset['train'].column_names,
    desc="Processing dataset",
    num_proc=8,
)

train_path = "/network/scratch/b/baldelld/LLM4CAD/cadmium/data/text2cad_v1.1/train_json_qwen_retokenized.parquet"

processed_dataset['train'].to_parquet(train_path)