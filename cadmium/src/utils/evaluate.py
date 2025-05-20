import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import copy
import json
import os
import re
from cadmium.src.utils.macro import CAD_CLASS_INFO, END_TOKEN, DATA_DIR

from copy import copy
import hashlib
import json
import os
import wandb
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
from trl import apply_chat_template
from datasets import load_dataset
from cadmium.src.utils.logger import CLGLogger
from cadmium.src.utils.utils import find_sublist
from torch.utils.tensorboard import SummaryWriter
import torch
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from datasets import Dataset
from tqdm import tqdm
import numpy as np
import pandas as pd
import datetime
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from cadmium.src.dataprep.t2c_dataset import Text2CADJSON_Dataset
from transformers import get_linear_schedule_with_warmup
import random
import gc
from cadmium.src.utils.prompts import SYSTEM_MESSAGE
from cadmium.src.utils.macro import (END_TOKEN, 
                                MAX_CAD_SEQUENCE_LENGTH, 
                                CAD_CLASS_INFO, 
                                )

from concurrent.futures import ProcessPoolExecutor
from functools import partial
import os
import pandas as pd
from tqdm import tqdm
import json


def switch_system_message(
    dataset, 
    new_system_message, 
    tokenizer,
):
    new_system_message_tokenized = tokenizer.apply_chat_template(
        [{"role": "system", "content": new_system_message}], 
        tokenize=True
    )
    # Get marker texts from config (or default to these values)
    user_marker = "<|im_start|>user\n"
    # Tokenize marker texts (without adding special tokens)
    user_marker_ids = tokenizer(user_marker, add_special_tokens=False)["input_ids"]
    # user_idx does not depend on the datapoint, we can find it once for all the dataset
    user_idx_char = find_sublist(dataset[0]['prompt'], user_marker)
    user_idx_tok = find_sublist(dataset[0]['input_ids'], user_marker_ids)

    dataset = dataset.map(
        lambda ex: switch_system_message_per_sample(
            ex, 
            tokenizer, 
            new_system_message, 
            new_system_message_tokenized, 
            user_idx_char, 
            user_idx_tok, 
            ),
        batched=False, 
    )

    return dataset

def switch_system_message_per_sample(
    example, 
    tokenizer, 
    new_system_message, 
    new_system_message_tokenized,
    user_idx_char,
    user_idx_tok,  
    ):
    prompt = example['prompt']
    input_ids = example['input_ids']
    pad_id = tokenizer.pad_token_id

    new_prompt = new_system_message + prompt[user_idx_char:]
    new_input_ids = new_system_message_tokenized + input_ids[user_idx_tok:]

    if len(new_input_ids) > len(input_ids):
        new_input_ids = new_input_ids[:len(input_ids)]
    else:
        new_input_ids = new_input_ids + [pad_id]*(len(input_ids) - len(new_input_ids))

    n_tokens = (np.array(new_input_ids)!=pad_id).sum()
    attention_mask = [1]*n_tokens + [0]*(len(input_ids)-n_tokens)

    assistant_marker = "<|im_start|>assistant\n"
    assistant_marker_ids = tokenizer(assistant_marker, add_special_tokens=False)["input_ids"]
    assistant_idx_tok = find_sublist(new_input_ids, assistant_marker_ids)
    

    labels = [-100]*(assistant_idx_tok+len(assistant_marker_ids)) + new_input_ids[assistant_idx_tok+len(assistant_marker_ids):]
    
    return {
        "uid": example["uid"],
        "input_ids": new_input_ids,
        "attention_mask": attention_mask,
        "prompt": new_prompt,
        "labels": labels,
    }
    
def collate_fn(batch):
    # For fields that are strings or dictionaries, just keep them as a list.
    uids = [sample['uid'] for sample in batch]
    prompts = [sample['prompt'] for sample in batch]
    # json_descs = [sample['json_desc'] for sample in batch]
    
    # For sequence fields, pad them to the same length.
    # (Here we assume they are lists of integers.)
    def pad_sequences(seqs, pad_value=0):
        max_len = max(len(seq) for seq in seqs)
        return [[pad_value]*(max_len - len(seq))+seq  for seq in seqs]
    
    input_ids = torch.tensor(pad_sequences([sample['input_ids'] for sample in batch]))
    attention_masks = torch.tensor(pad_sequences([sample['attention_mask'] for sample in batch]))
    labels = torch.tensor(pad_sequences([sample['labels'] for sample in batch], pad_value=-100))
    
    return {
        'uid': uids,
        'prompt': prompts,
        # 'json_desc': json_descs,
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'labels': labels
    }

def process_sample(sample, tokenizer, config):
    """
    Process a single sample from the dataset.
    Extracts and desrc the true response, finds marker indices,
    crops the input up to the assistant marker, and returns a dictionary
    containing all relevant data needed for generation.
    """
    # Extract raw fields
    input_ids = sample["input_ids"]
    attention_mask = sample["attention_mask"]

    # Process labels into the true response string
    labels = sample["labels"]
    labels = labels[labels != -100]
    true_response = tokenizer.decode(labels, skip_special_tokens=True)

    # Get marker texts from config (or default to these values)
    assistant_marker = getattr(config.eval, "marker_assistant", "<|im_start|>assistant\n")
    user_marker = getattr(config.eval, "marker_user", "<|im_start|>user\n")

    # Tokenize marker texts (without adding special tokens)
    assistant_marker_ids = tokenizer(assistant_marker, add_special_tokens=False)["input_ids"]
    user_marker_ids = tokenizer(user_marker, add_special_tokens=False)["input_ids"]

    # Find marker indices in the input_ids
    assistant_idx = find_sublist(input_ids.tolist(), assistant_marker_ids)
    user_idx = find_sublist(input_ids.tolist(), user_marker_ids)

    if assistant_idx == -1 or user_idx == -1:
        raise ValueError(f"Markers not found in sample uid {sample.get('uid', 'unknown')}.")

    # Get object description: slice from after the user marker to before the assistant marker.
    # (Adjust the offsets as needed; here we mimic your slicing: user_idx+3 to assistant_idx-2)
    desc = tokenizer.decode(input_ids[user_idx + 3 : assistant_idx - 2])

    # Crop input_ids and attention_mask up to the assistant marker (+3 tokens)
    cropped_input_ids = input_ids[: assistant_idx + 3]
    cropped_attention_mask = attention_mask[: assistant_idx + 3]

    # Convert to tensors (they might not be tensors already)
    input_tensor = cropped_input_ids
    mask_tensor = cropped_attention_mask
    original_length = len(cropped_input_ids)  # needed later to slice out only the generated part

    return {
        "uid": sample["uid"],
        "input_ids": input_tensor,
        "attention_mask": mask_tensor,
        "true_response": true_response,
        "desc": desc,
        "original_length": original_length,
    }

def evaluate_model(dataset, tokenizer, model, config, output_dir=None, save_per_batch=False):
    """
    Evaluates the model on the given validation dataset in batches.
    The dataset is expected to be a dictionary that can be converted to a Hugging Face Dataset.
    Hyperparameters for evaluation (batch size, generation parameters, etc.) are taken from config.eval.
    """
    if save_per_batch:
        if output_dir is None:
            raise ValueError("output_dir must be provided when save_per_batch is True.")
        os.makedirs(os.path.join(output_dir, 'predictions'), exist_ok=True)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if type(dataset) == dict:
        dataset = Dataset.from_dict(dataset)

    batch_size = config.eval.batch_size
    # Create a DataLoader that yields batches of raw samples
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    # Inside evaluate_model, after determining batch_size:
    if dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=False)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    results = []
    model.eval()
    batch_idx = 0
    for batch in tqdm(dataloader, desc=f"Evaluating on device {local_rank}"):
        processed_samples = []
        if save_per_batch:
            if os.path.exists(os.path.join(output_dir, 'predictions', f"{batch['uid'][0].replace('/','_')}.json")) or os.path.exists(os.path.join(output_dir, f"{batch['uid'][0].replace('/','_')}.json")):
                batch_idx += 1
                continue
        # Process each sample in the batch individually
        batch_size_actual = len(batch["uid"])
        for i in range(batch_size_actual):
            sample = {key: batch[key][i] for key in batch}
            try:
                proc = process_sample(sample, tokenizer, config)
            except Exception as e:
                print(f"Error processing sample {sample.get('uid', 'unknown')}: {e}")
                continue
            processed_samples.append(proc)

        if not processed_samples:
            continue

        # Collect tensors and original lengths from processed samples
        input_tensors = [s["input_ids"] for s in processed_samples]
        mask_tensors = [s["attention_mask"] for s in processed_samples]
        original_lengths = [s["original_length"] for s in processed_samples]

        # Pad sequences to the maximum length in the batch
        input_ids_padded = pad_sequence(
            input_tensors, batch_first=True, padding_value=tokenizer.pad_token_id, padding_side='left'
        ).to(device)
        attention_mask_padded = pad_sequence(
            mask_tensors, batch_first=True, padding_value=0, padding_side='left'
        ).to(device)
        batch_seq_len = len(input_ids_padded[0])

        # Optionally add more generation parameters if defined in config
        gen_kwargs = {}
        for param in ["temperature", "do_sample", "top_p", "top_k", "max_new_tokens"]:
            if hasattr(config.eval, param):
                gen_kwargs[param] = getattr(config.eval, param)

        with torch.no_grad():
            generated_batch = model.generate(
                input_ids=input_ids_padded,
                attention_mask=attention_mask_padded,
                **gen_kwargs
            )

        # For each sample in the batch, extract the generated tokens (i.e. after the original input)
        generated_texts = []
        for i in range(len(generated_batch)):
            gen_ids = generated_batch[i][batch_seq_len:]  # slice off the original prompt
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            generated_texts.append(text)

        # Prepare the final results for this batch
        for idx, s in enumerate(processed_samples):
            pred_response = generated_texts[idx]
            try:
                true_response_dict = eval(s["true_response"].replace("null", "None"))
            except Exception:
                true_response_dict = None
            try:
                pred_response_dict = eval(postprocess_response(pred_response))
            except Exception:
                pred_response_dict = None
            if save_per_batch:
                output_path = os.path.join(output_dir, 'predictions', f"{s['uid'].replace('/','_')}.json")
                with open(output_path, "w") as f:
                    json.dump(
                        {
                            "uid": s["uid"],
                            "object_description": s["desc"],
                            "true_response": s["true_response"],
                            "pred_response": pred_response,
                            "true_response_dict": true_response_dict,
                            "pred_response_dict": pred_response_dict,
                        },
                        f,
                    )
            else:
                results.append(
                    {
                        "uid": s["uid"],
                        "object_description": s["desc"],
                        "true_response": s["true_response"],
                        "pred_response": pred_response,
                        "true_response_dict": true_response_dict,
                        "pred_response_dict": pred_response_dict,
                    }
                )
        batch_idx += 1
    return results

def postprocess_response(response):
    return response.replace("null", "None").replace("`", "").replace("json", "")

def load_single_json(result_dir, filename):
    """Helper function to load a single JSON file."""
    filepath = os.path.join(result_dir, filename)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

class PredictionsLoader:
    def __init__(self, result_dir, filenames=None):
        self.result_dir = os.path.join(result_dir, 'predictions')
        if filenames is None:
            self.filenames = [f for f in os.listdir(self.result_dir) if f.endswith('.json')]
        else:
            self.filenames = filenames

    def load_parallel(self, max_workers=None, chunksize=100):
        """Load all JSON files in parallel and return a DataFrame."""
        # Bind result_dir to the helper function
        load_func = partial(load_single_json, self.result_dir)
        
        # Use ProcessPoolExecutor to parallelize loading
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(load_func, self.filenames, chunksize=chunksize))
        
        # Filter out any failed loads and create DataFrame
        valid_results = [res for res in results if res is not None]
        return pd.DataFrame(valid_results)



def save_generated_jsons(result_dir: str):    
    # if result_dir not in os.listdir(os.path.join(DATA_DIR, 'results')):
    #     raise FileNotFoundError(f"The specified directory '{result_dir}' does not exist in {os.path.join(DATA_DIR, 'results')}.")
    # else:
    #     result_dir = os.path.join(DATA_DIR, 'results', result_dir)

    files = os.listdir(result_dir)
    output_path = os.path.join(DATA_DIR, 'generated_jsons', result_dir.split('/')[-2])
    os.makedirs(output_path, exist_ok=True)
    
    for file in tqdm(files, desc=f"Processing files in {result_dir}"):
        if file.endswith('.csv') and 'results' in file:
            df = pd.read_csv(os.path.join(result_dir, file))
            len(df)
            for i in range(len(df)):
                name = df['uid'].loc[i]
                user_path = os.path.join(output_path, name)
                os.makedirs(user_path, exist_ok=True)
                
                json_file_path = os.path.join(user_path, f"{name.split('/')[1]}.json")
                try:
                    with open(json_file_path, 'w') as f:
                        json.dump(eval(df['pred_response_dict'][i]), f, indent=4)
                except Exception as e:
                    print(name)
                    # pass  # Ignore errors during JSON dumping

