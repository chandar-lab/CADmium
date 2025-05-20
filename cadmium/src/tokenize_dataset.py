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
from datasets import load_dataset, Dataset
from models.loss import CELoss
from cadmium.utils.logger import CLGLogger
from torch.utils.tensorboard import SummaryWriter
import torch
from tqdm import tqdm
import numpy as np
import datetime
from cadmium.utils.evaluate import switch_system_message, evaluate_model
from cadmium.utils.prompts import SYSTEM_MESSAGE, SYSTEM_MESSAGES
from datasets import Dataset
from huggingface_hub import login

# from peft import LoraConfig

from cadmium.codes.dataprep.t2c_dataset import Text2CADJSON_Dataset
from transformers import AdamW, get_linear_schedule_with_warmup
import random
from models.metrics import AccuracyCalculator
from loguru import logger
from cadmium.utils.prompts import SYSTEM_MESSAGE
from cadmium.utils.macro import (END_TOKEN, 
                                MAX_CAD_SEQUENCE_LENGTH, 
                                CAD_CLASS_INFO, 
                                )

@hydra.main(version_base=None, config_path="../config", config_name="process_data")
def main(config: DictConfig):
    # ----------------- Loading the model -----------------
    SYSTEM_MESSAGE = SYSTEM_MESSAGES['schema_imperative_noindent']
    print("Loading Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)

    login(token="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")  # Or set HF_TOKEN env variable
    dataset = load_dataset("LLM4CAD/LLM4CAD")

    if "train" in splits:
        print("Loading Train Dataset")
        train_dataset = Text2CADJSON_Dataset(
            prompt_path=config.data.prompt_path,
            json_desc_dir=config.data.json_desc_dir,
            split_filepath=config.data.split_filepath,
            subset="train",
            max_workers=32,
            #max_n_datapoints=config.data.max_n_datapoints_train,
            verbose=False,
        )

    print("Processing datasets")
    if "train" in splits:
        processed_train_dataset = train_dataset.map(
            lambda ex: process_batch(
                ex, 
                tokenizer, 
                SYSTEM_MESSAGE, 
                max_length=config.data.max_length),
            batched=True
        )
    if "validation" in splits:
        processed_val_dataset = val_dataset.map(
            lambda ex: process_batch(
                ex, 
                tokenizer, 
                SYSTEM_MESSAGE, 
                max_length=config.data.max_length),
            batched=True
        )
    if "test" in splits:
        processed_test_dataset = test_dataset.map(
            lambda ex: process_batch(
                ex, 
                tokenizer, 
                SYSTEM_MESSAGE, 
                max_length=config.data.max_length),
            batched=True
        )
   
    print("procesed dataset to parquet")
    if "train" in splits:
        processed_train_dataset.to_parquet(
            config.data.train_qwen_tokenized_parquet_path, 
        )
    if "validation" in splits:
        processed_val_dataset.to_parquet(
            config.data.validation_qwen_tokenized_parquet_path, 
        )
    if "test" in splits:
        processed_test_dataset.to_parquet(
            config.data.test_qwen_tokenized_parquet_path, 
        )

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
            {"role": "user", "content": prompt},
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


if __name__ == "__main__":
    main()