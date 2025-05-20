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
from cadmium.utils.utils import process_sample, process_batch
from torch.utils.tensorboard import SummaryWriter
import torch
from tqdm import tqdm
import numpy as np
import datetime
from cadmium.utils.evaluate import switch_system_message, evaluate_model
from cadmium.utils.prompts import SYSTEM_MESSAGE, SYSTEM_MESSAGES


# from peft import LoraConfig

from cadmium.codes.dataprep.t2c_dataset import Text2CADJSON_Dataset
# from transformers import AdamW, get_linear_schedule_with_warmup
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

    splits = ["train", "validation", "test"]

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
    
    if "validation" in splits:
        print("Loading Val Dataset")
        val_dataset = Text2CADJSON_Dataset(
            prompt_path=config.data.prompt_path,
            json_desc_dir=config.data.json_desc_dir,
            split_filepath=config.data.split_filepath,
            subset="validation",
            max_workers=32,
            #max_n_datapoints=config.data.max_n_datapoints_val,
            verbose=False,
        )

    if "test" in splits:
        print("Loading Test Dataset")
        test_dataset = Text2CADJSON_Dataset(
            prompt_path=config.data.prompt_path,
            json_desc_dir=config.data.json_desc_dir,
            split_filepath=config.data.split_filepath,
            subset="test",
            max_workers=32,
            #max_n_datapoints=config.data.max_n_datapoints_test,
            verbose=False,
        )

    print("Dataset to HF")
    if "train" in splits:
        train_dataset = train_dataset.to_hf_dataset()
    if "validation" in splits:
        val_dataset = val_dataset.to_hf_dataset()
    if "test" in splits:
        test_dataset = test_dataset.to_hf_dataset()


    print("Dataset to parquet")
    # if "train" in splits:
    #     train_dataset.to_parquet(
    #         config.data.train_parquet_path  
    #     )
    # if "validation" in splits:
    #     val_dataset.to_parquet(
    #         config.data.validation_parquet_path, 
    #     )
    # if "test" in splits:
    #     test_dataset.to_parquet(
    #         config.data.test_parquet_path, 
    #     )

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

if __name__ == "__main__":
    main()