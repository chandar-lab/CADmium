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
from models.loss import CELoss
from cadmium.utils.logger import CLGLogger
from cadmium.utils.utils import process_sample, process_batch, find_sublist
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

from cadmium.codes.dataprep.t2c_dataset import Text2CADJSON_Dataset
import random
from models.metrics import AccuracyCalculator
from loguru import logger
import gc
from cadmium.utils.evaluate import switch_system_message, evaluate_model, PredictionsLoader
# from cadmium.utils.save_result_jsons import process_results
from cadmium.utils.prompts import SYSTEM_MESSAGE, SYSTEM_MESSAGES
from cadmium.utils.macro import (END_TOKEN, 
                                MAX_CAD_SEQUENCE_LENGTH, 
                                CAD_CLASS_INFO, 
                                )
        

@hydra.main(version_base=None, config_path="../config/", config_name="config_json_eval")
def main(config: DictConfig):
    print("Entered the main")
    os.makedirs(config.eval.output_dir, exist_ok=True)
    config_to_save = OmegaConf.to_container(config, resolve=True)
    with open(os.path.join(config.eval.output_dir, 'config.json'), 'w') as f:
        json.dump(config_to_save, f, indent=4)
    print("\nCONFIG:")
    print(json.dumps(config_to_save))
    print("\n")

    # ----------------- Distributed init -----------------

    # Determine the local rank (default to 0 if not set)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=60*60))

    # ----------------- Loading the model -----------------
    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_checkpoint, 
        torch_dtype=torch.float16,  
        device_map=f"cuda:{local_rank}",  
        )
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_checkpoint, padding_side='left')

    # ----------------- Loading and Preprocessing the dataset -----------------
    data_path = config.data.qwen_tokenized_parquet_path
    ds = load_dataset(
        'parquet', 
        data_files={
            'ds':data_path, 
        })
    ds = ds['ds']
    # ----------------- Eval hyperparams -----------------

    if config.data.max_n_datapoints:
        ds = Dataset.from_dict(
            ds.shuffle(seed=config.seed)[:config.data.max_n_datapoints])

    if hasattr(config, 'prompt'):
        prompt = SYSTEM_MESSAGES[config.prompt]
        print(F"PROMPT: \n{prompt}")
        ds = switch_system_message(
            ds, 
            prompt, 
            tokenizer
            )

    res = evaluate_model(ds, tokenizer, model, config, output_dir=config.eval.output_dir, save_per_batch=config.eval.save_per_batch)
    if config.eval.save_per_batch:
        if dist.is_initialized():
            dist.barrier()
        if local_rank == 0 or not dist.is_initialized():
            loader = PredictionsLoader(config.eval.output_dir)
            res = loader.load_parallel(max_workers=8) 
            res.to_csv(os.path.join(config.eval.output_dir, 'results.csv'))
    else:
        res = pd.DataFrame(res)
        res.to_csv(os.path.join(config.eval.output_dir, f'results_rank{local_rank}.csv'))

    # Save JSONS for compute metrics
    #process_results(config.eval.output_dir)



if __name__ == "__main__":
    main()