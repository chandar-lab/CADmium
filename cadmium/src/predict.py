from copy import copy
import hashlib
import json
import os
import wandb
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
from datasets import Dataset
import pandas as pd
import datetime
import torch.distributed as dist
from cadmium.src.utils.evaluate import evaluate_model, PredictionsLoader

@hydra.main(version_base=None, config_path="../config/", config_name="predict")
def main(config: DictConfig):
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
    print(f"[PID {os.getpid():5d}] PRE‐DIST  RANK={os.environ.get('RANK', 0)}  LOCAL_RANK={local_rank}  WORLD_SIZE={os.environ.get('WORLD_SIZE', 1)}  -> cuda:{torch.cuda.current_device()}", flush=True)

    # ----------------- Loading the model -----------------
    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_checkpoint, 
        torch_dtype=torch.float16,  
        ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_checkpoint, padding_side='left')

    print("Model device:", model.device, flush=True)

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