import json
import os
import torch
import wandb
import hydra
import omegaconf
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed, TrainingArguments, Trainer
from transformers.trainer_utils import IntervalStrategy, get_last_checkpoint
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset, Dataset
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from cadmium.utils.customtrainer import OrderedSFTTrainer
from cadmium.utils.warmstabledecay import WarmStableDecayScheduler, WarmStableDecayCallback

@hydra.main(version_base=None, config_path="../config", config_name="config_json_train")
def main(config: DictConfig):
    set_seed(config.seed)

    # ----------------- Save the config -----------------
    os.makedirs(config.training.output_dir, exist_ok=True)
    config_to_save = OmegaConf.to_container(config, resolve=True)
    with open(os.path.join(config.training.output_dir, 'config.json'), 'w') as f:
        json.dump(config_to_save, f, indent=4)
    print("Config File:", config, flush=True)

    # ----------------- Distributed training init -----------------
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    rank       = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # 2) pin each process to its GPU
    torch.cuda.set_device(local_rank)

    # 3) print *before* you init, so you can see exactly what torchrun gave you
    print(f"[PID {os.getpid():5d}] PRE‐DIST  RANK={rank}  LOCAL_RANK={local_rank}  WORLD_SIZE={world_size}  -> cuda:{torch.cuda.current_device()}", flush=True)

    # ----------------- Wandb init -----------------

    # Only initialize wandb in the main process
    if local_rank == 0:
        wandb.init(
            project="cadmium",
            id=config.wandb.runid,    # Ensure this matches your resumed run if needed
            name=config.wandb.name,
            resume="allow",  # or "must" if you want an error when the run doesn't exist
        )

    # ----------------- Loading the model -----------------
    
    using_fsdp = hasattr(config.training, 'fsdp') and config.training.fsdp is not None
    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_name, 
        device_map="cpu", # None if using_fsdp else {"": f"cuda:{local_rank}"},
        **({"quantization_config": BitsAndBytesConfig(**config.model.quantize)} if hasattr(config.model, 'quantize') else {"torch_dtype": torch.float16})
        )
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)

    # ----------------- Loading the tokenized dataset -----------------
    train_ds = load_dataset(
        'parquet', 
        data_files={
            'train':config.data.train_qwen_tokenized_parquet_path, 
        })
    val_ds = load_dataset(
        'parquet', 
        data_files={
            'validation':config.data.validation_qwen_tokenized_parquet_path, 
        })
    processed_train_dataset, processed_val_dataset = train_ds['train'], val_ds['validation']
    
    if config.data.shuffle:
        processed_train_dataset = processed_train_dataset.shuffle(seed=config.seed)
        processed_val_dataset = processed_val_dataset.shuffle(seed=config.seed)

    # ----------------- Eventually reduce val set -----------------
    if config.data.max_n_datapoints_val:
        processed_val_dataset = Dataset.from_dict(
            processed_val_dataset.shuffle(seed=config.seed)[:config.data.max_n_datapoints_val])

    # ----------------- Defining training hyperparams -----------------
    if hasattr(config.model, 'lora'):
        if isinstance(config.model.lora.target_modules, omegaconf.listconfig.ListConfig):
            config.model.lora.target_modules = list(config.model.lora.target_modules)
        # Define LoRA configuration
        peft_config = LoraConfig(
            r=config.model.lora.r,
            lora_alpha=config.model.lora.lora_alpha,
            target_modules=config.model.lora.target_modules,
            lora_dropout=config.model.lora.lora_dropout,
            bias=config.model.lora.bias,
            task_type=config.model.lora.task_type,
        )
    else:
        peft_config=None

    training_args = TrainingArguments(
        **config.training, 
        # logging_dir=os.path.join(config.training.output_dir, "hf-logs"),
        # log_level="info",
        # log_level_replica="debug",
    )

    print("PEFT CONFIG:", peft_config, flush=True)
    trainer_class = SFTTrainer if config.data.shuffle else OrderedSFTTrainer
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=processed_train_dataset,
        eval_dataset=processed_val_dataset,
        peft_config=peft_config,
    )

    # 1) Detect last warm checkpoint by inverting the cold checkpoint
    output_dir     = config.training.output_dir
    last_checkpoint    = get_last_checkpoint(output_dir)  # e.g. “…/checkpoint-40”
    if last_checkpoint:
        print(f"Last checkpoint found: {last_checkpoint}", flush=True)
    else:
        last_checkpoint = None
        print("No checkpoint found, starting from scratch.", flush=True)
    
    trainer.train(resume_from_checkpoint=last_checkpoint)



if __name__ == "__main__":
    main()