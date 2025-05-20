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
from cadmium.src.utils.logger import CLGLogger
from torch.utils.tensorboard import SummaryWriter
import torch
from tqdm import tqdm
import numpy as np
import datetime
from cadmium.src.utils.evaluate import switch_system_message, evaluate_model
from cadmium.src.utils.prompts import SYSTEM_MESSAGE
from datasets import Dataset
from huggingface_hub import login

# from peft import LoraConfig

from cadmium.src.dataprep.t2c_dataset import Text2CADJSON_Dataset
import random
from loguru import logger
from cadmium.src.utils.prompts import SYSTEM_MESSAGE
from cadmium.src.utils.macro import (END_TOKEN, 
                                MAX_CAD_SEQUENCE_LENGTH, 
                                CAD_CLASS_INFO, 
                                )


# No module named models
# src -> src
# cadmium.src.utils -> cadmium.src.utils