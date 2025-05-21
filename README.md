# CADmium: Fine-Tuning Code Language Models for Text-Driven Sequential CAD Design

![CADmium Main Figure](./images/main_figure.png)


This repository implements the paper "*CADmium: Fine-Tuning Code Language Models for Text-Driven Sequential CAD Design*".

## Quick Setup

Ensure Anaconda (or Miniconda) is installed. From the project root directory, run the following to create the environment and install dependencies:

```bash
conda deactivate
conda create --prefix=venv python=3.11 -y
conda activate venv 
# Note: If 'conda activate venv' doesn't target the local './venv' directory, 
# you might need to use 'conda activate ./venv' instead.
conda install -c conda-forge pythonocc-core -y
pip install -r requirements.txt
pip install -e .
```

This sequence sets up a local Conda environment in the `venv` subdirectory, activates it, and installs all required packages, including the project itself in editable mode.

## Tokenizing the Dataset
To process and tokenize the dataset:

1.  **Prerequisites:**
    * Successful environment setup (see above).
    * Raw dataset (e.g., `cadmium_ds`) located in `data/cadmium_ds`.

2.  **Run the tokenization script** from the project root directory:
    ```bash
    python cadmium/src/tokenize_dataset.py
    ```

3.  **Output:**
    The script saves tokenized data into three Parquet files (train, validation, test splits) in the `data/` folder (e.g., `data/train_json_qwen_tokenized.parquet`).

## Training the Model

To train the model with the default configuration:

```bash
torchrun --nproc_per_node=4 cadmium/src/train.py --config-name train
```

The training configuration is defined in `cadmium/config/train.yaml`. By default, this setup is optimized for **4 GPUs** using Fully Sharded Data Parallel (FSDP). 

### Adjusting for Fewer GPUs:
1. **Remove FSDP:** Comment out or remove the `fsdp_config` block in the YAML file
2. **Maintain Effective Batch Size:** The original configuration uses `per_device_batch_size=4` with 4 GPUs (total batch size=16). To replicate this on fewer devices:
   - Single GPU: Set `per_device_batch_size=16` *or* increase `gradient_accumulation_steps=4` (with `per_device_batch_size=4`)
   - Intermediate GPUs: Adjust proportionally (e.g., 2 GPUs → `per_device_batch_size=8` or `gradient_accumulation_steps=2`)

## License

Licensed under the MIT License.