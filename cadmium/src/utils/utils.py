from prettytable import PrettyTable
import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
import json
import os
import re
from cadmium.utils.macro import CAD_CLASS_INFO, END_TOKEN


def count_parameters(model, description=True):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    if description:
        print(table)
    return total_params


def check_memory_usage(tensor):
    return tensor.element_size() * tensor.nelement() / 1024**2


def get_clones(module, num_layers=8):
    return nn.ModuleList([copy.deepcopy(module) for i in range(num_layers)])


def get_available_gpu_ids():
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices:
        gpu_ids = [int(gpu_id.strip()) for gpu_id in cuda_visible_devices.split(",")]
    else:
        gpu_ids = []  # Empty list means no GPUs available for training.

    return gpu_ids


def top_p_sampling(logits, top_p=0.9):
    logits_copy = (
        logits.clone()
    )  # Create a copy of the logits to avoid in-place modification

    sorted_logits, sorted_indices = torch.sort(logits_copy, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits_copy[..., indices_to_remove] = float("-inf")

    # Apply softmax and reshape back to original shape
    sampled_probs = F.softmax(logits_copy, dim=-1).reshape(-1, 1, 267)

    # Sample from the probability distribution
    sampled_indices = torch.multinomial(sampled_probs.view(1, -1), 1).view(1, 1, 1)
    return sampled_indices


def create_flag_vec(vec, prev_flag_vec):
    """
    Create flag vector given a CAD sequence and previous flag vectors. (Sketch First)

    Args:
        vec (torch.Tensor): The input tensor of shape (B, N, 2) representing CAD sketches.
        prev_flag_vec (torch.Tensor): The previous flag tensor of shape (B, N-1, 2).

    Returns:
        new_flag_vec (torch.Tensor): The updated flag tensor of shape (B, N, 2).
    """
    
    # Initialize a new flag tensor with zeros
    new_flag_vec = torch.zeros_like(prev_flag_vec[:, -1])

    # Determine positions of sketches
    sketch_pos = (prev_flag_vec[:, -1] == 0)

    # Check if there are at least 2 previous flag vectors
    if prev_flag_vec.shape[1] > 2:
        # Determine positions of extrude distance
        extrude_dist_pos = torch.logical_and(prev_flag_vec[:, -1] == 1, prev_flag_vec[:, -2] != 1)
        
        # Check if there are more than 11 tokens in the input vector (since first 11 tokens belong to extrusions)
        # Check if the previous token type indicates an end sketch
        prev_token_type = (vec[:, -2, 0] == END_TOKEN.index("END_SKETCH"))
        
        # Update flag values based on different conditions
        new_flag_vec[torch.logical_and(~sketch_pos, ~extrude_dist_pos)] = \
            (prev_flag_vec[:, -1][torch.logical_and(~sketch_pos, ~extrude_dist_pos)] + 1) % (CAD_CLASS_INFO['flag_size'] - 1)
        new_flag_vec[torch.logical_and(~sketch_pos, extrude_dist_pos)] = 1                      
        new_flag_vec[prev_token_type] = 1
        
        # Check if the current token type indicates a sequence start or end
        end_token_type = (vec[:, -1, 0] == END_TOKEN.index("START"))
        new_flag_vec[end_token_type] = 0
    else:
        pass

    # Identify positions of padding tokens and assign the corresponding flag value
    padding_pos = (vec[:, -1, 0] == END_TOKEN.index("PADDING"))
    new_flag_vec[padding_pos] = CAD_CLASS_INFO['flag_size'] - 1

    # Reshape the flag tensor
    return new_flag_vec.reshape(-1, 1)


def create_index_vec(vec, prev_index_vec):
    """
    Create index vector for the last token given previous tokens and previous index vectors. (Sketch First)

    Args:
        vec (torch.Tensor): The input tensor of shape (B, N, 2) representing CAD sketches.
        prev_index_vec (torch.Tensor): The previous index tensor of shape (B, N-1, 2).

    Returns:
        new_index_vec (torch.Tensor): The updated index tensor of shape (B, 1).
    """
    # Initialize a new index tensor with ones
    new_index_vec = torch.ones_like(prev_index_vec[:, -1]) * prev_index_vec[:, -1]

    # Determine positions where extrusion ends
    extrusion_end_pos = vec[:, -2, 0] == END_TOKEN.index("END_EXTRUSION")

    # Get the previous index values
    prev_index = prev_index_vec[:, -1]

    # Update index values for extrusion end positions
    new_index_vec[extrusion_end_pos] = torch.clip(
        prev_index + 1, min=0, max=CAD_CLASS_INFO["index_size"] - 1
    )[extrusion_end_pos]

    # Determine positions where sketch starts
    start_pos = vec[:, -1, 0] == END_TOKEN.index("START")

    # Update index values for sketch start positions
    new_index_vec[start_pos] = torch.clip(
        prev_index, min=0, max=CAD_CLASS_INFO["index_size"]
    )[start_pos]

    # Determine positions of padding tokens and those that were not padding previously
    padding_pos = vec[:, -1, 0] == END_TOKEN.index("PADDING")
    prev_not_padding_pos = vec[:, -2, 0] != END_TOKEN.index("PADDING")
    mask_new_pad = torch.logical_and(
        padding_pos, prev_not_padding_pos
    )  # For the first padding token
    mask_old_pad = torch.logical_and(
        padding_pos, ~prev_not_padding_pos
    )  # From the second padding token onwards

    # Update index values for new and old padding positions
    new_index_vec[mask_new_pad] = torch.clip(
        prev_index + 1, min=0, max=CAD_CLASS_INFO["index_size"] - 1
    )[mask_new_pad]
    new_index_vec[mask_old_pad] = torch.clip(
        prev_index, min=0, max=CAD_CLASS_INFO["index_size"] - 1
    )[mask_old_pad]

    # Reshape the index tensor
    return new_index_vec.reshape(-1, 1)

def get_messages(
    example, 
    tokenizer, 
    system_message, 
):
    # 1. Process the json_desc field: drop unwanted keys and convert to string.
    json_desc = example["json_desc"]
    if isinstance(json_desc, dict):
        json_desc.pop("final_name", None)
        json_desc.pop("final_shape", None)
        target_text = json.dumps(json_desc)
    else:
        target_text = example["json_desc"]
    
    prompt = example["prompt"]
    
    # 2. Build the conversation messages.
    #    Here we include the assistant message (i.e. the target text) directly.
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": target_text},
    ]
    return messages    


def process_sample(
    example, 
    tokenizer, 
    system_message, 
    max_length=2048, 
    ):
    """
    Processes one sample for autoregressive fine tuning in a chat-format.
    
    The final conversation is built as:
      [SYSTEM]   <system_message>
      [USER]     <prompt>
      [ASSISTANT] <target_text>
    
    We then tokenize the entire conversation (with special tokens included)
    in a single pass. Next, we search for the assistant marker token sequence
    and mask all tokens up to and including that marker (i.e. set them to -100)
    so that only the assistant's answer contributes to the loss.
    
    Args:
        example (dict): Contains "uid", "prompt", and "json_desc". The json_desc field
                        is originally a dictionary.
        tokenizer: A Hugging Face tokenizer that implements apply_chat_template.
        system_message (str): The system message text.
        max_total_length (int): Maximum length for the entire tokenized sequence.
    
    Returns:
        dict: A dictionary with keys "uid", "input_ids", "attention_mask", "labels".
    """
    
    # 1. Process the json_desc field: drop unwanted keys and convert to string.
    json_desc = example["json_desc"]
    prompt = example["prompt"]
    messages = get_messages(example, tokenizer, system_message)
    
    # 3. Tokenize the entire conversation in one pass.
    #    We assume that `apply_chat_template` in tokenize mode returns a dict with
    #    "input_ids" and "attention_mask" as lists.
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    full_tokens = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )
    
    input_ids = full_tokens["input_ids"]
    attention_mask = full_tokens["attention_mask"]
    
    # 4. Determine the start index of the assistant answer.
    #    We assume that the assistant marker is a known fixed string.
    #    For example, if the marker is "<|im_start|>assistant\n", then:
    marker_text = "<|im_start|>assistant\n"
    # Get the token ids corresponding to the marker.
    marker_ids = tokenizer(marker_text, add_special_tokens=False)

    idx = find_sublist(input_ids[0].tolist(), marker_ids['input_ids'])
    if idx == -1:
        # If we cannot find the marker, default to not masking any tokens.
        mask_boundary = 0
    else:
        # Mask up to (and including) the marker.
        mask_boundary = idx + len(marker_ids) + 1
    
    # 5. Create the labels: copy the input_ids and mask everything before mask_boundary.
    labels = input_ids.clone()
    labels[0, :mask_boundary] = -100

    # 6. Return the processed sample.
    return {
        "uid": example["uid"],
        "prompt": full_text,
        "input_ids": input_ids.squeeze(0).tolist(),
        "attention_mask": attention_mask.squeeze(0).tolist(),
        "labels": labels.squeeze(0).tolist(),
    }


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
        # If json_desc is a dict, remove unwanted keys and convert to string.
        if isinstance(json_desc, str):
            json_desc = json.loads(json_desc)
            # Copy the dict to avoid modifying the original.
            json_copy = dict(json_desc)
            json_copy.pop("final_name", None)
            json_copy.pop("final_shape", None)
            target_text = json.dumps(json_copy)
        else:
            target_text = json_desc
        target_texts.append(target_text)
    
    # 2. Build the conversation messages and full text for each sample.
    messages = []
    for prompt, target_text in zip(examples["prompt"], target_texts): ## annotation = prompt for Text2CAD data
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




# Helper function to search for a sublist in a list.
def find_sublist(sequence, sub):
    sub_len = len(sub)
    for i in range(len(sequence) - sub_len + 1):
        if sequence[i:i+sub_len] == sub:
            return i
    return -1