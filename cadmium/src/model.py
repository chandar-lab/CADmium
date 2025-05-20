import os, sys
import torch.nn as nn
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from layers.embedder import CADSequenceEmbedder, PositionalEncodingSinCos

MAX_CAD_SEQUENCE_LENGTH=272


class LLM4CAD(nn.Module):
    """
    LLM4CAD model
    """
    def __init__(self, model_config = {"device":"cuda",
                                "model_id":"meta-llama/Meta-Llama-3-8B"},
                emb_config = {"one_hot_size": 267,
                                "flag_size": 12,
                                "index_size": 11,
                                "cdim": 4096}):
        super().__init__()
        device = model_config["device"]
        model_id = model_config["model_id"]

        self.input_seq_emb = CADSequenceEmbedder(
            one_hot_size=emb_config["one_hot_size"], 
            flag_size=emb_config["flag_size"],
            index_size=emb_config["index_size"],
            d_model=emb_config['cdim'],
            device=device,
        )

        # self.output_seq_emb = CADSequenceEmbedder(
        #     one_hot_size=emb_config["one_hot_size"],  ## Check if arguments are correct
        #     flag_size=emb_config["flag_size"],
        #     index_size=emb_config["index_size"],
        #     d_model=262,
        #     device=device,
        # )

        self.out_mlp_x = nn.Linear(4096, emb_config['one_hot_size']) # 4096 -> 267
        self.out_mlp_y = nn.Linear(4096, emb_config['one_hot_size']) # 4096 -> 267

        self.pe = PositionalEncodingSinCos(
            embedding_size=emb_config['cdim'], max_seq_len=MAX_CAD_SEQUENCE_LENGTH, device=device ## define MAX_CAD_SEQUENCE_LENGTH
        )

        # self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.base_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                output_hidden_states=True, 
            )

    def forward(self, vec_dict, mask_cad_dict, prompt_inputs):

        """
        1. Embed the input sequence
        2. Add positional encoding
        3. Tokenize the prompt
        4. Concatenate tokenized prompt with the output CAD input embedding, and make the dimensions compatible
        5. Pass concatenated embeddings through the model
        6. Pass the output through the output sequence embedding
        7. Separate channel for text output from LLaMA
        8. Return the output
        """
        # prompt_inputs -- prompt_seq_length 
        ## CAD INPUT EMBEDDING
        input_seq_emb = self.input_seq_emb(vec_dict, mask_cad_dict["key_padding_mask"]) #cad_seq x 4096
        num_seq = vec_dict["cad_vec"].shape[1]
        input_seq_emb = self.pe(num_seq) + input_seq_emb #cad_seq x 4096
        hidden_size = self.base_model.config.hidden_size 
        with torch.no_grad():
            token_embeddings = self.base_model.get_input_embeddings()(prompt_inputs) #prompt_seq_length x 4096
        final_embedding = torch.cat([input_seq_emb, token_embeddings], dim=1) # (cad_seq + prompt_seq_length) x 4096 ;;; after padding, last 272 dimensions from prompt_seq_length should be removed
        
        with torch.no_grad():
            # CONCATENATE ATTENTION MASKS AND PASS AS ARGUMENT TO MODEL
            # bsize x seq_length x 120000
            outputs = self.base_model(inputs_embeds = final_embedding) # do we need attention mask?
        
        last_hidden_state = outputs.hidden_states[-1]
        output_seq_emb = self.output_seq_emb(last_hidden_state)
        
        return outputs, output_seq_emb
       

        