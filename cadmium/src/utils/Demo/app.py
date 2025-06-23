# source: https://github.com/SadilKhan/Text2CAD
import os, sys

sys.path.append("..")
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-1]))
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-2]))
from llm4cad.utils.CadSeqProc.cad_sequence import CADSequence
import gradio as gr
import yaml
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from types import SimpleNamespace

def load_model(config, device):
    # -------------------------------- Load Model -------------------------------- #

    print("Loading model...", config["test"]["checkpoint_path"])
    llm4cad = AutoModelForCausalLM.from_pretrained(
        config["test"]["checkpoint_path"],
        torch_dtype=torch.float16,  
        ).to(device)
    
    llm4cad.eval()
    return llm4cad

def format_prompt(user_prompt, system_message="You are a CAD designer"):
    """
    Formats the input prompt to match the model's expected structure.

    Args:
        user_prompt (str): The user's message.
        system_message (str, optional): The system message to guide the assistant.

    Returns:
        str: Formatted prompt.
    """
    # Start with system message if provided
    formatted_prompt = ""
    if system_message:
        formatted_prompt += f"<|im_start|>system\n{system_message}<|im_end|>\n"

    # Add user message
    formatted_prompt += f"<|im_start|>user\n{user_prompt}<|im_end|>\n"

    # Assistant marker for generation
    formatted_prompt += "<|im_start|>assistant\n"

    return formatted_prompt


def test_model(model, text, config, device):
    # if not isinstance(text, list):
    #     text = [text]
    text = format_prompt(text, SYSTEM_MESSAGE)
    with open("input_prompt.txt", "w") as f:
        f.write(text)
    inputs = tokenizer(text, return_tensors="pt", padding=False)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            max_new_tokens=4096,
            do_sample=False,
        )

    try:
        output_string = tokenizer.decode(outputs[0], skip_special_tokens=True).split('assistant\n')[1]
        pred_cad = eval(output_string)
        with open('sample_output.json', 'w') as f:
            json.dump(pred_cad, f, indent=4)
    except Exception as e:
        with open("error_log.txt", "a") as f:
            f.write(f"Error decoding output: {e}\n")
            f.write(f"Output string: {output_string}\n")
        assert(False), f"Error in decoding the output, {e}"


    pred_cad = CADSequence.minimal_json_to_seq(pred_cad, bit = 8)
    pred_cad.create_mesh()

    return pred_cad.mesh, pred_cad
    # except Exception as e:
    #     return None

def parse_config_file(config_file):
    with open(config_file, "r") as file:
        yaml_data = yaml.safe_load(file)
    return yaml_data


SYSTEM_MESSAGE = """Generate CAD model JSON EXACTLY matching this schema:\n { "parts": { "part_1": { // Always use sequential part_1, part_2... even if some are null "coordinate_system": { "Euler Angles": [0.0, 0.0, 0.0], // XYZ rotation angles in degrees "Translation Vector": [0.0, 0.0, 0.0] // X,Y,Z position offsets }, "description": { "height": 0.0, // Total vertical dimension "length": 0.0, // Total horizontal dimension "name": "", // (optional) Component identifier "shape": "", // (optional) Basic geometric classification "width": 0.0 // Total depth dimension }, "extrusion": { "extrude_depth_opposite_normal": 0.0, // Negative direction extrusion "extrude_depth_towards_normal": 0.0, // Positive direction extrusion "operation": "NewBodyFeatureOperation", // One of: NewBodyFeatureOperation, JoinFeatureOperation, CutFeatureOperation, IntersectFeatureOperation "sketch_scale": 0.0 // Scaling factor for sketch geometry }, "sketch": { "face_1": { // Use sequential face_1, face_2... (null if unused) "loop_1": { // Use sequential loop_1, loop_2... (null if unused) "circle_1": { // Use sequential circle_1, circle_2... "Center": [0.0, 0.0], // X,Y coordinates "Radius": 0.0 }, "arc_1": { // Use sequential arc_1, arc_2... "Start Point": [0.0, 0.0], "End Point": [0.0, 0.0], "Mid Point": [0.0, 0.0] }, "line_1": { // Use sequential line_1, line_2... "Start Point": [0.0, 0.0], "End Point": [0.0, 0.0] } // ... (other geometric elements as null/none) } // ... (other loops as null/none) } // ... (other faces as null/none) } }, "part_2": null, // Maintain sequential numbering even for null parts // ... (additional parts) } } \n\nSTRICT RULES:\n- OUTPUT ONLY RAW JSON (no formatting/text/comments/explanations)\n- NEVER COPY INSTRUCTIONAL TEXT FROM JSON SCHEMA EXAMPLES\n- ALL numbers as floats (0.0 not 0)\n- ALLOWED OPERATIONS: NewBodyFeatureOperation/JoinFeatureOperation/CutFeatureOperation/IntersectFeatureOperation \n- GEOMETRY REQUIREMENTS (these are the only available primitives):\n  • Circles: Center[X,Y] + Radius\n  • Arcs: Start[X,Y] + End[X,Y] + Mid[X,Y]\n  • Lines: Start[X,Y] + End[X,Y]\n- ENFORCE part_1, part_2... sequence (include nulls)\n- NO NEW FIELDS\n"""
config_path = "../../config/inference_user_input.yaml"
config = parse_config_file(config_path)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = load_model(config, device)
tokenizer = AutoTokenizer.from_pretrained(config["test"]["checkpoint_path"], padding_side='left')
OUTPUT_DIR="output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def genrate_cad_model_from_text(text):
    global model, config, tokenizer
    mesh,*extra = test_model(model=model, text=text, config=config, device=device)
    if mesh is not None:
        output_path = os.path.join(OUTPUT_DIR, "output.stl")
        mesh.export(output_path)
        return output_path
    else:
        raise Exception("Error generating CAD model from text")


examples = [
    "A ring.",
    "A rectangular prism.",
    "A 3D star shape with 5 points.",
    "The CAD model features a cylindrical object with a cylindrical hole in the center.",
    "The CAD model features a rectangular metal plate with four holes along its length."
]


title = "CADmium - Fine-Tuning Code Language Models for Text-Driven Sequential CAD Design"
description = """
Fine-tuning code LLMs for the text-to-CAD problem.

<div style="display: flex; justify-content: center; gap: 10px; align-items: center;">

<a href="https://arxiv.org/abs/2409.17106">
  <img src="https://img.shields.io/badge/Arxiv-3498db?style=for-the-badge&logoWidth=40&logoColor=white&labelColor=2c3e50&borderRadius=10" alt="Arxiv (Coming soon)" />
</a>
<a href="https://sadilkhan.github.io/text2cad-project/">
  <img src="https://img.shields.io/badge/Project-2ecc71?style=for-the-badge&logoWidth=40&logoColor=white&labelColor=27ae60&borderRadius=10" alt="Project (Coming soon)" />
</a>

</div>
"""

# Create the Gradio interface
demo = gr.Interface(
    fn=genrate_cad_model_from_text,
    inputs=gr.Textbox(label="Text", placeholder="Enter a text prompt here"),
    outputs=gr.Model3D(clear_color=[0.678, 0.847, 0.902, 1.0], label="3D CAD Model"),
    examples=examples,
    title=title,
    description=description,
    theme=gr.themes.Soft(), 
)

if __name__ == "__main__":
    demo.launch(share=True)
