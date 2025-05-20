import math
import matplotlib.pyplot
import numpy as np
from typing import List

import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import yaml
import warnings
import numpy as np
import math
from pathlib import Path
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.StlAPI import StlAPI_Writer, StlAPI_Reader
# from OCC.Display.SimpleGui import init_display
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Dir, gp_XYZ, gp_Ax3, gp_Trsf, gp_Pln
# from collections import OrderedDict
from torch.optim.lr_scheduler import LambdaLR
import json
import torch
from .macro import *
import pickle
import random
import trimesh
from plyfile import PlyData, PlyElement
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Builder, TopoDS_Compound
import joblib
from scipy.spatial import cKDTree
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut, BRepAlgoAPI_Common

warnings.filterwarnings("ignore")
import contextlib
from rich import print
import datetime
from hashlib import sha256

# from vec2cad import Vec2CAD
import copy
import concurrent.futures
import matplotlib.pyplot as plt
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeWire, BRepBuilderAPI_Transform
from OCC.Core.AIS import AIS_Shape
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
import torch.nn.functional as F
# from .decorator import measure_performance
from OCC.Core.BRepTools import breptools_Write




def find_response(prompt_ZE, emb_dir, response_dir):
    response_name=None
    all_responses=[os.path.join(emb_dir,f) for f in os.listdir(emb_dir) if f.endswith(".npz")]

    for response in all_responses:
        response_data=np.load(response,allow_pickle=True)['embedding']

        if np.allclose(response_data,prompt_ZE.cpu().numpy()[:response_data.shape[0]]):
            response_name= response.split("/")[-1].split(".")[0]
            break
    if response_name is not None:
        level="_".join(response_name.split("_")[-3:-1])
    response_path=os.path.join(response_dir, response_name+".json")
    
    with open(response_path,"r") as f:
        data=json.load(f)
    
    return data, level




def file_size(file):
    byte_size=sys.getsizeof(file)
    if byte_size<1024:
        return str(byte_size)+" bytes"
    elif byte_size<1024**2:
        return str(byte_size/1024)+" KB"
    elif byte_size<1024**3:
        return str(byte_size/1024**2)+" MB"
    else:
        return str(byte_size/1024**3)+" GB"

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


def choose_best_index(chamfer_distances, eps=1):
    min_cd = np.min(chamfer_distances)
    argmin_cd = np.argmin(chamfer_distances)
    sorted_indices = np.argsort(chamfer_distances)
    if np.sum(chamfer_distances) == -len(chamfer_distances):
        return 0  # All of them are invalid

    elif chamfer_distances[0] < eps and chamfer_distances[0] >= 0:
        return 0
    else:  # Check if it is invalid, otherwise return the next best
        for i in sorted_indices:
            if chamfer_distances[i] >= 0:
                return i
    return 0


def write_ply_with_binary_values(points, binary_value, filename, text=False):
    """input: Nx3, write points to filename as PLY format."""
    points = [
        (points[i, 0], points[i, 1], points[i, 2], binary_value[i])
        for i in range(points.shape[0])
    ]
    vertex = np.array(
        points, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("binary_value", "u1")]
    )
    el = PlyElement.describe(vertex, "vertex", comments=["vertices"])
    with open(filename, mode="wb") as f:
        PlyData([el], text=text).write(f)


def unique_preserve_order(tensor, dim=0):
    # Get unique values and indices
    unique_vals, indices = torch.unique(tensor, dim=dim, return_inverse=True)
    sorted_indices = []
    for i in indices:
        if i not in sorted_indices:
            sorted_indices.append(i.item())

    # Rearrange unique values according to sorted indices
    unique_sorted = unique_vals[sorted_indices]

    return unique_sorted


def clear_folder(folder_path):
    # Check if the folder exists
    if os.path.exists(folder_path):
        # Get a list of all the files in the folder
        files = os.listdir(folder_path)

        # Iterate through the files and remove them
        for file_name in files:
            file_path = os.path.join(folder_path, file_name)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                else:
                    print(f"Skipping {file_path} as it is a directory")
            except Exception as e:
                print(f"Error while deleting {file_path}: {e}")

        print(f"All files in {folder_path} have been removed.")
    else:
        print(f"The folder {folder_path} does not exist.")


def save_ais_shape_as_brep(ais_shape, filename):
    # Create a compound shape to hold the AIS_Shape
    compound = TopoDS_Compound()
    builder = TopoDS_Builder()
    builder.MakeCompound(compound)

    # Add the AIS_Shape to the compound
    builder.Add(compound, ais_shape.Shape())

    # Transform the compound to BRep
    brep_builder = BRepBuilderAPI_Transform(compound, ais_shape.Trsf())
    brep = brep_builder.Shape()

    # Save the BREP file
    breptools_Write(brep, filename)


def save_ais_shape_as_step(ais_shape, filename):
    breptools_Write(ais_shape.Shape(), filename)


def create_colored_wire(edges, apply_colors=True):
    # Create a wire
    wire_builder = BRepBuilderAPI_MakeWire()
    for edge in edges:
        wire_builder.Add(edge)
    wire = wire_builder.Wire()

    # Define 10 different colors
    colors = [
        Quantity_Color(1, 0, 0, Quantity_TOC_RGB),  # Red
        Quantity_Color(0, 1, 0, Quantity_TOC_RGB),  # Green
        Quantity_Color(0, 0, 1, Quantity_TOC_RGB),  # Blue
    ]

    # Create AIS object and assign colors
    wire_ais = AIS_Shape(wire)
    if apply_colors:
        wire_ais.SetColor(colors[0])

    return wire_ais


def get_file_size_mb(file_path):
    """
    Get the size of a file in megabytes (MB).

    Args:
        file_path (str): Path to the file.

    Returns:
        float: Size of the file in MB.
    """
    # Get the size in bytes
    size_bytes = os.path.getsize(file_path)

    # Convert to megabytes (MB)
    size_mb = size_bytes / (1024 * 1024)

    return size_mb


def calc_mean_precision_recall_f1_score(report_df, round=None):
    """
    Calculate mean precision, recall, and F1-score from a report DataFrame.

    Args:
        report_df (pd.DataFrame): DataFrame containing ground truth and prediction information.

    Returns:
        tuple: A tuple containing mean precision, mean recall, and mean F1-score.
    """

    # Calculate the sums of ground truth occurrences for different shapes
    line_total_gt_sum = np.sum(report_df["line_total_gt"] > 0)
    arc_total_gt_sum = np.sum(report_df["arc_total_gt"] > 0)
    circle_total_gt_sum = np.sum(report_df["circle_total_gt"] > 0)

    # Calculate the total sum of ground truth occurrences
    total_sum = line_total_gt_sum + arc_total_gt_sum + circle_total_gt_sum

    # Calculate mean precision
    mean_precision = (
        report_df["line_precision"]
        + report_df["arc_precision"]
        + report_df["circle_precision"]
    ).values[0] / total_sum

    # Calculate mean recall
    mean_recall = (
        report_df["line_recall"] + report_df["arc_recall"] + report_df["circle_recall"]
    ).values[0] / total_sum

    # Calculate mean F1-score
    mean_f1 = (
        report_df["line_f1"] + report_df["arc_f1"] + report_df["circle_f1"]
    ).values[0] / total_sum
    if round is not None:
        return (
            mean_precision.round(round),
            mean_recall.round(round),
            mean_f1.round(round),
        )
    else:
        return mean_precision, mean_recall, mean_f1


def create_matched_pair(list1, list2, row_indices, col_indices):
    """
    Creates a list of matched pairs based on the row and column indices.

    Args:
        list1 (list): The first list of elements.
        list2 (list): The second list of elements.
        row_indices (list): List of row indices based on Hungarian Matching
        col_indices (list): List of row indices based on Hungarian Matching

    Returns:
        list: List of matched pairs, where each pair is a list containing an element from list1 and an element from list2.
    """
    assert len(list1) == len(list2)
    assert len(row_indices) == len(col_indices)

    matched_pair = []
    for i in range(len(row_indices)):
        matched_pair.append([list1[row_indices[i]], list2[col_indices[i]]])

    return matched_pair


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0, 0], boxB[0, 0])
    yA = max(boxA[0, 1], boxB[0, 1])
    xB = min(boxA[1, 0], boxB[1, 0])
    yB = min(boxA[1, 1], boxB[1, 1])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[1, 0] - boxA[0, 0] + 1) * (boxA[1, 1] - boxA[0, 1] + 1)
    boxBArea = (boxB[1, 0] - boxB[0, 0] + 1) * (boxB[1, 1] - boxB[0, 1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def normalize_pc(points):
    scale = np.abs(np.max(points) - np.min(points))
    points = points / scale
    return points


def write_ply_colors(point_cloud, mask=None, filename=None):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    if mask is not None:
        colors = np.zeros((point_cloud.shape[0], 3))
        colors[mask == 0] = [0, 0, 0]  # Black for False
        colors[mask == 1] = [255, 0, 0]  # Red for True
        colors = colors.astype(np.uint8)
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Normalize to [0, 1]

    o3d.io.write_point_cloud(filename, pcd)


def get_last_extrusion(cad_vec, flag_vec, index_vec):
    """
    Get the information about the last extrusion in the CAD vector sequence.

    Args:
        cad_vec (torch.Tensor): CAD vector sequence of shape (batch_size, sequence_length, dimension).
        flag_vec (torch.Tensor): Flag vector indicating extrusion events of shape (batch_size, sequence_length).
        index_vec (torch.Tensor): Index vector indicating the extrusion index of shape (batch_size, sequence_length).

    Returns:
        last_ext (torch.Tensor): Information about the last extrusion of shape (batch_size, sequence_length, features).
    """
    # Create a mask to identify extrusion events
    ext_mask = torch.logical_and(flag_vec >= 1, flag_vec < 11).unsqueeze(dim=-1)

    # Apply the mask to the CAD vector sequence
    cad_seq_ext = cad_vec * ext_mask

    # Create a mask to identify the non-padding elements in the CAD vector
    non_padding_mask = cad_vec[:, :, 0] != 0

    # Find the maximum extrusion index for each sample
    max_ext_index = torch.max(index_vec * non_padding_mask, axis=1).values

    # Create a mask for the maximum extrusion index
    max_ext_mask = (index_vec == max_ext_index.unsqueeze(dim=1)).unsqueeze(dim=-1)

    # Apply the mask for the last extrusion
    last_ext = cad_seq_ext * max_ext_mask  # (B,N,2)

    batch_size = last_ext.shape[0]

    last_ext_list = []  # List of tensors containing last extrusion operations

    for b in range(batch_size):
        last_ext_list.append(
            last_ext[b][last_ext[b, :, 0] > END_TOKEN.index("END_EXTRUSION")]
        )

    return last_ext_list


def find_token(arr, val=4, remove_less=True):
    """

    Finds the position of arrays by spliting on the val

    """
    if remove_less:
        arr = arr[np.unique(np.where(arr >= val)[0])]
    indices = np.unique(np.where(arr == val)[0])
    return indices


def get_sketch_extrusion_pos(vec, ext_token=6, sketch_token=2, remove_less=True):
    skt_ext_pos_list = split_array_pos(vec, ext_token, False)
    skt_pos_list = []
    ext_pos_list = []
    for i in range(len(skt_ext_pos_list)):
        skt_ext_pos = skt_ext_pos_list[i]
        skt_pos = find_token(vec[skt_ext_pos[0] : skt_ext_pos[1]], sketch_token, False)[
            0
        ]
        skt_pos_list.append(
            [skt_ext_pos_list[i][0], skt_pos + skt_ext_pos_list[i][0] + 1]
        )
        ext_pos_list.append(
            [skt_pos + 1 + skt_ext_pos_list[i][0], skt_ext_pos_list[i][1] + 1]
        )

    return skt_pos_list, ext_pos_list


# def create_index_vec(vec, prev_index_vec):
#     """
#     Create index vector for the last token given previous tokens and previous index vectors. (Extrusion First)

#     Args:
#         vec (torch.Tensor): The input tensor of shape (B, N, 2) representing CAD sketches.
#         prev_index_vec (torch.Tensor): The previous index tensor of shape (B, N-1, 2).

#     Returns:
#         new_index_vec (torch.Tensor): The updated index tensor of shape (B, 1).
#     """
#     # Initialize a new index tensor with ones
#     new_index_vec = torch.ones_like(prev_index_vec[:, -1]) * prev_index_vec[:, -1]

#     # Determine positions where extrusion ends
#     extrusion_end_pos = vec[:, -2, 0] == END_TOKEN.index("END_SKETCH")

#     # Get the previous index values
#     prev_index = prev_index_vec[:, -1]

#     # Update index values for extrusion end positions
#     new_index_vec[extrusion_end_pos] = torch.clip(
#         prev_index + 1, min=0, max=CAD_CLASS_INFO["index_size"] - 1
#     )[extrusion_end_pos]

#     # Determine positions where sketch starts
#     start_pos = vec[:, -1, 0] == END_TOKEN.index("START")

#     # Update index values for sketch start positions
#     new_index_vec[start_pos] = torch.clip(
#         prev_index, min=0, max=CAD_CLASS_INFO["index_size"]
#     )[start_pos]

#     # Determine positions of padding tokens and those that were not padding previously
#     padding_pos = vec[:, -1, 0] == END_TOKEN.index("PADDING")
#     prev_not_padding_pos = vec[:, -2, 0] != END_TOKEN.index("PADDING")
#     mask_new_pad = torch.logical_and(
#         padding_pos, prev_not_padding_pos
#     )  # For the first padding token
#     mask_old_pad = torch.logical_and(
#         padding_pos, ~prev_not_padding_pos
#     )  # From the second padding token onwards

#     # Update index values for new and old padding positions
#     new_index_vec[mask_new_pad] = torch.clip(
#         prev_index + 1, min=0, max=CAD_CLASS_INFO["index_size"] - 1
#     )[mask_new_pad]
#     new_index_vec[mask_old_pad] = torch.clip(
#         prev_index, min=0, max=CAD_CLASS_INFO["index_size"] - 1
#     )[mask_old_pad]

#     # Reshape the index tensor
#     return new_index_vec.reshape(-1, 1)





# def create_flag_vec(vec, prev_flag_vec):
#     """
#     Create flag vector given a CAD sequence and previous flag vectors. (Extrusion First)

#     Args:
#         vec (torch.Tensor): The input tensor of shape (B, N, 2) representing CAD sketches.
#         prev_flag_vec (torch.Tensor): The previous flag tensor of shape (B, N-1, 2).

#     Returns:
#         new_flag_vec (torch.Tensor): The updated flag tensor of shape (B, N, 2).
#     """

#     # Initialize a new flag tensor with zeros
#     new_flag_vec = torch.zeros_like(prev_flag_vec[:, -1])

#     # Determine positions of sketches
#     sketch_pos = prev_flag_vec[:, -1] == 0

#     # Check if there are at least 3 previous flag vectors
#     if prev_flag_vec.shape[1] > 2:
#         # Determine positions of extrude distance
#         extrude_dist_pos = torch.logical_and(
#             prev_flag_vec[:, -1] == 1, prev_flag_vec[:, -2] != 1
#         )

#         # Check if there are more than 11 tokens in the input vector (since first 11 tokens belong to extrusions)
#         if vec.shape[1] > 11:
#             # Check if the previous token type indicates an end sketch
#             prev_token_type = vec[:, -2, 0] == END_TOKEN.index("END_SKETCH")

#             # Update flag values based on different conditions
#             new_flag_vec[torch.logical_and(~sketch_pos, ~extrude_dist_pos)] = (
#                 prev_flag_vec[:, -1][torch.logical_and(~sketch_pos, ~extrude_dist_pos)]
#                 + 1
#             ) % (CAD_CLASS_INFO["flag_size"] - 1)
#             new_flag_vec[torch.logical_and(~sketch_pos, extrude_dist_pos)] = 1
#             new_flag_vec[prev_token_type] = 1

#             # Check if the current token type indicates a sequence start or end
#             end_token_type = vec[:, -1, 0] == END_TOKEN.index("START")
#             new_flag_vec[end_token_type] = 0
#         else:
#             # Update flag values for extrude distances and sketches
#             new_flag_vec[~extrude_dist_pos] += (
#                 prev_flag_vec[:, -1][~extrude_dist_pos] + 1
#             ) % (CAD_CLASS_INFO["flag_size"] - 1)
#             new_flag_vec[extrude_dist_pos] = 1
#     else:
#         # If there are fewer than 3 previous flag vectors, simply increment the flag
#         new_flag_vec += 1

#     # Identify positions of padding tokens and assign the corresponding flag value
#     padding_pos = vec[:, -1, 0] == END_TOKEN.index("PADDING")
#     new_flag_vec[padding_pos] = CAD_CLASS_INFO["flag_size"] - 1

#     # Reshape the flag tensor
#     return new_flag_vec.reshape(-1, 1)

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


def make_unique_dict(data):
    # Get unique elements and their counts
    unique_elements, counts = np.unique(data, return_counts=True)

    # Create a dictionary from unique elements and their counts
    result_dict = dict(zip(unique_elements, counts))
    return result_dict


def intersection_with_order(A, B):
    """
    A: list, must be ordered
    B: list, order is optional

    """
    intersection = []
    b_set = set(B)

    for item in A:
        if item in b_set:
            intersection.append(item)

    return intersection


def int_round(arr):
    """
    Round arrays. For all int(x)+y=int(x)+1 where y>=0.5

    """

    rounded_arr = []

    for a in arr:
        if a - np.floor(a) >= 0.5:
            rounded_arr.append(np.ceil(a).astype(np.int32))
        else:
            rounded_arr.append(np.floor(a).astype(np.int32))

    return np.array(rounded_arr).astype(np.int32)


def float_round(arr, rounding=4):
    """
    Round the elements of a NumPy array to a specified number of decimal places.

    Args:
    - arr: NumPy array containing float values to be rounded.
    - rounding: Optional. Number of decimal places to round to. Default is 4.

    Returns:
    - NumPy array with elements rounded to the specified number of decimal places.

    This function rounds the elements of the input NumPy array to the specified number
    of decimal places using the np.round() function. It then checks for any negative
    zero values resulting from the rounding operation and replaces them with zero.
    """
    if not isinstance(float_round,np.ndarray):
        rounded_arr=np.array([float_round])
    # Round the elements of the input array to the specified number of decimal places
    rounded_arr = np.round(arr, rounding)

    # Fix negative zero values resulting from rounding
    array_fixed = np.where(np.abs(rounded_arr) == 0, 0, rounded_arr)

    return array_fixed



def get_files_scan_one_cpu(dir):
    """Get the path of all the files in nested folders"""
    all_files = []
    for entry in os.scandir(dir):
        if entry.is_file():
            all_files.append(entry.path)
        elif entry.is_dir():
            all_files.extend(get_files_scan(entry.path))
    if len(all_files) > 0:
        return all_files
    else:
        raise FileNotFoundError("No Files found. Please check your directory.")


def scan_files_in_folder(folder):
    """Scans files in a folder and returns a list of file paths."""
    files = []
    for entry in os.scandir(folder):
        if entry.is_file():
            files.append(entry.path)
        elif entry.is_dir():
            files.extend(scan_files_in_folder(entry.path))
    return files


def get_files_scan(dir, max_workers=1):
    """Get the path of all the files in nested folders using concurrent.futures."""
    all_files = []
    if max_workers == 1:
        return get_files_scan_one_cpu(dir)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_dir = {
                executor.submit(scan_files_in_folder, entry.path): entry.path
                for entry in os.scandir(dir)
            }
            for future in concurrent.futures.as_completed(future_to_dir):
                folder = future_to_dir[future]
                try:
                    all_files.extend(future.result())
                except Exception as e:
                    print(f"An error occurred while scanning {folder}: {e}")

        if len(all_files) > 0:
            return all_files
        else:
            raise FileNotFoundError("No Files found. Please check your directory.")


def get_principal_curvature_from_gaussian_mean_curvature(
    curvature_gauss, curvature_mean, eps=1e-1
):

    b = curvature_mean**2 - curvature_gauss
    b = np.where(b < 0, eps, b)
    curvature_min = curvature_mean - np.sqrt(b)
    curvature_max = curvature_mean + np.sqrt(b)

    return curvature_min, curvature_max


def compute_curvature_from_mesh(mesh, radius=0.001, normalize=False):
    """
    Given a mesh, it computes Min, Max, Gaussian and Normal Curvature
    """
    area = trimesh.curvature.sphere_ball_intersection(R=1, r=radius)
    # Calculate Gaussian and Mean Curvature
    curvature_gauss = (
        trimesh.curvature.discrete_gaussian_curvature_measure(
            mesh, mesh.vertices, radius=radius
        )
        / area
    )
    curvature_mean = (
        trimesh.curvature.discrete_mean_curvature_measure(
            mesh, mesh.vertices, radius=radius
        )
        / area
    )

    # Calculate Principal Curvature from Gaussian and Mean Curvature
    curvature_min, curvature_max = get_principal_curvature_from_gaussian_mean_curvature(
        curvature_gauss, curvature_mean
    )
    if normalize:
        curvature_min = standard_normalization(curvature_min)
        curvature_max = standard_normalization(curvature_max)
        curvature_gauss = standard_normalization(curvature_gauss)
        curvature_mean = standard_normalization(curvature_mean)

    return curvature_min, curvature_max, curvature_gauss, curvature_mean


def hash_map(arr):
    hash_vec = sha256(np.ascontiguousarray(arr)).hexdigest()
    return hash_vec


def add_axis(vec, val=0):
    return np.concatenate([vec, np.ones(1) * val])


def euler_to_axis(theta, phi, gamma):
    """
    Converts Euler angles to axis.

    Args:
        theta (float): The rotation angle around the z-axis in radians.
        phi (float): The rotation angle around the x-axis in radians.
        gamma (float): The rotation angle around the y-axis in radians.

    Returns:
        x_axis (np.ndarray): The x-axis.
        y_axis (np.ndarray): The y-axis.
        z_axis (np.ndarray): The z-axis.
    """

    # Convert angle to 3d rotation matrix
    z_axis, x_axis = polar_parameterization_inverse(theta, phi, gamma)
    y_axis = np.cross(z_axis, x_axis)

    return x_axis, y_axis, z_axis


def add_padding(tensor, M):
    """
    Pads a torch tensor with 0 to make it (N+M,K).

    Args:
        tensor (torch.Tensor): The tensor to be padded.
        M (int): The number of 0s to be padded.

    Returns:
        torch.Tensor: The padded tensor.
    """

    shape = tensor.shape
    assert len(shape) in [1, 2]
    if M == 0:
        return tensor
    elif M < 0:
        raise ValueError(
            f"Number of padding tensors can't be zero. Tensor shape {tensor.shape}"
        )

    if len(shape) == 1:
        new_shape = (shape[0] + M,)
    else:
        new_shape = (shape[0] + M, shape[1])
    padded_tensor = torch.zeros(new_shape, dtype=torch.int32)
    padded_tensor[: shape[0]] = tensor

    return padded_tensor


def perform_op(big, small, op_name):
    if op_name == "cut":
        op = BRepAlgoAPI_Cut(big, small)
    elif op_name == "fuse":
        op = BRepAlgoAPI_Fuse(big, small)
    elif op_name == "common":
        op = BRepAlgoAPI_Common(big, small)
    op.SetFuzzyValue(PRECISION)
    op.Build()
    return op.Shape()


def merge_end_tokens_from_loop(lp):
    """
    The merge_end_tokens_from_loop function takes a list lp as input,
    representing a loop of tokens.
    It creates pairs of start and end tokens for curves with a specific structure of
    either [start, end] or [start, mid, end].

    Args:
        lp (list): List representing a loop of tokens.

    Returns:
        list: List of paired curve tokens.
    """
    paired_curve_token = []  # List to store the paired curve tokens

    assert np.min(lp[:, 0]) == SKETCH_TOKEN.index(
        "END_CURVE"
    ), f"Invalid Loop {lp}"  # Assert that the minimum value in the loop is 5

    curve_tokens = split_array(
        lp, SKETCH_TOKEN.index("END_CURVE")
    )  # Split the loop into separate curve tokens of size 5
    n = len(curve_tokens)  # Get the number of curve tokens

    if n == 1:  # If there is only one curve token (e.g., for a circle)
        paired_curve_token.append(curve_tokens)
    else:  # If there are multiple curve tokens
        paired_curve_token_loop = (
            []
        )  # List to store the paired curve tokens for each loop
        for i in range(n):
            # Concatenate the current curve token with the next curve token (wrapping around to the first token)
            paired_curve_token_loop.append(
                np.concatenate([curve_tokens[i], curve_tokens[(i + 1) % n][:1]])
            )
        paired_curve_token.append(paired_curve_token_loop)

    return paired_curve_token


def fix_coord_seq(seq):
    new_seq = []
    for i, crd in enumerate(seq[:-2]):
        if (crd < 6).any() and (crd >= 6).any():
            if 5 in crd.tolist():
                if (seq[i + 1] == 5).all():
                    new_seq.append([crd.max().item(), crd.max().item()])
                elif (seq[i + 1] == 4).all():
                    new_seq.append([crd.min().item(), crd.min().item()])
                elif (seq[i + 1] > 5).all():
                    new_seq.append([crd.min().item(), crd.min().item()])
                else:
                    new_seq.append([5, 5])
            elif 4 in crd.tolist():
                if new_seq[i - 1][0] == 5 and seq[i + 1][0] == 3:
                    new_seq.append([crd.min().item(), crd.min().item()])
                elif (seq[i + 1] >= 5).any():
                    if (seq[i + 2] > 5).any():
                        new_seq.append([crd.min().item(), crd.min().item()])
                    else:
                        new_seq.append([crd.max().item(), crd.max().item()])
                else:
                    new_seq.append([4, 4])
            elif 3 in crd.tolist():
                if new_seq[i - 1][0] == 4 and seq[i + 1][0] == 2:
                    new_seq.append([crd.min().item(), crd.min().item()])
                elif (seq[i + 1] >= 5).any():
                    if (seq[i + 2] > 5).any():
                        new_seq.append([crd.min().item(), crd.min().item()])
                    else:
                        new_seq.append([crd.max().item(), crd.max().item()])
                else:
                    new_seq.append([3, 3])
            else:
                new_seq.append([crd.min().item(), crd.min().item()])
        else:
            new_seq.append(crd.tolist())

    new_seq.append(seq[-1].tolist())
    return torch.tensor(new_seq)


def get_plane_normal(A, B, C):
    return np.cross((B - A), (C - A))


def create_point_from_array(point_array, transform=None):
    pt2d = gp_Pnt(point_array[0], point_array[1], point_array[2])
    if transform is not None:
        return pt2d.Transformed(transform)
    else:
        return pt2d


def pairNotIn(pair_list, query_list):
    """
    Find pairs in the query_list that are not present in the pair_list.

    Args:
        pair_list (list): List of pairs to compare against.
        query_list (list): List of pairs to check.

    Returns:
        list: List of pairs from the query_list that are not in the pair_list.

    ```
    pairNotIn([[1,2],[4,3]],[[4,3],[5,4]]) -> [[5, 4]]
    pairNotIn([2, 4, 6], [1, 2, 3, 4, 5, 6]) -> [1, 3, 5]
    ```
    """
    if len(pair_list) == 0:
        return query_list
    else:
        new_list = []
        for q in query_list:
            if q not in pair_list:
                new_list.append(q)
        return new_list


def rads_to_degs(rads):
    """Convert an angle from radians to degrees"""
    return 180 * rads / math.pi


def angle_from_vector_to_x(vec):
    """computer the angle (0~2pi) between a unit vector and positive x axis"""
    angle = 0.0
    # 2 | 1
    # -------
    # 3 | 4
    if vec[0] >= 0:
        if vec[1] >= 0:
            # Qadrant 1
            angle = math.asin(vec[1])
        else:
            # Qadrant 4
            angle = 2.0 * math.pi - math.asin(-vec[1])
    else:
        if vec[1] >= 0:
            # Qadrant 2
            angle = math.pi - math.asin(vec[1])
        else:
            # Qadrant 3
            angle = math.pi + math.asin(-vec[1])
    return angle


def cartesian2polar(vec, with_radius=False):
    """convert a vector in cartesian coordinates to polar(spherical) coordinates"""
    vec = vec.round(6)
    norm = np.linalg.norm(vec)
    theta = np.arccos(vec[2] / norm)  # (0, pi)
    phi = np.arctan(
        vec[1] / (vec[0] + 1e-15)
    )  # (-pi, pi) # FIXME: -0.0 cannot be identified here
    if not with_radius:
        return np.array([theta, phi])
    else:
        return np.array([theta, phi, norm])


def polar2cartesian(vec):
    """convert a vector in polar(spherical) coordinates to cartesian coordinates"""
    r = 1 if len(vec) == 2 else vec[2]
    theta, phi = vec[0], vec[1]
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])


def rotate_by_x(vec, theta):
    mat = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ]
    )
    return np.dot(mat, vec)


def rotate_by_y(vec, theta):
    mat = np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )
    return np.dot(mat, vec)


def rotate_by_z(vec, phi):
    mat = np.array(
        [[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]]
    )
    return np.dot(mat, vec)


def polar_parameterization(normal_3d, x_axis_3d):
    """represent a coordinate system by its rotation from the standard 3D coordinate system

    Args:
        normal_3d (np.array): unit vector for normal direction (z-axis)
        x_axis_3d (np.array): unit vector for x-axis

    Returns:
        theta, phi, gamma: axis-angle rotation
    """
    normal_polar = cartesian2polar(normal_3d)
    theta = normal_polar[0]
    phi = normal_polar[1]

    ref_x = rotate_by_z(rotate_by_y(np.array([1, 0, 0]), theta), phi)

    gamma = np.arccos(np.dot(x_axis_3d, ref_x).round(6))
    if np.dot(np.cross(ref_x, x_axis_3d), normal_3d) < 0:
        gamma = -gamma
    return theta, phi, gamma


def polar_parameterization_inverse(theta, phi, gamma):
    """build a coordinate system by the given rotation from the standard 3D coordinate system"""
    normal_3d = polar2cartesian([theta, phi])
    ref_x = rotate_by_z(rotate_by_y(np.array([1, 0, 0]), theta), phi)
    ref_y = np.cross(normal_3d, ref_x)
    x_axis_3d = ref_x * np.cos(gamma) + ref_y * np.sin(gamma)
    return normal_3d, x_axis_3d


def merge_list(lists: List[List[int]]) -> List[List[int]]:
    """
    Merge lists in a list of lists.

    Args:
        lists: A list of lists containing integers.

    Returns:
        A list of lists where consecutive lists are merged if the end point of
        the first list is the same as the start point of the next list.

    ```python
    merge_list([[0, 1], [1, 2], [4, 5], [5, 6], [6, 7]]) --> [[0, 1, 2], [4, 5, 6, 7]]
    ```
    """
    merged_list = []

    prev_end = None
    for segment in lists:
        if len(segment) == 1:
            start = segment[0]
            end = start
        else:
            start, end = segment
        if prev_end is not None and prev_end == start:
            merged_list[-1].extend(segment)
        else:
            merged_list.append(segment)
        prev_end = end

    merged_list = [sorted(list(set(seg))) for seg in merged_list]
    return merged_list


def get_orientation(p1, p2, p3):
    """Returns clockwise, anticlockwise and collinear"""
    # Calculate the cross product of vectors formed by three points
    cross_product = (p2[0] - p1[0]) * (p3[1] - p2[1]) - (p2[1] - p1[1]) * (
        p3[0] - p2[0]
    )

    if cross_product > 0:
        return "counterclockwise"
    elif cross_product < 0:
        return "clockwise"
    else:
        # print(p1,p2,p3)
        return "collinear"


def create_paired_token_curve(sketch):
    """
    Creates a pair of start and end token of curve with [start,end] or [start,mid,end].
    It splits the token at loop level (index= END_LOOP in Utility.macro.SKETCH_TOKEN)

    """
    all_loops = split_array(sketch, val=4)
    paired_curve_token = []
    for lp in all_loops:
        curve_tokens = split_array(lp, 5)
        n = len(curve_tokens)
        if n == 1:  # For circle
            paired_curve_token.append(curve_tokens)
        else:
            paired_curve_token_loop = []
            for i in range(n):
                paired_curve_token_loop.append(
                    np.concatenate([curve_tokens[i], curve_tokens[(i + 1) % n][:1]])
                )
            paired_curve_token.append(paired_curve_token_loop)

    return paired_curve_token


def seq_to_num_curve(sketch, end_curve=5):
    """
    Given a sketch sequence,returns the number of curves
    """
    num_total = {"line": 0, "arc": 0, "circle": 0}
    all_curves = split_array(sketch, end_curve)
    for arr in all_curves:
        if len(arr) == 1:
            num_total["line"] += 1
        elif len(arr) == 2:
            num_total["arc"] += 1
        else:
            num_total["circle"] += 1
    return num_total


def post_process_extrude_token(sketch_token, extrude_token):
    if len(split_array(extrude_token, EXTRUSION_TOKEN.index("START"))) == 1:
        num_sketch = len(
            split_array(sketch_token, val=SKETCH_TOKEN.index("END_SKETCH"))
        )
        num_ext_token = num_sketch * ONE_EXT_SEQ_LENGTH
        new_ext = extrude_token[: num_ext_token + 2]
        new_ext[-1] = 1
        return new_ext
    else:
        return extrude_token


def delete_keys_from_dict(dictionary, key_list):
    if len(key_list) == 0:
        return dictionary
    for key in key_list:
        del dictionary[key]
    return dictionary


def create_path_with_time(directory):
    """
    Given a directory, it returns a modified path with date and time
    """
    now = datetime.datetime.now()
    time_str = now.strftime("%H:%M")
    date_str = datetime.date.today()
    saveDir = os.path.join(directory, f"{date_str}/{time_str}")
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    return saveDir


def reorder_loop(loop):
    """
    Reorder loop so that the first curve start from bottom left position

    """
    # Find the bottom left corner point
    if len(loop) == 1:
        # No need to reorder circle
        return loop
    min_x = min(curve.start[0] for curve in loop)
    min_y = min(curve.start[1] for curve in loop)
    bottom_left = np.array([min_x, min_y])
    min_index = -1
    min_distance = np.inf
    for i in range(len(loop)):
        curve_distance = point_distance(loop[i].start, bottom_left, "l2")
        if curve_distance < min_distance:
            min_distance = curve_distance
            min_index = i
    sorted_loop = loop[min_index:] + loop[:min_index]
    return sorted_loop


def calculate_angle(vec):
    return np.angle(complex(vec[0], vec[1]))


def reverse_loop(loop):
    """
    Rotate the loop in reverse direction

    """
    rotated_loop = copy.deepcopy(loop[::-1])
    for curve in rotated_loop:
        curve.reverse()
    return rotated_loop


def loop2token(loop):
    assert loop[0].token is not None, "Add tokens in the curve"
    tokens = np.array([], dtype=np.int32)
    if len(loop) == 1:
        return np.concatenate([loop[0].token, np.array([5], dtype=np.int32)])
    for curve in loop:
        tokens = np.append(
            tokens, np.concatenate([curve.token[:-1], np.array([5], dtype=np.int32)])
        )
    return tokens


def point_distance(p1, p2, type="l2"):
    if type.lower() == "l2":
        return np.sqrt(np.sum((p1 - p2) ** 2))
    elif type.lower() == "l1":
        return np.sum(np.abs(p1 - p2))
    else:
        raise NotImplementedError(f"Distance type {type} not yet supported")


def chamfer_dist(gt_points, gen_points, offset=0, scale=1):
    gen_points = gen_points / scale - offset

    # one direction
    gen_points_kd_tree = cKDTree(gen_points)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points)
    gt_to_gen_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = cKDTree(gt_points)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(gen_points)
    gen_to_gt_chamfer = np.mean(np.square(two_distances))

    return gt_to_gen_chamfer + gen_to_gt_chamfer


def print_args(args):
    for arg in vars(args):
        value = getattr(args, arg)
        print(f"{arg}: {value}")


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def ensure_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def separate_array(array, val=3, pos=0):
    """
    Create two new array of arrays, one having val at pos and one having no pos

    returns numpy array

    """
    pos_val = np.where(array[:, pos] == val)[0]
    pos_not = np.where(array[:, pos] != val)[0]
    row_val = []
    row_not_val = []
    for p in pos_val:
        row_val.append(array[p])
    for p in pos_not:
        row_not_val.append(array[p])
    return np.array(row_not_val), np.array(row_val)


def flatten_nested_list(lst):
    flattened_list = []
    for item in lst:
        if isinstance(item, list):
            flattened_list.extend(flatten_nested_list(item))
        else:
            flattened_list.append(item)
    return flattened_list


def flatten(l):
    return [item for sublist in l for item in sublist]


def create_connected_curve_seq(sequence):
    # sequence=flatten(sequence)
    new_sequence = []
    for i, seq in enumerate(sequence):
        if i < len(sequence) - 1:
            result = list(sequence[i]) + [sequence[i + 1][0]]
        if i == len(sequence) - 1:
            result = list(sequence[i]) + list([0])
        new_sequence.append(result)
    return new_sequence


def min_max_normalization(array):
    return (array - array.min()) / (array.max() - array.min())


def standard_normalization(array, eps=1e-6):
    return (array - array.mean()) / (array.std() + eps)


def random_sample_points(points, num_points=1024):
    """Randomly sample points"""
    num_sample = points.shape[0]
    if num_sample > num_points:
        random_choice = np.random.choice(
            num_sample, num_points, replace=False
        )  # Sampling without replacement
    else:
        random_choice = np.random.choice(
            num_sample, num_points, replace=True
        )  # Sampling without replacement
    return points[random_choice], random_choice


def get_all_files(dirname):
    """Get the path of all the files in nested folders"""
    all_files = []
    for path, subdirs, files in os.walk(dirname):
        for name in files:
            all_files.append(os.path.join(path, name))
    if len(all_files) > 0:
        return all_files
    else:
        raise FileNotFoundError("No Files found. Please check your directory.")


def save_yaml_file(yaml_data, filename, output_dir):
    with open(os.path.join(output_dir, filename), "w+") as f:
        yaml.dump(yaml_data, f, default_flow_style=False)


def split_array_pos(arr, val=4, remove_less=True):
    """

    Returns the position of arrays by spliting on the val

    """
    if remove_less:
        arr = arr[np.unique(np.where(arr >= val)[0])]
    indices = np.unique(np.where(arr == val)[0])
    splited_indices = []
    for i in range(len(indices)):
        if i == 0:
            splited_indices.append([0, indices[i]])
        else:
            splited_indices.append([indices[i - 1] + 1, indices[i]])

    return np.array(splited_indices)


def get_loops_start_end(sketch):
    """
    Find start and end position for loops

    """
    all_pos = np.concatenate([np.zeros(1, dtype=np.int32), np.where(sketch == 4)[0]])
    all_loop_start_end = []
    for i in range(len(all_pos) - 1):
        loop_pos = np.where(sketch[all_pos[i] : all_pos[i + 1]] >= 5)[0]
        all_loop_start_end.append(
            [loop_pos[0] + all_pos[i], loop_pos[-1] + 1 + all_pos[i]]
        )

    return all_loop_start_end


def get_curves_start_end(sketch):
    loop_start_end = get_loops_start_end(sketch)
    curve_start_end = []
    for lp in loop_start_end:
        curve_start_end.append(split_array_pos(sketch[lp[0] : lp[1]], val=5) + lp[0])
    return curve_start_end


def split_array(arr, val=4, include_val=False, remove_less=True):
    """

    Returns an array of arrays by spliting on the val

    """
    if remove_less:
        arr = arr[np.unique(np.where(arr >= val)[0])]
    indices = np.unique(np.where(arr == val)[0])
    splited_array = []
    for i in range(len(indices)):
        if i == 0:
            if include_val:
                splited_array.append(arr[: indices[i] + 1])
            else:
                splited_array.append(arr[: indices[i]])
        else:
            if include_val:
                splited_array.append(arr[indices[i - 1] + 1 : indices[i] + 1])
            else:
                splited_array.append(arr[indices[i - 1] + 1 : indices[i]])

    return splited_array


def get_curve_type_param_from_seq(arr):
    """
    Returns the sequence of curve types

    """
    curve_type_seq = []
    all_arr = []
    pos = np.where(arr == 1)[0]
    arr = arr[pos[0] : pos[1] + 1]
    for ar in split_array(arr, 4):
        all_arr += split_array(ar, 5)
    for a in all_arr:
        if len(a) == 1:
            curve_type_seq.append(["line", a])
        elif len(a) == 2:
            curve_type_seq.append(["arc", a])
        else:
            curve_type_seq.append(["circle", a])
    return curve_type_seq


def get_curve_type_accuracy(pred_skt_seq, target_skt_seq):
    """

    pred_skt_seq: An array of shape (B,N)
    target_skt_seq: An array of shape (B,N)

    Returns the accuracy for every curve types

    """
    batch_size = pred_skt_seq.shape[0]
    batch_acc = {"line": [], "arc": [], "circle": []}
    for b in range(batch_size):
        type_pred = get_curve_type_from_param(pred_skt_seq[b])
        type_gt = get_curve_type_from_param(target_skt_seq[b])
        acc = {"line": 0, "arc": 0, "circle": 0}
        total_dict = calculate_num_curve(target_skt_seq[b])
        for i, l in enumerate(type_gt):
            if l == type_pred[i]:
                acc[l] += 1

        for key, values in acc.items():
            try:
                acc[key] = values / total_dict[key]
                batch_acc[key].append(acc[key])
            except:
                batch_acc[key].append(acc[key])
                pass
    return acc


def calculate_num_curve(arr):
    """
    Returns the dictionary for number of curve types
    """
    curve_num = {"line": 0, "arc": 0, "circle": 0}
    all_loops = split_array(arr, 4)
    all_loops = [split_array(lp, 5) for lp in all_loops]

    for lp in all_loops:
        if len(lp) == 1:
            curve_num["circle"] += 1
        elif len(lp) > 2:
            for cv in lp:
                if len(cv):
                    curve_num["line"] += 1
                else:
                    curve_num["arc"] += 1
    return curve_num


def get_curve_type_from_param(arr):
    """
    Returns the sequence of curve types e.g ['line','line','line','arc','circle']

    """
    curve_type_seq = []
    all_arr = []
    for ar in split_array(arr, 4):
        all_arr += split_array(ar, 5)
    for a in all_arr:
        if len(a) == 1:
            curve_type_seq.append("line")
        elif len(a) == 2:
            curve_type_seq.append("arc")
        else:
            curve_type_seq.append("circle")
    return curve_type_seq


def calculate_num_bool(arr, val=2):
    """
    Returns the dictionary of number of boolean operations

    """
    bool_dict = {"new": 0, "join": 0, "cut": 0, "intersect": 0}
    splited_array = split_array(arr, val=2)
    for ar in splited_array:
        if ar[-2] == 3:
            bool_dict["new"] += 1
        elif ar[-2] == 4:
            bool_dict["join"] += 1
        elif ar[-2] == 5:
            bool_dict["cut"] += 1
        else:
            bool_dict["intersect"] += 1
    return bool_dict


def create_flag_seq(seq, t):
    """
    Tensor of shape (B,1)
    t: int

    """
    device = seq.device
    seq = seq.cpu()
    flag_seq = torch.zeros_like(seq)
    batch_size = seq.shape[0]
    length = ONE_EXT_SEQ_LENGTH
    for b in range(batch_size):
        if seq[b] > 1:
            if t % length >= 1 and t % length <= 2:  # For Extrusion Distance
                flag_seq[b] = 2
            elif t % length >= 3 and t % length <= 5:  # For Translation
                flag_seq[b] = 3
            elif t % length >= 6 and t % length <= 8:  # For Euler Angles
                flag_seq[b] = 4
            elif t % length == 9:  # For Booleans
                flag_seq[b] = 5
            elif t % length == 10:  # For Scaling
                flag_seq[b] = 6
            elif t % length == 0:  # For END Extrusion Sketch
                flag_seq[b] = 7
        elif seq[b] == 1:
            flag_seq[b] = 1

    return flag_seq


def create_index_seq(
    seq, prev_token, prev_index, end_token=2, max_index=None, batch_max_index=None
):
    """
    seq: tensor of shape (B,1)
    prev_token: tensor of shape (B,1)
    prev_index: tensor of shape (B,1)
    """
    device = seq.device
    seq = seq.cpu()
    prev_token = prev_token.cpu()
    prev_index = prev_index.cpu()
    index_seq = torch.zeros_like(seq)
    batch_size = seq.shape[0]
    assert max_index is not None

    for b in range(batch_size):
        if seq[b] > 1 and prev_token[b] > end_token:
            index_seq[b] = min(max_index, prev_index[b])
        elif seq[b] > 1 and prev_token[b] == end_token:
            index_seq[b] = min(max_index, prev_index[b] + 1)
        elif seq[b] == 0 and prev_token[b] > 1:
            index_seq[b] = min(max_index, prev_index[b] + 1)
        elif seq[b] == 0 and prev_token[b] == 1:
            index_seq[b] = min(max_index, batch_max_index[b])
        elif seq[b] == 1 and prev_token[b] > 1:
            index_seq[b] = min(max_index, batch_max_index[b] + 1)
        elif seq[b] == 0 and prev_token[b] == 0:
            index_seq[b] = min(max_index, prev_index[b])

    return index_seq.to(device)


def create_index_seq_whole(seq_dict, token_key="coord_seq"):
    """
    Creates index sequence for all tokens of seq_dict

    """
    if seq_dict[token_key].shape[-1] == 2:
        token = coord_to_pixel(seq_dict[token_key].reshape(1, -1, 2))
    else:
        token = seq_dict[token_key].reshape(-1, 1)
    new_ind_seq = torch.zeros_like(token)
    for t in range(1, token.shape[1]):
        new_ind_seq[:, t] = create_index_seq(
            token[:, t],
            token[:, t - 1 : t],
            new_ind_seq[:, t - 1 : t],
            2,
            max_index=MAX_EXTRUSION,
            batch_max_index=torch.max(new_ind_seq, dim=-1).values,
        )

    return new_ind_seq


def create_flag_seq_whole(extrude_seq_dict):
    """
    Creates a flag sequence for all tokens of extrude_seq_dict

    """
    device = extrude_seq_dict["ext_seq"].device
    new_flag_seq = torch.zeros_like(extrude_seq_dict["ext_seq"])
    for t in range(extrude_seq_dict["ext_seq"].shape[1]):
        new_flag_seq[:, t : t + 1] = create_flag_seq(
            extrude_seq_dict["ext_seq"][:, t : t + 1], t
        ).to(device)

    return new_flag_seq


def coord_to_pixel(coord_arr, bit=N_BIT):
    """
    Converts a 2d coordinate array to 1d pixel array.
    """
    arr_type = type(coord_arr)
    if arr_type != torch.Tensor:
        coord_arr = torch.tensor(coord_arr)

    if len(coord_arr.shape) != 3:
        coord_arr = coord_arr.reshape(1, -1, 2)

    device = coord_arr.device if isinstance(coord_arr, torch.Tensor) else None
    coord_arr = torch.tensor(coord_arr).to("cpu")
    pixel_arr = torch.ones_like(coord_arr[:, :, 0])

    # Coordinates less than [7,7] are for start and End tokens
    for i in range(0, END_PAD):
        pixel_arr[(coord_arr == i).any(dim=-1)] = i

    mask = (coord_arr >= END_PAD).all(dim=-1)
    pixel_arr[mask] = (
        (coord_arr[mask][:, 1] - END_PAD) * (2**bit)
        + (coord_arr[mask][:, 0] - END_PAD)
        + END_PAD
    )

    return pixel_arr.to(device) if device is not None else pixel_arr


def pixel_to_coord(pixel, bit=N_BIT):
    """Converts pixel to coordinates
    pixel: tensor of shape (B,N)

    returns:
    tensor of shape (B,N,2)

    """

    if isinstance(pixel, torch.Tensor):
        device = pixel.device
    elif isinstance(pixel, np.ndarray):
        device = None
        pixel = torch.from_numpy(pixel)
    else:
        device = "cpu"

    x = torch.linspace(0, 2**bit - 1, 2**bit) + bit
    y = torch.linspace(0, 2**bit - 1, 2**bit) + bit
    xx, yy = torch.broadcast_tensors(x.unsqueeze(-1), y.unsqueeze(0))
    vertices = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1).long()

    vertices = torch.cat(
        (
            torch.ones(bit, 2).long() * torch.arange(bit).long().reshape(bit, 1),
            vertices,
        ),
        dim=0,
    )
    vertices = vertices[:, [1, 0]]
    coord = vertices[pixel.long().cpu()]

    return coord.to(device)


def stl_to_stp(stl_file_path, stp_file_path):
    """Converts a stl file path to stp format."""
    # Create a shape from the STL file
    stl_reader = StlAPI_Reader()
    shape = TopoDS_Shape()
    stl_reader.Read(shape, stl_file_path)

    # Write the shape to the STP file
    step_writer = STEPControl_Writer()
    step_writer.Transfer(shape, STEPControl_AsIs)
    status = step_writer.Write(stp_file_path)
    if status != 0:
        print(f"Error writing STP file: {stp_file_path}")
    else:
        print(f"STP file written: {stp_file_path}")


def write_ply(points, filename, text=False):
    """input: Nx3, write points to filename as PLY format."""
    points = [
        (points[i, 0], points[i, 1], points[i, 2]) for i in range(points.shape[0])
    ]
    vertex = np.array(points, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    el = PlyElement.describe(vertex, "vertex", comments=["vertices"])
    with open(filename, mode="wb") as f:
        PlyData([el], text=text).write(f)


def generate_attention_mask(
    sz1: int, sz2: int = None, device="cpu", mask_start_token=True
):
    r"""Generate an attention mask for the sequence. The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).
    """
    if sz2 is None:
        sz2 = sz1
    mask = torch.triu(torch.full((sz1, sz2), float("-inf"), device=device), diagonal=1)

    # Add Mask for start token
    if mask_start_token:
        mask[1:, 0] = float("-inf")
    return mask


def generate_start_token_mask(sz1: int, sz2: int, device="cpu"):
    mask = generate_attention_mask(sz1, sz2, device, True)
    mask[:, 1:] = 0
    return mask


def round_float(point):
    point["x"] = round(point["x"], 9)
    point["y"] = round(point["y"], 9)
    point["z"] = round(point["z"], 9)
    return


def find_files(folder, extension):
    return sorted(
        [
            Path(os.path.join(folder, f))
            for f in os.listdir(folder)
            if f.endswith(extension)
        ]
    )


# def plot(shape_list):
#     pyqt5_display, start_display, add_menu, add_function_to_menu = init_display(
#         "qt-pyqt5"
#     )
#     for shape in shape_list:
#         pyqt5_display.DisplayShape(shape, update=True)
#     start_display()


def save_to_json(filepath, data):
    with open(filepath, "w+") as f:
        json.dump(data, f, indent=2)


def save_to_pickle(data, filename):
    # Check if file exists
    if os.path.isfile(filename):
        with open(filename, "rb") as f:
            existing_data = pickle.load(f)
        # Append new data to existing list
        existing_data += f"\n{data}\n"
        data = existing_data

    # Save data to pickle file
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def brep2mesh(a_shape, mode="ascii", linear_deflection=0.001, angular_deflection=0.5):
    """Create a mesh from brep
    a_shape: the topods_shape to export
    filename: the filename
    mode: optional, "ascii" by default. Can either be "binary"
    linear_deflection: optional, default to 0.001. Lower, more occurate mesh
    angular_deflection: optional, default to 0.5. Lower, more accurate_mesh
    """
    if a_shape.IsNull():
        raise AssertionError("Shape is null.")
    if mode not in ["ascii", "binary"]:
        raise AssertionError("mode should be either ascii or binary")
    # first mesh the shape
    mesh = BRepMesh_IncrementalMesh(
        a_shape, linear_deflection, False, angular_deflection, True
    )
    # mesh.SetDeflection(0.05)
    mesh.Perform()
    if not mesh.IsDone():
        raise AssertionError("Mesh is not done.")
    name = random.randint(0, 999999)
    filename = f".temp_{name}.stl"
    stl_exporter = StlAPI_Writer()
    if mode == "ascii":
        stl_exporter.SetASCIIMode(True)
    else:  # binary, just set the ASCII flag to False
        stl_exporter.SetASCIIMode(False)
    stl_exporter.Write(a_shape, filename)

    if not os.path.isfile(filename):
        raise IOError(f"File not written to disk on path {filename}")
    mesh = trimesh.load(filename)
    os.remove(filename)
    return mesh


def write_stl_file(
    a_shape, filename, mode="ascii", linear_deflection=0.001, angular_deflection=0.5, verbose=False
):
    """export the shape to a STL file
    Be careful, the shape first need to be explicitely meshed using BRepMesh_IncrementalMesh
    a_shape: the topods_shape to export
    filename: the filename
    mode: optional, "ascii" by default. Can either be "binary"
    linear_deflection: optional, default to 0.001. Lower, more occurate mesh
    angular_deflection: optional, default to 0.5. Lower, more accurate_mesh
    """
    if a_shape.IsNull():
        raise AssertionError("Shape is null.")
    if mode not in ["ascii", "binary"]:
        raise AssertionError("mode should be either ascii or binary")
    if verbose:
        if os.path.isfile(filename):
            print("Warning: %s file already exists and will be replaced" % filename)
    # first mesh the shape
    mesh = BRepMesh_IncrementalMesh(
        a_shape, linear_deflection, False, angular_deflection, True
    )
    # mesh.SetDeflection(0.05)
    mesh.Perform()
    if not mesh.IsDone():
        raise AssertionError("Mesh is not done.")

    stl_exporter = StlAPI_Writer()
    if mode == "ascii":
        stl_exporter.SetASCIIMode(True)
    else:  # binary, just set the ASCII flag to False
        stl_exporter.SetASCIIMode(False)
    stl_exporter.Write(a_shape, filename)

    if not os.path.isfile(filename):
        raise IOError(f"File not written to disk on path {filename}")


def same_plane(plane1, plane2):
    same = True
    trans1 = plane1["pt"]
    trans2 = plane2["pt"]
    for key in trans1.keys():
        v1 = trans1[key]
        v2 = trans2[key]
        if v1["x"] != v2["x"] or v1["y"] != v2["y"] or v1["z"] != v2["z"]:
            same = False
    return same


def create_xyz(xyz):
    return gp_XYZ(xyz["x"], xyz["y"], xyz["z"])


def get_ax3(transform_dict):
    origin = create_xyz(transform_dict["origin"])
    x_axis = create_xyz(transform_dict["x_axis"])
    y_axis = create_xyz(transform_dict["y_axis"])
    z_axis = create_xyz(transform_dict["z_axis"])
    # Create new coord (orig, Norm, x-axis)
    axis3 = gp_Ax3(gp_Pnt(origin), gp_Dir(z_axis), gp_Dir(x_axis))
    return axis3


def get_transform(transform_dict):
    axis3 = get_ax3(transform_dict)
    transform_to_local = gp_Trsf()
    transform_to_local.SetTransformation(axis3)
    return transform_to_local.Inverted()


def create_sketch_plane(transform_dict):
    axis3 = get_ax3(transform_dict)
    return gp_Pln(axis3)


def create_point(point_dict, transform=None):
    pt2d = gp_Pnt(point_dict["x"], point_dict["y"], point_dict["z"])
    if transform is not None:
        return pt2d.Transformed(transform)
    else:
        return pt2d


def create_vector(vec_dict, transform):
    vec2d = gp_Vec(vec_dict["x"], vec_dict["y"], vec_dict["z"])
    return vec2d.Transformed(transform)


def create_unit_vec(vec_dict, transform):
    vec2d = gp_Dir(vec_dict["x"], vec_dict["y"], vec_dict["z"])
    return vec2d.Transformed(transform)


def dequantize_verts(verts, n_bits=8, min_range=-0.5, max_range=0.5, add_noise=False):
    """Convert quantized vertices to floats."""
    range_quantize = 2**n_bits - 1
    verts = verts.astype("float32")
    verts = verts * (max_range - min_range) / range_quantize + min_range
    return verts


def quantize(data, n_bits=8, min_range=-1.0, max_range=1.0):
    """Convert vertices in [-1., 1.] to discrete values in [0, n_bits**2 - 1]."""
    range_quantize = 2**n_bits - 1
    data_quantize = (data - min_range) * range_quantize / (max_range - min_range)
    data_quantize = np.clip(data_quantize, a_min=0, a_max=range_quantize)  # clip values
    return data_quantize.astype("int32")


def find_files(folder, extension):
    return sorted(
        [
            Path(os.path.join(folder, f))
            for f in os.listdir(folder)
            if f.endswith(extension)
        ]
    )


def find_files_path(folder, extension):
    return sorted(
        [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(extension)]
    )


def quantize(data, n_bits=8, min_range=-1.0, max_range=1.0):
    """Convert vertices in [-1., 1.] to discrete values in [0, n_bits**2 - 1]."""
    range_quantize = 2**n_bits - 1
    data_quantize = (data - min_range) * range_quantize / (max_range - min_range)
    data_quantize = np.clip(data_quantize, a_min=0, a_max=range_quantize)  # clip values
    return data_quantize.astype("int32")


def normalize_vertices_scale(vertices):
    """Scale the vertices so that the long diagonal of the bounding box is one."""
    vert_min = vertices.min(axis=0)
    vert_max = vertices.max(axis=0)
    extents = vert_max - vert_min
    scale = 0.5 * np.sqrt(np.sum(extents**2))  # -1 ~ 1
    return vertices / scale, scale


def find_arc_geometry(a, b, c):
    A = b[0] - a[0]
    B = b[1] - a[1]
    C = c[0] - a[0]
    D = c[1] - a[1]

    E = A * (a[0] + b[0]) + B * (a[1] + b[1])
    F = C * (a[0] + c[0]) + D * (a[1] + c[1])

    G = 2.0 * (A * (c[1] - b[1]) - B * (c[0] - b[0]))

    if G == 0:
        raise Exception("zero G")

    p_0 = (D * E - B * F) / G
    p_1 = (A * F - C * E) / G

    center = np.array([p_0, p_1])
    radius = np.linalg.norm(center - a)

    c2s_vec = (a - center) / np.linalg.norm(a - center)
    c2m_vec = (b - center) / np.linalg.norm(b - center)
    c2e_vec = (c - center) / np.linalg.norm(c - center)
    angle_s, angle_m, angle_e = (
        angle_from_vector_to_x(c2s_vec),
        angle_from_vector_to_x(c2m_vec),
        angle_from_vector_to_x(c2e_vec),
    )
    angle_s, angle_e = min(angle_s, angle_e), max(angle_s, angle_e)

    # angles = []
    # for xx in [a, b, c]:
    #     angle = angle_from_vector_to_x(xx - center)
    #     angles.append(angle)

    # ab = b-a
    # ac = c-a
    # cp = np.cross(ab, ac)
    # if cp >= 0:
    #     start_angle_rads = angles[0]
    #     end_angle_rads = angles[2]
    # else:
    #     start_angle_rads = angles[2]
    #     end_angle_rads = angles[0]

    # Get Clock Sign
    clock = clock_sign(a, b, c)
    return center, radius, angle_s, angle_e, clock


def clock_sign(start_point, mid_point, end_point):
    """get a boolean sign indicating whether the arc is on top of s->e"""
    s2e = end_point - start_point
    s2m = mid_point - start_point
    sign = np.cross(s2m, s2e) >= 0  # counter-clockwise
    return sign


def get_constant_schedule_with_warmup(optimizer, num_warmup_steps, last_epoch=-1):
    """
    Create a schedule with a constant learning rate preceded by a warmup period during which the learning rate
    increases linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def split_sequence(data, seq_len):
    n_subseq = len(data) - seq_len + 1
    subsequences = torch.stack(
        data.unfold(0, seq_len, 1).split(1, dim=0)[:n_subseq]
    ).squeeze(
        1
    )  # (N,SEQ_LEN)
    return subsequences


def rotation_mat_to_euler_from_vec(vec, if_quantize=True, n_bit=N_BIT):
    """
    vec: Numpy array of Dimension (9,1)

    returns:
    theta,phi,gamma
    """
    mat = vec.reshape(3, 3)
    angles = R.from_matrix(mat).as_euler("xyz", degrees=False)

    if if_quantize:
        theta = quantize(theta, n_bits=6, min_range=-1)

    return theta, phi, gamma
