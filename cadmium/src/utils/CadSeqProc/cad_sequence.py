# Adding Python Path
import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append("..")
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-1]))
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-2]))

import numpy as np
import argparse
from CadSeqProc.sequence.transformation.extrude_sequence import ExtrudeSequence
from CadSeqProc.sequence.sketch.sketchsequence import SketchSequence
from CadSeqProc.utility.logger import CLGLogger
from CadSeqProc.utility.macro import *
from CadSeqProc.utility.decorator import *
from CadSeqProc.utility.utils import (
    add_axis,
    add_padding,
    brep2mesh,
    get_files_scan,
    make_unique_dict,
    perform_op,
    random_sample_points,
    split_array,
    write_ply,
    write_stl_file,
    normalize_pc,
    point_distance,
    ensure_dir,
    get_files_scan,
)

import os.path
from trimesh.sample import sample_surface, sample_surface_even
from loguru import logger
from rich import print
import json
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Extend.DataExchange import write_step_file
import signal
from contextlib import contextmanager
import copy
from OCC.Core import BRepAdaptor
from CadSeqProc.OCCUtils.Topology import Topo
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
import torch
from typing import List
from collections import OrderedDict
import logging

# from functools import reduce
import logging.config

logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": True,
    }
)
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (
    precision_score,
    recall_score,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
import matplotlib.pyplot as plt


@contextmanager
def timeout(time):
    # Register a function to raise a TimeoutError on the signal.
    signal.signal(signal.SIGALRM, raise_timeout)
    # Schedule the signal to be sent after ``time``.
    signal.alarm(time)
    try:
        yield
    except TimeoutError:
        raise Exception("time out")
    finally:
        # Unregister the signal so it won't be triggered
        # if the timeout is not reached.
        signal.signal(signal.SIGALRM, signal.SIG_IGN)


def raise_timeout(signum, frame):
    raise TimeoutError


clglogger = CLGLogger().configure_logger().logger


class CADSequence(object):
    """
    A Cad Model Sequence which consists of sketch and extrusion sequences.

    Requires:

    sketch_seq: List of SketchSequence objects
    extrude_seq: List of extrudeSequence objects
    bbox: optional
    """

    def __init__(
        self,
        sketch_seq: List[SketchSequence],
        extrude_seq: List[ExtrudeSequence],
        bbox=None,
    ):
        self.sketch_seq = sketch_seq
        self.extrude_seq = extrude_seq
        self.bbox = bbox
        self.create_variables()

    def create_variables(self):
        self.sketch_points3D = np.array([])
        self.sketch_points3D_color = np.array([])
        self.sketch_points2D = np.array([])
        self.sampled_points = []  # Sampled from the CAD Model
        self.cad_model = None
        self.cad_model = None
        self.points = []
        self.trimesh_obj = []
        self.sketch_points = np.array([])
        self.mesh = None
        self.sketches = []
        self.extrudes = []
        self.curve_dict = {"Line": 0, "Arc": 0, "Circle": 0}
        self.face_point_dict = None
        self.edge_point_dict = None
        # self.all_curves = []
        # self.all_loops = []
        self.all_orientation = []
        self.all_edges = []
        self.is_numerical = False

        self.sketch_vec = []  # Sketch sequence in vector format
        self.extrude_vec = []  # Extrude sequence in vector format
        self.sketch_pixel_vec = []
        self.sketch_coord_vec = []
        self.bbox_3d_per_model = []
        self.skt_pc_mask = []  # Point cloud Mask for every sketch
        self.cumulative_cad_seq = []
        self.cumulative_model_bbox = []
        self.all_sketch_figure = []
        self.color_code = [
            [1.00, 0.67, 0.60],
            [0.00, 0.00, 0.70],
            [1.00, 1.00, 0.40],
            [1.00, 0.60, 0.80],
            [0.10, 1.00, 1.00],
            [0.75, 0.70, 1.00],
            [1.00, 0.90, 0.70],
            [0.40, 0.70, 1.00],
            [0.60, 0.00, 0.30],
            [0.90, 1.00, 0.70],
            [0.40, 0.00, 0.40],
        ]

    @staticmethod
    def from_vec(
        cad_vec,
        bit=N_BIT,
        post_processing=False,
        denumericalize=True,
        fix_collinearity=False,
    ):

        if not isinstance(cad_vec, np.ndarray):
            cad_vec = cad_vec.numpy()

        # Remove Padding

        cad_vec = cad_vec[np.where(cad_vec[:, 0] != END_TOKEN.index("PADDING"))[0]]

        if post_processing:
            # If the first and last token is not Start/End token, then add them respectively

            if cad_vec[-1, 0] != END_TOKEN.index("START"):
                cad_vec = np.concatenate(
                    [cad_vec, np.array([END_TOKEN.index("START"), 0]).reshape(1, 2)]
                )

            if len(np.where(cad_vec[1:, 0] == END_TOKEN.index("START"))[0]) == 0:
                cad_vec = np.concatenate(
                    [np.array([END_TOKEN.index("START"), 0]).reshape(1, 2), cad_vec]
                )

        # Removing the start and end token
        cad_vec = split_array(cad_vec, END_TOKEN.index("START"))[1]

        # Splitting the cad seq tokens into number of sketch and extrusion tokens
        skt_ext_seq = split_array(
            cad_vec, END_TOKEN.index("END_EXTRUSION"), False, False
        )

        sketch_seq = []
        extrude_seq = []

        for i, skt_ext in enumerate(skt_ext_seq):
            num_tokens = len(skt_ext)

            # <---- Sketch First CAD Sequence ---->
            # sketch = skt_ext[:-11] # The last 10 tokens are extrusion tokens and [2,0] is excluded
            sketch = split_array(skt_ext, END_TOKEN.index("END_SKETCH"), False, False)[
                0
            ]
            extrude = skt_ext[-10:]

            # <---- Extrusion First CAD Sequence ---->
            # extrude=skt_ext[:10]
            # sketch = skt_ext[11:]

            if i == 0 and len(sketch) == 0:
                raise Exception("No Sketch for the first model")
            if i > 0 and len(sketch) == 0:
                break

            sketch_seq.append(
                SketchSequence.from_vec(sketch, bit, post_processing, fix_collinearity)
            )
            extrude_seq.append(ExtrudeSequence.from_vec(extrude, bit, post_processing))
            sketch_seq[-1].coordsystem = extrude_seq[-1].coordsystem

        if denumericalize:
            return CADSequence(
                sketch_seq=sketch_seq, extrude_seq=extrude_seq
            ).denumericalize(bit=bit)
        else:
            return CADSequence(
                sketch_seq=sketch_seq,
                extrude_seq=extrude_seq,
            )

    @logger.catch()
    def to_vec(self, padding=False, max_cad_seq_len=MAX_CAD_SEQUENCE_LENGTH):

        self.cad_vec = [[END_TOKEN.index("START"), 0]]
        self.flag_vec = [0]  # Type of token
        self.index_vec = [0]  # Index of sketch-extrusion

        for i in range(len(self.sketch_seq)):
            skt_vec = self.sketch_seq[i].to_vec()
            ext_vec = self.extrude_seq[i].to_vec()

            # Sketch First CAD Sequence
            self.cad_vec += skt_vec
            self.cad_vec += ext_vec

            self.flag_vec += [0] * (len(skt_vec))
            self.flag_vec += [1] + list(range(1, 11))
            self.index_vec += [i] * (len(skt_vec) + len(ext_vec))

        self.cad_vec.append([END_TOKEN.index("START"), 0])
        self.flag_vec.append(0)
        self.index_vec.append(self.index_vec[-1])

        self.cad_vec = torch.tensor(self.cad_vec, dtype=torch.int32)

        if padding:
            num_pad = max_cad_seq_len - self.cad_vec.shape[0]
            self.cad_vec = add_padding(self.cad_vec, num_pad)
            self.flag_vec += [11] * num_pad
            self.index_vec += [max(self.index_vec) + 1] * num_pad

        self.flag_vec = torch.tensor(self.flag_vec, dtype=torch.int32)
        self.index_vec = torch.tensor(self.index_vec, dtype=torch.int32)
        return self

    @staticmethod
    def json_to_vec(
        data: dict,
        bit: int = N_BIT,
        padding: bool = True,
        max_cad_seq_len: int = MAX_CAD_SEQUENCE_LENGTH,
    ):
        """
        Converts a JSON file to a vector.

        Args:
            data (dict): The JSON data.
            bit (int): The bit depth of the vector.
            padding (bool): Whether to pad the vector.

        Returns:
            cad_seq: CADSequence object
            cad_vec: The vector.
            flag_vec: The flag vector.
            index_vec: The index vector.

        """
        # Create a CADSequence object from the JSON data.
        cad_seq = CADSequence.from_dict(data)

        # Normalize the CADSequence object.
        cad_seq = cad_seq.normalize(bit=bit)

        # Numericalize (Quantization) the CADSequence object.
        cad_seq = cad_seq.numericalize(bit=bit)

        # Convert the CADSequence object to a vector.
        cad_seq = cad_seq.to_vec(padding=padding, max_cad_seq_len=max_cad_seq_len)

        return cad_seq, cad_seq.cad_vec, cad_seq.flag_vec, cad_seq.index_vec

    @staticmethod 
    def minimal_json_to_seq(data: dict, bit: int):
        cad_seq = CADSequence.from_minimal_dict(data)
        return cad_seq
        
    @staticmethod
    def json_to_NormalizedCAD(data: dict, bit: int):
        # Create a CADSequence object from the JSON data.
        cad_seq = CADSequence.from_dict(data)

        # Normalize the CADSequence object.
        cad_seq.normalize(bit=bit)

        # Denormalize the sketch profile only
        for i, skt in enumerate(cad_seq.sketch_seq):
            skt.denormalize(
                bbox_size=cad_seq.extrude_seq[i].metadata["sketch_size"],
                translate=0,
                bit=bit,
            )
        return cad_seq

    @staticmethod
    def json_to_pc(
        data: dict,
        bit: int,
        n_points: int,
        eps: float = 1e-8,
        method: int = 1,
        mul: float = 0.05,
    ):
        """
        Sample Points from CAD Model constructed from Json

        Args:
            data (dict): The JSON data.
            bit (int): The bit depth of the vector.

        Returns:
            numpy.ndarray: The vector.

        """

        # Create a CADSequence object from the JSON data.
        cad_seq = CADSequence.from_dict(data)

        # Normalize the CADSequence object.
        cad_seq.normalize(bit=bit)

        # Denormalize the sketch profile only
        for i, skt in enumerate(cad_seq.sketch_seq):
            skt.denormalize(
                bbox_size=cad_seq.extrude_seq[i].metadata["sketch_size"],
                translate=0,
                bit=bit,
            )

        # Sample Uniform points
        cad_seq.sample_points(n_points=n_points, type="uniform").get_skt_pc_mask(
            eps=eps, method=method, mul=mul
        )

        return cad_seq.points, cad_seq.skt_pc_mask, cad_seq

    @staticmethod
    def from_minimal_dict(data):
        """construct CADSequence from minimal json data"""

        sketch_seq = []
        extrude_seq = []

        for part_id, part_data in data["parts"].items():

            if part_data == None:
                continue
            
            sketch_ops = SketchSequence.from_minimal_dict(part_data)
            extrude_ops = ExtrudeSequence.from_minimal_dict(part_data)
            sketch_seq.append(sketch_ops)
            extrude_seq.append(extrude_ops)
            setattr(
                    extrude_seq[-1], "coordsystem", sketch_ops.coordsystem
                )
            extrude_seq[-1].add_info("sketch_size", sketch_ops.bbox_size)
        return CADSequence(sketch_seq, extrude_seq)

    @staticmethod
    def from_dict(all_stat):
        """construct CADSequence from json data"""

        # Loop through all the sequence

        sketch_seq = []
        extrude_seq = []

        for item in all_stat["sequence"]:
            if item["type"] == "ExtrudeFeature":
                extrude_ops = ExtrudeSequence.from_dict(
                    all_stat, item["entity"]
                )  # Passes the whole data and id
                uid_pairs = extrude_ops.get_profile_uids()
                if len(uid_pairs) == 0:
                    continue
                extrude_seq.append(extrude_ops)
                sketch_ops = SketchSequence.from_dict(all_stat, uid_pairs)


                # Add some information in the extrude seq from sketch coordinate system
                # (since sketch sequence will only contain points)
                setattr(
                    extrude_seq[-1], "coordsystem", sketch_ops.coordsystem
                )  # Add attribute coordinate system from sketch coordinate system

                extrude_seq[-1].add_info("sketch_size", sketch_ops.bbox_size)

                sketch_seq.append(sketch_ops)

        bbox_info = all_stat["properties"]["bounding_box"]
        max_point = np.array(
            [
                bbox_info["max_point"]["x"],
                bbox_info["max_point"]["y"],
                bbox_info["max_point"]["z"],
            ]
        )
        min_point = np.array(
            [
                bbox_info["min_point"]["x"],
                bbox_info["min_point"]["y"],
                bbox_info["min_point"]["z"],
            ]
        )
        bbox = np.stack([max_point, min_point], axis=0)

        return CADSequence(sketch_seq, extrude_seq, bbox)

    def __repr__(self) -> str:
        s = f"CAD Sequence:\n"
        for i, skt in enumerate(self.sketch_seq):
            s += f"\n    - {skt.__repr__()}"
            s += f"\n    - {self.extrude_seq[i].__repr__()}"
        return s

    def _json(self) -> dict:
        """
        Convert the CADSequence object to a JSON object.

        Returns:
            dict: The JSON object.

        """
        cad_seq_repr = OrderedDict()
        # For NeurIPS setting
        # cad_seq_repr["final_shape"] = ""  # self.final_name
        # Otherwise
        cad_seq_repr["final_name"] = "" # 
        cad_seq_repr["final_shape"] = ""
        cad_seq_repr["parts"] = {}

        self.get_bounding_box_per_model(mul=1)

        for i, sketch in enumerate(self.sketch_seq):

            # length, width, height = abs(
            #     self.bbox_3d_per_model[i][1] - self.bbox_3d_per_model[i][0]
            # )
            length, width = sketch.dimension
            height = (
                self.extrude_seq[i].metadata["extent_two"]
                + self.extrude_seq[i].metadata["extent_one"]
            )

            cad_seq_repr["parts"][f"part_{i+1}"] = OrderedDict()
            cad_seq_repr["parts"][f"part_{i+1}"][
                f"coordinate_system"
            ] = sketch.coordsystem._json()
            cad_seq_repr["parts"][f"part_{i+1}"][f"sketch"] = sketch._json()
            cad_seq_repr["parts"][f"part_{i+1}"][f"extrusion"] = self.extrude_seq[
                i
            ]._json()
            cad_seq_repr["parts"][f"part_{i+1}"]["description"] = {
                # For NeurIPS setting
                # "shape": "",
                # Otherwise
                "name": "",
                "shape": "",
                "length": length,
                "width": width,
                "height": height,
            }

        # cad_seq_repr["bounding_box"]=self.bbox

        return cad_seq_repr

    def to_txt(self, file_name, output):
        """
        Write the CADSequence metadata to a text file.

        Args:
            file_name (str): The file name.
            output (str): The output directory.

        """
        if not os.path.exists(output):
            os.makedirs(output)
        with open(os.path.join(output, file_name + ".txt"), "w") as f:
            f.write(self.__repr__())

        return self

    def transform3D(self):
        for skt in self.sketch_seq:
            skt.transform3D()
        return self

    def transform(self, translate, scale):
        for i, skt in enumerate(self.sketch_seq):
            skt.transform(translate, scale)
            # self.extrude_seq[i].transform(translate,scale)
            self.extrude_seq[i].metadata["extent_one"] *= scale
            self.extrude_seq[i].metadata["extent_two"] *= scale

        return self

    @property
    def all_curves(self):
        curves = []

        for seq in self.sketch_seq:
            curves += seq.all_curves

        return curves

    @property
    def all_loops(self):
        all_loops = []
        for seq in self.sketch_seq:
            all_loops += seq.all_loops

        return all_loops

    @property
    def all_faces(self):
        all_faces = []
        for seq in self.sketch_seq:
            all_faces += seq.facedata

        return all_faces

    @property
    def start_point(self):
        return self.sketch_seq[0].coordsystem.rotate_vec(self.sketch_seq[0].start_point)

    def numericalize(self, bit=N_BIT):
        self.is_numerical = True
        size = 2**bit
        for i, skt in enumerate(self.sketch_seq):
            # skt.transform(np.array((size / 2, size / 2)), 1)
            skt.numericalize(bit=bit)
        for i, ext in enumerate(self.extrude_seq):
            ext.numericalize(bit=bit)

        return self

    def normalize(self, size=1, bit=N_BIT):
        """
        Two Normalization happens here.

        1. Global normalization: which normalizes the cad model and only affects the extrusion parameters.
            This normalization is done once and it's not reversible.
        2. Local normalization: which normalizes the sketch profile based on their sketch size.
                Scale and translate the sketch such that their bbox is (0,0) to (63,63).
                This normalization is reversible. We need to transform the sketch to its original size after local scaling.
                The sketch size parameter in extrusion sequence is responsible for this.
        """

        scale = (
            size * NORM_FACTOR / np.max(np.abs(self.bbox[0] - self.bbox[1]))
        )  # bbox_max-bbox_min
        # scale=np.array([1.0])
        sketch_start_point = self.start_point

        # Shifting the sketches by their corresponding start position (different for each sketch)
        for i, ext in enumerate(self.extrude_seq):
            translate = add_axis(self.sketch_seq[i].start_point)  # -sketch_start_point

            # Update the translation vector for each sketch
            ext.transform(
                translate=ext.coordsystem.rotate_vec(translate, translation=False)
                - self.bbox[1],
                scale=scale,
            )
            self.sketch_seq[i].normalize(translate=None, bit=bit)
            self.sketch_seq[i].coordsystem = ext.coordsystem
        return self

    def denumericalize(self, bit=N_BIT, post_processing=True):

        self.is_numerical = False
        size = 2**bit
        for i, ext in enumerate(self.extrude_seq):
            ext.denumericalize(bit=bit)
        for i, skt in enumerate(self.sketch_seq):
            skt.coordsystem = self.extrude_seq[i].coordsystem
            skt.denormalize(
                bbox_size=self.extrude_seq[i].metadata["sketch_size"],
                translate=0,
                bit=bit,
            )

        return self

    def sample_sketch_points3D(self, n_points=1024, color=False):
        """
        Sample 3D sketch points.

        """
        all_points = []
        all_colors = []

        for i, skt in enumerate(self.sketch_seq):
            skt_points_3d = skt.sample_points(n_points=n_points, point_dimension=3)
            all_points.append(skt_points_3d)
            if color:
                all_colors.append([self.color_code[i]] * len(skt_points_3d))
        all_points = np.vstack(all_points)
        if color:
            all_colors = np.vstack(all_colors)

        self.sketch_points3D, index = random_sample_points(all_points, n_points)
        if color:
            self.sketch_points3D_color = all_colors[index]

        return self

    def sample_sketch_points2D(self, n_points=1024):
        """
        Sample 2D sketch points.

        """
        all_points = []

        for skt in self.sketch_seq:
            all_points.append(skt.sample_points(n_points=n_points, point_dimension=2))

        all_points = np.vstack(all_points)

        self.sketch_points2D = random_sample_points(all_points, n_points)[0]
        return self

    def save_sketch3d_brep(self, output_dir):
        """
        Saves the 3d sketch as a brep
        """
        all_solids = []

        for skt in self.sketch_seq:
            all_solids.append(skt.create_skt3d_edge())

        ensure_dir(output_dir)

        for i in range(len(all_solids)):
            write_step_file(all_solids[i], os.path.join(output_dir, f"sketch_{i}.step"))

        return self

    @property
    def bbox_size(self):
        bbox_min, bbox_max = self.bbox[1], self.bbox[0]
        size = np.max(np.abs(np.concatenate([bbox_max, bbox_min])))
        return size

    @property
    def volume(self):
        bbox_min, bbox_max = self.bbox[1], self.bbox[0]
        distance = bbox_max - bbox_min
        return distance[0] * distance[1] * distance[2]

    def center(self, type="sketch"):
        if type == "sketch":
            return (self.all_sketch_bbox[0] + self.all_sketch_bbox[1]) / 2
        elif type == "3d":
            return (self.bbox[0] + self.bbox[1]) / 2
        else:
            raise Exception(f"Unknown type {type}")

    @property
    def all_sketch_bbox(self):
        all_min_box = []
        all_max_box = []
        for skt in self.sketch_seq:
            bbox = skt.bbox
            all_min_box.append(bbox[0])
            all_max_box.append(bbox[1])
        return np.array([np.min(all_min_box, axis=0), np.max(all_max_box, axis=0)])

    @property
    def all_curve_dict(self):
        curve_dict = {"line": 0, "arc": 0, "circle": 0}
        curve_types = [curve.curve_type for curve in self.all_curves]
        update_dict = make_unique_dict(curve_types)

        for key, val in update_dict.items():
            curve_dict[key] += val

        return curve_dict

    @property
    def all_extrusion_dict(self):
        operation_dict = {"new": 0, "cut": 0, "join": 0, "intersect": 0}

        for ext in self.extrude_seq:
            operation = ext.get_boolean()
            if operation == 0:
                operation_dict["new"] += 1
            elif operation == 1:
                operation_dict["join"] += 1
            elif operation == 2:
                operation_dict["cut"] += 1
            else:
                operation_dict["intersect"] += 1

        return operation_dict

    @property
    def model_type(self):
        n_gt_curves = len(self.all_curves)
        # Model type: simple (# gt curves<=10), moderate (# 10<gt curves<=20), complex(# gt curves>20)
        if n_gt_curves <= 10:
            model_type = "simple"
        elif n_gt_curves <= 20:
            model_type = "moderate"
        else:
            model_type = "complex"
        return model_type

    def create_cumulative_model(self, skip_first=True):
        """
        Data Augmentation method.

        For a model with N sketches (N>2), the method generates (N-2) new cad models.

        """
        if len(self.sketch_seq) < 2:
            clglogger.warning(
                f"Skipping because number of sketches must be greater than 2"
            )

        else:
            if skip_first:
                sketch_start_index = 2
            else:
                sketch_start_index = 1
            # Create a cumulative bounding box
            self.get_cumulative_bounding_box()
            for i in range(sketch_start_index, len(self.sketch_seq) - 1):
                self.cumulative_cad_seq.append(
                    CADSequence(
                        sketch_seq=self.sketch_seq[:i],
                        extrude_seq=self.extrude_seq[:i],
                        bbox=self.cumulative_model_bbox[i - 1],
                    )
                )
        return self

    def create_intermediate_model(self):
        """
        Data Augmentation method.

        For a model with N sketches (N>2), the method generates (N-2) new cad models.

        """
        self.interm_cad_seq = []
        # Create a cumulative bounding box
        self.get_bounding_box_per_model(mul=1)
        for i in range(len(self.sketch_seq)):
            self.interm_cad_seq.append(
                CADSequence(
                    sketch_seq=[self.sketch_seq[i]],
                    extrude_seq=[self.extrude_seq[i]],
                    bbox=self.bbox_3d_per_model[i],
                )
            )
        return self

    def get_cumulative_bounding_box(self):

        self.get_bounding_box_per_model(mul=1)  # Bounding box per model

        for i in range(1, len(self.bbox_3d_per_model)):
            bbox_list = np.vstack(self.bbox_3d_per_model[:i])
            self.cumulative_model_bbox.append(
                np.vstack([bbox_list.max(axis=0), bbox_list.min(axis=0)])
            )
        self.cumulative_model_bbox.append(self.bbox)
        return self

    @staticmethod
    def create_bbox_from_ext_vec(vec, bit, post_processing, mul):
        ext_seq = ExtrudeSequence.from_vec(
            vec, bit=bit, post_processing=post_processing
        )
        ext_seq.denumericalize(bit=bit, post_processing=post_processing)
        # Get the bounding box of the 2d sketch
        bbox_sketch2d = np.array([[0, 0], [1, 1]]) * ext_seq.metadata["sketch_size"]
        # Transform it to 3d
        rt_bbox = ext_seq.coordsystem.rotate_vec(bbox_sketch2d)

        # Create bounding box for 3d sketch
        bbox_skt3d = np.array([np.min(rt_bbox, axis=0), np.max(rt_bbox, axis=0)])

        # Perform extrusion operation on the bounding box points
        extent_one = ext_seq.metadata["extent_one"] * mul
        extent_two = ext_seq.metadata["extent_two"] * mul

        normal = ext_seq.coordsystem.metadata["z_axis"]
        bbox_skt3d_e1 = (
            bbox_skt3d + extent_one * normal
        )  # Bbox coordinates after extrusions
        bbox_skt3d_e2 = bbox_skt3d - extent_two * normal

        # Bounding box per model
        bbox_model_min = np.vstack([bbox_skt3d_e1, bbox_skt3d_e2]).min(axis=0)
        bbox_model_max = np.vstack([bbox_skt3d_e1, bbox_skt3d_e2]).max(axis=0)
        bbox_model = np.vstack([bbox_model_min, bbox_model_max])
        return bbox_model

    @staticmethod
    def mask_point_cloud_in_bbox(points, bbox_list, eps=1e-8, method=1):
        mask_list = []
        for bbox in bbox_list:
            mask = np.logical_and.reduce(
                ((points - bbox[0]) >= -eps) & ((points - bbox[1]) <= eps), axis=1
            )
            if mask.sum() == 0:  # If now points are chosen, use the whole point cloud
                mask = np.ones_like(mask) == 1
            mask_list.append(mask)
        return mask_list

    def get_bounding_box_per_model(self, mul=0.05):
        """
        Create a bounding box per sketch-extrusion model

        mul: percentage of extrusions. Larger mul will create bigger bounding box. mul=1 will give the bounding box of the part.
        """
        self.bbox_3d_per_model = []
        for i, skt in enumerate(self.sketch_seq):
            # get the bounding box of the 2d sketch
            #  #0099/00991089
            bbox_sketch2d = (
                np.array([[0, 0], [1, 1]]) * self.extrude_seq[i].metadata["sketch_size"]
            )

            # Transform it to 3d
            rt_bbox = self.extrude_seq[i].coordsystem.rotate_vec(bbox_sketch2d)

            # Create bounding box for 3d
            bbox_skt3d = np.array([np.min(rt_bbox, axis=0), np.max(rt_bbox, axis=0)])

            # print(bbox_skt3d,mul)

            # Perform extrusion operation on the bounding box points
            extent_one = self.extrude_seq[i].metadata["extent_one"] * mul
            extent_two = self.extrude_seq[i].metadata["extent_two"] * mul

            normal = self.extrude_seq[i].coordsystem.metadata["z_axis"]
            bbox_skt3d_e1 = (
                bbox_skt3d + extent_one * normal
            )  # Bbox coordinates after extrusions
            bbox_skt3d_e2 = bbox_skt3d - extent_two * normal

            # Bounding box per model
            bbox_model_min = np.vstack([bbox_skt3d_e1, bbox_skt3d_e2]).min(axis=0)
            bbox_model_max = np.vstack([bbox_skt3d_e1, bbox_skt3d_e2]).max(axis=0)
            bbox_model = np.vstack([bbox_model_min, bbox_model_max])
            self.bbox_3d_per_model.append(bbox_model)
        return self

    def get_skt_pc_mask(self, eps=1e-8, method=1, mul=0.05):

        if len(self.bbox_3d_per_model) == 0:
            self.get_bounding_box_per_model(mul=mul)

        if len(self.points) == 0:
            clglogger.error(
                f"No points sampled before. Use sample_points() method before"
            )
            raise Exception
            # self.sample_points(n_points=8192)

        for bbox in self.bbox_3d_per_model:
            mask = CADSequence.mask_point_cloud_in_bbox(
                self.points, [bbox], eps, method
            )[0]
            # clglogger.debug(f"Number of point cloud {mask.sum()}")
            self.skt_pc_mask.append(mask)

        return self

    def create_cad_model(self):
        cur_solid = None
        # print(len(self.sketch_seq))
        for i, skt in enumerate(self.sketch_seq):
            ext = self.extrude_seq[i]
            extrude_params = {
                "extrude_values": [
                    ext.metadata["extent_one"],
                    ext.metadata["extent_two"],
                ]
            }
            try:
                ext_solid = skt.build_body(extrude_params=extrude_params)
            except:
                if i == 0:  # If the first model is incorrect, then it's invalid
                    raise Exception("Invalid Model")
                else:
                    break

            # Check empty solid (skip)
            props = GProp_GProps()
            brepgprop.VolumeProperties(ext_solid, props)
            solid_volume = props.Mass()
            if solid_volume == 0:
                continue
            prev_solid = copy.deepcopy(cur_solid)
            # Perform set operation and save current cad up to this step
            set_op = EXTRUDE_OPERATIONS[ext.metadata["boolean"]]
            if set_op == "NewBodyFeatureOperation" or set_op == "JoinFeatureOperation":
                if cur_solid is None:
                    cur_solid = ext_solid
                else:
                    cur_solid = perform_op(cur_solid, ext_solid, "fuse")
            elif set_op == "CutFeatureOperation":
                cur_solid = perform_op(cur_solid, ext_solid, "cut")
            elif set_op == "IntersectFeatureOperation":
                cur_solid = perform_op(cur_solid, ext_solid, "common")
            else:
                raise Exception("Unknown operation type")

            # Nothing happened
            if cur_solid is None:
                continue

            # Check solid changes (skip)
            if prev_solid is not None:
                props = GProp_GProps()
                brepgprop.VolumeProperties(prev_solid, props)
                solid_volume1 = props.Mass()
                brepgprop.VolumeProperties(cur_solid, props)
                solid_volume2 = props.Mass()
                if (solid_volume1 - solid_volume2) == 0:
                    continue

            # Solid not valid (skip)
            analyzer = BRepCheck_Analyzer(cur_solid)
            if not analyzer.IsValid():
                continue
        if cur_solid is not None:
            self.cad_model = cur_solid
        else:
            raise Exception("No Cad Model generated")
        return self

    def create_mesh(
        self, mode="ascii", linear_deflection=0.001, angular_deflection=0.5
    ):
        """
        Converts a brep to mesh

        """
        if self.cad_model is None:
            self.create_cad_model()

        self.mesh = brep2mesh(
            self.cad_model,
            mode=mode,
            linear_deflection=linear_deflection,
            angular_deflection=angular_deflection,
        )
        return self

    def sample_points(self, n_points=1024, type="uniform", normalize=False):
        """
        Sample points from mesh.
        type: "uniform" or "even"

        """
        if self.mesh is None:
            self.create_mesh()

        if type == "uniform":
            self.points, face_indices = sample_surface(self.mesh, n_points)
        elif type == "even":
            self.points, _ = sample_surface_even(self.mesh, n_points)
        else:
            raise AssertionError(f"Unknown sample type {type}")
        if normalize:
            self.points = normalize_pc(self.points)
        return self

    def save_points(
        self,
        filename=None,
        output_folder=None,
        n_points=1024,
        pointype="uniform",
        type="ply",
        **kwargs,
    ):
        """
        Saves the sampled points from mesh.

        Args:
            filename (str, optional): The filename of the output file. If None,
                the filename will be generated automatically. Defaults to None.
            output_folder (str, optional): The output folder. Defaults to ".".
            n_points (int, optional): The number of points to sample. Defaults to 1024.
            pointype (str, optional): The type of points to sample. Defaults to "uniform". Other type "even", "edge","face","sketch3D","sketch2D"
            type (str, optional): The type of output file. Defaults to "ply".

        Returns:
            self: The object itself.
        """
        # Uniform Sampling of the 3D object
        if pointype == "uniform":
            self.sample_points(n_points=n_points)
            points = self.points

        # Even Sampling of the 3D object
        elif pointype == "even":
            self.sample_points(n_points=n_points, type="even")
            points = self.points

        # Edge Sampling of the 3D object
        elif pointype == "edge" and self.edge_point_dict is None:
            self.sample_edge_points(num_samples_edge=n_points)
            points = self.edge_point_dict["coord"]

        # Face Sampling of the 3D object
        elif pointype == "face" and self.face_point_dict is None:
            self.sample_face_points(num_samples_face=n_points)
            points = self.face_point_dict["coord"]

        # 3D Sketch Points
        elif pointype == "sketch3D" and len(self.sketch_points3D) == 0:
            self.sample_sketch_points3D(n_points=n_points)
            points = self.sketch_points3D

        # 2D Sketch Points
        elif pointype == "sketch2D" and len(self.sketch_points2D) == 0:
            self.sample_sketch_points2D(n_points=n_points)
            points = self.sketch_points2D
            points = np.concatenate((points, np.zeros((points.shape[0], 1))), axis=1)

        elif pointype == "skt_pc_mask":
            import open3d as o3d

            self.get_skt_pc_mask(**kwargs)

            green = [0, 1, 0]
            blue = [0, 0, 1]
            yellow = [1, 1, 0]
            cyan = [0, 1, 1]
            magenta = [1, 0, 1]
            red = [1, 0, 0]
            gray = [0.5, 0.5, 0.5]
            orange = [1, 0.5, 0]

            # Create a dictionary mapping integers to Open3D color arrays
            color_dict = {
                1: np.array(magenta),
                2: np.array(blue),
                3: np.array(yellow),
                4: np.array(cyan),
                5: np.array(green),
                6: np.array(red),
                7: np.array(gray),
                8: np.array(orange),
                9: np.array(blue),  # Repeating blue for key 10 as an example
            }
            colors = np.array([[0, 0, 0]] * (len(self.points)))
            for i, mask in enumerate(self.skt_pc_mask):
                colors[mask] = ((mask.reshape(-1, 1)) * color_dict[i + 1])[mask]

            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(self.points)
            point_cloud.colors = o3d.utility.Vector3dVector(colors)

        else:
            raise Exception(f"Invalid Point type {pointype}")

        assert output_folder is not None
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        pc_name = f"{filename}" + f".{type}"
        output_path = os.path.join(output_folder, pc_name)

        if pointype == "skt_pc_mask":
            o3d.io.write_point_cloud(output_path, point_cloud)
            return self

        write_ply(points, output_path)
        return self

    def save_stp(self, filename=None, output_folder=None, type="step"):
        """
        Saves the CAD object to .step(brep) or .stl(mesh)

        type: str. "step" or "stl"

        """
        if filename is None:
            filename = "000"
        if self.cad_model is None:
            self.create_cad_model()

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # with timeout(30):
        if type == "stl":
            stl_name = f"{filename}" + ".stl"
            output_path = os.path.join(output_folder, stl_name)
            write_stl_file(
                self.cad_model,
                output_path,
                linear_deflection=0.001,
                angular_deflection=0.5,
            )
        elif type == "step":
            step_name = f"{filename}" + ".step"
            output_path = os.path.join(output_folder, step_name)
            write_step_file(self.cad_model, output_path)
        else:
            raise Exception(f"{type} format not supported")

        return self

    def sample_edge_points(self, num_samples_edge=100):
        """
        Samples edge points with edge type and grouping
        shape: TopoDs_Shape
        num_samples: Number of samples per edge
        """
        edges = []
        wireframe_points = []
        edge_type = []
        edge_grouping = []

        if self.cad_model is None:
            self.create_cad_model()

        # Extract all edges from the shape
        topo = Topo(self.cad_model)
        edges = list(topo.edges())

        # Sample points from each edge
        for i, edge in enumerate(edges):
            curve = BRepAdaptor.BRepAdaptor_Curve(edge)

            u_start = curve.FirstParameter()
            u_end = curve.LastParameter()
            u_samples = np.linspace(u_start, u_end, num=num_samples_edge)
            points = [curve.Value(u) for u in u_samples]
            wireframe_points.extend(points)
            curve_type = [curve.GetType()] * len(points)
            curve_grouping = [i] * len(points)
            edge_type.extend(curve_type)
            edge_grouping.extend(curve_grouping)

        sampled_points = [[p.X(), p.Y(), p.Z()] for p in wireframe_points]

        self.edge_point_dict = {
            "coord": np.array(sampled_points),
            "edge_type": np.array(edge_type),
            "edge_grouping": np.array(edge_grouping),
        }
        return self

    def sample_face_points(self, num_samples_face=100):
        """
        Extract points from face os TopoDS_Shape object
        """
        faces = []
        surface_points = []
        face_type = []
        face_grouping = []
        if self.cad_model is None:
            self.create_cad_model()

        # Extract all faces from the shape
        explorer = TopExp_Explorer(self.cad_model, TopAbs_FACE)
        while explorer.More():
            face = explorer.Current()
            faces.append(face)
            explorer.Next()

        for i, face in enumerate(faces):
            surface = BRepAdaptor.BRepAdaptor_Surface(face)
            num_u_points = int(np.sqrt(num_samples_face))
            num_v_points = int(np.sqrt(num_samples_face))
            umin, umax, vmin, vmax = (
                surface.FirstUParameter(),
                surface.LastUParameter(),
                surface.FirstVParameter(),
                surface.LastVParameter(),
            )
            u_values = np.linspace(umin, umax, num_u_points)
            v_values = np.linspace(vmin, vmax, num_v_points)

            sampled_points = []
            for u in u_values:
                for v in v_values:
                    point = surface.Value(u, v).Coord()
                    sampled_points.append(point)

            surface_face_type = [surface.GetType()] * len(sampled_points)
            surface_face_grouping = [i] * len(sampled_points)

            face_type.extend(surface_face_type)
            surface_points.extend(sampled_points)
            face_grouping.extend(surface_face_grouping)

        self.face_point_dict = {
            "coord": np.array(surface_points),
            "face_type": np.array(face_type),
            "face_grouping": np.array(face_grouping),
        }
        return self

    def save_mesh_vertices_as_point_cloud(self, filename, output_folder):
        """
        Save the mesh vertices. Mostly these are edge points

        """

        if self.mesh is None:
            self.create_mesh()

        pc_name = f"{filename}" + f".ply"
        vertices = np.array(self.mesh.vertices)
        output_path = os.path.join(output_folder, pc_name)
        write_ply(vertices, output_path)
        return self

    def sketchAccuracyReport(self, target, uid, is_invalid, tolerance):
        """
        [Number of correct, Number of values present in target, Number of values present in self]
        """
        self.type_correct = {
            "line": np.array([0, 0, 0]),
            "arc": np.array([0, 0, 0]),
            "circle": np.array([0, 0, 0]),
        }  # Number of correct type prediction
        self.line_parameter_correct = {
            "x": np.array([0, 0]),
            "y": np.array([0, 0]),
        }  # Number of correct line parameter prediction
        self.arc_parameter_correct = {
            "x": np.array([0, 0]),
            "y": np.array([0, 0]),
            "f": np.array([0, 0]),
            "alpha": np.array([0, 0]),
        }  # Number of correct arc parameter prediction
        self.circle_parameter_correct = {
            "x": np.array([0, 0]),
            "y": np.array([0, 0]),
            "r": np.array([0, 0]),
        }  # Number of correct circle parameter prediction

        # Type Accuracy calculation
        n_gt_curves = len(self.all_curves)
        n_pred_curves = len(target.all_curves)
        min_num_curves = min(n_pred_curves, n_gt_curves)

        all_gt_type = np.array(
            [self.all_curves[i].curve_type.lower() for i in range(n_gt_curves)]
        )  # All curves in the ground truth sequence
        all_pred_type = np.array(
            [target.all_curves[i].curve_type.lower() for i in range(n_pred_curves)]
        )  # All curves in the target sequence
        correct_type = (
            all_gt_type[:min_num_curves] == all_pred_type[:min_num_curves]
        ) * 1  # like [0,1,1,0,0,1] # All curves in the common sequence

        num_ext_gt = len(self.extrude_seq)
        num_ext_pred = len(target.extrude_seq)

        n_min_ext = min(num_ext_pred, num_ext_gt)

        all_gt_dict = make_unique_dict(all_gt_type)
        all_pred_dict = make_unique_dict(all_pred_type)

        cm = confusion_matrix(
            all_gt_type[:min_num_curves],
            all_pred_type[:min_num_curves],
            labels=["line", "arc", "circle"],
        )

        for i in range(min_num_curves):
            self.type_correct[all_gt_type[i]][0] += correct_type[
                i
            ]  # +1 for the correct type
            self.type_correct[all_gt_type[i]][
                2
            ] += 1  # +1 for the type from the ground truth
            self.type_correct[all_pred_type[i]][1] += 1  # +1 for the type from target

        # Line/Arc/circle Parameter Accuracy for correct prediction
        for i, acc in enumerate(correct_type):
            if acc == 1:
                curve_type = all_gt_type[
                    i
                ]  # target and input curve types are same for acc=1
                input_curve = self.all_curves[i]
                pred_curve = target.all_curves[i]
                if curve_type == "line":
                    line_parameter_correct = input_curve.accuracyReport(
                        pred_curve, tolerance
                    )
                    for key, val in self.line_parameter_correct.items():
                        self.line_parameter_correct[key] = (
                            val + line_parameter_correct[key]
                        )

                elif curve_type == "arc":
                    arc_parameter_correct = input_curve.accuracyReport(
                        pred_curve, tolerance
                    )
                    for key, val in self.arc_parameter_correct.items():
                        self.arc_parameter_correct[key] = (
                            val + arc_parameter_correct[key]
                        )

                elif curve_type == "circle":
                    circle_parameter_correct = input_curve.accuracyReport(
                        pred_curve, tolerance
                    )
                    for key, val in self.circle_parameter_correct.items():
                        self.circle_parameter_correct[key] = (
                            val + circle_parameter_correct[key]
                        )

        sketch_dataframe = pd.DataFrame(
            {
                "uid": [uid],
                "num_skt_gt": [len(self.sketch_seq)],
                "num_skt_pred": [len(target.sketch_seq)],
                "num_ext": [n_min_ext],
                "line_correct_type": [
                    self.type_correct["line"][0]
                ],  # Number of correctly predicted lines in the common sequence (same length sequence as gt)
                "line_total_type": [
                    self.type_correct["line"][2]
                ],  # TNumber of gt lines in the common sequence
                "total_line_pred_common": [
                    self.type_correct["line"][1]
                ],  # Number of predicted lines in the common sequence
                "total_line_pred": [
                    all_pred_dict["line"] if "line" in all_pred_dict else 0
                ],  # Number of predicted lines in the predicted sequence (can be of different len than gt)
                "total_line_gt": [
                    all_gt_dict["line"] if "line" in all_gt_dict else 0
                ],  # Number of ground truth lines in the ground truth sequence (can be of different len than predicted)
                "line_correct_param_x": [self.line_parameter_correct["x"][0]],
                "line_correct_param_y": [self.line_parameter_correct["y"][0]],
                "arc_correct_type": [self.type_correct["arc"][0]],
                "arc_total_type": [self.type_correct["arc"][2]],
                "total_arc_pred_common": [self.type_correct["arc"][1]],
                "total_arc_pred": [
                    all_pred_dict["arc"] if "arc" in all_pred_dict else 0
                ],
                "total_arc_gt": [all_gt_dict["arc"] if "arc" in all_gt_dict else 0],
                "arc_correct_param_x": [self.arc_parameter_correct["x"][0]],
                "arc_correct_param_y": [self.arc_parameter_correct["y"][0]],
                "arc_correct_param_alpha": [self.arc_parameter_correct["alpha"][0]],
                "arc_correct_param_f": [self.arc_parameter_correct["f"][0]],
                "circle_correct_type": [self.type_correct["circle"][0]],
                "circle_total_type": [self.type_correct["circle"][2]],
                "total_circle_pred_common": [self.type_correct["circle"][1]],
                "total_circle_pred": [
                    all_pred_dict["circle"] if "circle" in all_pred_dict else 0
                ],
                "total_circle_gt": [
                    all_gt_dict["circle"] if "circle" in all_gt_dict else 0
                ],
                "circle_correct_param_x": [self.circle_parameter_correct["x"][0]],
                "circle_correct_param_y": [self.circle_parameter_correct["y"][0]],
                "circle_correct_param_r": [self.circle_parameter_correct["r"][0]],
                "is_invalid": [is_invalid],
                "model_type": [self.model_type],  # "Simple", "Moderate" or "Complex"
            }
        )

        return sketch_dataframe, cm

    def extrusionAccuracyReport(self, target, uid):

        # Extrusion parameters are normalized between 0 to 0.75, for comparison, we normalize between 0 and 1
        SCALING_FACTOR = 0.75
        self.parameter_report = {
            "dist": 0,
            "o_x": 0,
            "o_y": 0,
            "o_z": 0,
            "theta": 0,
            "phi": 0,
            "gamma": 0,
            "s": 0,
            "b": 0,
        }

        # Reverse scale the parameters
        for ext in self.extrude_seq:
            ext.denumericalize(bit=8, post_processing=False, uid=uid)
            ext.transform(0, 1 / SCALING_FACTOR)

        for ext in target.extrude_seq:
            ext.denumericalize(bit=8, post_processing=False)
            ext.transform(0, 1 / SCALING_FACTOR)

        num_ext_gt = len(self.extrude_seq)
        num_ext_pred = len(target.extrude_seq)

        n_min_ext = min(num_ext_pred, num_ext_gt)
        for i in range(n_min_ext):
            self.parameter_report["dist"] += point_distance(
                self.extrude_seq[i].get_total_extent(return_quantized=False),
                target.extrude_seq[i].get_total_extent(return_quantized=False),
                "l1",
            )

            self.parameter_report["o_x"] += point_distance(
                self.extrude_seq[i].coordsystem.metadata["origin"][0],
                target.extrude_seq[i].coordsystem.metadata["origin"][0],
                "l1",
            )
            self.parameter_report["o_y"] += point_distance(
                self.extrude_seq[i].coordsystem.metadata["origin"][1],
                target.extrude_seq[i].coordsystem.metadata["origin"][1],
                "l1",
            )
            self.parameter_report["o_z"] += point_distance(
                self.extrude_seq[i].coordsystem.metadata["origin"][2],
                target.extrude_seq[i].coordsystem.metadata["origin"][2],
                "l1",
            )

            self.parameter_report["theta"] += point_distance(
                self.extrude_seq[i].coordsystem.metadata["euler_angles"][0],
                target.extrude_seq[i].coordsystem.metadata["euler_angles"][0],
                "l1",
            )

            self.parameter_report["phi"] += point_distance(
                self.extrude_seq[i].coordsystem.metadata["euler_angles"][1],
                target.extrude_seq[i].coordsystem.metadata["euler_angles"][1],
                "l1",
            )

            self.parameter_report["gamma"] += point_distance(
                self.extrude_seq[i].coordsystem.metadata["euler_angles"][2],
                target.extrude_seq[i].coordsystem.metadata["euler_angles"][2],
                "l1",
            )

            self.parameter_report["s"] += point_distance(
                self.extrude_seq[i].metadata["sketch_size"],
                target.extrude_seq[i].metadata["sketch_size"],
                "l1",
            )

            self.parameter_report["b"] += (
                self.extrude_seq[i].metadata["boolean"]
                == target.extrude_seq[i].metadata["boolean"]
            )

        extrusion_dataframe = pd.DataFrame(
            {
                "theta": [self.parameter_report["theta"]],
                "phi": [self.parameter_report["phi"]],
                "gamma": [self.parameter_report["gamma"]],
                "o_x": [self.parameter_report["o_x"]],
                "o_y": [self.parameter_report["o_y"]],
                "o_z": [self.parameter_report["o_z"]],
                "scale": [self.parameter_report["s"]],
                "dist": [self.parameter_report["dist"]],
                "b": [self.parameter_report["b"]],
                "num_ext_pred": [num_ext_pred],
                "num_ext_gt": [num_ext_gt],
                "num_ext": [n_min_ext],
            }
        )

        return extrusion_dataframe

    def accuracyReport(self, target, uid, is_invalid=0, tolerance=4):
        sketch_dataframe, cm = self.sketchAccuracyReport(
            target, uid, is_invalid, tolerance
        )
        extrusion_dataframe = self.extrusionAccuracyReport(
            target, uid, is_invalid, tolerance
        )

        return sketch_dataframe, cm, extrusion_dataframe

    def analysisReport(self):
        curve_dict = self.all_curve_dict
        face_str = ""
        loop_str = ""
        for skt in self.sketch_seq:
            face_str += " " + str(len(skt.facedata))
            for lp in skt.facedata:
                loop_str += " " + str(len(lp.loopdata))
            loop_str += "|"

        analysis_df = pd.DataFrame(
            {
                "sketch": [len(self.sketch_seq)],
                "line": [curve_dict["line"]],
                "arc": [curve_dict["arc"]],
                "circle": [curve_dict["circle"]],
                "face": [face_str],
                "loop": [loop_str],
                "num_curves": [len(self.all_curves)],
                "volume(10^3)": self.volume * 1000,
            }
        )

        return analysis_df

    def generate_report(self, pred_cad, uid="0000"):
        # Report Generation Post-Loop Matching

        # <------------- Generate Matching Curve Pair ----------------->

        gt_seq = copy.deepcopy(self.sketch_seq)
        pred_seq = copy.deepcopy(pred_cad.sketch_seq)

        n_gt = len(gt_seq)
        n_pred = len(pred_seq)
        n_max = max(n_gt, n_pred)

        if n_gt < n_max:
            gt_seq += [None] * (n_max - n_gt)
        elif n_pred < n_max:
            pred_seq += [None] * (n_max - n_pred)

        matched_curve_pair = []
        matched_loop_pair = []

        # Match predicted curve with gt curve in every sketch (same sketch for curve pair)
        for i in range(n_max):
            skt_matched_curve_pair, skt_matched_loop_pair = SketchSequence.loop_match(
                gt_sketch=gt_seq[i], pred_sketch=pred_seq[i], scale=1, multiplier=1
            )
            matched_curve_pair += skt_matched_curve_pair
            matched_loop_pair += skt_matched_loop_pair

        null_curve_index = len(CURVE_TYPE)

        y_true = np.array(
            [
                (
                    CURVE_TYPE.index(i[0].curve_type.capitalize())
                    if i[0] is not None
                    else null_curve_index
                )
                for i in matched_curve_pair
            ]
        )
        y_pred = np.array(
            [
                (
                    CURVE_TYPE.index(i[1].curve_type.capitalize())
                    if i[1] is not None
                    else null_curve_index
                )
                for i in matched_curve_pair
            ]
        )

        unique_labels = np.unique(y_true)
        unique_labels = unique_labels[unique_labels != null_curve_index]

        report_df = pd.DataFrame(
            {
                "uid": [uid],
                "line_recall": [0],  # Recall of line
                "line_precision": [0],  # Precision of line
                "line_f1": [0],  # F1 of line
                "line_correct_type": [0],  # Number of correctly predicted line
                "line_total_pred": [
                    0
                ],  # Total number of line predicted in the prediction
                "line_total_gt": [0],  # Total number of line in the ground truth
                "line_param_s_x": [
                    0
                ],  # Number of correctly predicted x parameter for start point for correctly predicted line
                "line_param_s_y": [
                    0
                ],  # Number of correctly predicted y parameter for start point for correctly predicted line
                "line_param_e_x": [
                    0
                ],  # Number of correctly predicted x parameter for end point for correctly predicted line
                "line_param_e_y": [
                    0
                ],  # Number of correctly predicted y parameter for end point for correctly predicted line
                "arc_recall": [0],
                "arc_precision": [0],
                "arc_f1": [0],
                "arc_correct_type": [0],
                "arc_total_pred": [0],
                "arc_total_gt": [0],
                "arc_param_s_x": [0],
                "arc_param_s_y": [0],
                "arc_param_m_x": [
                    0
                ],  # Number of correctly predicted x parameter for mid point for correctly predicted line
                "arc_param_m_y": [
                    0
                ],  # Number of correctly predicted y parameter for mid point for correctly predicted line
                "arc_param_e_x": [0],
                "arc_param_e_y": [0],
                "arc_param_x": [0],
                "arc_param_y": [0],
                "circle_recall": [0],
                "circle_precision": [0],
                "circle_f1": [0],
                "circle_total_pred": [0],
                "circle_total_gt": [0],
                "circle_correct_type": [0],
                "circle_param_c_x": [
                    0
                ],  # Number of correctly predicted x parameter for center point for correctly predicted line
                "circle_param_c_y": [
                    0
                ],  # Number of correctly predicted y parameter for center point for correctly predicted line
                "circle_param_r": [0],
                "macro_recall": [0],  # Macro-Average Recall
                "macro_precision": [0],  # Macro-Average Precision
                "macro_f1": [0],
                "micro_recall": [0],  # Micro-Average Recall
                "micro_precision": [0],  # Micro-Average Precision
                "micro_f1": [0],  # Micro-Average F1
                "total_type_acc": [0],
                "model_type": self.model_type,  # Accuracy of the type
                "num_skt_gt": len(self.sketch_seq),
                "num_skt_pred": len(pred_cad.sketch_seq),
            }
        )

        all_labels = list(range(len(CURVE_TYPE) + 1))  # [0,1,2,3] (3 for Null Curve)
        precs_score = precision_score(
            y_true, y_pred, labels=all_labels, average=None
        )  # +1 for Null curve
        rec_score = recall_score(
            y_true, y_pred, labels=all_labels, average=None
        )  # +1 for Null curve
        f1 = f1_score(
            y_true, y_pred, labels=all_labels, average=None
        )  # +1 for Null curve

        cm = confusion_matrix(y_true, y_pred, labels=all_labels)
        correct_prediction_position = np.where(y_true == y_pred)[
            0
        ]  # Positions where types are correctly predicted
        for i in range(
            len(CURVE_TYPE)
        ):  # No information for Null Curve (It should be mentioned only in confusion matrix)
            curve_type = CURVE_TYPE[i].lower()
            curve_position = np.where(y_true == i)
            report_df[f"{curve_type}_precision"] = precs_score[i]
            report_df[f"{curve_type}_recall"] = rec_score[i]
            report_df[f"{curve_type}_f1"] = f1[i]
            report_df[f"{curve_type}_correct_type"] = cm[i, i]
            report_df[f"{curve_type}_total_pred"] = np.sum(cm[:, i])
            report_df[f"{curve_type}_total_gt"] = np.sum(cm[i, :])

            param_acc_pos = np.intersect1d(correct_prediction_position, curve_position)

            # Calculate Parameter Accuracy for all the correctly predicted positions
            for pos in param_acc_pos:
                curve1 = matched_curve_pair[pos][0]
                curve2 = matched_curve_pair[pos][1]
                assert curve1.curve_type == curve2.curve_type
                accuracyReport = curve1.accuracyReport(curve2, 1)
                if i == 0:  # For Line (start and end point)
                    report_df["line_param_s_x"] += accuracyReport["s"][0]
                    report_df["line_param_s_y"] += accuracyReport["s"][1]
                    report_df["line_param_e_x"] += accuracyReport["e"][0]
                    report_df["line_param_e_y"] += accuracyReport["e"][1]
                elif i == 1:  # For Arc (start,mid,end point)
                    report_df["arc_param_s_x"] += accuracyReport["s"][0]
                    report_df["arc_param_s_y"] += accuracyReport["s"][1]
                    report_df["arc_param_m_x"] += accuracyReport["m"][0]
                    report_df["arc_param_m_y"] += accuracyReport["m"][1]
                    report_df["arc_param_e_x"] += accuracyReport["e"][0]
                    report_df["arc_param_e_y"] += accuracyReport["e"][1]
                elif i == 2:  # For Circle (Center and Radius)
                    report_df["circle_param_c_x"] += accuracyReport["c"][0]
                    report_df["circle_param_c_y"] += accuracyReport["c"][1]
                    report_df["circle_param_r"] += accuracyReport["r"][0]

        # mean_precision,mean_recall,mean_f1=calc_mean_precision_recall_f1_score(report_df=report_df,round=None)

        # Add Micro and Macro Average scores
        report_df["macro_recall"] = recall_score(
            y_true, y_pred, labels=unique_labels, average="macro"
        )
        report_df["macro_precision"] = precision_score(
            y_true, y_pred, labels=unique_labels, average="macro"
        )
        report_df["macro_f1"] = f1_score(
            y_true, y_pred, labels=unique_labels, average="macro"
        )

        report_df["micro_recall"] = recall_score(
            y_true, y_pred, labels=unique_labels, average="micro"
        )
        report_df["micro_precision"] = precision_score(
            y_true, y_pred, labels=unique_labels, average="micro"
        )
        report_df["micro_f1"] = f1_score(
            y_true, y_pred, labels=unique_labels, average="micro"
        )

        report_df["total_type_acc"] = accuracy_score(y_true, y_pred)

        # Generate Extrusion Report

        extrusion_df = self.extrusionAccuracyReport(target=pred_cad, uid=uid)

        report_df = pd.concat([report_df, extrusion_df], axis=1)

        return report_df, cm

    def draw(self, ax=None, colors=None):
        # TODO: FINISH THE CODES

        if colors is None:
            colors = [
                "red",
                "blue",
                "green",
                "brown",
                "pink",
                "yellow",
                "purple",
                "black",
            ] * 10
        else:
            colors = [colors] * 100

        for i, sketch in enumerate(self.sketch_seq):
            fig, ax = plt.subplots()

            sketch.draw(ax, colors[i])
            self.all_sketch_figure.append([fig, ax])

           

        return self

