import copy
import os, sys

sys.path.append("..")
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-3]))

import numpy as np
from CadSeqProc.utility.logger import CLGLogger
from CadSeqProc.utility.macro import *
from CadSeqProc.utility.utils import (
    coord_to_pixel,
    float_round,
    create_point_from_array,
    dequantize_verts,
    int_round,
    pixel_to_coord,
    quantize,
    point_distance,
)
from CadSeqProc.geometry.curve import Curve
from rich import print
import torch
from loguru import logger
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
from OCC.Core.gp import gp_Pnt

clglogger = CLGLogger().configure_logger().logger


class Line(Curve):
    def __init__(self, metadata):
        self.metadata = metadata

        self.is_numerical = False

    @staticmethod
    def from_minimal_dict(curve):
        metadata = {}

        metadata["start_point"] = np.array(curve["Start Point"])
        metadata["end_point"] = np.array(curve["End Point"])
        
        return Line(metadata)

    @staticmethod
    def from_dict(line_entity: dict):
        metadata = {}
        metadata["start_point"] = np.array(
            [line_entity["start_point"]["x"], line_entity["start_point"]["y"]]
        )
        metadata["end_point"] = np.array(
            [line_entity["end_point"]["x"], line_entity["end_point"]["y"]]
        )
        return Line(metadata)

    def to_vec(self):
        """
        vector representation of line
        """
        assert self.is_numerical is True, clglogger.error(
            "The points are not quantized."
        )
        coord_token = [
            (self.metadata["start_point"] + END_PAD + BOOLEAN_PAD).tolist(),
            [self.token_index, 0],
        ]
        return coord_token

    @staticmethod
    def from_vec(vec, bit=N_BIT, post_processing=False):
        metadata = {}
        vec -= END_PAD + BOOLEAN_PAD
        metadata["start_point"] = vec[0]
        metadata["end_point"] = vec[1]
        line = Line(metadata=metadata)
        line.quantized_metadata = metadata.copy()
        line.bit = bit
        return line

    def sample_points(self, n_points=32):
        points = np.linspace(
            self.metadata["start_point"], self.metadata["end_point"], num=n_points
        )
        
        return points

    @property
    def min_point(self):
        if np.all(
            self.metadata["start_point"] <= self.metadata["end_point"]
        ):
            return self.metadata["start_point"]
        else:
            return self.metadata["end_point"]
        

    def __repr__(self):
        line_repr = "{}: Start({}), End({})".format(
            self.__class__.__name__,
            self.metadata["start_point"].round(4),
            self.metadata["end_point"].round(4),
        )
        return line_repr

    @property
    def curve_type(self):
        return "line"

    @property
    def start_point(self):
        return self.metadata["start_point"]

    def get_point(self, point_type):
        return self.metadata[point_type]
       

    @property
    def bbox(self):
        points = np.stack(
            [self.metadata["start_point"], self.metadata["end_point"]], axis=0
        )
        return np.stack([np.min(points, axis=0), np.max(points, axis=0)], axis=0)


    def transform(self, translate, scale):
        """
        Transform the 2d points if 3D transformation is not done

        """
        # clglogger.debug(f"Before {translate} {scale} {self.metadata}")
        self.metadata["start_point"] = (
            self.metadata["start_point"] + translate
        ) * scale
        self.metadata["end_point"] = (
            self.metadata["end_point"] + translate
        ) * scale
        # clglogger.debug(f"After {self.metadata}")

    def reverse(self):
        self.metadata["start_point"], self.metadata["end_point"] = (
            self.metadata["end_point"],
            self.metadata["start_point"],
        )

    def merge(self, line: Curve):
        self.metadata["end_point"] = line.metadata["end_point"]

    def draw(self, ax=None, color="black"):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        xdata = [self.metadata["start_point"][0], self.metadata["end_point"][0]]
        ydata = [self.metadata["start_point"][1], self.metadata["end_point"][1]]
        l1 = lines.Line2D(xdata, ydata, lw=1, color=color, axes=ax)
        ax.add_line(l1)
        # ax.plot(self.metadata['start_point'][0], self.metadata['start_point'][1], 'ok', color=color)

    def is_collinear(self, curve: Curve):
        if curve.curve_type == "arc" or curve.curve_type == "circle":
            return False
        else:
            # Calculate the direction vectors of both lines
            direction_self = self.get_point("end_point") - self.get_point("start_point")
            direction_other = curve.get_point("end_point") - curve.get_point(
                "start_point"
            )

            # Normalize the direction vectors
            direction_self_norm = direction_self / np.linalg.norm(direction_self)
            direction_other_norm = direction_other / np.linalg.norm(direction_other)
            # Check if the direction vectors are parallel or anti-parallel
            dot_product = np.dot(direction_self_norm, direction_other_norm)
            if np.isclose(dot_product, 1.0) or np.isclose(dot_product, -1.0):
                return True
            else:
                return False

    def build_body(self, coordsystem=None):
        """
        Requires start point and end point ,transform(only for build type 2)
        """

        assert coordsystem is not None, clglogger.error(
            f"Requires Coordinate system for building {self.curve_type}."
        )
        start_point = create_point_from_array(
            coordsystem.rotate_vec(self.metadata["start_point"])
        )
        end_point = create_point_from_array(
            coordsystem.rotate_vec(self.metadata["end_point"])
        )

        topo_edge = BRepBuilderAPI_MakeEdge(start_point, end_point).Edge()

        return topo_edge

    @property
    def bbox_size(self):
        bbox_size = np.max(np.abs(self.bbox[1] - self.bbox[0]))
        if bbox_size == 0:
            return 1
        else:
            return bbox_size

    @property
    def one_point(self):
        return self.metadata["start_point"]

    def numericalize(self, bit=N_BIT):
        self.is_numerical = True
        self.bit = bit
        size = 2**bit - 1
        # clglogger.debug(f"{self.metadata['start_point']}")
        self.metadata["start_point"] = int_round(
            np.clip(self.metadata["start_point"], a_min=0, a_max=size)
        )
        self.metadata["end_point"] = int_round(
            np.clip(self.metadata["end_point"], a_min=0, a_max=size)
        )

    def denumericalize(self, bit=N_BIT):
        self.is_numerical = False
        self.metadata["start_point"] = dequantize_verts(
            verts=self.metadata["start_point"],
            n_bits=bit,
            min_range=-1,
            max_range=1,
        )
        self.metadata["end_point"] = dequantize_verts(
            verts=self.metadata["end_point"], n_bits=bit, min_range=-1, max_range=1
        )

    def accuracyReport(self, target, tolerance):

        # De-quantize the parameters between (0 and 1) for comparison purposes
        # self.transform(translate=0,scale=1/255)
        # target.transform(translate=0,scale=1/255)
        # print(self.bbox_size)

        self.line_parameter_correct = {"s": np.array([0, 0]), "e": np.array([0, 0])}

        # For Start Point
        self.line_parameter_correct["s"][0] += (
            np.abs(self.metadata["start_point"][0] - target.metadata["start_point"][0])
            / self.bbox_size
        )

        self.line_parameter_correct["s"][1] += (
            np.abs(self.metadata["start_point"][1] - target.metadata["start_point"][1])
            / self.bbox_size
        )

        # For End Point
        self.line_parameter_correct["e"][0] += (
            np.abs(self.metadata["end_point"][0] - target.metadata["end_point"][0])
            / self.bbox_size
        )

        self.line_parameter_correct["e"][1] += (
            np.abs(self.metadata["end_point"][1] - target.metadata["end_point"][1])
            / self.bbox_size
        )

        return self.line_parameter_correct

    def curve_distance(self, pred_curve, scale):
        return super().curve_distance(pred_curve, scale)

    def _json(self):
        line_json = {
            "Start Point": list(float_round(self.metadata["start_point"])),
            "End Point": list(float_round(self.metadata["end_point"]))
        }
        return line_json


if __name__ == "__main__":
    line_dict = {
        "type": "Line3D",
        "start_point": {"y": -0.01, "x": -0.01, "z": 0.0},
        "curve": "JGp",
        "end_point": {"y": -0.04871051, "x": -0.01, "z": 0.0},
    }

    line = Line.from_dict(line_dict)
    print(line)
