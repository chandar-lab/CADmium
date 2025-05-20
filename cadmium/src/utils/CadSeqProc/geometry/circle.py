import os, sys

sys.path.append("..")
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-3]))

import numpy as np
from CadSeqProc.utility.logger import CLGLogger
from CadSeqProc.utility.macro import *
from CadSeqProc.utility.utils import (
    create_point_from_array,
    get_plane_normal,
    quantize,
    dequantize_verts,
    point_distance,
    coord_to_pixel,
    create_point,
    pixel_to_coord,
    int_round,
    float_round
)
import torch
from CadSeqProc.geometry.curve import Curve
import matplotlib.patches as patches
from rich import print
from loguru import logger
import matplotlib.pyplot as plt
from OCC.Core.gp import gp_Circ, gp_Ax2, gp_Dir
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge

clglogger = CLGLogger().configure_logger().logger


class Circle(Curve):
    def __init__(self, metadata):
        self.metadata = metadata
        self.is_numerical = False
    
    @staticmethod
    def from_minimal_dict(curve):
        metadata = {
            "center": np.array(curve["Center"]),
            "radius": curve["Radius"],
            "normal": np.array([0, 0, 1]),
        }
        metadata["pt1"] = np.array(
            [metadata["center"][0], metadata["center"][1] + metadata["radius"]]
        )
        metadata["pt2"] = np.array(
            [metadata["center"][0], metadata["center"][1] - metadata["radius"]]
        )           
        metadata["pt3"] = np.array(
            [metadata["center"][0] + metadata["radius"], metadata["center"][1]]
        )
        metadata["pt4"] = np.array(
            [metadata["center"][0] - metadata["radius"], metadata["center"][1]]
        )
        return Circle(metadata)

    @staticmethod
    def from_dict(circle_entity: dict):
        metadata = {
            "center": np.array(
                [circle_entity["center_point"]["x"], circle_entity["center_point"]["y"]]
            ),
            "radius": circle_entity["radius"],
            "normal": np.array(
                [
                    circle_entity["normal"]["x"],
                    circle_entity["normal"]["y"],
                    circle_entity["normal"]["z"],
                ]
            ),
        }

        # Get 4 points on the circle
        metadata["pt1"] = np.array(
            [metadata["center"][0], metadata["center"][1] + metadata["radius"]]
        )
        metadata["pt2"] = np.array(
            [metadata["center"][0], metadata["center"][1] - metadata["radius"]]
        )
        metadata["pt3"] = np.array(
            [metadata["center"][0] + metadata["radius"], metadata["center"][1]]
        )
        metadata["pt4"] = np.array(
            [metadata["center"][0] - metadata["radius"], metadata["center"][1]]
        )

        return Circle(metadata)

    @property
    def bbox(self):
        return np.stack(
            [
                self.metadata["center"] - self.metadata["radius"],
                self.metadata["center"] + self.metadata["radius"],
            ],
            axis=0,
        )
    
    @property
    def bbox_size(self):
        bbox_size = np.max(np.abs(self.bbox[1] - self.bbox[0]))
        if bbox_size == 0:
            return 1
        else:
            return bbox_size

    def direction(self):
        return self.metadata["center"] - self.start_point

    @property
    def start_point(self):
        """Changing start point will change circle quantized values as well as its translation"""
        return self.bbox[0]

    @property
    def end_point(self):
        return np.array(
            [
                self.metadata["center"][0] + self.metadata["radius"],
                self.metadata["center"][1],
            ]
        )

    def to_vec(self):
        """
        vector representation of circle
        """
        assert self.is_numerical is True, clglogger.error(
            "The points are not quantized."
        )
        coord_token = [
            (self.metadata["center"] + END_PAD + BOOLEAN_PAD).tolist(),
            (self.metadata["pt1"] + END_PAD + BOOLEAN_PAD).tolist(),
            [self.token_index, 0],
        ]
        return coord_token

    @staticmethod
    def from_vec(vec, bit, post_processing):
        metadata = {}
        vec -= END_PAD + BOOLEAN_PAD
        metadata["center"] = vec[0]
        metadata["pt1"] = vec[1]
        metadata["radius"] = point_distance(metadata["center"], metadata["pt1"])
        circle = Circle(metadata=metadata)
        circle.quantized_metadata = metadata.copy()
        circle.bit = bit
        return circle

    def sample_points(self, n_points=1024):
        angles = np.linspace(0, np.pi * 2, num=n_points, endpoint=False)
        points = (
            np.stack([np.cos(angles), np.sin(angles)], axis=1) * self.metadata["radius"]
            + self.metadata["center"]
        )
        return points

    def draw(self, ax=None, color="black"):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        ap = patches.Circle(
            (self.metadata["center"][0], self.metadata["center"][1]),
            self.metadata["radius"],
            lw=1,
            fill=None,
            color=color,
        )
        ax.add_patch(ap)
        # ax.plot(self.metadata['center'][0], self.metadata['center'][1], 'ok')

    def __repr__(self) -> str:
        circle_repr = f"{self.__class__.__name__}: center({self.metadata['center'].round(4)}), \
            radius({round(self.metadata['radius'], 4)}), pt1 {self.metadata['pt1']}"
    
        return circle_repr

    @property
    def curve_type(self):
        return "circle"

    def get_point(self, point_type):
        return self.metadata[point_type]
        
    def is_collinear(self, curve: Curve):
        return super().is_collinear()

    def transform(self, translate, scale=1):
        self.metadata["center"] = (self.metadata["center"] + translate) * scale
        self.metadata["pt1"] = (self.metadata["pt1"] + translate) * scale
        if "radius" in self.metadata:
            self.metadata["radius"] *= scale
        else:
            self.metadata["radius"] = abs(
                float(
                    point_distance(
                        self.metadata["center"], self.metadata["pt1"], type="l1"
                    )
                )
            )
            if hasattr(self, "quantized_metadata"):
                self.quantized_metadata["radius"] = int_round(
                    [
                        np.clip(
                            self.metadata["radius"] / scale,
                            a_min=0,
                            a_max=2**self.bit - 1,
                        )
                    ]
                )[0]

    def build_body(self,normal=None, coordsystem=None):
        """
        Requires Center, uppermost point and normal, transform(optional for build_type 2)
        """
        
        assert coordsystem is not None and normal is not None, clglogger.error(
            f"Requires Coordinate System for building {self.curve_type}."
        )

        center = create_point_from_array(
            coordsystem.rotate_vec(self.metadata["center"])
        )
        radius = abs(
            float(
                point_distance(
                    self.metadata["center"], self.metadata["pt1"], type="l1"
                )
            )
        )

        axis = gp_Ax2(center, gp_Dir(*normal))
        circle = gp_Circ(axis, radius)
        topo_edge = BRepBuilderAPI_MakeEdge(circle).Edge()

        return topo_edge

    @property
    def one_point(self):
        return self.metadata["center"]

    def numericalize(self, bit=N_BIT):
        self.is_numerical = True
        self.bit = bit
        size = 2**bit - 1
        self.metadata["pt1"] = int_round(
            np.clip(self.metadata["pt1"], a_min=0, a_max=size)
        )
        self.metadata["pt2"] = int_round(
            np.clip(self.metadata["pt2"], a_min=0, a_max=size)
        )
        self.metadata["pt3"] = int_round(
            np.clip(self.metadata["pt3"], a_min=0, a_max=size)
        )
        self.metadata["pt4"] = int_round(
            np.clip(self.metadata["pt4"], a_min=0, a_max=size)
        )
        self.metadata["center"] = int_round(
            np.clip(self.metadata["center"], a_min=0, a_max=size)
        )
        self.metadata["radius"] = int_round(
            [np.clip(self.metadata["radius"], a_min=0, a_max=size)]
        )[0]

        if self.metadata["pt1"][1] == self.metadata["center"][1]:
            if self.metadata["pt1"][1] < 255:
                self.metadata["pt1"][1] += 1
            else:
                self.metadata["pt1"][1] -= 1

    def denumericalize(self, bit=N_BIT):
        self.is_numerical = False
        
        self.metadata["pt1"] = dequantize_verts(
            verts=self.metadata["pt1"], n_bits=bit, min_range=-1, max_range=1
        )
        self.metadata["pt2"] = dequantize_verts(
            verts=self.metadata["pt2"], n_bits=bit, min_range=-1, max_range=1
        )
        self.metadata["pt3"] = dequantize_verts(
            verts=self.metadata["pt3"], n_bits=bit, min_range=-1, max_range=1
        )
        self.metadata["pt4"] = dequantize_verts(
            verts=self.metadata["pt4"], n_bits=bit, min_range=-1, max_range=1
        )
        self.metadata["center"] = dequantize_verts(
            verts=self.metadata["center"], n_bits=bit, min_range=-1, max_range=1
        )
        self.metadata["radius"] = dequantize_verts(
            verts=self.metadata["radius"], n_bits=bit, min_range=-1, max_range=1
        )

    def accuracyReport(self, target, tolerance):

        # De-quantize the parameters between (0 and 1) for comparison purposes
        # self.transform(translate=0,scale=1/255)
        # target.transform(translate=0,scale=1/255)

        self.circle_parameter_report = {"c": np.array([0, 0]), "r": np.array([0, 0])}

        self.circle_parameter_report["c"][0] += (
            np.abs(self.metadata["center"][0] - target.metadata["center"][0])
            / self.bbox_size
        )

        self.circle_parameter_report["c"][1] += (
            np.abs(self.metadata["center"][1] - target.metadata["center"][1])
            / self.bbox_size
        )

        self.circle_parameter_report["r"][0] += (
            np.abs(self.metadata["radius"] - target.metadata["radius"]) / self.bbox_size
        )
        self.circle_parameter_report["r"][
            1
        ] += 1  # Number of radius considered (not used anymore. dummy value)

        return self.circle_parameter_report

    def curve_distance(self, pred_curve, scale):
        return super().curve_distance(pred_curve, scale)

    def _json(self):
        circle_json = {
            "Center": list(float_round(self.metadata["center"])),
            "Radius": float(float_round(self.metadata["radius"]))
        }
        return circle_json


if __name__ == "__main__":
    circle_dict = {
        "center_point": {"y": 0.0762, "x": 0.0, "z": 0.0},
        "type": "Circle3D",
        "radius": 0.06000001,
        "curve": "JGR",
        "normal": {"y": 0.0, "x": 1.0, "z": 0.0},
    }
    circle = Circle.from_dict(circle_dict)
    print(circle)
