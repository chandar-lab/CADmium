import os
import sys
from pathlib import Path

sys.path.append("..")

sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-3]))

import matplotlib.pyplot as plt
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
from OCC.Core.GC import GC_MakeArcOfCircle
from loguru import logger
from rich import print
from CadSeqProc.utility.utils import (
    angle_from_vector_to_x,
    rads_to_degs,
    create_point_from_array,
    quantize,
    dequantize_verts,
    coord_to_pixel,
    create_point,
    get_orientation,
    pixel_to_coord,
    int_round,
    float_round,
    find_arc_geometry,
    point_distance,
)
import torch
import matplotlib.patches as patches
from CadSeqProc.geometry.line import Line
from CadSeqProc.geometry.curve import Curve
from CadSeqProc.utility.macro import *
from CadSeqProc.utility.logger import CLGLogger
import numpy as np


clglogger = CLGLogger().configure_logger().logger


class Arc(Curve):
    def __init__(self, metadata):
        self.metadata = metadata
        self.is_numerical = False
    
    @staticmethod
    def get_reference_vector(start, center):

        dx = round(start[0] - center[0], 6)
        dy = round(start[1] - center[1], 6)
        reference_vector = np.array([dx, dy]) / np.linalg.norm([dx, dy])

        return reference_vector if (dx, dy) != (0, 0) else [1.0, 0.0]

    @staticmethod
    def get_center_three_points(A, M, B):
        """
        pass
        """
        # Extract coordinates
        xA, yA = A
        xB, yB = B
        xM, yM = M
        
        # Compute midpoints
        Px, Py = (xA + xB) / 2, (yA + yB) / 2
        Qx, Qy = (xM + xB) / 2, (yM + yB) / 2
        
        # Compute slopes
        def slope(x1, y1, x2, y2):
            return (y2 - y1) / (x2 - x1) if x2 != x1 else None
        
        mAB = slope(xA, yA, xB, yB)
        mMB = slope(xM, yM, xB, yB)
        
        # Compute perpendicular slopes
        m_perp_AB = -1 / mAB if mAB is not None else 0
        m_perp_MB = -1 / mMB if mMB is not None else 0
        
        # Solve for intersection of perpendicular bisectors
        A_matrix = np.array([
            [-m_perp_AB, 1],
            [-m_perp_MB, 1]
        ])
        B_matrix = np.array([
            Py - m_perp_AB * Px,
            Qy - m_perp_MB * Qx
        ])
        
        # Solve the system of equations Ax = B
        xC, yC = np.linalg.solve(A_matrix, B_matrix)
        
        return np.array((xC, yC))
    
    @staticmethod
    def from_minimal_dict(curve):
        metadata = {}
        metadata["start_point"] = np.array(curve["Start Point"])
        metadata["mid_point"] = np.array(curve["Mid Point"])
        metadata["end_point"] = np.array(curve["End Point"])
        metadata["center"] = Arc.get_center_three_points(metadata["start_point"], 
                                                         metadata["mid_point"],
                                                         metadata["end_point"])
        metadata["radius"] = point_distance(metadata["center"], metadata["start_point"])
        metadata["normal"] = np.array([0, 0, 1])
        metadata["start_angle"] = np.arctan2(metadata["start_point"][1] - metadata["center"][1], metadata["start_point"][0] - metadata["center"][0])
        metadata["end_angle"] = np.arctan2(metadata["end_point"][1] - metadata["center"][1], metadata["end_point"][0] - metadata["center"][0])
        metadata["ref_vec"] = Arc.get_reference_vector(metadata["start_point"], metadata["center"])
        mid_angle = (metadata["start_angle"] + metadata["end_angle"]) / 2
        rot_mat = np.array(
            [
                [np.cos(mid_angle), -np.sin(mid_angle)],
                [np.sin(mid_angle), np.cos(mid_angle)],
            ]
        )
            
        metadata["mid_angle"] = mid_angle
        metadata["rotation_matrix"] = rot_mat

        return Arc(metadata)    

    @staticmethod
    def from_dict(arc_entity: dict):
        metadata = {
            "start_point": np.array(
                [arc_entity["start_point"]["x"], arc_entity["start_point"]["y"]]
            ),
            "end_point": np.array(
                [arc_entity["end_point"]["x"], arc_entity["end_point"]["y"]]
            ),
            "center": np.array(
                [arc_entity["center_point"]["x"], arc_entity["center_point"]["y"]]
            ),
            "radius": arc_entity["radius"],
            "normal": np.array(
                [
                    arc_entity["normal"]["x"],
                    arc_entity["normal"]["y"],
                    arc_entity["normal"]["z"],
                ]
            ),
            "start_angle": arc_entity["start_angle"],
            "end_angle": arc_entity["end_angle"],
            "ref_vec": np.array(
                [
                    arc_entity["reference_vector"]["x"],
                    arc_entity["reference_vector"]["y"],
                ]
            ),
        }

        mid_angle = (metadata["start_angle"] + metadata["end_angle"]) / 2
        rot_mat = np.array(
            [
                [np.cos(mid_angle), -np.sin(mid_angle)],
                [np.sin(mid_angle), np.cos(mid_angle)],
            ]
        )
        mid_point = Arc.get_mid_point_arc(
            metadata["center"], metadata["radius"], metadata["ref_vec"], rot_mat
        )
        metadata["mid_point"] = mid_point
        metadata["rotation_matrix"] = rot_mat
        metadata["mid_angle"] = mid_angle

        return Arc(metadata)

    def to_vec(self):
        """
        vector representation of arc
        """
        assert self.is_numerical is True, clglogger.error(
            "The points are not quantized."
        )
        coord_token = [
            (self.metadata["start_point"] + END_PAD + BOOLEAN_PAD).tolist(),
            (self.metadata["mid_point"] + END_PAD + BOOLEAN_PAD).tolist(),
            [self.token_index, 0],
        ]
        return coord_token

    @staticmethod
    def get_mid_point_arc(center, radius, ref_vec, rot_mat):
        mid_vec = rot_mat @ ref_vec
        return center + mid_vec * radius

    def __repr__(self) -> str:
        if "center" in self.metadata:
            center = self.metadata["center"].round(4)
        else:
            center = "None"
        arc_repr = "{}: Start({}), Mid({}), End({}), Center({}) ".format(
            self.__class__.__name__,
            self.metadata["start_point"].round(4),
            self.metadata["mid_point"].round(4),
            self.metadata["end_point"].round(4),
            center,
        )
        
        return arc_repr


    @property
    def curve_type(self):
        return "arc"

    def reverse(self):
        self.metadata["start_point"], self.metadata["end_point"] = (
            self.metadata["end_point"],
            self.metadata["start_point"],
        )

    def get_point(self, point_type):
        return self.metadata[point_type]
   
    def transform(self, translate, scale):
        self.metadata["start_point"] = (
            self.metadata["start_point"] + translate
        ) * scale
        self.metadata["end_point"] = (
            self.metadata["end_point"] + translate
        ) * scale
        self.metadata["mid_point"] = (
            self.metadata["mid_point"] + translate
        ) * scale
        if "center" in self.metadata:
            self.metadata["center"] = (self.metadata["center"] + translate) * scale

        if "radius" in self.metadata:
            self.metadata["radius"] *= scale

    @staticmethod
    def from_vec(vec, bit=N_BIT, post_processing=False):
        
        metadata = {}

        vec -= END_PAD + BOOLEAN_PAD

        # pixel_to_coord_value=pixel_to_coord(vec,bit=bit).numpy()-(END_PAD+BOOLEAN_PAD)
        metadata["start_point"] = vec[0]
        metadata["mid_point"] = vec[1]
        metadata["end_point"] = vec[2]

        if (
            get_orientation(
                metadata["start_point"],
                metadata["mid_point"],
                metadata["end_point"],
            )
            == "collinear"
        ):
            if post_processing:
                # If three points are collinear,make them a line
                line = Line(metadata=metadata)
                line.quantized_metadata = metadata.copy()
                return line
            else:
                # raise Exception(f"Collinear points {metadata}")
                pass
        arc = Arc(metadata=metadata)
        arc.quantized_metadata = metadata.copy()
        arc.bit = bit
        return arc

    def get_angles_counterclockwise(self, eps=1e-8):
        c2s_vec = (self.metadata["start_point"] - self.metadata["center"]) / (
            np.linalg.norm(self.metadata["start_point"] - self.metadata["center"]) + eps
        )
        c2m_vec = (self.metadata["mid_point"] - self.metadata["center"]) / (
            np.linalg.norm(self.metadata["mid_point"] - self.metadata["center"]) + eps
        )
        c2e_vec = (self.metadata["end_point"] - self.metadata["center"]) / (
            np.linalg.norm(self.metadata["end_point"] - self.metadata["center"]) + eps
        )
        angle_s, angle_m, angle_e = (
            angle_from_vector_to_x(c2s_vec),
            angle_from_vector_to_x(c2m_vec),
            angle_from_vector_to_x(c2e_vec),
        )
        angle_s, angle_e = min(angle_s, angle_e), max(angle_s, angle_e)
        if not angle_s < angle_m < angle_e:
            angle_s, angle_e = angle_e - np.pi * 2, angle_s
        return angle_s, angle_e

    def direction(self, from_start=True):
        if from_start:
            return self.metadata["mid_point"] - self.metadata["start_point"]
        else:
            return self.metadata["end_point"] - self.metadata["mid_point"]

    @property
    def bbox(self):
        points = [
            self.metadata["start_point"],
            self.metadata["mid_point"],
            self.metadata["end_point"],
        ]
        points = np.stack(points, axis=0)
        
        return np.stack([np.min(points, axis=0), np.max(points, axis=0)], axis=0)

    @property
    def bbox_size(self):
        bbox_size = np.max(np.abs(self.bbox[1] - self.bbox[0]))
        if bbox_size == 0:
            return 1
        else:
            return bbox_size

    @property
    def start_point(self):
        return self.metadata["start_point"]

    @property
    def clock_sign(self):
        """get a boolean sign indicating whether the arc is on top of s->e"""
        s2e = self.metadata["end_point"] - self.metadata["start_point"]
        s2m = self.metadata["mid_point"] - self.metadata["start_point"]
        sign = np.cross(s2m, s2e) >= 0  # counter-clockwise
        return sign

    def draw(self, ax=None, color="black"):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10,10))
        ref_vec_angle = rads_to_degs(angle_from_vector_to_x(self.get_point("ref_vec")))
        start_angle = rads_to_degs(self.get_point("start_angle"))
        end_angle = rads_to_degs(self.get_point("end_angle"))
        diameter = 2.0 * self.metadata["radius"]
        ap = patches.Arc(
            (self.metadata["center"][0], self.metadata["center"][1]),
            diameter,
            diameter,
            angle=ref_vec_angle,
            theta1=start_angle,
            theta2=end_angle,
            lw=1,
            color=color,
        )
        ax.add_patch(ap)
        
    def sample_points(self, n_points=32):
        if "center" not in self.metadata.keys():
            center, radius, _, _, _ = find_arc_geometry(
                self.metadata["start_point"],
                self.metadata["mid_point"],
                self.metadata["end_point"],
            )
            self.metadata["center"] = center
            self.metadata["radius"] = radius
        # print(self.metadata)
        c2s_vec = (
            self.metadata["start_point"] - self.metadata["center"]
        ) / np.linalg.norm(self.metadata["start_point"] - self.metadata["center"])
        c2m_vec = (
            self.metadata["mid_point"] - self.metadata["center"]
        ) / np.linalg.norm(self.metadata["mid_point"] - self.metadata["center"])
        c2e_vec = (
            self.metadata["end_point"] - self.metadata["center"]
        ) / np.linalg.norm(self.metadata["end_point"] - self.metadata["center"])
        angle_s, angle_m, angle_e = (
            angle_from_vector_to_x(c2s_vec),
            angle_from_vector_to_x(c2m_vec),
            angle_from_vector_to_x(c2e_vec),
        )
        angle_s, angle_e = min(angle_s, angle_e), max(angle_s, angle_e)
        if not angle_s < angle_m < angle_e:
            angle_s, angle_e = angle_e - np.pi * 2, angle_s

        angles = np.linspace(angle_s, angle_e, num=n_points)
        points = (
            np.stack([np.cos(angles), np.sin(angles)], axis=1) * self.metadata["radius"]
            + self.metadata["center"][np.newaxis]
        )
        
        return points

    def is_collinear(self, curve: Curve):
        return super().is_collinear()

    def build_body(self, coordsystem=None):
        """
        Requires start point, end point and mid point

        """
        assert coordsystem is not None, clglogger.error(
            f"Requires Coordinate system for building {self.curve_type}."
        )
        start_point = create_point_from_array(
            coordsystem.rotate_vec(self.metadata["start_point"])
        )
        mid_point = create_point_from_array(
            coordsystem.rotate_vec(self.metadata["mid_point"])
        )
        end_point = create_point_from_array(
            coordsystem.rotate_vec(self.metadata["end_point"])
        )
        arc_occ = GC_MakeArcOfCircle(start_point, mid_point, end_point).Value()

        topo_edge = BRepBuilderAPI_MakeEdge(arc_occ).Edge()

        return topo_edge

    @property
    def one_point(self):
        return self.metadata["start_point"]

    def numericalize(self, bit=N_BIT):
        self.is_numerical = True
        self.bit = bit
        
        size = 2**bit - 1
        self.metadata["start_point"] = int_round(
            np.clip(self.metadata["start_point"], a_min=0, a_max=size)
        )
        self.metadata["mid_point"] = int_round(
            np.clip(self.metadata["mid_point"], a_min=0, a_max=size)
        )
        self.metadata["end_point"] = int_round(
            np.clip(self.metadata["end_point"], a_min=0, a_max=size)
        )

        # If the quantized values becomes invalid during json processing, perform some changes in the vector.
        # This slight change won't affect the model that much.

        if self.metadata["start_point"][0] == self.metadata["mid_point"][0]:
            if self.metadata["mid_point"][0] < 255:
                self.metadata["mid_point"][0] += 1
            else:
                self.metadata["mid_point"][0] -= 1
        if self.metadata["mid_point"][0] == self.metadata["end_point"][0]:
            if self.metadata["mid_point"][0] < 255:
                self.metadata["mid_point"][0] += 1
            else:
                self.metadata["mid_point"][0] -= 1

        if self.metadata["start_point"][1] == self.metadata["mid_point"][1]:
            if self.metadata["mid_point"][1] < 255:
                self.metadata["mid_point"][1] += 1
            else:
                self.metadata["mid_point"][1] -= 1
        if self.metadata["mid_point"][1] == self.metadata["end_point"][1]:
            if self.metadata["mid_point"][1] < 255:
                self.metadata["mid_point"][1] += 1
            else:
                self.metadata["mid_point"][1] -= 1

    def denumericalize(self, bit=N_BIT):
        self.is_numerical = True
        self.metadata["start_point"] = dequantize_verts(
            verts=self.metadata["start_point"], n_bits=bit, min_range=0, max_range=1
        )
        self.metadata["mid_point"] = dequantize_verts(
            verts=self.metadata["mid_point"], n_bits=bit, min_range=0, max_range=1
        )
        self.metadata["end_point"] = dequantize_verts(
            verts=self.metadata["end_point"], n_bits=bit, min_range=0, max_range=1
        )

    def accuracyReport(self, target, tolerance):
        # # De-quantize the parameters between (0 and 1) for comparison purposes
        # self.transform(translate=0,scale=1/255)
        # target.transform(translate=0,scale=1/255)


        self.arc_parameter_correct={"s":np.array([0,0]),
                                    "m":np.array([0,0]),
                                    "e":np.array([0,0])}
        # For Start Point 
        self.arc_parameter_correct['s'][0] +=  np.abs(self.metadata['start_point'][0]
                                                -target.metadata['start_point'][0])/self.bbox_size
        
        
        self.arc_parameter_correct['s'][1] +=  np.abs(self.metadata['start_point'][1]
                                                -target.metadata['start_point'][1])/self.bbox_size
        
        # For Mid Point 
        self.arc_parameter_correct['m'][0] +=  np.abs(self.metadata['mid_point'][0]
                                                -target.metadata['mid_point'][0])/self.bbox_size
        
        self.arc_parameter_correct['m'][1] +=  np.abs(self.metadata['mid_point'][1]
                                                -target.metadata['mid_point'][1])/self.bbox_size
        
        # For End Point 
        self.arc_parameter_correct['e'][0] +=  np.abs(self.metadata['end_point'][0]
                                                -target.metadata['end_point'][0])/self.bbox_size
        
        self.arc_parameter_correct['e'][1] += np.abs(self.metadata['end_point'][1]
                                                -target.metadata['end_point'][1])/self.bbox_size

        return self.arc_parameter_correct


    def curve_distance(self, pred_curve, scale):
        return super().curve_distance(pred_curve, scale)

    def _json(self):
        if "center" in self.metadata:
            center = self.metadata["center"].round(4)
        else:
            center = "None"
        arc_json = {
            "Start Point": list(float_round(self.metadata["start_point"])),
            "Mid Point": list(float_round(self.metadata["mid_point"])),
            "End Point": list(float_round(self.metadata["end_point"]))
        }

        return arc_json


if __name__ == "__main__":
    arc_dict = {
        "center_point": {"y": -0.00040928, "x": -0.00040928, "z": 0.0},
        "normal": {"y": 0.0, "x": 0.0, "z": 1.0},
        "end_point": {"y": -0.04871051, "x": -0.01, "z": 0.0},
        "start_angle": 0.0,
        "curve": "JGt",
        "end_angle": 1.1787740968698315,
        "radius": 0.0492442,
        "type": "Arc3D",
        "start_point": {"y": -0.01, "x": -0.04871051, "z": 0.0},
        "reference_vector": {
            "y": -0.19475838772210444,
            "x": -0.9808512478515213,
            "z": 0.0,
        },
    }

    arc = Arc.from_dict(arc_dict)
    print(arc._json())
