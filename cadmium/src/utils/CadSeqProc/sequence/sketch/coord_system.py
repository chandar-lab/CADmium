import os, sys

sys.path.append("..")
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-4]))


import numpy as np
from CadSeqProc.utility.logger import CLGLogger
from CadSeqProc.utility.utils import (
    float_round,
    int_round,
    polar_parameterization,
    quantize,
)
from CadSeqProc.utility.macro import *
from loguru import logger
from OCC.Core.gp import gp_Trsf, gp_Vec, gp_Ax3, gp_Dir, gp_Ax1, gp_Pnt
from rich import print
from scipy.spatial.transform import Rotation as R
import math

clglogger = CLGLogger().configure_logger().logger


class CoordinateSystem(object):

    def __init__(self, metadata) -> None:
        self.metadata = metadata
        self.is_numerical = False

    @staticmethod
    def from_dict(transform_dict):
        metadata = {}

        metadata["origin"] = np.array(
            [
                transform_dict["origin"]["x"],
                transform_dict["origin"]["y"],
                transform_dict["origin"]["z"],
            ]
        )
        metadata["x_axis"] = np.array(
            [
                transform_dict["x_axis"]["x"],
                transform_dict["x_axis"]["y"],
                transform_dict["x_axis"]["z"],
            ]
        )
        metadata["y_axis"] = np.array(
            [
                transform_dict["y_axis"]["x"],
                transform_dict["y_axis"]["y"],
                transform_dict["y_axis"]["z"],
            ]
        )
        metadata["z_axis"] = np.array(
            [
                transform_dict["z_axis"]["x"],
                transform_dict["z_axis"]["y"],
                transform_dict["z_axis"]["z"],
            ]
        )

        # theta,phi,gamma=polar_parameterization(metadata['z_axis'],metadata['x_axis'])
        euler_angles = R.from_matrix(
            np.vstack((metadata["x_axis"], metadata["y_axis"], metadata["z_axis"]))
        ).as_euler("zyx", degrees=False)

        metadata["euler_angles"] = euler_angles

        coord = CoordinateSystem(metadata=metadata)

        return coord

    @staticmethod
    def from_vec(vec, bit, post_processing):
        assert len(vec) == 6, clglogger.error(f"Wrong number of inputs {vec}")
        metadata = {}
        metadata["origin"] = vec[:3]
        metadata["euler_angles"] = vec[3:6]
        coord = CoordinateSystem(metadata=metadata)
        coord.quantized_metadata = metadata.copy()

        return coord

    def create_transform(self):
        """
        Requires Origin and the z-axis and gamma angle

        """
        transform = gp_Trsf()
        transform.SetTranslation(gp_Vec(*self.metadata["origin"]))

        # Calculate the rotation angle and axis
        rotation_axis = gp_Ax1(
            gp_Pnt(*self.metadata["origin"]),
            gp_Dir(*self.metadata["z_axis"].astype(np.float64)),
        )
        rotation_angle = self.metadata["euler_angles"][2]

        # Set the rotation using the calculated angle and axis
        transform.SetRotation(rotation_axis, rotation_angle)

        return transform

    @property
    def normal(self):
        return self.metadata["z_axis"]

    def get_property(self, key):
        return self.metadata[key]

    def rotate_vec(self, vec, translation=True):
        if vec.shape[-1] == 2:
            if len(vec.shape) == 1:
                vec = np.concatenate([vec, np.zeros(1)])
            else:
                vec = np.hstack([vec, np.zeros((len(vec), 1))])

        # Create a rotation matrix using the axes from metadata
        rotation_matrix = np.column_stack(
            (self.metadata["x_axis"], self.metadata["y_axis"], self.metadata["z_axis"])
        ).T

        # Rotate the vector
        rotated_vector = vec @ rotation_matrix

        rotated_vector = rotated_vector

        if translation:
            return rotated_vector + self.metadata["origin"]
        else:
            return rotated_vector

    def __repr__(self) -> str:
        try:
            rotation_matrix = (
                [*self.metadata["x_axis"]]
                + [*self.metadata["y_axis"]]
                + [*self.metadata["z_axis"]]
            )
        except:
            rotation_matrix = None
        s = f"{self.__class__.__name__}:\n            - Rotation Matrix {rotation_matrix},\n            - Translation {self.metadata['origin']}"

        return s

    def transform(self, translate, scale):
        if not isinstance(translate, int) and not isinstance(translate, float):
            if translate.shape[0] != 3:
                translate = np.concatenate([translate, np.zeros(3 - len(translate))])
        self.metadata["origin"] = (self.metadata["origin"] + translate) * scale

    def numericalize(self, bit: int):
        """
        Quantization
        """
        self.is_numerical = True
        size = 2**bit - 1
        self.metadata["origin"] = int_round(
            ((self.metadata["origin"] + 1.0) / 2 * (size + 1)).clip(min=0, max=size)
        )
        self.metadata["euler_angles"] = int_round(
            ((self.metadata["euler_angles"] / np.pi + 1.0) / 2 * (size + 1)).clip(
                min=0, max=size
            )
        )

    def denumericalize(self, bit):
        """
        Dequantization
        """

        self.is_numerical = False
        size = 2**bit
        self.metadata["origin"] = self.metadata["origin"] / size * 2 - 1.0
        self.metadata["euler_angles"] = (
            self.metadata["euler_angles"] / size * 2 - 1.0
        ) * np.pi

        rot_matrix = R.from_euler(
            seq="zyx", angles=self.metadata["euler_angles"], degrees=False
        ).as_matrix()

        # x_axis,y_axis,z_axis=euler_to_axis(*self.metadata['euler_angles'])
        self.metadata["x_axis"] = rot_matrix[0]
        self.metadata["y_axis"] = rot_matrix[1]
        self.metadata["z_axis"] = rot_matrix[2]


        self.is_numerical = False

    def _json(self):
        return {
            "Euler Angles": [
                float(float_round(math.degrees(r_val)))
                for r_val in self.metadata["euler_angles"]
            ],
            "Translation Vector": list(float_round(self.metadata["origin"]))
        }


if __name__ == "__main__":
    transform_dict = {
        "origin": {"y": 0.0, "x": 0.0, "z": 0.0},
        "y_axis": {"y": 0.0, "x": 0.0, "z": 1.0},
        "x_axis": {"y": 0.0, "x": 1.0, "z": 0.0},
        "z_axis": {"y": -1.0, "x": 0.0, "z": 0.0},
    }

    cd = CoordinateSystem.from_dict(transform_dict)
    print(cd._json())
