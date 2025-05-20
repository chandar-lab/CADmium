import os, sys
from typing import Any

sys.path.append("..")
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-4]))


import numpy as np
from CadSeqProc.utility.logger import CLGLogger
from CadSeqProc.utility.macro import *
from CadSeqProc.utility.utils import dequantize_verts, int_round, quantize, float_round
from loguru import logger
from CadSeqProc.sequence.sketch.coord_system import CoordinateSystem

clglogger = CLGLogger().configure_logger().logger 


class ExtrudeSequence(object):
    """
    Extrusion Sequence for a sketch.

    This class represents an extrusion sequence for a sketch. It stores metadata related to the extrusion operation.
    """

    def __init__(self, metadata: dict, coordsystem: CoordinateSystem = None):
        """
        Initialize the ExtrudeSequence object with the provided metadata.

        Args:
            metadata (dict): Metadata dictionary containing information about the extrusion.
        """
        self.metadata = metadata
        self.quantized_metadata = {}
        self.is_numerical = False
        self.coordsystem = coordsystem

    @staticmethod
    def from_minimal_dict(part):
        metadata = {}

        # Save the uids of the profiles
        metadata["profile_uids"] = [face for face in part["sketch"]]
        metadata["extent_one"] = part['extrusion']['extrude_depth_towards_normal']  #part["extent_one"]
        metadata["extent_two"] = part['extrusion']['extrude_depth_opposite_normal']  #part["extent_two"]
        metadata["boolean"] = EXTRUDE_OPERATIONS.index(part['extrusion']['operation'])  #part["boolean"]

        return ExtrudeSequence(metadata)

    @staticmethod
    def from_dict(all_stat, uid):
        """
        Create an ExtrudeSequence object from a dictionary.

        Args:
            all_stat (dict): Dictionary containing all the extrusion data.
            uid (str): Unique identifier for the extrusion entity.

        Returns:
            ExtrudeSequence: An instance of ExtrudeSequence created from the dictionary.

        Raises:
            AssertionError: If the extrusion entity type is not "ExtrudeFeature" or if the start extent type is not "ProfilePlaneStartDefinition".
        """
        metadata = {}
        extrude_entity = all_stat["entities"][uid]  # Only the extrusion entity

        # Verify the extrusion entity type
        assert extrude_entity["type"] == "ExtrudeFeature", clglogger.critical(
            f"uid {uid} is not extrusion"
        )

        # Verify the start extent type
        assert (
            extrude_entity["start_extent"]["type"] == "ProfilePlaneStartDefinition"
        ), clglogger.critical(f"Error with Extrusion uid {uid}")

        # Save the uids of the profiles
        metadata["profile_uids"] = [
            [profile["sketch"], profile["profile"]]
            for profile in extrude_entity["profiles"]
        ]

        # Extract extent values and boolean operation type
        metadata["extent_one"] = extrude_entity["extent_one"]["distance"][
            "value"
        ]  # Towards the direction of normal

        if extrude_entity["extent_type"] == "SymmetricFeatureExtentType":
            metadata["extent_two"] = metadata["extent_one"] / 2
            metadata["extent_one"] = metadata["extent_one"] / 2
        elif "extent_two" in metadata:
            metadata["extent_two"] = extrude_entity["extent_two"]["distance"][
                "value"
            ]  # Towards the opposite direction of normal
        else:
            metadata["extent_two"] = 0

        if metadata["extent_one"] < 0:
            metadata["extent_two"], metadata["extent_one"] = abs(
                metadata["extent_one"]
            ), abs(metadata["extent_two"])
        metadata["boolean"] = EXTRUDE_OPERATIONS.index(extrude_entity["operation"])
        return ExtrudeSequence(metadata)

    @property
    def token_index(self):
        return END_TOKEN.index("END_EXTRUSION")

    def add_info(self, key, val):
        self.metadata[key] = val

    def transform(self, translate, scale, merge_extent=False):
        # clglogger.debug(f"Extrusion Scale {scale}")
        if not isinstance(translate, int) and not isinstance(translate, float):
            if translate.shape[0] != 3:
                translate = np.concatenate([translate, np.zeros(3 - len(translate))])

        # Extrude Distance Transform
        self.metadata["extent_one"] *= scale
        self.metadata["extent_two"] *= scale

        # Two extents can be changed into one single extent by taking the mean and shifting the sketch position
        if merge_extent:
            self.metadata["extent"] = (
                abs(self.metadata["extent_one"]) + abs(self.metadata["extent_two"])
            ) / 2
            ext_translation = abs(self.metadata["extent"] - self.metadata["extent_one"])
        else:
            ext_translation = 0

        self.coordsystem.transform(translate, scale)  # Plane transformation
        self.metadata["sketch_size"] *= scale  # Sketch Size
        

    def __repr__(self) -> str:
        metadata_str = ", ".join(
            f"{key}: {value}" for key, value in self.metadata.items()
        )

        repr_str = f'{self.__class__.__name__}: ({metadata_str}) Euler Angles {self.coordsystem.metadata["euler_angles"]}'

        return repr_str

    def __setattr__(self, __name: str, __value: Any) -> None:
        super().__setattr__(__name, __value)

    def get_profile_uids(self):
        return self.metadata["profile_uids"]

    def get_total_extent(self, return_quantized=True):
        if hasattr(self, "quantized_metadata") and return_quantized:
            return (
                self.quantized_metadata["extent_one"]
                + self.quantized_metadata["extent_two"]
            )
        else:
            return abs(self.metadata["extent_one"]) + abs(self.metadata["extent_two"])

    def get_boolean(self):
        return self.metadata["boolean"]

    def numericalize(self, bit):
        self.is_numerical = True
        size = 2**bit - 1
        assert (
            -2.0 <= self.metadata["extent_one"] <= 2.0
            and -2.0 <= self.metadata["extent_two"] <= 2.0
        )
        self.metadata["extent_one"] = int_round(
            [
                ((self.metadata["extent_one"] + 1.0) / 2 * (size + 1)).clip(
                    min=0, max=size
                )
            ]
        )[0]
        self.metadata["extent_two"] = int_round(
            [
                ((self.metadata["extent_two"] + 1.0) / 2 * (size + 1)).clip(
                    min=0, max=size
                )
            ]
        )[0]
        self.metadata["boolean"] = int(self.metadata["boolean"])
        self.coordsystem.numericalize(bit)
        self.metadata["sketch_size"] = int_round(
            [(self.metadata["sketch_size"] / 2 * (size + 1)).clip(min=0, max=size)]
        )[0]

        # Due to quantization, small extent values can be quantized to zero so change the values to 1
        if self.metadata["extent_one"] == (2**bit) / 2 and self.metadata[
            "extent_two"
        ] == (2**bit / 2):
            self.metadata["extent_one"] = 1 + ((2**bit) // 2)
        if self.metadata["sketch_size"] == 0:
            self.metadata["sketch_size"] = 1

    def denumericalize(self, bit, post_processing=True, uid=None):

        self.is_numerical = False
        size = 2**bit
        self.metadata["extent_one"] = self.metadata["extent_one"] / size * 2 - 1.0
        self.metadata["extent_two"] = self.metadata["extent_two"] / size * 2 - 1.0
        self.coordsystem.denumericalize(bit)
        # self.metadata['sketch_pos'] = self.metadata['sketch_pos'] / size * 2 - 1.0
        self.metadata["sketch_size"] = self.metadata["sketch_size"] / size * 2
        if post_processing:
            if (
                self.metadata["extent_one"] == 0
                and self.metadata["extent_two"] == 0
            ):  # Post Processing Step
                self.metadata["extent_one"] = 0.01

    def to_vec(self):
        """
        default Value

        END_PAD = 3 # ONE END TOKEN, Start/END SEQ Token and Pad Token
        EXT_OPERATION_PAD=4 # Boolean Operations

        So 0,1,2 are preserved for End tokens.
        """
        assert self.is_numerical is True, clglogger.error("Values are not quantized")
        vec = []
        distance1 = [self.metadata["extent_one"] + END_PAD + BOOLEAN_PAD, 0]
        distance2 = [self.metadata["extent_two"] + END_PAD + BOOLEAN_PAD, 0]
        origin = [
            [i, 0]
            for i in self.coordsystem.metadata["origin"] + END_PAD + BOOLEAN_PAD
        ]
        euler_angles = [
            [i, 0]
            for i in self.coordsystem.metadata["euler_angles"]
            + END_PAD
            + BOOLEAN_PAD
        ]
        boolean = [self.metadata["boolean"] + END_PAD, 0]
        sketch_size = [self.metadata["sketch_size"] + END_PAD + BOOLEAN_PAD, 0]
        token = [self.token_index, 0]

        vec = (
            [distance1]
            + [distance2]
            + origin
            + euler_angles
            + [boolean]
            + [sketch_size]
            + [token]
        )
        # (e1,e2,ox,oy,oz,theta,phi,gamma,b,s,END_EXTRUDE_SKETCH) -> 11
        return vec

    @staticmethod
    def from_vec(vec, bit, post_processing):
        if vec[-1][0] == END_TOKEN.index("END_EXTRUSION"):
            vec = vec[:-1]
        metadata = {}
        metadata["extent_one"] = vec[0][0] - (END_PAD + BOOLEAN_PAD)
        metadata["extent_two"] = vec[1][0] - (END_PAD + BOOLEAN_PAD)
        metadata["boolean"] = vec[-2][0] - (END_PAD)
        metadata["sketch_size"] = vec[-1][0] - (END_PAD + BOOLEAN_PAD)
        coordsystem = CoordinateSystem.from_vec(
            vec[2:8, 0] - (END_PAD + BOOLEAN_PAD), bit, post_processing
        )
        # if post_processing and metadata['extent_one']==0 and metadata['extent_two']==0:
        #     metadata['extent_one']=1

        ext = ExtrudeSequence(metadata=metadata, coordsystem=coordsystem)
        ext.quantized_metadata = metadata.copy()

        return ext

    def _json(self):
        extrude_json = {
            "extrude_depth_towards_normal": float(float_round(self.metadata["extent_one"])),
            "extrude_depth_opposite_normal": float(float_round(self.metadata["extent_two"])),
            "sketch_scale": float(float_round(self.metadata["sketch_size"])),
            "operation": EXTRUDE_OPERATIONS[self.metadata["boolean"]],
        }

        return extrude_json
