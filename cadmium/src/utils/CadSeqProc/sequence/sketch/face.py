import os, sys

sys.path.append("..")
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-4]))


import numpy as np
from CadSeqProc.utility.logger import CLGLogger
from CadSeqProc.utility.macro import *
from .loop import LoopSequence
from CadSeqProc.utility.utils import (
    random_sample_points,
    perform_op,
    split_array,
    write_stl_file,
)
from loguru import logger
from OCC.Core.BRepCheck import (
    BRepCheck_Analyzer,
    BRepCheck_Result,
    BRepCheck_ListOfStatus,
)
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCC.Core.ShapeFix import ShapeFix_Face
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
import matplotlib.pyplot as plt
from typing import List

clglogger = CLGLogger().configure_logger().logger


class FaceSequence(object):

    def __init__(self, loopdata: List[LoopSequence], reorder: bool = True) -> None:
        self.loopdata = loopdata
        self.quantize_metadata = {}

        if reorder:
            # Reorder Faces according to the minimum bounding box coordinates
            self.reorder()

    @property
    def token_index(self):
        return SKETCH_TOKEN.index("END_FACE")
    
    @staticmethod
    def from_minimal_dict(face):
        loopdata = []
        for lp in face:
            loopdata.append(LoopSequence.from_minimal_dict(face[lp]))
        return FaceSequence(loopdata, True)
    
    @staticmethod
    def from_dict(face_entity: dict, loop_uid: str):
        # Faces consists of One loop
        loopdata = []

        loop_entity = face_entity["profiles"][loop_uid]
        for i, lp in enumerate(loop_entity["loops"]):
            loopdata.append(LoopSequence.from_dict(lp))

        return FaceSequence(loopdata, True)

    def to_vec(self):
        """
        Vector Representation of Face
        """
        coord_token = []
        for lp in self.loopdata:
            vec = lp.to_vec()
            coord_token += vec
        coord_token.append([self.token_index, 0])
        return coord_token

    def reorder(self):
        if len(self.loopdata) <= 1:
            return
        all_loops_bbox_min = np.stack(
            [loop.bbox[0] for loop in self.loopdata], axis=0
        ).round(6)
        ind = np.lexsort(all_loops_bbox_min.transpose()[[1, 0]])
        self.loopdata = [self.loopdata[i] for i in ind]

    @staticmethod
    def from_vec(vec, bit, post_processing, fix_collinearity):
        """
        Vec is the list of loops
        """
        lp = []
        merged_vec = split_array(vec, val=SKETCH_TOKEN.index("END_LOOP"))
        for lp_tokens in merged_vec:
            lp.append(
                LoopSequence.from_vec(
                    vec=lp_tokens,
                    bit=bit,
                    post_processing=post_processing,
                    fix_collinearity=fix_collinearity,
                )
            )
        if len(lp) == 0:
            raise Exception(f"No Loops Added for vec {vec}")
        return FaceSequence(
            loopdata=lp, reorder=False
        )  # No reordering during evaluation

    def __repr__(self):
        s = "Face:"  # Start with bold text for "Loop:"
        for loop in self.loopdata:
            s += f"\n          - {loop.__repr__()}"  # Add the curve representation with blue color

        return s + "\n"


    def transform(self, translate=None, scale=1):
        if translate is None:
            translate = 0
        for loop in self.loopdata:
            loop.transform(translate=translate, scale=scale)

    # @logger.catch()
    def sample_points(self, n_points):
        all_points = []

        for loop in self.loopdata:
            all_points.append(
                loop.sample_points(n_points=n_points)
            )

        all_points = np.vstack(all_points)
        random_points = random_sample_points(all_points, n_points)[0]
        # random_points=all_points
        return random_points

    @property
    def all_curves(self):
        all_curves = []
        for lp in self.loopdata:
            all_curves += lp.all_curves

        return all_curves

    @property
    def start_point(self):
        return self.loopdata[0].start_point

    @property
    def all_loops(self):
        all_loops = []
        for lp in self.loopdata:
            all_loops.append(lp)
        return all_loops

    @property
    def bbox(self):
        all_min_box = []
        all_max_box = []
        for lp in self.loopdata:
            bbox = lp.bbox
            all_min_box.append(bbox[0])
            all_max_box.append(bbox[1])
        return np.array([np.min(all_min_box, axis=0), np.max(all_max_box, axis=0)])

    def build_body(self, plane, normal, coordsystem):
        """
        plane: gp_Pln object. Sketch Plane where a face will be constructed
        normal: gp_Dir object
        transform: gp_Trsf object
        """
        face_list = []
        # plane=self.plane
        # Save all the loop
        for lp in self.loopdata:
            face_builder = BRepBuilderAPI_MakeFace(
                plane,
                lp.build_body(
                    normal=normal, coordsystem=coordsystem
                ),
            )
            if not face_builder.IsDone():
                raise Exception("face builder not done")
            face = face_builder.Face()

            # Fix face
            fixer = ShapeFix_Face(face)
            fixer.SetPrecision(PRECISION)
            fixer.FixOrientation()

            # analyzer = BRepCheck_Analyzer(fixer.Face())
            # if not analyzer.IsValid():
            #     clglogger.error(f"{lp}{normal}{coordsystem}")
            #     raise Exception(f"face check failed.")

            face_list.append(fixer.Face())

        # Find the outer wire (rest becomes inner wires)
        props = GProp_GProps()
        outer_idx = 0
        redo = True
        while redo:
            for f_idx, face in enumerate(face_list):
                # Skip outer face itself
                if f_idx == outer_idx:
                    continue
                # Cut inner face from outer
                cut_face = perform_op(face_list[outer_idx], face, "cut")
                # Compute area, check if inner is larger than outer
                brepgprop.SurfaceProperties(cut_face, props)
                area = props.Mass()
                if area == 0.0:
                    outer_idx = f_idx
                    break
            redo = False

        # Create final closed loop face
        inner_idx = list(set(list(range(0, len(face_list)))) - set([outer_idx]))
        inner_faces = [face_list[i] for i in inner_idx]
        final_face = face_list[outer_idx]
        for face in inner_faces:
            final_face = perform_op(final_face, face, "cut")

        return face_list[0], final_face

    def numericalize(self, bit=N_BIT):
        for lp in self.loopdata:
            lp.numericalize(bit=bit)

    def denumericalize(self, bit):
        for lp in self.loopdata:
            lp.denumericalize(bit=bit)

    def draw(self, ax=None, colors=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
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
        for i, loop in enumerate(self.loopdata):
            loop.draw(ax, colors[i])


    def _json(self):
        face_json = {}
        for i, loop in enumerate(self.loopdata):
            face_json[f"loop_{i+1}"] = loop._json()

        return face_json



if __name__ == "__main__":
    face_dict = {
        "transform": {
            "origin": {"y": 0.0, "x": 0.0, "z": 0.0},
            "y_axis": {"y": 0.0, "x": 0.0, "z": 1.0},
            "x_axis": {"y": 1.0, "x": 0.0, "z": 0.0},
            "z_axis": {"y": 0.0, "x": 1.0, "z": 0.0},
        },
        "type": "Sketch",
        "name": "Sketch 1",
        "profiles": {
            "JGC": {
                "loops": [
                    {
                        "is_outer": True,
                        "profile_curves": [
                            {
                                "center_point": {"y": 0.0762, "x": 0.0, "z": 0.0},
                                "type": "Circle3D",
                                "radius": 0.06000001,
                                "curve": "JGR",
                                "normal": {"y": 0.0, "x": 1.0, "z": 0.0},
                            }
                        ],
                    }
                ],
                "properties": {},
            },
            "JGK": {
                "loops": [
                    {
                        "is_outer": True,
                        "profile_curves": [
                            {
                                "type": "Line3D",
                                "start_point": {"y": 0.3048, "x": 0.3048, "z": 0.0},
                                "curve": "JGB",
                                "end_point": {"y": 0.3048, "x": -0.3048, "z": 0.0},
                            },
                            {
                                "type": "Line3D",
                                "start_point": {"y": 0.3048, "x": -0.3048, "z": 0.0},
                                "curve": "JGN",
                                "end_point": {"y": -0.3048, "x": -0.3048, "z": 0.0},
                            },
                            {
                                "type": "Line3D",
                                "start_point": {"y": -0.3048, "x": 0.3048, "z": 0.0},
                                "curve": "JGF",
                                "end_point": {"y": -0.3048, "x": -0.3048, "z": 0.0},
                            },
                            {
                                "type": "Line3D",
                                "start_point": {"y": 0.3048, "x": 0.3048, "z": 0.0},
                                "curve": "JGJ",
                                "end_point": {"y": -0.3048, "x": 0.3048, "z": 0.0},
                            },
                        ],
                    },
                    {
                        "is_outer": True,
                        "profile_curves": [
                            {
                                "center_point": {"y": 0.0762, "x": 0.0, "z": 0.0},
                                "type": "Circle3D",
                                "radius": 0.06000001,
                                "curve": "JGR",
                                "normal": {"y": 0.0, "x": 1.0, "z": 0.0},
                            }
                        ],
                    },
                    {
                        "is_outer": True,
                        "profile_curves": [
                            {
                                "center_point": {"y": -0.08540001, "x": 0.0, "z": 0.0},
                                "type": "Circle3D",
                                "radius": 0.06000001,
                                "curve": "JGV",
                                "normal": {"y": 0.0, "x": 1.0, "z": 0.0},
                            }
                        ],
                    },
                ],
                "properties": {},
            },
            "JGG": {
                "loops": [
                    {
                        "is_outer": True,
                        "profile_curves": [
                            {
                                "center_point": {"y": -0.08540001, "x": 0.0, "z": 0.0},
                                "type": "Circle3D",
                                "radius": 0.06000001,
                                "curve": "JGV",
                                "normal": {"y": 0.0, "x": 1.0, "z": 0.0},
                            }
                        ],
                    }
                ],
                "properties": {},
            },
        },
        "reference_plane": {},
    }

    face = FaceSequence.from_dict(face_dict, "JGK")
    print(face._json())
    # print(face.all_curves)

    # import open3d as o3d
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)

    # # Save PointCloud object as PLY file
    # o3d.io.write_point_cloud("/home/mkhan/Codes/point2cad/output/output.ply", pcd)
