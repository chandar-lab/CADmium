import os, sys
from typing import List

sys.path.append("..")
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-4]))

import numpy as np
from CadSeqProc.utility.logger import CLGLogger
from CadSeqProc.utility.macro import *
from CadSeqProc.geometry.curve import Curve
from CadSeqProc.geometry.line import Line
from CadSeqProc.geometry.arc import Arc
from CadSeqProc.geometry.circle import Circle
from CadSeqProc.utility.utils import (
    get_orientation,
    merge_list,
    flatten,
    random_sample_points,
    merge_end_tokens_from_loop,
    write_stl_file,
    write_ply,
    point_distance,
    create_matched_pair,
)
from rich import print
from loguru import logger
import matplotlib.pyplot as plt
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeWire
from OCC.Core.ShapeFix import ShapeFix_Wire
from OCC.Extend.DataExchange import write_step_file
from scipy.optimize import linear_sum_assignment

clglogger = CLGLogger().configure_logger().logger


class LoopSequence(object):

    def __init__(
        self,
        curvedata: List[Curve],
        is_outer=False,
        post_processing=True,
        fix_collinearity=False,
    ) -> None:
        self.curvedata = curvedata
        self.is_outer = is_outer
        self.collinear_curves = []

        # if post_processing:
        # Reorder the loop to fix connectivity, orientation and collinearity
        self.reorder(orientation_fix=True, collinearity_fix=fix_collinearity)

        # If post processing is set on, fix the line start and end point error
        n_curve = len(self.curvedata)

        if post_processing:
            for i, cv in enumerate(self.curvedata):
                if n_curve == 1:
                    continue
                else:
                    if cv.curve_type == "line":
                        if (
                            np.sum(
                                cv.metadata["start_point"] - cv.metadata["end_point"]
                            )
                            == 0
                        ):
                            cv.metadata["end_point"] += 1
                            self.curvedata[(i + 1) % n_curve].metadata[
                                "start_point"
                            ] = cv.metadata["end_point"]

    @property
    def token_index(self):
        return SKETCH_TOKEN.index("END_LOOP")
    
    @staticmethod
    def from_minimal_dict(loop):
        curvedata = []
        for curve in loop:
            if 'line' in curve:
                curvedata.append(Line.from_minimal_dict(loop[curve]))
            if 'arc' in curve:
                curvedata.append(Arc.from_minimal_dict(loop[curve]))
            if 'circle' in curve:
                curvedata.append(Circle.from_minimal_dict(loop[curve]))
        return LoopSequence(curvedata, True, False)

    @staticmethod
    def from_dict(loop_entity: dict):
        is_outer = loop_entity["is_outer"]
        curvedata = []
        curves = loop_entity["profile_curves"]
        for i in range(len(curves)):
            curve_type = curves[i]["type"]

            if curve_type == "Line3D":
                curvedata.append(Line.from_dict(curves[i]))
            if curve_type == "Arc3D":
                curvedata.append(Arc.from_dict(curves[i]))
            if curve_type == "Circle3D":
                curvedata.append(Circle.from_dict(curves[i]))

        return LoopSequence(curvedata, is_outer, False)

    @property
    def start_point(self):
        # if len(self.curvedata)>1:
        #     return self.curvedata[0].start_point
        # else:
        return self.curvedata[0].start_point

    @property
    def bbox(self):
        if len(self.curvedata) <= 1:
            return self.curvedata[0].bbox
        else:
            all_min_box = []
            all_max_box = []
            for curve in self.curvedata:
                if curve is not None:
                    bbox = curve.bbox
                    all_min_box.append(bbox[0])
                    all_max_box.append(bbox[1])
        return np.array([np.min(all_min_box, axis=0), np.max(all_max_box, axis=0)])

    def to_vec(self):
        """
        Vector Representation of Loop
        """
        coord_token = []
        for cv in self.curvedata:
            vec = cv.to_vec()
            coord_token += vec
        coord_token.append([self.token_index, 0])
        return coord_token

    @staticmethod
    def from_vec(vec, bit, post_processing, fix_collinearity):
        """
        Vec is the list of curves
        """
        cv = []
        merged_vec = merge_end_tokens_from_loop(vec)[0]
        for cv_tokens in merged_vec:
            if len(merged_vec) == 1:
                cv.append(
                    Circle.from_vec(
                        vec=cv_tokens,
                        bit=bit,
                        post_processing=post_processing,
                    )
                )
            elif len(cv_tokens) == 2:
                cv.append(
                    Line.from_vec(
                        vec=cv_tokens,
                        bit=bit,
                        post_processing=post_processing,
                    )
                )
            elif len(cv_tokens) == 3:
                cv.append(
                    Arc.from_vec(
                        vec=cv_tokens,
                        bit=bit,
                        post_processing=post_processing,
                    )
                )
            else:
                raise ValueError(f"Invalid Curve Tokens {cv_tokens}")
        if len(cv) == 0:
            raise Exception(f"No Curves Added for vec {vec}")
        return LoopSequence(
            curvedata=cv,
            post_processing=post_processing,
            fix_collinearity=fix_collinearity,
        )

    @property
    def direction(self):
        first_curve = self.curvedata[0]
        if first_curve.curve_type == "circle":
            try:
                return get_orientation(
                    first_curve.get_point("pt1"),
                    first_curve.get_point("pt2"),
                    first_curve.get_point("pt3"),
                )
            except:
                return "counterclockwise"
        else:
            return get_orientation(
                first_curve.get_point("start_point"),
                first_curve.get_point("end_point"),
                self.curvedata[1].get_point("end_point"),
            )

    @staticmethod
    def is_connected(curvedata: List[Curve]):
        """
        Check if curve is connected
        """
        n = len(curvedata)
        if n == 1:
            return True

        for i, curve in enumerate(curvedata):
            if (
                i > 0
                and i < n - 1
                and not np.allclose(
                    curvedata[i - 1].get_point("end_point"),
                    curve.get_point("start_point"),
                )
            ):
                clglogger.critical(
                    f"Curve is not connected {curvedata} at {curvedata[i-1],curve}"
                )
                return False
            elif i == n - 1 and not np.allclose(
                curvedata[0].get_point("start_point"), curve.get_point("end_point")
            ):
                clglogger.critical(
                    f"Curve is not connected {curvedata} at {curve,curvedata[0]}"
                )
                return False
        clglogger.success("Curve is connected")
        return True

    @staticmethod
    def ensure_connectivity(curvedata: List[Curve], verbose=False):
        """
        Create a connected loop from the existing curves
        """
        if len(curvedata) <= 1:
            return curvedata

        new_curvedata = [curvedata[0]]

        n = len(curvedata)
        for i, curve in enumerate(curvedata):
            if i > 0:
                if i < n - 1 and np.allclose(
                    new_curvedata[-1].get_point("end_point"),
                    curve.get_point("end_point"),
                ):
                    curve.reverse()
                    new_curvedata.append(curve)
                elif (
                    i == n - 1
                    and np.allclose(
                        new_curvedata[-1].get_point("end_point"),
                        curve.get_point("end_point"),
                    )
                    or np.allclose(
                        curve.get_point("start_point"),
                        new_curvedata[0].get_point("start_point"),
                    )
                ):
                    curve.reverse()
                    new_curvedata.append(curve)
                else:
                    new_curvedata.append(curve)
        if verbose:
            LoopSequence.is_connected(curvedata)

        return new_curvedata

    def reorder(self, orientation_fix=True, collinearity_fix=True):
        """reorder by starting left most and counter-clockwise. Fix Collinearity if exists by merging (for lines only)"""
        if len(self.curvedata) <= 1:
            return

        start_curve_idx = -1
        sx, sy = 10000, 10000
        total_curve = len(self.curvedata)

        self.curvedata = LoopSequence.ensure_connectivity(
            self.curvedata, verbose=False
        )  # Connected Loop
        # LoopSequence.is_connected(self.curvedata) # Check if the loop is connected

        # correct start-end point order and find left-most point
        for i, curve in enumerate(self.curvedata):
            if round(curve.get_point("start_point")[0], 6) < round(sx, 6) or (
                round(curve.get_point("start_point")[0], 6) == round(sx, 6)
                and round(curve.get_point("start_point")[1], 6) < round(sy, 6)
            ):
                start_curve_idx = i
                sx, sy = curve.get_point("start_point")

        self.curvedata = (
            self.curvedata[start_curve_idx:] + self.curvedata[:start_curve_idx]
        )

        # Fix Orientation so that loop is created counter-clockwise
        if (
            self.direction == "clockwise"
            and orientation_fix
            and len(self.curvedata) > 1
        ):
            self.curvedata = self.curvedata[::-1]

            for i in range(len(self.curvedata)):
                self.curvedata[i].reverse()

        # Fix Collinearity
        if len(self.curvedata) > 1 and collinearity_fix:
            collinear_pair = []
            for i in range(len(self.curvedata) - 1):
                if self.curvedata[i].is_collinear(self.curvedata[i + 1]):
                    collinear_pair.append([i, i + 1])
                    self.collinear_curves.append(
                        [self.curvedata[i], self.curvedata[i + 1]]
                    )
                else:
                    collinear_pair.append([i])
            if len(self.curvedata) - 1 not in flatten(collinear_pair):
                collinear_pair.append([len(self.curvedata) - 1])
            collinear_pair = merge_list(collinear_pair)
            self.new_curvedata = []

            for p in collinear_pair:
                if len(p) == 1:
                    self.new_curvedata.append(self.curvedata[p[0]])
                else:
                    curve = self.curvedata[p[0]]
                    curve.merge(self.curvedata[p[-1]])
                    self.new_curvedata.append(curve)

            self.curvedata = self.new_curvedata

    @property
    def all_curves(self):
        return self.curvedata

    def transform3D(self, coordsystem):
        for curve in self.curvedata:
            curve.transform3D(coordsystem=coordsystem)

    def transform(self, translate=None, scale=1):
        if translate is None:
            translate = 0
        for curve in self.curvedata:
            curve.transform(translate=translate, scale=scale)

    def __repr__(self):
        s = f"Loop: Start Point: {list(np.round(self.start_point,4))}, Direction: {self.direction}"  # bbox {list(np.round(self.bbox,4))}"  # Start with bold text for "Loop:"
        for curve in self.curvedata:
            s += f"\n              - {curve.__repr__()}"  # Add the curve representation with blue color
        return s + "\n"

    def add_info(self, key_, val_):
        self.curvedata[key_] = val_

    def sample_points(self, n_points):
        all_points = []

        for curve in self.curvedata:
            all_points.append(
                curve.sample_points(n_points=n_points)
            )
        all_points = np.vstack(all_points)
        random_points = random_sample_points(all_points, n_points)[0]
        # random_points=all_points
        return random_points

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
        for i, curve in enumerate(self.curvedata):
            curve.draw(ax, colors[i])

    def build_body(self, normal, coordsystem):
        topo_wire = BRepBuilderAPI_MakeWire()
        for cv in self.curvedata:
            if cv.curve_type.lower() == "circle":
                topo_wire.Add(
                    cv.build_body(
                        normal=normal, coordsystem=coordsystem
                    )
                )
            else:
                topo_wire.Add(
                    cv.build_body(coordsystem=coordsystem)
                )
            if not topo_wire.IsDone():
                raise Exception("wire builder not done")

        fixer = ShapeFix_Wire()
        fixer.Load(topo_wire.Wire())
        fixer.SetPrecision(PRECISION)
        fixer.FixClosed()
        fixer.Perform()

        return fixer.Wire()

    def numericalize(self, bit=N_BIT):
        for cv in self.curvedata:
            cv.numericalize(bit=bit)

        # Fix Invalidity
        # for i,cv in enumerate(self.curvedata[:-1]):
        #     if cv.curve_type.lower()=="line":
        #         if np.sum(cv.metadata['start_point']-cv.metadata['end_point'])==0:
        #             cv.metadata['end_point']+=1
        #             self.curvedata[i+1].metadata['start_point']=cv.metadata['end_point']

    def denumericalize(self, bit):
        for cv in self.curvedata:
            cv.denumericalize(bit=bit)

    def loop_distance(self, target_loop, scale: float):
        return point_distance(self.bbox * scale, target_loop.bbox * scale, type="l2")

    @staticmethod
    def match_primitives(gt_loop, pred_loop, scale: float, multiplier: int = 1):
        """
        Match primitives (e.g., curves) based on their bounding box distances.

        Args:
            gt_loop (object): Ground truth loop object.
            pred_loop (object): Predicted loop object.
            scale (float): The scaling factor.
            multiplier (int, optional): Multiplier for cost matrix. Defaults to 1.

        Returns:
            list: List containing matched ground truth and predicted curves.
        """
        if gt_loop is None:
            gt_curves = [None]
        else:
            gt_curves = gt_loop.all_curves

        if pred_loop is None:
            pred_curves = [None]
        else:
            pred_curves = pred_loop.all_curves

        n_gt = len(gt_curves)
        n_pred = len(pred_curves)
        n_max = max(n_gt, n_pred)

        # Initialize cost matrix with ones and apply multiplier
        cost_matrix = np.ones((n_max, n_max)) * multiplier

        # Pad lists with None if needed
        if n_gt < n_max:
            gt_curves += [None] * (n_max - n_gt)

        if n_pred < n_max:
            pred_curves += [None] * (n_max - n_pred)

        # Calculate Cost by calculating the distance between loops
        for ind_self in range(n_gt):
            for ind_pred in range(n_pred):
                if (
                    gt_curves[ind_self] is not None
                    and pred_curves[ind_pred] is not None
                ):
                    cost_matrix[ind_self, ind_pred] = gt_curves[
                        ind_self
                    ].curve_distance(pred_curves[ind_pred], scale)

        # Use Hungarian matching to find the best matching
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # row_indices=list(row_indices)
        # col_indices=list(col_indices)
        # print(row_indices, col_indices)

        # Create a new pair with matched ground truth and predicted curves
        new_pair = create_matched_pair(
            list1=gt_curves,
            list2=pred_curves,
            row_indices=row_indices,
            col_indices=col_indices,
        )
        return new_pair

    def _json(self):
        loop_json = {}
        curve_num_dict = {"line": 1, "arc": 1, "circle": 1}
        for i, curve in enumerate(self.curvedata):
            loop_json[f"{curve.curve_type}_{curve_num_dict[curve.curve_type]}"] = (
                curve._json()
            )
            curve_num_dict[curve.curve_type] += 1
        return loop_json


if __name__ == "__main__":
    loop_dict = {
        "is_outer": True,
        "profile_curves": [
            {
                "type": "Line3D",
                "start_point": {"y": 0.3048, "x": 0.3048, "z": 0.0},
                "curve": "JGB",
                "end_point": {"x": 0.166, "y": 0.3048, "z": 0.0},
            },
            {
                "type": "Line3D",
                "start_point": {"x": 0.166, "y": 0.3048, "z": 0.0},
                "curve": "JGB",
                "end_point": {"x": -0.3048, "y": 0.3048, "z": 0.0},
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
    }

    loop = LoopSequence.from_dict(loop_dict)
    loop.reorder()
    print(loop._json())
