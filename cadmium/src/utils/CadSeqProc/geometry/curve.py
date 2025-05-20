
import os
import sys
sys.path.append("..")
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-3])) # ROOT_DIR/CADLGen/DataProc

from abc import ABC, abstractmethod
from loguru import logger
from CadSeqProc.utility.utils import point_distance
from CadSeqProc.utility.macro import *
from CadSeqProc.utility.logger import CLGLogger
import numpy as np



clglogger = CLGLogger().configure_logger().logger


class Curve(ABC):

    def sample_points(self, n_points, sample_type):
        raise NotImplementedError

    def to_vec(self):
        raise NotImplementedError

    @staticmethod
    def from_dict():
        raise NotImplementedError

    @property
    def bbox(self):
        raise NotImplementedError

    def draw(self):
        raise NotImplementedError

    @property
    def curve_type(self):
        return self.__class__.__name__

    @property
    def bbox(self):
        raise NotImplementedError

    def is_collinear(self):
        return False

    def build_body(self):
        raise NotImplementedError

    def numericalize(self):
        raise NotImplementedError

    def denumericalize(self):
        raise NotImplementedError

    @property
    def token_index(self):
        return SKETCH_TOKEN.index("END_CURVE")

    def curve_distance(self, pred_curve, scale):
        return point_distance(self.bbox*scale, pred_curve.bbox*scale, type="l2")
