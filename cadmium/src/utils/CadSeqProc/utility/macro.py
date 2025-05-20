# ---------------------------------------------------------------------------- #
#                               Global Variables                               #
# ---------------------------------------------------------------------------- #


N_BIT=8

END_TOKEN=["PADDING", "START", "END_SKETCH",
                "END_FACE", "END_LOOP", "END_CURVE", "END_EXTRUSION"]

END_PAD=7
BOOLEAN_PAD=4

MAX_CAD_SEQUENCE_LENGTH=272

SKETCH_TOKEN = ["PADDING", "START", "END_SKETCH",
                "END_FACE", "END_LOOP", "END_CURVE", "CURVE"]
EXTRUSION_TOKEN = ["PADDING", "START", "END_EXTRUDE_SKETCH"]

CURVE_TYPE=["Line","Arc","Circle"]

EXTRUDE_OPERATIONS = ["NewBodyFeatureOperation", "JoinFeatureOperation",
                      "CutFeatureOperation", "IntersectFeatureOperation"]


NORM_FACTOR=0.75
EXTRUDE_R=1
SKETCH_R=1

PRECISION = 1e-5
eps = 1e-7


MAX_SKETCH_SEQ_LENGTH = 150
MAX_EXTRUSION = 10
ONE_EXT_SEQ_LENGTH = 10  # Without including start/stop and pad token ((e1,e2),ox,oy,oz,theta,phi,gamma,b,s,END_EXTRUSION) -> 10
VEC_TYPE=2 # Different types of vector representation (Keep only 2)


CAD_CLASS_INFO = {
    'one_hot_size': END_PAD+BOOLEAN_PAD+2**N_BIT,
    'index_size': MAX_EXTRUSION+1, # +1 for padding
    'flag_size': ONE_EXT_SEQ_LENGTH+2 # +2 for sketch and padding
}

