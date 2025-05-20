import re

# ------------------------------ Prompt Design ------------------------------

JSON_SCHEMA = """
{
  "parts": {
    "part_1": {  // Always use sequential part_1, part_2... even if some are null
      "coordinate_system": {
        "Euler Angles": [0.0, 0.0, 0.0],  // XYZ rotation angles in degrees
        "Translation Vector": [0.0, 0.0, 0.0]  // X,Y,Z position offsets
      },
      "description": {
        "height": 0.0,  // Total vertical dimension
        "length": 0.0,  // Total horizontal dimension
        "name": "",     // (optional) Component identifier
        "shape": "",    // (optional) Basic geometric classification
        "width": 0.0    // Total depth dimension
      },
      "extrusion": {
        "extrude_depth_opposite_normal": 0.0,  // Negative direction extrusion
        "extrude_depth_towards_normal": 0.0,   // Positive direction extrusion
        "operation": "NewBodyFeatureOperation", // One of: NewBodyFeatureOperation, JoinFeatureOperation, CutFeatureOperation, IntersectFeatureOperation
        "sketch_scale": 0.0  // Scaling factor for sketch geometry
      },
      "sketch": {
        "face_1": {  // Use sequential face_1, face_2... (null if unused)
          "loop_1": {  // Use sequential loop_1, loop_2... (null if unused)
            "circle_1": {  // Use sequential circle_1, circle_2...
              "Center": [0.0, 0.0],  // X,Y coordinates
              "Radius": 0.0
            },
            "arc_1": {  // Use sequential arc_1, arc_2...
              "Start Point": [0.0, 0.0],
              "End Point": [0.0, 0.0],
              "Mid Point": [0.0, 0.0]
            },
            "line_1": {  // Use sequential line_1, line_2...
              "Start Point": [0.0, 0.0],
              "End Point": [0.0, 0.0]
            }
            // ... (other geometric elements as null/none)
          }
          // ... (other loops as null/none)
        }
        // ... (other faces as null/none)
      }
    },
    "part_2": null,  // Maintain sequential numbering even for null parts
    // ... (additional parts)
  }
}
"""

# Imperative tone + Schema first 
SYSTEM_MESSAGE_TEMPLATE = """
Generate CAD model JSON EXACTLY matching this schema:
{JSON_SCHEMA}

STRICT RULES:
- OUTPUT ONLY RAW JSON (no formatting/text/comments/explanations)
- NEVER COPY INSTRUCTIONAL TEXT FROM JSON SCHEMA EXAMPLES
- ALL numbers as floats (0.0 not 0)
- ALLOWED OPERATIONS: NewBodyFeatureOperation/JoinFeatureOperation/CutFeatureOperation/IntersectFeatureOperation 
- GEOMETRY REQUIREMENTS (these are the only available primitives):
  • Circles: Center[X,Y] + Radius
  • Arcs: Start[X,Y] + End[X,Y] + Mid[X,Y]
  • Lines: Start[X,Y] + End[X,Y]
- ENFORCE part_1, part_2... sequence (include nulls)
- NO NEW FIELDS
"""

SYSTEM_MESSAGE = SYSTEM_MESSAGE_TEMPLATE.format(JSON_SCHEMA=re.sub('\s+',' ', JSON_SCHEMA))
