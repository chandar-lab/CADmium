import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
import bpy
import math
import mathutils


def main():

    fusion_dir = 'data/Fusion360/r1.0.1/reconstruction'
    uids_json = os.listdir('data/fusion360/Fusion360/r1.0.1/reconstruction')

    files = [f for f in os.listdir(fusion_dir) if len(f.split('_')) == 3]
#files = np.array(os.listdir(fusion_dir))
    file_types = np.unique(list(map(lambda x: x.split('.')[-1], files)))
    uids_render = np.unique(list(map(lambda x: x.split('.')[0], files)))

    views = ['top', 'bottom', '000', '001', '002', '003', '004', '005', '006', '007']

    OUTPUT_PARENT_DIR = "data/fusion360_rgb_images/"
    
    for uid in tqdm(uids_render):
        OBJ_PATH = os.path.join(fusion_dir, f'{uid}.obj')
        OUTPUT_DIR = os.path.join(OUTPUT_PARENT_DIR, uid)
        CAMERA_DATA_PATH = os.path.join(OUTPUT_DIR, f'camera_data.json')

        already_procesed = True
        for view in views:
            if not os.path.exists(os.path.join(OUTPUT_DIR, f'{view}.png')):
                already_procesed = False
                break
        if already_procesed:
            print(f"Already processed {uid}")
            continue

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        generate_camera_data(OBJ_PATH, CAMERA_DATA_PATH)
        generate_renders(OBJ_PATH, CAMERA_DATA_PATH, OUTPUT_DIR)

template_camera_data = {
    "universal": {
        "x_fov": 0.8575560450553894,
        "y_fov": 0.8575560450553894,
        "focal_length": 35.0
    },
    "cameras": {
        "top":    {"origin": [0,0,0], "x":[1.3333,0,0],       "y":[0,-1.3333,0],            "z":[0,0,-1.3333]},
        "bottom": {"origin": [0,0,0], "x":[-1.3333,0,0],      "y":[0,-1.3333,0],            "z":[0,0,1.3333]},
        "000":    {"origin": [0,0,0], "x":[0,1.3333,0],       "y":[0.6667,0,-1.1547],       "z":[-1.1547,0,-0.6667]},
        "001":    {"origin": [0,0,0], "x":[-0.9428,0.9428,0], "y":[0.4714,0.4714,-1.1547],  "z":[-0.8165,-0.8165,-0.6667]},
        "002":    {"origin": [0,0,0], "x":[-1.3333,8e-17,0],  "y":[4e-17,0.6667,-1.1547],   "z":[-7e-17,-1.1547,-0.6667]},
        "003":    {"origin": [0,0,0], "x":[-0.9428,-0.9428,0],"y":[-0.4714,0.4714,-1.1547], "z":[0.8165,-0.8165,-0.6667]},
        "004":    {"origin": [0,0,0], "x":[0,-1.3333,0],      "y":[-0.6667,0,-1.1547],      "z":[1.1547,0,-0.6667]},
        "005":    {"origin": [0,0,0], "x":[0.9428,-0.9428,0], "y":[-0.4714,-0.4714,-1.1547],"z":[0.8165,0.8165,-0.6667]},
        "006":    {"origin": [0,0,0], "x":[1.3333,0,0],       "y":[0,-0.6667,-1.1547],      "z":[0,1.1547,-0.6667]},
        "007":    {"origin": [0,0,0], "x":[0.9428,0.9428,0],  "y":[0.4714,-0.4714,-1.1547], "z":[-0.8165,0.8165,-0.6667]},
    }
}

def generate_camera_data(OBJ_PATH, CAMERA_DATA_PATH):
    # Clean scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # Import the OBJ
    bpy.ops.wm.obj_import(filepath=OBJ_PATH)
    obj = bpy.context.selected_objects[0]

    # Recenter at world origin
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY',      center='BOUNDS')
    obj.location = (0, 0, 0)

    # Compute half‐extents from bounding box
    bbox_world = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
    min_corner = mathutils.Vector((
        min(v.x for v in bbox_world),
        min(v.y for v in bbox_world),
        min(v.z for v in bbox_world),
    ))
    max_corner = mathutils.Vector((
        max(v.x for v in bbox_world),
        max(v.y for v in bbox_world),
        max(v.z for v in bbox_world),
    ))
    half_extents = (max_corner - min_corner) / 2.0

    # Compute camera distances
    camera_data = deepcopy(template_camera_data)
    x_fov = camera_data["universal"]["x_fov"]
    y_fov = camera_data["universal"]["y_fov"]
    # distance so that half‐width & half‐depth both fit
    d_top = max(
        half_extents.x / math.tan(x_fov/2),
        half_extents.y / math.tan(y_fov/2),
        half_extents.z
    )
    # We'll use the same radial distance for the 8 side‐views:
    radius = d_top

    # Fill in the origins
    camera_data["cameras"]["top"]["origin"]    = [0.0, 0.0,  d_top]
    camera_data["cameras"]["bottom"]["origin"] = [0.0, 0.0, -d_top]

    # Around views: keys "000" .. "007"
    for i in range(8):
        angle = 2*math.pi * i / 8
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        z = half_extents.z
        key = f"{i:03d}"
        camera_data["cameras"][key]["origin"] = [x, y, z]

    # save the camera data
    with open(CAMERA_DATA_PATH, 'w') as f:
        json.dump(camera_data, f, indent=4)
        print(f"Camera data saved to {CAMERA_DATA_PATH}")


def generate_renders(OBJ_PATH, CAMERA_DATA_PATH, OUTPUT_DIR):
    if 'cycles' not in bpy.context.preferences.addons:
        bpy.ops.preferences.addon_enable(module='cycles')

    # Choose your compute API ('CUDA', 'OPTIX', 'OPENCL' or 'NONE')
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'CUDA'    # or 'OPTIX' if you have an NVIDIA RTX card
    bpy.context.scene.cycles.device = 'GPU'

    offset_distance = 0   # extra shift along camera forward vector
    multiplier      = 1.7  # scale factor on the precomputed origins
    light_energy = 7000

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # Load camera parameters
    with open(CAMERA_DATA_PATH, 'r') as f:
        camera_data = json.load(f)

    univ  = camera_data['universal']
    views = camera_data['cameras']

    # Import the object (OBJ)
    bpy.ops.wm.obj_import(filepath=OBJ_PATH)
    obj = bpy.context.selected_objects[0]

    # Recenter the mesh at world origin
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY',      center='BOUNDS')
    obj.location = (0, 0, 0)
    scene = bpy.context.scene
    cam_target = bpy.data.objects.new("CamTarget", None)
    scene.collection.objects.link(cam_target)
    cam_target.location = (0.0, 0.0, 0.0)

    # Assign a simple grey material
    mat = bpy.data.materials.new(name="GreyMat")
    mat.use_nodes = True

    # Grab the Principled BSDF node
    bsdf = mat.node_tree.nodes.get("Principled BSDF")

    # Set base color
    bsdf.inputs["Base Color"].default_value = (0.5, 0.5, 0.5, 1.0)

    # Increase roughness for a duller, more diffuse look
    bsdf.inputs["Roughness"].default_value = 0.05 # range 0.0 (smooth) to 1.0 (very rough)

    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

    # Render settings
    scene = bpy.context.scene
    scene.render.engine          = 'CYCLES'
    scene.cycles.samples        = 8
    scene.render.film_transparent = True
    scene.render.resolution_x   = 512
    scene.render.resolution_y   = 512

    # turn on nodes for the world
    world = scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes

    # grab the existing Background node (created by default)
    bg = nodes.get("Background")

    # set it to pure white (1,1,1,1) and boost the strength
    bg.inputs["Color"].default_value    = (1.0, 1.0, 1.0, 1.0)
    bg.inputs["Strength"].default_value = 0.5   # ↑ increase until your object is evenly lit

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Loop over each view, place & track camera, render, clean up
    for view_name, params in views.items():
        # 1) create camera
        cam_data = bpy.data.cameras.new(name=f"Cam_{view_name}")
        cam_obj  = bpy.data.objects.new(name=f"Cam_{view_name}", object_data=cam_data)
        scene.collection.objects.link(cam_obj)

        # 2) compute & apply scaled/offset origin
        origin = mathutils.Vector(params['origin']) * multiplier
        forward = -mathutils.Vector(params['z']).normalized()
        origin += forward * offset_distance
        cam_obj.location = origin

        # 3) add a Track To constraint so the camera always points at our empty
        track = cam_obj.constraints.new(type='TRACK_TO')
        track.target     = cam_target
        track.track_axis = 'TRACK_NEGATIVE_Z'  # camera’s -Z points toward target
        track.up_axis    = 'UP_Y'              # Y-up for camera roll

        # 4) set FOV & focal length (universal parameters)
        cam_data.angle_x = univ['x_fov']
        cam_data.angle_y = univ['y_fov']
        cam_data.lens    = univ['focal_length']

        # 5) render
        scene.camera = cam_obj
        scene.render.filepath = os.path.join(OUTPUT_DIR, f"{view_name}.png")
        bpy.ops.render.render(write_still=True)

        # 6) cleanup camera
        bpy.data.objects.remove(cam_obj, do_unlink=True)
        bpy.data.cameras.remove(cam_data,   do_unlink=True)

if __name__ == "__main__":
    main()