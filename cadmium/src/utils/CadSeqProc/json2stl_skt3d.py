import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append("..")
from CadSeqProc.utility.logger import CLGLogger
from CadSeqProc.utility.macro import *
from cad_sequence import CADSequence
import argparse
import json
import open3d as o3d
import torch
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
  # ---------------------------------------------------------------------------- #
  #            DeepCAD Original Json or Vec to Mesh + 3D Sketch Points           #
  # ---------------------------------------------------------------------------- #

clglogger=CLGLogger().configure_logger(verbose=True).logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--input_dir", help="Input Directory for DeepCAD Json dataset", type=str
    )
    parser.add_argument("--split_json", help="Train-test-validation split", type=str)
    parser.add_argument("-o", "--output_dir", type=str)
    parser.add_argument("--num_sketch_points", type=int, default=30000)
    parser.add_argument("--max_workers", type=int, default=16)
    args = parser.parse_args()
    
    # print(args)
    clglogger.info(f"Running Task with {args.max_workers} workers.")

    # all_json_uids = sorted(get_files_scan(args.input_dir, max_workers=args.max_workers))

    with open(args.split_json, "r") as f:
        data = json.load(f)

    all_json_uids = (
        data["train"][:82000]
        + data["train"][84000:]
        + data["test"]
        + data["validation"]
    ) # Possible corruption in data causing ProcessPool to fail
    
    all_json_uids=sorted(all_json_uids)

    clglogger.info(f"Found {len(all_json_uids)} files.")
    clglogger.info(f"Saving Sequence in {args.output_dir}.")

    # --------------------------------- Method 1 --------------------------------- #
    executor = ProcessPoolExecutor(max_workers=args.max_workers)
    # Submit tasks to the executor
    futures = [
        executor.submit(
            process_one, os.path.join(args.input_dir, json_uid + ".json"), args
        )
        for json_uid in tqdm(all_json_uids, desc="Submitting Tasks")
    ]

    for future in tqdm(as_completed(futures), desc="Processing Files"):
        future.result()

    # --------------------------------- Method 2 --------------------------------- #
    for json_uid in tqdm(data["train"][:82000] + data["train"][84000:]):
        try:
            process_one(os.path.join(args.input_dir, json_uid + ".json"), args)
        except:
            pass
    

def process_one(file_path,args):
    
    try:
        file_type = file_path.split(".")[-1]
        if file_type == "json":
            uid = "/".join(file_path.split("/")[-2:]).split(".")[0]
            name = uid.split("/")[-1]
            with open(file_path, "r") as f:
                json_data = json.load(f)

            cad_seq = CADSequence.json_to_NormalizedCAD(
                data=json_data, bit=8
            )
        else:
            uid = "/".join(file_path.split("/")[-4:-2])
            name = uid.split("/")[-1]
            cad_vec = torch.load(file_path)['vec']["cad_vec"].numpy()
            cad_seq = CADSequence.from_vec(cad_vec, post_processing=True)

        # Generate Mesh and 3D sketch Points
        cad_seq.create_mesh().sample_sketch_points3D(n_points=args.num_sketch_points, color=True)
        
        # ---------------------------------- Output ---------------------------------- #
        output_dir=os.path.join(args.output_dir, uid, "mesh_skt_3d")
        
        # ----------------------------------- Mesh ----------------------------------- #
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Save Mesh using trimesh
        cad_seq.mesh.export(os.path.join(output_dir, f"{name}_mesh.stl"), file_type="stl")

        # -------------------------------- Point Cloud ------------------------------- #
        # Save 3D Sketch Points in open3d
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(cad_seq.sketch_points3D)
        # Set colors for each point
        colors = o3d.utility.Vector3dVector(cad_seq.sketch_points3D_color)
        point_cloud.colors = colors
        
        o3d.io.write_point_cloud(
            os.path.join(output_dir, f"{name}_skt_3d_color.ply"), point_cloud
        )
        # clglogger.success(f"Saved in {output_dir}.")
    except Exception as e:
        clglogger.error(f"Error in {file_path}: {e}")
        # print(traceback.print_exc())


if __name__ == "__main__":
    main()