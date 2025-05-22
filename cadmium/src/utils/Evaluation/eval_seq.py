import pandas as pd
import numpy as np
import pickle
import os,sys
import argparse
sys.path.append("..")
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-2]))
from tqdm import tqdm
from CadSeqProc.cad_sequence import CADSequence
from CadSeqProc.utility.utils import (create_path_with_time,ensure_dir)
from CadSeqProc.utility.logger import CLGLogger
from tqdm import tqdm
import traceback
from rich import print
import json
import trimesh
from CadSeqProc.utility.utils import chamfer_dist, normalize_pc
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings("ignore")
csnLogger=CLGLogger().configure_logger().logger


def remove_nulls(d):
    """Recursively remove keys with None (null) values from a dictionary."""
    if isinstance(d, dict):
        return {k: remove_nulls(v) for k, v in d.items() if v is not None}
    elif isinstance(d, list):
        return [remove_nulls(v) for v in d]
    else:
        return d

def main():
    os.chdir("/home/mila/p/prashant.govindarajan/scratch/objgen_project/LLM4CAD/llm4cad/utils/Evaluation")
    parser=argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--input_path",help="Predicted CAD Sequence in pkl format",required=True)
    parser.add_argument("--output_dir",help="Output dir",required=True)
    parser.add_argument("--verbose",action='store_true')
    parser.add_argument("--dataset",help="Output dir",required=False, default="deepcad")
    
    args=parser.parse_args()
    stl_path = "../../data/generated_stls/" + args.output_dir.split("/")[-1]
    stl_path = create_path_with_time(stl_path)
    os.makedirs(stl_path, exist_ok=True)
    output_dir=create_path_with_time(args.output_dir)
    

    if args.verbose:
        csnLogger.info("Evaluation for Design History")
        csnLogger.info(f"Output Path {output_path}")

    folders = os.listdir(args.input_path)

    for level in ["expert"]:
        uids = []
        csnLogger.info(f"Level {level}")
        for folder in folders:
            paths = os.listdir(os.path.join(args.input_path, folder))
            for path in paths:
                if args.dataset == "deepcad":
                    uids.append(path[:4] + '/' + path)
                elif args.dataset == "fusion360":
                    uids.append(path.split('_')[0] + '/' + path)
                elif args.dataset == "text2cad":
                    uids.append(path[:4] + '/' + path)
        output_path = os.path.join(output_dir, str(level))
        ensure_dir(output_path)
        generate_analysis_report(uids=uids, input_path = args.input_path, output_path=output_path, stl_path = stl_path,
                                    logger=csnLogger,verbose=args.verbose, level=str(level), dataset=args.dataset)





def generate_analysis_report(uids, input_path,output_path, stl_path,logger,verbose,level, dataset):
    report_df = pd.DataFrame() # Dataframe for analysis
    # cm=np.zeros((4,4)) # Confusion Matrix

    # uids=list(data.keys())

    # for uid in tqdm(uids):
    #     try:
    #         pred_json = json.load(open(os.path.join(input_path, f"{uid.split('/')[0]}/{uid.split('/')[1]}/{uid.split('/')[1]}.json")))
    #         gt_json = json.load(open(os.path.join('../../data/text2cad_v1.1/jsons',
    #                                             f"{uid}/minimal_json/{uid.split('/')[1]}.json")))
    #         best_report_df=process_uid_json(uid,pred_json, gt_json,level=level)
    #         if best_report_df is not None:
    #             report_df=pd.concat([report_df,best_report_df])
    #     except Exception as e:
    #         continue
    
    def compute_df(uid, level= "expert"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # print(os.path.join(input_path, f"{uid.split('/')[0]}/{uid.split('/')[1]}/{uid.split('/')[1]}.json"))
                pred_json = json.load(open(os.path.join(input_path, f"{uid.split('/')[0]}/{uid.split('/')[1]}/{uid.split('/')[1]}.json")))
                if dataset == "deepcad":
                    gt_json = json.load(open(os.path.join('../../data/text2cad_v1.1/jsons',
                                                            f"{uid}/minimal_json/{uid.split('/')[1]}.json")))
                elif dataset == "fusion360":
                    gt_json = json.load(open(os.path.join('../../data/fusion360/Fusion360/r1.0.1/reconstruction',
                                                            f"{uid.split('/')[1]}/minimal_json/{uid.split('/')[1]}.json")))
                elif dataset == "text2cad":
                    ref_uid = uid.split('/')[1].split('_')[0]
                    gt_json = json.load(open(os.path.join('../../data/text2cad_v1.1/jsons',
                                                            f"{uid.split('_')[0]}/minimal_json/{ref_uid}.json")))
                best_report_df=process_uid_json(uid,pred_json, gt_json,level=level, stl_path=stl_path)
                if best_report_df is not None:
                    # report_df=pd.concat([report_df,best_report_df])
                    return best_report_df
        except Exception as e:
            pass
    results = Parallel(n_jobs=6)(delayed(compute_df)(uid) for uid in tqdm(uids))
    report_df = pd.concat([report_df, *[r for r in results if r is not None]])
    csv_path=os.path.join(output_path,f"report_df_{level}.csv")
    try:
        report_df.to_csv(csv_path, index=None)
        # logger.success(f"Report is saved at {csv_path}")
    except Exception as e:
        logger.error(f"Error saving csv file at {csv_path}")
        if verbose:
           print(traceback.print_exc())

    if verbose:
        logger.info("Calculating Metrics...")

    eval_dict = {}
    if not report_df.empty:
        line_metrics = report_df[(report_df['line_total_gt'] > 0)][['line_recall', 'line_precision', 'line_f1']].mean() * 100
        eval_dict['line'] = {
            'recall': line_metrics['line_recall'],
            'precision': line_metrics['line_precision'],
            'f1': line_metrics['line_f1']
        }

        # Mean Recall, Precision, F1 for Arc
        arc_metrics = report_df[(report_df['arc_total_gt'] > 0)][['arc_recall', 'arc_precision', 'arc_f1']].mean() * 100
        eval_dict['arc'] = {
            'recall': arc_metrics['arc_recall'],
            'precision': arc_metrics['arc_precision'],
            'f1': arc_metrics['arc_f1']
        }

        # Mean Recall, Precision, F1 for Circle
        circle_metrics = report_df[(report_df['circle_total_gt'] > 0)][['circle_recall', 'circle_precision', 'circle_f1']].mean() * 100
        eval_dict['circle'] = {
            'recall': circle_metrics['circle_recall'],
            'precision': circle_metrics['circle_precision'],
            'f1': circle_metrics['circle_f1']
        }

        # Mean Recall, Precision, F1 for Extrusion
        ext_recall = report_df['num_ext'] / report_df['num_ext_gt']
        ext_precision = report_df['num_ext'] / report_df['num_ext_pred']
        ext_f1 = 2 * ext_recall * ext_precision / (ext_recall + ext_precision)
        extrusion_metrics = {
            'recall': ext_recall.mean() * 100,
            'precision': ext_precision.mean() * 100,
            'f1': ext_f1.mean() * 100
        }
        eval_dict.update({'extrusion': extrusion_metrics})

        # Update Chamfer Distance
        eval_dict['cd']={}
        eval_dict['cd']['median']=report_df['cd'][report_df['cd']>0].median()
        eval_dict['cd']['mean']=report_df['cd'][report_df['cd']>0].mean()
        eval_dict['invalidity_ratio_percentage'] = (1.0 - report_df['cd'][report_df['cd']>=0].count()/len(uids)) * 100.0 ## Change this to len(test_data)       

        # Update SAV Distance
        eval_dict['sav_distance']= report_df['sav_distance'][report_df['sav_distance']>=0].mean()


        # Update Euler Number
        eval_dict['score'] = report_df['euler_score'][report_df['euler_score']>=0].mean() * 100

        # Update Curvature
        eval_dict['curvature'] = {}
        eval_dict['curvature']['median'] = report_df['curvature'][np.isfinite(report_df['curvature'])].median()
        eval_dict['curvature']['mean'] = report_df['curvature'][np.isfinite(report_df['curvature'])].mean()

        # Update Watertightness
        eval_dict['watertightness'] = report_df['watertightness'][report_df['watertightness']>=0].mean() * 100

        if verbose:
            json_formatted_str = json.dumps(eval_dict, indent=4)
            print(json_formatted_str)

        mean_report_path=os.path.join(output_path,f"mean_report_{level}.json")

        with open(mean_report_path,"w") as f:
            json.dump(eval_dict,f, indent=4)
    else:
        print("Empty report_df")



def process_vec(pred_vec,gt_vec,bit,uid):
    try:
        pred_cad=CADSequence.from_vec(pred_vec,2,8,denumericalize=False)
        gt_cad=CADSequence.from_vec(gt_vec,2,8,denumericalize=False)
        report_df,cm=gt_cad.generate_report(pred_cad,uid)
        
        return report_df,cm
    except Exception as e:
        #print(e)
        return None,None

# def save_mesh(mesh, path):
#     try:
#         mesh.export(path)
#     except Exception as e:
#         # print(f"Error saving mesh: {e}")
#         return None
#     return path

def calculate_SAV_score(pred_cad,gt_cad, uid, stl_path):
    try:
        gt_cad.create_mesh()
        pred_cad.create_mesh()
        save_mesh(pred_cad.mesh, os.path.join(stl_path, f"{uid}_pred.stl"))
        watertightness = pred_cad.mesh.is_watertight 
        assert gt_cad.mesh.is_watertight
        assert pred_cad.mesh.is_watertight
        sa_volume_gt = gt_cad.mesh.area / gt_cad.mesh.volume
        sa_volume_pred = pred_cad.mesh.area / pred_cad.mesh.volume

        sphere_sa_gt = (((3 * gt_cad.mesh.volume) / (4 * np.pi)) ** (1/3))**2 * 4 * np.pi
        sphere_sa_pred = (((3 * pred_cad.mesh.volume) / (4 * np.pi)) ** (1/3))**2 * 4 * np.pi
        
        SAV_distance = abs(sphere_sa_pred/pred_cad.mesh.area - sphere_sa_gt/gt_cad.mesh.area)

        # Now calculate Euler Characteristic
        gt_cad.mesh.remove_unreferenced_vertices()
        pred_cad.mesh.remove_unreferenced_vertices()
        
        gt_euler = gt_cad.mesh.euler_number
        pred_euler = pred_cad.mesh.euler_number
        if gt_euler == pred_euler:
            euler_score = 1
        else:
            euler_score = 0

        # Calculate discrete mean curvature
        gt_curvature = trimesh.curvature.discrete_mean_curvature_measure(gt_cad.mesh, gt_cad.mesh.vertices, 0.02).mean()
        pred_curvature = trimesh.curvature.discrete_mean_curvature_measure(pred_cad.mesh, pred_cad.mesh.vertices, 0.02).mean()

        #curvature error

        curvature_error = np.abs(gt_curvature - pred_curvature)
    except Exception as e:
        SAV = -1
        SAV_distance = -1
        euler_score = -1
        curvature_error = -np.inf
    
    return SAV_distance, euler_score, curvature_error, watertightness


def process_min_json(pred,gt,uid,bit, stl_path):
    pred = remove_nulls(pred)
    gt_cad=CADSequence.minimal_json_to_seq(gt, bit = 8)
    pred_cad=CADSequence.minimal_json_to_seq(pred, bit = 8) 

    report_df,cm=gt_cad.generate_report(pred_cad,uid)
    pred_cad.sample_points(8192)
    gt_cad.sample_points(8192)
    cd = chamfer_dist(
        normalize_pc(pred_cad.points),
        normalize_pc(gt_cad.points),
    ) * 1000
    report_df['cd'] = cd

    sav_dist, euler_score, curvature, watertightness = calculate_SAV_score(pred_cad, gt_cad, uid, stl_path)
    try:
        pred_cad.mesh.export(os.path.join(stl_path, f"{uid.split('/')[-1]}_pred.stl"))
    except:
        pass

    report_df['euler_score'] = euler_score
    report_df['curvature'] = curvature
    report_df['sav_distance'] = sav_dist
    report_df['watertightness'] = watertightness
    
    return report_df,cm


def process_uid_(uid,data,level):
    try:
        gt_vec = data[uid][level]['gt_cad_vec']
        all_cd = data[uid][level]['cd']
        best_index = 0
        pred_vec = data[uid][level]['pred_cad_vec'][best_index]
        df, _ = process_vec(pred_vec, gt_vec, 8, uid)
        df['cd'] = all_cd[best_index]
        return df
    except Exception as e:
        return None

def process_uid_json(uid, pred, gt,level,stl_path):
    best_index = 0
    df, _ = process_min_json(pred, gt, uid, 8, stl_path)
    return df

    # except Exception as e:
    #     return None

if __name__=="__main__":
    main()