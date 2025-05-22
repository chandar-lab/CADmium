import os
import pandas as pd
from tqdm import tqdm
import json
from jsonargparse import ArgumentParser

def process_results(result_dir: str):
    os.chdir("../data/results/")
    
    if result_dir not in os.listdir():
        raise FileNotFoundError(f"The specified directory '{result_dir}' does not exist in '../data/results/'.")
    
    files = os.listdir(result_dir)
    output_path = f'../generated_jsons/{result_dir}'
    os.makedirs(output_path, exist_ok=True)
    
    for file in files:
        if file.endswith('.csv') and 'results' in file:
            df = pd.read_csv(os.path.join(result_dir, file))
            pbar = tqdm(range(len(df)), desc=f"Processing files in {result_dir}", unit="file")
            done = 0
            failed = 0
            for i in pbar:
                name = df['uid'].loc[i]
                user_path = os.path.join(output_path, name)
                os.makedirs(user_path, exist_ok=True)
                
                json_file_path = os.path.join(user_path, f"{name.split('/')[1]}.json")
                print()
                try:
                    with open(json_file_path, 'w') as f:
                        json.dump(eval(df['pred_response'][i].replace('null', 'None')), f, indent=4)
                    done += 1
                    pbar.set_postfix(done=done, failed=failed)
                except Exception as e:
                    print(f"Error while processing", name, df['pred_response_dict'][i])
                    print(e)
                    failed += 1
                    with open(json_file_path, 'w') as f:
                        f.write(str(df['pred_response'][i].replace('null', 'None')))
                    pbar.set_postfix(done=done, failed=failed)
                    # pass  # Ignore errors during JSON dumping

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--result_dir", type=str, help="Name of the result directory to process.")
    args = parser.parse_args()
    process_results(args.result_dir)
