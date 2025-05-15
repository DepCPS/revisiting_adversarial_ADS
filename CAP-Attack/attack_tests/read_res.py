#!/usr/bin/env python3
import pandas as pd
import numpy as np

def get_deviation_stats_from_df(df, col_type='optmask'):
    bounds = [0, 20, 40, 60, 80, 999]

    if 'ONNX_init_dRel0' not in df.columns or f'ONNX_{col_type}_dRel0' not in df.columns:
        return None, None, None, None
    init_preds = df['ONNX_init_dRel0'].to_numpy()
    mask_preds = df[f'ONNX_{col_type}_dRel0'].to_numpy()
    deviation = mask_preds - init_preds
    avg_dev = np.mean(deviation)
    std_dev = np.std(deviation)
    avg_dev_by_dist = {}
    std_dev_by_dist = {}
    for i in range(1, len(bounds)):
        mask = (init_preds >= bounds[i-1]) & (init_preds <= bounds[i])
        if mask.sum() > 0:
            avg_dev_by_dist[bounds[i]] = np.mean(deviation[mask])
            std_dev_by_dist[bounds[i]] = np.std(deviation[mask])
        else:
            avg_dev_by_dist[bounds[i]] = np.nan
            std_dev_by_dist[bounds[i]] = np.nan
    return avg_dev, std_dev, avg_dev_by_dist, std_dev_by_dist

def collect_results_rmse_from_df(df, patch_type='init'):
    pred_times = [0, 2, 4, 6, 8, 10]
    torch_cols = [f'Torch_{patch_type}_dRel{t}' for t in pred_times]
    onnx_cols = [f'ONNX_{patch_type}_dRel{t}' for t in pred_times]

    if any(col not in df.columns for col in torch_cols+onnx_cols):
        return None, None, None
    torch_preds = df[torch_cols].values
    onnx_preds = df[onnx_cols].values
    rmse = np.sqrt(np.mean((torch_preds - onnx_preds) ** 2))
    

    indiv_rmse = {}
    for t in pred_times:
        col_t = f'Torch_{patch_type}_dRel{t}'
        col_o = f'ONNX_{patch_type}_dRel{t}'
        indiv_rmse[f'dRel{t}'] = np.sqrt(np.mean((df[col_t] - df[col_o])**2))
    

    bounds = [0, 20, 40, 60, 80, 999]
    dist_rmse = {}

    torch_min = df[torch_cols].min(axis=1)
    for i in range(1, len(bounds)):
        mask = (torch_min >= bounds[i-1]) & (torch_min <= bounds[i])
        if mask.sum() > 0:
            diff = df.loc[mask, torch_cols].values - df.loc[mask, onnx_cols].values
            dist_rmse[bounds[i]] = np.sqrt(np.mean(diff ** 2))
        else:
            dist_rmse[bounds[i]] = np.nan
    return rmse, indiv_rmse, dist_rmse

def print_results(df, patch_type, bounds=[0,20,40,60,80,999]):

    stats = get_deviation_stats_from_df(df, col_type=patch_type)
    if stats[0] is None:
        print(f"攻击类型 {patch_type} 缺少必要数据，跳过偏差统计。")
        return
    avg_dev, std_dev, avg_dev_by_dist, std_dev_by_dist = stats
    if patch_type == 'optmask':
        print("### Optimized Patch ###")
    elif patch_type == 'randmask':
        print("### Random Patch ###")
    elif patch_type == 'fgsm':
        print("### FGSM Patch ###")
    elif patch_type == 'gauss':
        print("### Gaussian Noise Attack ###")
    elif patch_type == 'apgd':
        print("### Auto-PGD Attack ###")
    else:
        print(f"### {patch_type} ###")
    print(f'Overall Deviation (m): {avg_dev} (avg), {std_dev} (std)')
    print('Deviation by lead vehicle distance (m):')
    for i in range(1, len(bounds)):
        print(f"\t[{bounds[i-1]}, {bounds[i]}] : {avg_dev_by_dist[bounds[i]]} (avg), {std_dev_by_dist[bounds[i]]} (std)")
    

    rmse_results = collect_results_rmse_from_df(df, patch_type=patch_type)
    if rmse_results[0] is None:
        print("The necessary data are missing and the RMSE cannot be calculated.")
        return
    rmse, indiv_rmse, dist_rmse = rmse_results
    print("RMSE between Torch and ONNX models (overall):", rmse)
    print("RMSE between Torch and ONNX models by lead vehicle distance (m):")
    for i in range(1, len(bounds)):
        print(f"\t[{bounds[i-1]}, {bounds[i]}] : {dist_rmse[bounds[i]]}")
    print()

if __name__ == '__main__':
    csv_file = "results/vid_test1/chunk_3/video1.csv"  
    df = pd.read_csv(csv_file)
    

    for pt in ['optmask', 'randmask', 'fgsm', 'gauss', 'apgd']:
        print_results(df, pt)
    

    print("### No Patch ###")
    rmse_results = collect_results_rmse_from_df(df, patch_type='init')
    if rmse_results[0] is not None:
        rmse, indiv_rmse, dist_rmse = rmse_results
        print("RMSE between Torch and ONNX models (overall):", rmse)
        print("RMSE between Torch and ONNX models by lead vehicle distance (m):")
        for i in range(1, len([0,20,40,60,80,999])):
            b = [0,20,40,60,80,999][i]
            print(f"\t[{[0,20,40,60,80,999][i-1]}, {b}] : {dist_rmse[b]}")
    else:
        print("No attack data missing")
