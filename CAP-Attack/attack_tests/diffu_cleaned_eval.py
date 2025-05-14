import os
import sys
import cv2
import numpy as np
import torch
import pandas as pd
import copy
from datetime import datetime

# 添加模型所在路径，保证能正确导入 OpenPilotModel
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
from openpilot_torch import OpenPilotModel

TORCH_WEIGHTS_PATH = '../models/weights/supercombo_torch_weights.pth'

def load_torch_model():
    model = OpenPilotModel()
    model.load_state_dict(torch.load(TORCH_WEIGHTS_PATH))
    model.eval()
    return model

def parse_image_multi(frames):
    n = frames.shape[0]  # 应该为2（连续两帧）
    H = (frames.shape[1] * 2) // 3
    W = frames.shape[2]
    parsed = torch.zeros((n, 6, H // 2, W // 2), dtype=torch.float32)
    parsed[:, 0] = torch.tensor(frames[:, 0:H:2, 0::2])
    parsed[:, 1] = torch.tensor(frames[:, 1:H:2, 0::2])
    parsed[:, 2] = torch.tensor(frames[:, 0:H:2, 1::2])
    parsed[:, 3] = torch.tensor(frames[:, 1:H:2, 1::2])
    parsed[:, 4] = torch.tensor(frames[:, H:H + H // 4].reshape((n, H // 2, W // 2)))
    parsed[:, 5] = torch.tensor(frames[:, H + H // 4:H + H // 2].reshape((n, H // 2, W // 2)))
    return parsed

def estimate_torch(model, input_imgs, desire, traffic_convention, recurrent_state):
    if not isinstance(input_imgs, torch.Tensor):
        input_imgs = torch.tensor(input_imgs, dtype=torch.float32)
    if not isinstance(desire, torch.Tensor):
        desire = torch.tensor(desire, dtype=torch.float32)
        traffic_convention = torch.tensor(traffic_convention, dtype=torch.float32)
    if not isinstance(recurrent_state, torch.Tensor):
        recurrent_state = torch.tensor(recurrent_state, dtype=torch.float32)
    
    out = model(input_imgs, desire, traffic_convention, recurrent_state)
    lead = out[0, 5755:6010]
    drel = []
    for t in range(6):
        x_predt = lead[4*t::51]
        if t < 3:
            prob = lead[48+t::51]
            current_most_likely_hypo = torch.argmax(prob)
            drelt = x_predt[current_most_likely_hypo]
        else:
            drelt = torch.mean(x_predt)
        drel.append(drelt)
    rec_state = out[:, -512:]
    return drel, rec_state

def process_image_pairs(clean_folder, attacked_folder, output_csv):
    clean_files = sorted([f for f in os.listdir(clean_folder) if f.endswith('.png')])
    if len(clean_files) == 0:
        print("未找到干净图像。")
        return

    results = {
        'Frame': [],
        'Clean_dRel0': [],
        'Attacked_dRel0': [],
        'Deviation': []
    }
    
    model = load_torch_model()
    desire = np.zeros((1, 8), dtype='float32')
    traffic_convention = np.array([[1, 0]], dtype='float32')
    rec_state = np.zeros((1, 512), dtype='float32')
    
    clean_buffer = []
    attacked_buffer = []
    frame_ids = []
    
    # 保持与原始代码一致：resize 到 (512,256)（宽×高）
    resize_dim = (512, 256)
    
    for file in clean_files:
        clean_path = os.path.join(clean_folder, file)
        clean_img = cv2.imread(clean_path)
        if clean_img is None:
            continue
        clean_img = cv2.resize(clean_img, resize_dim)
        clean_img_yuv = cv2.cvtColor(clean_img, cv2.COLOR_BGR2YUV_I420).astype('float32')
        
        base_name = file[:-4]
        attacked_name = base_name + '.png'
        attacked_path = os.path.join(attacked_folder, attacked_name)
        attacked_img = cv2.imread(attacked_path)
        if attacked_img is None:
            print("未找到攻击后图像：", attacked_path)
            continue
        attacked_img = cv2.resize(attacked_img, resize_dim)
        attacked_img_yuv = cv2.cvtColor(attacked_img, cv2.COLOR_BGR2YUV_I420).astype('float32')
        
        try:
            frame_part = file.split('_')[1]  # 如 "frame1.png"
            frame_id = int(''.join([c for c in frame_part if c.isdigit()]))
        except Exception as e:
            frame_id = -1
        
        clean_buffer.append(clean_img_yuv)
        attacked_buffer.append(attacked_img_yuv)
        frame_ids.append(frame_id)
        
        if len(clean_buffer) == 2:
            clean_stack = np.stack(clean_buffer, axis=0)
            attacked_stack = np.stack(attacked_buffer, axis=0)
            
            clean_parsed = parse_image_multi(clean_stack)
            attacked_parsed = parse_image_multi(attacked_stack)
            
            # 保证 reshape 的总元素数正确：2*6*128*256 = 393216
            clean_input = clean_parsed.reshape(1, 12, 128, 256).numpy()
            attacked_input = attacked_parsed.reshape(1, 12, 128, 256).numpy()
            
            # 使用 no_grad 以减少内存消耗
            with torch.no_grad():
                clean_preds, rec_state = estimate_torch(model, clean_input, desire, traffic_convention, rec_state)
                attacked_preds, rec_state = estimate_torch(model, attacked_input, desire, traffic_convention, rec_state)
            
            clean_dRel0 = clean_preds[0].detach().numpy() if isinstance(clean_preds[0], torch.Tensor) else clean_preds[0]
            attacked_dRel0 = attacked_preds[0].detach().numpy() if isinstance(attacked_preds[0], torch.Tensor) else attacked_preds[0]
            deviation = attacked_dRel0 - clean_dRel0
            
            results['Frame'].append(frame_ids[-1])
            results['Clean_dRel0'].append(clean_dRel0)
            results['Attacked_dRel0'].append(attacked_dRel0)
            results['Deviation'].append(deviation)
            
            clean_buffer = [clean_buffer[-1]]
            attacked_buffer = [attacked_buffer[-1]]
            frame_ids = [frame_ids[-1]]
    
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print("预测结果已保存到:", output_csv)
    
    df['Clean_dRel0_float'] = df['Clean_dRel0'].apply(lambda x: float(x))
    df['Deviation_float'] = df['Deviation'].apply(lambda x: float(x))
    
    bounds = [0, 20, 40, 60, 80, 999]
    print("\n不同距离范围内（基于干净图像预测值）的平均误差（攻击后预测 - 干净预测）：")
    for i in range(1, len(bounds)):
        lower = bounds[i - 1]
        upper = bounds[i]
        bin_label = f"[{lower}, {upper})"
        df_bin = df[(df['Clean_dRel0_float'] >= lower) & (df['Clean_dRel0_float'] < upper)]
        if not df_bin.empty:
            avg_dev = df_bin['Deviation_float'].mean()
            print(f"{bin_label}: {avg_dev}")
        else:
            print(f"{bin_label}: 无数据")

if __name__ == '__main__':
    clean_folder = "results/vid_test1/chunk_3/clean"         # 干净图像文件夹（例如 video1_frame1.png）
    attacked_folder = "image_processing_results/result_gauss/randomization"     # 攻击后图像文件夹（例如 video1_frame1_diffusion_ffhq_10m.png）
    output_csv = "image_processing_results/results/gauss_randomization_results.csv"          # CSV 结果保存路径
    
    process_image_pairs(clean_folder, attacked_folder, output_csv)
