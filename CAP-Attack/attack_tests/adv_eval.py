import torch
import onnxruntime
import numpy as np
import cv2
from sklearn.metrics import mean_squared_error
import pandas as pd
import os
import sys
from datetime import datetime
from ultralytics import YOLO
import copy
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
from openpilot_torch import OpenPilotModel

# location of .onnx file for supercombo model
ONNX_MODEL_VERSION = 'original'
ONNX_MODEL_PATH = '../models/weights/supercombo_server3.onnx'
# ONNX_MODEL_PATH = '/home/student/Max/supercombo_alt_version2.onnx'

TORCH_WEIGHTS_PATH = '../models/weights/supercombo_torch_weights.pth'
YOLO_WEIGHTS_PATH = "../models/weights/yolov8n.pt"

RESULTS_DIRECTORY = 'results/png_test/'
verbose = 0

# -------------- 以下保持原有函数定义，不变 ------------------------------

def auto_pgd_attack(
    proc_images,  # 原始图像(YUV)列表，长度=2
    patch,        # 初始 patch, shape 与 random patch 相同
    h_bounds, w_bounds, 
    model_torch, model_onnx,
    desire, traffic_convention,
    rec_state_torch, rec_state_onnx,
    num_steps=5,
    step_size=1.0,
    alpha=0.75,
    thres=3
):
    # 攻击函数（本次不使用）
    pass

class AdamOptTorch:
    def __init__(self, size, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, dtype=torch.float32):
        self.exp_avg = torch.zeros(size, dtype=dtype)
        self.exp_avg_sq = torch.zeros(size, dtype=dtype)
        self.beta1 = torch.tensor(beta1)
        self.beta2 = torch.tensor(beta2)
        self.eps = eps
        self.lr = lr
        self.step = 0

    def update(self, grad):
        self.step += 1
        bias_correction1 = 1 - self.beta1 ** self.step
        bias_correction2 = 1 - self.beta2 ** self.step
        self.exp_avg = self.beta1 * self.exp_avg + (1 - self.beta1) * grad
        self.exp_avg_sq = self.beta2 * self.exp_avg_sq + (1 - self.beta2) * (grad ** 2)
        denom = (torch.sqrt(self.exp_avg_sq) / torch.sqrt(bias_correction2)) + self.eps
        step_size = self.lr / bias_correction1
        return step_size / denom * self.exp_avg

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def estimate_torch(model, input_imgs, desire, traffic_convention, recurrent_state):
    if not isinstance(input_imgs, torch.Tensor):
        input_imgs = torch.tensor(input_imgs).float()
    if not isinstance(desire, torch.Tensor):
        desire = torch.tensor(desire).float()
        traffic_convention = torch.tensor(traffic_convention).float()
    if not isinstance(recurrent_state, torch.Tensor): 
        recurrent_state = torch.tensor(recurrent_state).float()
    
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

def estimate_onnx(model, input_imgs, desire, traffic_convention, recurrent_state):
    out = model.run(['outputs'], {
        'input_imgs': input_imgs,
        'desire': desire,
        'traffic_convention': traffic_convention,
        'initial_state': recurrent_state
    })
    out = np.array(out[0])
    if ONNX_MODEL_VERSION == 'original':
        lead = out[0, 5755:6010]
    else:
        lead = out[0, 5755:5857]
    drel = []
    for t in range(6):
        x_predt = lead[4*t::51]
        if t < 3:
            prob = lead[48+t::51]
            current_most_likely_hypo = np.argmax(prob)
            drelt = x_predt[current_most_likely_hypo]
        else:
            drelt = np.mean(x_predt)
        drel.append(drelt)
    rec_state = out[:, -512:]
    return drel, rec_state

def parse_image(frame):
    H = (frame.shape[0]*2)//3
    W = frame.shape[1]
    parsed = np.zeros((6, H//2, W//2), dtype=np.uint8)
    parsed[0] = frame[0:H:2, 0::2]
    parsed[1] = frame[1:H:2, 0::2]
    parsed[2] = frame[0:H:2, 1::2]
    parsed[3] = frame[1:H:2, 1::2]
    parsed[4] = frame[H:H+H//4].reshape((-1, H//2, W//2))
    parsed[5] = frame[H+H//4:H+H//2].reshape((-1, H//2, W//2))
    return parsed

def parse_image_multi(frames):
    n = frames.shape[0]  # 应该始终为2
    H = (frames.shape[1]*2)//3
    W = frames.shape[2]
    parsed = torch.zeros(size=(n, 6, H//2, W//2)).float()
    parsed[:, 0] = frames[:, 0:H:2, 0::2]
    parsed[:, 1] = frames[:, 1:H:2, 0::2]
    parsed[:, 2] = frames[:, 0:H:2, 1::2]
    parsed[:, 3] = frames[:, 1:H:2, 1::2]
    parsed[:, 4] = frames[:, H:H+H//4].reshape((n, H//2, W//2))
    parsed[:, 5] = frames[:, H+H//4:H+H//2].reshape((n, H//2, W//2))
    return parsed

def load_torch_model():
    model = OpenPilotModel()
    model.load_state_dict(torch.load(TORCH_WEIGHTS_PATH))
    model.eval()
    return model

def dist(x, y):
    return torch.cdist(x, y, p=2.0)

def build_yuv_patch(thres, patch_dim, patch_start):
    patch_height, patch_width = patch_dim
    patch_y, patch_x = patch_start
    h_ratio = 256/874
    w_ratio = 512/1164
    y_patch_height = int(patch_height * h_ratio)
    y_patch_width = int(patch_width * w_ratio)
    y_patch_h_start = int(patch_y * h_ratio)
    y_patch_w_start = int(patch_x * w_ratio)
    y_patch = thres * np.random.rand(y_patch_height, y_patch_width).astype('float32')
    h_bounds = (y_patch_h_start, y_patch_h_start + y_patch_height)
    w_bounds = (y_patch_w_start, y_patch_w_start + y_patch_width)
    return y_patch, h_bounds, w_bounds

# ----------------- 以下为修改后的函数 -----------------
# 从指定文件夹中读取 .png 图像，不进行任何攻击，直接将两帧输入模型预测

def analyze_folder(folder_path):
    # 加载模型（这里不需要 YOLO 模块，因为不做攻击）
    onnx_model = onnxruntime.InferenceSession(ONNX_MODEL_PATH)
    torch_model = load_torch_model()
    
    # 设置图像尺寸（与原代码保持一致）
    width = 512
    height = 256
    dim = (width, height)
    
    # 初始化其他模型输入（desire、traffic_convention、rec_state 等）
    desire = np.zeros((1, 8), dtype='float32')
    traffic_convention = np.array([[1, 0]], dtype='float32')
    torch_rec_state = np.zeros((1, 512), dtype='float32')
    onnx_rec_state = np.zeros((1, 512), dtype='float32')
    
    # 模型预测的时间时刻（单位：秒）
    pred_times = [0, 2, 4, 6, 8, 10]
    
    # 用于记录预测结果
    torch_init_drel_hist = {'dRel0': [], 'dRel2': [], 'dRel4': [], 'dRel6': [], 'dRel8': [], 'dRel10': []}
    onnx_init_drel_hist = {'dRel0': [], 'dRel2': [], 'dRel4': [], 'dRel6': [], 'dRel8': [], 'dRel10': []}
    frame_hist = []
    rmse_hist = []
    
    # 读取文件夹中所有 .png 文件，并按文件名排序
    files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.png')])
    if len(files) < 2:
        print("指定文件夹中的图片不足两张，无法构造模型所需的两帧输入。")
        return

    # 为保证输入格式，初始化两个列表存放原始 YUV 图像与解析后的图像
    proc_images = []      # 用于存放原始 YUV 图像（形状应为 (384, 512)）
    parsed_images = []    # 用于存放解析后的图像（形状为 (6, 128, 256)）
    
    # 初始化时加入一个零数组，与原视频版本一致
    proc_images.append(np.zeros((384, 512), dtype='float32'))
    parsed_images.append(np.zeros((6, 128, 256), dtype='float32'))
    
    frame_idx = 0
    for file in files:
        frame_idx += 1
        file_path = os.path.join(folder_path, file)
        img = cv2.imread(file_path)
        if img is None:
            continue
        # 调整尺寸并转换为 YUV_I420 格式
        img = cv2.resize(img, dim)
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420).astype('float32')
        
        # 记录原始图像
        proc_images.append(img_yuv)
        # 对图像进行解析
        parsed = parse_image(img_yuv)
        parsed_images.append(parsed)
        
        # 当累积的解析图像数达到2时，取最近两帧进行预测
        if len(parsed_images) >= 2:
            # 取最后两帧构造输入
            input_imgs = np.array(parsed_images[-2:]).astype('float32')
            # reshape 成 (1, 12, 128, 256)
            input_imgs = input_imgs.reshape(1, 12, 128, 256)
            
            # 无攻击推理
            torch_drel, torch_rec_state = estimate_torch(torch_model, input_imgs, desire, traffic_convention, torch_rec_state)
            onnx_drel, onnx_rec_state = estimate_onnx(onnx_model, input_imgs, desire, traffic_convention, onnx_rec_state)
            
            torch_drel = np.array([v.detach().numpy() for v in torch_drel])
            for j, t in enumerate(pred_times):
                torch_init_drel_hist['dRel'+str(t)].append(torch_drel[j])
                onnx_init_drel_hist['dRel'+str(t)].append(onnx_drel[j])
            rmse = np.sqrt(mean_squared_error(torch_drel, onnx_drel))
            rmse_hist.append(rmse)
            frame_hist.append(frame_idx)
            print(f'Frame {frame_idx}: Torch dRel0 = {torch_drel[0]}, ONNX dRel0 = {onnx_drel[0]}, RMSE = {rmse}')
    
    # 记录结果至 CSV 文件
    data = {}
    data['Frame'] = frame_hist
    for t in pred_times:
        data['Torch_dRel'+str(t)] = torch_init_drel_hist['dRel'+str(t)]
        data['ONNX_dRel'+str(t)] = onnx_init_drel_hist['dRel'+str(t)]
    data['ONNX_Torch_RMSE'] = rmse_hist
    
    # 将结果保存到指定文件夹下的 CSV 文件中
    if not os.path.exists(RESULTS_DIRECTORY):
        os.makedirs(RESULTS_DIRECTORY)
    output_csv = os.path.join(RESULTS_DIRECTORY, 'predictions_from_png.csv')
    df = pd.DataFrame(data=data)
    df.to_csv(output_csv, index=False)
    print(f'预测结果已保存至: {output_csv}')

# --------------------- 主程序入口 ---------------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="从指定文件夹读取 .png 图像，直接输入模型进行预测（无攻击）")
    parser.add_argument('--folder', type=str, required=True, help="存放 .png 图像的文件夹路径")
    parser.add_argument('--verbose', type=int, default=0, help="详细信息输出级别")
    args = parser.parse_args()
    
    verbose = args.verbose
    analyze_folder(args.folder)
