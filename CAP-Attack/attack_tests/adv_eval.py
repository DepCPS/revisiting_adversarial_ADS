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



def auto_pgd_attack(
    proc_images,  
    patch,        
    h_bounds, w_bounds, 
    model_torch, model_onnx,
    desire, traffic_convention,
    rec_state_torch, rec_state_onnx,
    num_steps=5,
    step_size=1.0,
    alpha=0.75,
    thres=3
):
    
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
    n = frames.shape[0]  
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



def analyze_folder(folder_path):
    
    onnx_model = onnxruntime.InferenceSession(ONNX_MODEL_PATH)
    torch_model = load_torch_model()
    
    
    width = 512
    height = 256
    dim = (width, height)
    
    
    desire = np.zeros((1, 8), dtype='float32')
    traffic_convention = np.array([[1, 0]], dtype='float32')
    torch_rec_state = np.zeros((1, 512), dtype='float32')
    onnx_rec_state = np.zeros((1, 512), dtype='float32')
    
    
    pred_times = [0, 2, 4, 6, 8, 10]
    
   
    torch_init_drel_hist = {'dRel0': [], 'dRel2': [], 'dRel4': [], 'dRel6': [], 'dRel8': [], 'dRel10': []}
    onnx_init_drel_hist = {'dRel0': [], 'dRel2': [], 'dRel4': [], 'dRel6': [], 'dRel8': [], 'dRel10': []}
    frame_hist = []
    rmse_hist = []
    
    
    files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.png')])
    if len(files) < 2:
        print("There are not enough images in the specified folder to construct the two frames of input required by the model.")
        return

    
    proc_images = []      
    parsed_images = []    
    
    
    proc_images.append(np.zeros((384, 512), dtype='float32'))
    parsed_images.append(np.zeros((6, 128, 256), dtype='float32'))
    
    frame_idx = 0
    for file in files:
        frame_idx += 1
        file_path = os.path.join(folder_path, file)
        img = cv2.imread(file_path)
        if img is None:
            continue
        
        img = cv2.resize(img, dim)
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420).astype('float32')
        
        
        proc_images.append(img_yuv)
        
        parsed = parse_image(img_yuv)
        parsed_images.append(parsed)
        
        
        if len(parsed_images) >= 2:
            
            input_imgs = np.array(parsed_images[-2:]).astype('float32')
            # reshape  (1, 12, 128, 256)
            input_imgs = input_imgs.reshape(1, 12, 128, 256)
            
            
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
    
    #  CSV 
    data = {}
    data['Frame'] = frame_hist
    for t in pred_times:
        data['Torch_dRel'+str(t)] = torch_init_drel_hist['dRel'+str(t)]
        data['ONNX_dRel'+str(t)] = onnx_init_drel_hist['dRel'+str(t)]
    data['ONNX_Torch_RMSE'] = rmse_hist
    
    
    if not os.path.exists(RESULTS_DIRECTORY):
        os.makedirs(RESULTS_DIRECTORY)
    output_csv = os.path.join(RESULTS_DIRECTORY, 'predictions_from_png.csv')
    df = pd.DataFrame(data=data)
    df.to_csv(output_csv, index=False)
    print(f'Saved to: {output_csv}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Read .png images from the specified folder and directly input them into the model for prediction (without attack)")
    parser.add_argument('--folder', type=str, required=True, help="The path to the folder where the .png images are stored")
    parser.add_argument('--verbose', type=int, default=0, help="Verbose output level")
    args = parser.parse_args()
    
    verbose = args.verbose
    analyze_folder(args.folder)
