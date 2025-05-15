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

CHUNKS_DIRECTORY = '../data/'
RESULTS_DIRECTORY = 'results/vid_test1/'
verbose = 0

# -------------- Auto-PGD --------------
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

    rec_state_torch = rec_state_torch.detach()

    x0 = patch.clone().detach().requires_grad_(True)
    

    tmp_pimgs = torch.tensor(np.array(copy.deepcopy(proc_images))).float()
    tmp_pimgs[:, h_bounds[0]:h_bounds[1], w_bounds[0]:w_bounds[1]] += x0
    tmp_parsed = parse_image_multi(tmp_pimgs)
    input_imgs = tmp_parsed.reshape(1, 12, 128, 256)
    drel, _ = estimate_torch(model_torch, input_imgs, desire, traffic_convention, rec_state_torch)
    drel[0].backward()  # first backward
    grad = x0.grad
    x1 = x0 + step_size * torch.sign(grad)
    x1 = torch.clip(x1, -thres, thres)
    

    tmp_best = x0.clone().detach()
    tmp_best_score = drel[0].item()
    

    tmp_pimgs2 = torch.tensor(np.array(copy.deepcopy(proc_images))).float()
    tmp_pimgs2[:, h_bounds[0]:h_bounds[1], w_bounds[0]:w_bounds[1]] += x1
    tmp_parsed2 = parse_image_multi(tmp_pimgs2)
    input_imgs2 = tmp_parsed2.reshape(1, 12, 128, 256)
    drel2, _ = estimate_torch(model_torch, input_imgs2, desire, traffic_convention, rec_state_torch)
    score_x1 = drel2[0].item()
    if score_x1 > tmp_best_score:
        tmp_best = x1.clone().detach()
        tmp_best_score = score_x1


    x_km1 = x0.clone().detach()
    x_k = x1.clone().detach().requires_grad_(True)
    for k in range(1, num_steps):
        # y(k) = x(k) + alpha * (x(k) - x(k-1))
        y_k = x_k.detach() + alpha * (x_k.detach() - x_km1.detach())
        y_k = y_k.detach().requires_grad_(True)
        
        tmp_pimgs = torch.tensor(np.array(copy.deepcopy(proc_images))).float()
        tmp_pimgs[:, h_bounds[0]:h_bounds[1], w_bounds[0]:w_bounds[1]] += y_k
        tmp_parsed = parse_image_multi(tmp_pimgs)
        input_imgs = tmp_parsed.reshape(1, 12, 128, 256)
        drel, _ = estimate_torch(model_torch, input_imgs, desire, traffic_convention, rec_state_torch)
        drel[0].backward()
        grad = y_k.grad
        
        x_kplus1 = y_k + step_size * torch.sign(grad)
        x_kplus1 = torch.clip(x_kplus1, -thres, thres)
        
        tmp_pimgs2 = torch.tensor(np.array(copy.deepcopy(proc_images))).float()
        tmp_pimgs2[:, h_bounds[0]:h_bounds[1], w_bounds[0]:w_bounds[1]] += x_kplus1
        tmp_parsed2 = parse_image_multi(tmp_pimgs2)
        input_imgs2 = tmp_parsed2.reshape(1, 12, 128, 256)
        drel2, _ = estimate_torch(model_torch, input_imgs2, desire, traffic_convention, rec_state_torch)
        score_xkplus1 = drel2[0].item()
        if score_xkplus1 > tmp_best_score:
            tmp_best = x_kplus1.clone().detach()
            tmp_best_score = score_xkplus1

        x_km1 = x_k.clone().detach()
        x_k = x_kplus1.clone().detach().requires_grad_(True)

    return tmp_best

# from DRP-attack repo: https://github.com/ASGuard-UCI/DRP-attack
# used for patch optimization
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

# pass inputs through PyTorch version of supercombo model, returns relative distance
# predictions for all time horizons and updated recurrent state
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

# pass inputs through ONNX version of supercombo model, returns relative distance
# predictions for all time horizons and updated recurrent state
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

# prases single YUV image into format expected by supercombo model
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

# parses two YUV images into the format expected by supercombo model
def parse_image_multi(frames):
    n = frames.shape[0]  # should always be 2
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

# loads PyTorch version of OpenPilot's supercombo model
def load_torch_model():
    model = OpenPilotModel()
    model.load_state_dict(torch.load(TORCH_WEIGHTS_PATH))
    model.eval()
    return model

def dist(x, y):
    return torch.cdist(x, y, p=2.0)

# creates a random patch in the Y channel of a YUV image
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

# for each frame in the video, use YOLO model to find lead vehicle, create a random patch
# to around the lead vehicle, and then optimize the patch using FGSM
# records the lead vehicle distance predictions from the ONNX and PyTorch models with
# no patch, a random patch, and the optimized patch
def analyze_video(video_path, video_id):
    global chunk, RESULTS_DIRECTORY  # 提前声明全局变量

    ### load each model ###
    onnx_model = onnxruntime.InferenceSession(ONNX_MODEL_PATH)
    torch_model = load_torch_model()
    yolo_model = YOLO(YOLO_WEIGHTS_PATH)
    
    ### opens video file and prepares initial model inputs ###
    cap = cv2.VideoCapture(video_path)
    proc_images = [np.zeros(shape=(384, 512))]
    parsed_images = [np.zeros(shape=(6, 128, 256))]

    width = 512
    height = 256
    dim = (width, height)

    # initial values for non-image model inputs
    desire = np.zeros(shape=(1, 8)).astype('float32')
    traffic_convention = np.array([[1, 0]]).astype('float32')
    torch_rec_state = np.zeros(shape=(1, 512)).astype('float32')
    onnx_rec_state = np.zeros(shape=(1, 512)).astype('float32')

    # prediction time horizons given by model output
    pred_times = [0, 2, 4, 6, 8, 10]

    ### sets up objects to record relative distance predictions and other metrics ###
    torch_init_drel_hist = {'dRel0': [], 'dRel2': [], 'dRel4': [], 'dRel6': [], 'dRel8': [], 'dRel10': []}
    torch_optmask_drel_hist = copy.deepcopy(torch_init_drel_hist)
    onnx_init_drel_hist = copy.deepcopy(torch_init_drel_hist)
    onnx_optmask_drel_hist = copy.deepcopy(torch_init_drel_hist)
    torch_randmask_drel_hist = copy.deepcopy(torch_init_drel_hist)
    onnx_randmask_drel_hist = copy.deepcopy(torch_init_drel_hist)
    torch_fgsm_drel_hist = copy.deepcopy(torch_init_drel_hist)
    onnx_fgsm_drel_hist = copy.deepcopy(torch_init_drel_hist)

    # Gaussian Noise and Auto-PGD 
    torch_gauss_drel_hist = copy.deepcopy(torch_init_drel_hist)
    onnx_gauss_drel_hist = copy.deepcopy(torch_init_drel_hist)
    torch_apgd_drel_hist = copy.deepcopy(torch_init_drel_hist)
    onnx_apgd_drel_hist = copy.deepcopy(torch_init_drel_hist)

    rmse_hist = []
    frame_hist = []
    mask_effect_hist = []
    patch_h_hist = []
    patch_w_hist = []

    ### Set the directory to save the attack images ###
    chunk_dir = os.path.join(RESULTS_DIRECTORY, f'chunk_{chunk}')
    save_dir_gauss = os.path.join(chunk_dir, 'gauss')
    save_dir_apgd = os.path.join(chunk_dir, 'apgd')
    save_dir_fgsm = os.path.join(chunk_dir, 'fgsm')
    save_dir_opt = os.path.join(chunk_dir, 'opt')
    for d in [save_dir_gauss, save_dir_apgd, save_dir_fgsm, save_dir_opt]:
        if not os.path.exists(d):
            os.makedirs(d)
    
    ### reads video, detects objects, and applies patch to input ###
    img_center = torch.tensor([[437, 582, 437, 582]]).float()
    thres = 3
    mask_iterations = 5

    frame_idx = 0
    while cap.isOpened():
        frame_idx += 1
        ret, frame = cap.read()
        if not ret:
            break

        if frame is not None:
            results = yolo_model(frame, verbose=False)
            if verbose > 1:
                print('--- Frame {} ---'.format(frame_idx))
            elif verbose > 0 and frame_idx % 200 == 0:
                print(f'\t{frame_idx} frames processed')
            boxes = results[0].boxes

            max_size = 0
            box = None
            for i in range(len(boxes)):
                box_ = boxes[i].xyxy
                size = (box_[0, 3] - box_[0, 1]) * (box_[0, 2] - box_[0, 0])
                if size > max_size:
                    box = box_
                    max_size = size
            
            if box is None or len(boxes) == 0:
                box = torch.tensor([[437, 582, 438, 583]])

            if isinstance(box, torch.Tensor):
                box = box.int().cpu().numpy()
            else:
                box = box.astype('int32')
            if verbose > 1:
                print("Patch bounding box:", box)
            
            patch_start = (box[0, 1], box[0, 0])
            patch_dim = (box[0, 3] - box[0, 1], box[0, 2] - box[0, 0])
            patch_w_hist.append(patch_dim[0])
            patch_h_hist.append(patch_dim[1])
            if verbose > 1:
                print('Patch size:', patch_dim)
            
            img = cv2.resize(frame, dim)
            img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420).astype('float32')
            if len(proc_images) >= 2:
                del proc_images[0]
            proc_images.append(img_yuv)
            patch, h_bounds, w_bounds = build_yuv_patch(thres, patch_dim, patch_start)
            parsed = parse_image(img_yuv)

        if len(parsed_images) >= 2:
            del parsed_images[0]
        parsed_images.append(parsed)

        if len(parsed_images) >= 2:

            input_imgs = np.array(parsed_images).astype('float32')
            input_imgs.resize((1, 12, 128, 256))
            torch_drel, torch_rec_state = estimate_torch(torch_model, input_imgs, desire, traffic_convention, torch_rec_state)
            torch_rec_state = torch_rec_state.detach()
            onnx_drel, onnx_rec_state = estimate_onnx(onnx_model, input_imgs, desire, traffic_convention, onnx_rec_state)
            torch_drel = np.array([v.detach().numpy() for v in torch_drel])
            for j, t in enumerate(pred_times):
                torch_init_drel_hist['dRel'+str(t)].append(torch_drel[j])
                onnx_init_drel_hist['dRel'+str(t)].append(onnx_drel[j])
            rmse_hist.append(np.sqrt(mean_squared_error(torch_drel, onnx_drel)))
            frame_hist.append(frame_idx)
            if verbose > 1:
                print('Lead car rel dist, no patch: {}'.format(torch_drel[0]))


            tmp_gauss_imgs = copy.deepcopy(proc_images)
            tmp_gauss_imgs = torch.tensor(np.array(tmp_gauss_imgs)).float()
            gauss_std = 1.0  
            gauss_noise = torch.normal(mean=0.0, std=gauss_std, 
                                       size=(h_bounds[1]-h_bounds[0], w_bounds[1]-w_bounds[0]))
            tmp_gauss_imgs[:, h_bounds[0]:h_bounds[1], w_bounds[0]:w_bounds[1]] += gauss_noise
            tmp_gauss_imgs_parsed = parse_image_multi(tmp_gauss_imgs)
            gauss_input_imgs = tmp_gauss_imgs_parsed.reshape(1, 12, 128, 256)
            torch_drel_gauss, _ = estimate_torch(torch_model, gauss_input_imgs, desire, traffic_convention, torch_rec_state)
            onnx_drel_gauss, _ = estimate_onnx(onnx_model, gauss_input_imgs.detach().numpy(), desire, traffic_convention, onnx_rec_state)
            torch_drel_gauss = np.array([v.detach().numpy() for v in torch_drel_gauss])
            for j, t in enumerate(pred_times):
                torch_gauss_drel_hist['dRel'+str(t)].append(torch_drel_gauss[j])
                onnx_gauss_drel_hist['dRel'+str(t)].append(onnx_drel_gauss[j])

            gauss_frame = tmp_gauss_imgs[1].detach().cpu().numpy()
            gauss_frame = np.clip(gauss_frame, 0, 255).astype(np.uint8)
            gauss_bgr = cv2.cvtColor(gauss_frame, cv2.COLOR_YUV2BGR_I420)
            cv2.imwrite(os.path.join(save_dir_gauss, f'video{video_id}_frame{frame_idx}.png'), gauss_bgr)


            tmp_apgd_imgs = copy.deepcopy(proc_images)
            apgd_patch = torch.zeros((h_bounds[1]-h_bounds[0], w_bounds[1]-w_bounds[0]),
                                     dtype=torch.float32, requires_grad=True)
            apgd_final = auto_pgd_attack(
                proc_images=tmp_apgd_imgs,
                patch=apgd_patch,
                h_bounds=h_bounds,
                w_bounds=w_bounds,
                model_torch=torch_model,
                model_onnx=onnx_model,
                desire=desire,
                traffic_convention=traffic_convention,
                rec_state_torch=torch_rec_state,
                rec_state_onnx=onnx_rec_state,
                num_steps=5,
                step_size=1.0,
                alpha=0.75,
                thres=3
            )
            tmp_apgd_imgs = torch.tensor(np.array(tmp_apgd_imgs)).float()
            tmp_apgd_imgs[:, h_bounds[0]:h_bounds[1], w_bounds[0]:w_bounds[1]] += apgd_final
            tmp_parsed_apgd = parse_image_multi(tmp_apgd_imgs)
            apgd_input_imgs = tmp_parsed_apgd.reshape(1, 12, 128, 256)
            torch_drel_apgd, _ = estimate_torch(torch_model, apgd_input_imgs, desire, traffic_convention, torch_rec_state)
            onnx_drel_apgd, _ = estimate_onnx(onnx_model, apgd_input_imgs.detach().numpy(), desire, traffic_convention, onnx_rec_state)
            torch_drel_apgd = np.array([v.detach().numpy() for v in torch_drel_apgd])
            for j, t in enumerate(pred_times):
                torch_apgd_drel_hist['dRel'+str(t)].append(torch_drel_apgd[j])
                onnx_apgd_drel_hist['dRel'+str(t)].append(onnx_drel_apgd[j])

            apgd_frame = tmp_apgd_imgs[1].detach().cpu().numpy()
            apgd_frame = np.clip(apgd_frame, 0, 255).astype(np.uint8)
            apgd_bgr = cv2.cvtColor(apgd_frame, cv2.COLOR_YUV2BGR_I420)
            cv2.imwrite(os.path.join(save_dir_apgd, f'video{video_id}_frame{frame_idx}.png'), apgd_bgr)


            patch_tensor = torch.tensor(patch, requires_grad=True)
            adam = AdamOptTorch(patch_tensor.shape, lr=1)
            for it in range(mask_iterations):
                if it == 1:

                    fgsm_tmp = torch.tensor(np.array(copy.deepcopy(proc_images))).float()
                    fgsm_tmp[:, h_bounds[0]:h_bounds[1], w_bounds[0]:w_bounds[1]] += fgsm_patch

                    fgsm_frame = fgsm_tmp[1].detach().cpu().numpy()
                    fgsm_frame = np.clip(fgsm_frame, 0, 255).astype(np.uint8)
                    fgsm_bgr = cv2.cvtColor(fgsm_frame, cv2.COLOR_YUV2BGR_I420)
                    cv2.imwrite(os.path.join(save_dir_fgsm, f'video{video_id}_frame{frame_idx}.png'), fgsm_bgr)
                    
                    tmp_pimgs = torch.tensor(np.array(copy.deepcopy(proc_images))).float()
                    tmp_pimgs[:, h_bounds[0]:h_bounds[1], w_bounds[0]:w_bounds[1]] += fgsm_patch
                    tmp_pimgs = parse_image_multi(tmp_pimgs)
                    input_imgs = tmp_pimgs.reshape(1, 12, 128, 256)
                    torch_drel, _ = estimate_torch(torch_model, input_imgs, desire, traffic_convention, torch_rec_state)
                    onnx_drel, _ = estimate_onnx(onnx_model, input_imgs.detach().numpy(), desire, traffic_convention, onnx_rec_state)
                    if verbose > 1:
                        print('Lead car rel dist, with FGSM patch: {}'.format(torch_drel[0]))
                    torch_drel = np.array([v.detach().numpy() for v in torch_drel])
                    for j, t in enumerate(pred_times):
                        torch_fgsm_drel_hist['dRel'+str(t)].append(torch_drel[j])
                        onnx_fgsm_drel_hist['dRel'+str(t)].append(onnx_drel[j])
                patch_tensor = torch.clip(patch_tensor, -thres, thres)
                tmp_pimgs = torch.tensor(np.array(copy.deepcopy(proc_images))).float()
                tmp_pimgs[:, h_bounds[0]:h_bounds[1], w_bounds[0]:w_bounds[1]] += patch_tensor
                tmp_pimgs = parse_image_multi(tmp_pimgs)
                input_imgs = tmp_pimgs.reshape(1, 12, 128, 256)
                torch_drel, _ = estimate_torch(torch_model, input_imgs, desire, traffic_convention, torch_rec_state)
                if it == 0:
                    onnx_drel, _ = estimate_onnx(onnx_model, input_imgs.detach().numpy(), desire, traffic_convention, onnx_rec_state)
                    if verbose > 1:
                        print('Lead car rel dist, with random patch: {}'.format(onnx_drel[0]))
                    torch_drel_rand = np.array([v.detach().numpy() for v in torch_drel])
                    for j, t in enumerate(pred_times):
                        torch_randmask_drel_hist['dRel'+str(t)].append(torch_drel_rand[j])
                        onnx_randmask_drel_hist['dRel'+str(t)].append(onnx_drel[j])
                patch_tensor.retain_grad()
                torch_drel[0].backward()
                grad = patch_tensor.grad
                if it == 0:
                    fgsm_patch = patch_tensor + thres * torch.sign(grad)
                update = adam.update(grad)
                patch_tensor = patch_tensor.clone().detach().requires_grad_(True) + update
                torch_rec_state = torch_rec_state.clone().detach()
            onnx_drel, _ = estimate_onnx(onnx_model, input_imgs.detach().numpy(), desire, traffic_convention, onnx_rec_state)
            torch_drel, _ = estimate_torch(torch_model, input_imgs, desire, traffic_convention, torch_rec_state)
            torch_drel = np.array([v.detach().numpy() for v in torch_drel])
            for j, t in enumerate(pred_times):
                onnx_optmask_drel_hist['dRel'+str(t)].append(onnx_drel[j])
                torch_optmask_drel_hist['dRel'+str(t)].append(torch_drel[j])
            if verbose > 1:
                print('Lead car rel dist after {} updates: {}'.format(mask_iterations, onnx_drel[0]))
            mask_effect_hist.append(onnx_drel[0] - torch_drel_rand[0])

            tmp_pimgs_opt = torch.tensor(np.array(copy.deepcopy(proc_images))).float()
            tmp_pimgs_opt[:, h_bounds[0]:h_bounds[1], w_bounds[0]:w_bounds[1]] += patch_tensor
            opt_frame = tmp_pimgs_opt[1].detach().cpu().numpy()
            opt_frame = np.clip(opt_frame, 0, 255).astype(np.uint8)
            opt_bgr = cv2.cvtColor(opt_frame, cv2.COLOR_YUV2BGR_I420)
            cv2.imwrite(os.path.join(save_dir_opt, f'video{video_id}_frame{frame_idx}.png'), opt_bgr)

    ### record results and generate metrics ###
    data = {}
    data['Frame'] = frame_hist
    for t in pred_times:
        data['Torch_init_dRel'+str(t)] = torch_init_drel_hist['dRel'+str(t)]
        data['Torch_randmask_dRel'+str(t)] = torch_randmask_drel_hist['dRel'+str(t)]
        data['Torch_optmask_dRel'+str(t)] = torch_optmask_drel_hist['dRel'+str(t)]
        data['Torch_fgsm_dRel'+str(t)] = torch_fgsm_drel_hist['dRel'+str(t)]
        data['Torch_gauss_dRel'+str(t)] = torch_gauss_drel_hist['dRel'+str(t)]
        data['Torch_apgd_dRel'+str(t)] = torch_apgd_drel_hist['dRel'+str(t)]
        data['ONNX_init_dRel'+str(t)] = onnx_init_drel_hist['dRel'+str(t)]
        data['ONNX_randmask_dRel'+str(t)] = onnx_randmask_drel_hist['dRel'+str(t)]
        data['ONNX_optmask_dRel'+str(t)] = onnx_optmask_drel_hist['dRel'+str(t)]
        data['ONNX_fgsm_dRel'+str(t)] = onnx_fgsm_drel_hist['dRel'+str(t)]
        data['ONNX_gauss_dRel'+str(t)] = onnx_gauss_drel_hist['dRel'+str(t)]
        data['ONNX_apgd_dRel'+str(t)] = onnx_apgd_drel_hist['dRel'+str(t)]
    data['ONNX_Torch_RMSE'] = rmse_hist
    data['Mask_dRel_Effect'] = mask_effect_hist
    data['Patch_Height'] = patch_h_hist
    data['Patch_Width'] = patch_w_hist

    df = pd.DataFrame(data=data)
    res_path = os.path.join(RESULTS_DIRECTORY, f'chunk_{chunk}/video{video_id}.csv')
    df.to_csv(res_path, index=False)

    cap.release()
    cv2.destroyAllWindows()


def collect_results_rmse(res_path, name_filter='', patch_type='init'):
    all_mse = []
    indiv_mse = {}
    pred_times = [0, 2, 4, 6, 8, 10]
    for t in pred_times:
        indiv_mse['dRel'+str(t)] = []
    dist_mse = {}
    bounds = [0, 20, 40, 60, 80, 999]
    torch_cols = [f'Torch_{patch_type}_dRel{t}' for t in pred_times]
    onnx_cols = [f'ONNX_{patch_type}_dRel{t}' for t in pred_times]
    for i in range(1, len(bounds)):
        dist_mse[bounds[i]] = []
    for root, dirs, files in os.walk(res_path):
        for file in files:
            if file.endswith('.csv') and name_filter in file:
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                torch_preds = df.loc[:, torch_cols].to_numpy()
                onnx_preds = df.loc[:, onnx_cols].to_numpy()
                trace_mse = np.mean(np.power(torch_preds - onnx_preds, 2))
                all_mse.append(trace_mse)
                for t in pred_times:
                    pred_diff = df[f'Torch_{patch_type}_dRel{t}'] - df[f'ONNX_{patch_type}_dRel{t}']
                    pred_mse = pred_diff.pow(2).mean()
                    indiv_mse['dRel'+str(t)].append(pred_mse)
                for i in range(1, len(bounds)):
                    pred_within_bounds = np.all([df[torch_cols] >= bounds[i-1], df[torch_cols] <= bounds[i]], axis=0)
                    torch_mask = np.array(df[torch_cols]*pred_within_bounds)
                    onnx_mask = np.array(df[onnx_cols]*pred_within_bounds)
                    pred_mse = np.mean(np.power(torch_mask - onnx_mask, 2))
                    dist_mse[bounds[i]].append(pred_mse)
    rmse = np.sqrt(np.mean(all_mse))
    indiv_rmse = {}
    for t in pred_times:
        indiv_rmse['dRel'+str(t)] = np.sqrt(np.mean(indiv_mse['dRel'+str(t)]))
    dist_rmse = {}
    for i in range(1, len(bounds)):
        dist_rmse[bounds[i]] = np.sqrt(np.mean(dist_mse[bounds[i]]))
    return rmse, indiv_rmse, dist_rmse

def get_deviation_stats(res_path, blacklist, col_type='optmask'):
    bounds = [0, 20, 40, 60, 80, 999]
    init_predictions = None
    mask_predictions = None
    for root, dirs, files in os.walk(res_path):
        for file in files:
            if file.endswith('.csv') and file not in blacklist:
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                df = df.iloc[10:]
                #print(df.columns)
                if init_predictions is None:
                    init_predictions = df['ONNX_init_dRel0'].to_numpy()
                    mask_predictions = df[f'ONNX_{col_type}_dRel0'].to_numpy()
                else:
                    init_predictions = np.r_[init_predictions, df['ONNX_init_dRel0'].to_numpy()]
                    mask_predictions = np.r_[mask_predictions, df[f'ONNX_{col_type}_dRel0'].to_numpy()]
    deviation = mask_predictions - init_predictions
    avg_dev = np.mean(deviation)
    std_dev = np.std(deviation)
    avg_dev_by_dist = {}
    std_dev_by_dist = {}
    for i in range(1, len(bounds)):
        pred_within_bounds = np.all([init_predictions >= bounds[i-1], init_predictions <= bounds[i]], axis=0)
        deviation_within_bounds = deviation[pred_within_bounds]
        avg_dev_by_dist[bounds[i]] = np.mean(deviation_within_bounds)
        std_dev_by_dist[bounds[i]] = np.std(deviation_within_bounds)
    return avg_dev, std_dev, avg_dev_by_dist, std_dev_by_dist

def collect_results_patched(res_path, blacklist, name_filter=''):
    pred_times = [0, 2, 4, 6, 8, 10]
    opt_avg_dev, rand_avg_dev, fgsm_avg_dev = [], [], []
    opt_max_dev, rand_max_dev, fgsm_max_dev = 0, 0, 0
    opt_avg_dev_by_dist, rand_avg_dev_by_dist, fgsm_avg_dev_by_dist = {}, {}, {}
    opt_max_dev_by_dist, rand_max_dev_by_dist, fgsm_max_dev_by_dist = {}, {}, {}
    bounds = [0, 20, 40, 60, 80, 999]
    for i in range(1, len(bounds)):
        opt_avg_dev_by_dist[bounds[i]] = []
        rand_avg_dev_by_dist[bounds[i]] = []
        fgsm_avg_dev_by_dist[bounds[i]] = []
        opt_max_dev_by_dist[bounds[i]] = 0
        rand_max_dev_by_dist[bounds[i]] = 0
        fgsm_max_dev_by_dist[bounds[i]] = 0
    for root, dirs, files in os.walk(res_path):
        for file in files:
            if file.endswith('.csv') and name_filter in file and file not in blacklist:
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                df = df.iloc[10:]
                opt_effect = df['ONNX_optmask_dRel0'] - df['ONNX_init_dRel0']
                rand_effect = df['ONNX_randmask_dRel0'] - df['ONNX_init_dRel0']
                fgsm_effect = df['ONNX_fgsm_dRel0'] - df['ONNX_init_dRel0']
                opt_avg_dev.append(opt_effect.mean())
                rand_avg_dev.append(rand_effect.mean())
                fgsm_avg_dev.append(fgsm_effect.mean())
                opt_max_dev = max(opt_max_dev, opt_effect.max())
                rand_max_dev = max(rand_max_dev, rand_effect.max())
                fgsm_max_dev = max(fgsm_max_dev, fgsm_effect.max())
                for i in range(1, len(bounds)):
                    pred_within_bounds = np.all([df['ONNX_init_dRel0'] >= bounds[i-1], df['ONNX_init_dRel0'] <= bounds[i]], axis=0)
                    opt_dev_within_bounds = opt_effect[pred_within_bounds].to_numpy()
                    rand_dev_within_bounds = rand_effect[pred_within_bounds].to_numpy()
                    fgsm_dev_within_bounds = fgsm_effect[pred_within_bounds].to_numpy()
                    if len(opt_dev_within_bounds) == 0:
                        continue
                    opt_avg_dev_within_bounds = np.mean(opt_dev_within_bounds)
                    rand_avg_dev_within_bounds = np.mean(rand_dev_within_bounds)
                    fgsm_avg_dev_within_bounds = np.mean(fgsm_dev_within_bounds)
                    opt_max_dev_within_bounds = np.max(opt_dev_within_bounds)
                    rand_max_dev_within_bounds = np.max(rand_dev_within_bounds)
                    fgsm_max_dev_within_bounds = np.max(fgsm_dev_within_bounds)
                    opt_avg_dev_by_dist[bounds[i]].append(opt_avg_dev_within_bounds)
                    rand_avg_dev_by_dist[bounds[i]].append(rand_avg_dev_within_bounds)
                    fgsm_avg_dev_by_dist[bounds[i]].append(fgsm_avg_dev_within_bounds)
                    if opt_max_dev_within_bounds > opt_max_dev_by_dist[bounds[i]]:
                        opt_max_dev_by_dist[bounds[i]] = opt_max_dev_within_bounds
                    if rand_max_dev_within_bounds > rand_max_dev_by_dist[bounds[i]]:
                        rand_max_dev_by_dist[bounds[i]] = rand_max_dev_within_bounds
                    if fgsm_max_dev_within_bounds > fgsm_max_dev_by_dist[bounds[i]]:
                        fgsm_max_dev_by_dist[bounds[i]] = fgsm_max_dev_within_bounds
    opt_avg_dev_by_dist_collected, rand_avg_dev_by_dist_collected, fgsm_avg_dev_by_dist_collected = {}, {}, {}
    for i in range(1, len(bounds)):
        opt_avg_dev_by_dist_collected[bounds[i]] = np.mean(opt_avg_dev_by_dist[bounds[i]])
        rand_avg_dev_by_dist_collected[bounds[i]] = np.mean(rand_avg_dev_by_dist[bounds[i]])
        fgsm_avg_dev_by_dist_collected[bounds[i]] = np.mean(fgsm_avg_dev_by_dist[bounds[i]])
    opt_res = (np.mean(opt_avg_dev), opt_max_dev, opt_avg_dev_by_dist_collected, opt_max_dev_by_dist)
    rand_res = (np.mean(rand_avg_dev), rand_max_dev, rand_avg_dev_by_dist_collected, rand_max_dev_by_dist)
    fgsm_res = (np.mean(fgsm_avg_dev), fgsm_max_dev, fgsm_avg_dev_by_dist_collected, fgsm_max_dev_by_dist)
    return opt_res, rand_res, fgsm_res


if __name__ == '__main__':
    if len(sys.argv) > 1:
        print('Setting verbose level to', sys.argv[1])
        verbose = int(sys.argv[1])
    if not os.path.exists(RESULTS_DIRECTORY):
        os.mkdir(RESULTS_DIRECTORY)
    video_blacklist = []
    result_blacklist = []
    chunk = 3
    video_id = 0
    chunk_results_dir = os.path.join(RESULTS_DIRECTORY, f'chunk_{chunk}/')
    if not os.path.exists(chunk_results_dir):
        os.mkdir(chunk_results_dir)
    map_path = os.path.join(chunk_results_dir, 'filepaths.txt')
    if not os.path.exists(map_path):
        with open(map_path, 'x') as f:
            f.write('video_id|path\n')
    chunk_path = os.path.join(CHUNKS_DIRECTORY, f'Chunk_{chunk}/')
    for root, dirs, files in os.walk(chunk_path):
        for file in files:
            if file.endswith('.hevc'):
                video_path = os.path.join(root, file)
                video_id += 1
                if video_path in video_blacklist:
                    continue
                with open(map_path, 'a') as f:
                    f.write(str(video_id) + '|' + str(video_path) + '\n')
                print('Starting video', video_id)
                start = datetime.now()
                analyze_video(video_path, video_id)
                dur = datetime.now() - start
                print(dur, 'for file', video_id)
    results_path = os.path.join(RESULTS_DIRECTORY, f'chunk_{chunk}')
    bounds = [0, 20, 40, 60, 80, 999]

    for pt in ['optmask', 'randmask', 'fgsm', 'gauss', 'apgd']:
        if pt == 'optmask':
            print('### Optimized Patch ###')
        elif pt == 'randmask':
            print('### Random Patch ###')
        elif pt == 'fgsm':
            print('### FGSM Patch ###')
        elif pt == 'gauss':
            print('### Gaussian Noise Attack ###')
        elif pt == 'apgd':
            print('### Auto-PGD Attack ###')
        avg_dev, std_dev, avg_dev_by_dist, std_dev_by_dist = get_deviation_stats(results_path, result_blacklist, col_type=pt)
        print(f'Overall Deviation (m): {avg_dev} (avg), {std_dev} (std)')
        print('Deviation by lead vehicle distance (m):')
        for j in range(1, len(bounds)):
            print(f"\t[{bounds[j-1]}, {bounds[j]}] : {avg_dev_by_dist[bounds[j]]} (avg), {std_dev_by_dist[bounds[j]]} (std)")
        rmse, indiv_rmse, dist_rmse = collect_results_rmse(results_path, name_filter='', patch_type=pt)
        print('RMSE between Torch and ONNX models (overall):', rmse)
        print('RMSE between Torch and ONNX models by lead vehicle distance (m):')
        for j in range(1, len(bounds)):
            print(f"\t[{bounds[j-1]}, {bounds[j]}] : {dist_rmse[bounds[j]]}")
        print()
    print('### No Patch ###')
    rmse, indiv_rmse, dist_rmse = collect_results_rmse(results_path, name_filter='', patch_type='init')
    print('RMSE between Torch and ONNX models (overall):', rmse)
    print('RMSE between Torch and ONNX models by lead vehicle distance (m):')
    for i in range(1, len(bounds)):
        print(f"\t[{bounds[i-1]}, {bounds[i]}] : {dist_rmse[bounds[i]]}")
