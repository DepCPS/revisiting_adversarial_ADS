import os
import cv2
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from glob import glob
import sys


sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
from openpilot_torch import OpenPilotModel


def parse_image_bgr(frame_bgr):

    frame_resized = cv2.resize(frame_bgr, (512, 256))
    frame_resized = cv2.resize(frame_resized, (256, 128))  # (W,H) = (256,128)
    frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float()  # (3,128,256)
    frame_tensor_12 = frame_tensor.repeat(4, 1, 1)  # (12,128,256)
    return frame_tensor_12


def estimate_torch(model, input_imgs, desire=None, traffic_convention=None, recurrent_state=None, device="cpu"):

    if input_imgs.ndim == 3:
        input_imgs = input_imgs.unsqueeze(0)  # (1,12,128,256)
    B = input_imgs.size(0)
    if desire is None:
        desire = torch.zeros(B, 8, device=device, dtype=torch.float32)
    if traffic_convention is None:
        traffic_convention = torch.tensor([[1, 0]] * B, device=device, dtype=torch.float32)
    if recurrent_state is None:
        recurrent_state = torch.zeros(B, 512, device=device, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        out = model(input_imgs, desire, traffic_convention, recurrent_state)
    return out[0]


class AdvTrainDataset(Dataset):
    def __init__(self, clean_dir, adv_dir, model, device):

        super().__init__()
        self.clean_dir = clean_dir
        self.adv_dir = adv_dir
        self.model = model
        self.device = device


        self.adv_images = sorted(glob(os.path.join(self.adv_dir, '*.png')))
        self.adv_image_paths = []
        self.gt_distances = []

        print(f"Find {len(self.adv_images)} adversarial images in {self.adv_dir}.")
        self.model.eval()
        with torch.no_grad():
            for adv_path in self.adv_images:
                filename = os.path.basename(adv_path)  # "video1_frame1.png"
                clean_path = os.path.join(self.clean_dir, filename)
                if not os.path.exists(clean_path):
                    print(f"Warning: No corresponding file {filename} found in clean image, skipping.")
                    continue
                clean_bgr = cv2.imread(clean_path)
                if clean_bgr is None:
                    print(f"Warning: Failed to read {clean_path}, skipping.")
                    continue
                clean_input = parse_image_bgr(clean_bgr).to(self.device)  # (12,128,256)
                gt_dist = estimate_torch(self.model, clean_input, device=self.device)  # (6,)
                self.adv_image_paths.append(adv_path)
                self.gt_distances.append(gt_dist.cpu().numpy())
        
        if len(self.adv_image_paths) == 0:
            raise ValueError("No valid samples were found in the dataset. Please check whether the folder path and file name are correct.")
        else:
            print(f"The final dataset contains {len(self.adv_image_paths)} samples.")
    
    def __len__(self):
        return len(self.adv_image_paths)
    
    def __getitem__(self, idx):
        adv_path = self.adv_image_paths[idx]
        adv_bgr = cv2.imread(adv_path)
        adv_input = parse_image_bgr(adv_bgr)  # (12,128,256)
        gt_dist = self.gt_distances[idx]       # numpy (6,)
        return adv_input, torch.tensor(gt_dist, dtype=torch.float32)


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    criterion = nn.MSELoss()
    total_loss = 0.0
    for batch_idx, (adv_input, gt_dist) in enumerate(dataloader):
        adv_input = adv_input.to(device)       # (B,12,128,256)
        gt_dist = gt_dist.to(device)             # (B,6)
        optimizer.zero_grad()
        B = adv_input.size(0)
        desire = torch.zeros(B, 8, device=device, dtype=torch.float32)
        traffic_convention = torch.tensor([[1, 0]] * B, device=device, dtype=torch.float32)
        recurrent_state = torch.zeros(B, 512, device=device, dtype=torch.float32)
        pred_dist = model(adv_input, desire, traffic_convention, recurrent_state)  # (B,6)
        loss = criterion(pred_dist, gt_dist)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate_model_on_attacks(model, clean_dir, attack_dirs, device):
    model.eval()
    criterion = nn.MSELoss()
    for attack_dir in attack_dirs:
        adv_images = sorted(glob(os.path.join(attack_dir, '*.png')))
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for adv_path in adv_images:
                filename = os.path.basename(adv_path)
                clean_path = os.path.join(clean_dir, filename)
                if not os.path.exists(clean_path):
                    continue
                adv_bgr = cv2.imread(adv_path)
                clean_bgr = cv2.imread(clean_path)
                if adv_bgr is None or clean_bgr is None:
                    continue
                adv_input = parse_image_bgr(adv_bgr).to(device)      # (12,128,256)
                clean_input = parse_image_bgr(clean_bgr).to(device)    # (12,128,256)
                gt_dist = estimate_torch(model, clean_input, device=device).unsqueeze(0)  # (1,6)
                desire = torch.zeros(1, 8, device=device, dtype=torch.float32)
                traffic_convention = torch.tensor([[1, 0]], device=device, dtype=torch.float32)
                recurrent_state = torch.zeros(1, 512, device=device, dtype=torch.float32)
                pred_dist = model(adv_input.unsqueeze(0), desire, traffic_convention, recurrent_state).unsqueeze(0)  # (1,6)
                loss = criterion(pred_dist, gt_dist)
                total_loss += loss.item()
                count += 1
        avg_loss = total_loss / count if count > 0 else 0.0
        print(f"Attack folder '{os.path.basename(attack_dir)}': MSE = {avg_loss:.4f}")

def adversarial_training(
    clean_dir, 
    attack_dir_train, 
    attack_dirs_eval, 
    model_weights_path, 
    save_new_model_path, 
    device, 
    num_epochs=20, 
    batch_size=4,
    lr=1e-4
):

    model = OpenPilotModel()
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.to(device)
    

    train_dataset = AdvTrainDataset(clean_dir, attack_dir_train, model, device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        loss_val = train_one_epoch(model, train_loader, optimizer, device)
        print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {loss_val:.4f}")
    
    torch.save(model.state_dict(), save_new_model_path)
    print(f"Adversarially trained model saved to: {save_new_model_path}")
    
    print("Evaluating on attacks:")
    evaluate_model_on_attacks(model, clean_dir, attack_dirs_eval, device=device)


if __name__ == "__main__":
    # results/vid_test1/chunk_3: clean, gauss, fgsm, apgd, opt 文件夹
    chunk = 3
    results_root = os.path.join("results", "vid_test1")
    chunk_dir = os.path.join(results_root, f"chunk_{chunk}")
    
    clean_dir = os.path.join(chunk_dir, "clean")
    gauss_dir = os.path.join(chunk_dir, "gauss")
    fgsm_dir = os.path.join(chunk_dir, "fgsm")
    apgd_dir = os.path.join(chunk_dir, "apgd")
    opt_dir = os.path.join(chunk_dir, "opt")
    
    # Four attack samples used for training (loop training four new models)
    attack_dict = {
        "gauss": gauss_dir,
        "fgsm": fgsm_dir,
        "apgd": apgd_dir,
        "opt": opt_dir
    }
    # The images under all four attacks are used for evaluation.
    attack_dirs_eval = [gauss_dir, fgsm_dir, apgd_dir, opt_dir]
    
    model_weights_path = os.path.join("..", "models", "weights", "supercombo_torch_weights.pth")
    
    num_epochs = 20
    batch_size = 4
    learning_rate = 1e-4
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    
    # Cycle four attacks and train new models separately
    for attack_type, attack_train_dir in attack_dict.items():
        print(f"\n===== Start adversarial training using {attack_type} attack images =====")
        save_new_model_path = os.path.join("..", "models", "weights", f"supercombo_torch_weights_adv_{attack_type}.pth")
        adversarial_training(
            clean_dir=clean_dir,
            attack_dir_train=attack_train_dir,
            attack_dirs_eval=attack_dirs_eval,
            model_weights_path=model_weights_path,
            save_new_model_path=save_new_model_path,
            device=device,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=learning_rate
        )
