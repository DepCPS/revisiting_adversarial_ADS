import os
import cv2
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from glob import glob
import sys

# 添加模型所在路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
from openpilot_torch import OpenPilotModel

# --------------------- 图像预处理函数 ---------------------
def parse_image_bgr(frame_bgr):
    """
    将 BGR 格式图像转换为模型输入张量。
    假设原图尺寸为 512x256，输出 tensor 形状为 (12,128,256)。
    先 resize 到 (512,256)，再 resize 到 (256,128)，然后转换通道顺序并复制到 12 个通道。
    """
    frame_resized = cv2.resize(frame_bgr, (512, 256))
    frame_resized = cv2.resize(frame_resized, (256, 128))  # (W,H) = (256,128)
    frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float()  # (3,128,256)
    frame_tensor_12 = frame_tensor.repeat(4, 1, 1)  # (12,128,256)
    return frame_tensor_12

# --------------------- 模型推理函数 ---------------------
def estimate_torch(model, input_imgs, desire=None, traffic_convention=None, recurrent_state=None, device="cpu"):
    """
    使用模型进行前向推理。输入 tensor 期望形状为 (B,12,128,256)。
    若输入为 (12,128,256)，则自动 unsqueeze 添加 batch 维度。
    默认 desire 为全0张量 (B,8)，traffic_convention 为 [[1,0]] (B,2)，
    recurrent_state 为全0张量 (B,512)。
    返回模型输出的第一项（假设为 6 个相对距离）。
    """
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

# --------------------- 单一攻击数据集 ---------------------
class AdvTrainDataset(Dataset):
    def __init__(self, clean_dir, adv_dir, model, device):
        """
        :param clean_dir: 干净图像文件夹路径
        :param adv_dir:   单一攻击图像文件夹路径（例如 fgsm）
        :param model:     用于生成 ground truth 的模型
        :param device:    训练设备
        """
        super().__init__()
        self.clean_dir = clean_dir
        self.adv_dir = adv_dir
        self.model = model
        self.device = device

        self.adv_image_paths = sorted(glob(os.path.join(self.adv_dir, '*.png')))
        self.gt_distances = []

        print(f"在 {self.adv_dir} 中找到 {len(self.adv_image_paths)} 张对抗图像。")
        self.model.eval()
        with torch.no_grad():
            for adv_path in self.adv_image_paths:
                filename = os.path.basename(adv_path)
                clean_path = os.path.join(self.clean_dir, filename)
                if not os.path.exists(clean_path):
                    print(f"警告：干净图像中未找到 {filename}，跳过。")
                    continue
                clean_bgr = cv2.imread(clean_path)
                if clean_bgr is None:
                    print(f"警告：读取 {clean_path} 失败，跳过。")
                    continue
                clean_input = parse_image_bgr(clean_bgr).to(self.device)
                gt_dist = estimate_torch(self.model, clean_input, device=self.device)
                self.gt_distances.append(gt_dist.cpu().numpy())
        
        if len(self.gt_distances) == 0:
            raise ValueError("未在数据集中找到有效样本，请检查文件夹路径和文件名。")
        else:
            print(f"最终数据集中包含 {len(self.gt_distances)} 个样本。")
    
    def __len__(self):
        return len(self.gt_distances)
    
    def __getitem__(self, idx):
        adv_path = self.adv_image_paths[idx]
        adv_bgr = cv2.imread(adv_path)
        adv_input = parse_image_bgr(adv_bgr)
        gt_dist = self.gt_distances[idx]
        return adv_input, torch.tensor(gt_dist, dtype=torch.float32)

# --------------------- 多攻击数据集（随机采样每个文件夹25%图像） ---------------------
class CombinedAdvTrainDataset(Dataset):
    def __init__(self, clean_dir, attack_dirs, model, device, sample_ratio=0.25):
        """
        :param clean_dir: 干净图像文件夹路径
        :param attack_dirs: 包含多个攻击图像文件夹的列表，例如 [gauss, fgsm, apgd, opt]
        :param model:      用于生成 ground truth 的模型
        :param device:     训练设备
        :param sample_ratio: 从每个文件夹中随机采样的比例，默认 0.25（即 25%）
        """
        super().__init__()
        self.clean_dir = clean_dir
        self.attack_dirs = attack_dirs
        self.model = model
        self.device = device

        all_paths = []
        for ad in attack_dirs:
            folder_paths = glob(os.path.join(ad, '*.png'))
            # 随机采样25%的文件（至少采样1个）
            n_sample = max(1, int(len(folder_paths) * sample_ratio))
            sampled_paths = np.random.choice(folder_paths, size=n_sample, replace=False)
            all_paths.extend(sampled_paths.tolist())
        self.adv_image_paths = sorted(all_paths)
        self.gt_distances = []

        print(f"从文件夹 {attack_dirs} 中共采样到 {len(self.adv_image_paths)} 张对抗图像。")
        self.model.eval()
        with torch.no_grad():
            valid_paths = []
            for adv_path in self.adv_image_paths:
                filename = os.path.basename(adv_path)
                clean_path = os.path.join(self.clean_dir, filename)
                if not os.path.exists(clean_path):
                    print(f"警告：干净图像中未找到 {filename}，跳过。")
                    continue
                clean_bgr = cv2.imread(clean_path)
                if clean_bgr is None:
                    print(f"警告：读取 {clean_path} 失败，跳过。")
                    continue
                clean_input = parse_image_bgr(clean_bgr).to(self.device)
                gt_dist = estimate_torch(self.model, clean_input, device=self.device)
                self.gt_distances.append(gt_dist.cpu().numpy())
                valid_paths.append(adv_path)
            self.adv_image_paths = valid_paths
        
        if len(self.adv_image_paths) == 0:
            raise ValueError("未在组合数据集中找到有效样本，请检查文件夹路径和文件名。")
        else:
            print(f"最终组合数据集中包含 {len(self.adv_image_paths)} 个样本。")
    
    def __len__(self):
        return len(self.adv_image_paths)
    
    def __getitem__(self, idx):
        adv_path = self.adv_image_paths[idx]
        adv_bgr = cv2.imread(adv_path)
        adv_input = parse_image_bgr(adv_bgr)
        gt_dist = self.gt_distances[idx]
        return adv_input, torch.tensor(gt_dist, dtype=torch.float32)

# --------------------- 训练和评估函数 ---------------------
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
        pred_dist = model(adv_input, desire, traffic_convention, recurrent_state)
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

def adversarial_training(model, train_loader, optimizer, device, num_epochs):
    for epoch in range(num_epochs):
        loss_val = train_one_epoch(model, train_loader, optimizer, device)
        print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {loss_val:.4f}")
    return model

def adversarial_training_combined(clean_dir, attack_dirs_train, attack_dirs_eval, 
                                  model_weights_path, save_new_model_path, 
                                  device, num_epochs=20, batch_size=4, lr=1e-4, sample_ratio=0.25):
    # 加载模型
    model = OpenPilotModel()
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.to(device)
    
    dataset = CombinedAdvTrainDataset(clean_dir, attack_dirs_train, model, device, sample_ratio=sample_ratio)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model = adversarial_training(model, dataloader, optimizer, device, num_epochs)
    torch.save(model.state_dict(), save_new_model_path)
    print(f"Adversarially trained (combined) model saved to: {save_new_model_path}")
    print("Evaluating on attacks:")
    evaluate_model_on_attacks(model, clean_dir, attack_dirs_eval, device=device)

# --------------------- 主函数 ---------------------
if __name__ == "__main__":
    # 假设目录结构：results/vid_test1/chunk_3 下包含 clean, gauss, fgsm, apgd, opt 文件夹
    chunk = 3
    results_root = os.path.join("results", "vid_test1")
    chunk_dir = os.path.join(results_root, f"chunk_{chunk}")
    
    clean_dir = os.path.join(chunk_dir, "clean")
    gauss_dir = os.path.join(chunk_dir, "gauss")
    fgsm_dir  = os.path.join(chunk_dir, "fgsm")
    apgd_dir  = os.path.join(chunk_dir, "apgd")
    opt_dir   = os.path.join(chunk_dir, "opt")
    
    # 评估时采用所有 4 个攻击下的图像
    attack_dirs_eval = [gauss_dir, fgsm_dir, apgd_dir, opt_dir]
    
    # 原始模型权重（不用于训练，仅用于计算 ground truth）
    model_weights_path = os.path.join("..", "models", "weights", "supercombo_torch_weights.pth")
    
    num_epochs = 20
    batch_size = 4
    learning_rate = 1e-4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    
    # ------------------ 同时用 4 个攻击图像随机采样 25% 后合成训练数据生成一个新模型 ------------------
    print("\n===== 开始使用 4 个攻击图像（组合，采样25%）进行 adversarial training =====")
    save_path_combined = os.path.join("..", "models", "weights", "supercombo_torch_weights_adv_combined_new.pth")
    combined_train_dirs = [gauss_dir, fgsm_dir, apgd_dir, opt_dir]
    adversarial_training_combined(
        clean_dir=clean_dir,
        attack_dirs_train=combined_train_dirs,
        attack_dirs_eval=attack_dirs_eval,
        model_weights_path=model_weights_path,
        save_new_model_path=save_path_combined,
        device=device,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=learning_rate,
        sample_ratio=0.25  # 仅采样 25%
    )
