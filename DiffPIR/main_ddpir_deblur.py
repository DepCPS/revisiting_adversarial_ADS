import os.path
import cv2
import logging

import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
from collections import OrderedDict
import hdf5storage
from scipy import ndimage

from utils import utils_model
from utils import utils_logger
from utils import utils_sisr as sr
from utils import utils_image as util
from utils.utils_deblur import MotionBlurOperator, GaussialBlurOperator

# from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)

def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------

    noise_level_img         = 12.75/255.0           # AWGN noise level for LR image
    noise_level_model       = noise_level_img       
    model_name              = '256x256_diffusion_uncond'    # 可选: diffusion_ffhq_10m, 256x256_diffusion_uncond
    testset_name            = 'apgd_test'             # 测试集名称
    num_train_timesteps     = 1000
    iter_num                = 100                     # 迭代次数
    iter_num_U              = 1                       # 内部迭代次数
    skip                    = num_train_timesteps // iter_num   # skip interval

    show_img                = False                   # 是否显示图像
    save_L                  = False                   # 保存低质量图像
    save_E                  = True                    # 保存恢复后图像
    save_LEH                = False                   # 保存拼接图（LR/E/GT）
    save_progressive        = False                   # 保存生成过程
    border                  = 0
	
    sigma                   = max(0.001, noise_level_img)   # noise level associated with condition y
    lambda_                 = 1.0                     # 关键参数 lambda
    sub_1_analytic          = True                    # 是否使用解析解
    
    log_process             = False
    ddim_sample             = False                   # 采样方法
    model_output_type       = 'pred_xstart'           # 模型输出类型: 'pred_x_prev'、'pred_xstart'、'epsilon'、'score'
    generate_mode           = 'DiffPIR'               # 生成模式: DiffPIR; DPS; vanilla
    skip_type               = 'quad'                  # 跳步方式: uniform 或 quad
    eta                     = 0.0                     # eta 参数
    zeta                    = 0.1  
    guidance_scale          = 1.0   

    calc_LPIPS              = True
    use_DIY_kernel          = True
    blur_mode               = 'Gaussian'              # 'Gaussian' 或 'motion'
    kernel_size             = 61
    kernel_std              = 3.0 if blur_mode == 'Gaussian' else 0.5

    sf                      = 1
    task_current            = 'deblur'          
    n_channels              = 3                       # 固定为 3
    cwd                     = ''  
    model_zoo               = os.path.join(cwd, 'model_zoo')    
    testsets                = os.path.join(cwd, 'testsets')     
    results                 = os.path.join(cwd, 'results')      
    result_name             = f'{testset_name}_{task_current}_{generate_mode}_{model_name}_sigma{noise_level_img}_NFE{iter_num}_eta{eta}_zeta{zeta}_lambda{lambda_}_blurmode{blur_mode}'
    model_path              = os.path.join(model_zoo, model_name+'.pt')
    device                  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    # noise schedule 
    beta_start              = 0.1 / 1000
    beta_end                = 20 / 1000
    betas                   = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
    betas                   = torch.from_numpy(betas).to(device)
    alphas                  = 1.0 - betas
    alphas_cumprod          = np.cumprod(alphas.cpu(), axis=0)
    sqrt_alphas_cumprod     = torch.sqrt(alphas_cumprod)
    sqrt_1m_alphas_cumprod  = torch.sqrt(1. - alphas_cumprod)
    reduced_alpha_cumprod   = torch.div(sqrt_1m_alphas_cumprod, sqrt_alphas_cumprod)

    noise_model_t           = utils_model.find_nearest(reduced_alpha_cumprod, 2 * noise_level_model)
    noise_model_t           = 0
    
    noise_inti_img          = 50 / 255
    t_start                 = utils_model.find_nearest(reduced_alpha_cumprod, 2 * noise_inti_img)
    t_start                 = num_train_timesteps - 1              

    # ----------------------------------------
    # 设置文件路径
    # ----------------------------------------

    L_path = os.path.join(testsets, testset_name) # 输入图像所在目录
    E_path = os.path.join(results, result_name)   # 恢复结果保存目录
    util.mkdir(E_path)

    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    # ----------------------------------------
    # 加载模型
    # ----------------------------------------

    model_config = dict(
            model_path=model_path,
            num_channels=128,
            num_res_blocks=1,
            attention_resolutions="16",
        ) if model_name == 'diffusion_ffhq_10m' \
        else dict(
            model_path=model_path,
            num_channels=256,
            num_res_blocks=2,
            attention_resolutions="8,16,32",
        )
    args = utils_model.create_argparser(model_config).parse_args([])
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()))
    # model.load_state_dict(
    #     dist_util.load_state_dict(args.model_path, map_location="cpu")
    # )
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model.eval()
    if generate_mode != 'DPS_y0':
        for k, v in model.named_parameters():
            v.requires_grad = False
    model = model.to(device)

    logger.info('model_name:{}, image sigma:{:.3f}, model sigma:{:.3f}'.format(model_name, noise_level_img, noise_level_model))
    logger.info('eta:{:.3f}, zeta:{:.3f}, lambda:{:.3f}, guidance_scale:{:.2f}'.format(eta, zeta, lambda_, guidance_scale))
    logger.info('start step:{}, skip_type:{}, skip interval:{}, skipstep analytic steps:{}'.format(t_start, skip_type, skip, noise_model_t))
    logger.info('use_DIY_kernel:{}, blur mode:{}'.format(use_DIY_kernel, blur_mode))
    logger.info('Model path: {:s}'.format(model_path))
    logger.info(L_path)
    L_paths = util.get_image_paths(L_path)
    
    if calc_LPIPS:
        import lpips
        loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    
    def test_rho(lambda_=lambda_, zeta=zeta, model_output_type=model_output_type):
        logger.info('eta:{:.3f}, zeta:{:.3f}, lambda:{:.3f}, guidance_scale:{:.2f}'.format(eta, zeta, lambda_, guidance_scale))
        test_results = OrderedDict()
        test_results['psnr'] = []
        if calc_LPIPS:
            test_results['lpips'] = []

        for idx, img in enumerate(L_paths):
            # ---------------------------
            # (0) 根据设置生成或加载 kernel
            # ---------------------------
            if use_DIY_kernel:
                np.random.seed(seed=idx*10)  # 为每幅图保持 kernel 可复现
                if blur_mode == 'Gaussian':
                    kernel_std_i = kernel_std * np.abs(np.random.rand()*2+1)
                    kernel = GaussialBlurOperator(kernel_size=kernel_size, intensity=kernel_std_i, device=device)
                elif blur_mode == 'motion':
                    kernel = MotionBlurOperator(kernel_size=kernel_size, intensity=kernel_std, device=device)
                k_tensor = kernel.get_kernel().to(device, dtype=torch.float)
                k = k_tensor.clone().detach().cpu().numpy()
                k = np.squeeze(k)
            else:
                k_index = 0
                kernels = hdf5storage.loadmat(os.path.join(cwd, 'kernels', 'Levin09.mat'))['kernels']
                k = kernels[0, k_index].astype(np.float32)
            img_name, ext = os.path.splitext(os.path.basename(img))
            util.imsave(k*255.*200, os.path.join(E_path, f'motion_kernel_{img_name}{ext}'))
            k_4d = torch.from_numpy(k).to(device)
            k_4d = torch.einsum('ab,cd->abcd', torch.eye(3).to(device), k_4d)
            
            model_out_type = model_output_type

            # ---------------------------
            # (1) 读取图像并缩放到处理尺寸
            # ---------------------------
            img_name, ext = os.path.splitext(os.path.basename(img))
            img_H = util.imread_uint(img, n_channels=n_channels)
            # 记录原始尺寸
            orig_h, orig_w, _ = img_H.shape
            # 缩放到处理尺寸，例如设为 256x256（你也可以根据需要调整）
            proc_size = (256, 256)
            img_H_resized = cv2.resize(img_H, proc_size, interpolation=cv2.INTER_AREA)
            # 保证尺寸能被8整除（若处理尺寸已满足要求，可保留 modcrop）
            img_H_resized = util.modcrop(img_H_resized, 8)
            
            # ---------------------------
            # (2) 生成低质量图像 img_L
            # ---------------------------
            # 注意：此处使用缩放后的图像进行卷积操作
            img_L = ndimage.convolve(img_H_resized, np.expand_dims(k, axis=2), mode='wrap')
            if show_img:
                util.imshow(img_L)
            img_L = util.uint2single(img_L)

            np.random.seed(seed=0)  # 保证可复现
            img_L = img_L * 2 - 1
            img_L += np.random.normal(0, noise_level_img * 2, img_L.shape)  # 添加 AWGN
            img_L = img_L / 2 + 0.5

            # ---------------------------
            # (3) 计算 rhos 和 sigmas
            # ---------------------------
            sigmas = []
            sigma_ks = []
            rhos = []
            for i in range(num_train_timesteps):
                sigmas.append(reduced_alpha_cumprod[num_train_timesteps-1-i])
                if model_out_type == 'pred_xstart' and generate_mode == 'DiffPIR':
                    sigma_ks.append((sqrt_1m_alphas_cumprod[i] / sqrt_alphas_cumprod[i]))
                else:
                    sigma_ks.append(torch.sqrt(betas[i] / alphas[i]))
                rhos.append(lambda_ * (sigma**2) / (sigma_ks[i]**2))    
            rhos = torch.tensor(rhos).to(device)
            sigmas = torch.tensor(sigmas).to(device)
            sigma_ks = torch.tensor(sigma_ks).to(device)
            
            # ---------------------------
            # (4) 初始化 x 与预计算
            # ---------------------------
            y = util.single2tensor4(img_L).to(device)   # (1,3,256,256)
            t_y = utils_model.find_nearest(reduced_alpha_cumprod, 2 * noise_level_img)
            sqrt_alpha_effective = sqrt_alphas_cumprod[t_start] / sqrt_alphas_cumprod[t_y]
            x = sqrt_alpha_effective * (2*y - 1) + torch.sqrt(sqrt_1m_alphas_cumprod[t_start]**2 - 
                    sqrt_alpha_effective**2 * sqrt_1m_alphas_cumprod[t_y]**2) * torch.randn_like(y)

            k_tensor = util.single2tensor4(np.expand_dims(k, 2)).to(device)
            FB, FBC, F2B, FBFy = sr.pre_calculate(y, k_tensor, sf)

            # ---------------------------
            # (5) 主迭代过程
            # ---------------------------
            progress_img = []
            if skip_type == 'uniform':
                seq = [i * skip for i in range(iter_num)]
                if skip > 1:
                    seq.append(num_train_timesteps - 1)
            elif skip_type == "quad":
                seq = np.sqrt(np.linspace(0, num_train_timesteps**2, iter_num))
                seq = [int(s) for s in list(seq)]
                seq[-1] = seq[-1] - 1
            progress_seq = seq[::max(len(seq)//10, 1)]
            if progress_seq[-1] != seq[-1]:
                progress_seq.append(seq[-1])
            
            for i in range(len(seq)):
                curr_sigma = sigmas[seq[i]].cpu().numpy()
                t_i = utils_model.find_nearest(reduced_alpha_cumprod, curr_sigma)
                if t_i > t_start:
                    continue
                for u in range(iter_num_U):
                    if 'DPS' in generate_mode:
                        x = x.requires_grad_()
                        xt, x0 = utils_model.model_fn(x, noise_level=curr_sigma*255, model_out_type='pred_x_prev_and_start', 
                                    model_diffusion=model, diffusion=diffusion, ddim_sample=ddim_sample, alphas_cumprod=alphas_cumprod)
                    else:
                        x0 = utils_model.model_fn(x, noise_level=curr_sigma*255, model_out_type=model_out_type, 
                                model_diffusion=model, diffusion=diffusion, ddim_sample=ddim_sample, alphas_cumprod=alphas_cumprod)
                    
                    if seq[i] != seq[-1]:
                        if generate_mode == 'DiffPIR':
                            if sub_1_analytic:
                                if model_out_type == 'pred_xstart':
                                    tau = rhos[t_i].float().repeat(1, 1, 1, 1)
                                    if i < num_train_timesteps - noise_model_t: 
                                        x0_p = x0 / 2 + 0.5
                                        x0_p = sr.data_solution(x0_p.float(), FB, FBC, F2B, FBFy, tau, sf)
                                        x0_p = x0_p * 2 - 1
                                        x0 = x0 + guidance_scale * (x0_p - x0)
                                    else:
                                        model_out_type = 'pred_x_prev'
                                        x0 = utils_model.model_fn(x, noise_level=curr_sigma*255, model_out_type=model_out_type, 
                                                model_diffusion=model, diffusion=diffusion, ddim_sample=ddim_sample, alphas_cumprod=alphas_cumprod)
                                else:
                                    x = (y + rhos[t_i].float() * x).div(1 + rhos[t_i])
                            else:
                                x0 = x0.requires_grad_()
                                def Tx(x):
                                    x = x / 2 + 0.5
                                    pad_2d = torch.nn.ReflectionPad2d(k.shape[0]//2)
                                    x_deblur = F.conv2d(pad_2d(x), k_4d)
                                    return x_deblur
                                norm_grad, norm = utils_model.grad_and_value(operator=Tx, x=x0, x_hat=x0, measurement=y)
                                x0 = x0 - norm_grad * norm / (rhos[t_i])
                                x0 = x0.detach_()
                        elif 'DPS' in generate_mode:
                            def Tx(x):
                                x = x / 2 + 0.5
                                pad_2d = torch.nn.ReflectionPad2d(k.shape[0]//2)
                                x_deblur = F.conv2d(pad_2d(x), k_4d)
                                return x_deblur                         
                            if generate_mode == 'DPS_y0':
                                norm_grad, norm = utils_model.grad_and_value(operator=Tx, x=x, x_hat=x0, measurement=y)
                                x = xt - norm_grad * 1.
                                x = x.detach_()
                            elif generate_mode == 'DPS_yt':
                                y_t = sqrt_alphas_cumprod[t_i] * (2*y - 1) + sqrt_1m_alphas_cumprod[t_i] * torch.randn_like(y)
                                y_t = y_t / 2 + 0.5
                                norm_grad, norm = utils_model.grad_and_value(operator=Tx, x=xt, x_hat=xt, measurement=y_t)
                                x = xt - norm_grad * lambda_ * norm / (rhos[t_i]) * 0.35
                                x = x.detach_()
                    
                    if (generate_mode == 'DiffPIR' and model_out_type == 'pred_xstart') and not (seq[i] == seq[-1] and u == iter_num_U - 1):
                        t_im1 = utils_model.find_nearest(reduced_alpha_cumprod, sigmas[seq[i+1]].cpu().numpy())
                        eps = (x - sqrt_alphas_cumprod[t_i] * x0) / sqrt_1m_alphas_cumprod[t_i]
                        eta_sigma = eta * sqrt_1m_alphas_cumprod[t_im1] / sqrt_1m_alphas_cumprod[t_i] * torch.sqrt(betas[t_i])
                        x = sqrt_alphas_cumprod[t_im1] * x0 + np.sqrt(1-zeta) * (torch.sqrt(sqrt_1m_alphas_cumprod[t_im1]**2 - eta_sigma**2) * eps \
                                    + eta_sigma * torch.randn_like(x)) + np.sqrt(zeta) * sqrt_1m_alphas_cumprod[t_im1] * torch.randn_like(x)
                    if u < iter_num_U - 1 and seq[i] != seq[-1]:
                        sqrt_alpha_effective = sqrt_alphas_cumprod[t_i] / sqrt_alphas_cumprod[t_im1]
                        x = sqrt_alpha_effective * x + torch.sqrt(sqrt_1m_alphas_cumprod[t_i]**2 - 
                                sqrt_alpha_effective**2 * sqrt_1m_alphas_cumprod[t_im1]**2) * torch.randn_like(x)
                    
                    if save_progressive and (seq[i] in progress_seq):
                        x_show = x.clone().detach().cpu().numpy()
                        x_show = np.squeeze(x_show)
                        if x_show.ndim == 3:
                            x_show = np.transpose(x_show, (1, 2, 0))
                        progress_img.append(x_show)
                        if log_process:
                            logger.info('{:>4d}, steps: {:>4d}, np.max(x_show): {:.4f}, np.min(x_show): {:.4f}'.format(seq[i], t_i, np.max(x_show), np.min(x_show)))
                        if show_img:
                            util.imshow(x_show)
            
            # ---------------------------
            # (6) 恢复并上采样结果
            # ---------------------------
            # x_0 为处理尺寸下的结果，此处先转换为 uint，再上采样回原始尺寸
            x_0 = (x/2+0.5)
            x0_up = F.interpolate(x_0, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
            img_E = util.tensor2uint(x0_up)
                
            psnr = util.calculate_psnr(img_E, img_H, border=border)
            test_results['psnr'].append(psnr)
            
            if calc_LPIPS:
                img_H_tensor = np.transpose(img_H, (2, 0, 1))
                img_H_tensor = torch.from_numpy(img_H_tensor)[None,:,:,:].to(device)
                img_H_tensor = img_H_tensor / 255 * 2 - 1
                lpips_score = loss_fn_vgg(x0_up.detach()*2-1, img_H_tensor)
                lpips_score = lpips_score.cpu().detach().numpy()[0][0][0][0]
                test_results['lpips'].append(lpips_score)
                logger.info('{:->4d}--> {:>10s} PSNR: {:.4f} dB LPIPS: {:.4f} ave LPIPS: {:.4f}'.format(
                    idx+1, img_name+ext, psnr, lpips_score, sum(test_results['lpips']) / len(test_results['lpips'])))
            else:
                logger.info('{:->4d}--> {:>10s} PSNR: {:.4f} dB'.format(idx+1, img_name+ext, psnr))

            if n_channels == 1:
                img_H = img_H.squeeze()

            if save_E:
                util.imsave(img_E, os.path.join(E_path, img_name+'_'+model_name+ext))

            if save_progressive:
                now = datetime.now()
                current_time = now.strftime("%Y_%m_%d_%H_%M_%S")
                img_total = cv2.hconcat(progress_img)
                if show_img:
                    util.imshow(img_total, figsize=(80,4))
                util.imsave(img_total*255., os.path.join(E_path, img_name+'_sigma_{:.3f}_process_lambda_{:.3f}_{}_psnr_{:.4f}{}'.format(noise_level_img, lambda_, current_time, psnr, ext)))
                                                                            
            if save_LEH:
                img_L_uint = util.single2uint(img_L)
                k_v = k/np.max(k)*1.0
                k_v = util.single2uint(np.tile(k_v[..., np.newaxis], [1, 1, 3]))
                k_v = cv2.resize(k_v, (3*k_v.shape[1], 3*k_v.shape[0]), interpolation=cv2.INTER_NEAREST)
                img_I = cv2.resize(img_L_uint, (sf*img_L_uint.shape[1], sf*img_L_uint.shape[0]), interpolation=cv2.INTER_NEAREST)
                img_I[:k_v.shape[0], -k_v.shape[1]:, :] = k_v
                img_I[:img_L_uint.shape[0], :img_L_uint.shape[1], :] = img_L_uint
                if show_img:
                    util.imshow(np.concatenate([img_I, img_E, img_H], axis=1), title='LR / Recovered / Ground-truth')
                util.imsave(np.concatenate([img_I, img_E, img_H], axis=1), os.path.join(E_path, img_name+'_LEH'+ext))

            if save_L:
                util.imsave(util.single2uint(img_L), os.path.join(E_path, img_name+'_LR'+ext))
        
        # ---------------------------
        # 计算平均 PSNR 和 LPIPS
        # ---------------------------
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        logger.info('------> Average PSNR of ({}), sigma: ({:.3f}): {:.4f} dB'.format(testset_name, noise_level_model, ave_psnr))

        if calc_LPIPS:
            ave_lpips = sum(test_results['lpips']) / len(test_results['lpips'])
            logger.info('------> Average LPIPS of ({}) sigma: ({:.3f}): {:.4f}'.format(testset_name, noise_level_model, ave_lpips))
    
    # ---------------------------
    # 进行实验
    # ---------------------------
    lambdas = [lambda_ * i for i in range(7,8)]
    for lambda_ in lambdas:
        for zeta_i in [zeta * i for i in range(3,4)]:
            test_rho(lambda_, zeta=zeta_i, model_output_type=model_output_type)


if __name__ == '__main__':
    main()
