import os.path
import cv2
import logging

import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
from collections import OrderedDict
import hdf5storage

from utils import utils_model
from utils import utils_logger
from utils import utils_sisr as sr
from utils import utils_image as util
from utils.utils_resizer import Resizer
from functools import partial

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

    noise_level_img         = 12.75/255.0       # AWGN noise level for LR image
    noise_level_model       = noise_level_img   # set noise level of model
    model_name              = 'diffusion_ffhq_10m'  # diffusion_ffhq_10m, 256x256_diffusion_uncond
    testset_name            = 'apgd_test'    # testing set, e.g. 'imagenet_val' or 'ffhq_val'
    num_train_timesteps     = 1000
    iter_num                = 100             # number of sampling iterations
    iter_num_U              = 1               # inner iterations
    skip                    = num_train_timesteps // iter_num   # skip interval
    sr_mode                 = 'blur'          # 'blur' 或 'cubic' mode of sr up/down sampling

    show_img                = False           # 是否显示图像
    save_L                  = False            # 保存 LR 图像
    save_E                  = True           # 保存恢复后图像
    save_LEH                = False           # 保存 LR/E/GT 拼接图
    save_progressive        = False            # 保存生成过程

    sigma                   = max(0.001, noise_level_img)   # noise level associated with condition y
    lambda_                 = 1.0             # 关键参数 lambda
    sub_1_analytic          = True            # 是否使用解析解

    log_process             = False
    ddim_sample             = False           # sampling method
    model_output_type       = 'pred_xstart'   # 模型输出类型: 'pred_x_prev'、'pred_xstart'、'epsilon'、'score'
    generate_mode           = 'DiffPIR'       # DiffPIR; DPS; vanilla
    skip_type               = 'quad'          # 'uniform' 或 'quad'
    eta                     = 0.0             # eta for ddim sampling
    zeta                    = 0.1
    guidance_scale          = 1.0

    test_sf                 = [4]             # scale factor, e.g. [2, 3, 4]
    inIter                  = 1               # iter num for sr solution, e.g. 4~6
    gamma                   = 1/100           # coefficient for iterative sr solver
    classical_degradation   = False           # classical degradation or bicubic degradation
    task_current            = 'sr'            # 'sr' for super resolution
    n_channels              = 3               # 固定为 3
    cwd                     = ''
    model_zoo               = os.path.join(cwd, 'model_zoo')
    testsets                = os.path.join(cwd, 'testsets')
    results                 = os.path.join(cwd, 'results')
    result_name             = f'{testset_name}_{task_current}_{generate_mode}_{sr_mode}{str(test_sf)}_{model_name}_sigma{noise_level_img}_NFE{iter_num}_eta{eta}_zeta{zeta}_lambda{lambda_}'
    model_path              = os.path.join(model_zoo, model_name+'.pt')
    device                  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    calc_LPIPS              = True

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
    # L_path, E_path, H_path
    # ----------------------------------------

    L_path = os.path.join(testsets, testset_name)  # 输入图像目录
    E_path = os.path.join(results, result_name)    # 恢复结果保存目录
    util.mkdir(E_path)

    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    # ----------------------------------------
    # load model
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

    logger.info('model_name:{}, sr_mode:{}, image sigma:{:.3f}, model sigma:{:.3f}'.format(model_name, sr_mode, noise_level_img, noise_level_model))
    logger.info('eta:{:.3f}, zeta:{:.3f}, lambda:{:.3f}, guidance_scale:{:.2f}'.format(eta, zeta, lambda_, guidance_scale))
    logger.info('start step:{}, skip_type:{}, skip interval:{}, skipstep analytic steps:{}'.format(t_start, skip_type, skip, noise_model_t))
    logger.info('analytic iter num:{}, gamma:{}'.format(inIter, gamma))
    logger.info('Model path: {:s}'.format(model_path))
    logger.info(L_path)
    L_paths = util.get_image_paths(L_path)

    # --------------------------------
    # load kernel
    # --------------------------------

    if classical_degradation:
        kernels = hdf5storage.loadmat(os.path.join(cwd, 'kernels', 'kernels_12.mat'))['kernels']
    else:
        kernels = hdf5storage.loadmat(os.path.join(cwd, 'kernels', 'kernels_bicubicx234.mat'))['kernels']

    test_results_ave = OrderedDict()
    test_results_ave['psnr_sf_k'] = []
    test_results_ave['psnr_y_sf_k'] = []
    if calc_LPIPS:
        import lpips
        loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
        test_results_ave['lpips'] = []

    for sf in test_sf:
        border = sf
        k_num = 8 if classical_degradation else 1

        for k_index in range(k_num):
            logger.info('--------- sf:{:>1d} --k:{:>2d} ---------'.format(sf, k_index))

            if not classical_degradation:  # for bicubic degradation
                k_index = sf-2 if sf < 5 else 2
            k = kernels[0, k_index].astype(np.float64)

            util.surf(k) if show_img else None

            def test_rho(lambda_=lambda_, zeta=zeta, model_output_type=model_output_type):
                logger.info('eta:{:.3f}, zeta:{:.3f}, lambda:{:.3f}, inIter:{:.3f}, gamma:{:.3f}, guidance_scale:{:.2f}'.format(
                    eta, zeta, lambda_, inIter, gamma, guidance_scale))
                test_results = OrderedDict()
                test_results['psnr'] = []
                test_results['psnr_y'] = []
                if calc_LPIPS:
                    test_results['lpips'] = []
                for idx, img in enumerate(L_paths):
                    model_out_type = model_output_type

                    # --------------------------------
                    # (1) 获取 HR 图像并缩放到固定处理尺寸
                    # --------------------------------
                    img_name, ext = os.path.splitext(os.path.basename(img))
                    # 先读取原始高分辨率图像
                    img_H_orig = util.imread_uint(img, n_channels=n_channels)
                    orig_h, orig_w, _ = img_H_orig.shape
                    # 将 HR 图像缩放到固定处理尺寸（例如 256×256）
                    proc_size = (256, 256)
                    img_H = cv2.resize(img_H_orig, proc_size, interpolation=cv2.INTER_AREA)
                    # modcrop（保证尺寸能被 sf 整除）
                    img_H = util.modcrop(img_H, sf)

                    # --------------------------------
                    # (2) 生成低质量图像 img_L
                    # --------------------------------
                    if sr_mode == 'blur':
                        if classical_degradation:
                            img_L = sr.classical_degradation(img_H, k, sf)
                            if show_img:
                                util.imshow(img_L)
                            img_L = util.uint2single(img_L)
                        else:
                            img_L = util.imresize_np(util.uint2single(img_H), 1/sf)
                    elif sr_mode == 'cubic':
                        img_H_tensor = np.transpose(img_H, (2, 0, 1))
                        img_H_tensor = torch.from_numpy(img_H_tensor)[None,:,:,:].to(device)
                        img_H_tensor = img_H_tensor / 255
                        # set up resizers
                        up_sample = partial(F.interpolate, scale_factor=sf)
                        down_sample = Resizer(img_H_tensor.shape, 1/sf).to(device)
                        img_L = down_sample(img_H_tensor)
                        img_L = img_L.cpu().numpy()  # [0,1]
                        img_L = np.squeeze(img_L)
                        if img_L.ndim == 3:
                            img_L = np.transpose(img_L, (1, 2, 0))

                    np.random.seed(seed=0)  # 保证可复现
                    img_L = img_L * 2 - 1
                    img_L += np.random.normal(0, noise_level_img * 2, img_L.shape)  # add AWGN
                    img_L = img_L / 2 + 0.5

                    # --------------------------------
                    # (3) 计算 rhos 和 sigmas
                    # -------------------------------- 
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
                    
                    # --------------------------------
                    # (4) 初始化 x 与预计算
                    # --------------------------------
                    # 使用 cv2.resize 对 LR 图像进行上采样，得到初始 x（处理尺寸下的 HR 初始估计）
                    x = cv2.resize(img_L, (img_L.shape[1]*sf, img_L.shape[0]*sf), interpolation=cv2.INTER_CUBIC)
                    if np.ndim(x) == 2:
                        x = x[..., None]
                    if classical_degradation:
                        x = sr.shift_pixel(x, sf)
                    x = util.single2tensor4(x).to(device)

                    y = util.single2tensor4(img_L).to(device)  # (1,3,H_lr,W_lr) [0,1]

                    # 为 y 加上对应噪声
                    x = sqrt_alphas_cumprod[t_start] * (2*x - 1) + sqrt_1m_alphas_cumprod[t_start] * torch.randn_like(x)

                    k_tensor = util.single2tensor4(np.expand_dims(k, 2)).to(device)
                    FB, FBC, F2B, FBFy = sr.pre_calculate(y, k_tensor, sf)

                    # --------------------------------
                    # (5) 主迭代过程
                    # --------------------------------
                    progress_img = []
                    skip = num_train_timesteps // iter_num
                    if skip_type == 'uniform':
                        seq = [i*skip for i in range(iter_num)]
                        if skip > 1:
                            seq.append(num_train_timesteps-1)
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
                            # --------------------------------
                            # step 1, reverse diffusion step
                            # --------------------------------
                            if 'DPS' in generate_mode:
                                x = x.requires_grad_()
                                xt, x0 = utils_model.model_fn(x, noise_level=curr_sigma*255, model_out_type='pred_x_prev_and_start',
                                        model_diffusion=model, diffusion=diffusion, ddim_sample=ddim_sample, alphas_cumprod=alphas_cumprod)
                            else:
                                x0 = utils_model.model_fn(x, noise_level=curr_sigma*255, model_out_type=model_out_type,
                                        model_diffusion=model, diffusion=diffusion, ddim_sample=ddim_sample, alphas_cumprod=alphas_cumprod)
                            
                            # --------------------------------
                            # step 2, FFT / 解析求解
                            # --------------------------------
                            if seq[i] != seq[-1]:
                                if generate_mode == 'DiffPIR':
                                    if sub_1_analytic:
                                        if model_out_type == 'pred_xstart':
                                            if i < num_train_timesteps - noise_model_t:
                                                if sr_mode == 'blur':
                                                    tau = rhos[t_i].float().repeat(1, 1, 1, 1)
                                                    x0_p = x0 / 2 + 0.5
                                                    x0_p = sr.data_solution(x0_p.float(), FB, FBC, F2B, FBFy, tau, sf)
                                                    x0_p = x0_p * 2 - 1
                                                    x0 = x0 + guidance_scale * (x0_p - x0)
                                                elif sr_mode == 'cubic': 
                                                    for _ in range(inIter):
                                                        x0 = x0 / 2 + 0.5
                                                        x0 = x0 + gamma * up_sample((y - down_sample(x0))) / (1 + rhos[t_i])
                                                        x0 = x0 * 2 - 1
                                            else:
                                                model_out_type = 'pred_x_prev'
                                                x0 = utils_model.model_fn(x, noise_level=curr_sigma*255, model_out_type=model_out_type,
                                                        model_diffusion=model, diffusion=diffusion, ddim_sample=ddim_sample, alphas_cumprod=alphas_cumprod)
                                        else:
                                            x = (y + rhos[t_i].float() * x).div(1 + rhos[t_i])
                                    else:
                                        x0 = x0.requires_grad_()
                                        down_sample = Resizer(x.shape, 1/sf).to(device)
                                        norm_grad, norm = utils_model.grad_and_value(operator=down_sample, x=x0, x_hat=x0, measurement=2*y-1)
                                        x0 = x0 - norm_grad * norm / (rhos[t_i])
                                        x0 = x0.detach_()
                                elif 'DPS' in generate_mode:
                                    down_sample = Resizer(x.shape, 1/sf).to(device)
                                    if generate_mode == 'DPS_y0':
                                        norm_grad, norm = utils_model.grad_and_value(operator=down_sample, x=x, x_hat=x0, measurement=2*y-1)
                                        x = xt - norm_grad * 1.
                                        x = x.detach_()
                                    elif generate_mode == 'DPS_yt':
                                        y_t = sqrt_alphas_cumprod[t_i] * (2*y-1) + sqrt_1m_alphas_cumprod[t_i] * torch.randn_like(y)
                                        norm_grad, norm = utils_model.grad_and_value(operator=down_sample, x=xt, x_hat=xt, measurement=y_t)
                                        x = xt - norm_grad * lambda_ * norm / (rhos[t_i]) * 0.35
                                        x = x.detach_()
                            
                            if (generate_mode == 'DiffPIR' and model_out_type == 'pred_xstart') and not (seq[i] == seq[-1] and u == iter_num_U-1):
                                t_im1 = utils_model.find_nearest(reduced_alpha_cumprod, sigmas[seq[i+1]].cpu().numpy())
                                eps = (x - sqrt_alphas_cumprod[t_i] * x0) / sqrt_1m_alphas_cumprod[t_i]
                                eta_sigma = eta * sqrt_1m_alphas_cumprod[t_im1] / sqrt_1m_alphas_cumprod[t_i] * torch.sqrt(betas[t_i])
                                x = sqrt_alphas_cumprod[t_im1] * x0 + np.sqrt(1-zeta) * (torch.sqrt(sqrt_1m_alphas_cumprod[t_im1]**2 - eta_sigma**2) * eps \
                                            + eta_sigma * torch.randn_like(x)) + np.sqrt(zeta) * sqrt_1m_alphas_cumprod[t_im1] * torch.randn_like(x)
                            else:
                                pass
                            
                            if u < iter_num_U-1 and seq[i] != seq[-1]:
                                sqrt_alpha_effective = sqrt_alphas_cumprod[t_i] / sqrt_alphas_cumprod[t_im1]
                                x = sqrt_alpha_effective * x + torch.sqrt(sqrt_1m_alphas_cumprod[t_i]**2 - 
                                        sqrt_alpha_effective**2 * sqrt_1m_alphas_cumprod[t_im1]**2) * torch.randn_like(x)
                        
                        if save_progressive and (seq[i] in progress_seq):
                            x_0 = (x/2+0.5)
                            x_show = x_0.clone().detach().cpu().numpy()
                            x_show = np.squeeze(x_show)
                            if x_show.ndim == 3:
                                x_show = np.transpose(x_show, (1, 2, 0))
                            progress_img.append(x_show)
                            if log_process:
                                logger.info('{:>4d}, steps: {:>4d}, np.max(x_show): {:.4f}, np.min(x_show): {:.4f}'.format(seq[i], t_i, np.max(x_show), np.min(x_show)))
                            if show_img:
                                util.imshow(x_show)
                    
                    # --------------------------------
                    # (6) 恢复结果并上采样回原始尺寸
                    # --------------------------------
                    x_0 = (x/2+0.5)
                    # 将处理结果上采样回原始 HR 尺寸
                    x0_up = F.interpolate(x_0, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
                    img_E = util.tensor2uint(x0_up)
                        
                    psnr = util.calculate_psnr(img_E, img_H_orig, border=border)
                    test_results['psnr'].append(psnr)
                    
                    if calc_LPIPS:
                        img_H_tensor = np.transpose(img_H_orig, (2, 0, 1))
                        img_H_tensor = torch.from_numpy(img_H_tensor)[None,:,:,:].to(device)
                        img_H_tensor = img_H_tensor / 255 * 2 - 1
                        lpips_score = loss_fn_vgg(x0_up.detach()*2-1, img_H_tensor)
                        lpips_score = lpips_score.cpu().detach().numpy()[0][0][0][0]
                        test_results['lpips'].append(lpips_score)
                        logger.info('{:->4d}--> {:>10s} -- sf:{:>1d} --k:{:>2d} PSNR: {:.4f} dB LPIPS: {:.4f} ave LPIPS: {:.4f}'.format(
                            idx+1, img_name+ext, sf, k_index, psnr, lpips_score, sum(test_results['lpips']) / len(test_results['lpips'])))
                    else:
                        logger.info('{:->4d}--> {:>10s} -- sf:{:>1d} --k:{:>2d} PSNR: {:.4f} dB'.format(idx+1, img_name+ext, sf, k_index, psnr))

                    if save_E:
                        util.imsave(img_E, os.path.join(E_path, img_name+'_x'+str(sf)+'_k'+str(k_index)+'_'+model_name+ext))

                    if n_channels == 1:
                        img_H_orig = img_H_orig.squeeze()

                    if save_progressive:
                        now = datetime.now()
                        current_time = now.strftime("%Y_%m_%d_%H_%M_%S")
                        img_total = cv2.hconcat(progress_img)
                        if show_img:
                            util.imshow(img_total, figsize=(80,4))
                        util.imsave(img_total*255., os.path.join(E_path, img_name+'_sigma_{:.3f}_process_lambda_{:.3f}_{}_psnr_{:.4f}{}'.format(noise_level_img, lambda_, current_time, psnr, ext)))
                        
                    # --------------------------------
                    # (7) 拼接展示 LR / Recovered / GT（可选）
                    # --------------------------------
                    img_L_uint = util.single2uint(img_L).squeeze()
                    if save_LEH:
                        k_v = k/np.max(k)*1.0
                        if n_channels==1:
                            k_v = util.single2uint(k_v)
                        else:
                            k_v = util.single2uint(np.tile(k_v[..., np.newaxis], [1, 1, n_channels]))
                        k_v = cv2.resize(k_v, (3*k_v.shape[1], 3*k_v.shape[0]), interpolation=cv2.INTER_NEAREST)
                        img_I = cv2.resize(img_L_uint, (sf*img_L_uint.shape[1], sf*img_L_uint.shape[0]), interpolation=cv2.INTER_NEAREST)
                        img_I[:k_v.shape[0], -k_v.shape[1]:, ...] = k_v
                        img_I[:img_L_uint.shape[0], :img_L_uint.shape[1], ...] = img_L_uint
                        if show_img:
                            util.imshow(np.concatenate([img_I, img_E, img_H_orig], axis=1), title='LR / Recovered / Ground-truth')
                        util.imsave(np.concatenate([img_I, img_E, img_H_orig], axis=1), os.path.join(E_path, img_name+'_x'+str(sf)+'_k'+str(k_index)+'_LEH'+ext))

                    if save_L:
                        util.imsave(img_L_uint, os.path.join(E_path, img_name+'_x'+str(sf)+'_k'+str(k_index)+'_LR'+ext))
                    
                    if n_channels == 3:
                        img_E_y = util.rgb2ycbcr(img_E, only_y=True)
                        img_H_y = util.rgb2ycbcr(img_H_orig, only_y=True)
                        psnr_y = util.calculate_psnr(img_E_y, img_H_y, border=border)
                        test_results['psnr_y'].append(psnr_y)
                    
                # --------------------------------
                # Average PSNR and LPIPS for current scale and kernel
                # --------------------------------
                ave_psnr_k = sum(test_results['psnr']) / len(test_results['psnr'])
                logger.info('------> Average PSNR(RGB) of ({}) scale factor: ({}), kernel: ({}) sigma: ({:.3f}): {:.4f} dB'.format(
                    testset_name, sf, k_index, noise_level_model, ave_psnr_k))
                test_results_ave['psnr_sf_k'].append(ave_psnr_k)

                if n_channels == 3:
                    ave_psnr_y_k = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
                    logger.info('------> Average PSNR(Y) of ({}) scale factor: ({}), kernel: ({}) sigma: ({:.3f}): {:.4f} dB'.format(
                        testset_name, sf, k_index, noise_level_model, ave_psnr_y_k))
                    test_results_ave['psnr_y_sf_k'].append(ave_psnr_y_k)

                if calc_LPIPS:
                    ave_lpips_k = sum(test_results['lpips']) / len(test_results['lpips'])
                    logger.info('------> Average LPIPS of ({}) scale factor: ({}), kernel: ({}) sigma: ({:.3f}): {:.4f}'.format(
                        testset_name, sf, k_index, noise_level_model, ave_lpips_k))
                    test_results_ave['lpips'].append(ave_lpips_k)
                return test_results_ave

            # experiments: 对不同 lambda 和 zeta 进行测试
            lambdas = [lambda_ * i for i in range(2, 13)]
            for lambda_ in lambdas:
                for zeta_i in [0.25]:
                    test_results_ave = test_rho(lambda_, zeta=zeta_i, model_output_type=model_output_type)

    # ---------------------------------------
    # Overall Average PSNR and LPIPS for all scale factors and kernels
    # ---------------------------------------

    ave_psnr_sf_k = sum(test_results_ave['psnr_sf_k']) / len(test_results_ave['psnr_sf_k'])
    logger.info('------> Average PSNR of ({}) {:.4f} dB'.format(testset_name, ave_psnr_sf_k))
    if n_channels == 3:
        ave_psnr_y_sf_k = sum(test_results_ave['psnr_y_sf_k']) / len(test_results_ave['psnr_y_sf_k'])
        logger.info('------> Average PSNR-Y of ({}) {:.4f} dB'.format(testset_name, ave_psnr_y_sf_k))
    if calc_LPIPS:
        ave_lpips_sf_k = sum(test_results_ave['lpips']) / len(test_results_ave['lpips'])
        logger.info('------> Average LPIPS of ({}) {:.4f}'.format(testset_name, ave_lpips_sf_k))

if __name__ == '__main__':
    main()
