import cv2
import numpy as np
import os
import sys
from datetime import datetime

# 全局目录设置
CHUNKS_DIRECTORY = '../data/'
RESULTS_DIRECTORY = 'results/vid_test1/'
verbose = 0

# 预处理函数：对视频帧进行 resize，并按照攻击后图像的格式进行 YUV/BGR 转换
def process_frame(frame, width=512, height=256):
    # 调整尺寸
    dim = (width, height)
    img = cv2.resize(frame, dim)
    # 转换为 YUV_I420 格式（与攻击代码一致），再转换回 BGR 格式以保存
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420).astype('float32')
    clean_frame = np.clip(img_yuv, 0, 255).astype(np.uint8)
    clean_bgr = cv2.cvtColor(clean_frame, cv2.COLOR_YUV2BGR_I420)
    return clean_bgr

# 只保存干净视频帧，不执行任何攻击操作
def save_clean_video(video_path, video_id, chunk):
    cap = cv2.VideoCapture(video_path)
    
    # 设置保存目录，结构与攻击后一致：RESULTS_DIRECTORY/chunk_{chunk}/clean
    chunk_dir = os.path.join(RESULTS_DIRECTORY, f'chunk_{chunk}')
    save_dir_clean = os.path.join(chunk_dir, 'clean')
    if not os.path.exists(save_dir_clean):
        os.makedirs(save_dir_clean)
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        
        # 对视频帧进行预处理
        clean_bgr = process_frame(frame)
        # 按照 “video{video_id}_frame{frame_idx}.png” 的命名规则保存图像
        save_path = os.path.join(save_dir_clean, f'video{video_id}_frame{frame_idx}.png')
        cv2.imwrite(save_path, clean_bgr)
        
        if verbose > 0 and frame_idx % 200 == 0:
            print(f'Processed {frame_idx} frames for video {video_id}')
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # 根据命令行参数设置 verbose 级别
    if len(sys.argv) > 1:
        verbose = int(sys.argv[1])
    
    # 如果结果目录不存在则创建
    if not os.path.exists(RESULTS_DIRECTORY):
        os.mkdir(RESULTS_DIRECTORY)
    
    video_id = 0
    chunk = 3  # 根据需要修改 chunk 编号
    chunk_results_dir = os.path.join(RESULTS_DIRECTORY, f'chunk_{chunk}')
    if not os.path.exists(chunk_results_dir):
        os.mkdir(chunk_results_dir)
    
    # 保存视频路径映射文件
    map_path = os.path.join(chunk_results_dir, 'filepaths.txt')
    if not os.path.exists(map_path):
        with open(map_path, 'w') as f:
            f.write('video_id|path\n')
    
    # 遍历视频所在目录
    chunk_path = os.path.join(CHUNKS_DIRECTORY, f'Chunk_{chunk}/')
    for root, dirs, files in os.walk(chunk_path):
        for file in files:
            if file.endswith('.hevc'):
                video_path = os.path.join(root, file)
                video_id += 1
                with open(map_path, 'a') as f:
                    f.write(f'{video_id}|{video_path}\n')
                print('Starting video', video_id)
                start = datetime.now()
                save_clean_video(video_path, video_id, chunk)
                dur = datetime.now() - start
                print(dur, 'for file', video_id)
