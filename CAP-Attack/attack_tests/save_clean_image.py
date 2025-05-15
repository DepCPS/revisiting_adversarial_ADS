import cv2
import numpy as np
import os
import sys
from datetime import datetime


CHUNKS_DIRECTORY = '../data/'
RESULTS_DIRECTORY = 'results/vid_test1/'
verbose = 0


def process_frame(frame, width=512, height=256):

    dim = (width, height)
    img = cv2.resize(frame, dim)

    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420).astype('float32')
    clean_frame = np.clip(img_yuv, 0, 255).astype(np.uint8)
    clean_bgr = cv2.cvtColor(clean_frame, cv2.COLOR_YUV2BGR_I420)
    return clean_bgr


def save_clean_video(video_path, video_id, chunk):
    cap = cv2.VideoCapture(video_path)
    

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
        

        clean_bgr = process_frame(frame)

        save_path = os.path.join(save_dir_clean, f'video{video_id}_frame{frame_idx}.png')
        cv2.imwrite(save_path, clean_bgr)
        
        if verbose > 0 and frame_idx % 200 == 0:
            print(f'Processed {frame_idx} frames for video {video_id}')
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':

    if len(sys.argv) > 1:
        verbose = int(sys.argv[1])
    

    if not os.path.exists(RESULTS_DIRECTORY):
        os.mkdir(RESULTS_DIRECTORY)
    
    video_id = 0
    chunk = 3  
    chunk_results_dir = os.path.join(RESULTS_DIRECTORY, f'chunk_{chunk}')
    if not os.path.exists(chunk_results_dir):
        os.mkdir(chunk_results_dir)
    

    map_path = os.path.join(chunk_results_dir, 'filepaths.txt')
    if not os.path.exists(map_path):
        with open(map_path, 'w') as f:
            f.write('video_id|path\n')
    

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
