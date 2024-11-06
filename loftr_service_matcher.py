import os
#os.chdir("..")
from copy import deepcopy

import sys
import torch
import cv2
import numpy as np
#import matplotlib.cm as cm
#from src.utils.plotting import make_matching_figure
from src.loftr import LoFTR, default_cfg
#from kornia.feature import LoFTR, default_cfg
#import matplotlib
#import matplotlib.pyplot as plt
#matplotlib.use('TkAgg')
import socket
import time

#/mnt/ssd/data/GEO/slam-test/GS010035/images/000000.jpg /mnt/ssd/data/GEO/slam-test/tmp_data/Miroslav/images/001320.jpg /mnt/ssd/data/GEO/slam-test/tmp_data/Miroslav/images/001321.jpg /mnt/ssd/data/GEO/slam-test/tmp_data/Miroslav/images/001319.jpg

port = sys.argv[2]
try:
    port = int(port)  # Convert to integer
    print(f"The argument as integer: {port}")
except ValueError:
    port = 59434
    print(f"Error: '{port}' is not a valid integer")

output_path = sys.argv[3]

matcher = LoFTR(config=default_cfg)
matcher.load_state_dict(torch.load(sys.argv[1], {'weights_only'   : True})['state_dict'])
matcher = matcher.eval().cuda()

default_cfg['coarse']
{'d_model': 256,
 'd_ffn': 256,
 'nhead': 8,
 'layer_names': ['self',
  'cross',
  'self',
  'cross',
  'self',
  'cross',
  'self',
  'cross'],
 'attention': 'linear',
 'temp_bug_fix': False}

def receive_full_message(conn, buffer_size=1024):
    data = b''  
    while True:
        part = conn.recv(buffer_size)
        data += part
        if len(part) < buffer_size:
            break
    return data.decode('utf-8')

def image_processing_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('localhost', port))
    server_socket.listen(5)
    print("Server is listening on port", port)
    conn, addr = server_socket.accept()

    while True:
        file_paths = receive_full_message(conn)
        #print("Received message:", file_paths)

        if file_paths.strip() == "shutdown":
            print("Shutdown command received. Server is shutting down.")
            conn.sendall(b"Server shutting down")
            conn.close()
            break

        file_paths_list = file_paths.split('\n')
        process_images(file_paths_list)

        conn.sendall(b"done")
        #conn.close()

    server_socket.close()
    print("Server has shut down.")

def concatenate_filenames(paths):
    filenames = [os.path.splitext(os.path.basename(path))[0] for path in paths]
    concatenated_filenames = '_'.join(filenames)
    return concatenated_filenames

def safe_imread(path, retries=5, delay=0.2):
    """
    Attempts to read an image file with retries in case of rare access conflicts.
    """
    for attempt in range(retries):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img
        time.sleep(delay)  # Wait a bit before retrying
    raise FileNotFoundError(f"Failed to read {path} after {retries} retries.")

def process_images(file_paths_list):
    img_ref = file_paths_list[0]
    img_list = file_paths_list[1:-1]
    print(img_ref)
    print(img_list)

    #img_ref_raw = cv2.imread(img_ref, cv2.IMREAD_GRAYSCALE)
    img_ref_raw = safe_imread(img_ref)
    factor_raw = 64 if img_ref_raw.shape[0] > 3000 else 32
    img_ref_raw = cv2.resize(img_ref_raw, (img_ref_raw.shape[1]//factor_raw*8, img_ref_raw.shape[0]//factor_raw*8))
    img_ref_torch = torch.from_numpy(img_ref_raw)[None][None].cuda() / 255.
    # matching for each image in list
    cnt = 0
    for img in img_list:
        img1_pth = img
        img1_raw = cv2.imread(img1_pth, cv2.IMREAD_GRAYSCALE)

        factor1 = 64 if img1_raw.shape[0] > 3000 else 32
        img1_raw = cv2.resize(img1_raw, (img1_raw.shape[1]//factor1*8, img1_raw.shape[0]//factor1*8))
        img1_torch = torch.from_numpy(img1_raw)[None][None].cuda() / 255.

        batch = {'image0': img_ref_torch, 'image1': img1_torch}

        with torch.no_grad():
            matcher(batch)
            mkpts0 = batch['mkpts0_f'].cpu().numpy()
            mkpts1 = batch['mkpts1_f'].cpu().numpy()
            mconf = batch['mconf'].cpu().numpy()

        #outfile = output_path + concatenate_filenames([img_ref, img1_pth]) + "_" + str(cnt) + ".txt"

        tmpfile = output_path + concatenate_filenames([img_ref, img1_pth]) + "_" + str(cnt) + ".tmp"
        finalfile = output_path + concatenate_filenames([img_ref, img1_pth]) + "_" + str(cnt) + ".txt"

        with open(tmpfile, 'w') as f:
            f.write(str(mkpts0.shape[0]) + "\n")
            scaled_data0 = np.copy(mkpts0)
            scaled_data0[:, 0] /= img_ref_raw.shape[1]
            scaled_data0[:, 1] /= img_ref_raw.shape[0]
            np.savetxt(f, scaled_data0, fmt='%.6f', delimiter=' ')
            f.write(str(mkpts1.shape[0]) + "\n")
            scaled_data1 = np.copy(mkpts1)
            scaled_data1[:, 0] /= img1_raw.shape[1]
            scaled_data1[:, 1] /= img1_raw.shape[0]
            np.savetxt(f, scaled_data1, fmt='%.6f', delimiter=' ')

        os.rename(tmpfile, finalfile)
        cnt += 1

# Run the server
image_processing_server()