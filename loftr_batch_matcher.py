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

def concatenate_filenames(paths):
    filenames = [os.path.splitext(os.path.basename(path))[0] for path in paths]
    concatenated_filenames = '_'.join(filenames)
    return concatenated_filenames

# Check if the correct number of arguments are provided
if len(sys.argv) < 5:
    print("Usage: python loftr_batch_matcher.py weights_file img_ref img0 ... imgN path_to_output")
    sys.exit(1)

matcher = LoFTR(config=default_cfg)
#matcher.load_state_dict(torch.load("/mnt/ssd/data/LoFTR/weights/outdoor_ds.ckpt")['state_dict'])
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

img_ref = sys.argv[2]
img_list = sys.argv[3:-1]
print(img_ref)
print(img_list)
print(sys.argv[-1])

img_ref_raw = cv2.imread(img_ref, cv2.IMREAD_GRAYSCALE)
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

    outfile = sys.argv[-1] + concatenate_filenames([img_ref, img1_pth]) + "_" + str(cnt) + ".txt"

    with open(outfile, 'w') as f:
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

    cnt += 1