import os
#os.chdir("..")
from copy import deepcopy

import sys
import torch
import cv2
import numpy as np
import matplotlib.cm as cm
from src.utils.plotting import make_matching_figure
from src.loftr import LoFTR, default_cfg
#import matplotlib
#import matplotlib.pyplot as plt
#matplotlib.use('TkAgg')

def concatenate_filenames(paths):
    filenames = [os.path.splitext(os.path.basename(path))[0] for path in paths]
    concatenated_filenames = '_'.join(filenames)
    return concatenated_filenames

# Check if the correct number of arguments are provided
if len(sys.argv) != 4:
    print("Usage: python loftr_matcher.py img0 img1 path_to_output")
    sys.exit(1)

matcher = LoFTR(config=default_cfg)
matcher.load_state_dict(torch.load("/mnt/ssd/data/LoFTR/weights/outdoor_ds.ckpt")['state_dict'])
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

img0_pth = sys.argv[1]
img1_pth = sys.argv[2]
img0_raw = cv2.imread(img0_pth, cv2.IMREAD_GRAYSCALE)
img1_raw = cv2.imread(img1_pth, cv2.IMREAD_GRAYSCALE)

factor0 = 64 if img0_raw.shape[0] > 3000 else 32
factor1 = 64 if img1_raw.shape[0] > 3000 else 32

img0_raw = cv2.resize(img0_raw, (img0_raw.shape[1]//factor0*8, img0_raw.shape[0]//factor0*8))
img1_raw = cv2.resize(img1_raw, (img1_raw.shape[1]//factor1*8, img1_raw.shape[0]//factor1*8))

#mask0_raw = cv2.imread(sys.argv[3], cv2.IMREAD_GRAYSCALE)
#mask1_raw = cv2.imread(sys.argv[4], cv2.IMREAD_GRAYSCALE)
#mask0_raw = cv2.resize(mask0_raw, (mask0_raw.shape[1]//factor1, mask0_raw.shape[0]//factor1))
#mask1_raw = cv2.resize(mask1_raw, (mask1_raw.shape[1]//factor1, mask1_raw.shape[0]//factor1))

img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.

#mask0 = torch.from_numpy(mask0_raw)[None][None].cuda() / 255.
#mask1 = torch.from_numpy(mask1_raw)[None][None].cuda() / 255.
#mask0 = torch.squeeze(mask0, dim=1)
#mask1 = torch.squeeze(mask1, dim=1)

batch = {'image0': img0, 'image1': img1}
#batch = {'image0': img0, 'image1': img1, 'mask0': mask0, 'mask1': mask1}

with torch.no_grad():
	matcher(batch)
	mkpts0 = batch['mkpts0_f'].cpu().numpy()
	mkpts1 = batch['mkpts1_f'].cpu().numpy()
	mconf = batch['mconf'].cpu().numpy()

outfile = sys.argv[3] + concatenate_filenames([img0_pth, img1_pth]) + ".txt"

with open(outfile, 'w') as f:
        f.write(str(mkpts0.shape[0]) + "\n")
        scaled_data0 = np.copy(mkpts0)
        scaled_data0[:, 0] /= img0_raw.shape[1]
        scaled_data0[:, 1] /= img0_raw.shape[0]
        np.savetxt(f, scaled_data0, fmt='%.6f', delimiter=' ')
        f.write(str(mkpts1.shape[0]) + "\n")
        scaled_data1 = np.copy(mkpts1)
        scaled_data1[:, 0] /= img1_raw.shape[1]
        scaled_data1[:, 1] /= img1_raw.shape[0]
        np.savetxt(f, scaled_data1, fmt='%.6f', delimiter=' ')

#color = cm.jet(mconf)
#text = [
#    'LoFTR',
#    'Matches: {}'.format(len(mkpts0)),
#]
#fig = make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color, text=text)
#plt.show()