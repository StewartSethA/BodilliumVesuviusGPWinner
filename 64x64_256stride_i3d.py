import sys
print("Importing...")
import os.path as osp
import os
os.environ["WANDB_MODE"] = "offline"
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = 9331200000
import torch
import torch.nn as nn
import torch.nn.functional as F


import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import random
import yaml

import numpy as np

import wandb

from torch.utils.data import DataLoader
print("1")

import os
import random
from contextlib import contextmanager
import cv2

import scipy as sp
import numpy as np
import fastnumpyio as fnp

from tqdm.auto import tqdm

import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
print("2")
import datetime
import segmentation_models_pytorch as smp
print("3")
import numpy as np
print("4")
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from i3dallnl import InceptionI3d
print("5")
import torch.nn as nn
print("6")
import torch
print("7")
from warmup_scheduler import GradualWarmupScheduler
print("8")
from scipy import ndimage
print("9")
import time
print("10")
import json
print("11")
#import numba
print("12")
#from numba import jit
from torch.utils.tensorboard import SummaryWriter
print("Done importing")

from skimage.measure import block_reduce

print(sys.executable)
class CFG:
    # ============== comp exp name =============
    comp_name = 'vesuvius'

    # comp_dir_path = './'

    comp_dir_path = './' #'/content/gdrive/MyDrive/vesuvius_model/training/'
    comp_folder_name = './' #'/content/gdrive/MyDrive/vesuvius_model/training/'
    # comp_dataset_path = f'{comp_dir_path}datasets/{comp_folder_name}/'
    comp_dataset_path = './' #f'/content/gdrive/MyDrive/vesuvius_model/training/'
    basepath = "train_scrolls" #comp_dir_path
    exp_name = 'pretraining_all'

    # ============== pred target =============
    target_size = 1

    # ============== model cfg =============
    model_name = 'Unet'
    # backbone = 'efficientnet-b0'
    # backbone = 'se_resnext50_32x4d'
    backbone='resnet3d'
    in_chans = 30 # 65
    encoder_depth=5
    # ============== training cfg =============
    scale = 1
    size = 64
    tile_size = 256
    stride = tile_size // 8

    train_batch_size = 256 # 32
    valid_batch_size = train_batch_size
    use_amp = True

    scheduler = 'GradualWarmupSchedulerV2'
    # scheduler = 'CosineAnnealingLR'
    epochs = 30 # 30

    # adamW warmupあり
    warmup_factor = 10
    # lr = 1e-4 / warmup_factor
    # lr = 1e-4 / warmup_factor
    lr = 2e-5
    # ============== fold =============
    valid_id = '20230820203112'

    # objective_cv = 'binary'  # 'binary', 'multiclass', 'regression'
    metric_direction = 'maximize'  # maximize, 'minimize'
    # metrics = 'dice_coef'

    # ============== fixed =============
    pretrained = True
    inf_weight = 'best'  # 'best'

    min_lr = 1e-6
    weight_decay = 1e-6
    max_grad_norm = 100

    print_freq = 50
    num_workers = 10

    seed = 0

    # ============== set dataset path =============
    print('set dataset path')

    outputs_path = './' #f'/content/gdrive/MyDrive/vesuvius_model/training/outputs'

    submission_dir = outputs_path + 'submissions/'
    submission_path = submission_dir + f'submission_{exp_name}.csv'

    model_dir = outputs_path + \
        f'{comp_name}-models/'

    figures_dir = outputs_path + 'figures/'

    log_dir = outputs_path + 'logs/'
    log_path = log_dir + f'{exp_name}.txt'

    # ============== augmentation =============
    train_aug_list = [
        # A.RandomResizedCrop(
        #     size, size, scale=(0.85, 1.0)),
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # A.RandomRotate90(p=0.6),

        A.RandomBrightnessContrast(p=0.75),
        A.ShiftScaleRotate(rotate_limit=360,shift_limit=0.15,scale_limit=0.1,p=0.75),
        A.OneOf([
                A.GaussNoise(var_limit=[10, 50]),
                A.GaussianBlur(),
                A.MotionBlur(),
                ], p=0.4),
        # A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
        A.CoarseDropout(max_holes=2, max_width=int(size * 0.2), max_height=int(size * 0.2), 
                        mask_fill_value=0, p=0.5),
        # A.Cutout(max_h_size=int(size * 0.6),
        #          max_w_size=int(size * 0.6), num_holes=1, p=1.0),
        A.Normalize(
            mean= [0] * in_chans,
            std= [1] * in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ]

    valid_aug_list = [
        A.Resize(size, size),
        A.Normalize(
            mean= [0] * in_chans,
            std= [1] * in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ]
    rotate = A.Compose([A.Rotate(8,p=1)])
    #rotate = A.Compose([A.Rotate(90,p=1)])
def init_logger(log_file):
    from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

def set_seed(seed=None, cudnn_deterministic=True):
    if seed is None:
        seed = 42

    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = False
def make_dirs(cfg):
    for dir in [cfg.model_dir, cfg.figures_dir, cfg.submission_dir, cfg.log_dir]:
        os.makedirs(dir, exist_ok=True)
def cfg_init(cfg, mode='train'):
    set_seed(cfg.seed)
    # set_env_name()
    # set_dataset_path(cfg)

    if mode == 'train':
        make_dirs(cfg)
cfg_init(CFG)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = None
def read_image_mask(fragment_id,start_idx=15,end_idx=45,fragment_mask_only=False,pad0=0, pad1=0):
    scale = CFG.scale
    if fragment_mask_only:
      print("Reading fragment mask", f"train_scrolls/{fragment_id}/{fragment_id}_mask.png")
      fragment_mask=cv2.imread(f"train_scrolls/{fragment_id}/{fragment_id}_mask.png", 0)
      if fragment_id=='20230827161846':
        fragment_mask=cv2.flip(fragment_mask,0)
      if 'frag' in fragment_id:
        fragment_mask = cv2.resize(fragment_mask, (fragment_mask.shape[1]//2,fragment_mask.shape[0]//2), interpolation = cv2.INTER_AREA)
      #if scale != 1 and images is None: # Whoa there pardner! Scaling down the masks too I see!
      #  print("Resizing fragment mask since images are None!", fragment_mask.shape)
      #  fragment_mask = cv2.resize(fragment_mask , (fragment_mask.shape[1]//scale,fragment_mask.shape[0]//scale), interpolation = cv2.INTER_AREA)
      if scale != 1: #fragment_mask.shape[:2] != images.shape[:2]:
        print("Resizing fragment mask to equal images shape", fragment_mask.shape, images.shape)
        fragment_mask = cv2.resize(fragment_mask, (fragment_mask.shape[1]//scale,fragment_mask.shape[0]//scale), interpolation = cv2.INTER_AREA)
        return None, None, fragment_mask

    images = []

    # idxs = range(65)
    mid = 65 // 2
    start = mid - CFG.in_chans // 2
    end = mid + CFG.in_chans // 2
    idxs = range(start_idx, end_idx)


    t = time.time()
    rescaled = False
    scalerangenpy = f"train_scrolls/{fragment_id}_{start_idx}-{end_idx}_{scale}.npy"
    normalnpy = f"train_scrolls/{fragment_id}/layers/{fragment_id}.npy"
    rootnpy = f"train_scrolls/{fragment_id}.npy"
    print("Checking for", scalerangenpy, scale)
    if scale != 1 and os.path.isfile(scalerangenpy):
      images = np.load(scalerangenpy)
      pad0 = (CFG.tile_size - images.shape[0] % CFG.tile_size)
      pad1 = (CFG.tile_size - images.shape[1] % CFG.tile_size)
      print(time.time()-t, "seconds taken to load images from", scalerangenpy)
      rescaled = True
    else:
      print("Checking for", rootnpy, normalnpy, scale)
      for path in rootnpy, normalnpy:
        if os.path.isfile(path):
          images = np.load(path)
          pad0 = (CFG.tile_size - images.shape[0] % CFG.tile_size)
          pad1 = (CFG.tile_size - images.shape[1] % CFG.tile_size)
          print(time.time()-t, "seconds taken to load images from", path)
          break
    if isinstance(images, list):
      for i in idxs:
        
        image = cv2.imread(CFG.comp_dataset_path + f"train_scrolls/{fragment_id}/layers/{i:02}.tif", 0)

        pad0 = (CFG.tile_size - image.shape[0] % CFG.tile_size)
        pad1 = (CFG.tile_size - image.shape[1] % CFG.tile_size)

        image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)
        # image = ndimage.median_filter(image, size=5) # TODO: Why median filtering?
        
        # image = cv2.resize(image, (image.shape[1]//2,image.shape[0]//2), interpolation = cv2.INTER_AREA)
        if 'frag' in fragment_id:
            image = cv2.resize(image, (image.shape[1]//2,image.shape[0]//2), interpolation = cv2.INTER_AREA)
        image=np.clip(image,0,200)
        if fragment_id=='20230827161846':
            image=cv2.flip(image,0)
        images.append(image)
      print(time.time()-t, "seconds taken to load images.")
      images = np.stack(images, axis=2)
      t = time.time()
      print(time.time()-t, "seconds taken to stack images.")
      t = time.time()
      np.save(CFG.comp_dataset_path + f"train_scrolls/{fragment_id}.npy", images)
      print(time.time()-t, "seconds taken to save images as npy.")

    basepath = CFG.basepath
    if fragment_id=='20231022170900':
        mask = cv2.imread(f"train_scrolls/{fragment_id}/{fragment_id}_inklabels.tiff", 0)
    else:
        mask = cv2.imread(f"train_scrolls/{fragment_id}/{fragment_id}_inklabels.png", 0)
    if mask is None:
      print("Warning: No GT found for", fragment_id)
      mask = np.zeros_like(images[:,:,0])

    if scale != 1 and not rescaled:
      print("Rescaling image...", fragment_id, images.shape, "down by", scale)
      t = time.time()
      images = (block_reduce(images, block_size=(scale,scale,1), func=np.mean, cval=np.mean(images))+0.5).astype(np.uint8)
      print("Rescaling took", time.time()-t, "seconds.", images.shape)
      np.save(scalerangenpy, images) # Other parts too
      print("Saved rescaled array.")
      pad0 = (CFG.tile_size - images.shape[0] % CFG.tile_size)
      pad1 = (CFG.tile_size - images.shape[1] % CFG.tile_size)
    if isinstance(images, np.ndarray):
      images = np.pad(images, [(0,pad0), (0, pad1), (0, 0)], constant_values=0)

    if 'frag' in fragment_id:
      mask = cv2.resize(mask , (mask.shape[1]//2,mask.shape[0]//2), interpolation = cv2.INTER_AREA)

    #if scale != 1 or (images is not None and mask.shape[:2] != images.shape[:2]):
    #  print("resizing ink labels", mask.shape, scale, images.shape)
    #  #mask = cv2.resize(mask , (mask.shape[1]//scale,mask.shape[0]//scale), interpolation = cv2.INTER_AREA)
    if mask is not None and images is not None and scale != 1: #images.shape[:2] != mask.shape:
      print("resizing ink labels", mask.shape, scale, images.shape)
      mask = cv2.resize(mask , (mask.shape[1]//scale,mask.shape[0]//scale), interpolation = cv2.INTER_AREA)

    # mask = np.pad(mask, [(0, pad0), (0, pad1)], constant_values=0)
    print("Reading fragment mask", f"train_scrolls/{fragment_id}/{fragment_id}_mask.png")
    fragment_mask=cv2.imread(f"train_scrolls/{fragment_id}/{fragment_id}_mask.png", 0)
    if fragment_id=='20230827161846':
      fragment_mask=cv2.flip(fragment_mask,0)
    fragment_mask_only=False
    if fragment_mask is not None and images is not None and mask is not None and (not fragment_mask_only):
      print("Padding masks")
      p0 = max(0,images.shape[0]-fragment_mask.shape[0])
      p1 = max(0,images.shape[1]-fragment_mask.shape[1])
      fragment_mask = np.pad(fragment_mask, [(0, p0), (0, p1)], constant_values=0)
      p0 = max(0,images.shape[0]-mask.shape[0])
      p1 = max(0,images.shape[1]-mask.shape[1])
      mask = np.pad(mask, [(0, p0), (0, p1)], constant_values=0)
      mask = cv2.blur(mask, (3,3))
    elif not fragment_mask_only:
      return (None,)*3
    if 'frag' in fragment_id:
        fragment_mask = cv2.resize(fragment_mask, (fragment_mask.shape[1]//2,fragment_mask.shape[0]//2), interpolation = cv2.INTER_AREA)
    #if scale != 1 and images is None: # Whoa there pardner! Scaling down the masks too I see!
    #  print("Resizing fragment mask since images are None!", fragment_mask.shape)
    #  fragment_mask = cv2.resize(fragment_mask , (fragment_mask.shape[1]//scale,fragment_mask.shape[0]//scale), interpolation = cv2.INTER_AREA)
    if scale != 1: #fragment_mask.shape[:2] != images.shape[:2]:
      print("Resizing fragment mask to equal images shape", fragment_mask.shape, images.shape)
      fragment_mask = cv2.resize(fragment_mask, (fragment_mask.shape[1]//scale,fragment_mask.shape[0]//scale), interpolation = cv2.INTER_AREA)

    if mask is not None and images is not None and fragment_mask is not None:
      print("PYGO images.shape,dtype", images.shape, (images.dtype if isinstance(images, np.ndarray) else None), "mask", mask.shape, mask.dtype, "fragment_mask", fragment_mask.shape, fragment_mask.dtype)

    return images, mask,fragment_mask

#from numba import vectorize
#@vectorize
#@jit(nopython=True)
def generate_xyxys_ids(fragment_id, image, mask, fragment_mask, tile_size, size, stride, is_valid=False, scale=1):
        print("    Generating xyxys", fragment_id, image.shape, mask.shape, fragment_mask.shape, "mask min max", mask.min(), mask.max(), "tile_size", tile_size, "size", size, "stride", stride, "is_valid", is_valid)
        noink = os.path.join(CFG.basepath, fragment_id + "_noink.png")
        if os.path.isfile(noink):
          print("Reading NO INK file:", noink)
          noink = cv2.imread(noink, 0)
        else:
          noink = None
        if noink is not None and scale != 1: #mask.shape[:2] != noink.shape:
          print("MAJOR WARNING: NO INK MASK AND INK MASK SIZES DO NOT MATCH!", noink.shape, mask.shape, noink.shape[0]/mask.shape[0], noink.shape[1]/mask.shape[1])
          #noink = cv2.resize(noink, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_AREA)
          noink = cv2.resize(noink, (0,0), fx=1/scale, fy=1/scale, interpolation=cv2.INTER_AREA)
        if noink is None:
          noink = mask
        else:
          noink = mask + noink[:,:,np.newaxis]
        # TODO Use multiplication of masks and np.argwhere to produce suggested coordinates.
        xyxys = []
        ids = []
        #if fragment_id in ['20231007101619']: # SethS 4/17/2024 for suppressing false positive ink labels, include more true negative labels.
        #  is_valid = True
        #if is_valid:
        #  stride = stride * 2
        # TODO SethS:
        if is_valid: #Whoops! Was mask, not fragment_mask!
          xyxys = [(c[1],c[0],c[1]+size,c[0]+size) for c in np.argwhere(fragment_mask[:,:] >= -9999).tolist() if c[0] >= 0 and c[1] >= 0 and c[0]+tile_size < fragment_mask.shape[0] and c[1]+tile_size < fragment_mask.shape[1]] #[::int(stride)]
          ids = [fragment_id] * len(xyxys)
          return xyxys, ids
        if not is_valid: # Added 4/29 SethS
          #xyxys = [(c[1],c[0],c[1]+size,c[0]+size) for c in np.argwhere(mask[:,:,0] > 0).tolist() if c[0] >= 0 and c[1] >= 0 and c[0]+tile_size < mask.shape[0] and c[1]+tile_size < mask.shape[1]] #[::int(stride)
          t = time.time()
          xyxys = [(c[1],c[0],c[1]+size,c[0]+size) for c in np.argwhere(noink[:,:,0] > 0).tolist() if c[0] >= 0 and c[1] >= 0 and c[0]+tile_size < mask.shape[0] and c[1]+tile_size < mask.shape[1]] #[::int(stride)]
          print("xyxys for training generated in", time.time()-t, "seconds", fragment_id, len(xyxys), noink.shape)
          ids = [fragment_id] * len(xyxys)
          return xyxys, ids






        xyxys = []
        ids = []
        x1_list = list(range(0, image.shape[1]-tile_size+1, stride))
        y1_list = list(range(0, image.shape[0]-tile_size+1, stride))
        #x1_list = list(range(0, image.size()[1]-tile_size+1, stride))
        #y1_list = list(range(0, image.size()[0]-tile_size+1, stride))
        #windows_dict={}
        for a in y1_list:
            for b in x1_list:
                for yi in range(0,tile_size,size):
                    for xi in range(0,tile_size,size):
                        y1=a+yi
                        x1=b+xi
                        y2=y1+size
                        x2=x1+size
                # for y2 in range(y1,y1 + tile_size,size):
                #     for x2 in range(x1, x1 + tile_size,size):
                        if not is_valid:
                            if not np.all(np.less(mask[a:a + tile_size, b:b + tile_size],0.01)):
                                if not np.any(np.equal(fragment_mask[a:a+ tile_size, b:b + tile_size],0)):
                                    # if (y1,y2,x1,x2) not in windows_dict:
                                    #train_images.append(image[y1:y2, x1:x2])
                                    xyxys.append([x1,y1,x2,y2])
                                    ids.append(fragment_id)
                                    #train_masks.append(mask[y1:y2, x1:x2, None])
                                    assert image[y1:y2, x1:x2].shape==(size,size,CFG.in_chans)
                                        # windows_dict[(y1,y2,x1,x2)]='1'
                        else:
                            if not np.any(np.equal(fragment_mask[a:a + tile_size, b:b + tile_size], 0)):
                                    #valid_images.append(image[y1:y2, x1:x2])
                                    #valid_masks.append(mask[y1:y2, x1:x2, None])
                                    ids.append(fragment_id)
                                    xyxys.append([x1, y1, x2, y2])
                                    assert image[y1:y2, x1:x2].shape==(size,size,CFG.in_chans)
        return xyxys, ids


def get_xyxys(fragment_ids, is_valid=False):
#def get_xyxys(valid_ids, CFG, True, start_idx=start_idx, end_idx=end_idx, train_images=train_images, train_masks=train_masks, train_ids=train_ids, pads=pads, scale=1):
    xyxys = []
    ids = []
    images = {}
    masks = {}
    for fragment_id in fragment_ids:
        #start_idx = len(fragment_ids)
        print('reading ',fragment_id)
        image, mask,fragment_mask = read_image_mask(fragment_id)

        images[fragment_id] = image
        masks[fragment_id] = mask[:,:,None]
        t = time.time()
        #if os.path.isfile(fragment_id + ".ids.json"):
        #  with open(fragment_id + ".ids.json", 'r') as f:
        #    id = json.load(f)
        #if os.path.isfile(fragment_id + ".xyxys.json"):
        #  with open(fragment_id + ".xyxys.json", 'r') as f:
        #    xyxy = json.load(f)
        #else:
        if True:
          xyxy, id = generate_xyxys_ids(fragment_id, image, mask, fragment_mask, CFG.tile_size, CFG.size, CFG.stride, is_valid, scale=CFG.scale)
          with open(fragment_id + ".ids.json", 'w') as f:
            #if fragment_id != CFG.valid_id:
              json.dump(id, f) #[start_idx:], f)
            #else:
            #  json.dump(valid_ids, f)
          with open(fragment_id + ".xyxys.json", 'w') as f:
            #if fragment_id != CFG.valid_id:
            json.dump(xyxy, f) #[start_idx:],f)
            #else:
            #  json.dump(valid_xyxys, f)
        xyxys = xyxys + xyxy
        ids = ids + id

        print(time.time()-t, "seconds taken to generate crops for fragment", fragment_id)
    return images, masks, xyxys, ids

def get_xyxys(fragment_ids, cfg, is_valid=False, start_idx=15, end_idx=45, train_images={}, train_masks={}, train_ids=[], pads={}, scale=1):
    xyxys = []
    ids = []
    images = {}
    masks = {}

    for fragment_id in fragment_ids:
        myscale = scale
        if fragment_id in ['20231215151901', '20231111135340', '20231122192640']:
          myscale = scale * 2 # 2040 was extracted at 7.91 um, same as Scroll1
        start_idx = len(fragment_ids)
        print('reading', fragment_id)
        if fragment_id in train_images.keys():
          image, mask = train_images[fragment_id], train_masks[fragment_id]
          pad0, pad1 = pads.get(fragment_id)
          #_, _, fragment_mask,pad0,pad1 = read_image_mask(fragment_id, start_idx, end_idx, cfg, fragment_mask_only=True, pad0=pad0, pad1=pad1, scale=myscale, force_mem=is_valid)
          #_, _, fragment_mask, pad0, pad1 = read_image_mask(fragment_id, start_idx, end_idx, cfg, fragment_mask_only=True, pad0=pad0, pad1=pad1) #, scale=myscale, force_mem=is_valid)
          _, _, fragment_mask = read_image_mask(fragment_id, start_idx, end_idx, cfg, fragment_mask_only=True, pad0=pad0, pad1=pad1) #, scale=myscale, force_mem=is_valid)
        else:
          #image, mask,fragment_mask,pad0,pad1 = read_image_mask(fragment_id, start_idx, end_idx, cfg, scale=myscale, force_mem=is_valid)
          #image, mask,fragment_mask,pad0,pad1 = read_image_mask(fragment_id, start_idx, end_idx) #, cfg, scale=myscale, force_mem=is_valid)
          image, mask,fragment_mask = read_image_mask(fragment_id, start_idx, end_idx) #, cfg, scale=myscale, force_mem=is_valid)
          pad0, pad1 = 0, 0
        if image is None:
          print("Failed to load", fragment_id)
          continue
        print("Loading ink labels")
        pads[fragment_id] = (pad0, pad1)

        images[fragment_id] = image
        if image is None:
          masks[fragment_id] = None
          continue
          #return None, None, None, None
        if mask is None:
          print("Defaulting to empty mask! I hope this is just for inference!", fragment_id)
          mask = np.zeros_like(image[:,:,0])

        masks[fragment_id] = mask = mask[:,:,np.newaxis] if len(mask.shape) == 2 else mask
        t = time.time()
        validlabel="valid" if is_valid else "train"
        savename = os.path.join(cfg.basepath, fragment_id + validlabel+str(cfg.tile_size)+"_"+str(cfg.size)+"_"+str(cfg.stride)+("_s"+str(myscale) if myscale != 1 else ""))
        print("Loading IDs, xyxys", savename)
        if os.path.isfile(savename+".ids.json") and False:
          with open(savename + ".ids.json", 'r') as f:
            id = json.load(f)
        if os.path.isfile(savename + ".xyxys.json") and False:
          with open(savename + ".xyxys.json", 'r') as f:
            xyxy = json.load(f)
            print(savename, "xyxys len:", len(xyxy), image.shape)
        else:
          print("Generating new xyxys for", fragment_id, image.shape, "mask", mask.shape)
          if not is_valid:
            xyxy, id = generate_xyxys_ids(fragment_id, image, mask, fragment_mask, cfg.tile_size, cfg.size, cfg.stride, is_valid, scale=CFG.scale) #, CFG=cfg)
          else:
            xyxy, id = generate_xyxys_ids(fragment_id, image, mask, fragment_mask, cfg.valid_tile_size, cfg.valid_size, cfg.valid_stride, is_valid, scale=CFG.scale) #, CFG=cfg)
          print("saving xyxys and ids", len(xyxy), len(id), "to", savename)
          with open(savename + ".ids.json", 'w') as f:
            #if fragment_id != cfg.valid_id:
              json.dump(id, f) #[start_idx:], f)
            #else:
            #  json.dump(valid_ids, f)
          with open(savename +".xyxys.json", 'w') as f:
            #if fragment_id != cfg.valid_id:
            json.dump(xyxy, f) #[start_idx:],f)
            #else:
            #  json.dump(valid_xyxys, f)
        #print("xyxys", xyxys, xyxy, xyxy[-1], len(xyxys), len(xyxy))
        xyxys = xyxys + xyxy
        ids = ids + id

        print(time.time()-t, "seconds taken to generate crops for fragment", fragment_id)
    return images, masks, xyxys, ids, pads

#@jit(nopython=True)
def get_train_valid_dataset():
    train_images = {}
    train_masks = {}
    train_xyxys= []
    train_ids = []
    valid_images = {}
    valid_masks = {}
    valid_xyxys = []
    valid_ids = []
    train_ids = set(['20230702185753','20230929220926','20231005123336','20231007101619','20231012184423','20231016151002','20231022170901','20231031143852','20231106155351','20231210121321','20231221180251','20230820203112']) - set([CFG.valid_id])
    valid_ids = set([CFG.valid_id])
    train_images, train_masks, train_xyxys, train_ids = get_xyxys(train_ids, False)
    valid_images, valid_masks, valid_xyxys, valid_ids = get_xyxys(valid_ids, True)
    return train_images, train_masks, train_xyxys, train_ids, valid_images, valid_masks, valid_xyxys, valid_ids

def get_train_valid_dataset(CFG, train_ids=[], valid_ids=[], start_idx=15, end_idx=45, scale=1):
    start_idx = 0
    if len(train_ids) == 0:
      train_ids = set(["20231210132040", '20230929220926','20231005123336']) #,'20231007101619','20231016151002']) # - set([CFG.valid_id])
    if len(valid_ids) == 0:
      CFG.valid_id = "20231210132040" #"20231215151901" #20240304141530"
      valid_ids = set([CFG.valid_id]) #+list(train_ids))
    train_images, train_masks, train_xyxys, train_ids, pads = get_xyxys(train_ids, CFG, False, start_idx=start_idx, end_idx=end_idx, scale=scale)
    valid_images, valid_masks, valid_xyxys, valid_ids, _ = get_xyxys(valid_ids, CFG, True, start_idx=start_idx, end_idx=end_idx, train_images=train_images, train_masks=train_masks, train_ids=train_ids, pads=pads, scale=scale)
    return train_images, train_masks, train_xyxys, train_ids, valid_images, valid_masks, valid_xyxys, valid_ids

def get_transforms(data, cfg):
    if data == 'train':
        aug = A.Compose(cfg.train_aug_list)
    elif data == 'valid':
        aug = A.Compose(cfg.valid_aug_list)

    return aug

class CustomDataset(Dataset):
    def __init__(self, images, cfg, xyxys=None, labels=None, ids=None, transform=None):
        self.images = images
        self.cfg = cfg
        self.labels = labels
        self.transform = transform
        self.xyxys=xyxys
        self.ids = ids
        self.rotate=CFG.rotate
    def __len__(self):
        return len(self.xyxys)
    def cubeTranslate(self,y):
        x=np.random.uniform(0,1,4).reshape(2,2)
        x[x<.4]=0
        x[x>.633]=2
        x[(x>.4)&(x<.633)]=1
        mask=cv2.resize(x, (x.shape[1]*64,x.shape[0]*64), interpolation = cv2.INTER_AREA)
        x=np.zeros((self.cfg.size,self.cfg.size,self.cfg.in_chans)).astype(np.uint8)
        for i in range(3):
            x=np.where(np.repeat((mask==0).reshape(self.cfg.size,self.cfg.size,1), self.cfg.in_chans, axis=2),y[:,:,i:self.cfg.in_chans+i],x)
        return x
    def fourth_augment(self,image):
        image_tmp = np.zeros_like(image)
        cropping_num = random.randint(24, 30)

        start_idx = random.randint(0, self.cfg.in_chans - cropping_num)
        crop_indices = np.arange(start_idx, start_idx + cropping_num)

        start_paste_idx = random.randint(0, self.cfg.in_chans - cropping_num)

        tmp = np.arange(start_paste_idx, cropping_num)
        np.random.shuffle(tmp)

        cutout_idx = random.randint(0, 2)
        temporal_random_cutout_idx = tmp[:cutout_idx]

        image_tmp[..., start_paste_idx : start_paste_idx + cropping_num] = image[..., crop_indices]

        if random.random() > 0.4:
            image_tmp[..., temporal_random_cutout_idx] = 0
        image = image_tmp
        return image

    def __getitem__(self, idx):
        if self.xyxys is not None:
            id = self.ids[idx]
            x1,y1,x2,y2=xy=self.xyxys[idx]
            image = self.images[id][y1:y2,x1:x2] #,self.start:self.end] #[idx]
            label = self.labels[id][y1:y2,x1:x2]
            if np.product(image.shape) == 0:
                print("Invalid xy", self.xyxys[idx], self.images[id].shape)
                h,w=y2-y1,x2-x1
                x1,y1 = random.randint(0,self.images[id].shape[1]-w), random.randint(0,self.images[id].shape[0]-h)
                image = self.images[id][y1:y2,x1:x2] #,self.start:self.end] #[idx]
                label = self.labels[id][y1:y2,x1:x2] #,self.start:self.end] #[idx]
                #del self[idx]
                #return self[idx]
                #return self[(idx+1))%len(self)]
            if self.transform:
                data = self.transform(image=image, mask=label)
                image = data['image'].unsqueeze(0)
                label = data['mask']
                #label=F.interpolate((label/255).unsqueeze(0).float(),(max(1,self.cfg.size//4),max(self.cfg.size//4,1))).squeeze(0)
                label=(label/255).float()
            return image, label,xy,id
        else:
            image = self.images[idx]
            label = self.labels[idx]
            #3d rotate
            image=image.transpose(2,1,0)#(c,w,h)
            image=self.rotate(image=image)['image']
            image=image.transpose(0,2,1)#(c,h,w)
            image=self.rotate(image=image)['image']
            image=image.transpose(0,2,1)#(c,w,h)
            image=image.transpose(2,1,0)#(h,w,c)

            image=self.fourth_augment(image)
            
            if self.transform:
                data = self.transform(image=image, mask=label)
                image = data['image'].unsqueeze(0)
                label = data['mask']
                label=(label/255).float()
                #label=F.interpolate((label/255).unsqueeze(0).float(),(max(1,self.cfg.size//4),max(1,self.cfg.size//4))).squeeze(0)
            return image, label
class CustomDatasetTest(Dataset):
    def __init__(self, images, labels, xyxys, ids, cfg, transform=None):
        self.images = images
        self.labels = labels
        self.xyxys=xyxys
        self.ids = ids
        self.cfg = cfg
        self.transform = transform

    def __len__(self):
        # return len(self.df)
        return len(self.xyxys)

    def __getitem__(self, idx):
        x1,y1,x2,y2=xy=self.xyxys[idx]
        id = self.ids[idx]
        image = self.images[id][y1:y2,x1:x2]
        label = self.labels[id][y1:y2,x1:x2]
        if np.product(image.shape) == 0:
          print("Erroneous bounds", xy, id, self.images[id].shape)
          return None,None
        if self.transform:
            data = self.transform(image=image)
            image = data['image'].unsqueeze(0)

        return image,label,xy,id



# from resnetall import generate_model
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m, mode='fan_out', nonlinearity='relu')
class Decoder(nn.Module):
    def __init__(self, encoder_dims, upscale):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(encoder_dims[i]+encoder_dims[i-1], encoder_dims[i-1], 3, 1, 1, bias=False),
                nn.BatchNorm2d(encoder_dims[i-1]),
                nn.ReLU(inplace=True)
            ) for i in range(1, len(encoder_dims))])

        self.logit = nn.Conv2d(encoder_dims[0], 1, 1, 1, 0)
        self.up = nn.Upsample(scale_factor=upscale, mode="bilinear")

    def forward(self, feature_maps):
        for i in range(len(feature_maps)-1, 0, -1):
            f_up = F.interpolate(feature_maps[i], scale_factor=2, mode="bilinear")
            f = torch.cat([feature_maps[i-1], f_up], dim=1)
            f_down = self.convs[i-1](f)
            feature_maps[i-1] = f_down

        x = self.logit(feature_maps[0])
        mask = self.up(x)
        return mask



class RegressionPLModel(pl.LightningModule):

    def __init__(self,pred_shape,size=256,enc='',with_norm=False,total_steps=500, train_dataset=None, check_val_every_n_epoch=1, backbone=None, wandb_logger=None, val_masks=None,name="", complexity=16):
        super(RegressionPLModel, self).__init__()
        self.save_hyperparameters("size", "enc", "with_norm", "total_steps", "check_val_every_n_epoch", "name")
        self.name = name
        #self.save_hyperparameters()
        self.writer = SummaryWriter("runs/"+name)
        self.hparams.pred_shape = pred_shape
        self.val_masks = val_masks
        self.wandb_logger = wandb_logger
        if isinstance(self.hparams.pred_shape, dict):
          self.mask_pred = {}
          self.mask_count = {}
          for k,pred_shape in self.hparams.pred_shape.items():
            print("Initializing mask pred and count", k, pred_shape)
            self.mask_pred[k] = np.zeros(pred_shape)
            self.mask_count[k] = np.zeros(pred_shape)
        else:
          self.mask_pred = np.zeros(self.hparams.pred_shape)
          self.mask_count = np.zeros(self.hparams.pred_shape)

#    def __init__(self,pred_shape,size=256,enc='',with_norm=False, backbone="i3d", complexity=8):
#        super(RegressionPLModel, self).__init__()

#        self.save_hyperparameters()
#        self.mask_pred = np.zeros(self.hparams.pred_shape)
#        self.mask_count = np.zeros(self.hparams.pred_shape)

        self.loss_func1 = smp.losses.DiceLoss(mode='binary')
        self.loss_func2= smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25)
        self.loss_func= lambda x,y:0.5 * self.loss_func1(x,y)+0.5*self.loss_func2(x,y)
        self.model_name = backbone
        if "pygoflat" in backbone.lower():
            from pygoflat import InceptionI3d
            self.InceptionI3d = InceptionI3d
            self.backbone=InceptionI3d(in_channels=1,num_classes=128,non_local=False, complexity=complexity)
        elif "pygonet" in backbone.lower():
            from pygonet import InceptionI3d
            self.InceptionI3d = InceptionI3d
            self.backbone=InceptionI3d(in_channels=1,num_classes=512,non_local=False)
        elif "pygo" in backbone.lower():
            from pygoi3d_simple import InceptionI3d
            self.InceptionI3d = InceptionI3d
            self.backbone=InceptionI3d(in_channels=1,num_classes=512,non_local=False)
        elif "inception" in backbone or "i3d" in backbone.lower():
            from i3dallnl import InceptionI3d
            self.InceptionI3d = InceptionI3d
            self.backbone=InceptionI3d(in_channels=1,num_classes=512,non_local=True)

        #self.backbone=InceptionI3d(in_channels=1,num_classes=512,non_local=True)        
        self.decoder = Decoder(encoder_dims=[x.size(1) for x in self.backbone(torch.rand(1,1,20,256,256))], upscale=1)

        if self.hparams.with_norm:
            self.normalization=nn.BatchNorm3d(num_features=1)



            
    def forward(self, x):
        if x.ndim==4:
            x=x[:,None]
        if self.hparams.with_norm:
            x=self.normalization(x)
        feat_maps = self.backbone(x)
        feat_maps_pooled = [torch.max(f, dim=2)[0] for f in feat_maps]
        pred_mask = self.decoder(feat_maps_pooled)
        
        return pred_mask
    
    def training_step(self, batch, batch_idx):
        x, y, xys,ids = batch
        outputs = self(x)
        #print("outputs.shape", outputs.shape, "y.shape", y.shape)
        if outputs.shape != y.shape and y.shape[-2] > outputs.shape[-2]: # Downsample labels if bigger than outputs
          y = F.interpolate(y, outputs.shape[-2:], mode="area") # Make
        if outputs.shape != y.shape:
          #outputs = F.interpolate(outputs, y.shape[-2:], mode="bilinear") # Stretch outputs to equal label size
          outputs = F.interpolate(outputs, y.shape[-2:], mode="area") # Stretch outputs to equal label size
        loss1 = self.loss_func(outputs, y)
        diceloss = self.loss_func1(outputs, y)
        bceloss = self.loss_func2(outputs, y)
        loss1 = self.loss_func(outputs, y)
        mseloss = ((outputs - y) ** 2).mean()
        madloss = torch.abs(outputs - y).mean()
        loss1 = loss1 #+ mseloss * 0.5 + madloss * 0.5
        writer.add_scalar("loss_mse/train", mseloss, self.current_epoch * len(self.training_dataloader) + batch_idx)
        writer.add_scalar("loss_mad/train", madloss, self.current_epoch * len(self.training_dataloader) + batch_idx)
        writer.add_scalar("loss_bce/train", bceloss, self.current_epoch * len(self.training_dataloader) + batch_idx)
        writer.add_scalar("loss_dice/train", diceloss, self.current_epoch * len(self.training_dataloader) + batch_idx)
        writer.add_scalar("loss/train", loss1, self.current_epoch * len(self.training_dataloader) + batch_idx)
        if torch.isnan(loss1):
            print("Loss nan encountered")
        self.log("train/total_loss", loss1.item(),on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss1}

    def validation_step(self, batch, batch_idx):
        '''
        x,y,xyxys= batch
        batch_size = x.size(0)
        outputs = self(x)
        print("outputs.shape", outputs.shape, "y.shape", y.shape)
        if outputs.shape != y.shape:
          outputs = F.interpolate(outputs, y.shape[-2:], mode="bilinear")
        diceloss = self.loss_func1(outputs, y)
        bceloss = self.loss_func2(outputs, y)
        loss1 = self.loss_func(outputs, y)
        mseloss = ((outputs - y) ** 2).mean()
        madloss = torch.abs(outputs - y).mean()
        writer.add_scalar("loss_mse/valid", mseloss, self.current_epoch * len(self.valid_dataloader) + batch_idx)
        writer.add_scalar("loss_mad/valid", madloss, self.current_epoch * len(self.valid_dataloader) + batch_idx)
        writer.add_scalar("loss_bce/valid", bceloss, self.current_epoch * len(self.valid_dataloader) + batch_idx)
        writer.add_scalar("loss_dice/valid", diceloss, self.current_epoch * len(self.valid_dataloader) + batch_idx)
        writer.add_scalar("loss/valid", loss1, self.current_epoch * len(self.valid_dataloader) + batch_idx)
        y_preds = torch.sigmoid(outputs).to('cpu')
        for i, (x1, y1, x2, y2) in enumerate(xyxys):
            #self.mask_pred[y1:y2, x1:x2] += F.interpolate(y_preds[i].unsqueeze(0).float(),scale_factor=4,mode='bilinear').squeeze(0).squeeze(0).numpy()
            #ValueError: operands could not be broadcast together with shapes (64,32) (64,64) (64,32) # 2024-05-08
            y2,x2= min(y2, self.mask_pred.shape[0]), min(x2, self.mask_pred.shape[1])
            if x2 <= x1 or y2 <= y1 or x1 >= self.mask_pred.shape[1] or y1 >= self.mask_pred.shape[0]:
              # How did this happen? tensor([1440,  160, 1504,  224], device='cuda:3') (3712, 1408)
              print("How did this happen?", xyxys[i], self.mask_pred.shape)
              continue
            self.mask_pred[y1:y2, x1:x2] += F.interpolate(y_preds[i].unsqueeze(0).float(),(y2-y1,x2-x1),mode='bilinear').squeeze(0).squeeze(0).numpy()
            self.mask_count[y1:y2, x1:x2] += np.ones((y2-y1,x2-x1)) #(self.hparams.size, self.hparams.size))

        self.log("val/total_loss", loss1.item(),on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss1}
        '''

        x,y,xyxys,ids= batch
        batch_size = x.size(0)
        outputs = self(x)
        print("outputs.shape", outputs.shape)
        if y.shape != outputs.shape:
          #y=F.interpolate(y,(4,4)) # TODO SethS: DISABLE ME!
          #y=F.interpolate(y,outputs.shape[-2:], mode="bilinear") # Auto-adjust output shape.
          y=F.interpolate(y,outputs.shape[-2:], mode="area") # Auto-adjust output shape.
        y = F.interpolate(y, (1,1), mode="area")
        outputs = F.interpolate(outputs, (1,1), mode="area")

        loss1 = self.loss_func(outputs, y)
        y_preds = torch.sigmoid(outputs).to('cpu')
        #y_preds = torch.minimum(1, torch.relu(outputs).to('cpu'))
        #y_preds = torch.relu(outputs).to('cpu')
        print("min,max preds", y_preds.min().item(), y_preds.max().item())
        for i, (x1, y1, x2, y2) in enumerate(xyxys):
            x1,x2,y1,y2 = (int(c) for c in (x1,x2,y1,y2))
            print(x1,x2,y1,y2, "xyxys,", ids[i], "mask_pred.shape", self.mask_pred[ids[i]].shape)
            x1,x2,y1,y2 = x1,min(x2,self.mask_pred[ids[i]].shape[1]),y1,min(y2,self.mask_pred[ids[i]].shape[0]) # SethS Why would this be necessary? It isn't... Unless the validation masks and pred_shapes are different.
            if x2-x1 <= 0 or y2-y1 <= 0:
              print("Skipping OOB xyxys!",x1,y1,x2,y2)
              continue
            #print(x1, x2, y1, y2, "id", ids[i], [(k,v.shape) for k,v in self.mask_pred.items()], [(k,v.shape) for k,v in self.mask_count.items()])
            #self.mask_pred[ids[i]][y1:y2, x1:x2] += F.interpolate(y_preds[i].unsqueeze(0).float(),scale_factor=4,mode='bilinear').squeeze(0).squeeze(0).numpy()
            #self.mask_pred[ids[i]][y1:y2, x1:x2] += F.interpolate(y_preds[i].unsqueeze(0).float(),(y2-y1,x2-x1),mode='bilinear').squeeze(0).squeeze(0).numpy()
            if True or y2-y1 > 1 or x2-x1 > 1:
              #self.mask_pred[ids[i]][y1:y2, x1:x2] += np.clip(F.interpolate(y_preds[i].unsqueeze(0).float(),(y2-y1,x2-x1),mode='bilinear').squeeze(0).squeeze(0).numpy(), 0, 1)
              #self.mask_pred[ids[i]][y1:y2, x1:x2] += F.interpolate(y_preds[i].unsqueeze(0).float(),(y2-y1,x2-x1),mode='bilinear').squeeze(0).squeeze(0).numpy()
              #self.mask_pred[ids[i]][y1:y2, x1:x2] += F.interpolate(y_preds[i].unsqueeze(0).float(),(y2-y1,x2-x1),mode='area').squeeze(0).squeeze(0).numpy() # It has to be squashed DOWN, not upsampled. Same sampling algorithm problems I used to run i>
              self.mask_pred[ids[i]][y1:y2, x1:x2] += F.interpolate(y_preds[i].unsqueeze(0).float(),(y2-y1,x2-x1),mode='bilinear').squeeze(0).squeeze(0).numpy() # It has to be squashed DOWN, not upsampled. Same sampling algorithm problems I used to ru>
            else:
              #self.mask_pred[ids[i]][y1:y2, x1:x2] += np.clip(y_preds[i].unsqueeze(0).float().squeeze(0).squeeze(0).numpy(), 0, 1) # TODO: What if I don't apply np.clip while summing, but only AFTER???
              self.mask_pred[ids[i]][y1:y2, x1:x2] += y_preds[i].unsqueeze(0).float().squeeze(0).squeeze(0).numpy() # TODO: What if I don't apply np.clip while summing, but only AFTER???
            self.mask_count[ids[i]][y1:y2, x1:x2] += np.ones((y2-y1, x2-x1))
            #print("Keeping xyxys", x1,y1,x2,y2, xyxys[i], ids[i])
        self.log("val/total_loss", loss1.item(),on_step=True, on_epoch=True, prog_bar=True)
        self.writer.add_scalar('Loss/valid', loss1.item(), self.current_epoch)
        diceloss = self.loss_func1(outputs, y)
        bceloss = self.loss_func2(outputs, y)
        loss1 = self.loss_func(outputs, y)
        mseloss = ((outputs - y) ** 2).mean()
        madloss = torch.abs(outputs - y).mean()
        self.writer.add_scalar("loss_mse/valid", mseloss, self.current_epoch * len(self.valid_dataloaders) + batch_idx)
        self.writer.add_scalar("loss_mad/valid", madloss, self.current_epoch * len(self.valid_dataloaders) + batch_idx)
        self.writer.add_scalar("loss_bce/valid", bceloss, self.current_epoch * len(self.valid_dataloaders) + batch_idx)
        self.writer.add_scalar("loss_dice/valid", diceloss, self.current_epoch * len(self.valid_dataloaders) + batch_idx)
        self.writer.add_scalar("loss/valid", loss1, self.current_epoch * len(self.valid_dataloaders) + batch_idx)



    '''
    def on_validation_epoch_end(self):
        self.mask_pred = np.divide(self.mask_pred, self.mask_count, out=np.zeros_like(self.mask_pred), where=self.mask_count!=0)
        print("mask pred", self.mask_pred.shape)
        wandb_logger.log_image(key="masks", images=[np.clip(self.mask_pred,0,1)], caption=["probs"])
        cv2.imwrite(self.model_name+"_"+CFG.valid_id+"_scale"+str(CFG.scale)+"_size"+str(CFG.size)+"_tile_size"+str(CFG.tile_size)+"_stride"+str(CFG.stride)+"_epoch"+str(self.current_epoch)+".jpg", np.clip(self.mask_pred,0,1)*255) 
        writer.add_image("image/valid_"+CFG.valid_id, np.clip(self.mask_pred,0,1), self.current_epoch, dataformats="HW")
        #reset mask
        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)
    '''
    def on_validation_epoch_end(self):
        print("Experiment name", self.name)
        if isinstance(self.hparams.pred_shape, dict):
          #print("mask_pred has keys", self.mask_pred.keys())
          for k,pred_shape in self.hparams.pred_shape.items():
            self.mask_pred[k] = np.divide(self.mask_pred[k], self.mask_count[k], out=np.zeros_like(self.mask_pred[k]), where=self.mask_count[k]!=0)
            if self.mask_pred[k] is None or np.product(self.mask_pred[k].shape) == 0:
              print("No mask pred for key", k, self.mask_pred[k])
            else:
              self.wandb_logger.log_image(key=f"preds_{k}", images=[np.clip(self.mask_pred[k],0,1)], caption=["probs"])
              self.writer.add_image(f'{k}_preds', (self.mask_pred[k] - self.mask_pred[k].min()) / max(self.mask_pred[k].max()-self.mask_pred[k].min(), 0.01), self.current_epoch, dataformats="HW")
              if self.val_masks is not None and k in self.val_masks:
                self.wandb_logger.log_image(key=f"trues_{k}", images=[np.clip(self.val_masks[k],0,255)], caption=["probs"])
                self.writer.add_image(f'{k}_trues', np.clip(self.val_masks[k][:,:,0],0,255), self.current_epoch, dataformats="HW")
              else:
                print("val_masks missing", k, self.val_masks.keys())
        else:
          self.mask_pred = np.divide(self.mask_pred, self.mask_count, out=np.zeros_like(self.mask_pred), where=self.mask_count!=0)
          self.wandb_logger.log_image(key="preds", images=[np.clip(self.mask_pred,0,1)], caption=["probs"])
          self.writer.add_image(f'preds', self.mask_pred, self.current_epoch, dataformats="HW")
          if self.val_masks is not None:
            self.wandb_logger.log_image(key=f"trues_{k}", images=[np.clip(self.val_masks[:,:,0],0,255)], caption=["probs"])
        #print("Done logging! Resetting mask...")
        #reset mask
        if isinstance(self.hparams.pred_shape, dict):
          self.mask_pred = {}
          self.mask_count = {}
          for k,pred_shape in self.hparams.pred_shape.items():
            self.mask_pred[k] = np.zeros(pred_shape)
            self.mask_count[k] = np.zeros(pred_shape)
        else:
          self.mask_pred = np.zeros(self.hparams.pred_shape)
          self.mask_count = np.zeros(self.hparams.pred_shape)
        #print("Done resetting masks!")
        #self.train_dataset.labels = reload_masks(self.train_dataset.labels, CFG)
        #exit()




    def configure_optimizers(self):

        optimizer = AdamW(self.parameters(), lr=CFG.lr)
    
        scheduler = get_scheduler(CFG, optimizer)
        return [optimizer],[scheduler]



class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    """
    https://www.kaggle.com/code/underwearfitting/single-fold-training-of-resnet200d-lb0-965
    """
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(
            optimizer, multiplier, total_epoch, after_scheduler)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

def get_scheduler(cfg, optimizer):
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 10, eta_min=1e-6)
    scheduler = GradualWarmupSchedulerV2(
        optimizer, multiplier=1.0, total_epoch=1, after_scheduler=scheduler_cosine)

    return scheduler

def scheduler_step(scheduler, avg_val_loss, epoch):
    scheduler.step(epoch)
   


from argparse import ArgumentParser
#from config import CFG

parser = ArgumentParser()
parser.add_argument('--scale', type=int, default=1, required=False)
parser.add_argument('--tile_size', type=int, default=256, required=False)
parser.add_argument('--size', type=int, default=64, required=False)
parser.add_argument('--stride', type=int, default=32, required=False)
parser.add_argument('--model', type=str, default="pygoflat", required=False)
parser.add_argument('--load', type=str, default="", required=False)
parser.add_argument('--complexity', type=int, default=16, required=False)
parser.add_argument('--epochs', type=int, default=12, required=False)
parser.add_argument('--batch_size', type=int, default=256, required=False)
parser.add_argument('--val_batch_size', type=int, default=256, required=False)
parser.add_argument('--minbatches', type=int, default=1000000, required=False)
parser.add_argument('--mode', type=str, default="normal", required=False)
#parser.add_argument('--model', type=str, default="i3d", required=False)
args = parser.parse_args()

CFG.scale = args.scale
CFG.tile_size = args.tile_size
CFG.size = args.size
CFG.stride = args.stride
CFG.train_batch_size = args.batch_size

from dataloaders import *



#fragment_id = CFG.valid_id

#valid_mask_gt = cv2.imread(CFG.comp_dataset_path + f"train_scrolls/{fragment_id}/{fragment_id}_inklabels.png", 0)
# valid_mask_gt=cv2.resize(valid_mask_gt,(valid_mask_gt.shape[1]//2,valid_mask_gt.shape[0]//2),cv2.INTER_AREA)
#pred_shape=valid_mask_gt.shape
torch.set_float32_matmul_precision('medium')

scroll1val = ['20231012184423']
fragments = ['20230702185753','20230929220926','20231005123336','20231012184423','20231007101619','20231016151002','20231022170901','20231031143852','20231106155351','20231210121321','20231221180251','20230820203112']
scroll4_ids = ['20231111135340', '20231122192640', '20231210132040', '20240304141530', '20231215151901', '20240304144030'] + scroll1val
scroll3_ids = ['20231030220150', '20231031231220']
#scroll2_ids = []
with open("scroll2.ids", 'r') as f:
  scroll2_ids = [line.strip() for line in f.readlines()]
#train_scrolls = "train_scrolls" if os.path.isdir("train_scrolls") else "train_scrolls2"
train_scrolls = CFG.basepath #"train_scrolls" if os.path.isdir("train_scrolls") else "train_scrolls2"

'''
fragments=['20230820203112']
fragments=['20231012184423']
enc_i,enc,fold=0,'i3d',0
for fid in fragments:
    CFG.valid_id=fid
    fragment_id = CFG.valid_id
    run_slug=f'training_scrolls_valid={fragment_id}_{CFG.size}x{CFG.size}_submissionlabels_{args.model}11_scale{CFG.scale}_redo'
    writer = SummaryWriter("runs/"+run_slug)

    valid_mask_gt = cv2.imread(CFG.comp_dataset_path + f"train_scrolls/{fragment_id}/{fragment_id}_inklabels.png", 0)

    pred_shape=tuple(t//CFG.scale for t in valid_mask_gt.shape)
    train_images, train_masks, train_xyxys, train_ids, valid_images, valid_masks, valid_xyxys, valid_ids = get_train_valid_dataset()
'''


enc_i,enc,fold=0,'i3d',0
fid = CFG.valid_id
if True: #for fid in fragments:
    #train_images, train_masks, train_xyxys, train_ids, valid_images, valid_masks, valid_xyxys, valid_ids = [[,],]*8
    #origscale = CFG.scale # TODO Multiscale training
    #for scale in [CFG.scale // 2, CFG.scale, CFG.scale * 2]:
    #  train_images, train_masks, train_xyxys, train_ids, valid_images, valid_masks, valid_xyxys, valid_ids = get_train_valid_dataset(CFG, train_ids, valid_ids, scale=CFG.scale)
    ''' # Use only if one training image.
    r = CFG.size//2
    tm = train_masks[fid]
    #train_xyxys = [(c[1]-r,c[0]-r,c[1]+r,c[0]+r) for c in np.argwhere(train_masks[fid] >= 0).tolist() if c[0]-r > 0 and c[1]-r > 0 and c[0]+r < tm.shape[0] and c[1]+r < tm.shape[1]]
    #train_xyxys = [(c[1]-r,c[0]-r,c[1]+r,c[0]+r) for c in np.argwhere(train_masks[fid] >= 0).tolist() if c[0]-r > 0 and c[1]-r > 0 and c[0]+r < tm.shape[0] and c[1]+r < tm.shape[1] and c[0] % CFG.stride == 0 and c[1] % CFG.stride == 0]
    train_stride = 1 #CFG.stride // 16 # TODO Pygo optimize this!
    sidx = -1
    eidx = 0
    for i,v in enumerate(train_ids):
      if v == fid:
        if sidx < 0:
          sidx = i
        eidx = max(eidx,i)
    del train_ids[sidx:eidx+1]
    del train_xyxys[sidx:eidx+1]
    new_train_xyxys = [(c[1]-r,c[0]-r,c[1]+r,c[0]+r) for c in np.argwhere(train_masks[fid] >= 0).tolist() if c[0]-r > 0 and c[1]-r > 0 and c[0]+r < tm.shape[0] and c[1]+r < tm.shape[1] and c[0] % train_stride == 0 and c[1] % train_stride == 0]
    if len(new_train_xyxys) < 10000:
      new_train_xyxys = new_train_xyxys * (10000 // len(new_train_xyxys))
    new_train_ids = [fid] * len(new_train_xyxys) #train_ids * (100000 // len(train_ids))
    train_xyxys += new_train_xyxys
    train_ids += new_train_ids
    print(len(train_images)
    '''
    CFG.valid_id=fid
    fragment_id = CFG.valid_id
    import datetime
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = f'{args.model}_scale{CFG.scale}_size{CFG.size}_tilesize{CFG.tile_size}_stride{CFG.stride}_{ts}'
    run_slug=name #f'training_scrolls_valid={fragment_id}_{model}_{CFG.size}size_{CFG.tile_size}tile_size_{CFG.stride}stride_{CFG.scale}scale'
    pred_shape = {}
    valid_ids = list(set(scroll4_ids + scroll3_ids + scroll2_ids)) #['20231005123336']) #set(list(train_ids) + scroll4_ids + scroll2_ids + scroll3_ids)
    train_ids = list((set(fragments) - set(valid_ids)) - set(scroll1val)) #+ set( #scroll4_ids) 
    for scroll_id in set(valid_ids + train_ids): #fragments + scroll4_ids + scroll2_ids + scroll3_ids:
      valid_mask_gt = cv2.imread(f"{train_scrolls}/{scroll_id}/{scroll_id}_inklabels.png", 0)
      if valid_mask_gt is None:
        valid_mask_gt = cv2.imread(f"{train_scrolls}/{scroll_id}/{scroll_id}_mask.png", 0)
        if valid_mask_gt is None:
          print("Validation mask not found, skipping!", f"{train_scrolls}/{scroll_id}/{scroll_id}_mask.png")
          continue
        if not os.path.isfile(f"{train_scrolls}/{scroll_id}/{scroll_id}_inklabels.png"):
          print("Writing nonexistent ink labels file", f"{train_scrolls}/{scroll_id}/{scroll_id}_inklabels.png")
          print(scroll_id, f"{train_scrolls}/{scroll_id}/{scroll_id}_mask.png", "valid_mask_gt", valid_mask_gt)
          cv2.imwrite(f"{train_scrolls}/{scroll_id}/{scroll_id}_inklabels.png", np.zeros_like(valid_mask_gt))
      pred_shape[scroll_id]=tuple(int(c/CFG.scale) for c in valid_mask_gt.shape)
      print("pred shape for scroll", scroll_id, pred_shape[scroll_id])
      print("pred_shape", scroll_id, pred_shape[scroll_id])
    print("Got all validation scroll ID shapes for prediction and logging")
    norm=fold==1
    wandb_logger = WandbLogger(project="vesivus",name=run_slug+f'{enc}_finetune')
    print('FOLD : ',fold)
    multiplicative = lambda epoch: 0.9
    train_images, train_masks, train_xyxys, train_ids, valid_images, valid_masks, valid_xyxys, valid_ids = get_train_valid_dataset(CFG, train_ids, valid_ids, start_idx=0, end_idx=65, scale=CFG.scale)
    print(len(train_images))
    valid_xyxys = np.stack(valid_xyxys)
    train_dataset = CustomDataset(
        train_images, CFG, labels=train_masks, xyxys=train_xyxys, ids=train_ids, transform=get_transforms(data='train', cfg=CFG))
    valid_dataset = CustomDatasetTest(
        valid_images, CFG,xyxys=valid_xyxys, labels=valid_masks, ids=valid_ids, transform=get_transforms(data='valid', cfg=CFG))

    train_loader = DataLoader(train_dataset,
                                batch_size=CFG.train_batch_size,
                                shuffle=True,
                                num_workers=CFG.num_workers, pin_memory=True, drop_last=True,
                                )
    valid_loader = DataLoader(valid_dataset,
                                batch_size=CFG.valid_batch_size,
                                shuffle=False,
                                num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
    if args.mode == "dataonly":
      exit()

    wandb_logger = WandbLogger(project="vesivus",name=run_slug+f'{enc}_finetune')
    norm=fold==1

    non_local=True
    #if "pygo" in args.model:
    #  from pygoflat import 
    '''
    if len(args.load) == 0:
      model=RegressionPLModel(enc='i3d',pred_shape=pred_shape,size=CFG.size, backbone=args.model)
    else:
      model=RegressionPLModel.load_from_checkpoint(args.load, enc='i3d',pred_shape=pred_shape,size=CFG.size, name=run_slug, backbone=args.model)
    '''
    model=RegressionPLModel(enc='i3d',pred_shape=pred_shape,size=CFG.size, train_dataset=train_dataset, backbone=args.model, wandb_logger=wandb_logger, name=name, val_masks=valid_masks, complexity=args.complexity)
    if len(args.load) > 0:
      model=RegressionPLModel.load_from_checkpoint(args.load, backbone=args.model, wandb_logger=wandb_logger, enc="i3d", pred_shape=pred_shape, size=CFG.size, train_dataset=train_dataset, name=name, val_masks=valid_masks, complexity=args.complexity)
    wandb_logger.watch(model, log="all", log_freq=100)

    print('FOLD : ',fold)
    wandb_logger.watch(model, log="all", log_freq=100)
    multiplicative = lambda epoch: 0.9
    model.valid_dataloader = valid_loader
    model.training_dataloader = train_loader
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=torch.cuda.device_count(),
        logger=wandb_logger,
        default_root_dir='./', #"/content/gdrive/MyDrive/vesuvius_model/training/outputs",
        accumulate_grad_batches=1,
        precision='16-mixed',
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        strategy='ddp_find_unused_parameters_true',
        callbacks=[ModelCheckpoint(filename=f'{args.model}12_64_{fid}_{fold}_fr_{enc}_scale{CFG.scale}_size{CFG.size}_stride{CFG.stride}',dirpath=CFG.model_dir,monitor='train/total_loss',mode='min',save_top_k=5),

                    ],

    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    wandb.finish()
