import sys
print("Importing...")
import os.path as osp
import os
os.environ["WANDB_MODE"] = "offline"
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
import pandas as pd

import wandb

from torch.utils.data import DataLoader
print("1")

import pandas as pd
import os
import random
from contextlib import contextmanager
import cv2

import scipy as sp
import numpy as np
import fastnumpyio as fnp
import pandas as pd

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

def read_image_mask(fragment_id,start_idx=15,end_idx=45):

    images = []

    # idxs = range(65)
    mid = 65 // 2
    start = mid - CFG.in_chans // 2
    end = mid + CFG.in_chans // 2
    idxs = range(start_idx, end_idx)


    t = time.time()
    rescaled = False
    if CFG.scale != 1 and os.path.isfile(CFG.comp_dataset_path + f"train_scrolls/{fragment_id}_{start_idx}-{end_idx}_{CFG.scale}.npy"):
      images = np.load(CFG.comp_dataset_path + f"train_scrolls/{fragment_id}_{start_idx}-{end_idx}_{CFG.scale}.npy")
      pad0 = (CFG.tile_size - images.shape[0] % CFG.tile_size)
      pad1 = (CFG.tile_size - images.shape[1] % CFG.tile_size)
      print(time.time()-t, "seconds taken to load images from", CFG.comp_dataset_path + f"train_scrolls/{fragment_id}_{start_idx}-{end_idx}_{CFG.scale}.npy")
      rescaled = True
    elif os.path.isfile(CFG.comp_dataset_path + f"train_scrolls/{fragment_id}/layers/{fragment_id}.npy"):
      images = np.load(CFG.comp_dataset_path + f"train_scrolls/{fragment_id}/layers/{fragment_id}.npy")
      pad0 = (CFG.tile_size - images.shape[0] % CFG.tile_size)
      pad1 = (CFG.tile_size - images.shape[1] % CFG.tile_size)
      print(time.time()-t, "seconds taken to load images from", CFG.comp_dataset_path + f"train_scrolls/{fragment_id}/layers/{fragment_id}.npy")
    else:
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
      np.save(CFG.comp_dataset_path + f"train_scrolls/{fragment_id}/layers/{fragment_id}.npy", images)
      print(time.time()-t, "seconds taken to save images as npy.")

    scale = CFG.scale
    basepath = CFG.basepath
    if fragment_id=='20231022170900':
        mask = cv2.imread(f"{basepath}/{fragment_id}/{fragment_id}_inklabels.tiff", 0)
    else:
        mask = cv2.imread(f"{basepath}/{fragment_id}/{fragment_id}_inklabels.png", 0)
    if mask is None:
      print("Warning: No GT found for", fragment_id)
      mask = np.zeros_like(images[:,:,0])

    if scale != 1 and not rescaled:
      print("Rescaling image...", fragment_id, images.shape, "down by", scale)
      t = time.time()
      images = (block_reduce(images, block_size=(scale,scale,1), func=np.mean, cval=np.mean(images))+0.5).astype(np.uint8)
      print("Rescaling took", time.time()-t, "seconds.", images.shape)
      np.save(f"train_scrolls/{fragment_id}_{start_idx}-{end_idx}_{scale}.npy", images) # Other parts too
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
    print("Reading fragment mask", f"{basepath}/{fragment_id}/{fragment_id}_mask.png")
    fragment_mask=cv2.imread(f"{basepath}/{fragment_id}/{fragment_id}_mask.png", 0)
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
def generate_xyxys_ids(fragment_id, image, mask, fragment_mask, tile_size, size, stride, is_valid=False):
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
        if False and os.path.isfile(fragment_id + ".ids.json"):
          with open(fragment_id + ".ids.json", 'r') as f:
            id = json.load(f)
        if False and os.path.isfile(fragment_id + ".xyxys.json"):
          with open(fragment_id + ".xyxys.json", 'r') as f:
            xyxy = json.load(f)
        else:
          xyxy, id = generate_xyxys_ids(fragment_id, image, mask, fragment_mask, CFG.tile_size, CFG.size, CFG.stride, is_valid)
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
    train_ids = set(['20230702185753','20230929220926','20231005123336']) #,'20231007101619','20231012184423','20231016151002','20231022170901','20231031143852','20231106155351','20231210121321','20231221180251','20230820203112']) - set([CFG.valid_id])
    valid_ids = set([CFG.valid_id])
    train_images, train_masks, train_xyxys, train_ids = get_xyxys(train_ids, False)
    valid_images, valid_masks, valid_xyxys, valid_ids = get_xyxys(valid_ids, True)
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
                label=F.interpolate((label/255).unsqueeze(0).float(),(self.cfg.size//4,self.cfg.size//4)).squeeze(0)
            return image, label,xy
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
                label=F.interpolate((label/255).unsqueeze(0).float(),(self.cfg.size//4,self.cfg.size//4)).squeeze(0)
            return image, label
class CustomDatasetTest(Dataset):
    def __init__(self, images, xyxys, ids, cfg, transform=None):
        self.images = images
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
        if np.product(image.shape) == 0:
          print("Erroneous bounds", xy, id, self.images[id].shape)
          return None,None
        if self.transform:
            data = self.transform(image=image)
            image = data['image'].unsqueeze(0)

        return image,xy



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
    def __init__(self,pred_shape,size=256,enc='',with_norm=False):
        super(RegressionPLModel, self).__init__()

        self.save_hyperparameters()
        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)

        self.loss_func1 = smp.losses.DiceLoss(mode='binary')
        self.loss_func2= smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25)
        self.loss_func= lambda x,y:0.5 * self.loss_func1(x,y)+0.5*self.loss_func2(x,y)

        self.backbone=InceptionI3d(in_channels=1,num_classes=512,non_local=True)        
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
        x, y, xys = batch
        outputs = self(x)
        print("outputs.shape", outputs.shape, "y.shape", y.shape)
        if outputs.shape != y.shape:
          outputs = F.interpolate(outputs, y.shape[-2:], mode="bilinear")
        loss1 = self.loss_func(outputs, y)
        if torch.isnan(loss1):
            print("Loss nan encountered")
        self.log("train/total_loss", loss1.item(),on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss1}

    def validation_step(self, batch, batch_idx):
        x,y,xyxys= batch
        batch_size = x.size(0)
        outputs = self(x)
        loss1 = self.loss_func(outputs, y)
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
    
    def on_validation_epoch_end(self):
        self.mask_pred = np.divide(self.mask_pred, self.mask_count, out=np.zeros_like(self.mask_pred), where=self.mask_count!=0)
        wandb_logger.log_image(key="masks", images=[np.clip(self.mask_pred,0,1)], caption=["probs"])

        #reset mask
        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)
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
args = parser.parse_args()

CFG.scale = args.scale
CFG.tile_size = args.tile_size
CFG.size = args.size
CFG.stride = args.stride
CFG.train_batch_size = args.batch_size

#from dataloaders import *



#fragment_id = CFG.valid_id

#valid_mask_gt = cv2.imread(CFG.comp_dataset_path + f"train_scrolls/{fragment_id}/{fragment_id}_inklabels.png", 0)
# valid_mask_gt=cv2.resize(valid_mask_gt,(valid_mask_gt.shape[1]//2,valid_mask_gt.shape[0]//2),cv2.INTER_AREA)
#pred_shape=valid_mask_gt.shape
torch.set_float32_matmul_precision('medium')

fragments=['20230820203112']
enc_i,enc,fold=0,'i3d',0
for fid in fragments:
    CFG.valid_id=fid
    fragment_id = CFG.valid_id
    run_slug=f'training_scrolls_valid={fragment_id}_{CFG.size}x{CFG.size}_submissionlabels_wild11_scale{CFG.scale}'

    valid_mask_gt = cv2.imread(CFG.comp_dataset_path + f"train_scrolls/{fragment_id}/{fragment_id}_inklabels.png", 0)

    pred_shape=tuple(t//CFG.scale for t in valid_mask_gt.shape)
    train_images, train_masks, train_xyxys, train_ids, valid_images, valid_masks, valid_xyxys, valid_ids = get_train_valid_dataset()
    print(len(train_images))
    valid_xyxys = np.stack(valid_xyxys)
    train_dataset = CustomDataset(
        train_images, CFG, labels=train_masks, xyxys=train_xyxys, ids=train_ids, transform=get_transforms(data='train', cfg=CFG))
    valid_dataset = CustomDataset(
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

    wandb_logger = WandbLogger(project="vesivus",name=run_slug+f'{enc}_finetune')
    norm=fold==1
    if len(args.load) == 0:
      model=RegressionPLModel(enc='i3d',pred_shape=pred_shape,size=CFG.size)
    else:
      model=RegressionPLModel.load_from_checkpoint(args.load, enc='i3d',pred_shape=pred_shape,size=CFG.size, name=run_slug)

    print('FOLD : ',fold)
    wandb_logger.watch(model, log="all", log_freq=100)
    multiplicative = lambda epoch: 0.9

    trainer = pl.Trainer(
        max_epochs=24,
        accelerator="gpu",
        devices=torch.cuda.device_count(),
        logger=wandb_logger,
        default_root_dir='./', #"/content/gdrive/MyDrive/vesuvius_model/training/outputs",
        accumulate_grad_batches=1,
        precision='16-mixed',
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        strategy='ddp_find_unused_parameters_true',
        callbacks=[ModelCheckpoint(filename=f'wild12_64_{fid}_{fold}_fr_{enc}_scale{CFG.scale}_size{CFG.size}_stride{CFG.stride}',dirpath=CFG.model_dir,monitor='train/total_loss',mode='min',save_top_k=CFG.epochs),

                    ],

    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    wandb.finish()
