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

import datetime
import segmentation_models_pytorch as smp
import numpy as np
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch
from warmup_scheduler import GradualWarmupScheduler
from scipy import ndimage
import time
import json
import numba
from numba import jit

from argparse import ArgumentParser
from config import CFG

parser = ArgumentParser()
parser.add_argument('--scale', type=int, default=1, required=False)
parser.add_argument('--tile_size', type=int, default=64, required=False)
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
CFG.valid_id = "20240304144030"
CFG.train_batch_size = CFG.valid_batch_size = args.batch_size
import math
CFG.valid_batch_size = args.val_batch_size
CFG.valid_size = 1 #int(2**((10-math.log2(args.scale)))) # 1024 is 1, 512 is 2, 
CFG.valid_tile_size = 1 # CFG.valid_size * 4 # (10-math.log2(args.scale))+1 # MAKE BIGGER if possible!
CFG.valid_stride = 1 #CFG.valid_size // 2 #max(1, 10-math.log2(args.scale)) # Bigger scale? Smaller stride, tile size, etc.

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from plmodel import *
from dataloaders import *

scroll1val = ['20231012184423']
fragments = ['20230702185753','20230929220926','20231005123336','20231012184423','20231007101619','20231016151002','20231022170901','20231031143852','20231106155351','20231210121321','20231221180251','20230820203112']
scroll4_ids = ['20231111135340', '20231122192640', '20231210132040', '20240304141530', '20231215151901', '20240304144030'] + scroll1val
scroll3_ids = ['20231030220150', '20231031231220']
with open("scroll2.ids", 'r') as f:
  scroll2_ids = [line.strip() for line in f.readlines()]
#train_scrolls = "train_scrolls" if os.path.isdir("train_scrolls") else "train_scrolls2"
train_scrolls = CFG.basepath #"train_scrolls" if os.path.isdir("train_scrolls") else "train_scrolls2"

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
    name = f'{args.model}_scale{CFG.scale}_size{CFG.size}_tilesize{CFG.tile_size}_stride{CFG.stride}_withnoninkmasks_withpapyrusedges_simplerxygenerationfixedvalxys_normalizeinception_sigmoidagain_reluagain_valtilesize1_originalchannelrange'
    run_slug=name #f'training_scrolls_valid={fragment_id}_{model}_{CFG.size}size_{CFG.tile_size}tile_size_{CFG.stride}stride_{CFG.scale}scale'
    pred_shape = {}
    valid_ids = set(['20231005123336']) #set(list(train_ids) + scroll4_ids + scroll2_ids + scroll3_ids)
    for scroll_id in valid_ids: #fragments + scroll4_ids + scroll2_ids + scroll3_ids:
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
    train_ids = set(fragments) - set(scroll4_ids)
    train_images, train_masks, train_xyxys, train_ids, valid_images, valid_masks, valid_xyxys, valid_ids = get_train_valid_dataset(CFG, train_ids, valid_ids, start_idx=0, end_idx=65, scale=CFG.scale)
    if len(train_xyxys) < args.minbatches:
      train_xyxys = train_xyxys * (args.minbatches // len(train_xyxys))
      train_ids = train_ids * (args.minbatches // len(train_ids))
    print("Training images", len(train_images), train_images.keys(), train_masks.keys(), "xyxys", len(train_xyxys), len(train_ids))
    print("VALID xyxys:", len(valid_xyxys), valid_xyxys[-10:])
    valid_xyxys = np.stack(valid_xyxys)
    print("VX", valid_xyxys.shape, valid_xyxys[-10:])
    train_dataset = CustomDataset(
        train_images, CFG, labels=train_masks, xyxys=train_xyxys, ids=train_ids, transform=get_transforms(data='train', cfg=CFG), scale=CFG.scale)
    valid_dataset = CustomDatasetTest(
        valid_images, CFG,xyxys=valid_xyxys, labels=valid_masks, ids=valid_ids, transform=get_transforms(data='valid', cfg=CFG), scale=CFG.scale)
    train_loader = DataLoader(train_dataset,
                                batch_size=CFG.train_batch_size,
                                shuffle=True,
                                num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset,
                                batch_size=CFG.valid_batch_size,
                                shuffle=False,
                                num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
    print(len(valid_loader))
    model=RegressionPLModel(enc='i3d',pred_shape=pred_shape,size=CFG.size, train_dataset=train_dataset, backbone=args.model, wandb_logger=wandb_logger, name=name, val_masks=valid_masks, complexity=args.complexity)
    if len(args.load) > 0:
      model=RegressionPLModel.load_from_checkpoint(args.load, backbone=args.model, wandb_logger=wandb_logger, enc="i3d", pred_shape=pred_shape, size=CFG.size, train_dataset=train_dataset, name=name, val_masks=valid_masks, complexity=args.complexity)
    wandb_logger.watch(model, log="all", log_freq=100)

model.train_dataloaders = train_loader
model.valid_dataloaders = valid_loader
for precision in 'bf16-mixed', '16-mixed':
  try:
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=torch.cuda.device_count(),
        logger=wandb_logger,
        default_root_dir="./", #/content/gdrive/MyDrive/vesuvius_model/training/outputs",
        accumulate_grad_batches=1,
        check_val_every_n_epoch=1,
        precision=precision,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        strategy='ddp_find_unused_parameters_true',
        callbacks=[ModelCheckpoint(filename=f'{name}',dirpath=CFG.model_dir,monitor='train/total_loss',mode='min',save_top_k=CFG.epochs),],
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    break
  except RuntimeError as ex:
    if "bfloat" in str(ex):
      print(ex)
      continue
    else:
      raise(ex)
wandb.finish()
