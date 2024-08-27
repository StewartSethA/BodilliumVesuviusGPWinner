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
from i3dallnl import InceptionI3d
#from pygoi3d_simple import InceptionI3d
import torch.nn as nn
import torch
from warmup_scheduler import GradualWarmupScheduler
from scipy import ndimage
import time
import json
import numba
from numba import jit

from config import CFG
CFG.scale = 8 #1024
#CFG.tile_size = 64
CFG.tile_size = 256 #1024 #8
CFG.size = 128 #8 # WAS 64
CFG.stride = 64 # WAS 64
CFG.valid_id = "20231012184423"
#cfg_init(CFG)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from plmodel import *

from dataloaders import *

#fragments=[CFG.valid_id] + ['20230702185753','20230929220926','20231005123336','20231012184423'] #'20230820203112']
fragments = ['20230702185753','20230929220926','20231005123336','20231012184423','20231016151002','20231022170901','20231031143852','20231106155351','20231210121321','20231221180251','20230820203112']

#fragments= [] #['20230702185753','20230929220926','20231005123336','20231012184423'] #'20230820203112']
#scroll4_ids = ['20231111135340', '20231122192640', '20231210132040', '20240304141530', '20231215151901']
enc_i,enc,fold=0,'i3d',0
#if True: #for fid in fragments:
fid = CFG.valid_id
for xoffset in 512, 1024: #4, 8: #32, 16, 8, 48, 56:
    CFG.valid_id=fid
    fragment_id = CFG.valid_id
    run_slug=f'training_scrolls_valid={fragment_id}_{CFG.size}x{CFG.size}_submissionlabels_wild11'

    pred_shape = {}
    for scroll_id in fragments: ##scroll4_ids:
      valid_mask_gt = cv2.imread(CFG.comp_dataset_path + f"train_scrolls/{scroll_id}/{scroll_id}_inklabels.png", 0)
      if valid_mask_gt is None:
        valid_mask_gt = cv2.imread(CFG.comp_dataset_path + f"train_scrolls/{scroll_id}/{scroll_id}_mask.png", 0)
      pred_shape[scroll_id]=tuple(int(c/CFG.scale) for c in valid_mask_gt.shape)
      #print("pred_shape", scroll_id, pred_shape[scroll_id])
    #print("Got all validation scroll ID shapes for prediction and logging")
    norm=fold==1
    wandb_logger = WandbLogger(project="vesivus",name=run_slug+f'{enc}_finetune')

    #print('FOLD : ',fold)
    multiplicative = lambda epoch: 0.9

    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="gpu",
        devices=torch.cuda.device_count(),
        logger=wandb_logger,
        default_root_dir="./", #/content/gdrive/MyDrive/vesuvius_model/training/outputs",
        accumulate_grad_batches=1,
        check_val_every_n_epoch=1,
        precision='bf16-mixed',
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        strategy='ddp_find_unused_parameters_true',
        callbacks=[ModelCheckpoint(filename=f'wild12_64_Scroll1_{fid}_{fold}_fr_{enc}_scale{CFG.scale}'+f'xoffset{xoffset}',dirpath=CFG.model_dir,monitor='train/total_loss',mode='min',save_top_k=CFG.epochs),],
    )

    #train_ids = set(['20230702185753','20230929220926','20231005123336','20231012184423','20231007101619','20231016151002','20231022170901','20231031143852','20231106155351','20231210121321','20231221180251','20230820203112']) - set([CFG.valid_id])
    #train_ids = set(['20231210132040', '20230702185753','20230929220926','20231005123336','20231012184423','20231007101619','20231016151002','20231022170901','20231031143852','20231106155351','20231210121321','20231221180251','20230820203112']) - set([CFG.valid_id])
    #train_ids = set(['20230702185753','20230929220926','20231005123336','20231012184423','20231007101619','20231016151002','20231022170901','20231031143852','20231106155351','20231210121321','20231221180251','20230820203112']) - set([CFG.valid_id])
    #train_ids = set(['20230702185753','20230929220926','20231005123336','20231012184423']) - set([CFG.valid_id])
    train_ids = set(['20231012184423'])
    #train_ids = set(['20230702185753']) #,'20230929220926','20231005123336','20231012184423','20231007101619','20231016151002','20231022170901','20231031143852','20231106155351','20231210121321','20231221180251','20230820203112']) - set([CFG.valid_id])
    #train_ids = set(['20231210132040']) #, '20230702185753','20230929220926','20231005123336','20231012184423','20231007101619','20231016151002','20231022170901','20231031143852','20231106155351','20231210121321','20231221180251','20230820203112']) - set([CFG.valid_id])
    valid_ids = set(fragments) #set([CFG.valid_id])

    train_images, train_masks, train_xyxys, train_ids, valid_images, valid_masks, valid_xyxys, valid_ids = get_train_valid_dataset(CFG, train_ids, valid_ids, scale=CFG.scale)
    #print(train_images['20231012184423'].shape)
    inkwidth=xoffset * 3 // 2
    vtrim = 128 #1*1024//CFG.scale//CFG.size
    totwidth = inkwidth + xoffset
    #cv2.imwrite(f'20231012184423_{CFG.scale}.image.png', train_images['20231012184423'].mean(axis=2))
    cv2.imwrite(f'20231012184423_{CFG.scale}.mask.png', train_masks['20231012184423'].mean(axis=2))
    train_images['20231012184423'] = train_images['20231012184423'][vtrim:-vtrim,:totwidth]
    #print("TRAIN images.shape", train_images['20231012184423'].shape)
    #print("TRAIN mask.shape", train_masks['20231012184423'].shape)
    train_masks['20231012184423'] = train_masks['20231012184423'][vtrim:-vtrim,:totwidth,:]
    train_masks['20231012184423'][:,:xoffset,:] = 0 #255
    train_masks['20231012184423'][:,xoffset:totwidth,:] = 255
    #cv2.imwrite(f'20231012184423_{CFG.scale}_{xoffset}.imagenew.png', train_images['20231012184423'].mean(axis=2))
    cv2.imwrite(f'20231012184423_{CFG.scale}_{xoffset}.masknew.png', train_masks['20231012184423'].mean(axis=2))
    #train_masks['20231012184423'] = train_masks['20231012184423'][1:-1,:8+16,...]
    #train_xyxys = [] #[CFG.size]
    r = CFG.size//2
    tm = train_masks['20231012184423']
    #train_xyxys = [(c[1]-r,c[0]-r,c[1]+r,c[0]+r) for c in np.argwhere(train_masks['20231012184423'] >= 0).tolist() if c[0]-r > 0 and c[1]-r > 0 and c[0]+r < tm.shape[0] and c[1]+r < tm.shape[1]]
    #train_xyxys = [(c[1]-r,c[0]-r,c[1]+r,c[0]+r) for c in np.argwhere(train_masks['20231012184423'] >= 0).tolist() if c[0]-r > 0 and c[1]-r > 0 and c[0]+r < tm.shape[0] and c[1]+r < tm.shape[1] and c[0] % CFG.stride == 0 and c[1] % CFG.stride == 0]
    train_stride = CFG.stride // 8
    train_xyxys = [(c[1]-r,c[0]-r,c[1]+r,c[0]+r) for c in np.argwhere(train_masks['20231012184423'] >= 0).tolist() if c[0]-r > 0 and c[1]-r > 0 and c[0]+r < tm.shape[0] and c[1]+r < tm.shape[1] and c[0] % train_stride == 0 and c[1] % train_stride == 0]
    if len(train_xyxys) < 10000:
      train_xyxys = train_xyxys * (10000 // len(train_xyxys))
    train_ids = ['20231012184423'] * len(train_xyxys) #train_ids * (100000 // len(train_ids))
    #print(len(train_images))
    valid_xyxys = np.stack(valid_xyxys)
    train_dataset = CustomDataset(
        train_images, CFG, labels=train_masks, xyxys=train_xyxys, ids=train_ids, transform=get_transforms(data='train', cfg=CFG), scale=CFG.scale)
    valid_dataset = CustomDataset(
        valid_images, CFG,xyxys=valid_xyxys, labels=valid_masks, ids=valid_ids, transform=get_transforms(data='valid', cfg=CFG), scale=CFG.scale)

    train_loader = DataLoader(train_dataset,
                                batch_size=CFG.train_batch_size,
                                shuffle=True,
                                num_workers=CFG.num_workers, pin_memory=True, drop_last=True,
                                )
    valid_loader = DataLoader(valid_dataset,
                                batch_size=CFG.valid_batch_size,
                                shuffle=False,
                                num_workers=CFG.num_workers, pin_memory=True, drop_last=True)

    model=RegressionPLModel(enc='i3d',pred_shape=pred_shape,size=CFG.size, train_dataset=train_dataset, backbone="pygonet", wandb_logger=wandb_logger, val_masks = valid_masks)

    wandb_logger.watch(model, log="all", log_freq=100)

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    wandb.finish()
