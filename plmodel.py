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
from timesformer_pytorch import TimeSformer
import gc
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
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = 933120000

from torch.utils.tensorboard import SummaryWriter

torch.set_float32_matmul_precision('medium')

from config import CFG

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
            #print("i", i, feature_maps[i].shape, "scale factor 2")
            f_up = F.interpolate(feature_maps[i], scale_factor=2, mode="bilinear") #...
            f = torch.cat([feature_maps[i-1], f_up], dim=1)
            f_down = self.convs[i-1](f)
            feature_maps[i-1] = f_down

        x = self.logit(feature_maps[0])
        #print("x", x.shape)
        #mask = self.up(x)
        #print("upped x", x.shape)
        x = F.relu(x) #F.leaky_relu(x, 0.05) # SethS good nonlinearity?
        #print()
        return x #mask

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
        self.loss_func1 = smp.losses.DiceLoss(mode='binary')
        self.loss_func2= smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25)
        self.loss_func= lambda x,y:0.5 * self.loss_func1(x,y)+0.5*self.loss_func2(x,y) #+ (1.0 * (x-y) ** 2).mean() # PYGO added MSE
        #self.loss_func= lambda x,y:0.5 * self.loss_func1(x,y)+0.5*self.loss_func2(x,y) + (1.0 * (x-y) ** 2).mean() # PYGO added MSE
        #self.loss_func= lambda x,y:(1.0 * (x-y) ** 2).mean() # PYGO added MSE
        if backbone == "timesformer":
            self.backbone=TimeSformer(
                dim = 512,
                image_size = 64,
                patch_size = 16,
                num_frames = 26,
                num_classes = 16,
                channels=1,
                depth = 8,
                heads = 6,
                dim_head =  64,
                attn_dropout = 0.1,
                ff_dropout = 0.1
            )
        elif "pygoflat" in backbone.lower():
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
        else:
            self.backbone = backbone
        if self.hparams.with_norm:
            self.normalization=nn.BatchNorm3d(num_features=1)
        self.decoder = Decoder(encoder_dims=[x.size(1) for x in self.backbone(torch.rand(1,1,20,256,256))], upscale=1)
        if self.hparams.with_norm:
            self.normalization=nn.BatchNorm3d(num_features=1)
        self.train_dataset = train_dataset
    def forward(self, x):
        if x.ndim==4:
            x=x[:,None]
        if self.hparams.with_norm:
            x=self.normalization(x)
        if not isinstance(self.backbone, self.InceptionI3d):
          x = self.backbone(torch.permute(x, (0, 2, 1,3,4)))
          x=x.view(-1,1,4,4)
          return x
        else:
          feat_maps = self.backbone(x)
          feat_maps_pooled = [torch.max(f, dim=2)[0] for f in feat_maps]
          pred_mask = self.decoder(feat_maps_pooled)
          return pred_mask
    def training_step(self, batch, batch_idx):
        x, y, xys, ids = batch
        outputs = self(x)
        outputs = torch.relu(outputs)
        if y.shape != outputs.shape:
          #y=F.interpolate(y,(4,4)) # TODO SethS: DISABLE ME!
          #y=F.interpolate(y,outputs.shape[-2:], mode="bilinear") # TODO SethS: DISABLE ME!
          y=F.interpolate(y,outputs.shape[-2:], mode="area") # TODO SethS: DISABLE ME!
        #y = F.interpolate(y, (1,1), mode="area")
        #outputs = F.interpolate(outputs, (1,1), mode="area")
        #print("y.shape", y.shape, "outputs.shape", outputs.shape)
        loss1 = self.loss_func(outputs, y)
        #if batch_idx % 250 == 0:
        #  print("outputs min, max", outputs.min().item(), outputs.max().item(), "y min, max", y.min().item(), y.max().item(), "loss:", loss1.item())
        if torch.isnan(loss1):
            print("Loss nan encountered")
        self.log("train/total_loss", loss1.item(),on_step=True, on_epoch=True, prog_bar=True)
        self.writer.add_scalar('Loss/train', loss1.item(), self.current_epoch)
        self.writer.add_scalar('output/min', outputs.min().item(), self.current_epoch)
        self.writer.add_scalar('output/max', outputs.max().item(), self.current_epoch)
        diceloss = self.loss_func1(outputs, y)
        bceloss = self.loss_func2(outputs, y)
        loss1 = self.loss_func(outputs, y)
        mseloss = ((outputs - y) ** 2).mean()
        madloss = torch.abs(outputs - y).mean()
        self.writer.add_scalar("loss_mse/train", mseloss, self.current_epoch * len(self.train_dataloaders) + batch_idx)
        self.writer.add_scalar("loss_mad/train", madloss, self.current_epoch * len(self.train_dataloaders) + batch_idx)
        self.writer.add_scalar("loss_bce/train", bceloss, self.current_epoch * len(self.train_dataloaders) + batch_idx)
        self.writer.add_scalar("loss_dice/train", diceloss, self.current_epoch * len(self.train_dataloaders) + batch_idx)
        self.writer.add_scalar("loss/train", loss1, self.current_epoch * len(self.train_dataloaders) + batch_idx)

        return {"loss": loss1}
    def validation_step(self, batch, batch_idx):
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
              #self.mask_pred[ids[i]][y1:y2, x1:x2] += F.interpolate(y_preds[i].unsqueeze(0).float(),(y2-y1,x2-x1),mode='area').squeeze(0).squeeze(0).numpy() # It has to be squashed DOWN, not upsampled. Same sampling algorithm problems I used to run into!!!
              self.mask_pred[ids[i]][y1:y2, x1:x2] += F.interpolate(y_preds[i].unsqueeze(0).float(),(y2-y1,x2-x1),mode='bilinear').squeeze(0).squeeze(0).numpy() # It has to be squashed DOWN, not upsampled. Same sampling algorithm problems I used to run into!!! # USE bilinear for downsampling!!!
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


        return {"loss": loss1}
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
        if not isinstance(self.backbone, self.InceptionI3d):
          scheduler =torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-4,pct_start=0.15, steps_per_epoch=self.hparams.total_steps, epochs=150,final_div_factor=1e2)
        return [optimizer],[scheduler]

from dataloaders import *

class GradualWarmupSchedulerV2(GradualWarmupScheduler):
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
        optimizer, 30, eta_min=1e-6)
    scheduler = GradualWarmupSchedulerV2(
        optimizer, multiplier=1.0, total_epoch=1, after_scheduler=scheduler_cosine)

    return scheduler

def scheduler_step(scheduler, avg_val_loss, epoch):
    scheduler.step(epoch)
