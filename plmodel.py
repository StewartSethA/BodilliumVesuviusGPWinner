import os,sys
os.environ["WANDB_MODE"] = "offline"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
from timesformer_pytorch import TimeSformer
import numpy as np
import cv2
from torch.optim import AdamW
import datetime
import segmentation_models_pytorch as smp
from warmup_scheduler import GradualWarmupScheduler
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = 933120000
from torch.utils.tensorboard import SummaryWriter
torch.set_float32_matmul_precision('medium')
#from config import CFG

from dataloaders import id2scroll

# from resnetall import generate_model
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m, mode='fan_out', nonlinearity='relu')
class Decoder(nn.Module):
    def __init__(self, encoder_dims, upscale, is_main=False):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(encoder_dims[i]+encoder_dims[i-1], encoder_dims[i-1], 3, 1, 1, bias=False),
                nn.BatchNorm2d(encoder_dims[i-1]),
                nn.ReLU(inplace=True)
            ) for i in range(1, len(encoder_dims))])

        if is_main:
          print("plmodel Decoder encoder_dims", encoder_dims)
        self.is_main = is_main
        self.logit = nn.Conv2d(encoder_dims[0], 1, 1, 1, 0)
        self.up = nn.Upsample(scale_factor=upscale, mode="bilinear")

    def forward(self, feature_maps):
        for i in range(len(feature_maps)-1, 0, -1):
            if self.is_main:
              print("i", i, feature_maps[i].shape, "scale factor 2")
            #f_up = F.interpolate(feature_maps[i], scale_factor=2, mode="bilinear") #...
            f_up = F.interpolate(feature_maps[i], feature_maps[i-1].shape[-2:], mode="bilinear") #...
            if self.is_main:
              print("layer", i, "f_up.shape", f_up.shape, "fmi-1.shape", feature_maps[i-1].shape, file=sys.stderr)
            f = torch.cat([feature_maps[i-1], f_up], dim=1)
            f_down = self.convs[i-1](f)
            feature_maps[i-1] = f_down

        mask = self.up(feature_maps[0])
        mask = self.logit(mask)
        #print("x", x.shape)
        if self.is_main:
          print("upped x", x.shape)
        #x = F.relu(x) #F.leaky_relu(x, 0.05) # SethS good nonlinearity? # SethS NO FINAL RELU!!!
        #print()
        return feature_maps[-1][:,0:1,...] #mask #mask

class RegressionPLModel(pl.LightningModule):
    #def __init__(self,pred_shape,size=256,enc='',with_norm=False,total_steps=500, train_dataset=None, check_val_every_n_epoch=1, backbone=None, wandb_logger=None, val_masks=None,name="", complexity=16):
    def __init__(self,pred_shape=None,size=256,enc='',with_norm=False,total_steps=500, train_dataset=None, check_val_every_n_epoch=1, backbone=None, wandb_logger=None, val_masks=None,name="", complexity=16, train_loader=None, valid_loader=None, cfg=None, is_main=False, train_masks=None, train_noinkmasks=None):
        super(RegressionPLModel, self).__init__()
        self.cfg = cfg
        self.is_main = is_main
        self.save_hyperparameters("size", "enc", "with_norm", "total_steps", "check_val_every_n_epoch", "name", "cfg")
        self.name = name
        #self.save_hyperparameters()
        self.writer = SummaryWriter("runs/"+name)
        #self.hparams.pred_shape = pred_shape
        self.val_masks = val_masks
        self.train_masks = train_masks
        self.train_noinkmasks = train_noinkmasks
        self.valid_loader = self.valid_dataloaders = valid_loader
        self.train_loader = self.train_dataloaders = train_loader
        self.wandb_logger = wandb_logger
        #if isinstance(self.hparams.pred_shape, dict):
        if True:
          self.mask_pred = {}
          self.mask_count = {}
          #for k,pred_shape in self.hparams.pred_shape.items():
          for k,pred_mask in self.valid_loader.dataset.labels.items():
            pred_shape = pred_mask.shape[:2]
            #print("Initializing mask pred and count", k, pred_shape)
            self.mask_pred[k] = np.zeros(pred_shape)
            self.mask_count[k] = np.zeros(pred_shape)
        #else:
        #  self.mask_pred = np.zeros(self.hparams.pred_shape)
        #  self.mask_count = np.zeros(self.hparams.pred_shape)
        self.loss_func1 = smp.losses.DiceLoss(mode='binary')
        self.loss_func2= smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25)
        self.loss_func= lambda x,y:0.5 * self.loss_func1(x,y)+0.5*self.loss_func2(x,y) #+ (1.0 * (x-y) ** 2).mean() # PYGO added MSE
        #self.loss_func= lambda x,y:0.5 * self.loss_func1(x,y)+0.5*self.loss_func2(x,y) + (1.0 * (x-y) ** 2).mean() # PYGO added MSE
        #self.loss_func= lambda x,y:(1.0 * (x-y) ** 2).mean() # PYGO added MSE
        self.model_name = backbone
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
        elif "pygo1x1" in backbone.lower():
            from pygo1x1 import InceptionI3d
            self.InceptionI3d = InceptionI3d
            self.backbone=InceptionI3d(in_channels=1,num_classes=128,non_local=False, complexity=complexity, is_main=is_main)
            #from pygo1x1 import Decoder
            #self.decoder = Decoder(encoder_dims=[x.size(1) for x in self.backbone(torch.rand(1,1,30,64,64))], upscale=4) # Do it right to compensate for downsampling!!
            self.decoder = Decoder(encoder_dims=[x.size(1) for x in self.backbone(torch.rand(1,1,30,16,16))], upscale=4) # Do it right to compensate for downsampling!!
        elif "pygonarrow" in backbone.lower():
            from pygonarrow import InceptionI3d
            self.InceptionI3d = InceptionI3d
            self.backbone=InceptionI3d(in_channels=1,num_classes=128,non_local=False, complexity=complexity)
            #from pygonarrow import Decoder
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
        if "1x1" in backbone.lower():
          #self.decoder = Decoder(encoder_dims=[x.size(1) for x in self.backbone(torch.rand(1,1,65,64,64))], upscale=4) # Do it right to compensate for downsampling!!
          #xs = self.backbone(torch.rand(1,1,30,64,64))
          xs = self.backbone(torch.rand(1,1,30,64,64))
          if self.is_main:
            print("Backbone outputs", [x.shape for x in xs])
          self.decoder = Decoder(encoder_dims=[x.size(1) for x in xs], upscale=4) # Do it right to compensate for downsampling!!
        else:
          self.decoder = Decoder(encoder_dims=[x.size(1) for x in self.backbone(torch.rand(1,1,30,4,4))], upscale=4) # Do it right to compensate for downsampling!!

        if self.hparams.with_norm:
            self.normalization=nn.BatchNorm3d(num_features=1)
        #self.decoder = Decoder(encoder_dims=[x.size(1) for x in self.backbone(torch.rand(1,1,20,256,256))], upscale=4) # Do it right to compensate for downsampling!!
        #self.decoder = Decoder(encoder_dims=[x.size(1) for x in self.backbone(torch.rand(1,1,30,4,4))], upscale=4) # Do it right to compensate for downsampling!!
        if self.hparams.with_norm:
            self.normalization=nn.BatchNorm3d(num_features=1)
        self.train_dataset = train_dataset
    def forward(self, x):
        if x.ndim==4:
            if self.is_main:
              print("x.ndim==4", x.shape)
            x=x[:,None]
        if self.hparams.with_norm:
            if self.is_main:
              print("self.normalization", x.shape)
            x=self.normalization(x) # WHAT DOES THIS DO??? BATCH NORM ON INFERENCE TOO?
        if not isinstance(self.backbone, self.InceptionI3d):
          x = self.backbone(torch.permute(x, (0, 2, 1,3,4)))
          if self.is_main:
            print("NonI3D Output", x.shape)
          x=x.view(-1,1,4,4) # Ooh, ouch!! WHY?? DROPPING INFO???
          return x
        else:
          feat_maps = self.backbone(x)
          if not isinstance(feat_maps, list) and feat_maps.shape[-2:] == x.shape[-2:]:
            #print("Skipping decoder since feat maps match input shape!", x.shape, feat_maps.shape)
            pred_mask = self.decoder.logit(feat_maps)
          else:
            if self.is_main:
              print("Feat maps", [(f.shape,f.min(),f.max()) for f in feat_maps])
            feat_maps_pooled = [torch.max(f, dim=2)[0] for f in feat_maps]
            if self.is_main:
              print("Feat maps pooled", [(f.shape,f.min(),f.max()) for f in feat_maps_pooled])
            pred_mask = self.decoder(feat_maps_pooled)
          #print("Pred_mask", pred_mask.shape, pred_mask.min(), pred_mask.max())
          return pred_mask
    def training_step(self, batch, batch_idx):
        x, y, xys, ids = batch
        #if batch_idx > 100:
        #  return {"loss": None} #torch.zeros_like(x)[0,0,0,0,0]}
        #print("training input.shape", x.shape, "y.shape", y.shape, "xys", xys, "len(xys)", len(xys)) # xys look wrong. batched incorrectly. list of 4 elements of 12 when batch is 12 and there should be 4 elements per xy.
        try:
          outputs = self(x)
        except Exception as ex:
          print("Exception: training input.shape", x.shape, "y.shape", y.shape, "xys", xys, file=sys.stderr)
          raise(ex)
        if batch_idx % 100 == 0 and self.trainer.is_global_zero:
          self.writer.add_image("pred_mask/train", torchvision.utils.make_grid(outputs, nrow=16, normalize=True, scale_each=True), self.current_epoch) # * len(self.train_dataloaders) + batch_idx)
          self.writer.add_image("true_mask/train", torchvision.utils.make_grid(y, nrow=16, normalize=True, scale_each=True), self.current_epoch) # * len(self.train_dataloaders) + batch_idx)
          self.writer.add_image("inputsdepthslice/train", torchvision.utils.make_grid(x.mean(dim=3), nrow=16, normalize=True, scale_each=True), self.current_epoch) # * len(self.train_dataloaders) + batch_idx)
          self.writer.add_image("inputsdepthslice2/train", torchvision.utils.make_grid(x.mean(dim=4), nrow=16, normalize=True, scale_each=True), self.current_epoch) # * len(self.train_dataloaders) + batch_idx)
          self.writer.add_image("inputs2d/train", torchvision.utils.make_grid(x.mean(dim=2), nrow=16, normalize=True, scale_each=True), self.current_epoch) # * len(self.train_dataloaders) + batch_idx)
          self.writer.add_image("falsepositive/train", torchvision.utils.make_grid(torch.relu(outputs-y), nrow=16, normalize=True, scale_each=True), self.current_epoch) # * len(self.train_dataloaders) + batch_idx)
          self.writer.add_image("falsenegative/train", torchvision.utils.make_grid(torch.relu(y-outputs), nrow=16, normalize=True, scale_each=True), self.current_epoch) # * len(self.train_dataloaders) + batch_idx)
        #outputs = torch.relu(outputs) # SethS eliminate ReLU so we can make darker outputs!
        #if x2-x1 != y2-y1:
        #      print("Skipping misshapen xyxys!",x1,y1,x2,y2)
        #      continue
        #x2,y2 = x1+self.cfg.size, y1+self.cfg.size
        if batch_idx % 500 == 0 and self.trainer.is_global_zero:
          print("x.shape", x.shape, "y.shape", y.shape, "outputs.shape", outputs.shape, "y.stats", y.min(), y.max(), y.mean(), y.std(), "out.stats", outputs.min(), outputs.max(), outputs.mean(), outputs.std(), "xys", len(xys), xys[:3])

        if y.shape != outputs.shape:
          print("Outputs not same shape", outputs.shape, y.shape)
          #y=F.interpolate(y,(4,4)) # TODO SethS: DISABLE ME!
          outputs=F.interpolate(outputs,y.shape[-2:], mode="bilinear") # TODO SethS: DISABLE ME!
          #y=F.interpolate(y,outputs.shape[-2:], mode="area") # TODO SethS: DISABLE ME!
        #y = F.interpolate(y, (1,1), mode="area")
        #outputs = F.interpolate(outputs, (1,1), mode="area")
        mseloss = ((outputs - y) ** 2).mean()
        loss1 = self.loss_func(outputs, y) + mseloss
        #if batch_idx % 250 == 0:
        #  print("outputs min, max", outputs.min().item(), outputs.max().item(), "y min, max", y.min().item(), y.max().item(), "loss:", loss1.item())
        if torch.isnan(loss1):
            print("Loss nan encountered")
        self.log("train/total_loss", loss1.item(),on_step=True, on_epoch=True, prog_bar=True)
        if self.trainer.is_global_zero:
          self.writer.add_scalar('Loss/train', loss1.item(), self.current_epoch)
          self.writer.add_scalar('output/min', outputs.min().item(), self.current_epoch)
          self.writer.add_scalar('output/max', outputs.max().item(), self.current_epoch)
          diceloss = self.loss_func1(outputs, y)
          bceloss = self.loss_func2(outputs, y)
          loss1 = self.loss_func(outputs, y)
          madloss = torch.abs(outputs - y).mean()
          self.writer.add_scalar("loss_mse/train", mseloss, self.current_epoch) # * len(self.train_dataloaders) + batch_idx)
          self.writer.add_scalar("loss_mad/train", madloss, self.current_epoch) # * len(self.train_dataloaders) + batch_idx)
          self.writer.add_scalar("loss_bce/train", bceloss, self.current_epoch) # * len(self.train_dataloaders) + batch_idx)
          self.writer.add_scalar("loss_dice/train", diceloss, self.current_epoch) # * len(self.train_dataloaders) + batch_idx)
          self.writer.add_scalar("loss/train", loss1, self.current_epoch) # * len(self.train_dataloaders) + batch_idx)

        return {"loss": loss1}
    def validation_step(self, batch, batch_idx):
        x,y,xyxys,ids= batch
        #print("validation step", batch_idx, "x", x.shape, "y", y.shape, "xyxys[0]", xyxys[0], "len", len(xyxys), "ids", ids[:5])
        valid_includes = list(reversed(sorted(list(self.valid_loader.dataset.labels.keys())))) #[:self.current_epoch*5]
        if len(valid_includes) == 0:
          #print(batch_idx, "Validation step: Nothing in valid includes!", valid_includes)
          return
        #print(xyxys)
        #exit()
        #print("Included images on validation step:", valid_includes)
        if not np.array([id in valid_includes for id in ids]).any(): # WAS .all() 7/30/2024 # Ah, this is the way to get it to skip some images.
          #print("No IDs from batch in np array!:", ids, xyxys)
          return
        #print("IN NP ARRAY valid_includes:", ids, xyxys, "valid includes", valid_includes)
        batch_size = x.size(0)
        outputs = self(x)
        if batch_idx % 5000 == 0 and self.trainer.is_global_zero:
          self.writer.add_image("pred_mask/valid", torchvision.utils.make_grid(outputs, nrow=16, normalize=True, scale_each=True), self.current_epoch) #, self.current_epoch * len(self.train_dataloaders) + batch_idx)
          self.writer.add_image("true_mask/valid", torchvision.utils.make_grid(y, nrow=16, normalize=True, scale_each=True), self.current_epoch) # * len(self.train_dataloaders) + batch_idx)
          self.writer.add_image("inputsdepthslice/valid", torchvision.utils.make_grid(x.mean(dim=3), nrow=16, normalize=True, scale_each=True), self.current_epoch) # * len(self.train_dataloaders) + batch_idx)
          self.writer.add_image("inputsdepthslice2/valid", torchvision.utils.make_grid(x.mean(dim=4), nrow=16, normalize=True, scale_each=True), self.current_epoch) # * len(self.train_dataloaders) + batch_idx)
          self.writer.add_image("inputs2d/valid", torchvision.utils.make_grid(x.mean(dim=2), nrow=16, normalize=True, scale_each=True), self.current_epoch) # * len(self.train_dataloaders) + batch_idx)
          self.writer.add_image("falsepositive/valid", torchvision.utils.make_grid(torch.relu(outputs-y), nrow=16, normalize=True, scale_each=True), self.current_epoch) # * len(self.train_dataloaders) + batch_idx)
          self.writer.add_image("falsenegative/valid", torchvision.utils.make_grid(torch.relu(y-outputs), nrow=16, normalize=True, scale_each=True), self.current_epoch) # * len(self.train_dataloaders) + batch_idx)
          print("outputs.shape", outputs.shape)
          print("min,max preds", outputs.min().item(), outputs.max().item())
        if y.shape != outputs.shape:
          #y=F.interpolate(y,(4,4)) # TODO SethS: DISABLE ME!
          print("Outputs not same shape", outputs.shape, y.shape)
          outputs=F.interpolate(outputs,y.shape[-2:], mode="bilinear") # Auto-adjust output shape.
          #y=F.interpolate(y,outputs.shape[-2:], mode="area") # Auto-adjust output shape.
        #y = F.interpolate(y, (1,1), mode="area")
        #outputs = F.interpolate(outputs, (1,1), mode="area") # Reduction to a single point. Probably isn't helping.

        loss1 = self.loss_func(outputs, y) + ((outputs-y) ** 2).mean()
        y_preds = torch.sigmoid(outputs).to('cpu')
        #y_preds = #torch.clip(outputs.to('cpu'),0,1) #torch.sigmoid(outputs).to('cpu')
        #if batch_idx % 100 == 0:
        #  print("post-sigmoid min,max preds", y_preds.min().item(), y_preds.max().item())
        #y_preds = torch.minimum(1, torch.relu(outputs).to('cpu'))
        #y_preds = torch.relu(outputs).to('cpu')
        #print(batch_idx, "batch valid_xyxys", xyxys)
        for i, (x1, y1, x2, y2) in enumerate(xyxys):
            x1,x2,y1,y2 = (int(c) for c in (x1,x2,y1,y2))
            x2,y2 = x1+self.cfg.size, y1+self.cfg.size
            if batch_idx % 5000 == 0 and i == 0 and self.trainer.is_global_zero:
              print("validation CFG.size", self.cfg.size, "x1y1,x2y2:", x1,y1,x2,y2, "x.shape", x.shape, "y.shape", y.shape)
            origxs = (x1,x2,y1,y2)
            #print(x1,x2,y1,y2, "xyxys,", ids[i], "mask_pred.shape", self.mask_pred[ids[i]].shape)
            x1,x2,y1,y2 = x1,min(x2,self.mask_pred[ids[i]].shape[1]),y1,min(y2,self.mask_pred[ids[i]].shape[0]) # SethS Why would this be necessary? It isn't... Unless the validation masks and pred_shapes are different.
            #origxs = tuple(origxs)
            if (x1,x2,y1,y2) != origxs:
              print((x1,x2,y1,y2), "orig xys:", origxs)
            if x2-x1 <= 0 or y2-y1 <= 0:
              print("Skipping OOB xyxys!",x1,y1,x2,y2)
              continue
            if x2-x1 != y2-y1:
              print("Skipping misshapen xyxys!",x1,y1,x2,y2)
              continue
            #print(x1, x2, y1, y2, "id", ids[i], [(k,v.shape) for k,v in self.mask_pred.items()], [(k,v.shape) for k,v in self.mask_count.items()])
            #self.mask_pred[ids[i]][y1:y2, x1:x2] += F.interpolate(y_preds[i].unsqueeze(0).float(),scale_factor=4,mode='bilinear').squeeze(0).squeeze(0).numpy()
            #self.mask_pred[ids[i]][y1:y2, x1:x2] += F.interpolate(y_preds[i].unsqueeze(0).float(),(y2-y1,x2-x1),mode='bilinear').squeeze(0).squeeze(0).numpy()
            if True or y2-y1 > 1 or x2-x1 > 1: # TODO: Always upsampling predictions???
              #self.mask_pred[ids[i]][y1:y2, x1:x2] += np.clip(F.interpolate(y_preds[i].unsqueeze(0).float(),(y2-y1,x2-x1),mode='bilinear').squeeze(0).squeeze(0).numpy(), 0, 1)
              #self.mask_pred[ids[i]][y1:y2, x1:x2] += F.interpolate(y_preds[i].unsqueeze(0).float(),(y2-y1,x2-x1),mode='bilinear').squeeze(0).squeeze(0).numpy()
              #self.mask_pred[ids[i]][y1:y2, x1:x2] += F.interpolate(y_preds[i].unsqueeze(0).float(),(y2-y1,x2-x1),mode='area').squeeze(0).squeeze(0).numpy() # It has to be squashed DOWN, not upsampled. Same sampling algorithm problems I used to run into!!!
              #self.mask_pred[ids[i]][y1:y2, x1:x2] += F.interpolate(y_preds[i].unsqueeze(0).float(),(y2-y1,x2-x1),mode='bilinear').squeeze(0).squeeze(0).numpy() # It has to be squashed DOWN, not upsampled. Same sampling algorithm problems I used to run into!!! # USE bilinear for downsampling!!!
              # TODO SethS 7/30 commented the above!
              self.mask_pred[ids[i]][y1:y2, x1:x2] += y_preds[i].unsqueeze(0).float().squeeze(0).squeeze(0).numpy() # It has to be squashed DOWN, not upsampled. Same sampling algorithm problems I used to run into!!! # USE bilinear for downsampling!!!
            else:
              #self.mask_pred[ids[i]][y1:y2, x1:x2] += np.clip(y_preds[i].unsqueeze(0).float().squeeze(0).squeeze(0).numpy(), 0, 1) # TODO: What if I don't apply np.clip while summing, but only AFTER???
              self.mask_pred[ids[i]][y1:y2, x1:x2] += y_preds[i].unsqueeze(0).float().squeeze(0).squeeze(0).numpy() # TODO: What if I don't apply np.clip while summing, but only AFTER???
            self.mask_count[ids[i]][y1:y2, x1:x2] += np.ones((y2-y1, x2-x1))
            #print("Keeping xyxys", x1,y1,x2,y2, xyxys[i], ids[i])
        diceloss = self.loss_func1(outputs, y)
        bceloss = self.loss_func2(outputs, y)
        #lossv = self.loss_func(outputs, y)
        mseloss = ((outputs - y) ** 2).mean()
        madloss = torch.abs(outputs - y).mean()

        if self.trainer.is_global_zero:
          self.log("val/total_loss", loss1.item(),on_step=True, on_epoch=True, prog_bar=True)
          self.writer.add_scalar('Loss/valid', loss1.item(), self.current_epoch)
          self.writer.add_scalar("loss_mse/valid", mseloss, self.current_epoch) # * len(self.valid_dataloaders) + batch_idx)
          self.writer.add_scalar("loss_mad/valid", madloss, self.current_epoch) # * len(self.valid_dataloaders) + batch_idx)
          self.writer.add_scalar("loss_bce/valid", bceloss, self.current_epoch) # * len(self.valid_dataloaders) + batch_idx)
          self.writer.add_scalar("loss_dice/valid", diceloss, self.current_epoch) # * len(self.valid_dataloaders) + batch_idx)
          self.writer.add_scalar("loss/valid", loss1, self.current_epoch) # * len(self.valid_dataloaders) + batch_idx)
        return {"loss": loss1}
    def on_validation_epoch_end(self):
        if self.trainer.is_global_zero:
          valid_includes = list(reversed(sorted(list(self.valid_loader.dataset.labels.keys())))) #[:self.current_epoch*5]
          print("Experiment name", self.name, "Included images on validation epoch end:", valid_includes)
          if len(valid_includes) == 0:
            return
          #if isinstance(self.hparams.pred_shape, dict):
          # TODO: Should I call .gather here?
          #print("mask_pred has keys", self.mask_pred.keys())
          #for k,pred_shape in self.hparams.pred_shape.items():
          for k,pred_mask in self.valid_loader.dataset.labels.items():
            sid = id2scroll[k]
            pred_shape = pred_mask.shape[:2]
            self.mask_pred[k] = np.divide(self.mask_pred[k], self.mask_count[k], out=np.zeros_like(self.mask_pred[k]), where=self.mask_count[k]!=0)
            if self.mask_pred[k].std() == 0 or self.mask_count[k].sum() == 0:
              if k in valid_includes:
                print("Warning: included validation image has no nonzero predictions!", k, self.mask_pred[k].shape)
              continue
            #print("Writing image on validation epoch end", k, self.mask_count[k].sum(), "pred mean, std", self.mask_pred[k].mean(), self.mask_pred[k].std())
            if self.mask_pred[k] is None or np.product(self.mask_pred[k].shape) == 0:
              print("No mask pred for key", k, self.mask_pred[k])
            else:
              cv2.imwrite(self.model_name+"_"+self.name+"_"+sid+"_"+k+"_scale"+str(self.cfg.scale)+"_size"+str(self.cfg.size)+"_tile_size"+str(self.cfg.tile_size)+"_stride"+str(self.cfg.stride)+"_valstride"+str(self.cfg.valid_stride)+"_batch"+str(self.cfg.train_batch_size)+"_vbs"+str(self.cfg.valid_batch_size)+"_epoch"+str(self.current_epoch)+".jpg", np.clip(self.mask_pred[k],0,1)*255) 
              self.mask_pred[k][self.mask_pred[k]==0] = self.mask_pred[k].sum()/np.count_nonzero(self.mask_pred[k])
              self.wandb_logger.log_image(key=f"preds_{sid}_{k}", images=[np.clip(self.mask_pred[k],0,1)], caption=["probs"])
              self.writer.add_image(f'{sid}_{k}_preds', (self.mask_pred[k] - self.mask_pred[k].min()) / max(self.mask_pred[k].max()-self.mask_pred[k].min(), 0.01), self.current_epoch, dataformats="HW")
              self.writer.add_image(f'{sid}_{k}_predcount', self.mask_count[k]/max(1,self.mask_count[k].max()), self.current_epoch, dataformats="HW")
              if self.val_masks is not None and k in self.val_masks:
                self.wandb_logger.log_image(key=f"trues_{sid}_{k}", images=[np.clip(self.val_masks[k],0,255)], caption=["probs"])
                self.writer.add_image(f'{sid}_{k}_trues', np.clip(self.val_masks[k][:,:,0],0,255), self.current_epoch, dataformats="HW") # TODO check normalization range here?
                if self.train_masks is not None and k in self.train_masks:
                  self.wandb_logger.log_image(key=f"traintrues_{sid}_{k}", images=[np.clip(self.train_masks[k],0,255)], caption=["probs"])
                  self.writer.add_image(f'{sid}_{k}_traintrues', np.clip(self.train_masks[k][:,:,0],0,255), self.current_epoch, dataformats="HW") # TODO check normalization range here?
                if self.train_noinkmasks is not None and k in self.train_noinkmasks:
                  self.wandb_logger.log_image(key=f"trainnoink_{sid}_{k}", images=[np.clip(self.train_noinkmasks[k],0,255)], caption=["probs"])
                  self.writer.add_image(f'{sid}_{k}_trainnoink', np.clip(self.train_noinkmasks[k][:,:,0],0,255), self.current_epoch, dataformats="HW") # TODO check normalization range here?
              else:
                print("val_masks missing", k, self.val_masks.keys())
        '''
        else:
          self.mask_pred = np.divide(self.mask_pred, self.mask_count, out=np.zeros_like(self.mask_pred), where=self.mask_count!=0)
          self.wandb_logger.log_image(key="preds", images=[np.clip(self.mask_pred,0,1)], caption=["probs"])
          self.writer.add_image(f'preds', self.mask_pred, self.current_epoch, dataformats="HW")
          if self.val_masks is not None:
            self.wandb_logger.log_image(key=f"trues_{k}", images=[np.clip(self.val_masks[:,:,0],0,255)], caption=["probs"])
        '''
        #print("Done logging! Resetting mask...")
        #reset mask
        self.mask_pred = {}
        self.mask_count = {}
        for k,pred_mask in self.valid_loader.dataset.labels.items():
          pred_shape = pred_mask.shape[:2]
          self.mask_pred[k] = np.zeros(pred_shape)
          self.mask_count[k] = np.zeros(pred_shape)
        '''
        if isinstance(self.hparams.pred_shape, dict):
          self.mask_pred = {}
          self.mask_count = {}
          for k,pred_shape in self.hparams.pred_shape.items():
            self.mask_pred[k] = np.zeros(pred_shape)
            self.mask_count[k] = np.zeros(pred_shape)
        else:
          self.mask_pred = np.zeros(self.hparams.pred_shape)
          self.mask_count = np.zeros(self.hparams.pred_shape)
        '''
        #print("Done resetting masks!")
        #self.train_dataset.labels = reload_masks(self.train_dataset.labels, CFG)
        #exit()
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.cfg.lr)
        scheduler = get_scheduler(self.cfg, optimizer)
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
                #return self.after_scheduler.get_lr()
                return self.after_scheduler.get_last_lr()
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
