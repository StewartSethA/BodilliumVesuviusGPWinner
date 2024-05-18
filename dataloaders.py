from PIL import Image
Image.MAX_IMAGE_PIXELS = 933120000
import os.path as osp
import os
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
import zarr

from skimage.measure import block_reduce

#def ZarrArrayWrapper(zarr.core.Array):
#  def __getitem__(self, slice):
#    return super().__getitem((slice[2], slice[0], slice[1]))
class ZarrArrayWrapper:
  def __init__(self, array):
    self.array = array
    #self.shape = lambda: array.shape[1:] + array.shape[0]
  def __getitem__(self, slice):
    print("getting from zarr array wrapper", self.shape, "slice:", slice)
    return self.array[(slice[2], slice[0], slice[1])].transpose(1,2,0)
  @property
  def shape(self):
    return self.array.shape[1:] + self.array.shape[0:1]

#def ZarrArrayWrapper(zarr.core.Array):
#def __getitem__(a, slice):
#  return a[(slice[2], slice[0], slice[1])]

def read_image_mask(fragment_id,start_idx=15,end_idx=45,CFG=None, fragment_mask_only=False, pad0=0, pad1=0, scale=1, chunksize=128, force_mem=False):
  basepath = CFG.basepath
  scrollsdir = "train_scrolls" if os.path.isdir("train_scrolls") else "train_scrolls2"
  #if fragment_id in ['20231210132040']:
  #  scale = 1
  images = None
  fragment_mask = None
  mask = None
  #if scale != 1:
  #  start_idx = 0
  #  end_idx = 65
  if not fragment_mask_only:
    start_idx = 0
    idxs = range(start_idx, end_idx)
    print("idxs", idxs, start_idx, end_idx, fragment_id)
    images = []
    t = time.time()
    loaded = False
    rescaled = False
    print("Checking for", f"{basepath}/{fragment_id}_{chunksize}.zarr", "at scale", scale) # TODO: Bake in the scaling!
    mra = None
    if os.path.exists(f"{basepath}/{fragment_id}_{chunksize}_8level.zarr") and not force_mem and start_idx == 15 and end_idx == 45:
      print("Reading", f"{basepath}/{fragment_id}_{chunksize}_8level.zarr")
      mra = zarr.open(f"{basepath}/{fragment_id}_{chunksize}_8level.zarr")
    elif os.path.exists(f"{basepath}/{fragment_id}_{chunksize}_4levels.zarr") and not force_mem and start_idx == 15 and end_idx == 45:
      print("Reading", f"{basepath}/{fragment_id}_{chunksize}_4levels.zarr")
      mra = zarr.open(f"{basepath}/{fragment_id}_{chunksize}_4levels.zarr")
    elif os.path.exists(f"{basepath}/{fragment_id}_{chunksize}_0-30_4levels.zarr") and not force_mem and start_idx == 15 and end_idx == 45:
      print("Reading", f"{basepath}/{fragment_id}_{chunksize}_0-30_4levels.zarr")
      mra = zarr.open(f"{basepath}/{fragment_id}_{chunksize}_0-30_4levels.zarr")
    elif os.path.exists(f"{basepath}/{fragment_id}_{chunksize}_0-30.zarr") and not force_mem and start_idx == 15 and end_idx == 45:
      print("Reading", f"{basepath}/{fragment_id}_{chunksize}_0-30.zarr")
      mra = zarr.open(f"{basepath}/{fragment_id}_{chunksize}_0-30.zarr")
    elif os.path.exists(f"{basepath}/{fragment_id}_{chunksize}.zarr") and not force_mem:
      print("Reading", f"{basepath}/{fragment_id}_{chunksize}.zarr")
      mra = zarr.open(f"{basepath}/{fragment_id}_{chunksize}.zarr")
    if mra is not None:
      try:
        images = mra[0]
        if scale == 2:
          images = mra[1]
        elif scale == 4:
          images = mra[2]
        else:
          images = []
          rescaled = loaded = False
          a = 1/0 # SethS force punt on more downsampled arrays since these will more easily fit in memory.
        if scale == 8:
          images = mra[3]
        elif scale == 16:
          images = mra[4]
        elif scale == 32:
          images = mra[5]
        elif scale == 64:
          images = mra[6]
        elif scale == 128:
          images = mra[7]
        elif scale == 256:
          images = mra[8]
        elif scale == 512:
          images = mra[9]
        elif scale == 1024:
          images = mra[9]
        else:
          rescaled = loaded = False
        if images is not None:
          images = ZarrArrayWrapper(images)
          if images.shape[-1] < end_idx-start_idx:
            print("Read too little image stack data; skipping ZARR and looking for Numpy or Tiff!", images.shape, start_idx, end_idx, fragment_id)
            images = []
          else:
            rescaled = loaded = True
      except Exception as ex:
        rescaled = loaded = False
        print(ex, fragment_id)
    if loaded:
      print("Loaded shape", images.shape)
    elif os.path.isfile(f"{basepath}/{fragment_id}_{start_idx}-{end_idx}_{scale}.npy"):
      print("Reading", f"{basepath}/{fragment_id}_{start_idx}-{end_idx}_{scale}.npy")
      images = np.load(f"{basepath}/{fragment_id}_{start_idx}-{end_idx}_{scale}.npy") # Other parts too
      rescaled = loaded = True
    elif os.path.isfile(f"{basepath}/{fragment_id}_{start_idx}-{end_idx}.npy"):
      print("Reading", f"{basepath}/{fragment_id}_{start_idx}-{end_idx}.npy")
      images = np.load(f"{basepath}/{fragment_id}_{start_idx}-{end_idx}.npy", images) # Seths
      loaded = True
    '''
    elif scale != 1 and os.path.isfile(f"{basepath}/{fragment_id}_{scale}.npy"):
      print("Reading", f"{basepath}/{fragment_id}_{scale}.npy")
      np.load(f"{basepath}/{fragment_id}_{scale}.npy", images) # Other parts too
      rescaled = loaded = True
    '''
    '''
    elif os.path.isfile(f"{basepath}/{fragment_id}_{start_idx}-{end_idx}.npy"):
      print("Reading", f"{basepath}/{fragment_id}_{start_idx}-{end_idx}.npy", start_idx, end_idx)
      images = np.load(f"{basepath}/{fragment_id}_{start_idx}-{end_idx}.npy")
      if images.shape[-1] >= end_idx-start_idx:
        loaded = True
    elif os.path.isfile(f"{basepath}/{fragment_id}.npy"):
      print("Reading", f"{basepath}/{fragment_id}.npy", start_idx, end_idx)
      images = np.load(f"{basepath}/{fragment_id}.npy")
      if images.shape[-1] >= end_idx-start_idx:
        loaded = True
      else:
        print("Loaded shape rejected due to mismatch:", images.shape)
    elif  os.path.isfile(f"{basepath}/{fragment_id}/layers/{fragment_id}.npy"):
      print("READING", f"{basepath}/{fragment_id}/layers/{fragment_id}.npy", start_idx, end_idx)
      images = np.load(f"{basepath}/{fragment_id}/layers/{fragment_id}.npy")
      if images.shape[-1] >= end_idx-start_idx:
        loaded = True
    '''
    if loaded:
      print(time.time()-t, "seconds taken to load images from", f"{basepath}/{fragment_id}/layers/{fragment_id}"+" Something "+".npy", "Shape:", print(images.shape))
      pad0 = (CFG.tile_size - images.shape[0] % CFG.tile_size)
      pad1 = (CFG.tile_size - images.shape[1] % CFG.tile_size)
    elif not loaded:
      images = []
      if os.path.isfile(f"{basepath}/{fragment_id}/layers/015.tif") or os.path.isfile(f"{basepath}/{fragment_id}/layers/015.jpg"):
        idxs = range(0, 156)
      for idx in idxs:
        for ext in ['jpg', 'tif']:
          image = None
          if os.path.isfile(f"{basepath}/{fragment_id}/layers/{idx:02}.{ext}"):
            print("Loading", f"{basepath}/{fragment_id}/layers/{idx:02}.{ext}")
            image = cv2.imread(f"{basepath}/{fragment_id}/layers/{idx:02}.{ext}", 0)
          if image is not None:
            break
        for ext in ['jpg', 'tif']:
          if image is None and os.path.isfile(f"{basepath}/{fragment_id}/layers/{idx:03}.{ext}"): # TODO SethS: We need to accommodate a deeper stack of images here...
            print("Loading", f"{basepath}/{fragment_id}/layers/{idx:03}.{ext}")
            image = cv2.imread(f"{basepath}/{fragment_id}/layers/{idx:03}.{ext}", 0)
          if image is not None:
            break
        if image is None and "jpg" != ext:
          print("WARNING: FAILED TO LOAD!", f"{basepath}/{fragment_id}/layers/{idx:03}.{ext}")
          break
          #return None, None, None, None, None
        pad0 = (CFG.tile_size - image.shape[0] % CFG.tile_size)
        pad1 = (CFG.tile_size - image.shape[1] % CFG.tile_size)
        print("Un-padded image size", image.shape)
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
      if len(images) == 0:
        return None, None, None, None, None
      images = np.stack(images, axis=2)
      t = time.time()
      print(time.time()-t, "seconds taken to stack images.")
      t = time.time()
      if not os.path.isfile(f"{basepath}/{fragment_id}_{start_idx}-{end_idx}.npy") and not os.path.isfile(f"{basepath}/{fragment_id}_{start_idx}-{end_idx}.npy"):
        print("saving", f"{basepath}/{fragment_id}_{start_idx}-{end_idx}.npy", images.shape) # Seths
        np.save(f"{basepath}/{fragment_id}_{start_idx}-{end_idx}.npy", images) # Seths
        print(time.time()-t, "seconds taken to save images as npy.")
    if fragment_id in ['20230701020044','verso','20230901184804','20230901234823','20230531193658','20231007101615','20231005123333','20231011144857','20230522215721', '20230919113918', '20230625171244','20231022170900','20231012173610','20231016151000']:
        print("reversing", fragment_id)
        images=images[:,:,::-1]

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
      np.save(f"{basepath}/{fragment_id}_{start_idx}-{end_idx}_{scale}.npy", images) # Other parts too
      print("Saved rescaled array.")
      pad0 = (CFG.tile_size - images.shape[0] % CFG.tile_size)
      pad1 = (CFG.tile_size - images.shape[1] % CFG.tile_size)
    if isinstance(images, np.ndarray):
      images = np.pad(images, [(0,pad0), (0, pad1), (0, 0)], constant_values=0)

    if 'frag' in fragment_id:
      mask = cv2.resize(mask , (mask.shape[1]//2,mask.shape[0]//2), interpolation = cv2.INTER_AREA)

    if scale != 1: # or (images is not None and mask.shape[:2] != images.shape[:2]):
      print("resizing ink labels", mask.shape, scale, images.shape)
      mask = cv2.resize(mask , (mask.shape[1]//scale,mask.shape[0]//scale), interpolation = cv2.INTER_AREA)
      #mask = cv2.resize(mask , (images.shape[1],images.shape[0]), interpolation = cv2.INTER_AREA) # PYGO PYGO THIS WAS THE BUG DISCOVERED ON 5/8!!!
    #if mask is not None and images is not None:

  # mask = np.pad(mask, [(0, pad0), (0, pad1)], constant_values=0)
  print("Reading fragment mask", f"{basepath}/{fragment_id}/{fragment_id}_mask.png")
  fragment_mask=cv2.imread(f"{basepath}/{fragment_id}/{fragment_id}_mask.png", 0)
  if fragment_id=='20230827161846':
      fragment_mask=cv2.flip(fragment_mask,0)
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
    return (None,)*5
  if 'frag' in fragment_id:
      fragment_mask = cv2.resize(fragment_mask, (fragment_mask.shape[1]//2,fragment_mask.shape[0]//2), interpolation = cv2.INTER_AREA)
  if scale != 1: # and images is None: # Whoa there pardner! Scaling down the masks too I see!
    fragment_mask = cv2.resize(fragment_mask , (fragment_mask.shape[1]//scale,fragment_mask.shape[0]//scale), interpolation = cv2.INTER_AREA)
  #elif images is not None and fragment_mask.shape[:2] != images.shape[:2]:
  #  fragment_mask = cv2.resize(fragment_mask, (images.shape[1],images.shape[0]), interpolation = cv2.INTER_AREA)

  if mask is not None and images is not None and fragment_mask is not None:
    print("PYGO images.shape,dtype", images.shape, (images.dtype if isinstance(images, np.ndarray) else None), "mask", mask.shape, mask.dtype, "fragment_mask", fragment_mask.shape, fragment_mask.dtype)
    '''
    minh,minw = min(images.shape[0], mask.shape[0], fragment_mask.shape[0]), min(images.shape[1], mask.shape[1], fragment_mask.shape[1])
    print("Trimming all inputs to have the same spatial dimensions (padding would be meaningless)", minh, minw)
    if images.shape[:2] != (minh, minw):
      images = images[:minh, :minw]
    if mask.shape[:2] != (minh, minw):
      mask = mask[:minh, :minw]
    if fragment_mask.shape[:2] != (minh, minw):
      fragment_mask = fragment_mask[:minh, :minw]
    print("Done trimming.", "This could probably eliminate the necessity of any padding. So...")
    '''
  return images, mask,fragment_mask,pad0,pad1

# TODO: How to grow training set over time??? Automatically look in directory, but need .zarrs to be able to load dynamically.
def reload_masks(masks, CFG):
  print("reloadiing ink labels")
  newmasks = {}
  for fragment_id in masks.keys():
    print("reloading", fragment_id)
    if fragment_id=='20231022170900':
        mask = cv2.imread(f"{basepath}/{fragment_id}/{fragment_id}_inklabels.tiff", 0)
    else:
        mask = cv2.imread(f"{basepath}/{fragment_id}/{fragment_id}_inklabels.png", 0)
    if 'frag' in fragment_id:
        fragment_mask = cv2.resize(fragment_mask, (fragment_mask.shape[1]//2,fragment_mask.shape[0]//2), interpolation = cv2.INTER_AREA)
        mask = cv2.resize(mask , (mask.shape[1]//2,mask.shape[0]//2), interpolation = cv2.INTER_AREA)
    newmasks[fragment_id] = mask[:,:,None] if mask is not None else None
  return newmasks

def reload_validationset():
  pass

#from numba import vectorize
#@vectorize
#@jit(nopython=True)
import numpy as np
def generate_xyxys_ids(fragment_id, image, mask, fragment_mask, tile_size, size, stride, is_valid=False, CFG=None):
        print("    Generating xyxys", fragment_id, image.shape, mask.shape, fragment_mask.shape, "mask min max", mask.min(), mask.max(), "tile_size", tile_size, "size", size, "stride", stride, "is_valid", is_valid)
        noink = os.path.join(CFG.basepath, fragment_id + "_noink.png")
        if os.path.isfile(noink):
          print("Reading NO INK file:", noink)
          noink = cv2.imread(noink, 0)
        else:
          noink = None
        if noink is not None and mask.shape[:2] != noink.shape:
          noink = cv2.resize(noink, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_AREA)
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
          xyxys = [(c[1],c[0],c[1]+size,c[0]+size) for c in np.argwhere(mask[:,:,0] > 0).tolist() if c[0] >= 0 and c[1] >= 0 and c[0]+tile_size < mask.shape[0] and c[1]+tile_size < mask.shape[1]] #[::int(stride)]
          ids = [fragment_id] * len(xyxys)
          return xyxys, ids

        x1_list = list(range(0, image.shape[1]-tile_size+1, stride))
        y1_list = list(range(0, image.shape[0]-tile_size+1, stride))
        #x1_list = list(range(0, image.size()[1]-tile_size+1, stride))
        #y1_list = list(range(0, image.size()[0]-tile_size+1, stride))
        #windows_dict={}
        for a in y1_list:
            for b in x1_list:
                for yi in range(0,tile_size,size):
                    for xi in range(0,tile_size,size):
                        y1=a+yi # Reverse engineer what this is doing!!!
                        x1=b+xi
                        y2=y1+size
                        x2=x1+size
                # for y2 in range(y1,y1 + tile_size,size):
                #     for x2 in range(x1, x1 + tile_size,size):
                        if not is_valid:
                            if (not np.all(np.less(noink[a:a + tile_size, b:b + tile_size, 0],0.0001))): # Pick spots where the ink mask is not all less than a certain amount. TODO: ALSO check non-ink masks!
                                    #if not np.any(np.equal(fragment_mask[a:a+ tile_size, b:b + tile_size],0)): # This also ensures it does not pick up ANY edges. BUUT... Methinks the tile size should actually be smaller than this. ANd it's probably fine to include edges of the papyrus.
                                    xyxys.append([x1,y1,x2,y2])
                                    ids.append(fragment_id)
                                    # if (y1,y2,x1,x2) not in windows_dict:
                                    #train_images.append(image[y1:y2, x1:x2])
                                    #train_masks.append(mask[y1:y2, x1:x2, None])
                                    #assert image[y1:y2, x1:x2].shape==(size,size,in_chans)
                                        # windows_dict[(y1,y2,x1,x2)]='1'
                        else:
                                    #if not np.any(np.equal(fragment_mask[a:a + tile_size, b:b + tile_size], 0)):
                                    #valid_images.append(image[y1:y2, x1:x2])
                                    #valid_masks.append(mask[y1:y2, x1:x2, None])
                                    ids.append(fragment_id)
                                    xyxys.append([x1, y1, x2, y2])
                                    #assert image[y1:y2, x1:x2].shape==(size,size,in_chans)
        return xyxys, ids


import cv2
import json
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
          _, _, fragment_mask,pad0,pad1 = read_image_mask(fragment_id, start_idx, end_idx, cfg, fragment_mask_only=True, pad0=pad0, pad1=pad1, scale=myscale, force_mem=is_valid)
        else:
          image, mask,fragment_mask,pad0,pad1 = read_image_mask(fragment_id, start_idx, end_idx, cfg, scale=myscale, force_mem=is_valid)
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
            xyxy, id = generate_xyxys_ids(fragment_id, image, mask, fragment_mask, cfg.tile_size, cfg.size, cfg.stride, is_valid, CFG=cfg)
          else:
            xyxy, id = generate_xyxys_ids(fragment_id, image, mask, fragment_mask, cfg.valid_tile_size, cfg.valid_size, cfg.valid_stride, is_valid, CFG=cfg)
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
def get_train_valid_dataset(CFG, train_ids=[], valid_ids=[], start_idx=15, end_idx=45, scale=1):
    start_idx = 0
    if len(train_ids) == 0:
      #train_ids = set(["20231210132040", '20230702185753','20230929220926','20231005123336','20231007101619','20231012184423','20231016151002','20231022170901','20231031143852','20231106155351','20231210121321','20231221180251','20230820203112']) # - set([CFG.valid_id])
      #train_ids = set(["20231210132040", '20230929220926','20231005123336','20231007101619','20231016151002','20231022170901','20231031143852','20231106155351','20231210121321','20231221180251','20230820203112']) # - set([CFG.valid_id])
      train_ids = set(["20231210132040", '20230929220926','20231005123336']) #,'20231007101619','20231016151002']) # - set([CFG.valid_id])
      #train_ids = set(['20230702185753','20230929220926','20231007101619','20231012184423','2023101615100','20231031143852','20231106155351','20231221180251','20230820203112']) - set([CFG.valid_id])
      #train_ids  = set(["20240304141530", "20231210132030", "20231122192640", "20231215151901"]) #"20230702185753"])
      #train_ids  = set(["20240304141530", "20231210132030", "20231215151901"]) #, "20231122192640") #, "20231215151901"]) #"20230702185753"]) # TODO add 640 back in once it's there!!! HOW TO TRAIN ON IT???
      #train_ids  = set(["20240304141530"]) #, "20231215151901"]) #, "20231122192640") #, "20231215151901"]) #"20230702185753"]) # TODO add 640 back in once it's there!!! HOW TO TRAIN ON IT???
      #train_ids  = set(["20240304141530", "20231210132030", "20231210132040", "20231122192640", "20231215151901"]) #"20230702185753"])
      #train_ids  = set(["20231210132040"]) #"20230702185753"])
      #train_ids  = set(["20231210132030", "20231210132040", "20231215151901"]) #"20230702185753"])
      #train_ids  = set(["20240304141530", "20231210132030", "20231210132040", "20231215151901"]) #"20230702185753"])
      #train_ids = set(["20231210132040", '20230702185753','20230929220926'])
      train_ids = set(["20231210132040", '20230929220926','20231005123336']) #,'20231007101619']) # - set([CFG.valid_id])
    if len(valid_ids) == 0:
      CFG.valid_id = "20231210132040" #"20231215151901" #20240304141530"
      #valid_ids = set([CFG.valid_id, "20231111135340", "20231122192640"]+list(train_ids))
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
    def __init__(self, images, cfg, xyxys=None, labels=None, ids=None, transform=None, is_valid=False, randomize=False, scale=1, labelscale=4):
        self.images = images
        self.cfg = cfg
        self.labels = labels
        self.transform = transform
        # TODO: Only if scale != 1?
        #print("cleaning xyxys...", len(xyxys))
        #xids = [(xyxy,id) for (xyxy,id) in zip(xyxys,ids) if xyxy[2] < images[id].shape[1] and xyxy[3] < images[id].shape[0]]
        #xyxys,ids = [[xyxy for xyxy,id in xids],[id for xyxy,id in xids]]
        #print("cleaned.", len(xyxys))
        self.xyxys=xyxys
        self.ids = ids
        self.rotate=cfg.rotate
        self.is_valid = is_valid
        self.randomize = randomize
        self.labelscale = labelscale
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
        cropping_num = random.randint(18, 26)

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
        #print(self.xyxys)
        if self.xyxys is not None:
            t = time.time()
            id = self.ids[idx]
            x1,y1,x2,y2=xy=self.xyxys[idx]
            #print("xy,idx", xy,idx)
            start = 15 #0
            end = 45 #self.images[id].shape[-1]
            if self.images[id].shape[-1] == self.cfg.in_chans:
              start = 0
              end = self.cfg.in_chans
            #elif self.randomize and not self.is_valid and self.images[id].shape[-1] > self.cfg.in_chans:
            elif self.images[id].shape[-1] > self.cfg.in_chans:
              if random.random () > 0.5:
                # Squash or stretch channels some
                minextent = self.cfg.in_chans // 2 # Maximum extent is the whole scroll depth.
                extent = random.randint(minextent, self.images[id].shape[-1]+1)
                start = random.randint(0, self.images[id].shape[-1]-extent) if extent < self.images[id].shape[-1] else 0
                end = start + extent
              else:
                start = random.randint(0, self.images[id].shape[-1]-self.cfg.in_chans)
                end = start + self.cfg.in_chans
            #elif self.images[id].shape[-1] >= self.cfg.end_idx: #64:
            #  start = self.cfg.start_idx
            #  end = self.cfg.end_idx
            #else:
            #  end = self.images[id].shape[-1]
            #  start = end - self.cfg.in_chans
            #  #print("Exceeded channel depth bounds for", id, self.images[id].shape)
            #  #return self[idx+1]
            image = self.images[id][y1:y2,x1:x2,start:end] # SethS random depth select aug! #,self.start:self.end] #[idx]
            #if end-start > self.cfg.in_chans: # Stretch
            #  image = image# TODO Seth
            label = self.labels[id][y1:y2,x1:x2]

            if np.product(label.shape) == 0 or np.product(image.shape) == 0:
              print("image.shape", image.shape, "label.shape", label.shape, "id", id, "idx", idx, "x1,x2,y1,y2", x1, x2, y1, y2)
            # TODO: NEED different random crops!!! Including rotations!
            if image.shape[:2] != label.shape[:2]:
              print("MISMATCHED image, label", id, image.shape, label.shape)
              return self[idx+1]
            #print(label.shape)
            #print("Time to get item", time.time()-t)
            #3d rotate
            t = time.time()
            if random.random() < 0.05:
              image=image.transpose(2,1,0)#(c,w,h)
              image=self.rotate(image=image)['image']
              image=image.transpose(0,2,1)#(c,h,w)
              image=self.rotate(image=image)['image']
              image=image.transpose(0,2,1)#(c,w,h)
              image=image.transpose(2,1,0)#(h,w,c)
              #print("Time to augment 1", time.time()-t)
              t = time.time()

            if random.random() < 0.1 and image.shape[-1] == self.cfg.in_chans:
              image=self.fourth_augment(image)
              #print("Time to augment 2", time.time()-t)
              t = time.time()

            if self.transform:
                #if np.product(image.shape) == 0 or image.shape[-1] != self.cfg.in_chans:
                #  print("image.shape", image.shape, "image.dtype", image.dtype, "label.shape", label.shape, "label.dtype", label.dtype)
                data = self.transform(image=image, mask=label)
                image = data['image'].unsqueeze(0)
                #print(image.shape)
                #exit()
                label = data['mask']/255
                image = image.half().to(label.device)
                #if end-start > self.cfg.in_chans: # Squash
                if image.shape[-1] > self.cfg.in_chans: # Squash
                  image = F.interpolate(image.unsqueeze(1), (self.cfg.in_chans, image.shape[2], image.shape[3]), mode="area").squeeze(1) # TODO Seth
                #elif end-start < self.cfg.in_chans: # Stretch
                elif image.shape[-1] < self.cfg.in_chans: # Stretch
                  image = F.interpolate(image.unsqueeze(1), (self.cfg.in_chans, image.shape[2], image.shape[3]), mode="trilinear").squeeze(1) # TODO Seth
                #print("Final Image size", image.shape)
                  #image = F.interpolate(image, (image.shape[0], image.shape[1], self.cfg.in_chans), mode="bilinear") # TODO Seth
                #print("labels.shape", label.shape, self.cfg.size)
                #print("labels.shape", label.shape)
                #label=F.interpolate((label/255).unsqueeze(0).float(),(self.cfg.size//self.labelscale,self.cfg.size//self.labelscale)).squeeze(0) # Label patch size is patch size divided (downscaled) by label_scale # Not needed since automatically matched to network output shape
                #print("Time to augment 3", time.time()-t)
            else:
                print("No augment", image.shape)
            return image, label,xy,id
        else:
            #print("xyxys is None")
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
                label = data['mask']/255
                #label=F.interpolate((label/255).unsqueeze(0).float(),(self.cfg.size//self.labelscale,self.cfg.size//self.labelscale)).squeeze(0)
            return image, label
class CustomDatasetTest(Dataset):
    def __init__(self, images, cfg, xyxys=None, labels=None, ids=None, transform=None, is_valid=False, randomize=False, scale=1, labelscale=4):
    #def __init__(self, images, xyxys, ids, cfg, transform=None):
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
        if self.transform:
            data = self.transform(image=image, mask=label)
            label = data['mask']/255
            image = data['image'].unsqueeze(0)
            image = image.half().to(label.device)
            if image.shape[-1] > self.cfg.in_chans: # Squash
              if image.shape[-1] > 100: # 23%-70% for training.
                image = image[...,36:110]
              else:
                image = image[...,15:45]
              image = F.interpolate(image.unsqueeze(1), (self.cfg.in_chans, image.shape[2], image.shape[3]), mode="area").squeeze(1) # TODO Seth
            elif image.shape[-1] < self.cfg.in_chans: # Stretch
              image = F.interpolate(image.unsqueeze(1), (self.cfg.in_chans, image.shape[2], image.shape[3]), mode="trilinear").squeeze(1) # TODO Seth
            
        #print("val", "getitem", xy, id, image.shape)
        return image, label,xy,id
        #return image,xy



