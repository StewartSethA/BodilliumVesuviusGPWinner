from PIL import Image
Image.MAX_IMAGE_PIXELS = 933120000
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import numpy as np
import random
import cv2
from scipy import ndimage
from skimage.measure import block_reduce
import fastnumpyio as fnp
from tqdm.auto import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time
import json
import zarr
from termcolor import colored
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class ZarrArrayWrapper:
  def __init__(self, array):
    self.array = array
  def __getitem__(self, slice):
    #print("getting from zarr array wrapper", self.shape, "slice:", slice)
    return self.array[(slice[2], slice[0], slice[1])].transpose(1,2,0)
  @property
  def shape(self):
    return self.array.shape[1:] + self.array.shape[0:1]

def read_image_mask(fragment_id,start_idx=15,end_idx=45,CFG=None, fragment_mask_only=False, pad0=0, pad1=0, scale=1, chunksize=128, force_mem=False):
  basepath = CFG.basepath
  scrollsdir = "train_scrolls" if os.path.isdir("train_scrolls") else "train_scrolls2"
  images = None
  fragment_mask = None
  mask = None
  startt = t = time.time()
  if not fragment_mask_only:
    #start_idx = 0
    idxs = range(start_idx, end_idx)
    #print(bcolors.OKBLUE, "Loading", fragment_id, "idxs", idxs, start_idx, end_idx, bcolors.ENDC, end=" ")
    images = []
    loaded = False
    rescaled = False
    #print("Checking for", f"{basepath}/{fragment_id}_{chunksize}.zarr", "at scale", scale) # TODO: Bake in the scaling!
    mra = None
    print(bcolors.OKBLUE, end=" ")
    if scale > 4:
      pass
    elif os.path.exists(f"{basepath}/{fragment_id}_{chunksize}_8level.zarr") and not force_mem and start_idx == 15 and end_idx == 45:
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
        if scale == 1:
          images = mra[0]
        elif scale == 2:
          images = mra[1]
        elif scale == 4:
          images = mra[2]
        else:
          images = []
          rescaled = loaded = False
          print("Skipping zarr, using in-memory dataset instead because scale is too small")
          #a = 1/0 # SethS force punt on more downsampled arrays since these will more easily fit in memory.
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
            print("> Read too little image stack data; skipping ZARR and looking for Numpy or Tiff!", images.shape, start_idx, end_idx, fragment_id)
            images = []
          else:
            rescaled = loaded = True
      except Exception as ex:
        rescaled = loaded = False
        print("> Exception", ex, fragment_id)
    if loaded:
      pass
      #print("Loaded", fragment_id, "shape", images.shape, "at scale", scale)
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
    if loaded and images.shape[-1] < CFG.in_chans:
      print("> Loaded too few layers from numpy!", fragment_id, images.shape)
      loaded = False
    if loaded:
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
            print("---Loading", f"{basepath}/{fragment_id}/layers/{idx:02}.{ext}")
            image = cv2.imread(f"{basepath}/{fragment_id}/layers/{idx:02}.{ext}", 0)
          if image is not None:
            break
        for ext in ['jpg', 'tif']:
          if image is None and os.path.isfile(f"{basepath}/{fragment_id}/layers/{idx:03}.{ext}"): # TODO SethS: We need to accommodate a deeper stack of images here...
            print("---Loading", f"{basepath}/{fragment_id}/layers/{idx:03}.{ext}")
            image = cv2.imread(f"{basepath}/{fragment_id}/layers/{idx:03}.{ext}", 0)
          if image is not None:
            break
        if image is None and "jpg" != ext:
          print("> WARNING: FAILED TO LOAD!", f"{basepath}/{fragment_id}/layers/{idx:03}.{ext}")
          break
          #return None, None, None, None, None
        pad0 = (CFG.tile_size - image.shape[0] % CFG.tile_size)
        pad1 = (CFG.tile_size - image.shape[1] % CFG.tile_size)
        #print("Un-padded image size", image.shape)
        image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)
        # image = ndimage.median_filter(image, size=5) # TODO: Why median filtering?
        # image = cv2.resize(image, (image.shape[1]//2,image.shape[0]//2), interpolation = cv2.INTER_AREA)
        if 'frag' in fragment_id:
            image = cv2.resize(image, (max(1,image.shape[1]//2),max(1,image.shape[0]//2)), interpolation = cv2.INTER_AREA)
        image=np.clip(image,0,200)
        if fragment_id=='20230827161846':
            image=cv2.flip(image,0)
        images.append(image)
      print(time.time()-t, "s to load images;", end=" ")
      if len(images) == 0:
        return None, None, None, None, None
      images = np.stack(images, axis=2)
      t = time.time()
      print(time.time()-t, "seconds taken to stack images.")
      t = time.time()
      if not os.path.isfile(f"{basepath}/{fragment_id}_{start_idx}-{end_idx}.npy") and not os.path.isfile(f"{basepath}/{fragment_id}_{start_idx}-{end_idx}.npy"):
        print("saving", f"{basepath}/{fragment_id}_{start_idx}-{end_idx}.npy", images.shape, end=" ") # Seths
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
      print("Rescaling image...", fragment_id, images.shape, "down by", scale, end=" ")
      t = time.time()
      images = (block_reduce(images, block_size=(scale,scale,1), func=np.mean, cval=np.mean(images))+0.5).astype(np.uint8)
      print("Rescaling took", time.time()-t, "seconds.", images.shape, end=" ")
      np.save(f"{basepath}/{fragment_id}_{start_idx}-{end_idx}_{scale}.npy", images) # Other parts too
      print("Saved rescaled array.")
      pad0 = (CFG.tile_size - images.shape[0] % CFG.tile_size)
      pad1 = (CFG.tile_size - images.shape[1] % CFG.tile_size)
    if isinstance(images, np.ndarray):
      images = np.pad(images, [(0,pad0), (0, pad1), (0, 0)], constant_values=0)

    if 'frag' in fragment_id:
      mask = cv2.resize(mask , (max(1,mask.shape[1]//2),max(1,mask.shape[0]//2)), interpolation = cv2.INTER_AREA)

    if scale != 1: # or (images is not None and mask.shape[:2] != images.shape[:2]):
      print("resizing ink labels", mask.shape, scale, images.shape, end=" ")
      mask = cv2.resize(mask , (max(1,mask.shape[1]//scale),max(1,mask.shape[0]//scale)), interpolation = cv2.INTER_AREA)
      #mask = cv2.resize(mask , (max(1,mask.shape[1]//scale),max(1,mask.shape[0]//scale)), interpolation = cv2.INTER_AREA)
      #print("resized ink labels:", mask.shape)
      #mask = cv2.resize(mask , (images.shape[1],images.shape[0]), interpolation = cv2.INTER_AREA) # PYGO PYGO THIS WAS THE BUG DISCOVERED ON 5/8!!!
    #if mask is not None and images is not None:

  # mask = np.pad(mask, [(0, pad0), (0, pad1)], constant_values=0)
  #print("Reading fragment mask", f"{basepath}/{fragment_id}/{fragment_id}_mask.png")
  fragment_mask=cv2.imread(f"{basepath}/{fragment_id}/{fragment_id}_mask.png", 0)
  if fragment_mask is None:
    # Color output TODO SethS log warning
    if images is None:
      return (None,)*5
    fragment_mask = np.ones((images.shape[0]*scale, images.shape[1]*scale), dtype=np.uint8)*255
  if fragment_id=='20230827161846':
      fragment_mask=cv2.flip(fragment_mask,0)
  if fragment_mask is not None and images is not None and mask is not None and (not fragment_mask_only):
    #print("Padding masks")
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
      fragment_mask = cv2.resize(fragment_mask, (max(1,fragment_mask.shape[1]//2),max(1,fragment_mask.shape[0]//2)), interpolation = cv2.INTER_AREA)
  if scale != 1: # and images is None: # Whoa there pardner! Scaling down the masks too I see!
    fragment_mask = cv2.resize(fragment_mask , (max(1,fragment_mask.shape[1]//scale),max(1,fragment_mask.shape[0]//scale)), interpolation = cv2.INTER_AREA)
  #elif images is not None and fragment_mask.shape[:2] != images.shape[:2]:
  #  fragment_mask = cv2.resize(fragment_mask, (images.shape[1],images.shape[0]), interpolation = cv2.INTER_AREA)

  print(bcolors.ENDC, end=" ")
  if mask is not None and images is not None and fragment_mask is not None:
    print(bcolors.BOLD, time.time()-t, "s to load;", str(np.product(images.shape)/(time.time()-t)/1e9), "GPxCh/s", bcolors.ENDC, end=" ")
    pass
    #print(time.time()-startt, "s", f"{fragment_id}", "Shape:", images.shape, (images.dtype if isinstance(images, np.ndarray) else None), "mask", mask.shape, mask.dtype, "fragment_mask", fragment_mask.shape, fragment_mask.dtype)
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
        fragment_mask = cv2.resize(fragment_mask, (max(1,fragment_mask.shape[1]//2),max(1,fragment_mask.shape[0]//2)), interpolation = cv2.INTER_AREA)
        mask = cv2.resize(mask , (max(1,mask.shape[1]//2),max(1,mask.shape[0]//2)), interpolation = cv2.INTER_AREA)
    newmasks[fragment_id] = mask[:,:,None] if mask is not None else None
  return newmasks

def reload_validationset():
  pass

#from numba import vectorize
#@vectorize
#@jit(nopython=True)
import numpy as np
def generate_xyxys_ids(fragment_id, image, mask, fragment_mask, tile_size, size, stride, is_valid=False, scale=1, CFG=None):
        print(bcolors.OKGREEN, "Gen xyxys", fragment_id, image.shape, mask.shape, fragment_mask.shape, "mask min max", mask.min(), mask.max(), "tile_size", tile_size, "size", size, "stride", stride, "is_valid", is_valid, bcolors.ENDC, end=" ")
        if not is_valid:
          noink = os.path.join(CFG.basepath, fragment_id + "_noink.png")
          if os.path.isfile(noink):
            #print("Reading NO INK file:", noink, end=" ")
            noink = cv2.imread(noink, 0)
          else:
            noink = None
          if noink is not None and mask.shape[:2] != noink.shape:
            #print("noink is NOT same size as mask!", noink.shape, mask.shape)
            #noink = cv2.resize(noink, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_AREA) # TODO TODO TODO: This might introduce some offset / scaling errors!
            noink = cv2.resize(noink, (max(1,noink.shape[1]//scale),max(1,noink.shape[0]//scale)), interpolation=cv2.INTER_AREA) # was fx,fy 1/scale
          if noink is None:
            noink = mask
          else:
            noink = mask[:noink.shape[0],:noink.shape[1],...] + noink[:mask.shape[0],:mask.shape[1],np.newaxis]
          # TODO Use multiplication of masks and np.argwhere to produce suggested coordinates.
        xyxys = []
        ids = []
        #if fragment_id in ['20231007101619']: # SethS 4/17/2024 for suppressing false positive ink labels, include more true negative labels.
        #  is_valid = True
        #if is_valid:
        #  stride = stride * 2
        # TODO SethS:
        h,w = min(image.shape[0],mask.shape[0],fragment_mask.shape[0]), min(image.shape[1],mask.shape[1],fragment_mask.shape[1])
        if is_valid: #Whoops! Was mask, not fragment_mask!
          print("GENERATING xyxys with stride", stride, "size", size, "tile_size", tile_size, "h,w", h,w)
          #xyxys = [(c[1],c[0],c[1]+size,c[0]+size) for c in np.argwhere(fragment_mask[:,:] > 0).tolist() if c[0] >= 0 and c[1] >= 0 and c[0]+tile_size < fragment_mask.shape[0] and c[1]+tile_size < fragment_mask.shape[1]] #[::int(stride)]
          xyxys = [(c[1]*stride,c[0]*stride,c[1]*stride+size,c[0]*stride+size) for c in np.argwhere(fragment_mask[::stride,::stride] > 0).tolist() if c[0] >= 0 and c[1] >= 0 and c[0]*stride+tile_size < h and c[1]*stride+tile_size < w] #[::int(stride)]
          ids = [fragment_id] * len(xyxys)
          print(bcolors.OKCYAN, "len xyxys", len(xyxys), "first 3", xyxys[:3], bcolors.ENDC, end=" ") #, "validation", stride, fragment_mask.shape)
          return xyxys, ids
        if not is_valid: # Added 4/29 SethS
          #xyxys = [(c[1],c[0],c[1]+size,c[0]+size) for c in np.argwhere(noink[:,:,0] > 0).tolist() if c[0] >= 0 and c[1] >= 0 and c[0]+tile_size < mask.shape[0] and c[1]+tile_size < mask.shape[1]] #[::int(stride)]
          xyxys = [(c[1]*stride,c[0]*stride,c[1]*stride+size,c[0]*stride+size) for c in np.argwhere(noink[::stride,::stride,0] > 0).tolist() if c[0] >= 0 and c[1] >= 0 and c[0]*stride+tile_size < h and c[1]*stride+tile_size < w] #[::int(stride)]
          print(bcolors.OKCYAN, "len xyxys", len(xyxys), bcolors.ENDC, end=" ") #, "training", stride, fragment_mask.shape)
          ids = [fragment_id] * len(xyxys)
          return xyxys, ids

        x1_list = list(range(0, image.shape[1]-tile_size+1, stride))
        y1_list = list(range(0, image.shape[0]-tile_size+1, stride))
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

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def save_xyxys(images, masks, xyxys, ids, pads, imsizes, xysizes, cfg, is_valid, scale):
    validlabel="valid" if is_valid else "train"
    myscale=scale
    #if is_valid:
    #  load_id = os.path.join(cfg.basepath, "allfiles" + validlabel+"_tile"+str(cfg.valid_tile_size)+"_size"+str(cfg.valid_size)+"_stride"+str(cfg.valid_stride)+("_scale"+str(myscale) if myscale != 1 else ""))
    #else:
    #  load_id = os.path.join(cfg.basepath, "allfiles" + validlabel+"_tile"+str(cfg.tile_size)+"_size"+str(cfg.size)+"_stride"+str(cfg.stride)+("_scale"+str(myscale) if myscale != 1 else ""))
    if is_valid:
      load_id = os.path.join(cfg.basepath, "allfiles" + validlabel+""+str(cfg.valid_tile_size)+"_"+str(cfg.valid_size)+"_"+str(cfg.valid_stride)+("_s"+str(myscale) if myscale != 1 else ""))
    else:
      load_id = os.path.join(cfg.basepath, "allfiles" + validlabel+""+str(cfg.tile_size)+"_"+str(cfg.size)+"_"+str(cfg.stride)+("_s"+str(myscale) if myscale != 1 else ""))
    #img_load_id = os.path.join(cfg.basepath, ("allfiles_images_") + validlabel+("_scale"+str(myscale) if myscale != 1 else "")) + ".npz"
    #load_id = os.path.join(cfg.basepath, "allfiles" + validlabel+"_tile"+str(cfg.tile_size)+"_size"+str(cfg.size)+"_stride"+str(cfg.stride)+("_scale"+str(myscale) if myscale != 1 else ""))
    #load_id = os.path.join(cfg.basepath, "allfiles" + validlabel+str(cfg.tile_size)+"_"+str(cfg.size)+"_"+str(cfg.stride)+("_s"+str(myscale) if myscale != 1 else ""))
    img_load_id = os.path.join(cfg.basepath, ("allfiles_images_") + validlabel+("_scale"+str(myscale) if myscale != 1 else "")) + ".npz"
    np.savez(img_load_id, **images)
    masks_load_id = os.path.join(cfg.basepath, ("allfiles_masks_") + validlabel+("_scale"+str(myscale) if myscale != 1 else "")) + ".npz"
    np.savez(masks_load_id, **masks)
    xys_load_id = load_id+"_xys.npy"
    np.save(xys_load_id, xyxys)
    with open(load_id+"_ids.json", 'w') as f:
        json.dump(ids, f)
    with open(load_id+"_pads.json", 'w') as f:
        json.dump(pads, f)
    with open(load_id+"_imsizes.json", 'w') as f:
        json.dump(imsizes, f, cls=NpEncoder)
    with open(load_id+"_xysizes.json", 'w') as f:
        json.dump(xysizes, f, cls=NpEncoder)

def get_xyxys(fragment_ids, cfg, is_valid=False, start_idx=15, end_idx=45, train_images={}, train_masks={}, train_ids=[], pads={}, scale=1, is_main=False):
    #print("LOADING IMAGES!", fragment_ids, "is_valid", is_valid, "CFG:", cfg)
    validlabel="valid" if is_valid else "train"
    myscale=scale
    if is_valid:
      load_id = os.path.join(cfg.basepath, "allfiles" + validlabel+""+str(cfg.valid_tile_size)+"_"+str(cfg.valid_size)+"_"+str(cfg.valid_stride)+("_s"+str(myscale) if myscale != 1 else ""))
    else:
      load_id = os.path.join(cfg.basepath, "allfiles" + validlabel+""+str(cfg.tile_size)+"_"+str(cfg.size)+"_"+str(cfg.stride)+("_s"+str(myscale) if myscale != 1 else ""))
    #if is_valid:
    #  load_id = os.path.join(cfg.basepath, "allfiles" + validlabel+"_tile"+str(cfg.valid_tile_size)+"_size"+str(cfg.valid_size)+"_stride"+str(cfg.valid_stride)+("_scale"+str(myscale) if myscale != 1 else ""))
    #else:
    #  load_id = os.path.join(cfg.basepath, "allfiles" + validlabel+"_tile"+str(cfg.tile_size)+"_size"+str(cfg.size)+"_stride"+str(cfg.stride)+("_scale"+str(myscale) if myscale != 1 else ""))
    img_load_id = os.path.join(cfg.basepath, ("allfiles_images_") + validlabel+("_scale"+str(myscale) if myscale != 1 else "")) + ".npz"
    if os.path.exists(img_load_id):
      if is_main:
        print(bcolors.OKGREEN, "Loading all contents from", load_id, bcolors.ENDC)
      try:
        images = np.load(img_load_id)
        images = {k:v for k,v in images.items()}
        masks_load_id = os.path.join(cfg.basepath, ("allfiles_masks_") + validlabel+("_scale"+str(myscale) if myscale != 1 else "")) + ".npz"
        masks = np.load(masks_load_id)
        masks = {k:v for k,v in masks.items()}
        try:
          with open(load_id+"_pads.json", 'r') as f:
            pads = json.load(f)
          xys_load_id = load_id+"_xys.npy"
          xyxys = np.load(xys_load_id)
          if is_main:
            print("get_xyxys len(xyxys)", len(xyxys), "xyxys", xyxys[:3], "loaded from", load_id+"_xys.npy")
          #xyxys = {k:v for k,v in xyxys.items()}
          with open(load_id+"_ids.json", 'r') as f:
            ids = json.load(f)
        except Exception as ex:
          xyxys = []
          ids = []
          for fragment_id in tqdm(fragment_ids):
            image = images[fragment_id]
            mask = masks[fragment_id]
            pad0, pad1 = pads.get(fragment_id)
            _, _, fragment_mask,pad0,pad1 = read_image_mask(fragment_id, start_idx, end_idx, cfg, fragment_mask_only=True, pad0=pad0, pad1=pad1, scale=myscale, force_mem=is_valid)
            myscale = scale
            if fragment_id in ['20231215151901', '20231111135340', '20231122192640']: # TODO SethS need to add to this???
              myscale = scale * 2 # 2040 was extracted at 7.91 um, same as Scroll1
            if not is_valid:
              xyxy, id = generate_xyxys_ids(fragment_id, image, mask, fragment_mask, cfg.tile_size, cfg.size, cfg.stride, is_valid, scale=myscale, CFG=cfg)
            else:
              xyxy, id = generate_xyxys_ids(fragment_id, image, mask, fragment_mask, cfg.valid_tile_size, cfg.valid_size, cfg.valid_stride, is_valid, scale=myscale, CFG=cfg)
            xyxys = xyxys + xyxy
            ids = ids + id
            xysizes.append(len(xyxy))
            pads[fragment_id] = (pad0, pad1)
            if is_main:
              print("xyxy", xyxy, "id", id)

          xys_load_id = load_id+"_xys.npy"
          np.save(xys_load_id, xyxys)
          with open(load_id+"_ids.json", 'w') as f:
            json.dump(ids, f)
          with open(load_id+"_pads.json", 'w') as f:
            json.dump(pads, f)

        #with open(load_id+"_imsizes.json", 'r') as f:
        imsizes = [np.product(x.shape) for x in images.values()] #json.load(f, cls=NpEncoder)
        #with open(load_id+"_xysizes.json", 'r') as f:
        xysizes = [np.product(xyxys.shape)] #[len(xy) for xy in xyxys.values()] #json.load(f, cls=NpEncoder)
        #  xysizes = json.load(f, cls=NpEncoder)
        return images, masks, xyxys, ids, pads, imsizes, xysizes
      except Exception as ex:
        print(bcolors.FAIL, ex, bcolors.ENDC)

    import pickle
    if os.path.exists(load_id+".pickle"):
      with open(load_id+".pickle", 'rb') as handle:
        b = pickle.load(handle)
      images, masks, xyxys, ids, pads, imsizes, xysizes = b
      save_xyxys(images, masks, xyxys, ids, pads, imsizes, xysizes, cfg, is_valid, scale)
      return images, masks, xyxys, ids, pads, imsizes, xysizes

    xyxys = []
    ids = []
    images = {}
    masks = {}
    imsizes = []
    xysizes = []
    for fragment_id in tqdm(fragment_ids):
        myscale = scale
        if fragment_id in ['20231215151901', '20231111135340', '20231122192640']: # TODO SethS need to add to this???
          myscale = scale * 2 # 2040 was extracted at 7.91 um, same as Scroll1
        #start_idx = len(fragment_ids) # WHY ????
        if fragment_id in train_images.keys():
          image, mask = train_images[fragment_id], train_masks[fragment_id]
          pad0, pad1 = pads.get(fragment_id)
          _, _, fragment_mask,pad0,pad1 = read_image_mask(fragment_id, start_idx, end_idx, cfg, fragment_mask_only=True, pad0=pad0, pad1=pad1, scale=myscale, force_mem=is_valid)
        else:
          image, mask,fragment_mask,pad0,pad1 = read_image_mask(fragment_id, start_idx, end_idx, cfg, scale=myscale, force_mem=is_valid)
        if image is None and is_main:
          print(bcolors.WARNING, "WARNING: Failed to load images for", fragment_id, "skipping...", bcolors.ENDC)
          continue
        pads[fragment_id] = (pad0, pad1)
        images[fragment_id] = image
        if image is None:
          masks[fragment_id] = None
          continue
          #return None, None, None, None
        if mask is None and is_main:
          print(bcolors.WARNING, "Defaulting to empty mask! I hope this is just for inference!", fragment_id, bcolors.ENDC)
          mask = np.zeros_like(image[:,:,0])

        masks[fragment_id] = mask = mask[:,:,np.newaxis] if len(mask.shape) == 2 else mask
        t = time.time()
        savename = os.path.join(cfg.basepath, fragment_id + validlabel+str(cfg.tile_size)+"_"+str(cfg.size)+"_"+str(cfg.stride)+("_s"+str(myscale) if myscale != 1 else ""))
        #print("Loading IDs, xyxys", savename)
        if os.path.isfile(savename+".ids.json") and is_main and False:
          with open(savename + ".ids.json", 'r') as f:
            id = json.load(f)
        if os.path.isfile(savename + ".xyxys.json") and is_main and False:
          with open(savename + ".xyxys.json", 'r') as f:
            xyxy = json.load(f)
            print(bcolors.OKCYAN, savename, "xyxys len:", len(xyxy), image.shape, bcolors.ENDC)
        else:
          #print("Generating new xyxys for", fragment_id, image.shape, "mask", mask.shape)
          if not is_valid:
            xyxy, id = generate_xyxys_ids(fragment_id, image, mask, fragment_mask, cfg.tile_size, cfg.size, cfg.stride, is_valid, scale=myscale, CFG=cfg)
          else:
            xyxy, id = generate_xyxys_ids(fragment_id, image, mask, fragment_mask, cfg.valid_tile_size, cfg.valid_size, cfg.valid_stride, is_valid, scale=myscale, CFG=cfg)
          #print("saving xyxys and ids", len(xyxy), len(id), "to", savename)
          #with open(savename + ".ids.json", 'w') as f:
          #  if fragment_id != cfg.valid_id:
          #    json.dump(id, f) #[start_idx:], f)
          #  else:
          #    json.dump(id, f)
          #with open(savename +".xyxys.json", 'w') as f:
          #  if fragment_id != cfg.valid_id:
          #    json.dump(xyxy, f) #[start_idx:],f)
          #  else:
          #    json.dump(xyxy, f)
        #print("xyxys", xyxys, xyxy, xyxy[-1], len(xyxys), len(xyxy))
        xyxys = xyxys + xyxy
        ids = ids + id
        imsizes.append(np.product(image.shape))
        xysizes.append(len(xyxy))

        if is_main:
          print(bcolors.BOLD, time.time()-t, "seconds taken to generate crops for fragment", fragment_id, bcolors.ENDC)

    if is_main:
      print(bcolors.OKGREEN, "Saving whole dataset as NPZ", load_id, bcolors.ENDC)
      #print("Saving whole dataset as a pickle", load_id)
      #a = [images, masks, xyxys, ids, pads, imsizes, xysizes]
      #with open(load_id, 'wb') as handle:
      #  pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)
      save_xyxys(images, masks, xyxys, ids, pads, imsizes, xysizes, cfg, is_valid, scale)
    return images, masks, xyxys, ids, pads, imsizes, xysizes

#@jit(nopython=True)
def get_train_valid_dataset(CFG, train_ids=[], valid_ids=[], start_idx=15, end_idx=45, scale=1, is_main=False):
    if is_main:
      print(bcolors.OKCYAN, "get_train_valid_dataset: CFG.size", CFG.size, "stride", CFG.stride, "Valid size", CFG.valid_size, "stride", CFG.valid_stride, bcolors.ENDC)
    #exit()
    start_idx = 0
    if len(train_ids) == 0:
      train_ids = set(["20231210132040", '20230702185753','20230929220926','20231005123336','20231007101619','20231012184423','20231016151002','20231022170901','20231031143852','20231106155351','20231210121321','20231221180251','20230820203112']) # - set([CFG.valid_id])
    if len(valid_ids) == 0:
      CFG.valid_id = "20231210132040"
      valid_ids = set([CFG.valid_id])
    if is_main:
      print("Loading training images")
    t = time.time()
    train_images, train_masks, train_xyxys, train_ids, pads, imsizes, xysizes = get_xyxys(train_ids, CFG, False, start_idx=start_idx, end_idx=end_idx, scale=scale)
    if is_main:
      print("TRAIN image sizes", imsizes)
      print("TRAIN xy sizes", xysizes)
      print("TRAIN total image size", sum(imsizes)/1e9, "GB")
      print("TRAIN total xy size", sum(xysizes))
      t = time.time()-t
      print("Total time taken to load training images:", t, "seconds, GB/s:", sum(imsizes)/1e9 / t)
      print("Loading validation images")
      t = time.time()
    valid_images, valid_masks, valid_xyxys, valid_ids, _, imsizes, xysizes = get_xyxys(valid_ids, CFG, True, start_idx=start_idx, end_idx=end_idx, train_images=train_images, train_masks=train_masks, train_ids=train_ids, pads=pads, scale=scale, is_main=is_main)
    if is_main:
      print("VALID image sizes", imsizes)
      print("VALID xy sizes", xysizes)
      print("VALID total image size", sum(imsizes)/1e9, "GB")
      print("VALID total xy size", sum(xysizes))
      t = time.time()-t
      print("Total time taken to load validation images:", t, "seconds, GB/s:", sum(imsizes)/1e9 / t)
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
            if x2-x1 != y2-y1:
              print("MISMATCHED XY!", x1,y1,x2,y2)
            #if x2 > self.images[id].shape[1] or y2 > self.images[id].shape[0]:
            if x1 >= self.images[id].shape[1] or y1 >= self.images[id].shape[0]:
              print("OOB XY!", x1,y1,x2,y2, self.images[id].shape)
            #print(x1,y1,x2,y2)
            #exit()
            #print("xy,idx", xy,idx)
            start = 15 #0
            end = 45 #self.images[id].shape[-1]
            if self.images[id].shape[-1] == self.cfg.in_chans:
              start = 0
              end = self.cfg.in_chans
            #elif self.randomize and not self.is_valid and self.images[id].shape[-1] > self.cfg.in_chans:
            elif False and self.images[id].shape[-1] > self.cfg.in_chans:
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
            #image = torch.nn.functional.pad(image,(0,0,0,(x2-x1)-image.shape[1],0,(y2-y1)-image.shape[0]), value=0)
            if image.shape[0] != y2-y1 or image.shape[1] != x2-x1:
              #print("Image padding!", image.shape, x1,x2,y1,y2)
              image = np.pad(image,((0,(y2-y1)-image.shape[0]),(0,(x2-x1)-image.shape[1]),(0,0)), constant_values=0)
              #print("New shape:", image.shape)
            #if end-start > self.cfg.in_chans: # Stretch
            #  image = image# TODO Seth
            label = self.labels[id][y1:y2,x1:x2]
            if label.shape[0] != y2-y1 or label.shape[1] != x2-x1:
              #print("Label padding!", label.shape, x1,x2,y1,y2)
              #label = torch.nn.functional.pad(label,(0,0,0,(x2-x1)-label.shape[1],0,(y2-y1)-label.shape[0]), value=0) # SethS padding 8/27
              label = np.pad(label,((0,(y2-y1)-label.shape[0]),(0,(x2-x1)-label.shape[1]),(0,0)), constant_values=0)
              #label = torch.nn.functional.pad(label,(0,0,0,(x2-x1)-label.shape[1],0,(y2-y1)-label.shape[0]), value=0) # SethS padding 8/27
              #print("New shape:", label.shape)

            if np.product(label.shape) == 0 or np.product(image.shape) == 0:
              print("BAD image.shape", image.shape, "label.shape", label.shape, "id", id, "idx", idx, "x1,x2,y1,y2", x1, x2, y1, y2, self.images[id].shape, self.labels[id].shape)
              return self[idx+1]
            # TODO: NEED different random crops!!! Including rotations!
            if image.shape[:2] != label.shape[:2]:
              print("MISMATCHED image, label", id, image.shape, label.shape) # TODO: Should pad image to match labels???
              return self[idx+1]
            #print(label.shape)
            #print("Time to get item", time.time()-t)
            #3d rotate
            #print("trn image.shape", image.shape, label.shape, xy, id)
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
                #image = ((image - image.mean()) / max(0.001, image.std())).astype(np.float32) # Standardize (removing information)
                #image = (image - image.mean()) / max(0.001, image.std()).astype(np.float32) # Standardize (removing information)
                #if np.product(image.shape) == 0 or image.shape[-1] != self.cfg.in_chans:
                #  print("image.shape", image.shape, "image.dtype", image.dtype, "label.shape", label.shape, "label.dtype", label.dtype)
                data = self.transform(image=image, mask=label)
                image = data['image'].unsqueeze(0)
                #print(image.shape)
                #exit()
                label = data['mask']/255
                if random.random() > 0.5:
                  label = label * torch.rand_like(label) + torch.rand_like(label) * random.random() * 0.1 # SethS 2:31 p.m. LABEL SMOOTHING
                image = image.half().to(label.device)
                #if end-start > self.cfg.in_chans: # Squash
                #print("Image shape", image.shape, label.shape)
                if image.shape[1] > self.cfg.in_chans: # Squash
                  image = F.interpolate(image.unsqueeze(1), (self.cfg.in_chans, image.shape[2], image.shape[3]), mode="area").squeeze(1) # TODO Seth
                #elif end-start < self.cfg.in_chans: # Stretch
                elif image.shape[1] < self.cfg.in_chans: # Stretch
                  image = F.interpolate(image.unsqueeze(1), (self.cfg.in_chans, image.shape[2], image.shape[3]), mode="trilinear").squeeze(1) # TODO Seth
                #print("Post-resize Image shape", image.shape, label.shape)
                #print("Final Image size", image.shape)
                  #image = F.interpolate(image, (image.shape[0], image.shape[1], self.cfg.in_chans), mode="bilinear") # TODO Seth
                #print("labels.shape", label.shape, self.cfg.size)
                #print("labels.shape", label.shape)
                #label=F.interpolate((label/255).unsqueeze(0).float(),(self.cfg.size//self.labelscale,self.cfg.size//self.labelscale)).squeeze(0) # Label patch size is patch size divided (downscaled) by label_scale # Not needed since automatically matched to network output shape
                #print("Time to augment 3", time.time()-t)
            else:
                print("No augment", image.shape)
            #print("PT image.shape", image.shape, label.shape, xy, id)
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
        self.images = images
        self.labels = labels
        self.xyxys=xyxys
        self.ids = ids
        self.cfg = cfg
        self.transform = transform
    def __len__(self):
        return len(self.xyxys)
    def __getitem__(self, idx):
        x1,y1,x2,y2=xy=self.xyxys[idx]
        id = self.ids[idx]
        image = self.images[id][y1:y2,x1:x2,15:45] if self.images[id].shape[-1] > 30 else self.images[id][y1:y2,x1:x2]
        label = self.labels[id][y1:y2,x1:x2]
        #print("Test dataset image.shape", image.shape, "label.shape", label.shape)
        #print("Val image.shape", image.shape, label.shape, id, xy)
        if self.transform:
            #image = ((image - image.mean()) / max(0.001, image.std())).astype(np.float32) # Standardize (removing information)
            #image = (image - image.mean()) / max(0.001, image.std()) # Standardize (removing information)
            if image.dtype == np.uint16:
              print("Found uint16 image:", image.shape, image.dtype, id)
              #self.images[id] = self.images[id].astype(np.uint8)
              image = image.astype(np.uint8)
            data = self.transform(image=image, mask=label)
            label = data['mask']/255
            image = data['image'].unsqueeze(0)
            image = image.half().to(label.device)
            if image.shape[1] > self.cfg.in_chans: # Squash
              if image.shape[1] > 100: # 23%-70% for training.
                image = image[:,36:110,...]
              else:
                image = image[:,15:45,...]
              image = F.interpolate(image.unsqueeze(1), (self.cfg.in_chans, image.shape[2], image.shape[3]), mode="area").squeeze(1) # TODO Seth
            elif image.shape[1] < self.cfg.in_chans: # Stretch
              #print("image.shape", image.shape, "less than config in_chans", self.cfg.in_chans, "cfg.size", self.cfg.size, "tile_size", self.cfg.tile_size, "stride", self.cfg.stride, "valsize,tile,stride", self.cfg.valid_size, self.cfg.valid_tile_size, self.cfg.valid_stride)
              image = F.interpolate(image.unsqueeze(1), (self.cfg.in_chans, image.shape[2], image.shape[3]), mode="trilinear").squeeze(1) # TODO Seth
            #print("post resize image.shape", image.shape, label.shape)
        #print("Val PT image.shape", image.shape, label.shape, id, xy)
        return image, label, xy, id



scroll1_ids = fragments = ['20230702185753','20230929220926','20231005123336','20231012184423','20231007101619','20231016151002','20231022170901','20231031143852','20231106155351','20231210121321','20231221180251','20230820203112']
scroll4_ids = ['20231111135340', '20231122192640', '20231210132040', '20240304141531', '20231215151901', '20240304144031', '20240304161941'] #+ scroll1val
scroll3_ids = ['20231030220150', '20231031231220']
with open("scroll2.ids", 'r') as f:
  scroll2_ids = [line.strip() for line in f.readlines()]

id2scroll = {v:"1" for v in scroll1_ids}
id2scroll.update({v:"2" for v in scroll2_ids})
id2scroll.update({v:"3" for v in scroll3_ids})
id2scroll.update({v:"4" for v in scroll4_ids})

def custom_collate_fn(data): #images=None, labels=None, xys=None, ids=None):
  #print(data)
  images = [d[0] for d in data]
  labels = [d[1] for d in data]
  xys = [d[2] for d in data]
  ids = [d[3] for d in data]
  assert(len(data[0]) == 4)
  #	images, labels, xys, ids = data
  images = torch.stack(images)
  labels = torch.stack(labels)
  xys = xys
  ids = ids
  return images, labels, xys, ids
