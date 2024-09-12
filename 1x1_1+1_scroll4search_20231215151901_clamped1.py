#print("Importing...")
import sys
import os
os.environ["WANDB_MODE"] = "offline"
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = 9331200000
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import WandbLogger
import random
import numpy as np
import wandb
import cv2
from tqdm.auto import tqdm
import argparse
from torch.optim import AdamW
import datetime
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from i3dallnl import InceptionI3d
from scipy import ndimage
import time
import json
from torch.utils.tensorboard import SummaryWriter
from skimage.measure import block_reduce
#print("Done importing")
#print("Executable path", sys.executable)

from config import CFG

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
    print("set_seed1x1", cfg.seed)
    # set_env_name()
    # set_dataset_path(cfg)

    if mode == 'train':
        make_dirs(cfg)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = None
def get_transforms(data, cfg):
    if data == 'train':
        aug = A.Compose(cfg.train_aug_list)
    elif data == 'valid':
        aug = A.Compose(cfg.valid_aug_list)
    return aug

from plmodel import *
from argparse import ArgumentParser
#from config import CFG

parser = ArgumentParser()
parser.add_argument('--scale', type=int, default=1, required=False)
parser.add_argument('--tile_size', type=int, default=256, required=False)
parser.add_argument('--size', type=int, default=64, required=False)
parser.add_argument('--stride', type=int, default=32, required=False)
#parser.add_argument('--model', type=str, default="pygoflat", required=False)
parser.add_argument('--model', type=str, default="pygo1x1", required=False)
parser.add_argument('--name', type=str, default="default", required=False)
parser.add_argument('--load', type=str, default="", required=False)
parser.add_argument('--complexity', type=int, default=16, required=False)
parser.add_argument('--epochs', type=int, default=12, required=False)
parser.add_argument('--batch_size', type=int, default=256, required=False)
parser.add_argument('--val_batch_size', type=int, default=1, required=False)
parser.add_argument('--val_size', type=int, default=224, required=False)
parser.add_argument('--val_stride', type=int, default=112, required=False)
parser.add_argument('--minbatches', type=int, default=1000000, required=False)
parser.add_argument('--mode', type=str, default="normal", required=False)
parser.add_argument('--seed', type=int, default=42, required=False)
parser.add_argument('--scrollsval', type=str, default="1,2,3,4", required=False)
parser.add_argument('--fragmentsval', type=str, default="20231012184423", required=False)
parser.add_argument('--lr', type=float, default=2e-5, required=False)
#parser.add_argument('--model', type=str, default="i3d", required=False)
args = parser.parse_args()

CFG.scale = args.scale
CFG.tile_size = args.tile_size
CFG.size = args.size
CFG.stride = args.stride
CFG.train_batch_size = args.batch_size
CFG.valid_tile_size = args.val_size
CFG.valid_stride = args.val_stride
CFG.valid_size = args.val_size # TODO: Allow different validation size from training size
CFG.seed = args.seed
CFG.lr = args.lr
from dataloaders import *
torch.set_float32_matmul_precision('medium')

scroll1val = args.fragmentsval.split(",")
#scroll1_ids = fragments = ['20230702185753','20230929220926','20231005123336','20231012184423','20231007101619','20231016151002','20231022170901','20231031143852','20231106155351','20231210121321','20231221180251','20230820203112']

#scroll1_ids = fragments = ['20231005123336']
#scroll1_ids = fragments = ['20230702185753','20230929220926','20231005123336'] #,'20231012184423','20231007101619','20231016151002','20231022170901','20231031143852','20231106155351','20231210121321','20231221180251','20230820203112']
#scroll4_ids = ['20231111135340', '20231122192640', '20231210132040', '20240304141530', '20231215151901', '20240304144030', '20240304161940'] #+ scroll1val
#scroll3_ids = ['20231030220150', '20231031231220']
#scroll2_ids = []
#with open("scroll2.ids", 'r') as f:
#  scroll2_ids = [line.strip() for line in f.readlines()]
#train_scrolls = "train_scrolls" if os.path.isdir("train_scrolls") else "train_scrolls2"


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

enc_i,enc,fold=0,'i3d',0
fid = CFG.valid_id

name = f'{args.model}_scale{CFG.scale}_size{CFG.size}_tilesize{CFG.tile_size}_stride{CFG.stride}val{CFG.valid_stride}_batch{args.batch_size}_{args.name}'
origname = name
#exit()
CFG.model_dir = os.path.join("outputs", name)
cfg_init(CFG)

'''
train_scrolls = CFG.basepath #"train_scrolls" if os.path.isdir("train_scrolls") else "train_scrolls2"

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
    #pred_shape = {}
    valid_ids = scroll1val
    train_ids = scroll4_ids[0] # TODO: Brute force this more, man!
    if "4" in args.scrollsval or True:
      valid_ids = valid_ids + scroll4_ids
    if "3" in args.scrollsval:
      valid_ids = valid_ids + scroll3_ids
    if "2" in args.scrollsval:
      valid_ids = valid_ids + scroll2_ids
    if "1" in args.scrollsval:
      valid_ids = valid_ids + fragments #['20231005123336']) #set(list(train_ids) + scroll4_ids + scroll2_ids + scroll3_ids)
    #print("Valid ids", valid_ids)
    valid_ids = scroll4_ids + fragments + scroll3_ids + scroll2_ids #['20231005123336']
    #valid_ids = list(set(valid_ids))[:3] #['20231005123336']) #set(list(train_ids) + scroll4_ids + scroll2_ids + scroll3_ids)
    #valid_ids = list(set(valid_ids)) + ["20240304144031",] #[:3])) #[:3] #['20231005123336']) #set(list(train_ids) + scroll4_ids + scroll2_ids + scroll3_ids)
    #valid_ids = list(set(scroll4_ids + scroll3_ids + scroll2_ids + scroll1val + fragments)) #['20231005123336']) #set(list(train_ids) + scroll4_ids + scroll2_ids + scroll3_ids)
    #train_ids = scroll1_ids + list((set(fragments) - set(valid_ids)) - set(scroll1val)) + ["20240304144031", "20231210132040", "20231215151901", "20231122192640", "20231111135340", "20240304161941", "20240304141531"] #+ set( #scroll4_ids) 
    train_ids = scroll4_ids

    '''
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
    '''

    train_masks = None
    for randtrial in [3,4,5,6,7,9,10,11,12,13,14,15]: #range(11,15): #100 9/10
      name = origname+"_"+str(randtrial)
      CFG.model_dir = os.path.join("outputs", name)
      CFG.seed = random.randint(0,10000000)
      cfg_init(CFG)
      run_slug=name #f'training_scrolls_valid={fragment_id}_{model}_{CFG.size}size_{CFG.tile_size}tile_size_{CFG.stride}stride_{CFG.scale}scale'
      wandb_logger = WandbLogger(project="vesuvius",name=run_slug+f'{enc}_finetune')
      trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=torch.cuda.device_count(),
        logger=wandb_logger,
        default_root_dir='./', #"/content/gdrive/MyDrive/vesuvius_model/training/outputs",
        accumulate_grad_batches=1,
        #auto_scale_batch_size='binsearch',
        precision='16-mixed',
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        strategy='ddp', #_find_unused_parameters_true',
        callbacks=[ModelCheckpoint(filename=f'{args.model}_{fid}_{enc}_{name}_scale{CFG.scale}_size{CFG.size}_stride{CFG.stride}',dirpath=CFG.model_dir,monitor='train/total_loss',mode='min',save_top_k=1),],
                    #StochasticWeightAveraging(2e-5, annealing_epochs=5, device=None)],
      )
      if randtrial == 0 or train_masks is None:
        train_images, train_masks, train_xyxys, train_ids, valid_images, valid_masks, valid_xyxys, valid_ids = get_train_valid_dataset(CFG, train_ids, valid_ids, start_idx=0, end_idx=65, scale=CFG.scale, is_main=trainer.is_global_zero)
      if trainer.is_global_zero:
        print(bcolors.OKGREEN, "train_ids", len(train_ids), set(train_ids), bcolors.ENDC)
        print(bcolors.OKBLUE, "valid_ids", len(valid_ids), set(valid_ids), bcolors.ENDC)
      if trainer.is_global_zero:
        print(bcolors.OKGREEN, "fragments", set(fragments), "(from dataloaders import *)", bcolors.ENDC)

      maskmax = max([t.max() for t in train_masks.values() if t is not None])
      train_masks = {id:np.zeros_like(t) for id,t in train_masks.items()} #TODO SethS: Random pixel ON and OFF!
      train_noinkmasks = {id:np.zeros_like(t) for id,t in train_masks.items()} #TODO SethS: Random pixel ON and OFF!
      print("train_masks", train_masks.keys())
      whichmaskcandidates = [t for t in list(train_masks.keys()) if train_masks[t] is not None and len(train_masks[t].shape)>=2]
      whichmask = whichmaskcandidates[random.randint(0,len(whichmaskcandidates)-1)]
      whichmask = '20231215151901' # SethS 9/10/24
      print("whichmask", whichmask)
      maskshape = train_masks[whichmask].shape
      print("maskshape", maskshape)
      #randnoinkwidth,randnoinkheight = max(1,int(maskshape[1]*0.05)), int(maskshape[0]*0.67) #+1 -CFG.size #max(1,int(maskshape[0]*0.5-CFG.size/2)) #random.randint(1, max(1,int(maskshape[1] * 0.05))), random.randint(1, max(1,int(maskshape[0]*0.75)))
      #randnoinkwidth,randnoinkheight = max(1,int(maskshape[1]*0.05)), max(CFG.size+1,int(maskshape[0]*0.5)) # SethS 9/10 #+1 -CFG.size #max(1,int(maskshape[0]*0.5-CFG.size/2)) #random.randint(1, max(1,int(maskshape[1] * 0.05))), random.randint(1, max(1,int(maskshape[0]*0.75)))
      randnoinkwidth,randnoinkheight = max(2,int(maskshape[1]*0.05)), max(CFG.size+1,int(maskshape[0]*0.5)) # SethS 9/10 #+1 -CFG.size #max(1,int(maskshape[0]*0.5-CFG.size/2)) #random.randint(1, max(1,int(maskshape[1] * 0.05))), random.randint(1, max(1,int(maskshape[0]*0.75)))
      #randnoink = int(maskshape[0]*0.25), random.randint(0, max(1,maskshape[1]-randnoinkwidth-1-CFG.size)) #random.randint(0,max(1,int(maskshape[0]*0.95)-randnoinkheight-CFG.size)),random.randint(0,max(1,maskshape[1]-randnoinkwidth-CFG.size))
      #randnoink = 0, random.randint(0, max(1,int(maskshape[1]*0.8)-1-CFG.size)) #random.randint(0,max(1,int(maskshape[0]*0.95)-randnoinkheight-CFG.size)),random.randint(0,max(1,maskshape[1]-randnoinkwidth-CFG.size))
      #randnoink = int(maskshape[0]*.33*.5), random.randint(0, max(1,int(maskshape[1]*0.8)-1)) #random.randint(0,max(1,int(maskshape[0]*0.95)-randnoinkheight-CFG.size)),random.randint(0,max(1,maskshape[1]-randnoinkwidth-CFG.size))
      randnoink = int(maskshape[0]*.33*.5), 2 # SethS 9/10 #random.randint(0, max(1,int(maskshape[1]*0.8)-1)) #random.randint(0,max(1,int(maskshape[0]*0.95)-randnoinkheight-CFG.size)),random.randint(0,max(1,maskshape[1]-randnoinkwidth-CFG.size))
      randnoink = randnoink[0], int(float(randtrial)/48.0*maskshape[1]) # SethS 9/11/2024
      randink = randnoink[0],randnoinkwidth+randnoink[1] #random.randint(0,max(1,maskshape[0]-randinkheight-CFG.size)),random.randint(randnoink[1]+randnoinkwidth,maskshape[1]-CFG.size)
      #randinkwidth,randinkheight = random.randint(1, maskshape[1]-randink[1]-CFG.size),randnoinkheight
      #randinkwidth,randinkheight = random.randint(1+CFG.size, maskshape[1]-randink[1]),randnoinkheight
      randinkwidth,randinkheight = max(int(.1875*maskshape[1]), CFG.size+1),randnoinkheight

      # 9/11/2024: SWAP ink an non-ink sections.
      randink = randnoink
      randnoink = randink[0], randink[1]+randinkwidth

      randink = 2,15
      randnoink = 2,21
      randinkheight = 10
      randinkwidth = 6
      randnoinkheight = 10
      randnoinkwidth = randtrial

      #randnoink,randink = (randnoink[1],randnoink[0]), (randink[1],randink[0])

      '''
      randnoink = maskshape[0], maskshape[1]
      while randnoink[1] >= maskshape[1]-1:
        randnoinkwidth,randnoinkheight = random.randint(1, max(1,int(maskshape[1] * 0.05))), random.randint(1, max(1,int(maskshape[0]*0.75)))
        randnoink = random.randint(0,max(1,int(maskshape[0]*0.95)-randnoinkheight-CFG.size)),random.randint(0,max(1,maskshape[1]-randnoinkwidth-CFG.size))
      randinkwidth,randinkheight = random.randint(1, max(1,int(maskshape[1] * 0.10))), random.randint(1, max(1,int(maskshape[0]*0.9)))
      randink = random.randint(0,max(1,maskshape[0]-randinkheight-CFG.size)),random.randint(randnoink[1]+randnoinkwidth,maskshape[1]-CFG.size)
      randinkwidth = min(randinkwidth, maskshape[1]-CFG.size)
      '''

      # TODO: Set mask here and xyxys, THEN TRAIN!!!
      print(randtrial, "SethS train_masks shape, dtype", [(id,t.shape,t.dtype, t.min(), t.max(), t.mean(), t.std()) for (id,t) in train_masks.items()])
      print("SELECTED image for training", list(train_masks.keys())[0], "maskmax", maskmax)
      print("randink", randink, randinkheight, randinkwidth, "randnoink", randnoink, randnoinkheight, randnoinkwidth)
      print("randink", randink[0], ":", randink[0]+randinkheight, ",", randink[1], ":", randink[1] + randinkwidth, "randnoink", randnoink, randnoinkheight, randnoinkwidth)
      tmidx = np.zeros_like(train_masks[whichmask])
      tmnidx = np.zeros_like(train_masks[whichmask])
      #tmidx[randink[0]:randink[0]+randinkheight-CFG.size, randink[1]:randink[1]+randinkwidth-CFG.size] = maskmax #1.0 # TODO: Make larger rectangular area, including for noink!
      tmidx[randink[0]:randnoink[0]+randnoinkheight-CFG.size, randink[1]:randnoink[1]+randnoinkwidth-CFG.size] = maskmax #1.0 # TODO: Make larger rectangular area, including for noink!

      print("tmidx", randink[0],randink[0]+randinkheight-CFG.size, "x:x+w", randink[1],randink[1]+randinkwidth-CFG.size, "max", maskmax) #1.0 # TODO: Make larger rectangular area, including for noink!
      train_masks[whichmask][randink[0]:randink[0]+randinkheight, randink[1]:randink[1]+randinkwidth] = maskmax #1.0 # TODO: Make larger rectangular area, including for noink!
      train_noinkmasks[whichmask][randnoink[0]:randnoink[0]+randnoinkheight, randnoink[1]:randnoink[1]+randnoinkwidth] = maskmax #1.0 # TODO: Make larger rectangular area, including for noink!
      #tmnidx[randnoink[0]:randnoink[0]+randnoinkheight-CFG.size, randnoink[1]:randnoink[1]+randnoinkwidth] = maskmax #1.0 # TODO: Make larger rectangular area, including for noink!
      #trainlen=int(1280000/(np.count_nonzero(train_masks[whichmask]) + np.count_nonzero(train_noinkmasks[whichmask]))) #00
      print("ink samples", np.count_nonzero(tmidx), "non-ink samples", np.count_nonzero(tmnidx)) #00
      trainlen=int(1280000/(np.count_nonzero(tmidx) + np.count_nonzero(tmnidx))) #00
      #train_xyxys = [(randink[1],randink[0],randink[1]+CFG.size,randink[0]+CFG.size),]*trainlen # TODO SethS should make training set bigger?
      #train_xyxys += [(randnoink[1],randnoink[0],randnoink[1]+CFG.size,randnoink[0]+CFG.size),]*trainlen # TODO SethS should make training set bigger?
      #train_xyxys = [(c[1], c[0], c[1]+CFG.size, c[0]+CFG.size) for c in np.argwhere(train_masks[whichmask] > 0).tolist() if c[1]+CFG.size < randink[1]+randinkwidth and c[0]+CFG.size < randink[0] + randinkheight]*trainlen # TODO SethS should make training set bigger?
      #train_xyxys = [(c[1], c[0], c[1]+CFG.size, c[0]+CFG.size) for c in np.argwhere(train_masks[whichmask] > 0).tolist() if c[1]+CFG.size < randink[1]+randinkwidth and c[0]+CFG.size < randink[0] + randinkheight]*trainlen # TODO SethS should make training set bigger?
      train_xyxys = [(c[1], c[0], c[1]+CFG.size, c[0]+CFG.size) for c in np.argwhere(tmidx).tolist()]*trainlen # TODO SethS should make training set bigger?
      #train_xyxys += [(c[1], c[0], c[1]+CFG.size, c[0]+CFG.size) for c in np.argwhere(train_noinkmasks[whichmask] > 0).tolist()]*trainlen # TODO SethS should make training set bigger?
      train_xyxys += [(c[1], c[0], c[1]+CFG.size, c[0]+CFG.size) for c in np.argwhere(tmnidx > 0).tolist()]*trainlen # TODO SethS should make training set bigger?
      train_ids = [whichmask,] * len(train_xyxys)
      print("Len train_ids, xyxys:", len(train_ids), len(train_xyxys), set(train_ids), len(set(train_xyxys)), train_xyxys[:4])
      print("Len valid_ids, xyxys:", len(valid_ids), len(valid_xyxys), valid_xyxys[:3])
      #print("Got all validation scroll ID shapes for prediction and logging")
      wandb_logger = WandbLogger(project="vesuvius",name=run_slug+f'{enc}_finetune')
      multiplicative = lambda epoch: 0.9
      #train_xyxys, train_ids = train_xyxys[:100000], train_ids[:100000]
      #train_xyxys, train_ids = train_xyxys[:100000], train_ids[:100000]
      #print("Training images:", len(train_images))
      #print("len valid_xyxys", valid_xyxys, valid_xyxys.__class__.__name__)
      #valid_xyxys = np.stack(valid_xyxys) #[:100000])

      #print("stacked valid_xyxys", valid_xyxys.shape)
      #print("Creating datasets")
      train_dataset = CustomDataset(
        train_images, CFG, labels=train_masks, xyxys=train_xyxys, ids=train_ids, transform=get_transforms(data='train', cfg=CFG))
      valid_dataset = CustomDatasetTest(
        valid_images, CFG,xyxys=valid_xyxys, labels=valid_masks, ids=valid_ids, transform=get_transforms(data='valid', cfg=CFG))

      train_loader = DataLoader(train_dataset,
                                batch_size=CFG.train_batch_size,
                                shuffle=True,
                                num_workers=CFG.num_workers, pin_memory=True, drop_last=True, collate_fn=custom_collate_fn
                                )
      valid_loader = DataLoader(valid_dataset,
                                batch_size=CFG.valid_batch_size,
                                shuffle=False,
                                num_workers=CFG.num_workers, pin_memory=True, drop_last=True, collate_fn=custom_collate_fn)
      #print("Done creating datasets")
      if args.mode == "dataonly":
        exit()

      #print("Creating WandB logger")
      non_local=False
      #if "pygo" in args.model:
      #  from pygoflat import 
      '''
      if len(args.load) == 0:
        model=RegressionPLModel(enc='i3d',pred_shape=pred_shape,size=CFG.size, backbone=args.model)
      else:
        model=RegressionPLModel.load_from_checkpoint(args.load, enc='i3d',pred_shape=pred_shape,size=CFG.size, name=run_slug, backbone=args.model)
      '''
      #model=RegressionPLModel(enc='i3d',pred_shape=pred_shape,size=CFG.size, train_dataset=train_dataset, backbone=args.model, wandb_logger=wandb_logger, name=name, val_masks=valid_masks, complexity=args.complexity)
      #print("Creating model")
      model = RegressionPLModel(enc='i3d',size=CFG.size, train_dataset=train_dataset, backbone=args.model, wandb_logger=wandb_logger, name=name, val_masks=valid_masks, complexity=args.complexity, train_loader=train_loader, valid_loader=valid_loader, cfg=CFG, is_main=trainer.is_global_zero, train_masks=train_masks, train_noinkmasks = train_noinkmasks)
      if len(args.load) > 0:
        #model=RegressionPLModel.load_from_checkpoint(args.load, backbone=args.model, wandb_logger=wandb_logger, enc="i3d", pred_shape=pred_shape, size=CFG.size, train_dataset=train_dataset, name=name, val_masks=valid_masks, complexity=args.complexity)
        model=RegressionPLModel.load_from_checkpoint(args.load, backbone=args.model, wandb_logger=wandb_logger, enc="i3d", size=CFG.size, train_dataset=train_dataset, name=name, val_masks=valid_masks, complexity=args.complexity, train_loader=train_loader, valid_loader=valid_loader, cfg=CFG)
      #model.train_dataloaders = train_loader
      #model.valid_dataloaders = valid_loader
      #print('FOLD : ',fold)
      wandb_logger.watch(model, log="all", log_freq=100)
      multiplicative = lambda epoch: 0.9
      model.valid_dataloader = valid_loader
      model.training_dataloader = train_loader
      #print("Creating trainer")
      #tuner = pl.tuner.Tuner(trainer)
      #tuner.scale_batch_size(model, mode='binsearch')
      #print("Beginning training")
      trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader) #, auto_scale_batch_size='binsearch')

      wandb.finish()

