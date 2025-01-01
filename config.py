import os
import torch
import random
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CFG:
    # ============== comp exp name =============
    comp_name = 'vesuvius'

    # comp_dir_path = './'
    comp_dir_path = './' #'/content/gdrive/MyDrive/vesuvius_model/training'
    comp_folder_name = './' #'/content/gdrive/MyDrive/vesuvius_model/training'
    # comp_dataset_path = f'{comp_dir_path}datasets/{comp_folder_name}/'
    comp_dataset_path = './' #f'/content/gdrive/MyDrive/vesuvius_model/training'
    basepath = os.path.join(os.path.dirname(os.getcwd()), "Vesuvius") ##"/media/seth/WDC-4x1TB-Raid-10GBps/Vesuvius/"
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
    start_idx = 15
    end_idx = 45
    scale = 1
    size = 64
    tile_size = 64 #256
    stride = tile_size // 8
    valid_size = size
    valid_tile_size = valid_size
    valid_stride = valid_size // 2

    train_batch_size = 2 * 2 * 8 * 10 # 32
    train_batch_size = 2 * 2 * 10 # 32
    valid_batch_size = train_batch_size
    #valid_batch_size = 256
    use_amp = True

    scheduler = 'GradualWarmupSchedulerV2'
    # scheduler = 'CosineAnnealingLR'
    epochs = 30 # 30

    # adamW warmupあり
    warmup_factor = 10
    # lr = 1e-4 / warmup_factor
    # lr = 1e-4 / warmup_factor
    lr = 2e-5
    lr = 5e-5 # SethS
    # ============== fold =============
    #valid_id = '20230820203112'
    valid_id = '20231210132040'

    # objective_cv = 'binary'  # 'binary', 'multiclass', 'regression'
    metric_direction = 'maximize'  # maximize, 'minimize'
    # metrics = 'dice_coef'

    # ============== fixed =============
    pretrained = True
    inf_weight = 'best'  # 'best'

    min_lr = 1e-6
    weight_decay = 1e-6
    min_lr = 5e-6
    weight_decay = 6e-6
    max_grad_norm = 100

    print_freq = 50
    num_workers = 5

    seed = 0

    # ============== set dataset path =============
    #print('set dataset path')

    outputs_path = f'./outputs' #/content/gdrive/MyDrive/vesuvius_model/training/outputs'

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
        #A.Resize(size, size), # SethS TODO This was my bug on 6/8 that had been bugging me for quite a while!!
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # A.RandomRotate90(p=0.6),

        A.RandomBrightnessContrast(p=0.75),
        A.Equalize(p=0.5), # SethS 9/12/2024
        A.ShiftScaleRotate(rotate_limit=360,shift_limit=0.15,scale_limit=0.01,p=0.75),
        A.OneOf([
                A.GaussNoise(var_limit=[10, 50]),
                A.GaussianBlur(),
                A.MotionBlur(),
                ], p=0.4),
        #A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5), # SethS 9/11/2024 TODO # Disabled 9/25/2024
        #A.CoarseDropout(max_holes=2, max_width=int(size * 0.2), max_height=int(size * 0.2), 
        #A.CoarseDropout(max_holes=2, max_width=int(size * 0.25), max_height=int(size * 0.25), 
        #                mask_fill_value=0, p=0.5),
        A.CoarseDropout(max_holes=2, max_width=int(1), max_height=int(1), 
                        mask_fill_value=0, p=0.5),
        # A.Cutout(max_h_size=int(size * 0.6),
        #          max_w_size=int(size * 0.6), num_holes=1, p=1.0),
        #A.Normalize(
        #    mean= [0] * 65, #in_chans,
        #    std= [1] * 65, #in_chans
        #),
        ToTensorV2(transpose_mask=True),
    ]

    valid_aug_list = [
        #A.Resize(size, size), # TODO SethS 6/8/ bug FIXED
        #A.Normalize(
        #    mean= [0] * 65, #in_chans,
        #    std= [1] * 65, #in_chans
        #),
        ToTensorV2(transpose_mask=True),
    ]
    rotate = A.Compose([A.Rotate(90,p=1)])
    #rotate = A.Compose([A.Rotate(8,p=1)])

print("SETHS INSTANTIATED CONFIG, batch size", CFG.train_batch_size, "size", CFG.size)

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
    print("config_init", cfg.seed)
    set_seed(cfg.seed)
    # set_env_name()
    # set_dataset_path(cfg)

    if mode == 'train':
        make_dirs(cfg)
cfg_init(CFG)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

