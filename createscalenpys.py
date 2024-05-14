import sys, os
import numpy as np
from skimage.measure import block_reduce
import cv2
from tqdm import tqdm

# TODO: Put into dataloaders!
imgs = []
print("Loading...", sys.argv[1])
for i in tqdm(range(999)):
  p = os.path.join(sys.argv[1], "layers", f"{i:02}.tif")
  if os.path.exists(p):
    print(p)
    img = cv2.imread(p,0)
    imgs.append(img)
  else:
    break
print("images", img.shape, len(imgs))
print("Stacking")
imgs = np.stack(imgs, axis=2)
print("imgs.shape")
imgs=np.clip(imgs,0,200)
print("Saving scaled image")
np.save(sys.argv[1]+".npys", imgs)
oimgs = imgs
for scale in tqdm([2,4,8,16,32,64,128,256,512,1024]):
  print("Scaling")
  imgs = (block_reduce(oimgs, block_size=(scale,scale,1), func=np.mean, cval=np.mean(imgs))+0.5).astype(np.uint8)
  print("Saving")
  np.save(sys.argv[1]+"_"+str(scale)+".npys", imgs)
print("Done!")
