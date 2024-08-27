import sys
stride = int(sys.argv[1])
size = int(sys.argv[2])
tile_size = int(sys.argv[3])

w,h = int(sys.argv[4]), int(sys.argv[5])
import numpy as np

fragment_mask = np.ones((h,w))

xyxys = [(c[1]*stride,c[0]*stride,c[1]*stride+size,c[0]*stride+size) for c in np.argwhere(fragment_mask[::stride,::stride] > 0).tolist() if c[0] >= 0 and c[1] >= 0 and c[0]*stride+tile_size < h and c[1]*stride+tile_size < w]

print("xyxys:", xyxys)
print("len:", len(xyxys))
