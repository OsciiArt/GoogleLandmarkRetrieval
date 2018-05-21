# Getting some data
import numpy as np
import time

starttime = time.time()

d = 512                           # dimension
nb = 1091756                      # database size
nq = 114943                       # nb of queries
np.random.seed(1234)             # make reproducible
x_base = np.load("additional/vgg_pred_block4.npy")
xb = x_base[:nb]
xq = x_base[nb:]

import faiss
res = faiss.StandardGpuResources()  # use a single GPU

# build a flat (CPU) index
index_flat = faiss.IndexFlatL2(d)
# make it into a gpu index
gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)

gpu_index_flat.add(xb)         # add vectors to the index
print(gpu_index_flat.ntotal)

k = 100                          # we want to see 4 nearest neighbors
D, I = gpu_index_flat.search(xq, k)  # actual search
print(I[:5])                   # neighbors of the 5 first queries
print(I[-5:])                  # neighbors of the 5 last queries
print("done", time.time() - starttime) # done 18.42986536026001

np.save("additional/I_vgg_block4.npy", I)
np.save("additional/D_vgg_block4.npy", D)

import os
