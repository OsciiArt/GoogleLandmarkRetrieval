

# Getting some data
import numpy as np
import time

starttime = time.time()


basename = "vgg_block5"
d = 512                           # dimension
nb = 1091756                      # database size
nq = 114943                       # nb of queries
np.random.seed(1234)             # make reproducible
x_base = np.load("additional/vgg_pred_block5rgb.npy")
print(x_base.shape)
# x_base = pred_valid
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

np.save("additional/I_{}.npy".format(basename), I)
np.save("additional/D_{}.npy".format(basename), D)

import os, glob
import numpy as np
import pandas as pd

# I = np.load("additional/I_vgg_block4rgb.npy")
# D = np.load("additional/D_vgg_block4rgb.npy")

index_path = "input/train/"
# index_list = sorted(next(os.walk(index_path))[2])
index_list = sorted(glob.glob(index_path + "*")) # 1091756
index_list = pd.DataFrame(index_list, columns=['id'])
index_list['id'] = index_list['id'].apply(lambda x: os.path.basename(x)[:-4])
index_list = np.array(index_list['id'])
print(index_list[:10])
print(len(index_list))
query_path = "input/test/"
query_list = sorted(glob.glob(query_path + "*")) # 114943

sub = pd.DataFrame(query_list, columns=['id'])
sub['id'] = sub['id'].apply(lambda x: os.path.basename(x)[:-4])
print(sub.head())

images_list =  index_list[I]
print(images_list[:5])
images_list = images_list + " "
print(images_list[:5])
print(images_list.shape)
images_list = np.sum(images_list, axis=1)
print(images_list[:5])

# def images100(x):
#
#     for i in range(100):


sub['images'] = images_list
sub2 = pd.read_csv("input/sample_submission.csv")
sub2['images'] = ""
sub = pd.concat([sub, sub2])
sub = sub.drop_duplicates(['id'])
print(sub.shape)
sub.to_csv("output/sub_{}.csv".format(basename), index=None)
