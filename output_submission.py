import os, glob
import numpy as np
import pandas as pd

# feature_layer = "block3_conv3"
# feature_layer = "block4_conv3"
feature_layer = "block5_conv3"
model_name = "vgg" # or rotnet

d = 512                           # dimension
nb = 1091756                      # database size
nq = 114943                       # nb of queries
x_base = np.load("output/{}_feature_{}.npy".format(model_name, feature_layer))
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
# print(I[:5])                   # neighbors of the 5 first queries
# print(I[-5:])                  # neighbors of the 5 last queries

np.save("output/I_{}.npy".format(feature_layer), I)
np.save("output/D_{}.npy".format(feature_layer), D)

### make submission
index_path = "input/index/"
index_list = sorted(glob.glob(index_path + "*")) # 1091756
index_list = pd.DataFrame(index_list, columns=['id'])
index_list['id'] = index_list['id'].apply(lambda x: os.path.basename(x)[:-4])
index_list = np.array(index_list['id'])
query_path = "input/query/"
query_list = sorted(glob.glob(query_path + "*")) # 114943

sub = pd.DataFrame(query_list, columns=['id'])
sub['id'] = sub['id'].apply(lambda x: os.path.basename(x)[:-4])

images_list =  index_list[I]
images_list = images_list + " "
images_list = np.sum(images_list, axis=1)

sub['images'] = images_list
sub2 = pd.read_csv("input/sample_submission.csv")
sub2['images'] = ""
sub = pd.concat([sub, sub2])
sub = sub.drop_duplicates(['id'])
sub.to_csv("output/sub_{}_{}.csv".format(model_name, feature_layer), index=None)