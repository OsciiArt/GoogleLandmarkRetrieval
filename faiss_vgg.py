# Getting some data
import numpy as np
d = 512                           # dimension
nb = 1091756                      # database size
nq = 114943                       # nb of queries
np.random.seed(1234)             # make reproducible
x_base = np.load("additional/vgg_pred.npy")
xb = x_base[:nb]
xq = x_base[nb:]

# Building an index and adding the vectors to it
import faiss                   # make faiss available
index = faiss.IndexFlatL2(d)   # build the index
print(index.is_trained)
index.add(xb)                  # add vectors to the index
print(index.ntotal)

# Searching
k = 100                          # we want to see 4 nearest neighbors
D, I = index.search(xb[:5], k) # sanity check
print(I)
print(D)
D, I = index.search(xq, k)     # actual search
print(I[:5])                   # neighbors of the 5 first queries
print(I[-5:])                  # neighbors of the 5 last queries