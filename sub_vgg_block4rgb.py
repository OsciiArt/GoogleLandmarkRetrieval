import os, glob
import numpy as np
import pandas as pd

I = np.load("additional/I_vgg_block4rgb.npy")
D = np.load("additional/D_vgg_block4rgb.npy")

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
sub.to_csv("output/sub_vgg_block4rgb.csv", index=None)

