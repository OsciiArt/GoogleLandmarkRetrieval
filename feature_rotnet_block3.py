import numpy as np
import pandas as pd
import time, os, glob
import cv2

# from sklearn.model_selection import train_test_split

# from sklearn.metrics import log_loss
# from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
# from os.path import join as opj
# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import pylab
#
# # extract features by VGG
#
# #
from keras.applications.vgg16 import VGG16
# from keras.applications.densenet import DenseNet121
from keras.optimizers import SGD, Adam
from keras.layers import GlobalAveragePooling2D, Dense
from keras import Model
from keras.applications.imagenet_utils import preprocess_input

def get_rotnet(num_class, input_size):
    base_model = VGG16(weights='imagenet', include_top=False,
                       input_shape=[input_size,input_size,3], classes=num_class)
    x = base_model.get_layer('block5_pool').output
    x = GlobalAveragePooling2D()(x)
    x = Dense(4, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=x)

    # sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    # optimizer = Adam(lr=0.0001)
    # model.compile(loss='categorical_crossentropy',
    #               optimizer=optimizer,
    #               metrics=['accuracy'])

    optimizer = Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model

def get_model(base_model):
    x = base_model.get_layer('block3_conv3').output
    x = GlobalAveragePooling2D()(x)

    model = Model(inputs=base_model.input, outputs=x)

    # sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    # optimizer = Adam(lr=0.0001)
    # model.compile(loss='categorical_crossentropy',
    #               optimizer=optimizer,
    #               metrics=['accuracy'])

    optimizer = Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model

def test_generator(x_train, batch_size, input_size, shuffle=False):
    batch_index = 0
    n = x_train.shape[0]
    while 1:
        if batch_index == 0:
            index_array = np.arange(n)
            if shuffle:
                index_array = np.random.permutation(n)

        current_index = (batch_index * batch_size) % n
        if n >= current_index + batch_size:
            current_batch_size = batch_size
            batch_index += 1
        else:
            current_batch_size = n - current_index
            batch_index = 0

        batch_x = []
        batch_id = index_array[current_index: current_index + current_batch_size]
        for id in batch_id:
            img = cv2.imread(x_train['path'][id]).astype(np.float32)
            # img = cv2.resize(img, (input_size, input_size))
            # img = randomHueSaturationValue(img,
            #                                hue_shift_limit=(-5, 5),
            #                                sat_shift_limit=(-1, 1),
            #                                val_shift_limit=(-2, 2),
            #                                u=0.5)
            # img = randomShiftScaleRotate(img,
            #                                     shift_limit=(-0.2, 0.2),
            #                                     scale_limit=(-0.2, 0.2),
            #                                     rotate_limit=(-30, 30),
            #                              aspect_limit = (-0.2, 0.2),
            #                              u=0.5)
            # img = randomHorizontalFlip(img)
            img = img[:,:,::-1]
            img = preprocess_input(img)
            batch_x.append(img)
        batch_x = np.array(batch_x, np.float32)

        yield batch_x

#
basename = "rotnet_block3"
index_path = "input/train/"
# index_list = sorted(next(os.walk(index_path))[2])
index_list = sorted(glob.glob(index_path + "*")) # 1091756
print(len(index_list))
query_path = "input/test/"
index_list += sorted(glob.glob(query_path + "*")) # 114943
index_list = pd.DataFrame(index_list, columns=['path'])
print(len(index_list))
print(index_list[:5])
print(index_list[-5:])

# x = np.zeros([index_list.shape[0], 1024], dtype=np.float32)
# np.save("additional/zeros.npy", x)

rotnet = get_rotnet(1,128)
weight_path = "model/RotNet.hdf5"
rotnet.load_weights(weight_path)
model = get_model(rotnet)

batch_size = 128
input_size = 128
gen_test = test_generator(index_list, batch_size, input_size)
pred_valid = model.predict_generator(generator=gen_test,
                                     # steps=100,
                                     steps=np.ceil(index_list.shape[0] / batch_size),
                                     verbose=1)
print("pred_Valid", pred_valid.shape)
np.save("additional/feature_{}.npy".format(basename), pred_valid)

# # Getting some data
# import numpy as np
# import time
#
# starttime = time.time()
#
# d = 512                           # dimension
# nb = 1091756                      # database size
# nq = 114943                       # nb of queries
# np.random.seed(1234)             # make reproducible
# # x_base = np.load("additional/feature_{}.npy".format(basename))
# x_base = pred_valid
# xb = x_base[:nb]
# xq = x_base[nb:]
#
# import faiss
# res = faiss.StandardGpuResources()  # use a single GPU
#
# # build a flat (CPU) index
# index_flat = faiss.IndexFlatL2(d)
# # make it into a gpu index
# gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
#
# gpu_index_flat.add(xb)         # add vectors to the index
# print(gpu_index_flat.ntotal)
#
# k = 100                          # we want to see 4 nearest neighbors
# D, I = gpu_index_flat.search(xq, k)  # actual search
# print(I[:5])                   # neighbors of the 5 first queries
# print(I[-5:])                  # neighbors of the 5 last queries
# print("done", time.time() - starttime) # done 18.42986536026001
#
# np.save("additional/I_{}.npy".format(basename), I)
# np.save("additional/D_{}.npy".format(basename), D)
#
# import os, glob
# import numpy as np
# import pandas as pd
#
# # I = np.load("additional/I_vgg_block4rgb.npy")
# # D = np.load("additional/D_vgg_block4rgb.npy")
#
# index_path = "input/train/"
# # index_list = sorted(next(os.walk(index_path))[2])
# index_list = sorted(glob.glob(index_path + "*")) # 1091756
# index_list = pd.DataFrame(index_list, columns=['id'])
# index_list['id'] = index_list['id'].apply(lambda x: os.path.basename(x)[:-4])
# index_list = np.array(index_list['id'])
# print(index_list[:10])
# print(len(index_list))
# query_path = "input/test/"
# query_list = sorted(glob.glob(query_path + "*")) # 114943
#
# sub = pd.DataFrame(query_list, columns=['id'])
# sub['id'] = sub['id'].apply(lambda x: os.path.basename(x)[:-4])
# print(sub.head())
#
# images_list =  index_list[I]
# print(images_list[:5])
# images_list = images_list + " "
# print(images_list[:5])
# print(images_list.shape)
# images_list = np.sum(images_list, axis=1)
# print(images_list[:5])
#
# # def images100(x):
# #
# #     for i in range(100):
#
#
# sub['images'] = images_list
# sub2 = pd.read_csv("input/sample_submission.csv")
# sub2['images'] = ""
# sub = pd.concat([sub, sub2])
# sub = sub.drop_duplicates(['id'])
# print(sub.shape)
# sub.to_csv("output/sub_{}.csv".format(basename), index=None)
