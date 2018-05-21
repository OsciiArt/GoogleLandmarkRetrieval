import numpy as np
import pandas as pd
import time, os, glob
import cv2


from keras.applications.vgg16 import VGG16
from keras.optimizers import Adam
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

    optimizer = Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model


def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1])
        h = cv2.add(h, hue_shift)
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def randomShiftScaleRotate(image,
                           shift_limit=(-0.0625, 0.0625),
                           scale_limit=(-0.1, 0.1),
                           rotate_limit=(-45, 45), aspect_limit=(0, 0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])  # degree
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))

    return image


def randomHorizontalFlip(image, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)

    return image


def train_generator(train_id, x_train, batch_size, shuffle=True):
    batch_index = 0
    n = train_id.shape[0]
    while 1:
        if batch_index == 0:
            index_array = train_id
            if shuffle:
                index_array = train_id[np.random.permutation(n)]

        current_index = (batch_index * batch_size) % n
        if n >= current_index + batch_size:
            current_batch_size = batch_size
            batch_index += 1
        else:
            current_batch_size = n - current_index
            batch_index = 0

        batch_x = []
        batch_id = index_array[current_index: current_index + current_batch_size]
        batch_y = np.random.randint(0, 4, current_batch_size)
        # print(batch_y)
        for i, id in enumerate(batch_id):
            img = cv2.imread(x_train['path'][id]).astype(np.float32)
            img = randomHueSaturationValue(img,
                                           hue_shift_limit=(-50, 50),
                                           sat_shift_limit=(-5, 5),
                                           val_shift_limit=(-15, 15),
                                           u=0.5)
            img = randomShiftScaleRotate(img,
                                                shift_limit=(-0.2, 0.2),
                                                scale_limit=(-0.2, 0.2),
                                                rotate_limit=(-5, 5),
                                         aspect_limit = (-0.2, 0.2),
                                         u=0.5)
            img = randomHorizontalFlip(img)
            img = img[:,:,::-1]
            if batch_y[i]==1:
                img = img.transpose([1,0,2])
                img = img[::-1]
            elif batch_y[i]==2:
                img = img[::-1, ::-1]
            elif batch_y[i]==3:
                img = img.transpose([1,0,2])
                img = img[:, ::-1]
            batch_x.append(img)
        batch_x = np.array(batch_x, np.float32)
        batch_x = preprocess_input(batch_x)
        batch_y = np.eye(4)[batch_y]

        yield (batch_x, batch_y)


from keras.callbacks import ModelCheckpoint, CSVLogger

def get_callbacks(save_path):
    csv_logger = CSVLogger(save_path[:-5] + '_log.csv', append=True)
    check_path = save_path
    save_checkpoint = ModelCheckpoint(filepath=check_path, monitor='val_loss',
                                      save_best_only=True)
    Callbacks = [csv_logger, save_checkpoint]
    return Callbacks


index_path = "input/index/"
index_list = sorted(glob.glob(index_path + "*")) # 1091756
query_path = "input/query/"
index_list += sorted(glob.glob(query_path + "*")) # 114943
index_list = pd.DataFrame(index_list, columns=['path'])

from sklearn.model_selection import train_test_split

train_id, valid_id = train_test_split(np.arange(index_list.shape[0]), test_size=2048)

batch_size = 128
input_size = 128
epoch = 10
gen = train_generator(train_id, index_list, batch_size)
gen_val = train_generator(valid_id, index_list, batch_size, shuffle=False)

model = get_rotnet(1,128)
weight_path = "model/RotNet.hdf5"
callbacks = get_callbacks(weight_path)
model.fit_generator(generator=gen,
                     epochs=epoch,
                     steps_per_epoch=np.ceil(train_id.shape[0] / batch_size),
                     verbose=1,
                     callbacks=callbacks,
                     validation_data=gen_val,
                     validation_steps=np.ceil(valid_id.shape[0] / batch_size),
                     )