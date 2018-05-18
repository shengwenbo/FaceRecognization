import scipy.io as sio
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
from operator import itemgetter

def load_pie(path="./data/pie"):
    file_list = []
    if os.path.isdir(path):
        file_list = [os.path.join(path,f) for f in os.listdir(path)]
    else:
        file_list = [path]

    data = pd.DataFrame(columns=["fea","gnd","isTest"])
    for file in file_list:
        if not file.endswith(".mat"):
            continue
        content = sio.loadmat(file)
        content = {key:list(content[key]) for key in ["fea","gnd","isTest"]}
        data = data.append(pd.DataFrame(content), ignore_index=True)
    data["gnd"] = data.apply(lambda x: x[1][0],axis=1)
    data["isTest"] = data.apply(lambda x: float(x[2][0]), axis=1)
    data["dimX"] = 64
    data["dimY"] = 64
    return data

def load_pie_jpg(path="./data/pie/images"):
    file_list = []
    if os.path.isdir(path):
        file_list = [os.path.join(path,f) for f in os.listdir(path)]
    else:
        file_list = [path]

    data = pd.DataFrame(columns=["fea","gnd","isTest"])
    for file in file_list:
        if not file.endswith(".jpg"):
            continue
        image = Image.open(file)
        image.convert("L")
        image = np.array(image)
        image = list(image.reshape(1,-1))
        label,data_set,_ = os.path.basename(file).split("_", 2)
        data_set = 1.0 if data_set=="test" else 0.0
        label = int(label)
        content = {"fea":image, "gnd":np.array(label), "isTest":np.array(data_set)}
        data = data.append(pd.DataFrame(content), ignore_index=True)
    data["isTest"] = data.apply(lambda x: float(x[2]), axis=1)
    data["dimX"] = 64
    data["dimY"] = 64
    return data

def get_train_data(data):
    train_data = data[data.isTest < 0.5]
    images = train_data["fea"].values
    images = np.array([x.reshape(64,64,1) for x in images], dtype=float)
    labels = train_data["gnd"].values
    return images, labels


def get_test_data(data):
    test_data = data[data.isTest > 0.5]
    images = test_data["fea"].values
    images = np.array([x.reshape(64,64,1) for x in images])
    labels = test_data["gnd"].values
    return images, labels


def get_batch_data(images, labels, batch_size):
    images = tf.cast(images, tf.float32)
    label = tf.cast(labels, tf.float32)
    input_queue = tf.train.slice_input_producer([images, label], shuffle=False)
    image_batch, label_batch = tf.train.batch(input_queue, batch_size=batch_size, num_threads=1, capacity=64)
    return image_batch, label_batch


if __name__ == "__main__":
    data = load_pie()
    print(data)