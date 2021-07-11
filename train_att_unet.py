# import pandas as pd
import numpy as np
import tensorflow as tf
import os
# from tqdm import tqdm
from glob import glob
import gc
# import cv2
# import matplotlib.pyplot as plt
from sklearn.utils import shuffle
# import math
# import zipfile


os.environ["CUDA_VISIBLE_DEVICES"]="0"

BATCH_SIZE = 32
img_size = 256
#weights = None
weights = 'imagenet'
learning_rate = 1e-5
EPOCHS = 5
dropout_rate = 0.1

train_inp_files = glob(f'train_input_img_{img_size}/*.npy')
train_targ_files = glob(f'train_label_img_{img_size}/*.npy')

val_inp_files = glob(f'val_input_img_{img_size}/*.npy')
val_targ_files = glob(f'val_label_img_{img_size}/*.npy')

train_inp_files, train_targ_files = shuffle(train_inp_files, train_targ_files, random_state=42)

def train_map_func(inp_path, targ_path):
    inp = np.load(inp_path)
    inp = inp.astype(np.float32)/255
    targ = np.load(targ_path)
    targ = targ.astype(np.float32)/255
    inp, targ = augmentation(inp, targ)
    
    return inp, targ

def val_map_func(inp_path, targ_path):
    inp = np.load(inp_path)
    inp = inp.astype(np.float32)/255
    targ = np.load(targ_path)
    targ = targ.astype(np.float32)/255
    return inp, targ

def augmentation(inp, targ):
    inp, targ = random_rot(inp, targ)
    inp, targ = random_flip(inp, targ)
    
    return inp, targ

def random_rot(inp, targ):
    k = np.random.randint(4)
    inp = np.rot90(inp, k)
    targ = np.rot90(targ, k)
    
    return inp, targ

def random_flip(inp, targ):
    f = np.random.randint(2)
    if f == 0:
        inp = np.fliplr(inp)
        targ = np.fliplr(targ)
        
    return inp, targ

train_dataset = tf.data.Dataset.from_tensor_slices((train_inp_files, train_targ_files))
train_dataset = train_dataset.map(lambda item1, item2: tf.numpy_function(train_map_func, [item1, item2], [tf.float32, tf.float32]), num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((val_inp_files, val_targ_files))
val_dataset = val_dataset.map(lambda item1, item2: tf.numpy_function(val_map_func, [item1, item2], [tf.float32, tf.float32]), num_parallel_calls=tf.data.experimental.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE)
val_dataset = val_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

from model import MIRNet
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Concatenate, Conv2D, Input

optimizer = tf.keras.optimizers.Adam(learning_rate)
mir_x = MIRNet(16, 1, 2)
x = Input(shape=(256, 256, 3))
out = mir_x.main_model(x)
model = Model(inputs=x, outputs=out)
model.compile(loss='mae', optimizer=optimizer)

callbacks_list = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath = 'models/baseline_model.h5',
        monitor='val_loss',
        save_best_only=True
    )
]

hist = model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset, callbacks=callbacks_list)
