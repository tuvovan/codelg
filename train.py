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
import math
from tensorflow.keras import backend as K


os.environ["CUDA_VISIBLE_DEVICES"]="0"

BATCH_SIZE = 32
img_size = 256
#weights = None
weights = 'imagenet'
learning_rate = 1e-5
EPOCHS = 40
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

def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = tf.keras.layers.Conv2D(filters, size, strides=strides, padding=padding)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if activation == True:
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    return x

def residual_block(blockInput, num_filters=16):
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(blockInput)
    x = tf.keras.layers.BatchNormalization()(x)
    blockInput = tf.keras.layers.BatchNormalization()(blockInput)
    x = convolution_block(x, num_filters, (3,3) )
    x = convolution_block(x, num_filters, (3,3), activation=False)
    x = tf.keras.layers.Add()([x, blockInput])
    return x

def CA(X):
    c = list(X.shape)[-1]
    gap = tf.keras.layers.GlobalAveragePooling2D()(X)
    d = tf.reshape(gap, shape=(-1,1,1,c))
    d1 = tf.keras.layers.ReLU()(tf.keras.layers.Conv2D(filters=c, kernel_size=(1,1), kernel_initializer = 'he_normal')(d))
    d_bid = tf.math.sigmoid(tf.keras.layers.Conv2D(filters=c, kernel_size=(1,1), kernel_initializer = 'he_normal')(d1))

    return X*d_bid

def SA(X):
    gap = tf.reduce_max(X, axis=-1)
    gap = tf.expand_dims(gap, axis=-1)
    gmp = tf.reduce_mean(X, axis=-1)
    gmp = tf.expand_dims(gmp, axis=-1)
    
    ff = tf.keras.layers.concatenate([gap, gmp])

    f = tf.keras.layers.Conv2D(1, kernel_size=(1,1), kernel_initializer = 'he_normal')(ff)
    f = tf.math.sigmoid(f)

    return X * f

def dual_attention(X):
    c = list(X.shape)[-1]
    M = tf.keras.layers.Conv2D(c, kernel_size=(3,3), padding='same', kernel_initializer = 'he_normal')(X)
    M = tf.keras.layers.ReLU()(M)
    M = tf.keras.layers.Conv2D(c, kernel_size=(3,3), padding='same', kernel_initializer = 'he_normal')(M)

    ca = CA(M)
    sa = SA(M)

    concat = tf.keras.layers.concatenate([ca, sa])

    concat2 = tf.keras.layers.Conv2D(c, kernel_size=(1,1), kernel_initializer = 'he_normal')(concat)

    return tf.keras.layers.Add()([X, concat2])

# Define our custom metric
def PSNR(y_true, y_pred):
    y_true, y_pred = y_true.numpy(), y_pred.numpy()
    max_pixel = 1.0
    return 10.0 * math.log10((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true)))) 

# Define our custom loss function
def charbonnier(y_true, y_pred):
    epsilon = 1e-3
    error = y_true - y_pred
    p = K.sqrt(K.square(error) + K.square(epsilon))
    return K.mean(p)

def total_variation_loss(x):
    a = tf.square(
        x[:, : 256 - 1, : 256 - 1, :] - x[:, 1:, : 256 - 1, :]
    )
    b = tf.square(
        x[:, : 256 - 1, : 256 - 1, :] - x[:, : 256 - 1, 1:, :]
    )
    return tf.reduce_sum(tf.pow(a + b, 1.25))

def charbonnier_tv_loss(y_true, y_pred):
    loss = charbonnier(y_true, y_pred) + 1e-6 * total_variation_loss(y_pred)
    return loss

def ResUNet101V2(input_shape=(None, None, 3), dropout_rate=0.1, start_neurons = 16):
    backbone = tf.keras.applications.ResNet101V2(weights=weights, include_top=False, input_shape=input_shape)
    input_layer = backbone.input
    
    conv4 = backbone.layers[122].output
    conv4 = tf.keras.layers.LeakyReLU(alpha=0.1)(conv4)
    pool4 = tf.keras.layers.MaxPooling2D((2, 2))(conv4)
    pool4 = tf.keras.layers.Dropout(dropout_rate)(pool4)
    
    convm = tf.keras.layers.Conv2D(start_neurons * 32, (3, 3), activation=None, padding="same")(pool4)
    convm = residual_block(convm,start_neurons * 32)
    convm = residual_block(convm,start_neurons * 32)
    convm = tf.keras.layers.LeakyReLU(alpha=0.1)(convm)
    
    deconv4 = tf.keras.layers.Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = tf.keras.layers.concatenate([deconv4, conv4])
    uconv4 = tf.keras.layers.Dropout(dropout_rate)(uconv4)
    
    uconv4 = tf.keras.layers.Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(uconv4)
    uconv4 = residual_block(uconv4,start_neurons * 16)
    uconv4 = residual_block(uconv4,start_neurons * 16)
    uconv4 = tf.keras.layers.LeakyReLU(alpha=0.1)(uconv4)
    uconv4 = dual_attention(uconv4)
    
    deconv3 = tf.keras.layers.Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(uconv4)
    conv3 = backbone.layers[76].output
    uconv3 = tf.keras.layers.concatenate([deconv3, conv3])    
    uconv3 = tf.keras.layers.Dropout(dropout_rate)(uconv3)
    
    uconv3 = tf.keras.layers.Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3,start_neurons * 8)
    uconv3 = residual_block(uconv3,start_neurons * 8)
    uconv3 = tf.keras.layers.LeakyReLU(alpha=0.1)(uconv3)
    uconv3 = dual_attention(uconv3)

    deconv2 = tf.keras.layers.Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv3)
    conv2 = backbone.layers[30].output
    uconv2 = tf.keras.layers.concatenate([deconv2, conv2])
        
    uconv2 = tf.keras.layers.Dropout(0.1)(uconv2)
    uconv2 = tf.keras.layers.Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2,start_neurons * 4)
    uconv2 = residual_block(uconv2,start_neurons * 4)
    uconv2 = tf.keras.layers.LeakyReLU(alpha=0.1)(uconv2)
    uconv2 = dual_attention(uconv2)
    
    deconv1 = tf.keras.layers.Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv2)
    conv1 = backbone.layers[2].output
    uconv1 = tf.keras.layers.concatenate([deconv1, conv1])
    
    uconv1 = tf.keras.layers.Dropout(0.1)(uconv1)
    uconv1 = tf.keras.layers.Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1,start_neurons * 2)
    uconv1 = residual_block(uconv1,start_neurons * 2)
    uconv1 = tf.keras.layers.LeakyReLU(alpha=0.1)(uconv1)
    uconv1 = dual_attention(uconv1)
    
    uconv0 = tf.keras.layers.Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv1)   
    uconv0 = tf.keras.layers.Dropout(0.1)(uconv0)
    uconv0 = tf.keras.layers.Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv0)
    uconv0 = residual_block(uconv0,start_neurons * 1)
    uconv0 = residual_block(uconv0,start_neurons * 1)
    uconv0 = tf.keras.layers.LeakyReLU(alpha=0.1)(uconv0)
    uconv1 = dual_attention(uconv1)
    
    uconv0 = tf.keras.layers.Dropout(dropout_rate/2)(uconv0)
    output_layer = tf.keras.layers.Conv2D(3, (1,1), padding="same", activation="sigmoid")(uconv0)    
    
    model = tf.keras.models.Model(input_layer, output_layer)

    return model

optimizer = tf.keras.optimizers.Adam(learning_rate)
model = ResUNet101V2(input_shape=(img_size, img_size, 3),dropout_rate=dropout_rate)
model.compile(loss=charbonnier_tv_loss, optimizer=optimizer)

callbacks_list = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath = 'models/baseline_model_charnonnier_tv.h5',
        monitor='val_loss',
        save_best_only=True
    )
]

hist = model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset, callbacks=callbacks_list)
