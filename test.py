import pandas as pd
import numpy as np
import tensorflow as tf
import os
from tqdm import tqdm
from glob import glob
import gc
import cv2
import math
import zipfile
import math
from model import *
from tensorflow.keras import backend as K

os.environ["CUDA_VISIBLE_DEVICES"]="1"

BATCH_SIZE = 4
img_size = 1024
#weights = None
weights = 'imagenet'
learning_rate = 1e-5
EPOCHS = 5
dropout_rate = 0.1

# Define our custom loss function
def charbonnier(y_true, y_pred):
    epsilon = 1e-3
    error = y_true - y_pred
    p = K.sqrt(K.square(error) + K.square(epsilon))
    return K.mean(p)

model = ResUNet101V2(input_shape=(img_size, img_size, 3),dropout_rate=dropout_rate)
# model = tf.keras.models.load_model('models/baseline_model_charnonnier.h5')
model.load_weights('models/baseline_model_charnonnier.h5')

def predict(img_paths, stride=16, batch_size=128):
    results = []
    for img_path in img_paths:
        img = cv2.imread(img_path)
        img = img.astype(np.float32)/255
        crop = []
        position = []
        batch_count = 0

        result_img = np.zeros_like(img)
        voting_mask = np.zeros_like(img)

        for top in tqdm(range(0, img.shape[0], stride)):
            for left in range(0, img.shape[1], stride):
                piece = np.zeros([img_size, img_size, 3], np.float32)
                temp = img[top:top+img_size, left:left+img_size, :]
                piece[:temp.shape[0], :temp.shape[1], :] = temp
                crop.append(piece)
                position.append([top, left])
                batch_count += 1
                if batch_count == batch_size:
                    crop = np.array(crop)
                    pred = model(crop)*255
                    crop = []
                    batch_count = 0
                    for num, (t, l) in enumerate(position):
                        piece = pred[num]
                        h, w, c = result_img[t:t+img_size, l:l+img_size, :].shape
                        result_img[t:t+img_size, l:l+img_size, :] += piece[:h, :w]
                        voting_mask[t:t+img_size, l:l+img_size, :] += 1
                    position = []
        
        result_img = result_img/voting_mask
        result_img = result_img.astype(np.uint8)
        results.append(result_img)
        
    return results

def rmse_score(true, pred):
    score = math.sqrt(np.mean((true-pred)**2))
    return score

def psnr_score(true, pred, pixel_max):
    score = 20*np.log10(pixel_max/rmse_score(true, pred))
    return score

test_csv = pd.read_csv('data/test.csv')
test_input_files = 'data/test_input_img/'+test_csv['input_img']

test_result = predict(test_input_files, 32, 2)

def make_submission(result):
    os.makedirs('submission', exist_ok=True)
    os.chdir("./submission/")
    sub_imgs = []
    for i, img in enumerate(result):
        path = f'test_{20000+i}.png'
        cv2.imwrite(path, img)
        sub_imgs.append(path)
    submission = zipfile.ZipFile("submission.zip", 'w')
    for path in sub_imgs:
        submission.write(path)
    submission.close()

make_submission(test_result)

