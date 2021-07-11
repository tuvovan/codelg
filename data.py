import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import cv2

train_csv = pd.read_csv('data/train.csv')
test_csv = pd.read_csv('data/test.csv')

train_all_input_files = 'data/train_input_img/'+train_csv['input_img']
train_all_label_files = 'data/train_label_img/'+train_csv['label_img']

train_input_files = train_all_input_files[60:].to_numpy()
train_label_files = train_all_label_files[60:].to_numpy()

val_input_files = train_all_input_files[:60].to_numpy()
val_label_files = train_all_label_files[:60].to_numpy()

BATCH_SIZE = 8
img_size = 512
#weights = None
weights = 'imagenet'
learning_rate = 1e-5
EPOCHS = 5
dropout_rate = 0.1


def cut_img(img_path_list, save_path, stride):
    os.makedirs(f'{save_path}{img_size}', exist_ok=True)
    num = 0
    for path in tqdm(img_path_list):
        img = cv2.imread(path)
        for top in range(0, img.shape[0], stride):
            for left in range(0, img.shape[1], stride):
                piece = np.zeros([img_size, img_size, 3], np.uint8)
                temp = img[top:top+img_size, left:left+img_size, :]
                piece[:temp.shape[0], :temp.shape[1], :] = temp
                np.save(f'{save_path}{img_size}/{num}.npy', piece)
                num+=1


cut_img(train_input_files, 'train_input_img_', 128)
cut_img(train_label_files, 'train_label_img_', 128)
cut_img(val_input_files, 'val_input_img_', 128)
cut_img(val_label_files, 'val_label_img_', 128)
