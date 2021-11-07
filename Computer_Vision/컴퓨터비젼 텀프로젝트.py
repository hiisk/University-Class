# Commented out IPython magic to ensure Python compatibility.
import os, time, random
import numpy as np
import pandas as pd
import cv2, torch
from tqdm.auto import tqdm
import shutil as sh

from IPython.display import Image, clear_output
import matplotlib.pyplot as plt
# %matplotlib inline

from google.colab import drive
drive.mount('/gdrive')

!git clone https://github.com/ultralytics/yolov5
!pip install -U pycocotools
!pip install -qr yolov5/requirements.txt
!cp yolov5/requirements.txt

img_h, img_w, num_channels = (380, 676, 3)
df = pd.read_csv('/gdrive/My Drive/한밭대 20187100/4-1/컴퓨터비전/텀프로젝트/train_solution.csv')
df.rename(columns={'image':'image_id'}, inplace=True)
df['image_id'] = df['image_id'].apply(lambda x: x.split('.')[0])
df['x_center'] = (df['xmin'] + df['xmax'])/2
df['y_center'] = (df['ymin'] + df['ymax'])/2
df['w'] = df['xmax'] - df['xmin']
df['h'] = df['ymax'] - df['ymin']
df['classes'] = 0
df['x_center'] = df['x_center']/img_w
df['w'] = df['w']/img_w
df['y_center'] = df['y_center']/img_h
df['h'] = df['h']/img_h
df.head()

index = list(set(df.image_id))
image = random.choice(index)
print("Image ID: %s"%(image))
img = cv2.imread(f'/gdrive/My Drive/한밭대 20187100/4-1/컴퓨터비전/텀프로젝트/data/training_images/{image}.jpg')

image = random.choice(index)
Image(filename=f'/gdrive/My Drive/한밭대 20187100/4-1/컴퓨터비전/텀프로젝트/data/training_images/{image}.jpg',width=600)

source = 'training_images'
if True:
    for fold in [0]:
        val_index = index[len(index)*fold//5:len(index)*(fold+1)//5]
        for name,mini in tqdm(df.groupby('image_id')):
            if name in val_index:
                path2save = 'val2021/'
            else:
                path2save = 'train2021/'
            if not os.path.exists('/tmp/convertor/fold{}/labels/'.format(fold)+path2save):
                os.makedirs('/tmp/convertor/fold{}/labels/'.format(fold)+path2save)
            with open('/tmp/convertor/fold{}/labels/'.format(fold)+path2save+name+".txt", 'w+') as f:
                row = mini[['classes','x_center','y_center','w','h']].astype(float).values
                row = row.astype(str)
                for j in range(len(row)):
                    text = ' '.join(row[j])
                    f.write(text)
                    f.write("\n")
            if not os.path.exists('/tmp/convertor/fold{}/images/{}'.format(fold,path2save)):
                os.makedirs('/tmp/convertor/fold{}/images/{}'.format(fold,path2save))
            sh.copy("/gdrive/My Drive/한밭대 20187100/4-1/컴퓨터비전/텀프로젝트/data/{}/{}.jpg".format(source,name),'/tmp/convertor/fold{}/images/{}/{}.jpg'.format(fold,path2save,name))

!python yolov5/detect.py --weights yolov5/yolov5s.pt --img 676 --conf 0.4 --source "/gdrive/My Drive/한밭대 20187100/4-1/컴퓨터비전/텀프로젝트/data/testing_images"

predicted_files = []
for (dirpath, dirnames, filenames) in os.walk("runs/detect/exp") :
    predicted_files.extend(filenames)
print(predicted_files)

Image(filename=f'runs/detect/exp/{random.choice(predicted_files)}')

