import pandas as pd
import numpy as np
import os
import cv2

base_path = '/cver/jychen/Dataset/ISIC/ISIC2018_Task1-2_Test_Input'
write_path1 = os.path.join(base_path, '1') #nevus
write_path2 = os.path.join(base_path, '2') #melanoma
write_path3 = os.path.join(base_path, '3') #seborrheic_keratosis
os.mkdir(write_path1)
os.mkdir(write_path2)
os.mkdir(write_path3)

data = pd.read_csv('/cver/jychen/DMTNet/data/isic/class_id.csv')
data = np.array(data)
for idx, x in enumerate(data):
    img_id = x[0]
    img_class = x[1]
    read_path = os.path.join(base_path, img_id) + ".jpg"
    if img_class == 'nevus':
        write_path = os.path.join(write_path1, img_id) + ".jpg"
        img = cv2.imread(read_path)
        cv2.imwrite(write_path, img)
    elif img_class == 'melanoma':
        write_path = os.path.join(write_path2, img_id) + ".jpg"
        img = cv2.imread(read_path)
        cv2.imwrite(write_path, img)
    elif img_class == 'seborrheic keratosis':
        write_path = os.path.join(write_path3, img_id) + ".jpg"
        img = cv2.imread(read_path)
        cv2.imwrite(write_path, img)