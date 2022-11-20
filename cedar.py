import wget
import patoolib
import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import itertools
import random
    
def preprocessing(img):
    img = cv.imread(img)
    img = np.invert(img)
    img = tf.image.resize(img, size=(155,220), method=tf.image.ResizeMethod.BILINEAR).numpy()
    img = cv.cvtColor(img , cv.COLOR_BGR2GRAY)
    img = np.reshape(img , (img.shape[0] , img.shape[1] , 1))
    img = img/255.0
    return img    

def gen_pairs(org_folder_path , forg_folder_path , writer_strt_indx , writer_end_indx):
    
    all_pairs = []
    all_labels = []
    
    for i in range(writer_strt_indx , writer_end_indx + 1):
        
        writer_org = [os.path.join(org_folder_path, f'original_{i}_{j}.png') for j in range(1 , 25)]
        writer_forg = [os.path.join(forg_folder_path, f'forgeries_{i}_{j}.png') for j in range(1 , 25)]
        pairs = []
        labels = []
        
        for pair in itertools.combinations(writer_org , 2):
            all_pairs.append(pair)
            all_labels.append(0)
            
        for pair in itertools.product(writer_org , writer_forg):
            pairs.append(pair)
            labels.append(1)
        for pair , label in (random.sample(list(zip(pairs , labels)) , 276)):
            all_pairs.append(pair)
            all_labels.append(label)
        
        all_labels = np.asarray(all_labels , dtype='float32')
        return all_pairs , all_labels
        
def create_dataset(org_folder_path , forg_folder_path , M=50 , K=55):
    X_train = [[] , []]
    X_test = [[] , []]
    train_pairs , Y_train = gen_pairs(org_folder_path , forg_folder_path , 1 , M)
    test_pairs , Y_test = gen_pairs(org_folder_path , forg_folder_path, K-M+1, K)
    
    for pair1,pair2 in train_pairs:
        X_train[0].append(preprocessing(pair1))
        X_train[1].append(preprocessing(pair2))
    for pair1,pair2 in test_pairs:
        X_test[0].append(preprocessing(pair1))
        X_test[1].append(preprocessing(pair2))
    
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    
    return (X_train , Y_train) , (X_test , Y_test)

def load_data(path=None):
    if os.path.exists('signatures'):
        print('folder already exists')
        return
    url = "https://cedar.buffalo.edu/NIJ/data/signatures.rar"
    wget.download(url)
    if path == None:
        patoolib.extract_archive("signatures.rar")
    else:
        patoolib.extract_archive("signatures.rar" , outdir=path)
