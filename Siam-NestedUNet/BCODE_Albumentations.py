#!/usr/bin/env python
# coding: utf-8


import random

import cv2
from matplotlib import pyplot as plt

import albumentations as A


def visualize(image1, image2, mask):
    fontsize = 18
    
    f, ax = plt.subplots(1, 3, figsize=(8, 8))

    ax[0].imshow(image1)
    ax[0].set_title('image1', fontsize=fontsize)
    
    ax[1].imshow(image2)
    ax[1].set_title('image2', fontsize=fontsize)

    ax[2].imshow(mask)
    ax[2].set_title('mask', fontsize=fontsize)


# image1 = cv2.imread('./1.png')
# image2 = cv2.imread('./2.png')
# mask = cv2.imread('./3.png', cv2.IMREAD_GRAYSCALE)





def Albumentations(img1, img2, msk) :
    
    # original_height, original_width = img1.shape[:2]
    
    aug = A.Compose([
    # A.OneOf([
    #     A.RandomSizedCrop(min_max_height=(50, 101), height=original_height, width=original_width, p=0.5),
    #     A.PadIfNeeded(min_height=original_height, min_width=original_width, p=0.5)
    # ], p=1),    
    A.VerticalFlip(p=0.5),              
    A.RandomRotate90(p=0.5),
    A.OneOf([
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
        A.GridDistortion(p=0.5),
        A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1)                  
        ], p=0.8),
    A.CLAHE(p=0.8),
    A.RandomBrightnessContrast(p=0.8),    
    A.RandomGamma(p=0.8)],
    additional_targets = {'image0': 'image', 'image1': 'image', 'label' : 'mask'}
    )
    
    random.seed(11)
    
    s = aug(image=img1, image0=img2, mask=msk)
    
    return s