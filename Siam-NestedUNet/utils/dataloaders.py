import os
import numpy as np
import torch.utils.data as data
import torchvision
from PIL import Image
from utils import transforms as tr
import matplotlib.pyplot as plt
import album as BCODE_A


'''
Load all training and validation data paths
'''
def full_path_loader(data_dir):
    train_data = [i for i in os.listdir(data_dir + 'train/A/') if not
    i.startswith('.')]
    train_data.sort()

    valid_data = [i for i in os.listdir(data_dir + 'val/A/') if not
    i.startswith('.')]
    valid_data.sort()

    train_label_paths = []
    val_label_paths = []
    for img in train_data:
        train_label_paths.append(data_dir + 'train/OUT/' + img)
    for img in valid_data:
        val_label_paths.append(data_dir + 'val/OUT/' + img)


    train_data_path = []
    val_data_path = []

    for img in train_data:
        train_data_path.append([data_dir + 'train/', img])
    for img in valid_data:
        val_data_path.append([data_dir + 'val/', img])

    train_dataset = {}
    val_dataset = {}
    for cp in range(len(train_data)):
        train_dataset[cp] = {'image': train_data_path[cp],
                         'label': train_label_paths[cp]}
    for cp in range(len(valid_data)):
        val_dataset[cp] = {'image': val_data_path[cp],
                         'label': val_label_paths[cp]}


    return train_dataset, val_dataset

'''
Load all testing data paths
'''
def full_test_loader(data_dir):

    test_data = [i for i in os.listdir(data_dir + 'test/A/') if not
                    i.startswith('.')]
    test_data.sort()

    test_label_paths = []
    for img in test_data:
        test_label_paths.append(data_dir + 'test/OUT/' + img)

    test_data_path = []
    for img in test_data:
        test_data_path.append([data_dir + 'test/', img])

    test_dataset = {}
    for cp in range(len(test_data)):
        test_dataset[cp] = {'image': test_data_path[cp],
                           'label': test_label_paths[cp]}

    return test_dataset

def cdd_loader(img_path, label_path, aug):

# -------------------------------------------------------------------------------------
# img => img1, img2 split, mask => mask1 + mask2 
# 22.11. 8.(화)
    dir = img_path[0]
    name = img_path[1]

    image = plt.imread(dir + name)
    label = plt.imread(label_path)

    if image.shape == (754, 1508):
        img1 = image[:, :754]
        img2 = image[:, 754:]
    else:
        img1 = image[:, :750]
        img2 = image[:, 750:]

    if label.shape == (754, 1508):
        mask1 = label[:,:754]
        mask2 = label[:,754:]
    else:
        mask1 = label[:,:750]
        mask2 = label[:,750:]
    label = mask1 + mask2
    
    sample = {'image': (img1, img2), 'label': label}
    
    if aug :
        sample = BCODE_A.Albumentations(img1, img2, label)            
        # change to dic key
        sample = {'image': (sample['image'], sample['image0']), 'label': sample['mask']}

        # toTensor
    sample = tr.train_transforms(sample) 
    
    return sample['image'][0], sample['image'][1], sample['label']


    
# def cdd_loader(img_path, label_path, aug):
#     dir = img_path[0]
#     name = img_path[1]

#     img1 = Image.open(dir + 'A/' + name)
#     img2 = Image.open(dir + 'B/' + name)
#     label = Image.open(label_path).convert("L")

    
#     # img1 = img1.convert('RGB')
#     # img2 = img2.convert('RGB')
#     # label = label.convert('RGB')

#     re_size = (256,256)
#     if img1.size != (256, 256):
#         img1 = img1.resize(re_size)
#         img2 = img2.resize(re_size)
#         label = label.resize(re_size)



#     sample = {'image': (img1, img2), 'label': label}

#     if aug:
#         sample = tr.train_transforms(sample)
#     else:
#         sample = tr.test_transforms(sample)

#     return sample['image'][0], sample['image'][1], sample['label']
    
# -------------------------------------------------------------------------------------
# Albumentations 적용 코드
# 22.11. 7.(월)

    # # to nparray
    # img1 = np.asarray(img1)
    # img2 = np.asarray(img2)
    # label = np.asarray(label)
    
    # sample = {'image': (img1, img2), 'label': label}
    
    # if aug :
    #     sample = BCODE_A.Albumentations(img1, img2, label)            
    #     # change to dic key
    #     sample = {'image': (sample['image'], sample['image0']), 'label': sample['mask']}

    #     # toTensor
    # sample = tr.train_transforms(sample) 
    
    # return sample['image'][0], sample['image'][1], sample['label']

# -------------------------------------------------------------------------------------


class CDDloader(data.Dataset):

    def __init__(self, full_load, aug=False):

        self.full_load = full_load
        self.loader = cdd_loader
        self.aug = aug

    def __getitem__(self, index):

        img_path, label_path = self.full_load[index]['image'], self.full_load[index]['label']

        return self.loader(img_path,
                           label_path,
                           self.aug)

    def __len__(self):
        return len(self.full_load)