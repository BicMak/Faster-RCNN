import os
import PIL.Image
import numpy as np
import cv2
import PIL
from torch.utils.data import Dataset
import torch
import torchvision
import collections
import re

train_image_dir = 'archive\Train File\Train File\images'
train_annotation_dir = 'archive\Train File\Train File\\annotations'
test_image_dir = 'archive\\Test File\\Test File\\images'
test_annotation_dir = 'archive\\Test File\\Test File\\annotations'

class detection_Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, train, transforms,Means = [0.485, 0.456, 0.406],Stds = [0.229, 0.224, 0.225]):
        if train: 
            image_dir = train_image_dir
            annotation_dir = train_annotation_dir
        else:
            image_dir = test_image_dir
            annotation_dir = test_annotation_dir
        
        self.classifiation_dic = {'Apple':0, 'Banana':1, 'Grapes':2, 'Guava':3,
                                'HogPlum':4, 'Jackfruit':5, 'Litchi':6, 'Mango':7,
                                  'Orange':8, 'Papaya':9}

        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.img_lst = os.listdir(image_dir)
        self.annotation_lst = os.listdir(annotation_dir)
        self.detection_lst = (self.img_lst, self.annotation_lst)
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(Means,Stds),            
            torchvision.transforms.Resize((800,800)),
            torchvision.transforms.ToTensor(),
        ])
        self.pre_transform = transforms

    def __len__(self):
        return len(self.detection_lst[0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.image_dir, self.detection_lst[0][idx])
        anno_name = os.path.join(self.annotation_dir, self.detection_lst[1][idx])
        object_name = re.search(r'_(.*?)\.', self.detection_lst[1][idx])
        object_num = self.classifiation_dic.get(object_name.group(1))
        image = PIL.Image.open(img_name)
        with open(anno_name, 'r') as f:
            d = {}
            anno = np.array(list(map(np.float32, f.read().strip().split())))
            center = (anno[0], anno[1])
            size = (anno[2], anno[3])
            
            x1 = int(center[0] - size[0]/2)
            y1 = int(center[1] - size[1]/2)
            x2 = int(center[0] + size[0]/2)
            y2 = int(center[1] + size[1]/2)
            d['boxes'] = [x1, y1, x2, y2]
            d['labels'] = object_num

        
        if isinstance(image, np.ndarray):
            image = PIL.Image.fromarray(image)

        if self.pre_transform:
            image = self.transform(image)


        return {'image': image, 'annotation': d}




'''

'''
