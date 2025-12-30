import os
import cv2
import time
import copy
import torch
import random
import numpy as np
import torchvision
from PIL import Image
from tqdm import tqdm
import torch.utils.data as data

from models.common import config

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', 'bmp', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

class TrainDatasetFromFolder(data.Dataset):
    def __init__(self, dataset_dir_super, dataset_dir, dataset_bsd, block_size):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = []

        for path, dir_list, file_list in os.walk(dataset_dir):
            for file_name in file_list:
                self.image_filenames.extend([path + '/' + file_name])
        
        for i in range(6):
            self.image_filenames.extend(self.image_filenames)

        self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.Grayscale(num_output_channels=1),
                torchvision.transforms.RandomCrop(block_size),
                # torchvision.transforms.Normalize(mean=[0.45], std=[0.22])
            ])

    def __getitem__(self, index):
        for i in range(index, len(self.image_filenames)):
            # one_image = Image.open(self.image_filenames[i])

            Img = cv2.imread(self.image_filenames[i], flags=1)
            Img_yuv = cv2.cvtColor(Img, cv2.COLOR_BGR2YCrCb)
            one_image = Img_yuv[:, :, 0]
            one_image = np.float32(one_image) / 255.

            if one_image.shape[0] >= config.para.patch_size and one_image.shape[1] >= config.para.patch_size:
                one_image = self.transform(one_image)
                return one_image
            else:
                continue

    def __len__(self):
        return len(self.image_filenames)
