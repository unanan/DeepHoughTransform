import numpy as np 
import os
from os.path import join, split, isdir, isfile, abspath
import torch
from PIL import Image
import random
import collections
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class ImageTransform():
    def __init__(self, split):
        if split != 'test':
            self.transform = transforms.Compose([
                # TODO: Add Transforms
                transforms.Resize((400, 400)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((400, 400)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __call__(self, image):
        if len(image.split())!=3:
            image=image.convert('RGB')

        return self.transform(image)


class GtTransform():
    def __init__(self, split):
        pass

    def smooth_label(self,gt):
        pass

    def __call__(self, gt):
        if len(gt.split())>1:
            gt=gt.convert('L')



class SemanLineDatasetTest(Dataset):

    def __init__(self, root_dir, label_file, transform=None, t_transform=None):
        lines = [line.rstrip('\n') for line in open(label_file)]
        self.image_path = [join(root_dir, i) for i in lines]
        self.transform = transform
        self.t_transform = t_transform
        
    def __getitem__(self, item):
        assert isfile(self.image_path[item]), self.image_path[item]
        image = Image.open(self.image_path[item]).convert('RGB')
        w, h = image.size
        if self.transform is not None:
            image = self.transform(image)
            
        return image, self.image_path[item], [h, w]


    def __len__(self):
        return len(self.image_path)


class LineDataset(Dataset):
    '''
    Your Dataset.
    '''
    def __init__(self, label_file, transform=ImageTransform("test"), t_transform=GtTransform("test")):
        self.image_gt_pairs = [line.rstrip('\n').split(";") for line in open(label_file)]
        self.transform = transform
        self.t_transform = t_transform

    def __getitem__(self, item):
        image = Image.open(self.image_gt_pairs[item][0]).convert('RGB')
        w, h = image.size
        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            image_tensor=transforms.ToTensor(image)

        gt = Image.open(self.image_gt_pairs[item][1]).convert('L')
        w, h = gt.size
        if self.t_transform is not None:
            gt_tensor = self.t_transform(gt)
        else:
            gt_tensor=transforms.ToTensor(gt)

        return image_tensor, gt_tensor

    def __len__(self):
        return len(self.image_gt_pairs)


def get_loader(label_file, batch_size, num_thread=4, pin=True, split='train'):
    dataset = LineDataset(label_file, transform=ImageTransform(split), t_transform=GtTransform(split))
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True,
                             num_workers=num_thread,pin_memory=pin)
    return data_loader

        
if __name__=="__main__":
    LineDataset("./test/gt.txt")