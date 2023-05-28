import os
import PIL.Image as Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

'''ANNOTATION_PATH_TRAIN = '/home/shreya/Desktop/ANU/0Sem 2 2021/Adv Mech/Assignment/Lab 4/cifar-10-cam-seg-data/train_cls.txt'
ANNOTATION_PATH_TEST = '/home/shreya/Desktop/ANU/0Sem 2 2021/Adv Mech/Assignment/Lab 4/cifar-10-cam-seg-data/test_cls.txt'
IMG_PATH_TRAIN = '/home/shreya/Desktop/ANU/0Sem 2 2021/Adv Mech/Assignment/Lab 4/cifar-10-cam-seg-data/train'
IMG_PATH_TEST = '/home/shreya/Desktop/ANU/0Sem 2 2021/Adv Mech/Assignment/Lab 4/cifar-10-cam-seg-data/test'
SEG_PATH = '/home/shreya/Desktop/ANU/0Sem 2 2021/Adv Mech/Assignment/Lab 4/cifar-10-cam-seg-data/test_seg'''

ANNOTATION_PATH_TRAIN = 'engn8536/Datasets/cifar-10-cam-seg-data/train_cls.txt'
ANNOTATION_PATH_TEST = 'engn8536/Datasets/cifar-10-cam-seg-data/test_cls.txt'
IMG_PATH_TRAIN = 'engn8536/Datasets/cifar-10-cam-seg-data/train'
IMG_PATH_TEST = 'engn8536/Datasets/cifar-10-cam-seg-data/test'
SEG_PATH = 'engn8536/Datasets/cifar-10-cam-seg-data/test_seg'

Transform_L = transforms.Compose([transforms.Resize((224,224)),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
Transform_S = transforms.Compose([transforms.Resize((112,112)),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
Transform_Seg = transforms.Compose([transforms.Resize((112,112)),
                                    transforms.ToTensor()])

class Cifar10Loader(Dataset):
    # Cifar-10 Dataset
    def __init__(self, args, split='train'):
        assert split in ['train', 'test']
        self.dict = {}
        self.split = split

        if self.split == 'test': # train or test mode
            self.img_path = IMG_PATH_TEST
            self.seg_path = SEG_PATH
            self.anno_path = ANNOTATION_PATH_TEST
        else:
            self.img_path = IMG_PATH_TRAIN
            self.anno_path = ANNOTATION_PATH_TRAIN

        with open(self.anno_path, 'r') as file:
            data = file.readlines()
            for idx, line in enumerate(data):
                img_name = line.split(' ')[0]
                img_label = line.split(' ')[1]
                self.dict[idx] = (img_name, int(img_label))

        self.Transform_L = Transform_L
        self.Transform_S = Transform_S
        if args.augmentation1:
            self.Transform_Seg = transforms.Compose([transforms.Resize((112, 112)),
                                                     transforms.RandomVerticalFlip(p=1),
                                                     transforms.ToTensor()])
        elif args.augmentation2:
            self.Transform_Seg = transforms.Compose([transforms.Resize((112, 112)),
                                                     transforms.RandomVerticalFlip(p=1),
                                                     transforms.RandomErasing(p=0.3),
                                                     transforms.ToTensor()])
        else:
            self.Transform_Seg = Transform_Seg

    def __len__(self):
        return len(self.dict)

    def __getitem__(self, idx):
        img_name, img_label = self.dict[idx]

        img = Image.open(os.path.join(self.img_path, img_name))
        img_L = self.Transform_L(img)

        if self.split == 'test':
            sample = {'img_L': img_L, 'label': torch.tensor(img_label)}
            return sample
        else:
            img_S = self.Transform_S(img)
            sample = {'img_L': img_L, 'img_S': img_S, 'label': torch.tensor(img_label)}
            return sample
