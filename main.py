import argparse
from model import CAMModel
from dataloader import Cifar10Loader
from train import train, resume, evaluate, train_CV
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=str, default='exp_CAM')
    parser.add_argument('--mode', type=str, default='CAM', help='CAM or SEG')

    parser.add_argument('--resume', type=int, default=0, help='resume the trained model')
    parser.add_argument('--test', type=int, default=0, help='test with trained model')

    parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

    parser.add_argument('--seed', type=int, default=1, help='random seed')

    parser.add_argument('--cv', type=int, default=0, help='cross validation for cam')

    parser.add_argument('--augmentation1', type=int, default=0, help='vertically flip image - data augmentation')
    parser.add_argument('--augmentation2', type=int, default=0,
                        help='vertically flip and randomly erase some pixels of image - data augmentation')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # dataloaders
    trainloader = DataLoader(Cifar10Loader(args, split='train'),
        batch_size=args.batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(Cifar10Loader(args, split='test'),
        batch_size=args.batch_size, shuffle=False, num_workers=2)
    dataloaders = (trainloader, testloader)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # network
    model = CAMModel(args).to(device)
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9,0.999))

    # resume the trained model
    if args.resume:
        model, optimizer = resume(args, model, optimizer)

    if args.test == 1: # test mode
        testing_accuracy = evaluate(args, model, testloader)
        print('testing finished, accuracy: {:.3f}'.format(testing_accuracy))
    else: # train mode, train the network from scratch
        if args.cv:
            train_CV(args, model, optimizer, dataloaders)
        else:
            train(args, model, optimizer, dataloaders)
        print('training finished')
