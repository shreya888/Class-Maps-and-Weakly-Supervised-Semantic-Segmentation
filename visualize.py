# author: Yicong Hong (yicong.hong@anu.edu.au) for Lab4, 2020, ENGN8536 @ANU

import argparse
from train import resume
from model import CAMModel
import os
import glob
import cv2
import numpy as np
import PIL.Image as Image
import torch
import torch.nn.functional as F
from torchvision import transforms

''' Refer to CAM visualization code by BoleiZhou (bzhou@csail.mit.edu),
    the author of paper Learning Deep Features for Discriminative Localization,
    see https://github.com/zhoubolei/CAM/blob/master/pytorch_CAM.py '''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=str, default='exp_CAM')
    parser.add_argument('--mode', type=str, default='CAM', help='CAM or SEG')
    args = parser.parse_args()
    return args

def returnCAM(feature_conv, weights):
    # generate the class activation maps upsample to 224x224
    size_upsample = (224, 224)
    nc, h, w = feature_conv.shape

    weight_softmax = F.softmax(weights, dim=0)
    cam = weight_softmax.unsqueeze(-1)*(feature_conv.reshape((nc, h*w)))
    cam = cam.sum(0).cpu().numpy()

    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    output_cam = cv2.resize(cam_img, size_upsample)
    return output_cam

preprocess = transforms.Compose([
   transforms.Resize((224,224)),
   transforms.ToTensor(),
   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

if __name__ == '__main__':
    args = parse_args()

    # network
    model = CAMModel(args).to(device)
    model.eval()
    # optimizer (useless)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00, betas=(0.9,0.999))
    # resume the trained model (assume trained CAM/SEG models exist)
    model, optimizer = resume(args, model, optimizer)

    # source images
    img_list = glob.glob('engn8536/Datasets/cifar-10-cam-seg-data/test_seg_source/*')
    for img_path in img_list:
        img = Image.open(img_path)
        img_tensor = preprocess(img).unsqueeze(0).cuda()

        with torch.no_grad():
            feature_conv, weights = model(img_tensor)

        # render the CAM and output
        # feature_conv, weights = cam_source
        output_cam = returnCAM(feature_conv.squeeze(), weights)

        out_img = cv2.imread(img_path)
        out_img = cv2.resize(out_img,(224, 224))
        heatmap = cv2.applyColorMap(output_cam, cv2.COLORMAP_JET)
        result = heatmap * 0.15 + out_img * 0.3
        img_name = img_path.split('/')[-1]

        # threshold the heatmap, use it as the segmentation map
        ret, thresh_img = cv2.threshold(output_cam, 175, 255, cv2.THRESH_BINARY)

        if args.mode == 'CAM':
            cv2.imwrite('./visualize/CAM/'+img_name, result)
            thresh_img_name = img_name.split('.')[0] + '_seg.png'
            cv2.imwrite('./visualize/CAM/'+thresh_img_name, thresh_img)

        elif args.mode == 'SEG':
            cv2.imwrite('./visualize/SEG/'+img_name, result)
            thresh_img_name = img_name.split('.')[0] + '_seg.png'
            cv2.imwrite('./visualize/SEG/'+thresh_img_name, thresh_img)

    else:
        NotImplementedError
