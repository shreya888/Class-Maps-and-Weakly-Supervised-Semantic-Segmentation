import cv2
import numpy as np

def compute_iou(seg_path_gt, seg_path_cam, seg_path_seg):
    # COMPUTE THE INTERSECTION OVER UNION
    im_gt = cv2.imread(seg_path_gt)
    im_cam = cv2.resize(cv2.imread(seg_path_cam), (im_gt.shape[0:2]))
    im_seg = cv2.resize(cv2.imread(seg_path_seg), (im_gt.shape[0:2]))
    _, intersection_cam = np.unique(cv2.bitwise_and(im_gt, im_cam), return_counts=True)
    _, intersection_seg = np.unique(cv2.bitwise_and(im_gt, im_seg), return_counts=True)
    _, union_cam = np.unique(cv2.bitwise_or(im_gt, im_cam), return_counts=True)
    _, union_seg = np.unique(cv2.bitwise_or(im_gt, im_seg), return_counts=True)
    iou_CAM = intersection_cam[-1] / union_cam[-1]
    iou_SEG = intersection_seg[-1] / union_seg[-1]

    return iou_CAM, iou_SEG


if __name__ == '__main__':
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    for cls in classes:
        seg_path_gt = './data/test_seg/{}.png'.format(cls)       # ground-truth seg map
        seg_path_cam = './visualize/CAM/{}_seg.png'.format(cls)  # output seg map from CAM
        seg_path_seg = './visualize/SEG/{}_seg.png'.format(cls)  # output seg map from SEG

        iou_CAM, iou_SEG = compute_iou(seg_path_gt, seg_path_cam, seg_path_seg)

        print('Class: {} | CAM IoU: {:.3f} | SEG IoU: {:.3f}'.format(cls, iou_CAM, iou_SEG))
