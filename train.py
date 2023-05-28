import time
import os
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DEFINE THE LOSS FUNCTIONS
criterion = nn.CrossEntropyLoss()
seg_criterion = nn.MSELoss()


# Function to get feature map using weights of classification layer
def returnCAM_SEG(args, f_l, f_s, w_l, w_s):
    # Downsampling to get same size CAM
    m = nn.Upsample(scale_factor=-0.5, mode='bilinear')
    f_l = m(f_l)
    w_l = torch.softmax(w_l, dim=1)
    w_s = torch.softmax(w_s, dim=1)
    batch, nc, h, w = f_s.size()
    cam_l = torch.zeros((args.batch_size, h, w)).to(device)
    cam_s = torch.zeros((args.batch_size, h, w)).to(device)
    for i in range(batch):
        cam_l[i] = torch.sum(w_l[i] * f_l[i].reshape((nc, h * w)).T, dim=1).reshape((h, w))
        cam_s[i] = torch.sum(w_s[i] * f_s[i].reshape((nc, h * w)).T, dim=1).reshape((h, w))
    return cam_l, cam_s


def train(args, model, optimizer, dataloaders):
    trainloader, testloader = dataloaders

    best_testing_accuracy = 0.0

    # training
    print('Network training starts ...')
    for epoch in range(args.epochs):
        model.train()

        batch_time = time.time();
        iter_time = time.time()

        for i, data in enumerate(trainloader):

            img_L = data['img_L'];
            img_S = data['img_S'];
            labels = data['label']
            img_L, img_S, labels = img_L.to(device), img_S.to(device), labels.to(device)

            if args.mode == 'CAM':
                # INPUT TO CAM MODEL AND COMPUTE THE LOSS
                cls_scores, _, _ = model(img_L)
                loss = criterion(cls_scores, labels)

            elif args.mode == 'SEG':
                # INPUT TO SEG MODEL, DEFINE THE SCALE EQUIVARIANT LOSS
                # AND COMPUTE THE TOTAL LOSS
                cls_scores_l, f_l, w_l, cls_scores_s, f_s, w_s = model(img_L, img_S)
                # Classification losses
                loss_l = criterion(cls_scores_l, labels)
                loss_s = criterion(cls_scores_s, labels)

                cam_l, cam_s = returnCAM_SEG(args, f_l, f_s, w_l, w_s)

                '''w_l = torch.softmax(w_l, 1)
                w_s = torch.softmax(w_s, 1)
                cam_l = returnCAM_1(f_l, w_l)
                cam_s = returnCAM_1(f_s, w_s)'''
                loss_seg = seg_criterion(cam_l, cam_s)
                loss = (loss_l + loss_s) / 2 + loss_seg

            else:
                NotImplementedError

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # logging
            if i % 100 == 0 and i != 0:
                print('epoch:{}, iter:{}, time:{:.2f}, loss:{:.5f}'.format(epoch, i,
                                                                           time.time() - iter_time, loss.item()))
                iter_time = time.time()
        batch_time = time.time() - batch_time
        print('[epoch {} | time:{:.2f} | loss:{:.5f}]'.format(epoch, batch_time, loss.item()))

        # evaluation
        if epoch % 1 == 0:
            testing_accuracy = evaluate(args, model, testloader)
            print('testing accuracy: {:.3f}'.format(testing_accuracy))
            print('-------------------------------------------------')

            if testing_accuracy > best_testing_accuracy:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, './checkpoints/{}_checkpoint.pth'.format(args.exp_id))
                best_testing_accuracy = testing_accuracy
                print('new best model saved at epoch: {}'.format(epoch))
                print('-------------------------------------------------')
    print('-------------------------------------------------')
    print('best testing accuracy achieved: {:.3f}'.format(best_testing_accuracy))


def train_CV(args, model, optimizer, dataloaders):
    # IMPLEMENT 5-FOLD CROSS VALIDATION
    # YOU CAN REUSE CODES FROM TRAIN()
    trainloader, testloader = dataloaders
    splitter = KFold(n_splits=5, shuffle=True)
    splits = []
    for train_idx, val_idx in splitter.split(trainloader['label']):
        splits.append((train_idx, val_idx))

    # Split train data into train and val
    for i, split in enumerate(range(len(splits))):
        train_folds_idx = split[0]
        valid_folds_idx = split[1]
        train_sampler = ImageSampler(train_folds_idx)
        val_sampler = ImageSampler(valid_folds_idx)
        train_batch_sampler = ImageBatchSampler(train_sampler,
                                                args.batch_size)
        val_batch_sampler = ImageBatchSampler(val_sampler,
                                              args.batch_size,
                                              drop_last=False)
        train_loader = DataLoader(dataset, batch_sampler=train_batch_sampler)
        val_loader = DataLoader(dataset, batch_sampler=val_batch_sampler)
        args.exp_id = 'exp_CAM_fold_{}'.format(str(i))
        train(args, model, optimizer, (train_loader, val_loader))


def evaluate(args, model, testloader):
    total_count = torch.tensor([0.0]).to(device);
    correct_count = torch.tensor([0.0]).to(device)

    for i, data in enumerate(testloader):
        img_L = data['img_L'];
        labels = data['label']
        img_L, labels = img_L.to(device), labels.to(device)
        total_count += labels.size(0)

        with torch.no_grad():
            cls_L_scores, _, _ = model(img_L, img_S=None)
            predict_L = torch.argmax(cls_L_scores, dim=1)
            correct_count += (predict_L == labels).sum()
    testing_accuracy = correct_count / total_count

    return testing_accuracy.item()


def resume(args, model, optimizer):
    checkpoint_path = './checkpoints/{}_checkpoint.pth'.format(args.exp_id)
    assert os.path.exists(checkpoint_path), ('checkpoint do not exits for %s' % checkpoint_path)

    checkpoint_saved = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint_saved['model_state_dict'])
    optimizer.load_state_dict(checkpoint_saved['optimizer_state_dict'])

    print('Resume completed for the model\n')

    return model, optimizer
