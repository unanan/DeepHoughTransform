import os
import time
import yaml
import logging
import argparse
import numpy as np
import plotext.plot as plx

import torch
from torch.utils.data import DataLoader
from torch import optim
from skimage.measure import label, regionprops

from data.TestPaper_dataset.testpaperdataset import TestPaper
from dataloader import get_loader
from model.network import Net
from utils.avgmeter import AverageMeter
from utils.misc import reverse_mapping, visulize_mapping, get_boundary_point

def train(args):
    # CONFIGS = yaml.load(open(args.config)) # deprecated, please set the configs in parse_args()

    # Set device
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device.strip()
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")  # Not suggested

    # Set save folder & logging config
    subfolder = time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime(time.time()))
    if not args.save_folder or (not os.path.isdir(args.save_folder)):
        print("Warning: Not invalid value of 'save_folder', set as default value: './save_folder'..")
        save_folder = "./save_folder"
    else:
        save_folder = args.save_folder
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    save_folder = os.path.join(save_folder,subfolder)
    os.mkdir(save_folder)
    #TODO:logging

    # Load Dataset
    trainloader = get_loader(args.train_gtfile,
                             batch_size=args.batch_size,
                             num_thread=args.num_workers)
    valloader = get_loader(args.val_gtfile,
                           batch_size=args.batch_size,
                           num_thread=args.num_workers)

    # Init Net
    model = Net(numAngle=args.num_angle, numRho=args.num_rho, backbone=args.backbone)
    if args.resume:
        model.load_state_dict(torch.load(args.resume))
    model = torch.nn.DataParallel(model).to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters())

    # Loss
    criterion = torch.nn.CrossEntropyLoss()
    losses = AverageMeter()

    # Start Training
    model.train();iter = 0  # iter id start from 1
    for epoch in range(args.max_epoch):

        for batch in trainloader:
            start = time.time()
            iter += 1
            img_tensor, gt_tensor = batch
            optimizer.zero_grad()

            # Forwarding
            preds = model(img_tensor)

            # Calculate Loss
            loss = criterion(preds, gt_tensor)
            loss.backward()
            optimizer.step()
            losses.update(loss.item(), args.batch_size)

            if iter%args.show_interval==0:
                logging.info(f"Training [{epoch}/{args.max_epoch}][{iter}] Loss:{losses.avg} Time:{time.time()-start:.1f}s")

            if iter%args.val_interval==0:
                pass
                # vallosses = AverageMeter()
                # valaccs = AverageMeter()
                # valstart = time.time()
                # # Start Validating
                # for valbatch in enumerate(valloader):
                #     val_img_tensor, val_label_tensor = valbatch
                #     # Forwarding
                #     preds = model(img_tensor)
                #
                #     # Calculate Loss
                #     loss = criterion(preds, label_tensor)
                #     vallosses.update(loss.item(), args.val_batch_size)
                #
                #     # Calculate accuracy metrics
                #     acc = None  #TODO
                #     valaccs.update(acc, args.val_batch_size)
                # logging.info(f"Validating: Loss:{vallosses.avg} Acc:{valaccs.avg} Time:{time.time() - valstart:.1f}s")
                #
                # key_points = model(img_tensor)
                # key_points = torch.sigmoid(key_points)
                # binary_kmap = key_points.squeeze().cpu().numpy() > args.threshold
                # kmap_label = label(binary_kmap, connectivity=1)
                # props = regionprops(kmap_label)
                # plist = []
                # for prop in props:
                #     plist.append(prop.centroid)
                #
                # b_points = reverse_mapping(plist, numAngle=args.num_angle, numRho=args.num_rho, size=(400, 400))
                # size = (img_tensor.shape[2].item(), img_tensor.shape[3].item())
                # scale_w = size[1] / 400
                # scale_h = size[0] / 400
                # for i in range(len(b_points)):
                #     y1 = int(np.round(b_points[i][0] * scale_h))
                #     x1 = int(np.round(b_points[i][1] * scale_w))
                #     y2 = int(np.round(b_points[i][2] * scale_h))
                #     x2 = int(np.round(b_points[i][3] * scale_w))
                #     if x1 == x2:
                #         angle = -np.pi / 2
                #     else:
                #         angle = np.arctan((y1 - y2) / (x1 - x2))
                #     (x1, y1), (x2, y2) = get_boundary_point(y1, x1, angle, size[0], size[1])
                #     b_points[i] = (y1, x1, y2, x2)
                #
                # # # Show current accuracy
                # # plx.scatter(x, y, rows= 17, cols = 70)
                # # plx.show()



def parse_args():
    parser = argparse.ArgumentParser(description='Training Deep Hough Network')

    # Training
    parser.add_argument('--device', default="0,1", type=str, help='device id(s) for data-parallel during training.')
    parser.add_argument('--batch_size', default=6, type=int, help='batch size for training.')
    parser.add_argument('--num_workers', default=4, type=int, help='number of workers for training.')
    parser.add_argument('--max_epoch', default=800, type=int, help='number of epoches for training.')
    parser.add_argument('--base_lr', default=0.001, type=float, help='learning rate at the beginning.')
    parser.add_argument('--show_interval', default=50, type=int, help='steps(iters) between two training logging output.')

    parser.add_argument('--backbone', default="res2net50", type=str, help='resnet18 | resnet50 | resnet101 | resnext50 | vgg16 | mobilenetv2 | res2net50')
    parser.add_argument('--num_angle', default=100, type=int, help='')
    parser.add_argument('--num_rho', default=100, type=int, help='')
    parser.add_argument('--threshold', default=0.01, type=float, help='')


    # Validating
    parser.add_argument('--val_batch_size', default=6, type=int, help='batch size for validating')
    parser.add_argument('--val_interval', default=200, type=int, help='steps(iters) between two validating phase.')


    # Datasets
    parser.add_argument('--train_gtfile', default="test/gt.txt", type=str, help='')
    parser.add_argument('--val_gtfile', default="", type=str, help='')


    # Miscs
    # parser.add_argument('--config', default="./config.yml", help="default configs")
    parser.add_argument('--save_folder', default="./save_folder", type=str, help='')


    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)