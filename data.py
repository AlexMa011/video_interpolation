import os
from collections import deque
from time import time

import cv2
import numpy as np
import torch
from torch import utils
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from pwc.run import estimate
from warpFlow import warp, flowproj


class NewDataset(Dataset):
    """
    custom defined dataset
    """

    def __init__(self, dataDir, dataList, model1, model2, transform=None,
                 devices=0, refine=False, cuth=256, cutw=400, aug=True):
        self.buffer_size = 500
        self.devices = devices
        self.transform = transform
        self.buffer_index = deque([], self.buffer_size)
        self.buffer = {}
        self.extract_feature = model1  # .cuda(self.devices)
        self.extract_flow = model2
        with open(dataList, 'r') as f:
            self.filenames = f.readlines()
        self.filenames = [fn.strip() for fn in self.filenames]
        self.num = len(self.filenames) * 2 * 3 if aug else len(self.filenames)
        self.height = 256
        self.width = 256
        self.cuth = cuth
        self.cutw = cutw
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()

        torch.cuda.device(self.devices)
        self.dataDir = dataDir
        self.refine = refine
        self.aug = aug

    def __getitem__(self, index):
        start = time()
        oindex = index
        index = index // 6 if self.aug else index
        if index not in self.buffer:
            # if True:
            if len(self.buffer_index) >= self.buffer_size:
                idd = self.buffer_index.pop()
                self.buffer.pop(idd)
            filename = self.filenames[index]
            img1 = cv2.imread(
                os.path.join(self.dataDir, '{}/im1.png'.format(filename)))
            img2 = cv2.imread(
                os.path.join(self.dataDir, '{}/im3.png'.format(filename)))
            img3 = cv2.imread(
                os.path.join(self.dataDir, '{}/im2.png'.format(filename)))
            self.buffer_index.append(index)
            self.buffer[index] = [img1, img2, img3]
        else:
            img1, img2, img3 = self.buffer[index]
        if not self.aug:
            self.cutw= img1.shape[1]
            self.cuth= img1.shape[0]
        try:
            x_start = np.random.randint(img1.shape[1] - self.cutw + 1)
            y_start = np.random.randint(img1.shape[0] - self.cuth + 1)
            img1 = img1[y_start:y_start + self.cuth,
                   x_start:x_start + self.cutw, :]
            img2 = img2[y_start:y_start + self.cuth,
                   x_start:x_start + self.cutw, :]
            img3 = img3[y_start:y_start + self.cuth,
                   x_start:x_start + self.cutw, :]
            # h = img1.shape[0]
            # w = img1.shape[1]
        except:
            print(filename)
            print(x_start)
            print(y_start)
            print('-----------')

        h = self.height
        w = self.width

        if self.aug:

            if oindex % 2 == 0:
                tmp = img1
                img1 = img2
                img2 = tmp

            if oindex % 3 == 0:
                img1 = np.fliplr(img1).copy()
                img2 = np.fliplr(img2).copy()
                img3 = np.fliplr(img3).copy()
            elif oindex % 3 == 1:
                img1 = np.flipud(img1).copy()
                img2 = np.flipud(img2).copy()
                img3 = np.flipud(img3).copy()

        img1_r = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2_r = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img3_r = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
        input1 = self.normalize(self.to_tensor(img1_r)).unsqueeze(0).cuda(0)
        input2 = self.normalize(self.to_tensor(img2_r)).unsqueeze(0).cuda(0)
        feature1 = self.extract_feature(input1)
        feature2 = self.extract_feature(input2)
        del input1
        del input2
        img1_tensor = (torch.Tensor(np.rollaxis(img1, 2, 0)[np.newaxis, :]) * 1.0 / 255.0).cuda(0)
        img2_tensor = (torch.Tensor(np.rollaxis(img2, 2, 0)[np.newaxis, :]) * 1.0 / 255.0).cuda(0)
        # img3_tensor = (torch.Tensor(np.rollaxis(img3, 2, 0)[np.newaxis, :]) * 1.0 / 255.0).cuda()

        flow12 = estimate(img1_tensor[0], img2_tensor[0], self.extract_flow, self.devices)
        flow21 = estimate(img2_tensor[0], img1_tensor[0], self.extract_flow, self.devices)

        flow12[0] /= ((w - 1) / 2.0)
        flow12[1] /= ((h - 1) / 2.0)
        flow21[0] /= ((w - 1) / 2.0)
        flow21[1] /= ((h - 1) / 2.0)
        flow12 = flow12.unsqueeze(0)
        flow21 = flow21.unsqueeze(0)

        flowt1, flowt2 = flowproj(flow12, flow21)

        max_val = max(torch.mean(abs(flow12)),torch.mean(abs(flow21)))

        flow12 /= 0.02
        flow21 /= 0.02
        flowt1 /= 0.02
        flowt2 /= 0.02



        img1_warp = warp(img1_tensor, flowt1)
        img2_warp = warp(img2_tensor, flowt2)

        img1_tensor = torch.flip(img1_tensor, [1])
        img2_tensor = torch.flip(img2_tensor, [1])
        img1_warp = torch.flip(img1_warp, [1])
        img2_warp = torch.flip(img2_warp, [1])

        img3_tensor = (torch.Tensor(np.rollaxis(img3_r, 2, 0)[np.newaxis, :]) * 1.0 / 255.0).cuda(0)

        all_img = torch.cat((img1_tensor, img2_tensor), 0)

        # mu = all_img.mean(0,keepdim=True).mean(2,keepdim=True).mean(3,keepdim=True)
        # std = torch.zeros_like(mu)
        # for idx in range(all_img.shape[1]):
        #     std[0,idx,0,0] = all_img[:,idx].std()
        # std = std + 1e-6

        mu = all_img.mean()
        std = all_img.std() + 1e-6
        img1_tensor = (img1_tensor - mu) / std
        img2_tensor = (img2_tensor - mu) / std
        img1_warp = (img1_warp - mu) / std
        img2_warp = (img2_warp - mu) / std
        img3_tensor = (img3_tensor - mu) / std




        # if self.refine:
        # inp = [img1_tensor, img2_tensor, img1_warp, img2_warp, abs(img1_tensor - img2_tensor),
        #        abs(img1_warp - img2_warp), flowt1, flowt2]
        inp = [img1_tensor, img2_tensor, img1_warp, img2_warp, flow12, flow21, flowt1, flowt2]
        if not self.refine:
            inp += [feature1, feature2]
            # feature1_warp = warp(feature1, flowt1)
            # feature2_warp = warp(feature2, flowt2)
            # feature1_warp = (feature1_warp - mu_ft) / std_ft
            # feature2_warp = (feature2_warp - mu_ft) / std_ft
            # inp = (img1_warp, img2_warp, feature1_warp, feature2_warp)
        inp = torch.cat(inp, 1)
        # label = (img3_tensor)
        # label = torch.cat(label, 1)
        return (inp[0], img3_tensor[0], [mu, std],max_val)
        # return (inp[0], img3_tensor[0], mu[0], std[0])

    def __len__(self):
        return self.num  # of how many examples(images?) you have


def load_data(dataDir, dataList, model1, model2, batch_size, devices, refine=False, shuffle=True, cuth=256, cutw=400,
              aug=True,drop_last=True):
    datas = NewDataset(dataDir, dataList, model1, model2, devices=devices, refine=refine, cuth=cuth, cutw=cutw, aug=aug)
    Dataloader = torch.utils.data.DataLoader(datas, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return Dataloader
