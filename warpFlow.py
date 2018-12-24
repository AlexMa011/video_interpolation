# import cv2
import torch
import numpy as np
import torch.nn.functional as F
from time import time
#
# def warp_flow(img, flow):
#     h, w = flow.shape[:2]
#     flow = -flow
#     flow[:,:,0] += np.arange(w)
#     flow[:,:,1] += np.arange(h)[:,np.newaxis]
#     res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
#     return res


def warp(img, flow):
    # torch.cuda.device(devices)

    h = 256
    w = 256
    flow_real = torch.zeros_like(flow)
    flow_real[:,0] = flow[:,0] * 1.0 *  (w-1) / (img.shape[3]-1)
    flow_real[:,1] = flow[:,1] * 1.0 *  (h-1) / (img.shape[2]-1)
    ctime = time()
    if flow.is_cuda:
        grid = meshgrid(img.shape[2], img.shape[3], img.shape[0],flow.get_device())
    else:
        grid = meshgrid(img.shape[2], img.shape[3], img.shape[0])
    # print(time() - ctime)
    # print('!!!!!!!!')
    grid = grid + flow_real * 0.02
    warp_img = torch.nn.functional.grid_sample(img, grid.permute(0, 2, 3,1))
    # print(time() - ctime)
    # print('!!!!!!!!')
    return warp_img


# def interp(im1, im2, flow1, flow2, devices, num):
#     torch.cuda.device(devices)
#     # grid = meshgrid(im1.shape[2], im1.shape[3]).cuda()
#     # flowt1 = -0.25 * flow1 + 0.25 * flow2
#     # flowt2 = 0.25 * flow1 - 0.25 * flow2
#     flowt1, flowt2 = flowproj(flow1,flow2)
#     return warp(im1, flowt1, devices, num), warp(im2, flowt2, devices, num)


def meshgrid(height, width, num,device):
    ctime = time()
    # x_t = torch.matmul(
    #     torch.ones(height, 1), torch.linspace(-1.0, 1.0, width).view(1, width)).cuda(device)
    # y_t = torch.matmul(
    #     torch.linspace(-1.0, 1.0, height).view(height, 1),
    #     torch.ones(1, width)).cuda(device)
    y_t,x_t = torch.meshgrid([torch.linspace(-1.0, 1.0, height).cuda(device),torch.linspace(-1.0, 1.0,
                                                                                                  width).cuda(device)])
    # print(torch.sum(x_newt-x_t))
    # print(torch.sum(y_newt - y_t))

    grid_x = x_t.view(1, height, width)
    grid_y = y_t.view(1, height, width)
    grid = torch.cat((grid_x, grid_y), 0)
    grid = grid.unsqueeze(0).repeat(num, 1, 1, 1)

    # print(time()-ctime)
    # print('!!!!!!!!')
    return grid

def ceil_conv(input_tensor,filters,padding):
    output = []
    c = input_tensor.shape[1]
    for i in range(c):
        output.append(F.conv2d(input_tensor[:,i:i+1],filters,padding=padding))
    return torch.cat(output,1)


def flowproj(flow1, flow2):
    num = flow1.shape[0]
    h = flow1.shape[2]
    w = flow1.shape[3]
    filters = torch.ones(1, 1, 3, 3).cuda()
    filters[:, :, 1, 1] = 0

    grid1 = meshgrid(h, w, num,flow1.get_device()).cuda()
    map1 = grid1 + flow1 / 2.0
    flowt1 = torch.zeros_like(flow1).cuda()
    cnt = torch.zeros_like(flow1).cuda()
    one = torch.ones_like(flow1).cuda()
    map1 = torch.clamp(map1, -1, 1)
    map1[:, 0] = ((map1[:, 0] + 1) / 2 * (w - 1)).int()
    map1[:, 1] = ((map1[:, 1] + 1) / 2 * (h - 1)).int()
    map1_np = map1.cpu().numpy().astype(np.int64)
    map1_f = np.ravel_multi_index((map1_np[:, 1], map1_np[:, 0]), (h, w))
    map1_f = torch.from_numpy(map1_f).cuda()
    for i in range(num):
        flowt1[i][0].put_(map1_f[i].flatten(), flow1[i][0].flatten(), accumulate=True)
        flowt1[i][1].put_(map1_f[i].flatten(), flow1[i][1].flatten(), accumulate=True)
        cnt[i][0].put_(map1_f[i].flatten(), one[i][0].flatten(), accumulate=True)
        cnt[i][1].put_(map1_f[i].flatten(), one[i][1].flatten(), accumulate=True)
    flowt1[cnt > 0] /= cnt[cnt > 0]
    flowt1 /= -2.0

    flowt1_conv = ceil_conv(flowt1, filters, padding=1)
    cnt_conv = ceil_conv((cnt > 0).float(), filters, padding=1)
    flowt1_conv[cnt_conv > 0] /= cnt_conv[cnt_conv > 0]
    flowt1[cnt == 0] = flowt1_conv[cnt == 0]

    grid2 = meshgrid(h, w, num,flow2.get_device()).cuda()
    map2 = grid2 + flow2 / 2.0
    flowt2 = torch.zeros_like(flow2).cuda()
    cnt = torch.zeros_like(flow2).cuda()
    one = torch.ones_like(flow2).cuda()
    map2 = torch.clamp(map2, -1, 1)
    map2[:, 0] = ((map2[:, 0] + 1) / 2 * (w - 1)).int()
    map2[:, 1] = ((map2[:, 1] + 1) / 2 * (h - 1)).int()
    map2_np = map2.cpu().numpy().astype(np.int64)
    map2_f = np.ravel_multi_index((map2_np[:, 1], map2_np[:, 0]), (h, w))
    map2_f = torch.from_numpy(map2_f).cuda()
    for i in range(num):
        flowt2[i][0].put_(map2_f[i].flatten(), flow2[i][0].flatten(), accumulate=True)
        flowt2[i][1].put_(map2_f[i].flatten(), flow2[i][1].flatten(), accumulate=True)
        cnt[i][0].put_(map2_f[i].flatten(), one[i][0].flatten(), accumulate=True)
        cnt[i][1].put_(map2_f[i].flatten(), one[i][1].flatten(), accumulate=True)
    flowt2[cnt > 0] /= cnt[cnt > 0]
    flowt2 /= -2.0

    flowt2_conv = ceil_conv(flowt2, filters, padding=1)
    cnt_conv = ceil_conv((cnt > 0).float(), filters, padding=1)
    flowt2_conv[cnt_conv > 0] /= cnt_conv[cnt_conv > 0]
    flowt2[cnt == 0] = flowt2_conv[cnt == 0]

    flowt1 = torch.clamp(flowt1, -2, 2)
    flowt2 = torch.clamp(flowt2, -2, 2)

    return flowt1, flowt2

