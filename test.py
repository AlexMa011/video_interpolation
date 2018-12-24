import torch
# from OCNet.network import get_segmentation_model
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torch import nn
import os
from data import load_data
from model import LapLoss, PerLoss, Feature, RefineNet, ANet, init_weights
from pwc.run import Network
from time import time

split_model = True
batch_size = 1
devices = 0
mode = 'syn'
name = mode + 'nnn'
log_dir = './test_logs/{}'.format(name)
model_path = 'lf.pth'
output_dir = './out'
torch.cuda.device(devices)
torch.backends.cudnn.enabled = True
torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True

net = ANet(mode=mode,test=True,use_bn=False,new=True,split_model=split_model)

# net = RefineNet(20, mode=mode, use_bn=False, test=True)
# net = GridNet(20, 5,refine=True)
# net = FlowRefine(16, 6)
# net = get_segmentation_model('resnet101_baseline',num_classes=3)
# net = UNet(12, 3)
if split_model:
    net.load_state_dict(torch.load(model_path))
else:
    net.load_state_dict(torch.load(model_path,map_location='cuda:0'))
net.float()
net.eval()




model1 = Feature().cuda().eval()
model2 = Network().cuda().eval()

data_dir = '/home/alex/data/vimeo_triplet/sequences/'
# data_list = '/home/alex/data/vimeo_triplet/tri_trainlist.txt'
data_list = '/home/alex/data/vimeo_triplet/tri_testlist.txt'
# data_list = '/home/alex/data/vimeo_triplet/demo.txt'
kk = load_data(data_dir, data_list, model1, model2, batch_size, devices, refine=False, shuffle=False,aug=False,
               drop_last=False)
# kk = load_data(data_dir, data_list, model1, model2, batch_size, devices, refine=False, shuffle=False,aug=False)

# if torch.cuda.device_count() > 1:
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     net = nn.DataParallel(net)

writer = SummaryWriter(log_dir)

mse_loss = nn.MSELoss()
mse_loss2 = nn.MSELoss(reduce=False)
# lp_loss = LapLoss()
# l1_loss = nn.L1Loss()
# per_loss = PerLoss()

end = 0
total_psnr = 0
cnt = 0
len_dl = len(kk)
idle = 20000
last_loss = 0
print(len_dl)
epsilon = 1e-6
name = 0
total_ie = 0

for index,data in enumerate(kk):

    inp = data[0]
    label = data[1]
    mu, std = data[2]
    mu = mu.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    std = std.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    cnt += 1
    max_val = data[3][0].item()
    # scheduler.step()
    # fname = datas.filenames[index]
    # fname = fname.split('/')[-1]

    xx = inp
    torch.cuda.empty_cache()
    # xx = torch.cat((inp[:,:12],flow),1)
    xx = xx.cuda(0)
    out = net(xx)

    # out1 = net(xx1)
    # out = out + out1
    # out = out / 2.0
    # out = out[0]

    if split_model:
        out = out[0]
        gt = label[:, :3].cuda(1)
        std = std.cuda(1)
        mu = mu.cuda(1)
    else:
        gt = label[:, :3]

    out = out * std + mu
    gt = gt * std + mu

    im1 = xx[:,:3].cuda(1)*std + mu
    im2 = xx[:, 3:6].cuda(1) * std + mu

    out *= 255
    out = torch.round(out)
    gt *= 255
    gt = torch.round(gt)

    im1 *= 255
    im1 = torch.round(im1)
    im2 *= 255
    im2 = torch.round(im2)

    ie = mse_loss(out,gt) ** 0.5

    out /= 255.0
    gt /= 255.0

    im1 /= 255.0
    im2 /= 255.0

    score = mse_loss2(out, gt)
    score = torch.mean(score.view(xx.shape[0], -1), dim=1)
    psnr = -10 * torch.log10(score)
    psnr = torch.mean(psnr)

    # print(max_val)
    # print(psnr.item())
    # print('----')

    if True:
    # if psnr.item()>37 and max_val>0.05:
        print(max_val)
        print(psnr.item())
        print('----')
        for i in range(out.shape[0]):
            # print(fname)
            # os.mkdir('{}/{}'.format(output_dir,fname))
            # vutils.save_image(im1[i], '{}/{}_im1.png'.format(output_dir, name), nrow=1, padding=0)
            # vutils.save_image(im2[i], '{}/{}_im2.png'.format(output_dir, name), nrow=1, padding=0)
            vutils.save_image(out[i], '{}/{}_pred1.png'.format(output_dir, name), nrow=1, padding=0)
            # vutils.save_image(out[i],'{}/{}.png'.format(output_dir,name),nrow=1,padding=0)
            # vutils.save_image(gt[i], '{}/{}_gt.png'.format(output_dir, name),nrow=1,padding=0)
            name += 1


    diff = torch.abs(out - gt)
    aa = diff ** 2


    total_psnr += psnr.item()
    total_ie += ie.item()
    step = cnt % idle
    step = idle if step == 0 else step
    writer.add_scalar('train/psnr', total_psnr / step, cnt)
    writer.add_scalar('train/ie', total_ie / step, cnt)
    if True:
    # if cnt % 2 == 0:
        pred = vutils.make_grid(out, normalize=True, scale_each=True)
        gt = vutils.make_grid(gt, normalize=True, scale_each=True)
        diff = vutils.make_grid(diff, normalize=True, scale_each=True)
        img1 = inp[:, :3, :, :]
        img2 = inp[:, 3:6, :, :]
        # wimg1 = inp[:, 6:9, :, :]
        # wimg2 = inp[:, 9:12, :, :]
        img1s = vutils.make_grid(img1, normalize=True, scale_each=True)
        img2s = vutils.make_grid(img2, normalize=True, scale_each=True)
        # wimg1s = vutils.make_grid(wimg1, normalize=True, scale_each=True)
        # wimg2s = vutils.make_grid(wimg2, normalize=True, scale_each=True)
        writer.add_image('train/pred', pred, 0)
        writer.add_image('train/gt', gt, 0)
        writer.add_image('train/img1', img1s, 0)
        writer.add_image('train/img2', img2s, 0)
        # writer.add_image('train/wimg1', wimg1s, cnt)
        # writer.add_image('train/wimg2', wimg2s, cnt)
        writer.add_image('train/diff', diff, 0)

    if cnt % idle == 0:
        total_psnr = 0


