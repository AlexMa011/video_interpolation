import numpy as np
import torch
# from OCNet.network import get_segmentation_model
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torch import nn
from torch import optim
from copy import deepcopy
from data import load_data
from model import LapLoss, PerLoss, Feature, RefineNet, init_weights, ANet
from pwc.run import Network
from data import load_data as load_test

split_model = True
batch_size = 4
val_bs = 4
lr = 1e-6
devices = 0
first_ratio = 0.01
mode = 'syn'
name = mode + 'feat4'
log_dir = './logs/{}'.format(name)
model_path = './models/{}.pth'.format(name)
torch.cuda.device(devices)
torch.backends.cudnn.enabled = True

net = ANet(mode=mode,use_bn=False,skip=False,new=True)
# net = RefineNet(20, mode=mode, use_bn=False)
# net = GridNet(20, 5,refine=True)
# net = FlowRefine(16, 6)
# net = get_segmentation_model('resnet101_baseline',num_classes=3)
net.float()
# net = UNet(12, 3)
net.apply(init_weights)

# pretrained_dict = torch.load('./models/synbn1_10000.pth')
# model_dict = net.state_dict()
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'encode' not in k}
# print(pretrained_dict.keys())
# model_dict.update(pretrained_dict)
# net.load_state_dict(model_dict)

#net.load_state_dict(torch.load('./models/synfeat1_75000.pth'))

# net.rn.load_state_dict(torch.load('./models/flownew1.pth'))
#
# net1 = ANet(mode=mode,use_bn=False,skip=True,new=True)
# net1.apply(init_weights)
# net1.rn.flownet = deepcopy(net.rn.flownet)
# net1.syn = deepcopy(net.syn)
# net1.rn = net1.rn.cuda(0)
# net1.syn = net1.syn.cuda(1)
# net = net1

if not split_model:
    net.cuda()

model1 = Feature().cuda().eval()
model2 = Network().cuda().eval()

data_dir = '/home/alex/data/vimeo_triplet/sequences/'
data_list = '/home/alex/data/vimeo_triplet/tri_trainlist.txt'
# data_list = '/home/alex/data/vimeo_triplet/tri_testlist.txt'
# data_list = '/home/alex/data/vimeo_triplet/demo.txt'
# kk = load_data(data_dir, data_list, model1, model2, batch_size, devices, refine=False)
kk = load_data(data_dir, data_list, model1, model2, batch_size, devices, refine=False, shuffle=True,cutw=384,
               aug=True)
data_list = '/home/alex/data/vimeo_triplet/tri_testlist.txt'
# data_list = '/home/alex/data/vimeo_triplet/demo.txt'
kk_val = load_test(data_dir, data_list, model1, model2, val_bs, devices, refine=False, shuffle=False,cutw=448,
               aug=False)

# if torch.cuda.device_count() > 1:
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     net = nn.DataParallel(net)

writer = SummaryWriter(log_dir)

mse_loss = nn.MSELoss()
mse_loss2 = nn.MSELoss(reduce=False)
lp_loss = LapLoss()
l1_loss = nn.L1Loss()
per_loss = PerLoss()

optimizer = optim.Adamax(net.parameters(), lr=lr)
# optimizer = optim.Adamax(
#     [
#         {'params': net.flownet.encode.parameters()},
#         {'params': net.flownet.laterals.parameters()},
#         {'params': net.flownet.downs.parameters()},
#         {'params': net.flownet.ups.parameters()},
#         {'params': net.flownet.decode.parameters(),'lr': 0.01*lr},
#     ], lr=lr)
# optimizer = optim.Adam(net.parameters(), lr=lr)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[400,4000])
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3000, cooldown=1000)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, cooldown=1)
end = 0
total_loss = 0
total_loss1 = 0
total_loss2 = 0
total_psnr = 0
cnt = 0
len_dl = len(kk)
idle = 20000
last_loss = 0
print(len_dl)
epsilon = 1e-6
best = 0
# net.flownet.decode[3].bias.requires_grad = False
for epoch in range(50):
    # if True:
    # for epoch in range(1000000 // len_dl):
    for data in kk:
        torch.cuda.empty_cache()
        net.train()
        inp = data[0]
        label = data[1]

        # mu = data[2]
        # std = data[3]

        mu, std = data[2]
        mu = mu.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        std = std.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        cnt += 1
        # scheduler.step()
        torch.set_grad_enabled(True)
        xx = inp
        # xx = torch.cat((inp[:,:12],flow),1)
        xx = xx.cuda(0)
        torch.cuda.empty_cache()
        out_list = net(xx)
        out = out_list[0]
        mid = out_list[1]
        # out = out[0]

        if split_model:
            gt = label[:, :3].cuda(1)
            std = std.cuda(1)
            mu = mu.cuda(1)
        else:
            gt = label[:, :3]

        out = out * std + mu
        mid = mid * std + mu
        gt = gt * std + mu

        score = mse_loss2(out, gt)
        score = torch.mean(score.view(xx.shape[0], -1), dim=1)
        psnr = -10 * torch.log10(score)
        psnr = torch.mean(psnr)
        total_psnr += psnr.item()

        diff = torch.abs(out - gt)

        outh = out#[diff>0.01]
        gth = gt#[diff>0.01]


        # loss1 = l1_loss(outh, gth)
        # loss2 = lp_loss(out, gt)
        # loss3l = mse_loss2(outh, gth)
        # loss3 = loss3l + epsilon ** 2
        # loss3 = loss3 ** 0.5
        # loss3 = torch.mean(loss3)
        # loss3 = 0.3 * loss3
        # loss3 = 0.5*l1_loss(mid,gt)
        # loss2 = loss3
        # loss3 = 0.005*per_loss(out, gt)
        # loss = loss1 + loss2 + loss3
        # loss = loss1 + loss2 + loss3
        # loss = per_loss(out,gt)
        loss = per_loss(out,gt)

        aa = diff ** 2
        loss.backward()
        total_loss += loss.item()
        # total_loss1 += loss1.item()
        # total_loss2 += loss2.item()

        step = cnt % idle
        step = idle if step == 0 else step
        ratio = first_ratio if epoch == 0 else 0.25
        if step > ratio * idle:
            if total_psnr / step > best:
                best = total_psnr / step
                torch.save(net.state_dict(), './models/{}.pth'.format(name))
        # if step > 0.4 * idle:
        #     scheduler.step(total_psnr / step)
        writer.add_scalar('train/loss', total_loss / step, cnt)
        # writer.add_scalar('train/l1_loss', total_loss1 / step, cnt)
        # writer.add_scalar('train/lp_loss', total_loss2 / step, cnt)
        writer.add_scalar('train/psnr', total_psnr / step, cnt)
        for param_group in optimizer.param_groups:
            lrc = param_group['lr']
            # param_group['lr'] *= 1.025
        writer.add_scalar('train/lr', -np.log10(lrc), cnt)
        # writer.add_scalar('train/lossperlr', loss.item(), np.log10(lrc)*100)
        # writer.add_scalar('train/losschange', loss.item() - last_loss, np.log10(lrc) * 100)
        # wimg1s = vutils.make_grid(wimg1, normalize=True, scale_each=True)
        # wimg2s = vutils.make_grid(wimg2, normalize=True, scale_each=True)
        optimizer.step()
        # if False:
        if cnt % 100 == 1:
            print('loss at step {}:{}'.format(cnt, total_loss / step))
        if False:
            pred = vutils.make_grid(out, normalize=True, scale_each=True)
            gt = vutils.make_grid(gt, normalize=True, scale_each=True)
            diff = vutils.make_grid(diff, normalize=True, scale_each=True)
            img1 = inp[:, :3, :, :]
            img2 = inp[:, 3:6, :, :]
            # wimg1 = inp[:, 6:9, :, :]
            # wimg2 = inp[:, 9:12, :, :]
            img1s = vutils.make_grid(img1, normalize=True, scale_each=True)
            img2s = vutils.make_grid(img2, normalize=True, scale_each=True)

            writer.add_image('train/pred', pred, 0)
            writer.add_image('train/gt', gt, 0)
            writer.add_image('train/img1', img1s, 0)
            writer.add_image('train/img2', img2s, 0)
            # writer.add_image('train/wimg1', wimg1s, cnt)
            # writer.add_image('train/wimg2', wimg2s, cnt)
            writer.add_image('train/diff', diff, 0)

        if cnt % idle == 0:
            # if False:
            total_loss = 0
            total_loss1 = 0
            total_loss2 = 0
            total_psnr = 0
        if cnt % (idle//4) == 0:
            torch.cuda.empty_cache()
            print('-----------------')
            net.eval()

            total_psnr_val = 0
            cnt_val = 0
            len_dl_val = len(kk_val)
            print(len_dl_val)
            epsilon = 1e-6

            for data in kk_val:
                inp = data[0]
                label = data[1]
                # mu = data[2]
                # std = data[3]
                mu, std = data[2]
                mu = mu.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                std = std.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                cnt_val += 1
                # scheduler.step()

                xx = inp
                torch.set_grad_enabled(False)
                # xx = torch.cat((inp[:,:12],flow),1)
                xx = xx.cuda(0)
                torch.cuda.empty_cache()
                out = net(xx)
                out = out[0]

                if split_model:
                    gt = label[:, :3].cuda(1)
                    std = std.cuda(1)
                    mu = mu.cuda(1)
                else:
                    gt = label[:, :3]

                out = out * std + mu
                gt = gt * std + mu

                out *= 255
                out = torch.round(out)
                gt *= 255
                gt = torch.round(gt)

                out /= 255.0
                gt /= 255.0

                # for i in range(out.shape[0]):
                #     vutils.save_image(out[i], '{}/{}_pred.png'.format(output_dir, name), nrow=1, padding=0)
                #     vutils.save_image(gt[i], '{}/{}_gt.png'.format(output_dir, name), nrow=1, padding=0)
                #     name += 1

                diff = torch.abs(out - gt)
                aa = diff ** 2

                score = mse_loss2(out, gt)
                score = torch.mean(score.view(xx.shape[0], -1), dim=1)
                psnr = -10 * torch.log10(score)
                psnr = torch.mean(psnr)
                total_psnr_val += psnr.item()
                step = cnt_val
                # step = idle if step == 0 else step

                # if cnt_val % 100 == 1:
                #     pred = vutils.make_grid(out, normalize=True, scale_each=True)
                #     gt = vutils.make_grid(gt, normalize=True, scale_each=True)
                #     diff = vutils.make_grid(diff, normalize=True, scale_each=True)
                #     img1 = inp[:, :3, :, :]
                #     img2 = inp[:, 3:6, :, :]
                #     # wimg1 = inp[:, 6:9, :, :]
                #     # wimg2 = inp[:, 9:12, :, :]
                #     img1s = vutils.make_grid(img1, normalize=True, scale_each=True)
                #     img2s = vutils.make_grid(img2, normalize=True, scale_each=True)
                #     # wimg1s = vutils.make_grid(wimg1, normalize=True, scale_each=True)
                #     # wimg2s = vutils.make_grid(wimg2, normalize=True, scale_each=True)
                #     writer.add_image('val/pred', pred, 0)
                #     writer.add_image('val/gt', gt, 0)
                #     writer.add_image('val/img1', img1s, 0)
                #     writer.add_image('val/img2', img2s, 0)
                #     # writer.add_image('train/wimg1', wimg1s, cnt+cnt_val)
                #     # writer.add_image('train/wimg2', wimg2s, cnt+cnt_val)
                #     writer.add_image('val/diff', diff, 0)
            # scheduler.step(total_psnr_val / cnt_val)
            writer.add_scalar('val/psnr', total_psnr_val / step, cnt + cnt_val)

            print('-----------------')
            torch.save(net.state_dict(), './models/{}_{}.pth'.format(name, cnt))


        # cnts = 0
        # alls = 0
        # alls2 = 0
        #
        # for name, param in net.flownet.named_parameters():
        #     if param.requires_grad:
        #         # alls += abs(param).mean().item()
        #         # alls2 += abs(param.grad).mean().item()
        #         alls = (abs(param.grad/param)).mean().item()
        #         print(name)
        #         print(alls)
        #         print('--------')
        #         # cnts += 1
        # print(alls/cnts)
        # print(alls2/cnts)
        # print('-------')
        # break
