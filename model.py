import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from warpFlow import meshgrid, warp
import time

def init_weights(m):
    if type(m) == nn.Linear:
        # torch.nn.init.normal_(m.weight)
        # torch.nn.init.kaiming_normal_(m.weight)
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(1e-6)
    elif type(m) == nn.Conv2d:
        # torch.nn.init.normal_(m.weight)
        # torch.nn.init.kaiming_normal_(m.weight)
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(1e-6)


def build_gauss_kernel(size=5, sigma=1.0, n_channels=1, cuda=False, device=0):
    if size % 2 != 1:
        raise ValueError("kernel size must be uneven")
    grid = np.float32(np.mgrid[0:size, 0:size].T)
    gaussian = lambda x: np.exp((x - size // 2) ** 2 / (-2 * sigma ** 2)) ** 2
    kernel = np.sum(gaussian(grid), axis=2)
    kernel /= np.sum(kernel)
    # repeat same kernel across depth dimension
    kernel = np.tile(kernel, (n_channels, 1, 1))
    # conv weight should be (out_channels, groups/in_channels, h, w),
    # and since we have depth-separable convolution we want the groups
    # dimension to be 1
    kernel = torch.FloatTensor(kernel[:, None, :, :])
    if cuda:
        kernel = kernel.cuda(device)
    kernel.requires_grad = True
    return kernel
    # return Variable(kernel, requires_grad=False)


def conv_gauss(img, kernel):
    """ convolve img with a gaussian kernel that has been built with
    build_gauss_kernel """
    n_channels, _, kw, kh = kernel.shape
    img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
    return F.conv2d(img, kernel, groups=n_channels)


def laplacian_pyramid(img, kernel, max_levels=5):
    current = img
    pyr = []

    for level in range(max_levels):
        filtered = conv_gauss(current, kernel)
        diff = current - filtered
        pyr.append(diff)
        current = F.avg_pool2d(filtered, 2)

    # pyr.append(current)
    return pyr


class LapLoss(nn.Module):
    def __init__(self, max_levels=5, k_size=5, sigma=2.0, split_model=False):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.k_size = k_size
        self.sigma = sigma
        self._gauss_kernel = None
        self.loss = nn.L1Loss()
        self.split_model = split_model

    def forward(self, input, target):
        if self._gauss_kernel is None or self._gauss_kernel.shape[1] != \
            input.shape[1]:
            self._gauss_kernel = build_gauss_kernel(size=self.k_size, sigma=self.sigma, n_channels=input.shape[1],
                                                    cuda=input.is_cuda, device=input.get_device())
        pyr_input = laplacian_pyramid(input, self._gauss_kernel,
                                      self.max_levels)
        pyr_target = laplacian_pyramid(target, self._gauss_kernel,
                                       self.max_levels)
        values = [self.loss(a, b) for a, b in zip(pyr_input, pyr_target)]
        for i in range(self.max_levels):
            values[i] *= (2.0 ** i)
        values = [values[i] / pyr_target[i].nelement() for i in
                  range(len(pyr_target))]
        # values = [torch.log(x) for x in values]
        return sum(values) / self.max_levels


class PerLoss(nn.Module):
    def __init__(self):
        super(PerLoss, self).__init__()
        self.loss = nn.L1Loss()
        self.feature = list(models.vgg16(pretrained=True).children())[0][:21].cuda(1)
        for param in self.feature.parameters():
            param.requires_grad = False
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).unsqueeze(0).unsqueeze(2).unsqueeze(3).cuda(1)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).unsqueeze(0).unsqueeze(2).unsqueeze(3).cuda(1)

    def forward(self, input, target):
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        input_ft = self.feature(input)
        target_ft = self.feature(target)
        return self.loss(input_ft, target_ft)


class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()
        first_layer = list(models.resnet18(pretrained=True).children())[0]
        self.layer = nn.Conv2d(3, 64, kernel_size=7, padding=3, bias=False)
        self.layer.weight.data.copy_(first_layer.weight.data)
        self.layer.requires_grad = False

    def forward(self, image):
        return self.layer(image)


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch, kernel_size):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            # nn.PReLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=(kernel_size - 1)
                                                          // 2),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(),
            # nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size, padding=(kernel_size - 1)
                                                           // 2),
            nn.BatchNorm2d(out_ch),
            nn.PReLU()
            # nn.ReLU(inplace=True)
        )
        self.convr = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        xo = self.conv(x)
        xr = self.convr(x)
        return xo + xr


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch, kernel_size=kernel_size)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            # nn.MaxPool2d(2),
            nn.AvgPool2d(kernel_size=2, stride=2),
            double_conv(in_ch, out_ch, kernel_size=kernel_size)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear',
                                  align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch, kernel_size=kernel_size)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        super(outconv, self).__init__()
        self.oconv = double_conv(in_ch, out_ch, kernel_size=kernel_size)

    def forward(self, x):
        x = self.oconv(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64, 3)
        self.down1 = down(64, 128, 3)
        self.down2 = down(128, 256, 3)
        self.down3 = down(256, 512, 3)
        self.down4 = down(512, 512, 3)
        self.up1 = up(1024, 256, 3)
        self.up2 = up(512, 128, 3)
        self.up3 = up(256, 64, 3)
        self.up4 = up(128, 64, 3)
        self.outc = outconv(64, n_classes, 3)
        self.convl = nn.Conv2d(n_classes, n_classes, 3, padding=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        xo = self.up1(x5, x4)
        xo = self.up2(xo, x3)
        xo = self.up3(xo, x2)
        xo = self.up4(xo, x1)
        xo = self.outc(xo)
        return xo


class LateralBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, use_bn=False, test=False):
        super(LateralBlock, self).__init__()
        if use_bn:
            self.block = nn.Sequential(
                nn.BatchNorm2d(in_ch),
                nn.PReLU(),
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
                nn.BatchNorm2d(out_ch),
                nn.PReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            )
        else:
            self.block = nn.Sequential(
                nn.PReLU(),
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
                nn.PReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            )
        self.p = torch.distributions.bernoulli.Bernoulli(torch.Tensor([0.9]))
        self.test = test

    def forward(self, x):
        xo = self.block(x)
        p = self.p.sample().item()
        if self.test:
            return x + xo
        else:
            return x + xo * p


class DownSamplingBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, use_bn=False):
        super(DownSamplingBlock, self).__init__()
        if use_bn:
            self.block = nn.Sequential(
                nn.BatchNorm2d(in_ch),
                nn.PReLU(),
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, stride=2),
                nn.BatchNorm2d(out_ch),
                nn.PReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
            )
        else:
            self.block = nn.Sequential(
                nn.PReLU(),
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, stride=2),
                nn.PReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
            )

    def forward(self, x):
        return self.block(x)


class UpSamplingBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, use_bn=False):
        super(UpSamplingBlock, self).__init__()
        if use_bn:
            self.block = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.BatchNorm2d(in_ch),
                nn.PReLU(),
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
                nn.BatchNorm2d(out_ch),
                nn.PReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
            )
        else:
            self.block = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.PReLU(),
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
                nn.PReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
            )

    def forward(self, x):
        return self.block(x)


class GridNet(nn.Module):
    def __init__(self, in_ch, out_ch, nrows=3, ncols=6, channels=[32, 64, 96], use_bn=False, test=False, skip=False):
        super(GridNet, self).__init__()
        self.nrows = nrows
        self.ncols = ncols
        self.skip = skip
        if skip:
            chann = channels[0] + 3
            ratio = 4
        else:
            chann = channels[0]
            ratio = 1
        if use_bn:
            self.encode = nn.Sequential(
                nn.Conv2d(in_ch, channels[0], kernel_size=5, padding=2),
                nn.BatchNorm2d(channels[0]),
                nn.PReLU(),
                nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=1),
                nn.BatchNorm2d(channels[0]),
                nn.PReLU(),
                nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=1),
            )
            self.decode = nn.Sequential(
                nn.BatchNorm2d(channels[0]),
                nn.PReLU(),
                nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=1),
                nn.BatchNorm2d(channels[0]),
                nn.PReLU(),
                nn.Conv2d(channels[0], out_ch, kernel_size=3, padding=1),
            )
        else:
            self.encode = nn.Sequential(
                nn.Conv2d(in_ch, channels[0], kernel_size=5, padding=2),
                nn.PReLU(),
                nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=1),
                nn.PReLU(),
                nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=1),
            )
            self.decode = nn.Sequential(
                nn.PReLU(),
                nn.Conv2d(chann, channels[0]//ratio, kernel_size=3, padding=1),
                nn.PReLU(),
                nn.Conv2d(channels[0]//ratio, out_ch, kernel_size=3, padding=1),
            )
        laterals = [[None] * (ncols - 1) for _ in range(nrows)]
        for i in range(nrows):
            for j in range(ncols - 1):
                laterals[i][j] = LateralBlock(channels[i], channels[i], 3, use_bn=use_bn, test=test)
        downs = [[None] * (ncols // 2) for _ in range(nrows - 1)]
        ups = [[None] * (ncols // 2) for _ in range(nrows - 1)]
        for i in range(nrows - 1):
            for j in range(ncols // 2):
                downs[i][j] = DownSamplingBlock(channels[i], channels[i + 1], 3, use_bn=use_bn)
                ups[i][j] = UpSamplingBlock(channels[i + 1], channels[i], 3, use_bn=use_bn)
        for i in range(len(laterals)):
            laterals[i] = nn.ModuleList(laterals[i])
        for i in range(len(downs)):
            downs[i] = nn.ModuleList(downs[i])
        for i in range(len(ups)):
            ups[i] = nn.ModuleList(ups[i])
        self.laterals = nn.ModuleList(laterals)
        self.downs = nn.ModuleList(downs)
        self.ups = nn.ModuleList(ups)

    def forward(self, x):
        xi = self.encode(x)
        xs = [xi]
        for i in range(self.nrows - 1):
            xi = self.downs[i][0](xi)
            xs.append(xi)
        half = self.ncols // 2 - 1

        for j in range(self.ncols - 1):
            if j < self.ncols // 2 - 1:
                xi = self.laterals[0][j](xs[0])
                new_xs = [xi]
                for i in range(1, self.nrows):
                    xi = self.laterals[i][j](xs[i]) + self.downs[i - 1][j + 1](xi)
                    new_xs.append(xi)
                xs = new_xs
            else:
                xi = self.laterals[-1][j](xs[-1])
                new_xs = [xi]
                for i in range(self.nrows - 2, -1, -1):
                    xi = self.laterals[i][j](xs[i]) + self.ups[i][j - half](xi)
                    new_xs.append(xi)
                xs = list(reversed(new_xs))
        if self.skip:
            inp_t = torch.cat((xs[0],x[:,6:9]),1)
        else:
            inp_t = xs[0]
        xo = self.decode(inp_t)
        return xo


class RefineNet(nn.Module):
    def __init__(self, in_ch, mode='flow', use_bn=False, deep=False, all=False, test=False, new=False):
        super(RefineNet, self).__init__()
        mask_num = 2 if new else 1
        self.new = new
        if deep:
            self.flownet = GridNet(in_ch, 4, nrows=4, channels=[32, 64, 96, 128], use_bn=False, test=test)
            self.masknet = GridNet(in_ch, mask_num, ncols=4, use_bn=use_bn, test=test)
        else:
            self.flownet = GridNet(in_ch, 4, use_bn=False, test=test)
            self.masknet = GridNet(in_ch, mask_num, use_bn=use_bn, test=test)
        if mode == 'flow':
            for param in self.masknet.parameters():
                param.requires_grad = False
        elif mode == 'mask':
            for param in self.flownet.parameters():
                param.requires_grad = False
        self.mode = mode
        self.all = all

    def forward(self, x):
        ctime = time.perf_counter()
        x = x[:,:20]
        flow = self.flownet(x)
        # print(x.std().item())
        # print(flow.std().item())
        # print(abs(x).mean().item())
        # print(abs(flow).mean().item())
        # print('----')
        flow = flow / 100
        flowt1 = x[:, -4:-2] + flow[:, :2]
        flowt2 = x[:, -2:] + flow[:, 2:]


        mask = self.masknet(torch.cat((x[:, :16], flowt1, flowt2), 1))


        mask = mask + 0.5
        mask = torch.clamp(mask,min=0,max=1)


        # mask = torch.sigmoid(mask)
        mask1 = mask[:,:1]
        mask2 = mask[:,1:]

        flowt1 = flowt1*2*mask1
        flowt2 = flowt2*2*(1-mask1)





        im1_warp = warp(x[:, :3], flowt1)
        im2_warp = warp(x[:, 3:6], flowt2)

        if self.mode == 'flow':
            mask2 = 0.5
        # else:
            # mask = self.masknet(torch.cat((x[:, :6], im1_warp, im2_warp, x[:, 12:16], flowt1, flowt2), 1))
            # mask = torch.sigmoid(mask)
            # mask = 0.5

        ret = im1_warp * mask2 + im2_warp * (1 - mask2)
        if self.all:
            return [ret, flowt1, flowt2, mask2]
        else:
            return ret


class ANet(nn.Module):
    def __init__(self, mode, use_bn=False, test=False, skip=False, new=False,split_model=True):
        super(ANet, self).__init__()
        if not split_model:
            self.device = 0
        else:
            self.device = 1
        self.rn = RefineNet(20, mode='all', use_bn=False, all=True, test=test, new=new)
        self.rn = self.rn.cuda(0)
        self.syn = GridNet(201, 3, use_bn=use_bn, test=test, skip=skip)
        self.syn = self.syn.cuda(self.device)
        if mode == 'pre':
            for param in self.rn.parameters():
                param.requires_grad = False
        self.mode = mode
        self.split_model = split_model

    def forward(self, x):
        ctime = time.perf_counter()
        inp1 = x[:, :20]

        out, flowt1, flowt2, mask = self.rn(inp1)
        inp2 = x[:, 20:]

        feature1 = inp2[:, :64]
        feature2 = inp2[:, 64:]

        all_ft = torch.cat((feature1, feature2), 0)


        # mu_ft = all_ft.mean(0,keepdim=True).mean(2,keepdim=True).mean(3,keepdim=True)
        # std_ft = torch.zeros_like(mu_ft)
        # for idx in range(all_ft.shape[1]):
        #     std_ft[0,idx,0,0] = all_ft[:,idx].std()
        # std_ft = std_ft + 1e-6
        mu_ft = all_ft.mean()
        std_ft = all_ft.std() + 1e-5

        feature1_warp = warp(feature1, flowt1)
        feature2_warp = warp(feature2, flowt2)
        im1_warp = warp(x[:, :3], flowt1)
        im2_warp = warp(x[:, 3:6], flowt2)
        feature1_warp = (feature1_warp - mu_ft) / std_ft
        feature2_warp = (feature2_warp - mu_ft) / std_ft
        feature = feature1_warp * mask + feature2_warp * (1 - mask)

        inp2r = torch.cat([im1_warp, im2_warp, out, feature1_warp, feature2_warp, feature], 1)
        # inp2r = torch.cat([im1_warp, im2_warp, feature1_warp, feature2_warp, mask], 1)
        # inp2r = torch.cat([im1_warp, im2_warp, out, feature], 1)
        if self.split_model:
            out = out.cuda(self.device)
            inp2r = inp2r.cuda(self.device)
        if self.mode == 'pre':
            inp2r.requires_grad = False
        outf = self.syn(inp2r)
        outf = out + outf / 50.0

        if not self.split_model:
            return outf
        else:
            return [outf,out]
