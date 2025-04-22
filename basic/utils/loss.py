import torch.nn as nn
import torch
import torch.nn.functional as F

class AMPLoss(torch.nn.Module):
    def __init__(self):
        super(AMPLoss, self).__init__()
        self.cri = torch.nn.L1Loss()

    def forward(self, x, y):
        x = torch.fft.rfft2(x, norm='backward')
        x_mag =  torch.abs(x)
        y = torch.fft.rfft2(y, norm='backward')
        y_mag = torch.abs(y)

        return self.cri(x_mag,y_mag)


class PhaLoss(torch.nn.Module):
    def __init__(self):
        super(PhaLoss, self).__init__()
        self.cri = torch.nn.L1Loss()

    def forward(self, x, y):
        x = torch.fft.rfft2(x, norm='backward')
        x_mag = torch.angle(x)
        y = torch.fft.rfft2(y, norm='backward')
        y_mag = torch.angle(y)

        return self.cri(x_mag, y_mag)
    
    
class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class SSIMLoss(torch.nn.Module):
    def __init__(self, win_size=11, win_sigma=1.5, data_range=None, size_average=True, channel=3):
        r""" class for ssim
        Args:
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            channel (int, optional): input channels (default: 3)
        """

        super(SSIMLoss, self).__init__()
        self.win = _fspecial_gauss_1d(
            win_size, win_sigma).repeat(channel, 1, 1, 1)
        self.size_average = size_average
        self.data_range = data_range

    def forward(self, X, Y):
        if X.ndimension() == 5:
            X = X[:,0,...]
            Y = Y[:,0,...]
        return 1-ssim(X, Y, win=self.win, data_range=self.data_range, size_average=self.size_average)

class SAMLoss(torch.nn.Module):
    def __init__(self, size_average = False):
        super(SAMLoss, self).__init__()

    def forward(self, img_base, img_out):
        if img_base.ndimension() == 5:
            img_base = img_base[:,0,...]
        if img_out.ndimension() == 5:
            img_out = img_out[:,0,...]
        sum1 = torch.sum(img_base * img_out, 1)
        sum2 = torch.sum(img_base * img_base, 1)
        sum3 = torch.sum(img_out * img_out, 1)
        t = (sum2 * sum3) ** 0.5
        numlocal = torch.gt(t, 0)
        num = torch.sum(numlocal)
        t = sum1 / t
        angle = torch.acos(t)
        sumangle = torch.where(torch.isnan(angle), torch.full_like(angle, 0), angle).sum()
        if num == 0:
            averangle = sumangle
        else:
            averangle = sumangle / num
        SAM = averangle * 180 / 3.14159256
        return SAM

class MultipleLoss(nn.Module):
    def __init__(self, losses, weight=None):
        super(MultipleLoss, self).__init__()
        self.losses = nn.ModuleList(losses)
        self.weight = weight or [1/len(self.losses)] * len(self.losses)
    
    def forward(self, predict, target):
        total_loss = 0
        for weight, loss in zip(self.weight, self.losses):
            total_loss += loss(predict, target) * weight
        return total_loss

    def extra_repr(self):
        return 'weight={}'.format(self.weight)


def get_loss(type):
    
    if type == 'l2':
        return nn.MSELoss()
    if type == 'l1':
        return nn.L1Loss()
    if type == 'smooth_l1':
        return nn.SmoothL1Loss()
    if type == 'ssim':
        return SSIMLoss(data_range=1, channel=31)
    if type == 'l2_ssim':
        return MultipleLoss([nn.MSELoss(), SSIMLoss(data_range=1, channel=31)], weight=[1, 2.5e-3])
    if type == 'l2_sam':
        return MultipleLoss([nn.MSELoss(),SAMLoss()],weight=[1, 1e-3])
    if type == 'fidloss':
        return MultipleLoss([nn.MSELoss(),AMPLoss(),PhaLoss()],weight=[1,0.01,0.01])


def _fspecial_gauss_1d(size, sigma):
    
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size//2

    g = torch.exp(-(coords**2) / (2*sigma**2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)

def ssim(X, Y, win_size=11, win_sigma=1.5, win=None, data_range=255, size_average=True, full=False):

    if len(X.shape) != 4:
        raise ValueError('Input images must 4-d tensor.')

    if not X.type() == Y.type():
        raise ValueError('Input images must have the same dtype.')

    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    win_sigma = win_sigma
    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(X.shape[1], 1, 1, 1)
    else:
        win_size = win.shape[-1]

    ssim_val, cs = _ssim(X, Y,
                         win=win,
                         data_range=data_range,
                         size_average=False,
                         full=True)
    if size_average:
        ssim_val = ssim_val.mean()
        cs = cs.mean()

    if full:
        return ssim_val, cs
    else:
        return ssim_val
    
def _ssim(X, Y, win, data_range=255, size_average=True, full=False):

    K1 = 0.01
    K2 = 0.03
    batch, channel, height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range)**2
    C2 = (K2 * data_range)**2

    concat_input = torch.cat([X, Y, X*X, Y*Y, X*Y], dim=1)
    concat_win = win.repeat(5, 1, 1, 1).to(X.device, dtype=X.dtype)
    concat_out = gaussian_filter(concat_input, concat_win)
    
    mu1, mu2, sigma1_sq, sigma2_sq, sigma12 = (
        concat_out[:, idx*channel:(idx+1)*channel, :, :] for idx in range(5))

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (sigma1_sq - mu1_sq)
    sigma2_sq = compensation * (sigma2_sq - mu2_sq)
    sigma12 = compensation * (sigma12 - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    if size_average:
        ssim_val = ssim_map.mean()
        cs = cs_map.mean()
    else:
        ssim_val = ssim_map.mean(-1).mean(-1).mean(-1)  # reduce along CHW
        cs = cs_map.mean(-1).mean(-1).mean(-1)

    if full:
        return ssim_val, cs
    else:
        return ssim_val
    
def gaussian_filter(input, win):
    
    N, C, H, W = input.shape
    out = F.conv2d(input, win, stride=1, padding=0, groups=C)
    out = out.transpose(2, 3).contiguous()
    out = F.conv2d(out, win, stride=1, padding=0, groups=C)
    return out.transpose(2, 3).contiguous()