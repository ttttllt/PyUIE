import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Callable
import torch.nn.init as init
from typing import Union
import torchvision
import numbers
from einops import rearrange


#############################################common#############################################
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


def catcat(inputs1, inputs2):
    return torch.cat((inputs1, inputs2), 2)


def get_conv2d_layer(in_c, out_c, k, s, p=0, dilation=1, groups=1):
    return nn.Conv2d(in_channels=in_c,
                     out_channels=out_c,
                     kernel_size=k,
                     stride=s,
                     padding=p, dilation=dilation, groups=groups)


class Conv2d(nn.Conv2d):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, *,
                 stride: int = 1, padding: Union[str, int] = "preserve", dilation: int = 1, groups: int = 1,
                 bias: bool = True, padding_mode: str = "zeros",
                 init_weights: Callable = init.xavier_normal_, device=None, dtype=None):
        if padding == "preserve":
            assert stride == 1, "Preserve padding only works with stride=1"
            assert (dilation * (kernel_size - 1)) % 2 == 0, "Preserve padding only works with odd kernel sizes"
            padding = (dilation * (kernel_size - 1)) // 2
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, dilation=dilation, groups=groups,
                         bias=bias, padding_mode=padding_mode, device=device, dtype=dtype)
        init_weights(self.weight)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

#############################################NeuralPreset Net#############################################
class MobileNetV3Large(nn.Module):
    def __init__(self, pretrained=False, weights_path="weight.pth"):
        super().__init__()

        if pretrained and weights_path is not None:
            self.model = torchvision.models.mobilenet_v3_large(pretrained=False)
            state_dict = torch.load(weights_path)
            self.model.load_state_dict(state_dict)
        else:
            self.model = torchvision.models.mobilenet_v3_large(pretrained=pretrained)


        del self.model.classifier

    def forward(self, x):
        return self.model.features(x)


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(self.in_channels, self.out_channels)

    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class NCMPredictor(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.fc1 = torch.nn.Linear(self.in_channels, self.embed_dim ** 2, bias=False)
        self.fc2 = torch.nn.Linear(self.embed_dim ** 2, self.embed_dim ** 2, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class NCMPerformer(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

        self.encoder = torch.nn.Parameter(torch.randn(1, 3, self.embed_dim), requires_grad=True)
        self.decoder = torch.nn.Parameter(torch.randn(1, self.embed_dim, 3), requires_grad=True)

    def forward(self, x, ncm):
        n = x.size(0)

        x = torch.bmm(x, self.encoder.expand(n, -1, -1))
        x = torch.tanh(x)

        x = torch.bmm(x, ncm.view(n, self.embed_dim, self.embed_dim))
        x = torch.tanh(x)

        x = torch.bmm(x, self.decoder.expand(n, -1, -1))
        x = torch.sigmoid(x)
        return x


class NeuralPreset(nn.Module):
    def __init__(self, input_size=384, embed_dim=16, latent_channels=640, pretrained=True):
        super().__init__()
        # arguments
        self.input_size = input_size
        self.embed_dim = embed_dim
        self.latent_channels = latent_channels
        self.pretrained = pretrained

        # components
        backbone_channels = 960
        self.bottleneck = Bottleneck(backbone_channels, self.latent_channels)
        self.nncm_predictor = NCMPredictor(self.latent_channels, self.embed_dim)
        self.nncm_performer = NCMPerformer(self.embed_dim)
        self._init_weights()
        self.backbone = MobileNetV3Large(pretrained=self.pretrained)

    def _init_weights(self):
        # NOTE: initialization of the nn.Linear layer is critical for converging
        #       using the default initialization of torch 1.9.0 is fine
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                if m.weight is not None:
                    torch.nn.init.constant_(m.weight, 1)
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Parameter):
                torch.nn.init.kaiming_uniform_(m.data, a=0, mode='fan_in', nonlinearity='relu')

    def _extract_features(self, x):
        x = F.interpolate(x, (self.input_size, self.input_size), mode='bilinear', align_corners=True)
        x = self.backbone(x)
        x = self.bottleneck(x)
        return x

    def forward(self, img_l, img):
        n, c, h, w = img.shape

        feature = self._extract_features(img_l)
        nncm = self.nncm_predictor(feature)
        img = img.reshape(n, c, -1).permute(0, 2, 1)

        img = self.nncm_performer(img, nncm)
        img = self.nncm_performer(img, nncm)
        img = self.nncm_performer(img, nncm)
        img = self.nncm_performer(img, nncm)
        img = self.nncm_performer(img, nncm)
        img = self.nncm_performer(img, nncm)
        img = img.permute(0, 2, 1).view(n, c, h, w)

        return img


##########################################################################
## SE Block
class SEBlock(nn.Module):
    def __init__(self, inplanes, reduction=16):
        super(SEBlock, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inplanes, inplanes // reduction, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes // reduction, inplanes, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        input = x
        x = self.se(x)
        return input * x


class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out


##  adaptive hybrid attention
class AHAttention(nn.Module):  # adaptive hybrid attention
    def __init__(self, dim):
        super(AHAttention, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d((None, 1))
        self.max = nn.AdaptiveMaxPool2d((1, None))
        self.conv1x1 = nn.Conv2d(dim, dim // 2, kernel_size=1, padding=(1 // 2), bias=True)
        self.conv3x3 = nn.Conv2d(dim // 2, dim, kernel_size=3, padding=(3 // 2), bias=True)
        self.con3x3 = nn.Conv2d(dim, dim, kernel_size=3, padding=(3 // 2), bias=True)
        self.GELU = nn.GELU()
        self.mix1 = Mix(m=-1)
        self.mix2 = Mix(m=-0.6)

    def forward(self, x):
        batch_size, channel, height, width = x.size()
        x_h = self.avg(x)
        x_w = self.max(x)
        x_h = torch.squeeze(x_h, 3)
        x_w = torch.squeeze(x_w, 2)
        x_h1 = x_h.unsqueeze(3)
        x_w1 = x_w.unsqueeze(2)
        x_h_w = torch.cat((x_h, x_w), 2)
        x_h_w = x_h_w.unsqueeze(3)
        x_h_w = self.conv1x1(x_h_w)
        x_h_w = self.GELU(x_h_w)
        x_h_w = torch.squeeze(x_h_w, 3)
        x1, x2 = torch.split(x_h_w, [height, width], 2)
        x1 = x1.unsqueeze(3)
        x2 = x2.unsqueeze(2)
        x1 = self.conv3x3(x1)
        x2 = self.conv3x3(x2)
        mix1 = self.mix1(x_h1, x1)
        mix2 = self.mix2(x_w1, x2)
        x1 = self.con3x3(mix1)
        x2 = self.con3x3(mix2)
        matrix = torch.matmul(x1, x2)
        matrix = torch.sigmoid(matrix)
        final = torch.mul(x, matrix)
        final = x + final
        return final



#############################################LPTN Model#############################################





class GDFN(nn.Module):
    def __init__(self, dim):
        super(GDFN, self).__init__()

        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1, bias=True)
        self.conv1_1 = nn.Conv2d(dim, dim, 3, padding=1, bias=True)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(dim, dim, 1, bias=True)
        self.conv2_2 = nn.Conv2d(dim, dim, 1, bias=True)
        self.act2 = nn.GELU()
        self.conv3 = nn.Conv2d(dim, dim, 3, padding=1, bias=True)
        self.attention = AHAttention(dim)

    def forward(self, x):
        res1 = self.act1(self.conv1(x))
        res1 = self.act1(self.conv1_1(x))
        res2 = self.act2(self.conv2(x))
        res2 = self.act2(self.conv2_2(x))
        res = res1 + res2
        res = x + res
        res = self.attention(res)
        res = self.conv3(res)
        res = x + res
        return res


##########################################################################

class MultiGDFN(nn.Module):
    def __init__(self, num_blocks, dim, bias=False):
        super(MultiGDFN, self).__init__()

        self.num_blocks = num_blocks
        self.dim = dim
        self.blocks = nn.ModuleList([GDFN(dim) for _ in range(num_blocks)])
        self.se = SEBlock(dim * num_blocks)
        self.gelu = nn.GELU()
        self.project_out = nn.Conv2d(dim * num_blocks, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        outputs = []
        for block in self.blocks:
            x = block(x)
            outputs.append(x)
        x_concat = self.gelu(torch.cat(outputs, dim=1))
        output = self.project_out(self.se(x_concat))
        return output


##########################################################################
#####Lap_Pyramid#####
class Lap_Pyramid_Conv(nn.Module):
    def __init__(self, num_high=3):
        super(Lap_Pyramid_Conv, self).__init__()

        self.num_high = num_high
        self.kernel = self.gauss_kernel()

    def gauss_kernel(self, device=torch.device('cuda:0'), channels=3):
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]])
        kernel /= 256.
        kernel = kernel.repeat(channels, 1, 1, 1)
        kernel = kernel.to(device)
        return kernel

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def upsample(self, x):
        cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
        cc = cc.permute(0, 1, 3, 2)
        cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
        x_up = cc.permute(0, 1, 3, 2)
        return self.conv_gauss(x_up, 4 * self.kernel)

    def conv_gauss(self, img, kernel):
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
        return out

    def pyramid_decom(self, img):
        current = img
        pyr = []
        for _ in range(self.num_high):
            filtered = self.conv_gauss(current, self.kernel)
            down = self.downsample(filtered)
            up = self.upsample(down)
            if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
                up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
            diff = current - up
            pyr.append(diff)
            current = down
        pyr.append(current)
        return pyr

    def pyramid_recons(self, pyr):
        image = pyr[-1]
        for level in reversed(pyr[:-1]):
            up = self.upsample(image)
            if up.shape[2] != level.shape[2] or up.shape[3] != level.shape[3]:
                up = nn.functional.interpolate(up, size=(level.shape[2], level.shape[3]))
            image = up + level
        return image


#####Low Map#####
class Trans_low(nn.Module):
    def __init__(self):
        super(Trans_low, self).__init__()
        self.neural_preset = NeuralPreset(input_size=64)

    def forward(self, x, low):
        out = self.neural_preset(x, low)
        out = torch.tanh(out)
        return out


#####High Map#####
class Trans_high(nn.Module):
    def __init__(self, num_high=3, n_feats=64, num_blocks=3):
        super(Trans_high, self).__init__()

        self.num_high = num_high
        self.num_blocks = num_blocks
        self.conv = nn.Conv2d(3, n_feats, kernel_size=3, padding=1)

        self.DCSEB1 = nn.Sequential(
            nn.Conv2d(3, n_feats, kernel_size=3, padding=1),
            nn.GELU(),
            SELayer(n_feats),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
            nn.GELU()
        )

        self.DCSEB = nn.Sequential(
            nn.Conv2d(6, n_feats, kernel_size=3, padding=1),
            nn.GELU(),
            SELayer(n_feats),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
            nn.GELU()
        )

        self.multigdfn = MultiGDFN(num_blocks=num_blocks, dim=n_feats)

       
        for i in range(self.num_high):
            DCSEB2 = nn.Sequential(
                nn.Conv2d(n_feats, 16, 1),
                nn.GELU(),
                SELayer(16),
                nn.Conv2d(16, 3, 1),
                nn.GELU())
            setattr(self, 'DCSEB2_{}'.format(str(i)), DCSEB2)
            
    def forward(self, x, pyr_original, fake_low):
        pyr_result = []
        x = self.DCSEB(x)
        for i in range(self.num_high):
            x = self.multigdfn(x)
            mask = x
            pyr_original_adjusted = self.DCSEB1(pyr_original[-2 - i])
            mask = nn.functional.interpolate(mask,
                                             size=(pyr_original_adjusted.shape[2], pyr_original_adjusted.shape[3]))

            result_highfreq = torch.mul(pyr_original_adjusted, mask)
            x = result_highfreq
            self.DCSEB2 = getattr(self, 'DCSEB2_{}'.format(str(i)))
            result_highfreq = self.DCSEB2(result_highfreq)
            result_highfreq = result_highfreq + pyr_original[-2 - i]
            setattr(self, 'result_highfreq_{}'.format(str(i)), result_highfreq)
        for i in reversed(range(self.num_high)):
            result_highfreq = getattr(self, 'result_highfreq_{}'.format(str(i)))
            pyr_result.append(result_highfreq)
        pyr_result.append(fake_low)

        return pyr_result


class LPTN(nn.Module):
    def __init__(self, num_high=3):
        super(LPTN, self).__init__()
        self.num_high = num_high
        self.lap_pyramid = Lap_Pyramid_Conv(num_high)
        trans_low = Trans_low()
        trans_high = Trans_high(num_high=num_high)
        self.trans_low = trans_low.cuda(0)
        self.trans_high = trans_high.cuda(0)

    def forward(self, real_A_full):
        pyr_A = self.lap_pyramid.pyramid_decom(img=real_A_full)
        fake_B_low = self.trans_low(real_A_full, pyr_A[-1])
        real_A_up = nn.functional.interpolate(pyr_A[-1], size=(pyr_A[-2].shape[2], pyr_A[-2].shape[3]))
        fake_B_up = nn.functional.interpolate(fake_B_low, size=(pyr_A[-2].shape[2], pyr_A[-2].shape[3]))
        high_with_low = torch.cat([real_A_up, fake_B_up], 1)
        pyr_A_trans = self.trans_high(high_with_low, pyr_A, fake_B_low)
        fake_B_full = self.lap_pyramid.pyramid_recons(pyr_A_trans)

        restored_images = []
        for i in range(self.num_high - 1):
            restored_image = self.lap_pyramid.pyramid_recons(pyr_A_trans[-(i + 2):])
            restored_images.append(restored_image)
        return fake_B_low, fake_B_full, restored_images
