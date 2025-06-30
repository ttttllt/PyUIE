import os
import torch
import torchvision
from PIL import Image
from path import Path
from torch.utils import data
from torchvision.transforms import InterpolationMode
from utils import DataSetUtils
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

class RandomCropPair:
    def __init__(self, size):
        self.size = size
        self.resize = transforms.Resize((512, 512))

    def __call__(self, img1, img2):
        # 先将 img1 和 img2 resize 到 384x384
        img1_resized = self.resize(img1)
        img2_resized = self.resize(img2)

        # 获取随机裁剪的参数
        i, j, h, w = transforms.RandomCrop.get_params(img1_resized, output_size=self.size)

        # 对两张图片使用相同的裁剪参数
        img1_cropped = F.crop(img1_resized, i, j, h, w)
        img2_cropped = F.crop(img2_resized, i, j, h, w)

        return img1_cropped, img2_cropped

class RandomCropPair1:
    def __init__(self, size):
        self.size = size
        self.resize = transforms.Resize((512, 512))
        self.center_crop = transforms.CenterCrop(self.size)

    def __call__(self, img1, img2):
        img1_resized = self.resize(img1)
        img2_resized = self.resize(img2)

        img1_cropped = self.center_crop(img1_resized)
        img2_cropped = self.center_crop(img2_resized)

        return img1_cropped, img2_cropped


class RawAndReferenceDataset(data.Dataset):
    def __init__(self, raw_file_path, reference_file_path, size=None):
        self.size = size
        if self.size is not None:
            self.random_crop_pair = RandomCropPair(size=(self.size, self.size))

        self.raw_file_path = raw_file_path
        self.reference_file_path = reference_file_path

        self.transforms = torchvision.transforms.ToTensor()
        self.raw_files = list(Path(self.raw_file_path).glob('*'))
        self.reference_files = list(Path(self.reference_file_path).glob('*'))

        self.file_names = {rf.name for rf in self.raw_files} & {cf.name for cf in self.reference_files}

        self.files_pairs = [(Path(self.raw_file_path) / n, Path(self.reference_file_path) / n) for n in self.file_names]

    def color_compensation(self, img):
        R = img[:, 2, :, :]
        G = img[:, 1, :, :]
        B = img[:, 0, :, :]

        Irm = torch.mean(R, dim=[1, 2], keepdim=True) / 256.0
        Igm = torch.mean(G, dim=[1, 2], keepdim=True) / 256.0
        Ibm = torch.mean(B, dim=[1, 2], keepdim=True) / 256.0

        a = 1
        Irc = R + a * (Igm - Irm) * (1 - Irm) * G
        Ibc = B + a * (Igm - Ibm) * (1 - Ibm) * G

        Irc = Irc.unsqueeze(1)
        Ibc = Ibc.unsqueeze(1)
        G = G.unsqueeze(1)

        img = torch.cat([Ibc, G, Irc], dim=1)
        img = torch.clamp(img, 0, 1)
        return img

    def gray_world(self, img):
        batch_size, channels, height, width = img.shape
        avg = torch.mean(img, dim=[2, 3], keepdim=True)

        out = torch.zeros_like(img)
        for j in range(channels):
            m = torch.sum(img[:, j, :, :], dim=[1, 2], keepdim=True)
            n = height * width
            scale = n / m
            g_weight = avg[:, j, :, :] * scale
            g_weight = g_weight.expand(batch_size, height, width)
            out[:, j, :, :] = img[:, j, :, :] * g_weight

        out = torch.clamp(out, 0, 1)
        return out

    def __getitem__(self, index):
        raw_img_path, challenging_img_path = self.files_pairs[index]

        raw_img = Image.open(raw_img_path).convert("RGB")
        challenging_img = Image.open(challenging_img_path).convert("RGB")

        raw_img = self.transforms(raw_img)

        challenging_img = self.transforms(challenging_img)

        raw_img = self.color_compensation(raw_img.unsqueeze(0)).squeeze(0)
        raw_img = self.gray_world(raw_img.unsqueeze(0)).squeeze(0)

        if self.size is not None:
            raw_img_cropped, challenging_img_cropped = self.random_crop_pair(raw_img, challenging_img)

            return raw_img_cropped, challenging_img_cropped

        return raw_img, challenging_img

    def __len__(self):
        return len(self.files_pairs)

class RawAndReferenceWithNameDataset(data.Dataset):
    def __init__(self, raw_file_path, reference_file_path, size=None):
        self.size = size
        if self.size is not None:
            self.random_crop_pair = RandomCropPair1(size=(self.size, self.size))

        self.raw_file_path = raw_file_path
        self.reference_file_path = reference_file_path

        self.transforms = torchvision.transforms.ToTensor()
        self.raw_files = list(Path(self.raw_file_path).glob('*'))
        self.reference_files = list(Path(self.reference_file_path).glob('*'))

        self.file_names = {rf.name for rf in self.raw_files} & {cf.name for cf in self.reference_files}

        self.files_pairs = [(Path(self.raw_file_path) / n, Path(self.reference_file_path) / n) for n in self.file_names]

    def color_compensation(self, img):
        R = img[:, 2, :, :]
        G = img[:, 1, :, :]
        B = img[:, 0, :, :]

        Irm = torch.mean(R, dim=[1, 2], keepdim=True) / 256.0
        Igm = torch.mean(G, dim=[1, 2], keepdim=True) / 256.0
        Ibm = torch.mean(B, dim=[1, 2], keepdim=True) / 256.0

        a = 1
        Irc = R + a * (Igm - Irm) * (1 - Irm) * G
        Ibc = B + a * (Igm - Ibm) * (1 - Ibm) * G

        Irc = Irc.unsqueeze(1)
        Ibc = Ibc.unsqueeze(1)
        G = G.unsqueeze(1)

        img = torch.cat([Ibc, G, Irc], dim=1)
        img = torch.clamp(img, 0, 1)
        return img

    def gray_world(self, img):
        batch_size, channels, height, width = img.shape
        avg = torch.mean(img, dim=[2, 3], keepdim=True)

        out = torch.zeros_like(img)
        for j in range(channels):
            m = torch.sum(img[:, j, :, :], dim=[1, 2], keepdim=True)
            n = height * width
            scale = n / m
            g_weight = avg[:, j, :, :] * scale
            g_weight = g_weight.expand(batch_size, height, width)
            out[:, j, :, :] = img[:, j, :, :] * g_weight

        out = torch.clamp(out, 0, 1)
        return out
    def __getitem__(self, index):
        raw_img_path, challenging_img_path = self.files_pairs[index]

        raw_img = Image.open(raw_img_path).convert("RGB")
        challenging_img = Image.open(challenging_img_path).convert("RGB")

        raw_img = self.transforms(raw_img)

        challenging_img = self.transforms(challenging_img)
        raw_img = self.color_compensation(raw_img.unsqueeze(0)).squeeze(0)
        raw_img = self.gray_world(raw_img.unsqueeze(0)).squeeze(0)
        
        if self.size is not None:
            raw_img_cropped, challenging_img_cropped = self.random_crop_pair(raw_img, challenging_img)

            return raw_img_cropped, challenging_img_cropped, os.path.basename(raw_img_path)

        return raw_img, challenging_img, os.path.basename(raw_img_path)

    def __len__(self):
        return len(self.files_pairs)

class FolderWithNameDataset(data.Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path

        self.transforms = torchvision.transforms.ToTensor()

        self.files = list(Path(self.folder_path).glob('*'))

    def __getitem__(self, index):
        img_path = self.files[index]
        img = Image.open(img_path).convert("RGB")
        img = self.transforms(img)
        return img, img_path

    def __len__(self):
        return len(self.files)


class ConditionalResize(object):
    def __init__(self, size):
        self.size = size
        self.resize = torchvision.transforms.Resize(self.size, InterpolationMode.BICUBIC)

    def __call__(self, img):
        if min(img.size) < self.size:
            return self.resize(img)
        else:
            return img
