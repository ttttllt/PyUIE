import argparse
import os
import numpy as np
import torch
from PIL import Image
from ignite.metrics import SSIM, PSNR
from path import Path
from torch.utils.data import Dataset

from dataset import RawAndReferenceWithNameDataset
from utils import ImageUtils
from PyramidNet import LPTN

def run_model(model, dataloader, device):
    model.eval()
    model.to(device)
    with torch.no_grad():
        for data in dataloader:
            raw_img, challenging_img, name = data
            raw_img = raw_img.to(device)
            challenging_img = challenging_img.to(device)
            

            fake_B_low, fake_B_full, restored_images = model(raw_img)
            
            yield raw_img, fake_B_full, name
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--initmodel", type=str, help="initmodel path")
    parser.add_argument("--model", type=str, help="model path")
    parser.add_argument("-x", type=str, help="input path")
    parser.add_argument("-o", "--output", type=str, help="output path")

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    dataset = RawAndReferenceWithNameDataset(args.x, args.x, 384)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)


    model = LPTN()


    model.load_state_dict(torch.load(args.model, map_location=device), strict=False)

    print("Started, please wait...")


    ssim = SSIM(data_range=1.0)
    psnr = PSNR(data_range=1.0)

    ssim_with_img_names = []
    psnr_with_img_names = []


    for xs, y_preds, names in run_model(model, dataloader, device):
        y_preds = y_preds.cpu()

        for x, y_pred, name in zip(xs, y_preds, names):
            ImageUtils.save_image(y_pred, os.path.join(args.output, name))

    print("Processing completed.")
