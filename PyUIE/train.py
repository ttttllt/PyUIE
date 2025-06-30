from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import random
from typing import Sequence
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ignite.contrib.handlers import TensorboardLogger, global_step_from_engine
from ignite.contrib.handlers.neptune_logger import *
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Checkpoint
from ignite.metrics import Loss, PSNR, SSIM
from neptune.types import File
from path import Path
from reactivex import subject
from torch.utils.data import DataLoader
from torchsummary import summary
from ignite.metrics import Metric
import pytorch_ssim
from config import settings
from dataset import RawAndReferenceDataset, split_dataset
from loss import L1CharbonnierLoss, PerceptualLoss, ColorLosstwo, DynamicLossWeighting, UnContrastLoss

from PyramidNet import LPTN


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


set_seed(38)


def weight_init(m):
    if isinstance(m, nn.Linear):
        if m.weight is not None:
            nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        if m.weight is not None:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        if m.weight is not None:
            nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


if __name__ == "__main__":

    neptune_logger = NeptuneLogger(
        project=settings.neptune_logger.project,
        api_token=settings.neptune_logger.api_token,
        mode=settings.neptune_logger.mode,
        tags=settings.neptune_logger.tags,
    )

    torchvision_models_path = settings.torchvision_models_path
    if not os.path.exists(torchvision_models_path):
        Path(torchvision_models_path).makedirs_p()
    torch.hub.set_dir(torchvision_models_path)

    
    class MyLoss(nn.Module):
        def __init__(self):
            super(MyLoss, self).__init__()
            self.perceptual = PerceptualLoss()
            self.colortwo = ColorLosstwo()
            self.UnContrastLoss = UnContrastLoss()
            self.dynamic_weighting_full = DynamicLossWeighting(num_losses=4)
            self.dynamic_weighting_low = DynamicLossWeighting(num_losses=2)

        def forward(self, x, fake_B_low, fake_B_full, restored_images, Jgt_low, Jgt_full):

            l1_loss_low = F.l1_loss(fake_B_low, Jgt_low)
            total_loss_low = l1_loss_low

            l1_loss_full = F.l1_loss(fake_B_full, Jgt_full)
            perceptual_loss_full = self.perceptual(fake_B_full, Jgt_full)
            colortwo_loss_full = self.colortwo(fake_B_full, Jgt_full)
            UnContrastLoss_full = self.UnContrastLoss(x, fake_B_full, Jgt_full)


            losses_full = [l1_loss_full, perceptual_loss_full, colortwo_loss_full, UnContrastLoss_full]
            total_loss_full = self.dynamic_weighting_full(losses_full)

            l1_loss_restored = 0
            perceptual_loss_restored = 0
            colortwo_loss_restored = 0
            UnContrastLoss_restored = 0
            for restored_img in restored_images:
                Jgt_restored = F.interpolate(Jgt_full, size=(restored_img.shape[2], restored_img.shape[3]),
                                             mode='bicubic', align_corners=True)
                x_restored = F.interpolate(x, size=(restored_img.shape[2], restored_img.shape[3]),
                                             mode='bicubic', align_corners=True)
                l1_loss_restored += F.l1_loss(restored_img, Jgt_restored)
                perceptual_loss_restored += self.perceptual(restored_img, Jgt_restored)
                colortwo_loss_restored += self.colortwo(restored_img, Jgt_restored)
                UnContrastLoss_restored = self.UnContrastLoss(x_restored, restored_img, Jgt_restored)


            losses_restored = [l1_loss_restored, perceptual_loss_restored,colortwo_loss_restored,UnContrastLoss_restored]
            total_loss_restored = self.dynamic_weighting_full(losses_restored)


            total_loss = total_loss_low + total_loss_full + 0.5 * total_loss_restored
            return total_loss



    crop_size = settings.dataset.train.crop_size
    crop_size_val = settings.dataset.valid.crop_size
    train_raw_path = settings.dataset.raw.train
    train_reference_path = settings.dataset.reference.train
    val_raw_path = settings.dataset.raw.val
    val_reference_path = settings.dataset.reference.val


    train_dataset = RawAndReferenceDataset(train_raw_path, train_reference_path, crop_size)
    val_dataset = RawAndReferenceDataset(val_raw_path, val_reference_path, crop_size_val)


    ts = settings.dataset.train
    train_loader = DataLoader(train_dataset,
                              batch_size=ts.batch_size, shuffle=ts.shuffle,
                              num_workers=ts.num_workers)

    vs = settings.dataset.valid
    val_loader = DataLoader(val_dataset,
                            batch_size=vs.batch_size, shuffle=vs.shuffle,
                            num_workers=vs.num_workers)


    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    model = LPTN().to(device)

    unet_path = settings.model.unet
    if unet_path is not None:
        if os.path.isfile(unet_path):
            print("Loading model: ", unet_path)
            model.load_state_dict(torch.load(unet_path))
        else:
            raise FileNotFoundError("Unet model not found")

    model.apply(weight_init)
    loss_model = MyLoss().to(device)

    lr = settings.train.lr
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    step_lr = settings.train.step_lr

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=step_lr.T_max, eta_min=step_lr.eta_min)


    def my_loss_wrapper(x, y):
        fake_B_low, fake_B_full, restored_images = model(x)
        Jgt_low = F.interpolate(y, size=(fake_B_low.shape[2], fake_B_low.shape[3]), mode='bicubic', align_corners=True)
        return loss_model(x, fake_B_low, fake_B_full, restored_images, Jgt_low, y)


    val_metrics = {
        "loss": Loss(my_loss_wrapper),
        "psnr": PSNR(data_range=1.0),
        "ssim": SSIM(data_range=1.0),

        
    }

    training_info_subject = subject.Subject()


    def get_current_lr(optimizer):
        return optimizer.param_groups[0]['lr']



    def train_step(engine, batch):
        model.train()
        optimizer.zero_grad()

        x, y = batch[0].to(device), batch[1].to(device)
        fake_B_low, fake_B_full, restored_images = model(x)
        Jgt_low = F.interpolate(y, size=(fake_B_low.shape[2], fake_B_low.shape[3]), mode='bicubic', align_corners=True)

        loss = loss_model(x, fake_B_low, fake_B_full, restored_images, Jgt_low, y)
        loss.backward()


        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        return loss.item()


    trainer = Engine(train_step)


    def validation_step(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = batch[0].to(device), batch[1].to(device)
            fake_B_low, fake_B_full, restored_images = model(x)

            engine.state.metrics["x"] = x
            engine.state.metrics["y"] = y
            engine.state.metrics["y_pred"] = fake_B_full
            return fake_B_full, y


    train_evaluator = Engine(validation_step)
    val_evaluator = Engine(validation_step)


    for name, metric in val_metrics.items():
        metric.attach(train_evaluator, name)

    for name, metric in val_metrics.items():
        metric.attach(val_evaluator, name)

    log_interval = settings.log.iteration_interval


    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def log_training_loss(engine):
        print(f"Epoch[{engine.state.epoch}], Iter[{engine.state.iteration}] Loss: {engine.state.output:.4f}")


    @trainer.on(Events.EPOCH_COMPLETED)
    def update_lr_scheduler(engine):
        scheduler.step()
        current_lr = get_current_lr(optimizer)
        neptune_logger.experiment['learning_rate'].append(current_lr)


    def get_matrics_message(metrics, names: Sequence[str]):
        message = ""
        for name in names:
            message += f"{name}: {metrics[name]:.4f}, "
        return message


    val_epoch_interval = settings.valid.epoch_interval


    @trainer.on(Events.EPOCH_COMPLETED(every=val_epoch_interval))
    def log_training_results(trainer):
        train_evaluator.run(train_loader)
        metrics = train_evaluator.state.metrics
        print(
            f"Training Results - Epoch[{trainer.state.epoch}] {get_matrics_message(metrics, [name for name, _ in val_metrics.items()])}")


    @trainer.on(Events.EPOCH_COMPLETED(every=val_epoch_interval))
    def log_validation_results(trainer):
        val_evaluator.run(val_loader)
        metrics = val_evaluator.state.metrics
        print(
            f"Validation Results - Epoch[{trainer.state.epoch}] {get_matrics_message(metrics, [name for name, _ in val_metrics.items()])}\n\n")


    @val_evaluator.on(Events.COMPLETED)
    def log_generated_images(engine):
        x_tensor = engine.state.metrics["x"].cpu()
        y_tensor = engine.state.metrics["y"].cpu()
        y_pred_tensor = engine.state.metrics["y_pred"].cpu()

        x_img = torch.moveaxis(x_tensor[0], 0, -1)
        y_img = torch.moveaxis(y_tensor[0], 0, -1)
        y_pred_img = torch.moveaxis(y_pred_tensor[0], 0, -1)
        y_pred_img = torch.clamp(y_pred_img, 0, 1)

        neptune_logger.experiment["I"].append(File.as_image(x_img))
        neptune_logger.experiment["Jgt"].append(File.as_image(y_img))
        neptune_logger.experiment["J"].append(File.as_image(y_pred_img))


    def score_function_for(score_name):
        return lambda engine: engine.state.metrics[score_name]


    cps = settings.checkpoint


    model_checkpoint = ModelCheckpoint(
        cps.dirname,
        n_saved=cps.n_saved,
        filename_prefix=cps.filename_prefix,
        score_function=score_function_for(cps.score_name),
        score_name=cps.score_name,
        global_step_transform=global_step_from_engine(trainer),
        require_empty=cps.require_empty,
    )


    val_evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {"model": model})

    if cps.upload:
        to_save = {"model": model}
        handler = Checkpoint(
            to_save,
            NeptuneSaver(neptune_logger),
            n_saved=cps.n_saved,
            filename_prefix=cps.filename_prefix,
            score_function=score_function_for(cps.score_name),
            score_name=cps.score_name,
            global_step_transform=global_step_from_engine(trainer),
        )
        val_evaluator.add_event_handler(Events.COMPLETED, handler)


    tb_logger = TensorboardLogger(log_dir=settings.tb_logger.log_dir)


    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED,
        tag="training",
        output_transform=lambda loss: {"batch_loss": loss},
    )

    neptune_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED,
        tag="training",
        output_transform=lambda loss: {"batch_loss": loss},
    )


    for tag, evaluator in [("training", train_evaluator), ("validation", val_evaluator)]:
        tb_logger.attach_output_handler(
            evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag=tag,
            metric_names=[name for name, metric in val_metrics.items()],
            global_step_transform=global_step_from_engine(trainer),
        )
        neptune_logger.attach_output_handler(
            evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag=tag,
            metric_names=[name for name, metric in val_metrics.items()],
            global_step_transform=global_step_from_engine(trainer),
        )

    summary(model, (3, crop_size, crop_size))
    trainer.run(train_loader, max_epochs=settings.train.max_epochs)

    neptune_logger.close()
