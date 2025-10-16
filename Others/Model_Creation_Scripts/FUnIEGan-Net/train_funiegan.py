"""
 > Training pipeline for FUnIE-GAN (paired) model
   * Paper: arxiv.org/pdf/1903.09766.pdf
 > Maintainer: Ashiwin Rajendran
"""

# py libs
import os
import sys
import yaml
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import csv

matplotlib.use("Agg")
import pickle

# pytorch libs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler

# Import the cosine annealing scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR

scaler = GradScaler()

# local libs
from nets.commons import Weights_Normal, VGG19_PercepLoss
from nets.funiegan import GeneratorFunieGAN, DiscriminatorFunieGAN
from utils.data_utils import GetTrainingPairs, GetValImage

## get configs and training options
parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", type=str, default="configs/train_euvp.yaml")
parser.add_argument("--epoch", type=int, default=45, help="which epoch to start from")
parser.add_argument("--num_epochs", type=int, default=201, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of 1st order momentum")
parser.add_argument("--b2", type=float, default=0.99, help="adam: decay of 2nd order momentum")
args = parser.parse_args()

## training params
epoch = args.epoch
num_epochs = args.num_epochs
batch_size = args.batch_size
lr_rate, lr_b1, lr_b2 = args.lr, args.b1, args.b2
# load the data config file
with open(args.cfg_file) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
# get info from config file
dataset_name = cfg["dataset_name"]
dataset_path = cfg["dataset_path"]
channels = cfg["chans"]
img_width = cfg["im_width"]
img_height = cfg["im_height"]
val_interval = cfg["val_interval"]
ckpt_interval = cfg["ckpt_interval"]

## create dir for model, validation data, and loss graphs
samples_dir = os.path.join("samples/FunieGAN/", dataset_name)
checkpoint_dir = os.path.join("checkpoints/FunieGAN/", dataset_name)
loss_graph_path = os.path.join("checkpoints/FunieGAN/", dataset_name, "loss_graph.png")
loss_data_path = os.path.join("checkpoints/FunieGAN/", dataset_name, "loss_data.pkl")

# Create CSV file and write headers
csv_file_path = os.path.join("checkpoints/FunieGAN/", dataset_name, "training_log.csv")
if not os.path.exists(csv_file_path):
    with open(csv_file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Batch", "Length", "D_loss", "G_loss", "Adv_loss"])

os.makedirs(samples_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# Initialize lists to store losses
if epoch == 0 and os.path.exists(loss_data_path):
    os.remove(loss_data_path)

if os.path.exists(loss_data_path):
    with open(loss_data_path, "rb") as f:
        loss_history = pickle.load(f)
else:
    loss_history = {"G_loss": [], "D_loss": [], "GAN_loss": []}


def save_loss_graph():
    plt.figure()
    plt.plot(loss_history["G_loss"], label="G_loss")
    plt.plot(loss_history["D_loss"], label="D_loss")
    plt.plot(loss_history["GAN_loss"], label="GAN_loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(loss_graph_path)
    plt.close()


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data)


""" FunieGAN specifics: loss functions and patch-size
-----------------------------------------------------"""

Adv_cGAN = torch.nn.MSELoss()
L1_G = torch.nn.L1Loss()  # similarity loss (l1)
L_vgg = VGG19_PercepLoss()  # content loss (vgg)
lambda_1, lambda_con = 5, 2  # 7:3 (as in paper)
patch = (1, img_height // 16, img_width // 16)  # 16x16 for 256x256

# Initialize generator and discriminator
generator = GeneratorFunieGAN()
discriminator = DiscriminatorFunieGAN()

# see if cuda is available
if torch.cuda.is_available():
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    Adv_cGAN.cuda()
    L1_G = L1_G.cuda()
    L_vgg = L_vgg.cuda()
    Tensor = torch.cuda.FloatTensor
else:
    Tensor = torch.FloatTensor

# Initialize weights or load pretrained models
if args.epoch == 0:
    generator.apply(init_weights)
    discriminator.apply(init_weights)
else:
    generator.load_state_dict(torch.load("checkpoints/FunieGAN/%s/generator_%d.pth" % (dataset_name, args.epoch)))
    discriminator.load_state_dict(torch.load("checkpoints/FunieGAN/%s/discriminator_%d.pth" % (dataset_name, epoch)))
    print("Loaded model from epoch %d" % (epoch))

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_rate, betas=(lr_b1, lr_b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_rate, betas=(lr_b1, lr_b2))

# Define the scheduler with a reduced T_max for more aggressive decay
scheduler_G = CosineAnnealingLR(optimizer_G, T_max=10, eta_min=1e-6)
scheduler_D = CosineAnnealingLR(optimizer_D, T_max=10, eta_min=1e-6)

## Data pipeline
transforms_ = [
    transforms.Resize((img_height, img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

accumulation_steps = 4  # Adjust as needed

dataloader = DataLoader(
    GetTrainingPairs(dataset_path, dataset_name, transforms_=transforms_),
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,  # default num_workers 8
)

val_dataloader = DataLoader(
    GetValImage(dataset_path, dataset_name, transforms_=transforms_, sub_dir="validation"),
    batch_size=batch_size,
    shuffle=True,
    num_workers=1,  # default num_workers 1
)

## Training pipeline
for epoch in range(epoch, num_epochs):
    optimizer_G.zero_grad()  # Ensure gradients are reset at the start of each epoch
    for i, batch in enumerate(dataloader):
        # Model inputs
        imgs_distorted = Variable(batch["A"].type(Tensor))
        # Add Gaussian noise to the input images
        noise = torch.randn_like(imgs_distorted) * 0.05  # Scale the noise as needed
        imgs_distorted = imgs_distorted + noise

        imgs_good_gt = Variable(batch["B"].type(Tensor))
        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((imgs_distorted.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((imgs_distorted.size(0), *patch))), requires_grad=False)

        ## Train Discriminator
        with autocast():  # Mixed precision enabled for discriminator
            imgs_fake = generator(imgs_distorted)
            pred_real = discriminator(imgs_good_gt, imgs_distorted)
            loss_real = Adv_cGAN(pred_real, valid)
            pred_fake = discriminator(imgs_fake, imgs_distorted)
            loss_fake = Adv_cGAN(pred_fake, fake)
            loss_D = 0.5 * (loss_real + loss_fake) * 10.0  # 10x scaled for stability

        # Scale gradients and backward pass
        scaler.scale(loss_D).backward()

        if (i + 1) % accumulation_steps == 0:  # Accumulate gradients and step
            scaler.step(optimizer_D)
            scaler.update()
            optimizer_D.zero_grad()

        ## Train Generator
        with autocast():  # Mixed precision enabled for generator
            imgs_fake = generator(imgs_distorted)
            pred_fake = discriminator(imgs_fake, imgs_distorted)
            loss_GAN = Adv_cGAN(pred_fake, valid)  # GAN loss
            loss_1 = L1_G(imgs_fake, imgs_good_gt)  # similarity loss
            loss_con = L_vgg(imgs_fake, imgs_good_gt)  # content loss
            lambda_gan = 15  # 10 New scaling factor for GAN loss
            loss_G = lambda_gan * loss_GAN + lambda_1 * loss_1 + lambda_con * loss_con

        # Scale gradients and backward pass
        scaler.scale(loss_G).backward()

        if (i + 1) % accumulation_steps == 0:  # Accumulate gradients and step
            scaler.step(optimizer_G)
            scaler.update()
            optimizer_G.zero_grad()

        ## Store loss values
        loss_history["G_loss"].append(loss_G.item())
        loss_history["D_loss"].append(loss_D.item())
        loss_history["GAN_loss"].append(loss_GAN.item())

        ## Print log
        if not i % 50:
            print(
                "[Epoch %d/%d: batch %d/%d] [DLoss: %.3f, GLoss: %.3f, AdvLoss: %.3f]"
                % (
                    epoch,
                    num_epochs,
                    i,
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_GAN.item(),
                )
            )

            ## Write loss values to CSV
            with open(csv_file_path, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([epoch, i, len(dataloader), loss_D.item(), loss_G.item(), loss_GAN.item()])

        ## If at sample interval save image
        batches_done = epoch * len(dataloader) + i
        if batches_done % val_interval == 0:
            imgs = next(iter(val_dataloader))
            imgs_val = Variable(imgs["val"].type(Tensor))
            imgs_gen = generator(imgs_val)
            img_sample = torch.cat((imgs_val.data, imgs_gen.data), -2)
            save_image(img_sample, "samples/FunieGAN/%s/%s.png" % (dataset_name, batches_done), nrow=5, normalize=True)

    ## Save model checkpoints
    if epoch % ckpt_interval == 0:
        torch.save(generator.state_dict(), "checkpoints/FunieGAN/%s/generator_%d.pth" % (dataset_name, epoch))
        torch.save(discriminator.state_dict(), "checkpoints/FunieGAN/%s/discriminator_%d.pth" % (dataset_name, epoch))
        with open(loss_data_path, "wb") as f:
            pickle.dump(loss_history, f)
        save_loss_graph()

    # Update learning rate
    scheduler_G.step()
    scheduler_D.step()
