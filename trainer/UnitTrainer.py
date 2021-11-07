#!/usr/bin/python3

import argparse
import itertools
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from .utils import LambdaLR, Logger, ReplayBuffer
from .utils import weights_init_normal, get_config
from .datasets import ImageDataset, ValDataset
from Model.Unit import *
from .utils import Resize, ToTensor, smooothing_loss
from .utils import Logger
from .reg import Reg
from torchvision.transforms import RandomAffine
from torchvision.transforms import RandomAffine, ToPILImage
from .transformer import Transformer_2D
from skimage import measure
import numpy as np
import cv2
import os


class Unit_Trainer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        ## def networks
        self.netG_A2B = VAEGen(
            config["input_nc"], config["gen"]
        ).cuda()  # auto-encoder for domain a
        self.netG_B2A = VAEGen(
            config["input_dim_b"], config["gen"]
        ).cuda()  # auto-encoder for domain b
        self.netD_B = MsImageDis(
            config["input_dim_b"], config["dis"]
        ).cuda()  # discriminator for domain b
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.optimizer_G = torch.optim.Adam(
            itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()),
            lr=config["lr"],
            betas=(0.5, 0.999),
        )
        if config["regist"]:
            self.R_A = Reg(config['size'], config['size'],config["input_nc"],config["input_nc"]).cuda()
            self.spatial_transform = Transformer_2D().cuda()
            self.optimizer_R_A = torch.optim.Adam(
                self.R_A.parameters(), lr=config["lr"], betas=(0.5, 0.999)
            )

        if config["bidirect"]:

            self.netD_A = MsImageDis(
                config["input_dim_a"], config["dis"]
            ).cuda()  # discriminator for domain a

            self.optimizer_D = torch.optim.Adam(
                itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                lr=config["lr"],
                betas=(0.5, 0.999),
            )
        else:
            self.optimizer_D = torch.optim.Adam(
                self.netD_B.parameters(), lr=config["lr"], betas=(0.5, 0.999)
            )

        # Lossess
        self.MSE_loss = torch.nn.MSELoss()
        self.L1_loss = torch.nn.L1Loss()

        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if config["cuda"] else torch.Tensor
        self.input_A = Tensor(
            config["batchSize"], config["input_nc"], config["size"], config["size"]
        )
        self.input_B = Tensor(
            config["batchSize"], config["output_nc"], config["size"], config["size"]
        )
        self.target_real = Variable(Tensor(1, 1).fill_(1.0), requires_grad=False)
        self.target_fake = Variable(Tensor(1, 1).fill_(0.0), requires_grad=False)

        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

        # Dataset loader
        level = config['noise_level']  # set noise level
        transforms_1 = [
            ToPILImage(),
            RandomAffine(
                degrees=level,translate=[0.02*level, 0.02*level],scale=[1-0.02*level, 1+0.02*level],fillcolor=-1
            ),
            ToTensor(),
            Resize(size_tuple=(config["size"], config["size"])),
        ]

        transforms_2 = [
            ToPILImage(),
            RandomAffine(
                degrees=level, translate=[0.02 * level, 0.02 * level], scale=[1 - 0.02 * level, 1 + 0.02 * level],
                fillcolor=-1
            ),
            ToTensor(),
            Resize(size_tuple=(config["size"], config["size"])),
        ]

        self.dataloader = DataLoader(
            ImageDataset(
                config["dataroot"],
                transforms_1=transforms_1,
                transforms_2=transforms_2,
                unaligned=False,
            ),
            batch_size=config["batchSize"],
            shuffle=True,
            num_workers=config["n_cpu"],
        )

        val_transforms = [
            ToTensor(),
            Resize(size_tuple=(config["size"], config["size"])),
        ]

        self.val_data = DataLoader(
            ValDataset(
                config["val_dataroot"], transforms_=val_transforms, unaligned=False
            ),
            batch_size=config["batchSize"],
            shuffle=False,
            num_workers=config["n_cpu"],
        )

        # Loss plot
        self.logger = Logger(config["name"], config['port'],config["n_epochs"], len(self.dataloader))

    def train(self):
        ###### Training ######
        for epoch in range(self.config["epoch"], self.config["n_epochs"]):
            for i, batch in enumerate(self.dataloader):
                # Set model input
                real_A = Variable(self.input_A.copy_(batch["A"]))
                real_B = Variable(self.input_B.copy_(batch["B"]))
                if self.config["bidirect"]:  # b dir
                    if self.config["regist"]:  # + reg
                        self.optimizer_R_A.zero_grad()
                        self.optimizer_G.zero_grad()
                        x_a = real_A
                        x_b = real_B
                        # GAN loss # encode
                        h_a, n_a = self.netG_A2B.encode(x_a)
                        h_b, n_b = self.netG_B2A.encode(x_b)
                        # decode (within domain)
                        x_a_recon = self.netG_A2B.decode(h_a + n_a)
                        x_b_recon = self.netG_B2A.decode(h_b + n_b)
                        # decode (cross domain)
                        x_ba = self.netG_A2B.decode(h_b + n_b)
                        x_ab = self.netG_B2A.decode(h_a + n_a)
                        # encode again
                        h_b_recon, n_b_recon = self.netG_A2B.encode(x_ba)
                        h_a_recon, n_a_recon = self.netG_B2A.encode(x_ab)
                        # decode again (if needed)
                        x_aba = self.netG_A2B.decode(h_a_recon + n_a_recon)
                        x_bab = self.netG_B2A.decode(h_b_recon + n_b_recon)
                        # reconstruction loss
                        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
                        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
                        self.loss_gen_recon_kl_a = self.__compute_kl(h_a)
                        self.loss_gen_recon_kl_b = self.__compute_kl(h_b)
                        self.loss_gen_cyc_x_a = self.recon_criterion(x_aba, x_a)
                        self.loss_gen_cyc_x_b = self.recon_criterion(x_bab, x_b)
                        self.loss_gen_recon_kl_cyc_aba = self.__compute_kl(h_a_recon)
                        self.loss_gen_recon_kl_cyc_bab = self.__compute_kl(h_b_recon)

                        # GAN loss
                        self.loss_gen_adv_a = self.netD_A.calc_gen_loss(x_ba)
                        self.loss_gen_adv_b = self.netD_B.calc_gen_loss(x_ab)
                        # Total loss
                        loss_G = (
                            self.loss_gen_adv_a * self.config['Adv_lamda']
                            + self.loss_gen_adv_b * self.config['Adv_lamda']
                            + self.loss_gen_recon_x_a * self.config['Cyc_lamda']
                            + self.loss_gen_recon_x_b * self.config['Cyc_lamda']
                            + self.loss_gen_recon_kl_a * self.config['Recon_kl_lamda']
                            + self.loss_gen_recon_kl_b * self.config['Recon_kl_lamda']
                            + self.loss_gen_cyc_x_a * self.config['Cyc_lamda']
                            + self.loss_gen_cyc_x_b * self.config['Cyc_lamda']
                            + self.loss_gen_recon_kl_cyc_aba * self.config['Recon_kl_lamda']
                            + self.loss_gen_recon_kl_cyc_bab * self.config['Recon_kl_lamda']
                        )

                        ################ Reg ##############

                        fake_B = x_ab
                        Trans = self.R_A(x_ab, real_B)
                        SysRegist_A2B = self.spatial_transform(x_ab, Trans)
                        SR_loss = self.config['Corr_lamda'] * self.L1_loss(SysRegist_A2B, real_B)  ###SR
                        SM_loss = self.config['Smooth_lamda'] * smooothing_loss(Trans)

                        # Total loss
                        loss_Total = loss_G + SR_loss + SM_loss
                        loss_Total.backward()
                        self.optimizer_G.step()
                        self.optimizer_R_A.step()

                        ###### Discriminator ######
                        self.optimizer_D.zero_grad()
                        h_a, n_a = self.netG_A2B.encode(x_a)
                        h_b, n_b = self.netG_B2A.encode(x_b)
                        # decode (cross domain)
                        x_ba = self.netG_A2B.decode(h_b + n_b)
                        x_ab = self.netG_B2A.decode(h_a + n_a)
                        # D loss
                        loss_D_A = self.netD_A.calc_dis_loss(x_ba.detach(), x_a)
                        loss_D_B = self.netD_B.calc_dis_loss(x_ab.detach(), x_b)
                        self.loss_dis_total = self.config['Adv_lamda'] * (loss_D_A + loss_D_B)
                        self.loss_dis_total.backward()
                        self.optimizer_D.step()
                        ###################################

                    else:
                        self.optimizer_G.zero_grad()
                        x_a = real_A
                        x_b = real_B
                        # GAN loss # encode
                        h_a, n_a = self.netG_A2B.encode(x_a)
                        h_b, n_b = self.netG_B2A.encode(x_b)
                        # decode (within domain)
                        x_a_recon = self.netG_A2B.decode(h_a + n_a)
                        x_b_recon = self.netG_B2A.decode(h_b + n_b)
                        # decode (cross domain)
                        x_ba = self.netG_A2B.decode(h_b + n_b)
                        x_ab = self.netG_B2A.decode(h_a + n_a)
                        # encode again
                        h_b_recon, n_b_recon = self.netG_A2B.encode(x_ba)
                        h_a_recon, n_a_recon = self.netG_B2A.encode(x_ab)
                        # decode again (if needed)
                        x_aba = self.netG_A2B.decode(h_a_recon + n_a_recon)
                        x_bab = self.netG_B2A.decode(h_b_recon + n_b_recon)
                        # reconstruction loss
                        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
                        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
                        self.loss_gen_recon_kl_a = self.__compute_kl(h_a)
                        self.loss_gen_recon_kl_b = self.__compute_kl(h_b)
                        self.loss_gen_cyc_x_a = self.recon_criterion(x_aba, x_a)
                        self.loss_gen_cyc_x_b = self.recon_criterion(x_bab, x_b)
                        self.loss_gen_recon_kl_cyc_aba = self.__compute_kl(h_a_recon)
                        self.loss_gen_recon_kl_cyc_bab = self.__compute_kl(h_b_recon)

                        # GAN loss
                        self.loss_gen_adv_a = self.netD_A.calc_gen_loss(x_ba)
                        self.loss_gen_adv_b = self.netD_B.calc_gen_loss(x_ab)
                        # Total loss
                        loss_G = (
                            self.loss_gen_adv_a * self.config['Adv_lamda']
                            + self.loss_gen_adv_b * self.config['Adv_lamda']
                            + self.loss_gen_recon_x_a * self.config['Cyc_lamda']
                            + self.loss_gen_recon_x_b * self.config['Cyc_lamda']
                            + self.loss_gen_recon_kl_a * self.config['Recon_kl_lamda']
                            + self.loss_gen_recon_kl_b * self.config['Recon_kl_lamda']
                            + self.loss_gen_cyc_x_a * self.config['Cyc_lamda']
                            + self.loss_gen_cyc_x_b * self.config['Cyc_lamda']
                            + self.loss_gen_recon_kl_cyc_aba * self.config['Recon_kl_lamda']
                            + self.loss_gen_recon_kl_cyc_bab * self.config['Recon_kl_lamda']
                        )

                        fake_B = x_ab
                        loss_G.backward()
                        self.optimizer_G.step()
                        ###### Discriminator ######
                        self.optimizer_D.zero_grad()
                        h_a, n_a = self.netG_A2B.encode(x_a)
                        h_b, n_b = self.netG_B2A.encode(x_b)
                        # decode (cross domain)
                        x_ba = self.netG_A2B.decode(h_b + n_b)
                        x_ab = self.netG_B2A.decode(h_a + n_a)
                        # D loss
                        loss_D_A = self.netD_A.calc_dis_loss(x_ba.detach(), x_a)
                        loss_D_B = self.netD_B.calc_dis_loss(x_ab.detach(), x_b)
                        self.loss_dis_total = self.config['Adv_lamda'] * (loss_D_A + loss_D_B)
                        self.loss_dis_total.backward()
                        self.optimizer_D.step()
                        ###################################

                else:  # s dir
                    if self.config["regist"]:  # + reg
                        self.optimizer_R_A.zero_grad()
                        self.optimizer_G.zero_grad()
                        x_a = real_A
                        x_b = real_B
                        # GAN loss # encode
                        h_a, n_a = self.netG_A2B.encode(x_a)
                        h_b, n_b = self.netG_B2A.encode(x_b)
                        # decode (within domain)
                        x_a_recon = self.netG_A2B.decode(h_a + n_a)
                        x_b_recon = self.netG_B2A.decode(h_b + n_b)
                        # decode (cross domain)
                        x_ba = self.netG_A2B.decode(h_b + n_b)
                        x_ab = self.netG_B2A.decode(h_a + n_a)
                        # encode again
                        h_b_recon, n_b_recon = self.netG_A2B.encode(x_ba)
                        h_a_recon, n_a_recon = self.netG_B2A.encode(x_ab)
                        # decode again (if needed)
                        x_aba = self.netG_A2B.decode(h_a_recon + n_a_recon)
                        x_bab = self.netG_B2A.decode(h_b_recon + n_b_recon)
                        # reconstruction loss
                        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
                        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
                        self.loss_gen_recon_kl_a = self.__compute_kl(h_a)
                        self.loss_gen_recon_kl_b = self.__compute_kl(h_b)
                        self.loss_gen_cyc_x_a = self.recon_criterion(x_aba, x_a)
                        self.loss_gen_recon_kl_cyc_aba = self.__compute_kl(h_a_recon)

                        # GAN loss
                        self.loss_gen_adv_b = self.netD_B.calc_gen_loss(x_ab)
                        # Total loss
                        loss_G = (
                            # self.loss_gen_adv_a
                            self.loss_gen_adv_b * self.config['Adv_lamda']
                            + self.loss_gen_recon_x_a * self.config['Cyc_lamda']
                            + self.loss_gen_recon_x_b * self.config['Cyc_lamda']
                            + self.loss_gen_recon_kl_a * self.config['Recon_kl_lamda']
                            + self.loss_gen_recon_kl_b * self.config['Recon_kl_lamda']
                            + self.loss_gen_cyc_x_a * self.config['Cyc_lamda']
                            + self.loss_gen_recon_kl_cyc_aba * self.config['Recon_kl_lamda']
                        )

                        ################ Reg ##############

                        fake_B = x_ab
                        Trans = self.R_A(x_ab, real_B)
                        SysRegist_A2B = self.spatial_transform(x_ab, Trans)
                        SR_loss = self.config['Corr_lamda'] * self.L1_loss(SysRegist_A2B, real_B)  ###SR
                        SM_loss = self.config['Smooth_lamda'] * smooothing_loss(Trans)

                        # Total loss
                        loss_Total = loss_G + SR_loss + SM_loss
                        loss_Total.backward()
                        self.optimizer_G.step()
                        self.optimizer_R_A.step()

                        ###### Discriminator ######
                        self.optimizer_D.zero_grad()
                        h_a, n_a = self.netG_A2B.encode(x_a)
                        # decode (cross domain)
                        x_ab = self.netG_B2A.decode(h_a + n_a)
                        # D loss
                        loss_D_B = self.netD_B.calc_dis_loss(x_ab.detach(), x_b)
                        self.loss_dis_total = self.config['Adv_lamda'] * loss_D_B
                        self.loss_dis_total.backward()
                        self.optimizer_D.step()
                        ###################################

                    else:
                        self.optimizer_G.zero_grad()
                        x_a = real_A
                        x_b = real_B
                        # GAN loss # encode
                        h_a, n_a = self.netG_A2B.encode(x_a)
                        h_b, n_b = self.netG_B2A.encode(x_b)
                        # decode (within domain)
                        x_a_recon = self.netG_A2B.decode(h_a + n_a)
                        x_b_recon = self.netG_B2A.decode(h_b + n_b)
                        # decode (cross domain)
                        x_ba = self.netG_A2B.decode(h_b + n_b)
                        x_ab = self.netG_B2A.decode(h_a + n_a)
                        # encode again
                        h_b_recon, n_b_recon = self.netG_A2B.encode(x_ba)
                        h_a_recon, n_a_recon = self.netG_B2A.encode(x_ab)
                        # decode again (if needed)
                        x_aba = self.netG_A2B.decode(h_a_recon + n_a_recon)
                        x_bab = self.netG_B2A.decode(h_b_recon + n_b_recon)
                        # reconstruction loss
                        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
                        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
                        self.loss_gen_recon_kl_a = self.__compute_kl(h_a)
                        self.loss_gen_recon_kl_b = self.__compute_kl(h_b)
                        self.loss_gen_cyc_x_a = self.recon_criterion(x_aba, x_a)
                        self.loss_gen_recon_kl_cyc_aba = self.__compute_kl(h_a_recon)

                        # GAN loss
                        # self.loss_gen_adv_a = self.netD_A.calc_gen_loss(x_ba)
                        self.loss_gen_adv_b = self.netD_B.calc_gen_loss(x_ab)
                        # Total loss
                        loss_G = (
                            # self.loss_gen_adv_a
                            self.loss_gen_adv_b * self.config['Adv_lamda']
                            + self.loss_gen_recon_x_a * self.config['Cyc_lamda']
                            + self.loss_gen_recon_x_b * self.config['Cyc_lamda']
                            + self.loss_gen_recon_kl_a * self.config['Recon_kl_lamda']
                            + self.loss_gen_recon_kl_b * self.config['Recon_kl_lamda']
                            + self.loss_gen_cyc_x_a * self.config['Cyc_lamda']
                            + self.loss_gen_recon_kl_cyc_aba * self.config['Recon_kl_lamda']
                        )

                        ################ Reg ##############

                        fake_B = x_ab
                        loss_G.backward()
                        self.optimizer_G.step()


                        ###### Discriminator ######
                        self.optimizer_D.zero_grad()
                        h_a, n_a = self.netG_A2B.encode(x_a)
                        h_b, n_b = self.netG_B2A.encode(x_b)
                        # decode (cross domain)
                        # x_ba = self.netG_A2B.decode(h_b + n_b)
                        x_ab = self.netG_B2A.decode(h_a + n_a)
                        # D loss
                        # loss_D_A = self.netD_A.calc_dis_loss(x_ba.detach(), x_a)
                        loss_D_B = self.netD_B.calc_dis_loss(x_ab.detach(), x_b)
                        self.loss_dis_total = self.config['Adv_lamda'] * loss_D_B
                        self.loss_dis_total.backward()
                        self.optimizer_D.step()
                        ###################################
                        ###################################

                self.logger.log(
                    {
                        "loss_G": loss_G,
                    },
                    images={
                        "real_A": real_A,
                        "real_B": real_B,
                        "fake_B": fake_B,
                        # "warped": SysRegist_A2B
                    },
                )  # ,'SR':SysRegist_A2B

            #         # Save models checkpoints
            if not os.path.exists(self.config["save_root"]):
                os.makedirs(self.config["save_root"])
            torch.save(
                self.netG_A2B.state_dict(), self.config["save_root"] + "netG_A2B.pth"
            )
            torch.save(
                self.netG_B2A.state_dict(), self.config["save_root"] + "netG_B2A.pth"
            )
            # torch.save(netG_B2A.state_dict(), 'output/netG_B2A_3D.pth')
            # torch.save(netD_A.state_dict(), 'output/netD_A_3D.pth')
            # torch.save(netD_B.state_dict(), 'output/netD_B_3D.pth')

            #############val###############
            with torch.no_grad():
                MAE = 0
                num = 0
                for i, batch in enumerate(self.val_data):
                    real_A = Variable(self.input_A.copy_(batch["A"]))
                    real_B = (
                        Variable(self.input_B.copy_(batch["B"]))
                        .detach()
                        .cpu()
                        .numpy()
                        .squeeze()
                    )
                    h_A, _ = self.netG_A2B.encode(real_A)
                    fake_B = self.netG_B2A.decode(h_A).detach().cpu().numpy().squeeze()
                    mae = self.MAE(fake_B, real_B)
                    MAE += mae
                    num += 1

                print("MAE:", MAE / num)

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def __compute_kl(self, mu):
        # def _compute_kl(self, mu, sd):
        # mu_2 = torch.pow(mu, 2)
        # sd_2 = torch.pow(sd, 2)
        # encoding_loss = (mu_2 + sd_2 - torch.log(sd_2)).sum() / mu_2.size(0)
        # return encoding_loss
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    def test(
        self,
    ):
        self.netG_A2B.load_state_dict(
            torch.load(self.config["save_root"] + "netG_A2B.pth")
        )
        self.netG_B2A.load_state_dict(
            torch.load(self.config["save_root"] + "netG_B2A.pth")
        )
        with torch.no_grad():
            MAE = 0
            PSNR = 0
            SSIM = 0
            num = 0

            for i, batch in enumerate(self.val_data):
                real_A = Variable(self.input_A.copy_(batch["A"]))
                real_B = (
                    Variable(self.input_B.copy_(batch["B"]))
                    .detach()
                    .cpu()
                    .numpy()
                    .squeeze()
                )
                h_A, _ = self.netG_A2B.encode(real_A)
                fake_B = self.netG_B2A.decode(h_A).detach().cpu().numpy().squeeze()

                mae = self.MAE(fake_B, real_B)

                psnr = self.PSNR(fake_B, real_B)
                ssim = measure.compare_ssim(fake_B, real_B)

                MAE += mae
                PSNR += psnr
                SSIM += ssim
                num += 1
                image_FB = 255 * ((fake_B + 1) / 2)
                cv2.imwrite(self.config["image_save"] + str(num) + ".png", image_FB)

            print("MAE:", MAE / num)
            print("PSNR:", PSNR / num)
            print("SSIM:", SSIM / num)

    def PSNR(self, fake, real):
        x, y = np.where(real != -1)
        mse = np.mean(((fake[x][y] + 1) / 2.0 - (real[x][y] + 1) / 2.0) ** 2)
        if mse < 1.0e-10:
            return 100
        else:
            PIXEL_MAX = 1
            return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

    def MAE(self, fake, real):
        x, y = np.where(real != -1)  # coordinate of target points
        # points = len(x)  #num of target points
        mae = np.abs(fake[x, y] - real[x, y]).mean()

        return mae / 2
