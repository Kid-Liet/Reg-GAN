#!/usr/bin/python3

import argparse
import itertools
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from .datasets import ImageDataset,ValDataset
from Model.NiceGan import *
from .utils import Resize,ToTensor,smooothing_loss
from .utils import Logger
from .reg import Reg
from torchvision.transforms import RandomAffine
from torchvision.transforms import RandomAffine,ToPILImage
from .transformer import Transformer_2D
from skimage import measure
import numpy as np
import cv2

class Nice_Trainer():
    def __init__(self, config):
        super().__init__()
        self.config = config
        ## def networks
        
        if config['regist']:
            self.R_A = Reg(config['size'], config['size'],config['input_nc'],config['input_nc']).cuda()
            self.spatial_transform = Transformer_2D().cuda()
            self.optimizer_R_A = torch.optim.Adam(self.R_A.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        if config['bidirect']:
            self.netG_A2B = ResnetGenerator(config['input_nc']*128, config['output_nc']).cuda()
            self.netG_B2A = ResnetGenerator(config['input_nc']*128, config['output_nc']).cuda()
            
            self.netD_A = Discriminator(config['input_nc']).cuda()
            self.netD_B = Discriminator(config['input_nc']).cuda()
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()),lr=config['lr'], betas=(0.5, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=config['lr'], betas=(0.5, 0.999))
            self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        else:
            self.netD_B = Discriminator2(config['input_nc']).cuda()
            self.netG_A2B = ResnetGenerator2(config['input_nc'], config['output_nc']).cuda()
            self.optimizer_G = torch.optim.Adam(self.netG_A2B.parameters(), lr=config['lr'], betas=(0.5, 0.999))
            self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=config['lr'], betas=(0.5, 0.999))
                
        # Lossess
        self.MSE_loss = torch.nn.MSELoss()
        self.L1_loss = torch.nn.L1Loss()
        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if config['cuda'] else torch.Tensor
        self.input_A = Tensor(config['batchSize'], config['input_nc'], config['size'], config['size'])
        self.input_B = Tensor(config['batchSize'], config['output_nc'], config['size'], config['size'])

        #Dataset loader
        level = config['noise_level']
        transforms_1 = [ToPILImage(),
                   RandomAffine(degrees=level,translate=[0.02*level, 0.02*level],scale=[1-0.02*level, 1+0.02*level],fillcolor=-1),
                   ToTensor(),
                   Resize(size_tuple = (config['size'], config['size']))]
    
        transforms_2 = [ToPILImage(),
                   RandomAffine(degrees=level,translate=[0.02*level, 0.02*level],scale=[1-0.02*level, 1+0.02*level],fillcolor=-1),
                   ToTensor(),
                   Resize(size_tuple = (config['size'], config['size']))]

        self.dataloader = DataLoader(ImageDataset(config['dataroot'], transforms_1=transforms_1, transforms_2=transforms_2, unaligned=False),
                                batch_size=config['batchSize'], shuffle=True, num_workers=config['n_cpu'])
        
        
        val_transforms = [ToTensor(),
                    Resize(size_tuple = (config['size'], config['size']))]
        
        self.val_data = DataLoader(ValDataset(config['val_dataroot'], transforms_ =val_transforms, unaligned=False),
                                batch_size=config['batchSize'], shuffle=False, num_workers=config['n_cpu'])

 
       # Loss plot
        self.logger = Logger(config['name'],config['port'],config['n_epochs'], len(self.dataloader))       
        
    def train(self):
        ###### Training ######
        for epoch in range(self.config['epoch'], self.config['n_epochs']):
            for i, batch in enumerate(self.dataloader):
                # Set model input
                real_A = Variable(self.input_A.copy_(batch['A']))
                real_B = Variable(self.input_B.copy_(batch['B']))
                if self.config['bidirect']:   # b dir
                    if self.config['regist']:    # + reg 
                        self.optimizer_D_A.zero_grad()
                        self.optimizer_D_B.zero_grad()

                        real_LA_logit,real_GA_logit, real_A_cam_logit, _, real_A_z = self.netD_A(real_A)
                        real_LB_logit,real_GB_logit, real_B_cam_logit, _, real_B_z = self.netD_B(real_B)

                        fake_A2B = self.netG_A2B(real_A_z)
                        fake_B2A = self.netG_B2A(real_B_z)

                        fake_B2A = fake_B2A.detach()
                        fake_A2B = fake_A2B.detach()

                        fake_LA_logit, fake_GA_logit, fake_A_cam_logit, _, _ = self.netD_A(fake_B2A)
                        fake_LB_logit, fake_GB_logit, fake_B_cam_logit, _, _ = self.netD_B(fake_A2B)

                        D_ad_loss_GA = self.MSE_loss(real_GA_logit, torch.ones_like(real_GA_logit).cuda()) +                                                   self.MSE_loss(fake_GA_logit, torch.zeros_like(fake_GA_logit).cuda())
                        D_ad_loss_LA = self.MSE_loss(real_LA_logit, torch.ones_like(real_LA_logit).cuda()) +                                                   self.MSE_loss(fake_LA_logit, torch.zeros_like(fake_LA_logit).cuda())
                        D_ad_loss_GB = self.MSE_loss(real_GB_logit, torch.ones_like(real_GB_logit).cuda()) +                                                   self.MSE_loss(fake_GB_logit, torch.zeros_like(fake_GB_logit).cuda())
                        D_ad_loss_LB = self.MSE_loss(real_LB_logit, torch.ones_like(real_LB_logit).cuda()) +                                                   self.MSE_loss(fake_LB_logit, torch.zeros_like(fake_LB_logit).cuda())            
                        D_ad_cam_loss_A = self.MSE_loss(real_A_cam_logit, torch.ones_like(real_A_cam_logit).cuda()) +                                               self.MSE_loss(fake_A_cam_logit, torch.zeros_like(fake_A_cam_logit).cuda())
                        D_ad_cam_loss_B = self.MSE_loss(real_B_cam_logit, torch.ones_like(real_B_cam_logit).cuda()) +                                               self.MSE_loss(fake_B_cam_logit, torch.zeros_like(fake_B_cam_logit).cuda())

                        loss_D_A = (D_ad_loss_GA + D_ad_cam_loss_A + D_ad_loss_LA)
                        loss_D_B = (D_ad_loss_GB + D_ad_cam_loss_B + D_ad_loss_LB)

                        Discriminator_loss = self.config['Adv_lamda'] * (loss_D_A + loss_D_B)
                        Discriminator_loss.backward()
                        self.optimizer_D_A.step()
                        self.optimizer_D_B.step()
                        # Update G
                        self.optimizer_G.zero_grad()
                        self.optimizer_R_A.zero_grad()

                        _,  _,  _, _, real_A_z = self.netD_A(real_A)
                        _,  _,  _, _, real_B_z = self.netD_B(real_B)

                        fake_A2B = self.netG_A2B(real_A_z)
                        fake_B2A = self.netG_B2A(real_B_z)
                        
                        fake_A2A = self.netG_B2A(real_A_z)
                        fake_B2B = self.netG_A2B(real_B_z)

                        fake_LA_logit, fake_GA_logit, fake_A_cam_logit, _, fake_A_z = self.netD_A(fake_B2A)
                        fake_LB_logit, fake_GB_logit, fake_B_cam_logit, _, fake_B_z = self.netD_B(fake_A2B)

                        fake_B2A2B = self.netG_A2B(fake_A_z)
                        fake_A2B2A = self.netG_B2A(fake_B_z)

                        Trans = self.R_A(fake_A2B,real_B) 
                        SysRegist_A2B = self.spatial_transform(fake_A2B,Trans)
                        SR_loss = self.config['Corr_lamda'] * self.L1_loss(SysRegist_A2B,real_B)###SR
                        SM_loss = self.config['Smooth_lamda'] * smooothing_loss(Trans)
                        
                        
                        G_ad_loss_GA = self.MSE_loss(fake_GA_logit, torch.ones_like(fake_GA_logit).cuda())
                        G_ad_loss_LA = self.MSE_loss(fake_LA_logit, torch.ones_like(fake_LA_logit).cuda())
                        G_ad_loss_GB = self.MSE_loss(fake_GB_logit, torch.ones_like(fake_GB_logit).cuda())
                        G_ad_loss_LB = self.MSE_loss(fake_LB_logit, torch.ones_like(fake_LB_logit).cuda())
                        G_ad_cam_loss_A = self.MSE_loss(fake_A_cam_logit, torch.ones_like(fake_A_cam_logit).cuda())
                        G_ad_cam_loss_B = self.MSE_loss(fake_B_cam_logit, torch.ones_like(fake_B_cam_logit).cuda())

                        G_cycle_loss_A = self.L1_loss(fake_A2B2A, real_A)
                        G_cycle_loss_B = self.L1_loss(fake_B2A2B, real_B)

                       
                        G_ident_loss_A = self.L1_loss(fake_A2A, real_A)
                        G_ident_loss_B = self.L1_loss(fake_B2B, real_B) 
                        


                        G_loss_A = self.config['Adv_lamda'] * (G_ad_loss_GA + G_ad_cam_loss_A + G_ad_loss_LA ) + self.config['Cyc_lamda'] * ( G_cycle_loss_A +  G_ident_loss_A )
                        G_loss_B = self.config['Adv_lamda'] * (G_ad_loss_GB + G_ad_cam_loss_B + G_ad_loss_LB ) + self.config['Cyc_lamda'] * (G_cycle_loss_B + G_ident_loss_B )

                        Generator_loss = G_loss_A + G_loss_B
                        
                        Total_loss = Generator_loss + SR_loss + SM_loss
                        Total_loss.backward()
                        self.optimizer_G.step()
                        self.optimizer_R_A.step()
                        ################################### 
                    
                    else: 
                        self.optimizer_D_A.zero_grad()
                        self.optimizer_D_B.zero_grad()

                        real_LA_logit,real_GA_logit, real_A_cam_logit, _, real_A_z = self.netD_A(real_A)
                        real_LB_logit,real_GB_logit, real_B_cam_logit, _, real_B_z = self.netD_B(real_B)

                        fake_A2B = self.netG_A2B(real_A_z)
                        fake_B2A = self.netG_B2A(real_B_z)

                        fake_B2A = fake_B2A.detach()
                        fake_A2B = fake_A2B.detach()

                        fake_LA_logit, fake_GA_logit, fake_A_cam_logit, _, _ = self.netD_A(fake_B2A)
                        fake_LB_logit, fake_GB_logit, fake_B_cam_logit, _, _ = self.netD_B(fake_A2B)

                        D_ad_loss_GA = self.MSE_loss(real_GA_logit, torch.ones_like(real_GA_logit).cuda()) +                                                   self.MSE_loss(fake_GA_logit, torch.zeros_like(fake_GA_logit).cuda())
                        D_ad_loss_LA = self.MSE_loss(real_LA_logit, torch.ones_like(real_LA_logit).cuda()) +                                                   self.MSE_loss(fake_LA_logit, torch.zeros_like(fake_LA_logit).cuda())
                        D_ad_loss_GB = self.MSE_loss(real_GB_logit, torch.ones_like(real_GB_logit).cuda()) +                                                   self.MSE_loss(fake_GB_logit, torch.zeros_like(fake_GB_logit).cuda())
                        D_ad_loss_LB = self.MSE_loss(real_LB_logit, torch.ones_like(real_LB_logit).cuda()) +                                                   self.MSE_loss(fake_LB_logit, torch.zeros_like(fake_LB_logit).cuda())            
                        D_ad_cam_loss_A = self.MSE_loss(real_A_cam_logit, torch.ones_like(real_A_cam_logit).cuda()) +                                               self.MSE_loss(fake_A_cam_logit, torch.zeros_like(fake_A_cam_logit).cuda())
                        D_ad_cam_loss_B = self.MSE_loss(real_B_cam_logit, torch.ones_like(real_B_cam_logit).cuda()) +                                               self.MSE_loss(fake_B_cam_logit, torch.zeros_like(fake_B_cam_logit).cuda())

                        loss_D_A = self.config['Adv_lamda'] * (D_ad_loss_GA + D_ad_cam_loss_A + D_ad_loss_LA)
                        loss_D_B = self.config['Adv_lamda'] * (D_ad_loss_GB + D_ad_cam_loss_B + D_ad_loss_LB)

                        Discriminator_loss = loss_D_A + loss_D_B
                        Discriminator_loss.backward()
                        self.optimizer_D_A.step()
                        self.optimizer_D_B.step()
                        # Update G
                        self.optimizer_G.zero_grad()

                        _,  _,  _, _, real_A_z = self.netD_A(real_A)
                        _,  _,  _, _, real_B_z = self.netD_B(real_B)

                        fake_A2B = self.netG_A2B(real_A_z)
                        fake_B2A = self.netG_B2A(real_B_z)
                        
                        fake_A2A = self.netG_B2A(real_A_z)
                        fake_B2B = self.netG_A2B(real_B_z)

                        fake_LA_logit, fake_GA_logit, fake_A_cam_logit, _, fake_A_z = self.netD_A(fake_B2A)
                        fake_LB_logit, fake_GB_logit, fake_B_cam_logit, _, fake_B_z = self.netD_B(fake_A2B)

                        fake_B2A2B = self.netG_A2B(fake_A_z)
                        fake_A2B2A = self.netG_B2A(fake_B_z)

                        G_ad_loss_GA = self.MSE_loss(fake_GA_logit, torch.ones_like(fake_GA_logit).cuda())
                        G_ad_loss_LA = self.MSE_loss(fake_LA_logit, torch.ones_like(fake_LA_logit).cuda())
                        G_ad_loss_GB = self.MSE_loss(fake_GB_logit, torch.ones_like(fake_GB_logit).cuda())
                        G_ad_loss_LB = self.MSE_loss(fake_LB_logit, torch.ones_like(fake_LB_logit).cuda())
                        G_ad_cam_loss_A = self.MSE_loss(fake_A_cam_logit, torch.ones_like(fake_A_cam_logit).cuda())
                        G_ad_cam_loss_B = self.MSE_loss(fake_B_cam_logit, torch.ones_like(fake_B_cam_logit).cuda())

                        G_cycle_loss_A = self.L1_loss(fake_A2B2A, real_A)
                        G_cycle_loss_B = self.L1_loss(fake_B2A2B, real_B)

                        G_ident_loss_A = self.L1_loss(fake_A2A, real_A)
                        G_ident_loss_B = self.L1_loss(fake_B2B, real_B) 

                        G_loss_A = self.config['Adv_lamda'] * (G_ad_loss_GA + G_ad_cam_loss_A + G_ad_loss_LA ) + self.config['Cyc_lamda'] * (G_cycle_loss_A +  G_ident_loss_A )
                        G_loss_B = self.config['Adv_lamda'] * (G_ad_loss_GB + G_ad_cam_loss_B + G_ad_loss_LB ) + self.config['Cyc_lamda'] * (G_cycle_loss_B +  G_ident_loss_B )

                        Generator_loss = G_loss_A + G_loss_B
                        Generator_loss.backward()
                        self.optimizer_G.step()
                        ###################################
                        
                        
                        
                else:                  # s dir
                    if self.config['regist']:    # + reg 
                        self.optimizer_R_A.zero_grad()
                        self.optimizer_G.zero_grad()
                        #### regist sys loss

                        fake_A2B = self.netG_A2B(real_A)
                        Trans = self.R_A(fake_A2B,real_B)
                        SysRegist_A2B = self.spatial_transform(fake_A2B,Trans)
  

                        SR_loss = self.config['Corr_lamda'] * self.L1_loss(SysRegist_A2B,real_B)###SR


                        fake_LB_logit, fake_GB_logit, fake_B_cam_logit = self.netD_B(fake_A2B)


                        adv_loss = self.config['Adv_lamda'] * (self.MSE_loss(fake_GB_logit, torch.ones_like(fake_GB_logit).cuda())+                                                        self.MSE_loss(fake_LB_logit, torch.ones_like(fake_LB_logit).cuda())+                                                        self.MSE_loss(fake_B_cam_logit, torch.ones_like(fake_B_cam_logit).cuda()))
                        ####smooth loss
                        SM_loss = self.config['Smooth_lamda'] * smooothing_loss(Trans)

                        toal_loss = SM_loss+adv_loss+SR_loss

                        toal_loss.backward()
                        self.optimizer_R_A.step()
                        self.optimizer_G.step()
                        
                        self.optimizer_D_B.zero_grad()
                        with torch.no_grad():
                            fake_B = self.netG_A2B(real_A)
              
                        fake_LB_logit, fake_GB_logit, fake_B_cam_logit = self.netD_B(fake_B)
                        real_LB_logit, real_GB_logit, real_B_cam_logit = self.netD_B(real_B)
                        
                        loss_D_B = self.config['Adv_lamda'] * (self.MSE_loss(fake_GB_logit, torch.zeros_like(fake_GB_logit).cuda())+                                                        self.MSE_loss(fake_LB_logit, torch.zeros_like(fake_LB_logit).cuda())+                                                        self.MSE_loss(fake_B_cam_logit, torch.zeros_like(fake_B_cam_logit).cuda())+                                                    self.MSE_loss(real_GB_logit, torch.ones_like(real_GB_logit).cuda())+                                                        self.MSE_loss(real_LB_logit, torch.ones_like(real_LB_logit).cuda()) +                                                        self.MSE_loss(real_B_cam_logit, torch.ones_like(real_B_cam_logit).cuda())    )


                        loss_D_B.backward()
                        self.optimizer_D_B.step()
                        
                        
                        
                    else:
                        self.optimizer_G.zero_grad()
                        fake_A2B = self.netG_A2B(real_A)
                        #### GAN aligin loss
                        fake_LB_logit, fake_GB_logit, fake_B_cam_logit= self.netD_B(fake_A2B)
 
                        adv_loss = self.config['Adv_lamda'] * (self.MSE_loss(fake_GB_logit, torch.ones_like(fake_GB_logit).cuda())+                                                        self.MSE_loss(fake_LB_logit, torch.ones_like(fake_LB_logit).cuda())+                                                        self.MSE_loss(fake_B_cam_logit, torch.ones_like(fake_B_cam_logit).cuda()))
          
                        adv_loss.backward()
                        self.optimizer_G.step()
                        ###### Discriminator B ######
                        self.optimizer_D_B.zero_grad()
                        with torch.no_grad():
                            fake_A2B = self.netG_A2B(real_A)
                        # Real loss
                        real_LB_logit,real_GB_logit, real_B_cam_logit = self.netD_B(real_B)
                        
                        
                        loss_D_real = self.MSE_loss(real_GB_logit, torch.ones_like(real_GB_logit).cuda())+                                                        self.MSE_loss(real_LB_logit, torch.ones_like(real_LB_logit).cuda())+                                                          self.MSE_loss(real_B_cam_logit, torch.ones_like(real_B_cam_logit).cuda())      
        
                        # Fake loss
                        
                        fake_LB_logit, fake_GB_logit, fake_B_cam_logit = self.netD_B(fake_A2B)
                        loss_D_fake = self.MSE_loss(fake_GB_logit, torch.zeros_like(fake_GB_logit).cuda())+                                                        self.MSE_loss(fake_LB_logit, torch.zeros_like(fake_LB_logit).cuda())+                                                        self.MSE_loss(fake_B_cam_logit, torch.zeros_like(fake_B_cam_logit).cuda())
                        # Total loss
                        loss_D_B = self.config['Adv_lamda'] * (loss_D_real + loss_D_fake)
                        loss_D_B.backward()
                        self.optimizer_D_B.step()
                        ###################################
                        #'adv_loss':adv_loss,'SM':SM_loss,'SR_loss':SR_loss

                self.logger.log({'loss_D_B': loss_D_B,},
                       images={'real_A': real_A, 'real_B': real_B, 'fake_B': fake_A2B})#,'SR':SysRegist_A2B



    #         # Save models checkpoints
            if not os.path.exists(self.config["save_root"]):
                os.makedirs(self.config["save_root"])
            torch.save(self.netG_A2B.state_dict(), self.config['save_root'] + 'netG_A2B.pth')
            # torch.save(self.netG_A2B.state_dict(), self.config['save_root'] + 'netG_A2B.pth')
            # torch.save(self.netG_B2A.state_dict(), self.config['save_root'] +'netG_B2A.pth')
            # torch.save(self.netD_A.state_dict(), self.config['save_root'] +'netD_A.pth')
            # torch.save(self.netD_B.state_dict(), self.config['save_root'] +'netD_B.pth')
            
            
            #############val###############
            with torch.no_grad():
                MAE = 0
                num = 0
                for i, batch in enumerate(self.val_data):
                    real_A = Variable(self.input_A.copy_(batch['A']))
                    real_B = Variable(self.input_B.copy_(batch['B'])).detach().cpu().numpy().squeeze()
                    if self.config['bidirect']:
                        _,  _,  _, _, real_A_z = self.netD_A(real_A)
                        fake_B = self.netG_A2B(real_A_z).detach().cpu().numpy().squeeze()
                    else:
                        fake_B = self.netG_A2B(real_A).detach().cpu().numpy().squeeze()
                    mae = self.MAE(fake_B,real_B)
                    MAE += mae
                    num += 1
                 
                print ('MAE:',MAE/num)
                
                    
                         
    def test(self,):
        self.netG_A2B.load_state_dict(torch.load(self.config['save_root'] + 'netG_A2B.pth'))
        with torch.no_grad():
                MAE = 0
                PSNR = 0
                SSIM = 0
                num = 0
                
                for i, batch in enumerate(self.val_data):
                    real_A = Variable(self.input_A.copy_(batch['A']))
                    real_B = Variable(self.input_B.copy_(batch['B'])).detach().cpu().numpy().squeeze()
                    fake_B = self.netG_A2B(real_A).detach().cpu().numpy().squeeze()
                    
                    mae = self.MAE(fake_B,real_B)
                    
                    psnr = self.PSNR(fake_B,real_B)
                    ssim = measure.compare_ssim(fake_B,real_B)
                    
                    MAE += mae
                    PSNR += psnr
                    SSIM += ssim 
                    num += 1

                print ('MAE:',MAE/num)
                print ('PSNR:',PSNR/num)
                print ('SSIM:',SSIM/num)
    
    def PSNR(self,fake,real):
       x,y = np.where(real!= -1)
       mse = np.mean(((fake[x][y]+1)/2. - (real[x][y]+1)/2.) ** 2 )
       if mse < 1.0e-10:
          return 100
       else:
           PIXEL_MAX = 1
           return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
            
            
    def MAE(self,fake,real):
        x,y = np.where(real!= -1)  # coordinate of target points
        #points = len(x)  #num of target points
        mae = np.abs(fake[x,y]-real[x,y]).mean()
            
        return mae/2    
            
            
            

