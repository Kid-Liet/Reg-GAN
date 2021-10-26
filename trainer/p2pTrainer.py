#!/usr/bin/python3

import argparse
import itertools
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from .utils import LambdaLR,Logger,ReplayBuffer
from .utils import weights_init_normal,get_config
from .datasets import ImageDataset,ValDataset
from Model.CycleGan import *
from .utils import Resize,ToTensor,smooothing_loss
from .utils import Logger
from .reg import Reg
from torchvision.transforms import RandomAffine
from torchvision.transforms import RandomAffine,ToPILImage
from .transformer import Transformer_2D
from skimage import measure
import numpy as np
import cv2

class P2p_Trainer():
    def __init__(self, config):
        super().__init__()
        self.config = config
        ## def networks
        self.netG_A2B = Generator(config['input_nc'], config['output_nc']).cuda()
        self.netD_B = Discriminator(config['input_nc']*2).cuda()
        self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        self.optimizer_G = torch.optim.Adam(self.netG_A2B.parameters(), lr=config['lr'], betas=(0.5, 0.999))
            
            
                
        # Lossess
        self.MSE_loss = torch.nn.MSELoss()
        self.L1_loss = torch.nn.L1Loss()

        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if config['cuda'] else torch.Tensor
        self.input_A = Tensor(config['batchSize'], config['input_nc'], config['size'], config['size'])
        self.input_B = Tensor(config['batchSize'], config['output_nc'], config['size'], config['size'])
        self.target_real = Variable(Tensor(1,1).fill_(1.0), requires_grad=False)
        self.target_fake = Variable(Tensor(1,1).fill_(0.0), requires_grad=False)

        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

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
               
                self.optimizer_G.zero_grad()
                fake_B = self.netG_A2B(real_A)
                loss_L1 = self.L1_loss(fake_B, real_B) * self.config['P2P_lamda']
                # gan loss: 
                fake_AB = torch.cat((real_A, fake_B), 1)
                pred_fake = self.netD_B(fake_AB)
                loss_GAN_A2B = self.MSE_loss(pred_fake, self.target_real) * self.config['Adv_lamda']

                # Total loss
                toal_loss = loss_L1 + loss_GAN_A2B
                toal_loss.backward()
                self.optimizer_G.step()
        

                self.optimizer_D_B.zero_grad()
                with torch.no_grad():
                    fake_B = self.netG_A2B(real_A)
                pred_fake0 = self.netD_B(torch.cat((real_A, fake_B), 1)) * self.config['Adv_lamda']
                pred_real = self.netD_B(torch.cat((real_A, real_B), 1)) * self.config['Adv_lamda']
                loss_D_B = self.MSE_loss(pred_fake0, self.target_fake)+self.MSE_loss(pred_real, self.target_real)


                loss_D_B.backward()
                self.optimizer_D_B.step()
                self.logger.log({'loss_D_B': loss_D_B,'loss_G':toal_loss,},
                       images={'real_A': real_A, 'real_B': real_B, 'fake_B': fake_B,})#,'SR':SysRegist_A2B



                    
                    
                
                
                
    #         # Save models checkpoints
            if not os.path.exists(self.config["save_root"]):
                os.makedirs(self.config["save_root"])
            torch.save(self.netG_A2B.state_dict(), self.config['save_root'] + 'netG_A2B.pth')
            #torch.save(netG_B2A.state_dict(), 'output/netG_B2A_3D.pth')
            #torch.save(netD_A.state_dict(), 'output/netD_A_3D.pth')
            #torch.save(netD_B.state_dict(), 'output/netD_B_3D.pth')
            
            
            #############val###############
            with torch.no_grad():
                MAE = 0
                num = 0
                for i, batch in enumerate(self.val_data):
                    real_A = Variable(self.input_A.copy_(batch['A']))
                    real_B = Variable(self.input_B.copy_(batch['B'])).detach().cpu().numpy().squeeze()
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
            
            
            

