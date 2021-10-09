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
        self.netD_B = Discriminator(config['input_nc']+1).cuda()
        self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        
        if config['regist']:
            self.R_A = Reg().cuda()
            self.spatial_transform = Transformer_2D().cuda()
            
            self.optimizer_R_A = torch.optim.Adam(self.R_A.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        if config['bidirect']:
            self.netG_B2A = Generator(config['input_nc'], config['output_nc']).cuda()
            self.netD_A = Discriminator(config['input_nc']).cuda()
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()),lr=config['lr'], betas=(0.5, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=config['lr'], betas=(0.5, 0.999))
            
            
        else:
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
        transforms_1 = [ToPILImage(),
                  RandomAffine(degrees=5,translate=[0.1, 0.1],scale=[0.9, 1.1],fillcolor=-1),#degrees=2,translate=[0.05, 0.05],scale=[0.9, 1.1]
                   ToTensor(),
                   Resize(size_tuple = (config['size'], config['size']))]
    
        transforms_2 = [ToPILImage(),
                   RandomAffine(degrees=5,translate=[0.1, 0.1],scale=[0.9, 1.1],fillcolor=-1),
                   ToTensor(),
                   Resize(size_tuple = (config['size'], config['size']))]

        self.dataloader = DataLoader(ImageDataset(config['dataroot'], transforms_1=transforms_1, transforms_2=transforms_2, unaligned=False),
                                batch_size=config['batchSize'], shuffle=True, num_workers=config['n_cpu'])
        
        
        val_transforms = [ToTensor(),
                    Resize(size_tuple = (config['size'], config['size']))]
        
        self.val_data = DataLoader(ValDataset(config['val_dataroot'], transforms_ =val_transforms, unaligned=False),
                                batch_size=config['batchSize'], shuffle=False, num_workers=config['n_cpu'])

 
       # Loss plot
        self.logger = Logger(config['name'],config['n_epochs'], len(self.dataloader))       
        
    def train(self):
        ###### Training ######
        for epoch in range(self.config['epoch'], self.config['n_epochs']):
            for i, batch in enumerate(self.dataloader):
                # Set model input
                real_A = Variable(self.input_A.copy_(batch['T1']))
                real_B = Variable(self.input_B.copy_(batch['T2']))
               
                self.optimizer_G.zero_grad()

                
                fake_B = self.netG_A2B(real_A)
                loss_L1 = self.L1_loss(fake_B, real_B)*100
                
                # gan loss: 
                fake_AB = torch.cat((real_A, fake_B), 1)
                pred_fake = self.netD_B(fake_AB)
                loss_GAN_A2B = self.MSE_loss(pred_fake, self.target_real)

                # Total loss
                toal_loss = loss_L1 + loss_GAN_A2B
                toal_loss.backward()
                self.optimizer_G.step()
        

                self.optimizer_D_B.zero_grad()
                with torch.no_grad():
                    fake_B = self.netG_A2B(real_A)
                    #Trans = self.R_A(fake_B,real_B)
                    #SysRegist_A2B = self.spatial_transform(fake_B,Trans)



                pred_fake0 = self.netD_B(torch.cat((real_A, fake_B), 1))
                pred_real = self.netD_B(torch.cat((real_A, real_B), 1))
                loss_D_B = self.MSE_loss(pred_fake0, self.target_fake)+self.MSE_loss(pred_real, self.target_real)


                loss_D_B.backward()
                self.optimizer_D_B.step()
                self.logger.log({'loss_D_B': loss_D_B,'loss_G':toal_loss,},
                       images={'real_A': real_A, 'real_B': real_B, 'fake_B': fake_B,})#,'SR':SysRegist_A2B



                    
                    
                
                
                
    #         # Save models checkpoints
            torch.save(self.netG_A2B.state_dict(), self.config['save_root'] + 'netG_A2B.pth')
            #torch.save(netG_B2A.state_dict(), 'output/netG_B2A_3D.pth')
            #torch.save(netD_A.state_dict(), 'output/netD_A_3D.pth')
            #torch.save(netD_B.state_dict(), 'output/netD_B_3D.pth')
            
            
            #############val###############
            with torch.no_grad():
                MAE = 0
                num = 0
                for i, batch in enumerate(self.val_data):
                    real_A = Variable(self.input_A.copy_(batch['T1']))
                    real_B = Variable(self.input_B.copy_(batch['T2'])).detach().cpu().numpy().squeeze()
                    fake_B = self.netG_A2B(real_A).detach().cpu().numpy().squeeze()
                    mae = self.MAE(fake_B,real_B)
                    MAE += mae
                    num += 1
                
                
                    
                #LD = loss_D_B.item()
                #ADV = loss_G.item()
                #SM = SM_loss.item()
                #SR = SR_loss.item()
                L = [[MAE/num]] #LD,ADV,SR,MAE/num
                arr = np.array(L)
                
                if epoch == 0:
                    np.save(self.config['save_root']+'save', arr)
                else:
                    old_save = np.load(self.config['save_root']+'save.npy')
                    new_save = np.concatenate([old_save,arr],0)
                    
                    np.save(self.config['save_root']+'save', new_save)
                    
                    
                    
                print ('MAE:',MAE/num)
                
                    
                         
    def test(self,):
        self.netG_A2B.load_state_dict(torch.load(self.config['save_root'] + 'netG_A2B.pth'))
        with torch.no_grad():
                MAE = 0
                PSNR = 0
                SSIM = 0
                num = 0
                
                for i, batch in enumerate(self.val_data):
                    real_A = Variable(self.input_A.copy_(batch['T1']))
                    real_B = Variable(self.input_B.copy_(batch['T2'])).detach().cpu().numpy().squeeze()
                    fake_B = self.netG_A2B(real_A).detach().cpu().numpy().squeeze()
                    
                    mae = self.MAE(fake_B,real_B)
                    
                    psnr = self.PSNR(fake_B,real_B)
                    ssim = measure.compare_ssim(fake_B,real_B)
                    
                    MAE += mae
                    PSNR += psnr
                    SSIM += ssim 
                    num += 1
                    image_FB = 255*((fake_B+1)/2)
                    cv2.imwrite(self.config['image_save']+ str(num)+'.png',image_FB)
                    
                    
                    
                 
#                 print ('MAE:',MAE/num)
#                 print ('PSNR:',PSNR/num)
#                 print ('SSIM:',SSIM/num)
    
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
            
            
            

