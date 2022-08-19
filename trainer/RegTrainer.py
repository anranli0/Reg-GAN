#!/usr/bin/python3

import argparse
import itertools
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from .utils import LambdaLR,Logger,ReplayBuffer
from .utils import weights_init_normal,get_config
from .datasets import get_datasets
from Model.CycleGan import *
from .utils import Resize,ToTensor,smooothing_loss
from .utils import Logger
from .reg import Reg
from .transformer import Transformer_2D
from skimage import measure
import numpy as np
import cv2
import re

import SimpleITK as sitk

class Reg_Trainer():
    def __init__(self, config):
        super().__init__()
        assert config['regist'] and (not config['bidirect'])
        self.config = config
        ## def networks
        #FIXME:
        self.netG_A2B = Generator(config['input_nc'], config['output_nc']).cuda()
        self.netD_B = Discriminator(config['output_nc']).cuda()
        self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        
        self.R_A = Reg(config['size'], config['size'], config['output_nc'], config['output_nc']).cuda()
        self.spatial_transform = Transformer_2D().cuda()
        self.optimizer_R_A = torch.optim.Adam(self.R_A.parameters(), lr=config['lr'], betas=(0.5, 0.999))
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

        # Dataset loader

        # transformsA = [ToTensor(), Resize((config['size'], config['size'], config['input_nc']))]
        # transformsB = [ToTensor(), Resize((config['size'], config['size'], config['output_nc']))]

        train, val = get_datasets(
                        config['source_root'], config['target_root'], 
                        config['input_nc'], config['output_nc'], config['size']
                        )

        print(f"train {len(train)} samples, val {len(val)} samples")

        self.dataloader = DataLoader(
            train,
            batch_size=config['batchSize'], 
            shuffle=True, 
            num_workers=config['n_cpu'])
        
        self.val_data = DataLoader(
            val,
            batch_size=config['batchSize'], 
            shuffle=False, 
            num_workers=config['n_cpu'])

       # Loss plot
        self.logger = Logger(config['name'],config['port'],config['n_epochs'], len(self.dataloader))       
        
    def train(self):
        # load weights if saved from previous training
        weight_path = self.config['save_root'] + 'netG_A2B.pth'
        if os.path.exists(weight_path):
            print("loading weights A2B")
            self.netG_A2B.load_state_dict(torch.load(weight_path))
        
        weight_path_c = self.config['save_root'] + 'netG_B2A.pth'
        if self.config['bidirect'] and os.path.exists(weight_path_c):
            print("loading weights B2A")
            self.netG_B2A.load_state_dict(torch.load(weight_path_c))

        weight_path_reg = self.config['save_root'] + 'Regist.pth'
        if self.config['regist'] and os.path.exists(weight_path_reg):
            print("loading weights registration ")
            self.R_A.load_state_dict(torch.load(weight_path_reg))

        ###### Training ######
        for epoch in range(self.config['epoch'], self.config['n_epochs']):
            skip_count = 0
            for i, batch in enumerate(self.dataloader):

                if torch.quantile(batch['B'], 0.9) == -1:
                    skip_count += 1
                    continue

                # Set model input
                real_A = Variable(self.input_A.copy_(batch['A']))
                real_B = Variable(self.input_B.copy_(batch['B']))
                
                # NC+R
                self.optimizer_R_A.zero_grad()
                self.optimizer_G.zero_grad()
                #### regist sys loss
                fake_B = self.netG_A2B(real_A)
                Trans = self.R_A(fake_B,real_B) 
                SysRegist_A2B = self.spatial_transform(fake_B,Trans)
                SR_loss = self.config['Corr_lamda'] * self.L1_loss(SysRegist_A2B,real_B)###SR
                pred_fake0 = self.netD_B(fake_B)
                adv_loss = self.config['Adv_lamda'] * self.MSE_loss(pred_fake0, self.target_real)
                ####smooth loss
                SM_loss = self.config['Smooth_lamda'] * smooothing_loss(Trans)
                toal_loss = SM_loss + adv_loss + SR_loss
                toal_loss.backward()
                self.optimizer_R_A.step()
                self.optimizer_G.step()
                self.optimizer_D_B.zero_grad()
                with torch.no_grad():
                    fake_B = self.netG_A2B(real_A)
                pred_fake0 = self.netD_B(fake_B)
                pred_real = self.netD_B(real_B)
                loss_D_B = self.config['Adv_lamda'] * self.MSE_loss(pred_fake0, self.target_fake)+ \
                           self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)

                loss_D_B.backward()
                self.optimizer_D_B.step()

                
                self.logger.log(
                    {'loss_D_B': loss_D_B, 'SR_loss':SR_loss, 'adv_loss':adv_loss},
                    images={'real_A': real_A, 'real_B': real_B, 'fake_B': fake_B})#,'SR':SysRegist_A2B

            # Save models checkpoints
            if not os.path.exists(self.config["save_root"]):
                os.makedirs(self.config["save_root"])
            torch.save(self.netG_A2B.state_dict(), self.config['save_root'] + 'netG_A2B.pth')
            torch.save(self.netD_B.state_dict(), self.config['save_root'] + 'netD_B.pth')
            torch.save(self.R_A.state_dict(), self.config['save_root'] + 'Regist.pth')

            if epoch == 0:
                print(f"\n*** skipped {skip_count} images ***")
            #############val###############
            with torch.no_grad():
                MAE = 0
                num = 0
                for i, batch in enumerate(self.val_data):
                    real_A = Variable(self.input_A.copy_(batch['A']))
                    real_B = Variable(self.input_B.copy_(batch['B']))
                    fake_B = self.netG_A2B(real_A).detach().cpu().numpy().squeeze()
                    
                    # trans = self.R_A(fake_B, real_B).squeeze()
                    # self.save_deformation(self, trans, root='')
                    mae = self.MAE(fake_B,real_B.detach().cpu().numpy().squeeze())
                    MAE += mae
                    num += 1

                    if i % 50 == 0:
                        if not os.path.exists(self.config["image_save"]):
                            os.makedirs(self.config["image_save"])
                        serial = re.split('/|\.', self.val_data.dataset.files_A[i])[-2]
                        
                        image_FB = 255 * ((fake_B + 1) / 2)
                        size = self.config['size']
                        image_FB = image_FB.reshape(size, size, -1)

                        image_fname = self.config["image_save"] + serial + "fakeT"
                        if self.config['output_nc'] == 1 or self.config['output_nc'] == 3:
                            cv2.imwrite(image_fname + ".png", image_FB)
                        else:
                            image_FB = image_FB.reshape(image_FB.shape[::-1])
                            sitk.WriteImage(sitk.GetImageFromArray(image_FB), image_fname + ".tif")

                print ('\nVal MAE:',MAE/num)
                
                    
    def PSNR(self,fake,real):
       x,y = np.where(real!= -1)# Exclude background
       mse = np.mean(((fake[x][y]+1)/2. - (real[x][y]+1)/2.) ** 2 )
       if mse < 1.0e-10:
          return 100
       else:
           PIXEL_MAX = 1
           return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
            
            
    def MAE(self,fake,real):
        if len(real.shape) == 2:
            x, y = np.where(real!= -1)  # coordinate of target points
            mae = np.abs(fake[x,y]-real[x,y]).mean()
        else:
            x, y, z = np.where(real != -1)
            mae = np.abs(fake[x,y,z] - real[x,y,z]).mean()
            
        return mae/2 
            

    def save_deformation(self,defms,root):
        heatmapshow = None
        defms_ = defms.data.cpu().float().numpy()
        dir_x = defms_[0]
        dir_y = defms_[1]
        x_max,x_min = dir_x.max(),dir_x.min()
        y_max,y_min = dir_y.max(),dir_y.min()
        dir_x = ((dir_x-x_min)/(x_max-x_min))*255
        dir_y = ((dir_y-y_min)/(y_max-y_min))*255
        tans_x = cv2.normalize(dir_x, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        tans_x[tans_x<=150] = 0
        tans_x = cv2.applyColorMap(tans_x, cv2.COLORMAP_JET)
        tans_y = cv2.normalize(dir_y, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        tans_y[tans_y<=150] = 0
        tans_y = cv2.applyColorMap(tans_y, cv2.COLORMAP_JET)
        gradxy = cv2.addWeighted(tans_x, 0.5,tans_y, 0.5, 0)

        cv2.imwrite(root, gradxy) 
