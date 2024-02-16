#This code belongs to the paper
#P. Hagemann, J. Hertrich, F. Altekrüger, R. Beinert, J. Chemseddine, G. Steidl
#Posterior Sampling Based on Gradient Flows of the MMD with Negative Distance Kernel
#International Conference on Learning Representations.
#
#It provides some helpful functions.

import torch
from torch import nn
from torchvision.utils import make_grid
import math
import matplotlib
import matplotlib.pyplot as plt
import skimage.io as io
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float

def MMD_derivative_1d(x,y,only_potential=False):
    '''
    compute the derivate of MMD in 1D
    '''
    N=x.shape[1]
    P=1
    if len(x.shape)>1:
        P=x.shape[0]
    # potential energy
    if y is None:
        grad=torch.zeros(P,N,dtype=dtype,device=device)
    else:
        M=y.shape[1]
        _,inds=torch.sort(torch.cat((x,y),1))
        grad=torch.where(inds>=N,1.,0.).type(dtype)
        grad=(2*torch.cumsum(grad,-1)-M) / (N*M)
        _,inverted_inds=torch.sort(inds)
        inverted_inds=inverted_inds[:,:N]+torch.arange(P,device=device).unsqueeze(1)*(N+M)
        inverted_inds=torch.flatten(inverted_inds)
        grad=grad.flatten()
        grad=grad[inverted_inds].reshape(P,-1)


    if not only_potential:
        _,inds_x=torch.sort(x)
        inds_x=inds_x+torch.arange(P,device=device).unsqueeze(1)*N
        inds_x=torch.flatten(inds_x)
        # interaction energy
        interaction=2*torch.arange(N,dtype=dtype,device=device)-N+1
        interaction=(1/(N**2)) * interaction
        interaction=interaction.tile(P,1)
        grad=grad.flatten()
        grad[inds_x]=grad[inds_x]-interaction.flatten()
        grad=grad.reshape(P,-1)

    return grad

def sliced_factor(d):
    '''
    compute the scaling factor of sliced MMD
    '''
    k=(d-1)//2
    fac=1.
    if (d-1)%2==0:
        for j in range(1,k+1):
            fac=2*fac*j/(2*j-1)
    else:
        for j in range(1,k+1):
            fac=fac*(2*j+1)/(2*j)
        fac=fac*math.pi/2
    return fac

class cut_patches_periodic_padding(torch.nn.Module):
    '''
    extract patch
    '''
    def __init__(self,img_height,img_width,channels,patch_size):
        super(cut_patches_periodic_padding,self).__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.channels = channels
        self.patch_size = patch_size
        
        self.patch_width=torch.zeros((channels,patch_size,patch_size),dtype=torch.long,device=device)
        self.patch_width+=torch.arange(patch_size,device=device)[None,None,:]
        self.patch_height=torch.zeros((channels,patch_size,patch_size),dtype=torch.long,device=device)
        self.patch_height+=torch.arange(patch_size,device=device)[None,:,None]
        
    def forward(self,imgs,position_inds_height,position_inds_width):
        N=imgs.shape[0]
        n_projections=position_inds_height.shape[0]
        patches_width=(self.patch_width[None,:,:,:].tile(n_projections,1,1,1)+position_inds_width[:,None,None,None])%self.img_width
        patches_height=(self.patch_height[None,:,:,:].tile(n_projections,1,1,1)+position_inds_height[:,None,None,None])%self.img_height
        linear_inds=patches_width+self.img_width*patches_height+(self.img_width*self.img_height)*torch.arange(self.channels,device=device)[None,:,None,None]
        linear_inds=linear_inds.reshape(n_projections,1,-1).tile(1,N,1)
        linear_inds+=(self.channels*self.img_height*self.img_width)*torch.arange(N,device=device)[None,:,None]
        linear_inds=linear_inds.reshape(-1)
        patches=imgs.reshape(-1)[linear_inds].reshape(n_projections,N,self.channels,self.patch_size,self.patch_size)
        return patches,linear_inds

def imread(img_name):
    '''
    loads an image as torch.tensor on the selected device
    '''
    np_img = io.imread(img_name)
    tens_img = torch.tensor(np_img, dtype=torch.float, device=device)
    if torch.max(tens_img) > 1:
        tens_img/=255                                                   
    if len(tens_img.shape) < 3:
        tens_img = tens_img.unsqueeze(2)                        
    if tens_img.shape[2] > 3:                                       
        tens_img = tens_img[:,:,:3]
    tens_img = tens_img.permute(2,0,1)  
    return tens_img.unsqueeze(0)    

def save_image(trajectory,name,rows=10):
    grid = make_grid(trajectory,nrow=rows,padding=1,pad_value=.5)
    if trajectory.shape[0] == 1:
        tmp = 0.5*torch.ones(1,1,30,30)
        tmp[...,1:-1,1:-1] = trajectory
        grid = tmp.squeeze(0).tile(3,1,1)
    plt.imsave(name,torch.clip(grid.permute(1,2,0),0,1).cpu().numpy())
    return

def createTrainset(img_path, operator, std, size = 100):
    '''
    Create a training set
    '''
    train = []
    picts = os.listdir(img_path)
    for img in picts:
        real = imread(f'{img_path}/{img}')
        for i in range(real.shape[2]//size):
            for j in range(real.shape[3]//size): 
                hr = real[:,:,i*size:(i+1)*size,j*size:(j+1)*size]
                lr = operator(hr).clone()
                lr = lr + std * torch.randn_like(lr)
                train.append([lr,hr])
    return train

class gaussian_downsample(nn.Module):
    '''
    Downsampling module with Gaussian filtering
    ''' 
    def __init__(self, kernel_size, sigma, stride, pad=False):
        super(gaussian_downsample, self).__init__()
        self.gauss = nn.Conv2d(1, 1, kernel_size, stride=stride, groups=1, bias=False)      
        gaussian_weights = self.init_weights(kernel_size, sigma)
        self.gauss.weight.data = gaussian_weights.to(device)
        self.gauss.weight.requires_grad_(False)
        self.pad = pad
        self.padsize = kernel_size-1

    def forward(self, x):
        if self.pad:
            x = torch.cat((x, x[:,:,:self.padsize,:]), 2)
            x = torch.cat((x, x[:,:,:,:self.padsize]), 3)
        return self.gauss(x)

    def init_weights(self, kernel_size, sigma):
        x_cord = torch.arange(kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t() 
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)
        mean = (kernel_size - 1)/2.
        variance = sigma**2.
        gaussian_kernel = (1./(2.*math.pi*variance))*torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1)/(2*variance))
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        return gaussian_kernel.view(1, 1, kernel_size, kernel_size)

def Downsample(scale = 0.25, gaussian_std = 2):
    ''' 
    downsamples an img by factor 4 using gaussian downsample from utils.py
    '''
    if scale > 1:
        print('Error. Scale factor is larger than 1.')
        return
    gaussian_std = gaussian_std
    kernel_size = 16
    gaussian_down = gaussian_downsample(kernel_size,gaussian_std,int(1/scale),pad=True) #gaussian downsample with zero padding
    return gaussian_down.to(device)
