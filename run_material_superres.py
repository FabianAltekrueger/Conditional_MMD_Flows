#This code belongs to the paper
#P. Hagemann, J. Hertrich, F. Altekrüger, R. Beinert, J. Chemseddine, G. Steidl
#Posterior Sampling Based on Gradient Flows of the MMD with Negative Distance Kernel
#International Conference on Learning Representations.
#
#It reproduces the material's microstructure superresolution example from Section 4.

import torch
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader
import os
import skimage.io as io
import torch.nn.functional as F
import dival
from dival.reconstructors.networks.unet import get_unet_model
import utils as ut

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float

parser = argparse.ArgumentParser()
parser.add_argument('--visualize', type = bool, default = False,
                    help='Visualize the generated samples')
parser.add_argument('--save', type = bool, default = True,
                    help='Save images of particles during training')
args = parser.parse_args()

def gen_i():
    N = 100
    gt = ut.imread('material_data/test/hr_img.png')
    gt = F.pad(gt,pad=[40,40,40,40],mode='reflect')
    lr = operator(gt)
    lr += std*torch.randn_like(lr)
    hr_shape = gt.shape[2]
    x_new = torch.rand((N,channel,hr_shape,hr_shape),dtype=dtype,device=device)
    obs_tmp = F.interpolate(lr,scale_factor=scale_factor,mode='bicubic').clone()
    obs = obs_tmp.tile(N,1,1,1)
    x_new = torch.cat([x_new,obs],dim=1)
    with torch.no_grad():
        for n in tqdm(range(len(os.listdir(f'{img_path}/nets')))):
            new_net = get_unet_model(in_ch=2*channel,out_ch=channel,scales=6,
                        use_sigmoid=False,use_norm=False).to(device)
            new_net.load_state_dict(torch.load(f'{img_path}/nets/net{n}.pt',map_location=torch.device(device)))
            x_l = []
            for j in range(N//batch_size):
                x_tmp = x_new[j*batch_size:(j+1)*batch_size,...]
                x_tmp_z = x_tmp[:,:channel]
                x_l.append(x_tmp_z-new_net(x_tmp).detach())
            x_new = torch.cat(x_l,dim=0)
            x_new = torch.cat([x_new,obs],dim=1)
       
    x_new = x_new[:,:channel][...,40:-40,40:-40]
    mean = torch.mean(x_new,dim=0,keepdim=True)
    stds = torch.std(x_new,dim=0,keepdim=True)
    idx_2 = torch.randperm(x_new.shape[0])[:3]
    tmp = torch.cat([gt[...,40:-40,40:-40],F.interpolate(lr,scale_factor=scale_factor,mode='nearest')[...,40:-40,40:-40]],dim=0)
    ut.save_image(torch.cat([tmp,x_new[idx_2],mean,stds/stds.max()],dim=0),f'{img_path}/material_superresolution.png')
    exit()

if __name__ == '__main__':
    img_path = 'material_superresolution'
    if not os.path.isdir(f'{img_path}'):
        os.mkdir(f'{img_path}')  
    if not os.path.isdir(f'{img_path}/nets'):
        os.mkdir(f'{img_path}/nets')  
        
    #experiment configurations
    channel = 1
    hr_size = 100
    scale_factor = 4
    lr_size = hr_size//scale_factor
    m = 1000 #number samples
    d = channel*hr_size**2
    obs_dim = channel*lr_size**2
    n_projections = int(d/2)
    proj_batches = 1
    momentum = 0.7
    std = 0.01
    operator = ut.Downsample(scale=1/scale_factor,gaussian_std=int(scale_factor/2))
    s_factor = ut.sliced_factor(d+obs_dim)
    
    step_size = 1
    step_exp = 3
    opt_steps = 2**step_exp
    step = 0
    final_step = 30 #maximal number of networks
    
    new_net = get_unet_model(in_ch=2*channel,out_ch=channel,scales=6,
                            use_sigmoid=False,use_norm=False).to(device)
    net_num = 0
    train_steps = 4000
    batch_size = 10
    
    data_path = 'material_data'
    SiC_set = ut.createTrainset(img_path=f'{data_path}/train',operator=operator,std=std,size=hr_size)
    data = DataLoader(dataset=SiC_set,batch_size=m)
    SiC = next(iter(data))
    ground_truth = SiC[1].view(m,d).to(device)
    observation = SiC[0].view(m,obs_dim).to(device)
    target = torch.cat([ground_truth,observation],dim=1)
    
    if args.visualize:
        gen_i()

    particles = torch.rand((int(m),d),dtype=dtype,device=device)
    particles = torch.cat([particles,observation],dim=1)
    for i in tqdm(range(final_step)):
        p_old=torch.clone(particles)
        old_grad = torch.zeros_like(particles)
        for _ in range(opt_steps):
            MMD_grad=0.
            for _ in range(proj_batches):
                #fully-connected projections
                xi = torch.randn((n_projections,d+obs_dim),dtype=dtype,device=device)
                xi = xi/torch.sqrt(torch.sum(xi**2,-1,keepdim=True))
                xi = xi.unsqueeze(1)
                
                #slice particles
                particles_proj = F.conv1d(particles.reshape(1,1,-1),xi,stride=d+obs_dim).reshape(n_projections,-1)
                target_proj = F.conv1d(target.reshape(1,1,-1),xi,stride=d+obs_dim).reshape(n_projections,-1)
                
                #compute 1D gradient of MMD
                grad = ut.MMD_derivative_1d(particles_proj,target_proj)
                grad = grad.transpose(0,1)
                
                #compute MMD gradient based on 1D gradient
                xi = xi.reshape([n_projections,d+obs_dim]).transpose(0,1).flatten()
                MMD_grad += s_factor* F.conv1d(xi.reshape([1,1,-1]), grad.unsqueeze(1),
                            stride=n_projections).squeeze(0)/(n_projections*proj_batches)
            MMD_grad += momentum * old_grad
            MMD_grad[:,d:] = 0
            
            #update the flow
            particles -= step_size*m*MMD_grad
            old_grad = MMD_grad
            step += 1
        
        #train network
        many_grad = (p_old-particles)
        optim = torch.optim.Adam(new_net.parameters(), lr=0.0005)
        for ts in range(train_steps):
            perm = torch.randperm(many_grad.shape[0])[:batch_size]
            y_in = many_grad[perm]
            x_in = p_old[perm]
            x_in_z = x_in[:,:d].reshape(-1,channel,hr_size,hr_size)
            x_in_y = x_in[:,d:].reshape(-1,channel,lr_size,lr_size)
            x_in_y = F.interpolate(x_in_y,scale_factor=scale_factor,mode='bicubic')
            x_in = torch.cat([x_in_z,x_in_y],dim=1)
            loss = torch.sum((new_net(x_in).reshape(-1,d)-y_in[:,:d])**2)/batch_size
            optim.zero_grad()
            loss.backward()
            optim.step()
        torch.save(new_net.state_dict(),f'{img_path}/nets/net{net_num}.pt')
        net_num += 1
        
        #update particles
        with torch.no_grad():
            x_new=[]
            i=0
            while i<m:
                x_in = p_old[i:i+batch_size]
                x_in_z = x_in[:,:d].reshape(-1,channel,hr_size,hr_size)
                x_in_y = x_in[:,d:].reshape(-1,channel,lr_size,lr_size)
                x_in_y = F.interpolate(x_in_y,scale_factor=scale_factor,mode='bicubic')
                x_in = torch.cat([x_in_z,x_in_y],dim=1)
                x_new.append(x_in_z-new_net(x_in).detach())
                i += batch_size
            x_new = torch.cat(x_new,0).reshape(-1,d)
            x_new = torch.cat([x_new,observation],dim=1)
        particles = x_new.detach()    
        opt_plus = min(2**step_exp,1024)
        opt_steps = min(opt_steps+opt_plus,10000)
        step_exp += 1

        if args.save:
            ut.save_image(particles[:100,:d].reshape(-1,channel,hr_size,hr_size),f'{img_path}/flow{step}.png',10)
        
        
