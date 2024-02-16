#This code belongs to the paper
#P. Hagemann, J. Hertrich, F. Altekrüger, R. Beinert, J. Chemseddine, G. Steidl
#Posterior Sampling Based on Gradient Flows of the MMD with Negative Distance Kernel
#International Conference on Learning Representations.
#
#It reproduces the CIFAR10 classconditional example from Section 4.

import torch
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import torchvision.datasets as td
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import os
import skimage.io as io
import utils as ut
from unet import UNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float

parser = argparse.ArgumentParser()
parser.add_argument('--visualize', type = bool, default = False,
                    help='Visualize the generated samples')
parser.add_argument('--save', type = bool, default = True,
                    help='Save images of particles during training')
args = parser.parse_args()

def gen_i():
    N = 10
    batch_size = N
    down_factors = [8,4,2,1]
    net_changes = torch.load(f'{img_path}/net_changes.pt')
    x_classes = torch.tensor([],device=device,dtype=dtype)
    for i in tqdm(range(10)):
        obs = torch.zeros(1,10,device=device)
        obs[:,i] = amplifier
        x_new = torch.rand((N,channel,img_size//down_factors[0],img_size//down_factors[0]),dtype=dtype,device=device)
        obs = obs.tile(N,1)
        net_num = 0 
        factor_num = 0
        for n in range(len(os.listdir(f'{img_path}/nets'))):
            if net_num in net_changes:
                factor_num += 1
            input_h = img_size//down_factors[factor_num]
            net = get_UNET(input_h=input_h)
            net.load_state_dict(torch.load(f'{img_path}/nets/net{net_num}.pt',map_location=device))
            net.eval()
            x_l = []
            for i in range(N//batch_size):
                x_tmp = x_new[i*batch_size:(i+1)*batch_size,...]
                x_tmp_z = x_tmp.clone().reshape(-1,channel,input_h,input_h)
                x_tmp_y = obs.clone().reshape(-1,10,1,1).tile(1,1,input_h,input_h)
                x_tmp = torch.cat([x_tmp_z,x_tmp_y],dim=1)
                x_l.append(x_tmp_z-net(x_tmp).detach())
            x_new = torch.cat(x_l,0)
            net_num += 1
            if net_num in net_changes and down_factors[factor_num]>1:
                x_new = x_new.reshape(N,channel,input_h,1,input_h,1).tile(1,1,1,2,1,2).reshape(N,channel,2*input_h,2*input_h)
                x_new = x_new+.07*torch.randn_like(x_new)
        x_classes = torch.cat([x_classes,x_new.reshape(-1,channel,img_size,img_size)],dim=0)
    ut.save_image(x_classes,f'{img_path}/CIFAR_classconditional.png',rows=N)
    exit()

def get_UNET(input_h):
    return UNet(
        input_channels=10+channel,
        output_channels=channel,
        input_height=input_h,
        ch=32,
        ch_mult=(1, 2, 4),
        num_res_blocks=2,
        attn_resolutions=(256,),
        resamp_with_conv=True,).to(device) 

if __name__ == '__main__':
    dataset = 'CIFAR'
    img_path = 'CIFAR_classconditional'
    if not os.path.isdir(f'{img_path}'):
        os.mkdir(f'{img_path}')  
    if not os.path.isdir(f'{img_path}/nets'):
        os.mkdir(f'{img_path}/nets')  
    
    #experiment configurations
    channel = 3
    img_size = 32
    d = channel*img_size**2
    obs_dim = 10
    m = 40000 #number samples
    momentum = 0.
    patch_projections = 400
    proj_batches = 1
    amplifier = 10
    
    #steps of particle flow
    down_factors = [8,4,2,1]
    num_steps = [700000,700000,700000,1000000]
    step_sizes = [.5,1,1,1]
    step_exps = [4,10,13,13]
    net_changes = []
    
    #network
    train_steps = 4000
    batch_size = 100
    net_num = 0
    
    p_size = 7 #local size for projections
    p_dim = channel * p_size**2
    scales = [0,1,2,3]
    step_sizes_scale = [1,1,1,1]
    
    cifar = td.CIFAR10('cifar',transform=transforms.ToTensor(),download=True)
    data = DataLoader(dataset=cifar,batch_size=m)
    data = next(iter(data))
    data_tmp = data[0].view(m,channel*img_size**2).to(device)
    
    ground_truth = data_tmp.clone()
    label = F.one_hot(data[1]).to(device)
    observation = amplifier * label.clone()
    
    #if pretrained networks should be evaluated
    if args.visualize:
        gen_i()
    
    #pyramidal approach
    x = torch.rand((m,channel*(img_size//down_factors[0])**2),dtype=dtype,device=device)
    x = torch.cat([x,observation],dim=1)
    for factor_num,down_factor in enumerate(down_factors):
        y = F.avg_pool2d(ground_truth.reshape(m,channel,img_size,img_size),down_factor).reshape(m,-1)
        d = y.shape[-1]
        y = torch.cat([y,observation],dim=1)
        cur_size = img_size//down_factor
        cut_p = []
        for i in range(4):
            cut_p.append(ut.cut_patches_periodic_padding(cur_size,cur_size,channel,(i+1)*p_size))
        step_size = step_sizes[factor_num]
        step_exp = step_exps[factor_num]
        opt_steps = 2**step_exp
        step = 0
        
        s_factor = ut.sliced_factor(d+obs_dim)
        n_projections = max(500,d)
        proj_batches = 1
        new_net=get_UNET(input_h=cur_size)
        while True:
            old_grad = torch.zeros((m,d+obs_dim), device = device)
            x_old=torch.clone(x)
            for _ in tqdm(range(opt_steps)):
                if cur_size < 32:
                    MMD_grad=0.
                    for _ in range(proj_batches):
                        #fully-connected projections
                        xi = torch.randn((n_projections,d+obs_dim),dtype=dtype,device=device)
                        xi = xi/torch.sqrt(torch.sum(xi**2,-1,keepdim=True))
                        xi = xi.unsqueeze(1)
                        
                        x_proj = F.conv1d(x.reshape(1,1,-1),xi,stride=d+obs_dim).reshape(n_projections,-1)
                        y_proj = F.conv1d(y.reshape(1,1,-1),xi,stride=d+obs_dim).reshape(n_projections,-1)
                        
                        grad = ut.MMD_derivative_1d(x_proj,y_proj)
                        grad = grad.transpose(0,1)
                        
                        xi = xi.reshape([n_projections,d+obs_dim]).transpose(0,1).flatten()
                        MMD_grad = s_factor* F.conv1d(xi.reshape([1,1,-1]),grad.unsqueeze(1),
                                    stride=n_projections).squeeze()/n_projections + MMD_grad
                        MMD_grad[:,-obs_dim:] = 0
                else:
                    momentum = 0
                    MMD_grads = torch.zeros(m*(d),device=device)
                    for s in scales:
                        MMD_grad = 0.
                        for _ in range(proj_batches):
                            #locally-connected projections
                            xi_x = torch.randn((patch_projections,p_dim),dtype=dtype,device=device)
                            xi_x = xi_x.reshape(-1,channel,p_size,1,p_size,1).tile(1,1,(s+1),1,(s+1)).reshape(-1,channel*((s+1)*p_size)**2)
                            xi_y = torch.randn((patch_projections,obs_dim),dtype=dtype,device=device)
                            xi = torch.cat([xi_x,xi_y],dim=1)
                            xi = xi/torch.sqrt(torch.sum(xi**2,-1,keepdim=True))
                            xi = xi.unsqueeze(1)
                            
                            #extract patches
                            position_inds_height = torch.randint(0,cur_size,(1,),device=device)
                            position_inds_width = torch.randint(0,cur_size,(1,),device=device)
                            patches_x,linear_inds = cut_p[s](x[:,:d].reshape(-1,channel,cur_size,cur_size),
                                                        position_inds_height,position_inds_width)
                            patches_y,linear_inds = cut_p[s](y[:,:d].reshape(-1,channel,cur_size,cur_size),
                                                        position_inds_height,position_inds_width)
                            x_tmp = torch.cat([patches_x.reshape(m,-1),x[:,d:]],dim=1)
                            y_tmp = torch.cat([patches_y.reshape(m,-1),y[:,d:]],dim=1)
                            
                            #slice the flow
                            x_proj = F.conv1d(x_tmp.reshape(1,1,-1),xi,
                                        stride=channel*((s+1)*p_size)**2+obs_dim).reshape(patch_projections,-1)
                            y_proj = F.conv1d(y_tmp.reshape(1,1,-1),xi,
                                        stride=channel*((s+1)*p_size)**2+obs_dim).reshape(patch_projections,-1)
                            
                            #compute 1D gradient of MMD
                            grad = ut.MMD_derivative_1d(x_proj,y_proj)
                            grad = grad.transpose(0,1)
                            
                            #compute MMD gradient based on 1D gradient
                            xi = xi.reshape([patch_projections,channel*((s+1)*p_size)**2+obs_dim]).transpose(0,1).flatten()
                            MMD_grad = s_factor* F.conv1d(xi.reshape([1,1,-1]), grad.unsqueeze(1),
                                            stride=patch_projections).squeeze()/patch_projections + MMD_grad
                            MMD_grad = MMD_grad[:,:3*((s+1)*p_size)**2]
                        MMD_grads[linear_inds] += step_sizes_scale[s]*MMD_grad.reshape(-1)
                     
                    MMD_grad = MMD_grads.reshape(m,-1) 
                    MMD_grad = torch.cat([MMD_grad,torch.zeros(m,obs_dim,device=device,dtype=dtype)],dim=1)
                MMD_grad = MMD_grad/proj_batches + momentum*old_grad
                
                #update particles
                x -= step_size*m*MMD_grad
                old_grad = MMD_grad
                step += 1

            #train network
            many_grad = (x_old-x)
            optim = torch.optim.Adam(new_net.parameters(), lr=0.0005)
            for ts in range(train_steps):
                perm = torch.randperm(many_grad.shape[0])[:batch_size]
                y_in = many_grad[perm]
                x_in = x_old[perm]
                x_in_z = x_in[:,:d].reshape(-1,channel,cur_size,cur_size)
                x_in_y = x_in[:,d:].reshape(-1,10,1,1).tile(1,1,cur_size,cur_size)
                x_in = torch.cat([x_in_z,x_in_y],dim=1)
                loss = torch.sum((new_net(x_in).reshape(-1,d)-y_in[:,:d])**2)/batch_size
                optim.zero_grad()
                loss.backward()
                optim.step()
            torch.save(new_net.state_dict(),f'{img_path}/nets/net{net_num}.pt')
            net_num+=1
            
            #update particles
            with torch.no_grad():
                x_new=[]
                i=0
                while i<m:
                    x_in = x_old[i:i+batch_size]
                    x_in_z = x_in[:,:d].reshape(-1,channel,cur_size,cur_size)
                    x_in_y = x_in[:,d:].reshape(-1,10,1,1).tile(1,1,cur_size,cur_size)
                    x_in = torch.cat([x_in_z,x_in_y],dim=1)
                    x_new.append(x_in_z-new_net(x_in).detach())
                    i += batch_size
                x_new = torch.cat(x_new,0).reshape(-1,d)
                x_new = torch.cat([x_new,observation],dim=1)
            x = x_new.reshape(m,-1).detach()
            
            if args.save:
                ut.save_image(x[:100,:d].reshape(-1,channel,cur_size,cur_size),f'{img_path}/flow_net{net_num}.png',10)
            
            #update number of flow steps
            opt_plus = min(2**step_exp,1024)
            opt_steps = min(opt_steps+opt_plus,30000)
            momentum = min(0.8,momentum + 0.01)
            step_exp += 1
            if step>=num_steps[factor_num]:
                break
                
        #upsample to higher resolution        
        net_changes.append(net_num)
        torch.save(torch.tensor(net_changes),f'{img_path}/net_changes.pt')
        if down_factor>1:
            x = x[:,:d].reshape(m,channel,cur_size,1,cur_size,1).tile(1,1,1,2,1,2).reshape(m,3,2*cur_size,2*cur_size)
            x += .07*torch.randn_like(x)
            x = x.reshape(m,-1)
            x = torch.cat([x,observation],dim=1)
            
