#This code belongs to the paper
#P. Hagemann, J. Hertrich, F. Altekrüger, R. Beinert, J. Chemseddine, G. Steidl
#Posterior Sampling Based on Gradient Flows of the MMD with Negative Distance Kernel
#International Conference on Learning Representations.
#
#It reproduces the CT low-dose and limited angle example from Section 4.

import torch
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader
import os
import skimage.io as io
import torch.nn.functional as F
import dival
from dival import get_standard_dataset
import odl
from odl.contrib.torch import OperatorModule

from dival.reconstructors.networks.unet import get_unet_model
import utils as ut

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float

parser = argparse.ArgumentParser()
parser.add_argument('--visualize', type = bool, default = False,
                    help='Visualize the generated samples')
parser.add_argument('--save', type = bool, default = True,
                    help='Save images of particles during training')
parser.add_argument('--lowdose', type = bool, default = False,
                    help='Decide between limited angle CT and low-dose CT. You run limited angle CT per default, for lowdose set lowdose=True')
args = parser.parse_args()

def gen_i():
    test = dataset.create_torch_dataset(part='test',
                        reshape=((1,1) + dataset.space[0].shape,
                        (1,1) + dataset.space[1].shape))
    N = 100
    idx = torch.randint(0,3553,size=[])
    gt = test[idx][1].to(device)
    observation = fbp(test[idx][0]).to(device)
    
    x_new = torch.rand((N,channel,img_size,img_size),dtype=dtype,device=device)
    obs = observation.tile(N,1,1,1)
    x_new = torch.cat([x_new,obs],dim=1)
    with torch.no_grad():
        for n in tqdm(range(len(os.listdir(f'{img_path}/nets')))):
            new_net = get_unet_model(in_ch=2*channel,out_ch=channel,
                            use_sigmoid=False,use_norm=False).to(device)
            new_net.load_state_dict(torch.load(f'{img_path}/nets/net{n}.pt'))
            x_l = []
            for i in range(N//batch_size):
                x_tmp = x_new[i*batch_size:(i+1)*batch_size,...]
                x_tmp_z = x_tmp[:,:channel]
                x_l.append(x_tmp_z-new_net(x_tmp).detach())
            x_new = torch.cat(x_l,0)
            x_new = torch.cat([x_new,obs],dim=1)
    imgs = x_new[:,:channel]
    mean = torch.mean(imgs,dim=0,keepdim=True)
    std = torch.std(imgs,dim=0,keepdim=True)
    idx_2 = torch.randperm(imgs.shape[0])[:3]
    ut.save_image(torch.cat([gt,observation,imgs[idx_2],mean,std/std.max()],dim=0),f'{img_path}/CT_reconstruction.png')
    exit()

if __name__ == '__main__':
    if args.lowdose:
        img_path = 'lowdoseCT'
    else:
        img_path = 'limangleCT'
    if not os.path.isdir(f'{img_path}'):
        os.mkdir(f'{img_path}')  
    if not os.path.isdir(f'{img_path}/nets'):
        os.mkdir(f'{img_path}/nets')  
        
    #experiment configurations
    img_size = 362
    channel = 1
    d = channel*img_size**2
    obs_dim = d
    m = 500 #number samples
    step = 0
    step_exp = 4
    opt_steps = 2**step_exp
    
    new_net = get_unet_model(in_ch=2*channel,out_ch=channel,use_sigmoid=False,use_norm=False).to(device)
    train_steps = 4000
    batch_size = 10
    final_step = 40 #maximal number of networks
    net_num = 0

    s_factor = ut.sliced_factor(d+obs_dim)
    n_projections = 400
    proj_batches = 1

    p_size = 15 #local size for projections
    p_dim = channel * p_size**2
    cut_p = []
    step_sizes=[]
    for i in range(21):
        cut_p.append(ut.cut_patches_periodic_padding(img_size,img_size,channel,(i+1)*p_size))
        if i < 6:
            step_sizes.append(.1)
        elif i < 10:
            step_sizes.append(.2)
        elif i < 20:
            step_sizes.append(.5)
        else:
            step_sizes.append(1)
    scales = [0,1,2,3,4,5,6,7,8,9,10,20]
    
    #create LoDoPaB dataset
    dataset = get_standard_dataset('lodopab', impl='astra_cuda')                 
    if not args.lowdose:
        dataset = dival.datasets.angle_subset_dataset.AngleSubsetDataset(dataset,
                    slice(100,900),impl='astra_cuda')  
        
    ray_trafo = dataset.ray_trafo                    
    operator = OperatorModule(ray_trafo).to(device)
    fbp = odl.tomo.analytic.filtered_back_projection.fbp_op(ray_trafo,
	                    filter_type = 'Hann', frequency_scaling = 0.641025641025641)
    fbp = OperatorModule(fbp).to(device)
    
    if args.visualize:
        gen_i()
    
    #create data as tensor
    train = dataset.create_torch_dataset(part='train',
                        reshape=((1,) + dataset.space[0].shape,
                        (1,) + dataset.space[1].shape))
    data = DataLoader(dataset=train,batch_size=m,shuffle=False)
    data = next(iter(data))
    ground_truth = data[1].to(device)
    ground_truth = ground_truth.view(m,d)
    observation = fbp(data[0]).to(device)
    observation = observation.view(m,obs_dim)
    target = torch.cat([ground_truth,observation],dim=1)
    
    particles = torch.rand((int(m),d),dtype=dtype,device=device)
    particles = torch.cat([particles,observation],dim=1)
    for i in tqdm(range(final_step)):
        p_old = torch.clone(particles)
        for _ in tqdm(range(opt_steps)):
            MMD_grads = torch.zeros(m*(d),device=device)
            for s in scales:
                MMD_grad = 0.
                for _ in range(proj_batches):
                    #locally-connected projections
                    xi_x = torch.randn((n_projections,p_dim),dtype=dtype,device=device)
                    xi_x = xi_x.reshape(-1,channel,p_size,1,p_size,1).tile(1,1,(s+1),1,(s+1)).reshape(-1,channel*((s+1)*p_size)**2)
                    xi_y = torch.randn((n_projections,p_dim),dtype=dtype,device=device)
                    xi_y = xi_y.reshape(-1,channel,p_size,1,p_size,1).tile(1,1,(s+1),1,(s+1)).reshape(-1,channel*((s+1)*p_size)**2)
                    xi = torch.cat([xi_x,xi_y],dim=1)
                    xi = xi/torch.sqrt(torch.sum(xi**2,-1,keepdim=True))
                    xi = xi.unsqueeze(1)
                    
                    #cut patches
                    position_inds_height=torch.randint(0,img_size,(1,),device=device)
                    position_inds_width=torch.randint(0,img_size,(1,),device=device)
                    patches_x,linear_inds=cut_p[s](particles[:,:d].reshape(-1,channel,img_size,img_size),
                                                position_inds_height,position_inds_width)
                    patches_y,linear_inds=cut_p[s](target[:,:d].reshape(-1,channel,img_size,img_size),
                                                position_inds_height,position_inds_width)
                    patches_x_obs,_=cut_p[s](particles[:,d:].reshape(-1,channel,img_size,img_size),
                                                position_inds_height,position_inds_width)
                    patches_y_obs,_=cut_p[s](target[:,d:].reshape(-1,channel,img_size,img_size),
                                                position_inds_height,position_inds_width)
                    x_tmp = torch.cat([patches_x.reshape(m,-1),patches_x_obs.reshape(m,-1)],dim=1)
                    y_tmp = torch.cat([patches_y.reshape(m,-1),patches_y_obs.reshape(m,-1)],dim=1)

                    #slice particles
                    x_proj = F.conv1d(x_tmp.reshape(1,1,-1),xi,
                                stride=channel*2*((s+1)*p_size)**2).reshape(n_projections,-1)
                    y_proj = F.conv1d(y_tmp.reshape(1,1,-1),xi,
                                stride=channel*2*((s+1)*p_size)**2).reshape(n_projections,-1)
                    
                    #compute 1D gradient of MMD
                    grad = ut.MMD_derivative_1d(x_proj,y_proj)
                    grad = grad.transpose(0,1)
                    
                    #compute MMD gradient based on 1D gradient
                    xi = xi.reshape([n_projections,channel*2*((s+1)*p_size)**2]).transpose(0,1).flatten()
                    MMD_grad += s_factor* F.conv1d(xi.reshape([1,1,-1]), grad.unsqueeze(1),
                                stride=n_projections).squeeze()/(n_projections*proj_batches)
                MMD_grads[linear_inds] += step_sizes[s]*MMD_grad[:,:channel*((s+1)*p_size)**2].reshape(-1)
            MMD_grads = MMD_grads.reshape(m,-1)    
            
            #update the flow
            particles[:,:d] -= m*MMD_grads
            step += 1
        
        #train network
        many_grad=(p_old-particles)
        optim = torch.optim.Adam(new_net.parameters(), lr=0.0005)
        for ts in range(train_steps):
            perm = torch.randperm(many_grad.shape[0])[:batch_size]
            y_in = many_grad[perm]
            x_in = p_old[perm]
            x_in_z = x_in[:,:d].reshape(-1,channel,img_size,img_size)
            x_in_y = x_in[:,d:].reshape(-1,channel,img_size,img_size)
            x_in = torch.cat([x_in_z,x_in_y],dim=1)
            loss = torch.sum((new_net(x_in).reshape(-1,d)-y_in[:,:d])**2)/batch_size
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(new_net.parameters(), 1)
            optim.step()
        del many_grad
        torch.cuda.empty_cache()
        torch.save(new_net.state_dict(),f'{img_path}/nets/net{net_num}.pt')
        net_num += 1
        
        #update particles
        with torch.no_grad():
            x_new = []
            i = 0
            while i<m:
                x_in = p_old[i:i+batch_size]
                x_in_z = x_in[:,:d].reshape(-1,channel,img_size,img_size)
                x_in_y = x_in[:,d:].reshape(-1,channel,img_size,img_size)
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
            ut.save_image(particles[:16,:d].reshape(-1,channel,img_size,img_size),f'{img_path}/flow{step}.png',4)
        
