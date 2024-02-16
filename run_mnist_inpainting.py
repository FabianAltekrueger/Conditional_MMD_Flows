#This code belongs to the paper
#P. Hagemann, J. Hertrich, F. Altekrüger, R. Beinert, J. Chemseddine, G. Steidl
#Posterior Sampling Based on Gradient Flows of the MMD with Negative Distance Kernel
#International Conference on Learning Representations.
#
#It reproduces the MNIST inpainting example from Section 4.

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
    mnist = td.MNIST('mnist',train=False,transform=transforms.ToTensor(),download=True)
    numb_im = 4
    N=25
    batch_size = N
    idx = torch.randint(0,10000,(numb_im,))
    obs = torch.tensor([],device=device)
    for i in idx:
        tmp = mnist[i][0].view(1,d).to(device)
        obs = torch.cat([obs,tmp],dim=0)
    
    gt = obs.reshape(-1,channel,img_size,img_size)
    ut.save_image(obs.reshape(-1,channel,img_size,img_size),f'{img_path}/{dataset}_gt.png',rows=1)
    o = obs.reshape(-1,channel,img_size,img_size)[...,-pixel_y:,:].clone()
    obs=torch.cat([0.6*torch.ones([o.shape[0],channel,img_size-pixel_y,img_size],device=device,dtype=dtype),o],dim=2)
    ut.save_image(obs,f'{img_path}/{dataset}_obs.png',rows=1)
    o = o.reshape(-1,obs_dim)
    recos = torch.tensor([],device=device)
    stds = torch.tensor([],device=device)
    for i in range(numb_im):
        x_new = torch.rand((int(N),d),dtype=dtype,device=device)
        observation = o[i].reshape(1,-1)
        obs = observation.tile(N,1)
        x_new = torch.cat([x_new,obs],dim=1)
        x_old = x_new.clone()
        for n in tqdm(range(len(os.listdir(f'{img_path}/nets')))):
            new_net=get_UNET()
            new_net.load_state_dict(torch.load(f'{img_path}/nets/net{n}.pt'))
            x_l = []
            save_counter = 0
            for i in range(N//batch_size):
                x_tmp = x_new[i*batch_size:(i+1)*batch_size,...]
                x_tmp_z=x_tmp[:,:d].clone().reshape(-1,channel,img_size,img_size)
                x_tmp_y=x_tmp[:,d:].reshape(-1,channel,pixel_y,img_size)
                x_tmp_y=torch.cat([torch.zeros([x_tmp_y.shape[0],channel,img_size-pixel_y,img_size],device=device,dtype=dtype),x_tmp_y],dim=2)
                x_tmp = torch.cat([x_tmp_z,x_tmp_y],dim=1)
                x_l.append(x_tmp_z-new_net(x_tmp).detach())
            x_new = torch.cat(x_l,0).reshape(-1,d)
            x_new = torch.cat([x_new,obs],dim=1)
        recos = torch.cat([recos,x_new[:,:d].reshape(-1,channel,img_size,img_size)],dim=0)
        std_tmp = torch.std(x_new[:,:d].reshape(-1,channel,img_size,img_size),dim=0,keepdim=True)
        std_tmp = std_tmp/std_tmp.max()
        stds = torch.cat([stds,std_tmp],dim=0)

    ut.save_image(stds,f'{img_path}/{dataset}_std.png',rows=1)
    ut.save_image(recos,f'{img_path}/{dataset}_rec.png',rows=N)
    a=ut.imread(f'{img_path}/{dataset}_obs.png')
    b=ut.imread(f'{img_path}/{dataset}_rec.png')
    c=ut.imread(f'{img_path}/{dataset}_gt.png')
    e=ut.imread(f'{img_path}/{dataset}_std.png')
    white = torch.ones(1,3,c.shape[2],4,device='cuda')
    tmp = torch.cat([a,white,b,white,e,white,c],dim=3)
    os.remove(f'{img_path}/{dataset}_obs.png')
    os.remove(f'{img_path}/{dataset}_rec.png')
    os.remove(f'{img_path}/{dataset}_gt.png')
    os.remove(f'{img_path}/{dataset}_std.png')
    io.imsave(f'{img_path}/{dataset}_inpainting.png',tmp.squeeze().permute(1,2,0).detach().cpu().numpy())
    exit()

def get_UNET():
    return UNet(
        input_channels=2*channel,
        output_channels=channel,
        input_height=img_size,
        ch=32,
        ch_mult=(1, 2, 4),
        num_res_blocks=2,
        attn_resolutions=(256,),
        resamp_with_conv=True,).to(device) 

if __name__ == '__main__':
    dataset = 'MNIST'
    img_path = 'MNIST_inpainting'
    if not os.path.isdir(f'{img_path}'):
        os.mkdir(f'{img_path}')  
    if not os.path.isdir(f'{img_path}/nets'):
        os.mkdir(f'{img_path}/nets')  
    
    #experiment configurations
    channel = 1
    img_size = 28
    d = channel*img_size**2
    pixel_y = 10
    obs_dim = channel*img_size*pixel_y
    m = 20000 #number samples
    momentum = 0.
    s_factor = ut.sliced_factor(d+obs_dim)
    n_projections = 500
    proj_batches = 1
    final_step = 90 #maximal number of trained networks
    
    #steps of particle flow
    step_exp = 4
    opt_steps = 2**step_exp
    step = 0
    
    #network
    new_net = get_UNET()
    train_steps = 5000
    batch_size = 100
    net_num = 0
    
    p_size = 5 #local size for projections
    p_dim = channel * p_size**2
    cut_p = []
    for i in range(5):
        cut_p.append(ut.cut_patches_periodic_padding(img_size,img_size,channel,(i+1)*p_size))
    scales = [0,1,2,3]
    step_sizes = [.25,.5,.85,.85,.85]
    
    mnist = td.MNIST('mnist',transform=transforms.ToTensor(),download=True)
    data = DataLoader(dataset=mnist,batch_size=m)
    data = next(iter(data))
    data = data[0].view(m,channel*img_size**2).to(device)
    
    ground_truth = data.clone()
    observation = ground_truth.reshape(-1,channel,img_size,img_size)[...,-pixel_y:,:].clone()
    observation = observation.reshape(-1,obs_dim)
    
    #target pairs and initial particles
    y = torch.cat([ground_truth,observation],dim=1)
    particles = torch.cat([torch.rand(m,d,device=device),observation],dim=1)
    
    #if pretrained networks should be evaluated
    if args.visualize:
        gen_i()

    for i in tqdm(range(final_step)):
        p_old = torch.clone(particles)
        old_grad = torch.zeros_like(particles[:,:d])
        for _ in tqdm(range(opt_steps)):
            MMD_grads = torch.zeros(m*(d),device=device)
            for s in scales:
                MMD_grad = 0.
                for _ in range(proj_batches):
                    #locally-connected projections
                    xi_x = torch.randn((n_projections,p_dim),dtype=dtype,device=device)
                    xi_x = xi_x.reshape(-1,channel,p_size,1,p_size,1).tile(1,1,(s+1),1,(s+1)).reshape(-1,channel*((s+1)*p_size)**2)
                    xi_y = torch.randn((n_projections,obs_dim),dtype=dtype,device=device)
                    xi = torch.cat([xi_x,xi_y],dim=1)
                    xi = xi/torch.sqrt(torch.sum(xi**2,-1,keepdim=True)) 
                    xi = xi.unsqueeze(1)
                    
                    #extract patches
                    position_inds_height = torch.randint(0,img_size,(1,),device=device)
                    position_inds_width = torch.randint(0,img_size,(1,),device=device)
                    patches_x,linear_inds = cut_p[s](particles[:,:d].reshape(-1,channel,img_size,img_size),
                                                position_inds_height,position_inds_width)
                    patches_y,linear_inds = cut_p[s](y[:,:d].reshape(-1,channel,img_size,img_size),
                                                position_inds_height,position_inds_width)
                    x_tmp = torch.cat([patches_x.reshape(m,-1),particles[:,d:]],dim=1)
                    y_tmp = torch.cat([patches_y.reshape(m,-1),y[:,d:]],dim=1)
                    
                    #slice the flow
                    x_proj = F.conv1d(x_tmp.reshape(1,1,-1),xi,
                                stride=channel*((s+1)*p_size)**2+obs_dim).reshape(n_projections,-1)
                    y_proj = F.conv1d(y_tmp.reshape(1,1,-1),xi,
                                stride=channel*((s+1)*p_size)**2+obs_dim).reshape(n_projections,-1)
                    
                    #compute 1D gradient of MMD
                    grad = ut.MMD_derivative_1d(x_proj,y_proj)
                    grad = grad.transpose(0,1)

                    #compute MMD gradient based on 1D gradient
                    xi = xi.reshape([n_projections,channel*((s+1)*p_size)**2+obs_dim]).transpose(0,1).flatten()
                    MMD_grad = s_factor* F.conv1d(xi.reshape([1,1,-1]),grad.unsqueeze(1),
                                stride=n_projections).squeeze()/n_projections + MMD_grad
                    MMD_grad[:,-obs_dim:] = 0
                    
                MMD_grads[linear_inds] += step_sizes[s]*MMD_grad[:,:channel*((s+1)*p_size)**2].reshape(-1)
            MMD_grads = MMD_grads.reshape(m,-1)    
            MMD_grads = MMD_grads/proj_batches + momentum*old_grad
            
            #update particles
            particles[:,:d] -= m*MMD_grads
            old_grad = MMD_grads            
            step += 1
        
        #train network
        many_grad = (p_old-particles)
        optim = torch.optim.Adam(new_net.parameters(), lr=0.0005)
        for ts in range(train_steps):
            perm = torch.randperm(many_grad.shape[0])[:batch_size]
            y_in = many_grad[perm]
            x_in = p_old[perm]
            x_in_z = x_in[:,:d].reshape(-1,channel,img_size,img_size)
            x_in_y = x_in[:,d:].reshape(-1,channel,pixel_y,img_size)
            x_in_y = torch.cat([torch.zeros([x_in_y.shape[0],1,img_size-pixel_y,img_size],
                                device=device,dtype=dtype),x_in_y],dim=2)
            x_in = torch.cat([x_in_z,x_in_y],dim=1)
            loss = torch.sum((new_net(x_in).reshape(-1,d)-y_in[:,:d])**2)/batch_size
            optim.zero_grad()
            loss.backward()
            optim.step()
        torch.save(new_net.state_dict(),f'{img_path}/nets/net{net_num}.pt')
        net_num += 1
        
        #update particles
        with torch.no_grad():
            x_new = []
            i = 0
            while i<m:
                x_in = p_old[i:i+batch_size]
                x_in_z = x_in[:,:d].reshape(-1,channel,img_size,img_size)
                x_in_y = x_in[:,d:].reshape(-1,channel,pixel_y,img_size)
                x_in_y = torch.cat([torch.zeros([x_in_y.shape[0],1,img_size-pixel_y,img_size],
                                    device=device,dtype=dtype),x_in_y],dim=2)
                x_in = torch.cat([x_in_z,x_in_y],dim=1)
                x_new.append(x_in_z-new_net(x_in).detach())
                i += batch_size
            x_new = torch.cat(x_new,0).reshape(-1,d)
            x_new = torch.cat([x_new,observation],dim=1)
        particles = x_new.detach()    
        
        #update number of flow steps
        opt_plus = min(2**step_exp,1024)
        opt_steps = min(opt_steps+opt_plus,30000)
        step_exp += 1 
        
        if args.save:
            ut.save_image(particles[:100,:d].reshape(-1,channel,img_size,img_size),f'{img_path}/flow{step}.png',10)
        
    
