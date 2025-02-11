# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from models.classifier_guidance_model import ClassifierGuidanceModel
from utils.degredations import build_degredation_model
from utils.functions import postprocess
from .ddim import DDIM

import matplotlib.pyplot as plt
import numpy as np

class REDDIFF(DDIM):
    def __init__(self, model: ClassifierGuidanceModel, cfg: DictConfig):
        self.model = model
        self.diffusion = model.diffusion
        self.H = build_degredation_model(cfg)
        self.cfg = cfg
        self.awd = cfg.algo.awd
        self.cond_awd = cfg.algo.cond_awd
        self.grad_term_weight = cfg.algo.grad_term_weight
        self.obs_weight = cfg.algo.obs_weight
        self.eta = cfg.algo.eta
        self.lr = cfg.algo.lr
        self.denoise_term_weight = cfg.algo.denoise_term_weight
        self.sigma_x0 = cfg.algo.sigma_x0
        
        print('self.lr', self.lr)
        print('self.sigma_x0', self.sigma_x0)
        print('self.beta', self.diffusion.betas)
        print('self.alpha', self.diffusion.alphas)
        print('self.eta', self.eta)
        print('self.grad_term_weight', self.grad_term_weight)
        print('self.obs_weight', self.obs_weight)
        print('self.denoise_term_weight', self.denoise_term_weight)
        print('self.awd', self.awd)
        print('self.cond_awd', self.cond_awd)
        print('self.num_diffusion_timesteps', self.diffusion.num_diffusion_timesteps)

    def sample(self, x, y, ts, **kwargs):
        y_0 = kwargs["y_0"]
        sigma_y = self.cfg.algo.sigma_y
        n = x.size(0)
        H = self.H
    
        x = self.initialize(x, y, ts, y_0=y_0)
        ss = [-1] + list(ts[:-1])
        xt_s = [x.cpu()]
        x0_s = []
        
        mu_s = x.cpu()
        x0_pred_s = x.cpu()
        mu_fft_abs_s = torch.fft.fftshift(torch.abs(torch.fft.fft2(mu_s)))
        mu_fft_ang_s = torch.fft.fftshift(torch.angle(torch.fft.fft2(mu_s)))

        #optimizer
        dtype = torch.FloatTensor
        
        # mu is the mean of the variational distribution and needs to be optimized
        mu = torch.autograd.Variable(x, requires_grad=True)   #, device=device).type(dtype)
        
        optimizer = torch.optim.Adam([mu], lr=self.lr, betas=(0.9, 0.99), weight_decay=0.0)   #original: 0.999
        #optimizer = torch.optim.SGD([mu], lr=1e6, momentum=0.9)  #momentum=0.9

        for ti, si in zip(reversed(ts), reversed(ss)):
                
            t = torch.ones(n).to(x.device).long() * ti
            s = torch.ones(n).to(x.device).long() * si
            alpha_t = self.diffusion.alpha(t).view(-1, 1, 1, 1)
            alpha_s = self.diffusion.alpha(s).view(-1, 1, 1, 1)
            
            sigma_x0 = self.sigma_x0  #0.0001
            noise_x0 = torch.randn_like(mu)
            noise_xt = torch.randn_like(mu)

            # randomly initialized 
            x0_pred = mu + sigma_x0*noise_x0
            
            # conditional distribution q(x_t | y)
            xt = alpha_t.sqrt() * x0_pred + (1 - alpha_t).sqrt() * noise_xt
            
            #scale = 0.0
            # below is the same as DDIM
            c1 = ((1 - alpha_t / alpha_s) * (1 - alpha_s) / (1 - alpha_t)).sqrt() * self.eta
            c2 = ((1 - alpha_s) - c1 ** 2).sqrt()
            #xt = xt.clone().to('cuda').requires_grad_(True)
            if self.cond_awd:
                scale = alpha_s.sqrt() / (alpha_s.sqrt() - c2 * alpha_t.sqrt() / (1 - alpha_t).sqrt())
                scale = scale.view(-1)[0].item()
            else:
                scale = 1.0
            
            # model output from DDIM
            et, x0_hat = self.model(xt, y, t, scale=scale)   #et, x0_hat
            
            if not self.awd:
                et = (xt - x0_hat * alpha_t.sqrt()) / (1 - alpha_t).sqrt()
                
            # detach the score network output to avoid backpropagating through it
            # as for w(0)=0, we have analytical form of gradient of the regularization term
            et = et.detach()
            
            ############################################
            # Below is solving inverse problem by optimizing mu
            # note to get the analytic gradient, must set sigma=0 and w(0)=0
            # which makes the Gaussian degenrate to a point mass
            
            e_obs = y_0 - H.H(x0_pred)
            
            # MSE loss ||y-f(mu)||^2 * 1/2
            loss_obs = (e_obs**2).mean()/2
            
            # Regularization loss
            # grad = E_{t~U(0,1) and noise~N(0,1)} [lambda_t (score - noise)]
            # thus loss is lambda_t (network_out - noise)^T mu
            loss_noise = torch.mul((et - noise_xt).detach(), x0_pred).mean()
            
            snr_inv = (1-alpha_t[0]).sqrt()/alpha_t[0].sqrt()  #1d torch tensor
            
            if self.denoise_term_weight == "linear":
                snr_inv = snr_inv
            elif self.denoise_term_weight == "sqrt":
                snr_inv = torch.sqrt(snr_inv)
            elif self.denoise_term_weight == "square":
                snr_inv = torch.square(snr_inv)
            elif self.denoise_term_weight == "log":
                snr_inv = torch.log(snr_inv + 1.0)
            elif self.denoise_term_weight == "trunc_linear":
                snr_inv = torch.clip(snr_inv, max=1.0)
            elif self.denoise_term_weight == "power2over3":
                snr_inv = torch.pow(snr_inv, 2/3)
            elif self.denoise_term_weight == "const":
                snr_inv = torch.pow(snr_inv, 0.0)
            
            
            w_t = self.grad_term_weight*snr_inv   #0.25
            v_t = self.obs_weight

            loss = w_t*loss_noise + v_t*loss_obs
            
            #adam step
            optimizer.zero_grad()  #initialize
            loss.backward()
            optimizer.step()
            
            # #save for visualization
            if self.cfg.exp.save_evolution:
                if (ti/((self.cfg.exp.start_step - self.cfg.exp.end_step)//len(ts))) % (len(ts)//10) == 0:
                    mu_s = torch.cat((mu_s, mu.detach().cpu()), dim=3)
                    mu_fft_abs_s = torch.cat((mu_fft_abs_s, torch.fft.fftshift(torch.abs(torch.fft.fft2(mu.detach().cpu())))), dim=3)
                    mu_fft_ang_s = torch.cat((mu_fft_ang_s, torch.fft.fftshift(torch.angle(torch.fft.fft2(mu.detach().cpu())))), dim=3)
                    x0_pred_s = torch.cat((x0_pred_s, x0_pred.detach().cpu()), dim=3)
                
        if self.cfg.exp.save_evolution:
            return x0_pred, mu, mu_s, x0_pred_s, mu_fft_abs_s, mu_fft_ang_s
        else:
            return x0_pred, mu  

        
    def initialize(self, x, y, ts, **kwargs):
        deg = self.cfg.algo.deg
        
        y_0 = kwargs['y_0']
        print(f"y_0.shape: {y_0.shape}")
        # plt.imshow(y_0[0].permute(1, 2, 0).cpu().numpy())
        
        H = self.H
        # n = x.size(0)
        # ti = ts[-1]
        
        # plt.imshow(postprocess(y_0)[0].reshape(3, 256, 256).permute(1, 2, 0).cpu().numpy())
        # plt.savefig('y_0.png')
        
        # for consistency with other methods, we use H_pinv
        x_0 = H.H_pinv(y_0).view(*x.size()).detach()
        
        # x_0 = H.H(y_0).view(*x.size()).detach() # what happens if we use this? Seems fine.
        
        # why need to apply mask twice??
        # make x_0 a clone of y. Seems fine.
        # x_0 = y_0.clone().detach()
        # x_0 = x_0.view(*x.size())
        
        
        # now if just use y_0, things go wrong. Why?
        # x_0 = y_0.view(*x.size()).detach()
        
        # plt.imshow(postprocess(x_0)[0].permute(1, 2, 0).cpu().numpy())
        # plt.savefig('x_0.png')
        
        # print(f"Torch allclose gives: {torch.allclose(x_0, y_0.view(*x.size()))}")
        
        # t = torch.ones(n).to(x.device).long() * ti
        # alpha_t = self.diffusion.alpha(t).view(-1, 1, 1, 1)  #it is zero
        
        return x_0   #alpha_t.sqrt() * x_0 + (1 - alpha_t).sqrt() * torch.randn_like(x_0)    #x_0


    def plot_weight_den(self, ts, **kwargs):
    
        #ts.reverse()
        alpha = self.diffusion.alpha(torch.tensor(ts).cuda())
        
        snr_inv = (1-alpha).sqrt()/alpha.sqrt()  #1d torch tensor
        snr_inv = snr_inv.detach().cpu().numpy()
            
        # plot lines
        plt.plot(ts, snr_inv, label = "1/snr", linewidth=2)
        plt.plot(ts, np.sqrt(snr_inv), label = "sqrt(1/snr)", linewidth=2)
        #plt.plot(ts, np.power(snr_inv, 2/3), label = "(1/snr)^2/3")
        plt.plot(ts, np.square(snr_inv), label = "square(1/snr)", linewidth=2)
        plt.plot(ts, np.log(snr_inv+1), label = "log(1+1/snr)", linewidth=2)   #ln
        plt.plot(ts, np.clip(snr_inv, None, 1), label = "clip(1/snr,max=1)", linewidth=2)
        plt.plot(ts, np.power(snr_inv, 0.0), label = "const", linewidth=2)

        plt.legend()
        #plt.xscale('log')
        plt.yscale('log')
        plt.xlim(max(ts), min(ts))
        plt.xlabel("timestep", fontsize = 15)
        plt.ylabel("denoiser weight", fontsize = 15)
        
        plt.legend(fontsize = 13)
        plt.xticks(fontsize = 13)
        plt.yticks(fontsize = 13)

        plt.savefig('weight_type_vs_step.png')

        return 0







