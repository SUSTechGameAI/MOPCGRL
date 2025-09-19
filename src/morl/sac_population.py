import os
import time
from torch import tensor
import torch
import numpy as np
import torch.nn.functional as F

import sys
import pickle
import importlib
import copy
import random

from pymoo.util.ref_dirs import get_reference_directions

# absolute path to src
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(src_path)

from src.algo.sac import SAC_Model, OffRewSAC_Trainer, plot_levels
from src.environment.reward_function import *
from src.environment.env import make_vec_offrew_env
from src.designer.train_designer import load_sac_model
from src.algo.replay_memory import ReplayMem
from src.gan.gan_use import *
from src.repairer.repairer import DivideConquerRepairer, Repairer

from smb import *

class SAC_Population:
    def __init__(self, pop_size, moea_opt, mutation_std, mutation_mu, constr_p=23, partial_iter=5, d_eval_size=3, device='cuda:0'):
        self.pop_size = pop_size
        # optimizer function
        self.moea_opt = moea_opt
        self.mutation_std = mutation_std
        self.mutation_mu = mutation_mu
        self.device = device
        self.off = []
        self.pretrain_iter = 1000
        self.rep_mem = None
        self.d_eval_size = d_eval_size
        self.nz = 20
        self.d_eval = np.random.rand(self.d_eval_size, 4, self.nz).astype(np.float32) * 2 - 1
        self.rfunc = RewardFunc(FunContent(), FunBehaviour(), Playability())
        self.generator = get_generator(device=self.device)

        self.repairer = DivideConquerRepairer()
        self.mario_proxy = MarioProxy() if self.rfunc.require_simlt else None

        self.F = None
        self.G = None
        self.constr_p = constr_p
        self.CV = None
        
        self.da = []
        self.ca = []
        self.d_trainer = None
        self.partial_iter = partial_iter

    # network crossover
    def crossover(self, p1, p2):
        # Save current optimizers
        actor_optimizer_p1 = p1.netA_optimizer
        actor_optimizer_p2 = p2.netA_optimizer

        for param1, param2 in zip(p1.netA.parameters(), p2.netA.parameters()):
            W1 = param1.data
            W2 = param2.data

            if len(W1.shape) == 2:
                num_variables = W1.shape[0]
                num_cross_overs = random.randrange(num_variables*2)
                for i in range(num_cross_overs):
                    receiver_choice = random.random()
                    if receiver_choice < 0.5:
                        ind_cr = random.randrange(W1.shape[0])
                        W1[ind_cr, :] = W2[ind_cr, :]
                    else:
                        ind_cr = random.randrange(W1.shape[0])
                        W2[ind_cr, :] = W1[ind_cr, :]

            elif len(W1.shape) == 1:
                num_variables = W1.shape[0]
                num_cross_overs = random.randrange(num_variables)
                for i in range(num_cross_overs):
                    receiver_choice = random.random()
                    if receiver_choice < 0.5:
                        ind_cr = random.randrange(W1.shape[0])
                        W1[ind_cr] = W2[ind_cr]
                    else:
                        ind_cr = random.randrange(W1.shape[0])
                        W2[ind_cr] = W1[ind_cr]
            # update parameters
            param1.data = W1
            param2.data = W2
        # Reinitialize optimizers
        p1.netA_optimizer = torch.optim.Adam(p1.netA.parameters(), 3e-4)
        p1.netA_optimizer.load_state_dict(actor_optimizer_p1.state_dict())

        p2.netA_optimizer = torch.optim.Adam(p2.netA.parameters(), 3e-4)
        p2.netA_optimizer.load_state_dict(actor_optimizer_p2.state_dict())

    # mutation part
    def mutation(self):  
        for sac_model in self.off[:,0]:
            with torch.no_grad():
                # pass actor
                actor_optimizer = sac_model.netA_optimizer
                for name, param in sac_model.netA.named_parameters():
                    weighs = param.detach()
                    if 'bias' not in name:
                        # add noise
                        weighs += torch.normal(mean=self.mutation_mu, std=self.mutation_std, size=param.shape).to(sac_model.device)
                    # update parameter
                    sac_model.netA.state_dict()[name].data.copy_(torch.Tensor(weighs))
                # initialize optimizer
                sac_model.netA_optimizer = torch.optim.Adam(sac_model.netA.parameters(), 3e-4)
                sac_model.netA_optimizer.load_state_dict(actor_optimizer.state_dict())

    # update models in self.off using replay memory
    def model_update(self, batch_size, models, cfgs, env):
        if self.d_trainer is None:
                self.d_trainer = OffRewSAC_Trainer(
                    env, cfgs.total_steps, cfgs.update_freq, cfgs.batch_size, ReplayMem(int(cfgs.mem_size/5)),
                    cfgs.res_path, cfgs.check_points
                )
        if self.rep_mem is None:
            self.rep_mem = ReplayMem(cfgs.mem_size)
        for model in models:
            self.rep_mem = self.d_trainer.train_model(model, self.partial_iter, self.rep_mem)

    # evaluate model performance
    def model_eval(self, model, eplen, play_style):
        eval_fc_matrix = []
        eval_fb_matrix = []
        eval_p_matrix = []
        eval_fc = []
        eval_fb = []
        eval_p = []
        # d_eval_size is the number of generated levels by one model
        # print("#################### Evaluation ####################")
        for d in range(self.d_eval_size):
            if np.mod(d, 10) == 0:
                print(f"      level {d} is done")
            start_gen_time = time.time()

            d_eval = tensor(self.d_eval[d]).view(-1, self.nz, 1, 1).to(self.device)
            st_seg = copy.deepcopy(d_eval)
            # plot start segments
            obs_level = process_levels(self.generator(st_seg), True)
            level_segs = copy.deepcopy(st_seg)

            for i in range(eplen):
                # generate a new segment
                obs_tensor = st_seg.reshape(1, -1)
                gen_seg = model.make_decision(obs_tensor)
                # turn the generated segment to tensor
                next_seg = tensor(gen_seg).view(-1, self.nz, 1, 1).to(self.device)
                # concatenate new segment to level 
                concat_seg = torch.cat((st_seg.clone().detach().view(-1, self.nz, 1, 1).to(self.device), next_seg), dim=0)
                concat_re = concat_seg.reshape(1, -1)
                level_segs = torch.cat((level_segs, next_seg), dim=0)
                # get new obs
                st_seg = concat_re[:, -80:]
            level = process_levels(self.generator(level_segs), True)

            gen_time = time.time() - start_gen_time
            print(f"Level {d} generation time: {gen_time:.2f} seconds")

            # level simulation
            full_level = level_sum(level)
            
            use_repair = False
            
            if use_repair:
            
                start_repair_time = time.time()
                
                full_level = self.repairer.repair(full_level)
                repair_time = time.time() - start_repair_time
            else:
                full_level = full_level
            
            w = MarioLevel.default_seg_width
            segs = [full_level[:, s: s + w] for s in range(0, full_level.w, w)]
            jagent = MarioJavaAgents.__getitem__(play_style)
            simlt_k = 4. if play_style == 'Runner' else 10.

            start_sim_time = time.time()

            if self.mario_proxy:
                raw_simlt_res = self.mario_proxy.simulate_long(level_sum(segs), jagent, simlt_k)
                simlt_res = MarioProxy.get_seg_infos(raw_simlt_res)
            else:
                simlt_res = None

            sim_time = time.time() - start_sim_time
            print(f"Level {d} simulation time: {sim_time:.2f} seconds")

            # show results
            rewards = self.rfunc.get_rewards(segs=segs, simlt_res=simlt_res)
            info = {}
            total_score = 0
            info['LevelStr'] = str(full_level)
            for key in rewards:
                info[f'{key}_reward_list'] = rewards[key][-eplen:]
                info[f'{key}'] = sum(rewards[key][-eplen:])
                total_score += info[f'{key}']

            fc_matrix = info['FunContent_reward_list']
            fb_matrix = info['FunBehaviour_reward_list']
            p_matrix = info['Playability_reward_list']
            eval_fc_matrix.append(fc_matrix)
            eval_fb_matrix.append(fb_matrix)
            eval_p_matrix.append(p_matrix)

            eval_fc.append(np.mean(fc_matrix))
            eval_fb.append(np.mean(fb_matrix))
            eval_p.append(np.sum(p_matrix))

        F_eval_fc = np.mean(np.array(eval_fc))
        F_eval_fb = np.mean(np.array(eval_fb))
        F = [1-F_eval_fc, 1-F_eval_fb]
        G = eplen + np.mean(np.array(eval_p))
        return F, G

    # save model
    def save_morl_model(self, save_path, model, pop_i, iter, list_seed):
        
        torch.save({
            'iter_id': iter,
            'pop_index': pop_i,
            'seed': list_seed,
            'model_state_dict': model.log_alpha,
            'optimizer_state_dict': model.alpha_optimizer.state_dict()
        }, save_path+f'/log_alpha_iter{iter}_i{pop_i}_s{list_seed}.pth')
        
        torch.save({
            'iter_id': iter,
            'pop_index': pop_i,
            'seed': list_seed,
            'model_state_dict': model.netA.state_dict(),
            'optimizer_state_dict': model.netA_optimizer.state_dict()
        }, save_path+f'/netA_iter{iter}_i{pop_i}_s{list_seed}.pth')
        
        torch.save({
            'iter_id': iter,
            'pop_index': pop_i,
            'seed': list_seed,
            'model_state_dict': model.netQ1.state_dict(),
            'optimizer_state_dict': model.netQ1_optimizer.state_dict()
        }, save_path+f'/netQ1_iter{iter}_i{pop_i}_s{list_seed}.pth')
        
        torch.save({    
            'iter_id': iter,
            'pop_index': pop_i,
            'seed': list_seed,
            'model_state_dict': model.netQ2.state_dict(),
            'optimizer_state_dict': model.netQ2_optimizer.state_dict()
        }, save_path+f'/netQ2_iter{iter}_i{pop_i}_s{list_seed}.pth')
        
        torch.save({
            'iter_id': iter,
            'pop_index': pop_i,
            'seed': list_seed,
            'model_state_dict': model.tar_netQ1.state_dict(),
        }, save_path+f'/tar_netQ1_iter{iter}_i{pop_i}_s{list_seed}.pth')
        
        torch.save({
            'iter_id': iter,
            'pop_index': pop_i,
            'seed': list_seed,
            'model_state_dict': model.tar_netQ2.state_dict(),
        }, save_path+f'/tar_netQ2_iter{iter}_i{pop_i}_s{list_seed}.pth')
    
    # save replay memory
    def save_rep_mem(self, save_path, iter, list_seed):
        with open(save_path+f"/rep_mem_iter{iter}_s{list_seed}.pkl", 'wb') as f:
            pickle.dump(self.rep_mem, f)
    
    def delete_rep_mem(self, save_path, iter, list_seed):
        rm_path = os.path.join(save_path, f"rep_mem_iter{iter}_s{list_seed}.pkl")
        if os.path.exists(rm_path):
            os.remove(rm_path)
            print(f"Deleted {rm_path}")
        else:
            print(f"The file {rm_path} does not exist.")

    # load replay memory
    def load_rep_mem(self, save_path, iter, list_seed):
        with open(save_path+f"/rep_mem_iter{iter}_s{list_seed}.pkl", 'rb') as f:
            self.rep_mem = pickle.load(f)

    # load population randomly from pretrained models
    def load_list(self, list_seed):
        random.seed(list_seed)
        model_list = set()  
        run_range = range(1, 6)  
        iter_range = range(200000, 300001, 10000) 

        while len(model_list) < self.pop_size:
            run_id = random.choice(run_range)
            iter_id = random.choice(iter_range)
            model_list.add((run_id, iter_id))  
        return list(model_list)
                   
# load trained pop
def load_morl_model(save_path, model, iter, pop_i, list_seed):
    # load log_alpha
    log_alpha_state = torch.load(save_path+f'/log_alpha_iter{iter}_i{pop_i}_s{list_seed}.pth')
    model.log_alpha = log_alpha_state['model_state_dict']
    model.alpha_optimizer.load_state_dict(log_alpha_state['optimizer_state_dict'])
    
    # load netA
    netA_state = torch.load(save_path+f'/netA_iter{iter}_i{pop_i}_s{list_seed}.pth')
    model.netA.load_state_dict(netA_state['model_state_dict'])
    model.netA_optimizer.load_state_dict(netA_state['optimizer_state_dict'])
    
    # load netQ1
    netQ1_state = torch.load(save_path+f'/netQ1_iter{iter}_i{pop_i}_s{list_seed}.pth')
    model.netQ1.load_state_dict(netQ1_state['model_state_dict'])
    model.netQ1_optimizer.load_state_dict(netQ1_state['optimizer_state_dict'])
    
    # load netQ2
    netQ2_state = torch.load(save_path+f'/netQ2_iter{iter}_i{pop_i}_s{list_seed}.pth')
    model.netQ2.load_state_dict(netQ2_state['model_state_dict'])
    model.netQ2_optimizer.load_state_dict(netQ2_state['optimizer_state_dict'])
    
    # load tar_netQ1
    tar_netQ1_state = torch.load(save_path+f'/tar_netQ1_iter{iter}_i{pop_i}_s{list_seed}.pth')
    model.tar_netQ1.load_state_dict(tar_netQ1_state['model_state_dict'])
    
    # load tar_netQ2
    tar_netQ2_state = torch.load(save_path+f'/tar_netQ2_iter{iter}_i{pop_i}_s{list_seed}.pth')
    model.tar_netQ2.load_state_dict(tar_netQ2_state['model_state_dict'])

    return model





    


    
    
