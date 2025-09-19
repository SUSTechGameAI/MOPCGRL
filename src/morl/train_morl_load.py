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

from c_taea import CADASurvival, RestrictedMating, merge, constr_to_cv
from sac_population import SAC_Population, load_morl_model
from pymoo.util.ref_dirs import get_reference_directions

# absolute path to src
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(src_path)

from src.algo.sac import SAC_Model, OffRewSAC_Trainer, plot_levels
from src.environment.reward_function import *
from src.environment.env import make_vec_offrew_env
from src.designer.train_designer import load_sac_model, load_sac_model_new, get_env
from src.gan.gan_use import *
from src.utils.filesys import auto_dire, create_directory

from smb import *

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     # torch.backends.cudnn.deterministic = True

if __name__=='__main__':
    
    seed = 1234
    setup_seed(seed)
    
    iterations = 100
    pop_size = 10
    moea_opt = 1
    mutation_std = 0.02
    mutation_mu = 0.0
    constr_p = 24
    partial_iter = 5000
    d_eval_size = 20
    device = 'cuda:0'
    load_iter = 18
    load_exp = 8

    # sac_population.pop = [] always empty, but F, CV are not empty and alined with sac_population.ca
    # sac_population.ca --> CA
    # sac_population.da --> DA
    # sac_population.off --> Offspring
    # abc = np.array([1,2, 3])
    # np.savetxt("abc1.txt", abc)
    cfgs = pickle.load(open("cfgs.pkl","rb"))
    cfgs.n_envs = 10
    # cfgs.res_path = 'exp_data'
    # model_path = '../MOPCGRL/model4pop'
    # print("preparing to create rfunc")
    env = get_env(cfgs)
    # np.savetxt("abc2.txt", abc)
    
    rfunc = (
        importlib.import_module('src.environment.rfuncs')
        .__getattribute__(f'{cfgs.rfunc_name}')
    )
    # np.savetxt("abc3.txt", abc)
    info_path = '../training_exp'
    new_dir = auto_dire(info_path, name='experiment')
    print(f"Directory '{new_dir}' created successfully")
    
    # Record the start time
    start_time = time.time()
    start_time_str = time.strftime('%Y-%m-%d %H:%M', time.localtime(start_time))

    params_info = f"""
    Start time: {start_time_str}
    
    # Parameters:
    seed = {seed}
    iterations = {iterations}
    pop_size = {pop_size}
    moea_opt = {moea_opt}
    mutation_std = {mutation_std}
    mutation_mu = {mutation_mu}
    constr_p = {constr_p}
    partial_iter = {partial_iter}
    d_eval_size = {d_eval_size}
    load_iter = {load_iter}
    load_exp = {load_exp}
    """
    with open(os.path.join(new_dir, 'exp_setting.txt'), 'w') as f:
        f.write(params_info)

    
    # print("preparing to create rfunc")
    # env = make_vec_offrew_env(
    #     cfgs.n_envs, rfunc, cfgs.res_path, cfgs.eplen, play_style=cfgs.play_style,
    #     log_itv=cfgs.n_envs * 2, device=cfgs.device, log_targets=['file', 'std']
    # )
    sac_population = SAC_Population(pop_size, moea_opt, mutation_std, mutation_mu, constr_p, partial_iter, d_eval_size, device)
    sac_population.ca = np.empty((pop_size, 3), dtype=object)
    sac_population.da = np.empty((pop_size, 3), dtype=object)
    # np.savetxt("abc4.txt", abc)
    
    # pop_list = sac_population.load_list(list_seed)
    pop_list = [(i, 200000) for i in range(101, 111)]
    print(f"initial population: {pop_list}")
    with open(os.path.join(new_dir, 'exp_setting.txt'), 'a') as f:
        f.write(f"\nPopulation list: {pop_list}")

    
    saved_ca_path = info_path+f'/experiment{load_exp}/iteration{load_iter}/morl_pop/ca'
    saved_da_path = info_path+f'/experiment{load_exp}/iteration{load_iter}/morl_pop/da'
    ca_data = np.loadtxt(info_path+f'/experiment{load_exp}/iteration{load_iter}/iter_pop/ca/gen{load_iter}_test{seed}.txt')
    da_data = np.loadtxt(info_path+f'/experiment{load_exp}/iteration{load_iter}/iter_pop/da/gen{load_iter}_test{seed}.txt')
    for i in range(pop_size):
        print("process individual", i)
        sac_model = load_sac_model_new(cfgs, env)
        sac_population.ca[i][0] = load_morl_model(saved_ca_path, sac_model, load_iter, i, seed)
        sac_population.ca[i][1] = np.array([ca_data[i][0], ca_data[i][1]])
        sac_population.ca[i][2] = np.array([ca_data[i][2]])
        sac_population.da[i][0] = load_morl_model(saved_da_path, sac_model, load_iter, i, seed)
        sac_population.da[i][1] = np.array([da_data[i][0], da_data[i][1]])
        sac_population.da[i][2] = np.array([da_data[i][2]])
    
    rep_mem_path = info_path+f'/experiment{load_exp}/iteration{load_iter}/morl_pop'
    sac_population.load_rep_mem(rep_mem_path, load_iter, seed)
    # rfunc = (
    #     importlib.import_module('src.environment.rfuncs')
    #     .__getattribute__(f'{cfgs.rfunc_name}')
    # )
    
    # print("preparing to create rfunc")
    # env = make_vec_offrew_env(
    #     cfgs.n_envs, rfunc, cfgs.res_path, cfgs.eplen, play_style=cfgs.play_style,
    #     log_itv=cfgs.n_envs * 2, device=cfgs.device, log_targets=['file', 'std']
    # )

    print("preparing to algorithm")
    ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=pop_size-1)
    algorithm = CADASurvival(ref_dirs)
    # print("preparing to evolve")
    # train model set
    # sac_population.initialize(OffRewSAC_Trainer, env, cfgs)
    # F, G = [], []
    # for i in range(len(sac_population.ca)):
    #     print("******************individual eval", i)
    #     # np.savetxt(f"abcii{i}.txt", abc)
        
    #     F_model, G_model = sac_population.model_eval(sac_population.ca[i][0], cfgs.eplen, cfgs.play_style)
    #     F.append(F_model)
    #     G.append(G_model)
    # G_array = constr_to_cv(G, sac_population.constr_p)
    # for i in range(len(sac_population.ca)):
    #     sac_population.ca[i][1] = np.array(F[i])
    #     sac_population.ca[i][2] = np.array(G_array[i])
    # sac_population.F = np.array(F)
    # sac_population.CV = np.array(constr_to_cv(G, sac_population.constr_p))
    # print("F and CV of SAC Population:", sac_population.F, sac_population.CV)
    
    # sac_population.pop = np.array(sac_population.pop)
    # sac_population.ca, sac_population.da = algorithm.do(sac_population.ca, sac_population.da, sac_population.off, sac_population, n_survive=pop_size)
    print(f'ITERATION: {load_iter}')
    print("ca: ", np.hstack((np.vstack(sac_population.ca[:,1]), np.vstack(sac_population.ca[:,2]))))
    print("da: ", np.hstack((np.vstack(sac_population.da[:,1]), np.vstack(sac_population.da[:,2]))))
    
    for iter in range(load_iter+1, iterations):
        # Record the start time of every iteration
        iter_time = time.time()
        iter_time_str = time.strftime('%Y-%m-%d %H:%M', time.localtime(iter_time))

        # Append the end time and total runtime to the run.txt file
        with open(os.path.join(new_dir, 'exp_setting.txt'), 'a') as f:
            f.write(f"\nIteration{iter} loop start time: {iter_time_str}")
        # mating selection
        Hm = merge(sac_population.ca, sac_population.da)
        selection = RestrictedMating()
        n_parents = 2
        n_select = pop_size
        # mating parent p1(from CA) and p2(from DA)
        mating = selection._do(Hm, n_select=n_select, n_parents=n_parents) 
        print(mating)

        # crossover and mutation
        off = []
        sac_population.off = np.empty((pop_size, 3), dtype=object)
        for i in range(mating.shape[0]):
            sac_population.crossover(Hm[mating[i, 0], 0], Hm[mating[i, 1], 0])
            if np.random.random() < 0.5:
                off.append(Hm[mating[i, 1], 0])
            else:
                off.append(Hm[mating[i, 0], 0])
        sac_population.off[:, 0] = np.array(off)

        # add mutation process to actors
        sac_population.mutation()
        
        # update offspring
        sac_population.model_update(cfgs.batch_size, sac_population.off[:,0], cfgs, env)

        # evaluate performance of self.off
        F, G = [], []
        for i in range(len(sac_population.off)):
            print("******************eval", i)

            F_model, G_model = sac_population.model_eval(sac_population.off[i][0], cfgs.eplen, cfgs.play_style)
            F.append(F_model)
            G.append(G_model)
        G_array = constr_to_cv(G, sac_population.constr_p)
        for i in range(len(sac_population.off)):
            sac_population.off[i][1] = np.array(F[i])
            sac_population.off[i][2] = np.array(G_array[i])
        print(f"F and CV of off in {iter}:\n", np.hstack((np.vstack(sac_population.off[:, 1]), np.vstack(sac_population.off[:, 2]))))
        
        # environmantal selection, update CA and DA
        sac_population.ca, sac_population.da = algorithm.do(sac_population.ca, sac_population.da, sac_population.off, sac_population, n_survive=pop_size)
        
        ca_pop_info = np.hstack((np.vstack(sac_population.ca[:, 1]), np.vstack(sac_population.ca[:, 2])))
        da_pop_info = np.hstack((np.vstack(sac_population.da[:, 1]), np.vstack(sac_population.da[:, 2])))
        print(f"F and CV of ca in {iter}:\n", ca_pop_info)
        print(f"F and CV of da in {iter}:\n", da_pop_info)

        # save the population
        iter_dir = create_directory(new_dir+f'/iteration{iter}')  
        ca_dir = create_directory(iter_dir+'/morl_pop/ca') 
        da_dir = create_directory(iter_dir+'/morl_pop/da')
        ca_eval_dir = create_directory(iter_dir+'/iter_pop/ca') 
        da_eval_dir = create_directory(iter_dir+'/iter_pop/da') 
        np.savetxt(ca_eval_dir+f'/gen{iter}_test{seed}.txt', ca_pop_info)
        np.savetxt(da_eval_dir+f'/gen{iter}_test{seed}.txt', da_pop_info)
        for ca_i in range(len(sac_population.ca)):
            sac_population.save_morl_model(ca_dir, sac_population.ca[ca_i][0], ca_i, iter, seed)
        for da_i in range(len(sac_population.da)):
            sac_population.save_morl_model(da_dir, sac_population.da[da_i][0], da_i, iter, seed)
        # save rep_mem of the new iteration
        sac_population.save_rep_mem(iter_dir+'/morl_pop', iter, seed)
        # delete old rep_mem
        if iter >= 1:
            iter_dir_old = new_dir+f'/iteration{iter-1}'+'/morl_pop'
            sac_population.delete_rep_mem(iter_dir_old, iter-1, seed)

    # Record the end time
    end_time = time.time()
    end_time_str = time.strftime('%Y-%m-%d %H:%M', time.localtime(end_time))
    
    # Append the end time and total runtime to the run.txt file
    with open(os.path.join(new_dir, 'exp_setting.txt'), 'a') as f:
        f.write(f"\nEnd time: {end_time_str}")