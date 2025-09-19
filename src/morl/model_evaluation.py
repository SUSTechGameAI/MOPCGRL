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
import json

# absolute path to src
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(src_path)
from src.algo.sac import plot_levels
from src.designer.train_designer import load_sac_model, load_sac_model_new, get_env
from src.repairer.repairer import DivideConquerRepairer, Repairer
from src.gan.gan_use import *
from smb import *
from testing_segs import load_testing_segs

from sac_population import load_morl_model

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)

def pretrain_eval(run_id, iter_id, sac_model, play_style, eplen, rfunc, path):
    device = 'cuda:0'
    # constr_p = 22
    # d_eval_size = 100
    d_eval_group = 1
    nz = 20
    mario_proxy = MarioProxy() if rfunc.require_simlt else None
    repairer = DivideConquerRepairer()
    generator = get_generator(device=device)
    
    for g in range(d_eval_group):
        dict_list = []
        aver_list = []
        group_seed = 111 + g
        np.random.seed(group_seed)
        print(f"#################### Evaluation: Model{run_id}_{iter_id} Group{g+1}####################")
        # d_eval_total = np.random.rand(d_eval_size, 4, nz).astype(np.float32) * 2 - 1
        d_eval_total = load_testing_segs()
        eval_fc = []
        eval_fb = []
        eval_p = []
        for d in range(len(d_eval_total)):
            if np.mod(d, 10) == 0:
                print(f"      level {d} is done")
            # start_gen_time = time.time()

            d_eval = tensor(d_eval_total[d]).view(-1, nz, 1, 1).to(device)
            st_seg = copy.deepcopy(d_eval)
            # plot start segments
            obs_level = process_levels(generator(st_seg), True)
            level_segs = copy.deepcopy(st_seg)

            for i in range(eplen):
                # generate a new segment
                obs_tensor = st_seg.reshape(1, -1)
                gen_seg = sac_model.make_decision(obs_tensor)
                # turn the generated segment to tensor
                next_seg = tensor(gen_seg).view(-1, nz, 1, 1).to(device)
                # concatenate new segment to level 
                concat_seg = torch.cat((st_seg.clone().detach().view(-1, nz, 1, 1).to(device), next_seg), dim=0)
                concat_re = concat_seg.reshape(1, -1)
                level_segs = torch.cat((level_segs, next_seg), dim=0)
                # get new obs
                st_seg = concat_re[:, -80:]
            level = process_levels(generator(level_segs), True)
            # gen_time = time.time() - start_gen_time

            # level simulation
            full_level = level_sum(level)
            
            use_repair = False 
            if use_repair:
                full_level = repairer.repair(full_level)
            else:
                full_level = full_level
            
            w = MarioLevel.default_seg_width
            segs = [full_level[:, s: s + w] for s in range(0, full_level.w, w)]
            jagent = MarioJavaAgents.__getitem__(play_style)
            simlt_k = 4. if play_style == 'Runner' else 10.

            # start_sim_time = time.time()
            if mario_proxy:
                raw_simlt_res = mario_proxy.simulate_long(level_sum(segs), jagent, simlt_k)
                simlt_res = MarioProxy.get_seg_infos(raw_simlt_res)
            else:
                simlt_res = None
            # sim_time = time.time() - start_sim_time

            # show results
            rewards = rfunc.get_rewards(segs=segs, simlt_res=simlt_res)
            info = {}
            total_score = 0
            info['LevelStr'] = str(full_level)
            for key in rewards:
                info[f'{key}_reward_list'] = rewards[key][-eplen:]
                info[f'{key}'] = sum(rewards[key][-eplen:])
                total_score += info[f'{key}']
            
            # they are lists
            fc_matrix = info['FunContent_reward_list']
            fb_matrix = info['FunBehaviour_reward_list']
            p_matrix = info['Playability_reward_list']

            eval_fc.append(np.mean(fc_matrix))
            eval_fb.append(np.mean(fb_matrix))
            eval_p.append(np.sum(p_matrix))

            data_dict = {
                "Model ID": [run_id, iter_id],
                "D_eval": d_eval.reshape(1, -1).tolist(), 
                "Fun Content Rewards": fc_matrix,
                "Fun Behaviour Rewards": fb_matrix,
                "Playability Rewards": p_matrix
            }
            dict_list.append(data_dict)

        with open(path+f'/test_20w/model_r{run_id}_i{iter_id}_g{g+1}.json', 'w', encoding='utf-8') as f:
            json.dump(dict_list, f, ensure_ascii=False, indent=4)
                
        F_eval_fc = np.mean(np.array(eval_fc))
        F_eval_fb = np.mean(np.array(eval_fb))
        G = eplen + np.mean(np.array(eval_p))

        average_dict = {
            "Model ID": [run_id, iter_id],
            "Fun Content Average": F_eval_fc,
            "Fun Behaviour Average": F_eval_fb,
            "Playability Average": G
        }
        aver_list.append(average_dict)
        with open(path+f'/test_20w/model_aver_r{run_id}_i{iter_id}_g{g+1}.json', 'w', encoding='utf-8') as f:
            json.dump(aver_list, f, ensure_ascii=False, indent=4)


def morl_eval(path, sac_model, load_exp, load_iter, pop_ind, play_style, eplen, rfunc, device, archive):
    # constr_p = 22
    # d_eval_size = 100
    d_eval_group = 1
    nz = 20
    mario_proxy = MarioProxy() if rfunc.require_simlt else None
    repairer = DivideConquerRepairer()
    generator = get_generator(device=device)
    
    for g in range(d_eval_group):
        dict_list = []
        aver_list = []
        # group_seed = 111 + g
        # np.random.seed(group_seed)
        print(f"#################### Evaluation: Experiment{load_exp}_Model{pop_ind}_Iteration{load_iter}####################")
        # d_eval_total = np.random.rand(d_eval_size, 4, nz).astype(np.float32) * 2 - 1
        d_eval_total = load_testing_segs()
        eval_fc = []
        eval_fb = []
        eval_p = []
        for d in range(len(d_eval_total)):
            if np.mod(d, 10) == 0:
                print(f"      level {d} is done")
            # start_gen_time = time.time()  
            d_eval = tensor(d_eval_total[d]).view(-1, nz, 1, 1).to(device)
            st_seg = copy.deepcopy(d_eval)
            # print(st_seg.shape)
            # plot start segments
            obs_level = process_levels(generator(st_seg), True)
            # plot_levels(obs_level, save_path="test_obs.png")
            # level = self.generator(st_seg)
            level_segs = copy.deepcopy(st_seg)

            for i in range(eplen):
                # generate a new segment
                obs_tensor = st_seg.reshape(1, -1)
                gen_seg = sac_model.make_decision(obs_tensor)
                # turn the generated segment to tensor
                next_seg = tensor(gen_seg).view(-1, nz, 1, 1).to(device)
                # concatenate new segment to level 
                concat_seg = torch.cat((st_seg.clone().detach().view(-1, nz, 1, 1).to(device), next_seg), dim=0)
                concat_re = concat_seg.reshape(1, -1)
                level_segs = torch.cat((level_segs, next_seg), dim=0)
                # get new obs
                st_seg = concat_re[:, -80:]
            level = process_levels(generator(level_segs), True)
            # plot_levels(level, path+f'/training_exp/experiment{load_exp}/test_level.png')

            # gen_time = time.time() - start_gen_time
            # print(f"Level {d} generating: {gen_time:.2f} seconds")

            # level simulation
            full_level = level_sum(level)
            
            use_repair = False 
            if use_repair:
            
                # start_repair_time = time.time()
                
                full_level = repairer.repair(full_level)
                # repair_time = time.time() - start_repair_time
                # print(f"Level {d} repairing: {repair_time:.2f} seconds")
                # plot_levels(full_level, save_path="test_level_repaired.png")
            else:
                full_level = full_level
            
            w = MarioLevel.default_seg_width
            segs = [full_level[:, s: s + w] for s in range(0, full_level.w, w)]
            # plot_levels(segs, path+f'/training_exp/experiment{load_exp}/test_segs.png')
            jagent = MarioJavaAgents.__getitem__(play_style)
            simlt_k = 4. if play_style == 'Runner' else 10.

            start_sim_time = time.time()

            if mario_proxy:
                raw_simlt_res = mario_proxy.simulate_long(level_sum(segs), jagent, simlt_k)
                simlt_res = MarioProxy.get_seg_infos(raw_simlt_res)
            else:
                simlt_res = None
            
            sim_time = time.time() - start_sim_time
            print(f"关卡 {d} 模拟时间: {sim_time:.2f} 秒")
            # trace_r = mario_proxy.simulate_long(full_level, MarioJavaAgents.Runner)['trace']
            # show results
            rewards = rfunc.get_rewards(segs=segs, simlt_res=simlt_res)
            info = {}
            total_score = 0
            info['LevelStr'] = str(full_level)
            for key in rewards:
                info[f'{key}_reward_list'] = rewards[key][-eplen:]
                info[f'{key}'] = sum(rewards[key][-eplen:])
                total_score += info[f'{key}']
            
            # they are lists
            fc_matrix = info['FunContent_reward_list']
            fb_matrix = info['FunBehaviour_reward_list']
            p_matrix = info['Playability_reward_list']

            eval_fc.append(np.mean(fc_matrix))
            eval_fb.append(np.mean(fb_matrix))
            eval_p.append(np.sum(p_matrix))

            data_dict = {
                "Model ID": [load_iter, pop_ind],
                "D_eval": d_eval.reshape(1, -1).tolist(), 
                "Fun Content Rewards": fc_matrix,
                "Fun Behaviour Rewards": fb_matrix,
                "Playability Rewards": p_matrix
            }
            dict_list.append(data_dict)

        with open(path+f'/experiment{load_exp}/testing_{archive}/model_iter{load_iter}_ind{pop_ind}.json', 'w', encoding='utf-8') as f:
            json.dump(dict_list, f, ensure_ascii=False, indent=4)
                
        F_eval_fc = np.mean(np.array(eval_fc))
        F_eval_fb = np.mean(np.array(eval_fb))
        G = eplen + np.mean(np.array(eval_p))

        average_dict = {
            "Model ID": [load_iter, pop_ind],
            "Fun Content Average": F_eval_fc,
            "Fun Behaviour Average": F_eval_fb,
            "Playability Average": G
        }
        aver_list.append(average_dict)
       
        with open(path+f'/experiment{load_exp}/testing_{archive}/model_aver_iter{load_iter}_ind{pop_ind}.json', 'w', encoding='utf-8') as f:
            json.dump(aver_list, f, ensure_ascii=False, indent=4)


if __name__=='__main__':
    setup_seed(1)

    pop_size = 10
    path = '../MOPCGRL'

    cfgs = pickle.load(open("cfgs.pkl","rb"))
    cfgs.n_envs = 10
    # cfgs.play_style = 'Collector'
    env = get_env(cfgs)
    
    rfunc = (
        importlib.import_module('src.environment.rfuncs')
        .__getattribute__(f'{cfgs.rfunc_name}')
    )
    play_style = cfgs.play_style
    eplen = cfgs.eplen

    sac_model = load_sac_model_new(cfgs, env)

    iter_id = 1000000
    while iter_id <= 1000000:
        for run_id in range(206, 211):
            sac_model.load4pop(path=path, run_id=run_id, iter_id=iter_id)
            # for parameters in sac_model.netA.parameters():
                # print(parameters)
            pretrain_eval(run_id, iter_id, sac_model, play_style, eplen, rfunc, path)
        iter_id = iter_id + 10000
    # load_exp = 1
    # start_iter = 9
    # end_iter = 100
    # list_seed = 23456
    # device = 'cuda:0'
    # archive = 'da'
    # for load_iter in range(start_iter, end_iter, 10):
    #     # saved_ca_path = path+f'/training_exp/experiment{load_exp}/iteration{load_iter}/morl_pop/ca'
    #     saved_da_path = path+f'/experiment{load_exp}/iteration{load_iter}/morl_pop/da'
    #     for pop_ind in range(pop_size):
            
    #         # loaded_model = load_morl_model(saved_ca_path, sac_model, load_iter, pop_ind, list_seed)
    #         loaded_model = load_morl_model(saved_da_path, sac_model, load_iter, pop_ind, list_seed)
    #         morl_eval(path, loaded_model, load_exp, load_iter, pop_ind, play_style, eplen, rfunc, device, archive)
        
    print("END OF EVALUATION")

