"""
  @Time : 2021/11/29 19:56 
  @Author : Ziqi Wang
  @File : sac.py 
"""
from src.gan.gan_use import *
import os
import time
from torch import tensor
import torch
import numpy as np
import torch.nn.functional as F
from stable_baselines3.common.utils import polyak_update
from src.algo.replay_memory import ReplayMem
from src.gan.gan_config import nz
from src.utils.filesys import get_path
from src.utils.img import make_img_sheet
from src.utils.img import vsplit_img
import pickle

class SAC_Model:
    def __init__(self, netA_builder, netQ_builder, gamma=0.99, tau=0.005, tar_entropy=-nz, device='cuda:0'):
        self.netA = netA_builder().to(device)
        self.netQ1 = netQ_builder().to(device)
        self.netQ2 = netQ_builder().to(device)
        self.netA_optimizer = torch.optim.Adam(self.netA.parameters(), 3e-4)
        self.netQ1_optimizer = torch.optim.Adam(self.netQ1.parameters(), 3e-4)
        self.netQ2_optimizer = torch.optim.Adam(self.netQ2.parameters(), 3e-4)

        self.tar_netQ1 = netQ_builder().to(device)
        self.tar_netQ2 = netQ_builder().to(device)
        self.tar_netQ1.load_state_dict(self.netQ1.state_dict())
        self.tar_netQ2.load_state_dict(self.netQ2.state_dict())
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.log_alpha = torch.tensor([1], dtype=torch.float, device=device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], 3e-4)
        self.tar_entropy = torch.tensor([tar_entropy], device=device, requires_grad=False)

    def make_decision(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            if torch.is_tensor(obs):
                a, _ = self.netA(obs.clone().detach(), with_logprob=False)
            else:
                a, _ = self.netA(torch.tensor(obs, device=self.device), with_logprob=False)
                
            a = a.to('cpu').numpy()
        return a.astype(np.float32)

    def update(self, batch):
        s, a, _, _, _ = batch
        y = self.process_batch(batch)
        self.update_critic(s, a, y)
        self.update_actor(s)
        self.update_alpha(s)
        self.update_tar_nets()

    def process_batch(self, batch):
        # s, ztraces, r, sp, d = batch
        s, _, r, sp, _ = batch
        with torch.no_grad():
            alpha = torch.exp(self.log_alpha)
            ap, log_ap = self.netA(sp)
            tar_q1 = self.tar_netQ1(sp, ap)
            tar_q2 = self.tar_netQ2(sp, ap)
            tar_q = torch.min(tar_q1, tar_q2).squeeze()
            # The terminate is fake, thus no (1-d) is multiplied
            y = r + self.gamma * (tar_q - alpha * log_ap)
        return y.float().unsqueeze(-1)

    def update_critic(self, s_batch, a_batch, y):
        self.netQ1_optimizer.zero_grad()
        self.netQ2_optimizer.zero_grad()
        q1_loss = F.mse_loss(self.netQ1(s_batch, a_batch), y)
        q2_loss = F.mse_loss(self.netQ2(s_batch, a_batch), y)
        q1_loss.backward()
        q2_loss.backward()
        self.netQ1_optimizer.step()
        self.netQ2_optimizer.step()

    def update_actor(self, s_batch):
        alpha = torch.exp(self.log_alpha)
        a, log_a = self.netA(s_batch)
        value_a = torch.min(self.netQ1(s_batch, a), self.netQ2(s_batch, a))
        self.netA_optimizer.zero_grad()
        a_loss = (alpha * log_a - value_a).mean()
        a_loss.backward()
        self.netA_optimizer.step()
        pass

    def update_alpha(self, s_batch):
        with torch.no_grad():
            self.alpha_optimizer.zero_grad()
        a, log_a = self.netA(s_batch)
        loss_alpha = -(self.log_alpha * (log_a + self.tar_entropy).detach()).mean()
        loss_alpha.backward()
        self.alpha_optimizer.step()
        pass

    def update_tar_nets(self):
        polyak_update(self.netQ1.parameters(), self.tar_netQ1.parameters(), self.tau)
        polyak_update(self.netQ2.parameters(), self.tar_netQ2.parameters(), self.tau)

    def save(self, path, fmt='%s', only_actor=True):
        torch.save(self.netA, path + '/' + fmt % 'actor' + '.pth')
        if not only_actor:
            torch.save(self.netQ1, path + '/' + fmt % 'critic1' + '.pth')
            torch.save(self.netQ2, path + '/' + fmt % 'critic2' + '.pth')
            torch.save(self.tar_netQ1, path + '/' + fmt % 'tar_critic1' + '.pth')
            torch.save(self.tar_netQ2, path + '/' + fmt % 'tar_critic2' + '.pth')
            
            # save optimizer
    
    def save4pop(self, path='model4pop', run_id=0, iter_id=0, rep_mem=None):
        
        torch.save({
            'run_id': run_id,
            'iter_id': iter_id,
            'model_state_dict': self.log_alpha,
            'optimizer_state_dict': self.alpha_optimizer.state_dict()
        }, path+f'/log_alpha_r{run_id}_i{iter_id}.pth')
        
        torch.save({
            'run_id': run_id,
            'iter_id': iter_id,
            'model_state_dict': self.netA.state_dict(),
            'optimizer_state_dict': self.netA_optimizer.state_dict()
        }, path+f'/netA_r{run_id}_i{iter_id}.pth')
        
        torch.save({
            'run_id': run_id,
            'iter_id': iter_id,
            'model_state_dict': self.netQ1.state_dict(),
            'optimizer_state_dict': self.netQ1_optimizer.state_dict()
        }, path+f'/netQ1_r{run_id}_i{iter_id}.pth')
        
        torch.save({    
            'run_id': run_id,
            'iter_id': iter_id,
            'model_state_dict': self.netQ2.state_dict(),
            'optimizer_state_dict': self.netQ2_optimizer.state_dict()
        }, path+f'/netQ2_r{run_id}_i{iter_id}.pth')
        
        torch.save({
            'run_id': run_id,
            'iter_id': iter_id,
            'model_state_dict': self.tar_netQ1.state_dict(),
        }, path+f'/tar_netQ1_r{run_id}_i{iter_id}.pth')
        
        torch.save({
            'run_id': run_id,
            'iter_id': iter_id,
            'model_state_dict': self.tar_netQ2.state_dict(),
        }, path+f'/tar_netQ2_r{run_id}_i{iter_id}.pth')
        
        # with open(path+f"/rep_mem_r{run_id}_i{iter_id}.pkl", 'wb') as f:
        #     pickle.dump(rep_mem, f)
        
    
    def load4pop(self, path='model4pop', run_id=0, iter_id=0):
        # def print_optimizer_device(name, optimizer):
        #     print(f"{name} optimizer device(s):")
        #     for i, group in enumerate(optimizer.param_groups):
        #         for j, p in enumerate(group['params']):
        #             print(f"  group {i}, param {j}: {p.device}")
        
        # def _print_device_info(name, obj, is_optimizer=False):
        #     """打印设备信息的工具函数"""
        #     # 检查模型参数
        #     if hasattr(obj, 'parameters'):
        #         params = list(obj.parameters())
        #         if len(params) > 0:
        #             print(f"[检查模型] {name}: 设备 -> {params[0].device}")
        #         else:
        #             print(f"[检查模型] {name}: 无参数")
        #     else:
        #         print(f"[检查张量] {name}: 设备 -> {obj.device}")

        # ----------------------------
        # 1. 加载前的设备检查（可选）
        # ----------------------------
        # print("\n=== 加载前的设备状态 ===")
        # _print_device_info("log_alpha", self.log_alpha)
        # _print_device_info("netA", self.netA)
        # _print_device_info("netQ1", self.netQ1)
        # _print_device_info("netQ2", self.netQ2)
        # _print_device_info("tar_netQ1", self.tar_netQ1)
        # _print_device_info("tar_netQ2", self.tar_netQ2)
        # print_optimizer_device("Alpha", self.alpha_optimizer)
        # print_optimizer_device("NetA", self.netA_optimizer)
        # print_optimizer_device("NetQ1", self.netQ1_optimizer)
        # print_optimizer_device("NetQ2", self.netQ2_optimizer)
        
        # load log_alpha
        log_alpha_state = torch.load(path+f'/log_alpha_r{run_id}_i{iter_id}.pth', map_location=self.device)
        # log_alpha_state = torch.load(path+f'/log_alpha_r{run_id}_i{iter_id}.pth')
        self.log_alpha = log_alpha_state['model_state_dict']
        self.alpha_optimizer.load_state_dict(log_alpha_state['optimizer_state_dict'])
        
        # load netA
        netA_state = torch.load(path+f'/netA_r{run_id}_i{iter_id}.pth', map_location=self.device)
        # netA_state = torch.load(path+f'/netA_r{run_id}_i{iter_id}.pth')
        self.netA.load_state_dict(netA_state['model_state_dict'])
        self.netA_optimizer.load_state_dict(netA_state['optimizer_state_dict'])
        
        # load netQ1
        netQ1_state = torch.load(path+f'/netQ1_r{run_id}_i{iter_id}.pth', map_location=self.device)
        # netQ1_state = torch.load(path+f'/netQ1_r{run_id}_i{iter_id}.pth')
        self.netQ1.load_state_dict(netQ1_state['model_state_dict'])
        self.netQ1_optimizer.load_state_dict(netQ1_state['optimizer_state_dict'])
        
        # load netQ2
        netQ2_state = torch.load(path+f'/netQ2_r{run_id}_i{iter_id}.pth', map_location=self.device)
        # netQ2_state = torch.load(path+f'/netQ2_r{run_id}_i{iter_id}.pth')
        self.netQ2.load_state_dict(netQ2_state['model_state_dict'])
        self.netQ2_optimizer.load_state_dict(netQ2_state['optimizer_state_dict'])
        
        # load tar_netQ1
        tar_netQ1_state = torch.load(path+f'/tar_netQ1_r{run_id}_i{iter_id}.pth', map_location=self.device)
        # tar_netQ1_state = torch.load(path+f'/tar_netQ1_r{run_id}_i{iter_id}.pth')
        self.tar_netQ1.load_state_dict(tar_netQ1_state['model_state_dict'])
        
        # load tar_netQ2
        tar_netQ2_state = torch.load(path+f'/tar_netQ2_r{run_id}_i{iter_id}.pth', map_location=self.device)
        # tar_netQ2_state = torch.load(path+f'/tar_netQ2_r{run_id}_i{iter_id}.pth')
        self.tar_netQ2.load_state_dict(tar_netQ2_state['model_state_dict'])

        # self.netA.to(self.device)
        # self.netQ1.to(self.device)
        # self.netQ2.to(self.device)
        # self.tar_netQ1.to(self.device)
        # self.tar_netQ2.to(self.device)

        # models = [self.netA, self.netQ1, self.netQ2, self.tar_netQ1, self.tar_netQ2]
        # for model in models:
        #     model.to(self.device)
        
        # optimizers = [self.alpha_optimizer, self.netA_optimizer, 
        #             self.netQ1_optimizer, self.netQ2_optimizer]
        # for optim in optimizers:
        #     for state in optim.state.values():
        #         for k, v in state.items():
        #             if isinstance(v, torch.Tensor):
        #                 state[k] = v.to(self.device)

        # print("\n=== 加载后的设备状态 ===")
        # _print_device_info("log_alpha", self.log_alpha)
        # _print_device_info("netA", self.netA)
        # _print_device_info("netQ1", self.netQ1)
        # _print_device_info("netQ2", self.netQ2)
        # _print_device_info("tar_netQ1", self.tar_netQ1)
        # _print_device_info("tar_netQ2", self.tar_netQ2)
        # print_optimizer_device("Alpha", self.alpha_optimizer)
        # print_optimizer_device("NetA", self.netA_optimizer)
        # print_optimizer_device("NetQ1", self.netQ1_optimizer)
        # print_optimizer_device("NetQ2", self.netQ2_optimizer)
    
    
class AsyncGenerativeSAC_Trainer:
    def __init__(self, update_itv=5, update_ratio=5, batch_size=256, rep_mem=None):
        # self.update_freq=update_freq
        self.update_itv = update_itv
        self.update_ratio = update_ratio
        self.batch_size = batch_size
        self.rep_mem = ReplayMem() if rep_mem is None else rep_mem
        pass

    def train(self, model, env, step_budget, save_path='./', check_points=None):
        steps =  0
        update_counter = 0
        # self.rep_mem.clear()

        miss_counter = 0
        start_time = time.time()
        ep = 0
        print('Start to train AGSAC')
        while steps < step_budget:
            ep += 1
            done = False
            obs = env.reset()
            while not done:
                action = model.make_decision(np.expand_dims(obs, 0))
                next_ob, done = env.step(action)
                update_counter += 1
                steps += 1
            transitions = env.roll_out()
            for t in transitions:
                self.rep_mem.add(*t)
            # self.rep_mem.add_trajectories(*transitions)
            if len(self.rep_mem) < self.batch_size:
                miss_counter += update_counter
                update_counter = 0
            elif ep % self.update_itv == 0:
                repeats = update_counter // self.update_ratio
                for _ in range(repeats):
                    batch_data = self.rep_mem.sample(self.batch_size, device=model.device)
                    model.update(batch_data)
                    env.logger.on_model_update()
                update_counter %= self.update_ratio

            if check_points and steps >= check_points[0]:
                check_point_path = get_path(save_path + f'/model_at_{steps}')
                os.makedirs(check_point_path, exist_ok=True)
                model.save(check_point_path)
                check_points.pop(0)

        transitions = env.close()
        for t in transitions:
            self.rep_mem.add(*t)
        repeats = miss_counter // self.update_ratio
        print(f'Update model for {repeats} times compensatorily')
        for _ in range(repeats):
            batch_data = self.rep_mem.sample(self.batch_size, device=model.device)
            model.update(batch_data)
        print(f'Training finished in {time.time()-start_time}s')
        model.save(save_path)
        pass


def plot_levels(levels, save_path=f'test1.png'):
    if isinstance(levels, list):
        total_lvls = levels[0]
        for lel_id in range(1, len(levels)):
            total_lvls += levels[lel_id]
    else:
        total_lvls = levels
        
    img = total_lvls.to_img(save_path=save_path)
    seg_path = save_path.split(".")
    save_path_new = f"{seg_path[0]}_seg.{seg_path[1]}"
    vsplit_img(img, save_path=save_path_new)


class OffRewSAC_Trainer:
    def __init__(self, env, step_budget, update_freq=10, batch_size=384, rep_mem=None, save_path = '.', check_points = None, run_id=0):
        self.env = env
        self.n_parallel = env.num_envs
        self.step_budget = step_budget
        self.update_freq = update_freq
        self.batch_size = batch_size
        self.rep_mem = ReplayMem() if rep_mem is None else rep_mem
        self.steps = 0
        self.check_points = [] if not check_points else check_points
        # self.check_points.sort(reverse=True)
        self.check_points = [100, 500, 5000]
        self.save_path = save_path
        self.run_id = run_id
        pass
    
    def init_first_seg(self):
        obs = self.env.reset()

    def train(self, model):
        self.steps = 0

        obs = self.env.reset()
        print('Start to train SAC')
        obs_buffer = [[] for _ in range(self.env.num_envs)]
        action_buffer = [[] for _ in range(self.env.num_envs)]
        next_obs_buffer = [[] for _ in range(self.env.num_envs)]
        new_transitions = 0
        while self.steps < self.step_budget:
            print(f"steps: {self.steps}")
            actions = model.make_decision(obs) # 20 (len of latent vector) * 4
            next_obs, _, dones, infos = self.env.step(actions)
            
            # for action_, obs_ in zip(actions, obs):
            #     z = torch.clamp(tensor(obs_.astype(np.float32), device=self.env.device), -1, 1).view(-1, nz, 1, 1)
            #     levels_obv = process_levels(self.env.generator(z), True)
            #     z = torch.clamp(tensor(action_.astype(np.float32), device=self.env.device), -1, 1).view(-1, nz, 1, 1)
            #     levels_act = process_levels(self.env.generator(z), True)
                
            #     # plot_levels(levels_obv+levels_act, save_path="abc.png")
                
            #     pass
            
            for i in range(len(next_obs)):
                if dones[i]:
                    next_obs[i] = infos[i]['terminal_observation']
            for i, (ob, action, next_ob) in enumerate(zip(obs, actions, next_obs)):
                obs_buffer[i].append(ob)
                action_buffer[i].append(action)
                next_obs_buffer[i].append(next_ob)

            del obs
            obs = next_obs
            self.steps += self.n_parallel
            for i, (done, info) in enumerate(zip(dones, infos)):
                if not done:
                    continue
                print("enter the loop")
                reward_lists = []
                for key in info.keys():
                    if 'reward_list' not in key:
                        continue
                    reward_lists.append(info[key])
                rewards = []

                for j in range(len(reward_lists[0])):
                    step_reward = 0
                    for item in reward_lists:
                        step_reward += item[j]
                    rewards.append(step_reward)
                self.rep_mem.add_batched(
                    obs_buffer[i], action_buffer[i], rewards, next_obs_buffer[i],
                    [False] * (len(reward_lists[0]) - 1) + [True]
                )
                obs_buffer[i].clear()
                action_buffer[i].clear()
                next_obs_buffer[i].clear()

                new_transitions += len(reward_lists[0])
            if new_transitions > self.update_freq and len(self.rep_mem) > self.batch_size:
                update_times = new_transitions // self.update_freq
                for _ in range(update_times):
                    batch_data = self.rep_mem.sample(self.batch_size, device=model.device)
                    model.update(batch_data)   # update parameters of actor critic
                new_transitions = new_transitions % self.update_freq
                # print(len(self.rep_mem))

            if len(self.check_points) and self.steps >= self.check_points[-1]:
                check_point_path = get_path(self.save_path + f'/model_at_{self.steps}')
                os.makedirs(check_point_path, exist_ok=True)
                model.save(check_point_path)
                self.check_points.pop()
                pass

            # model.save(self.save_path)
            # if (np.mod(self.steps, 50000) == 0) | (self.steps == 1):
            if 200000 <= self.steps <= 290000 and self.steps % 10000 == 0:
            # if self.steps in [250000, 290000] and self.steps % 10000 == 0:
                model.save4pop(run_id=self.run_id, iter_id=self.steps, rep_mem=self.rep_mem)
            # model.load4pop(run_id=self.run_id, iter_id=self.steps)
            if self.steps == 1000000:
                model.save4pop(run_id=self.run_id, iter_id=self.steps, rep_mem=self.rep_mem)
        pass

    def pretrain(self, model, itera):

        obv = []
        self.steps = 0

        obs = self.env.reset()
        print('Start to train SAC')
        obs_buffer = [[] for _ in range(self.env.num_envs)]
        action_buffer = [[] for _ in range(self.env.num_envs)]
        next_obs_buffer = [[] for _ in range(self.env.num_envs)]
        new_transitions = 0
        while self.steps < itera:
            actions = model.make_decision(obs) # 20 (len of latent vector) * 4
            next_obs, _, dones, infos = self.env.step(actions)
            
            for action_, obs_ in zip(actions, obs):
                z = torch.clamp(tensor(obs_.astype(np.float32), device=self.env.device), -1, 1).view(-1, nz, 1, 1)
                levels_obv = process_levels(self.env.generator(z), True)
                z = torch.clamp(tensor(action_.astype(np.float32), device=self.env.device), -1, 1).view(-1, nz, 1, 1)
                levels_act = process_levels(self.env.generator(z), True)
                
                # plot_levels(levels_obv+levels_act, save_path="abc.png")
                
                pass
            
            for i in range(len(next_obs)):
                if dones[i]:
                    next_obs[i] = infos[i]['terminal_observation']
            for i, (ob, action, next_ob) in enumerate(zip(obs, actions, next_obs)):
                obs_buffer[i].append(ob)
                action_buffer[i].append(action)
                next_obs_buffer[i].append(next_ob)

            del obs
            obs = next_obs
            self.steps += self.n_parallel
            for i, (done, info) in enumerate(zip(dones, infos)):
                if not done:
                    continue
                print("enter the loop")
                reward_lists = []
                for key in info.keys():
                    if 'reward_list' not in key:
                        continue
                    reward_lists.append(info[key])
                rewards = []

                for j in range(len(reward_lists[0])):
                    step_reward = 0
                    for item in reward_lists:
                        step_reward += item[j]
                    rewards.append(step_reward)
                self.rep_mem.add_batched(
                    obs_buffer[i], action_buffer[i], rewards, next_obs_buffer[i],
                    [False] * (len(reward_lists[0]) - 1) + [True]
                )
                obs_buffer[i].clear()
                action_buffer[i].clear()
                next_obs_buffer[i].clear()

                new_transitions += len(reward_lists[0])
            if new_transitions > self.update_freq and len(self.rep_mem) > self.batch_size:
                update_times = new_transitions // self.update_freq
                for _ in range(update_times):
                    batch_data = self.rep_mem.sample(self.batch_size, device=model.device)
                    model.update(batch_data)   # update parameters of actor critic
                new_transitions = new_transitions % self.update_freq
                print(len(self.rep_mem))


        # model.save(self.save_path)
        
        return self.rep_mem

    def train_model(self, model, itera, rep_mem):
        print("train_model")

        obv = []
        self.steps = 0

        obs = self.env.reset()
        print('Start to train SAC')
        obs_buffer = [[] for _ in range(self.env.num_envs)]
        action_buffer = [[] for _ in range(self.env.num_envs)]
        next_obs_buffer = [[] for _ in range(self.env.num_envs)]
        new_transitions = 0
        while self.steps < itera:
            # print(f"steps: {self.steps}")
            actions = model.make_decision(obs) # 20 (len of latent vector) * 4
            next_obs, _, dones, infos = self.env.step(actions)
            #     pass
            
            for i in range(len(next_obs)):
                if dones[i]:
                    next_obs[i] = infos[i]['terminal_observation']
            for i, (ob, action, next_ob) in enumerate(zip(obs, actions, next_obs)):
                obs_buffer[i].append(ob)
                action_buffer[i].append(action)
                next_obs_buffer[i].append(next_ob)

            del obs
            obs = next_obs
            self.steps += self.n_parallel
            for i, (done, info) in enumerate(zip(dones, infos)):
                if not done:
                    continue
                # print("enter the loop")
                reward_lists = []
                for key in info.keys():
                    if 'reward_list' not in key:
                        continue
                    reward_lists.append(info[key])
                rewards = []

                for j in range(len(reward_lists[0])):
                    step_reward = 0
                    for item in reward_lists:
                        # reward function
                        step_reward += item[j]
                    rewards.append(step_reward)
                rep_mem.add_batched(
                    obs_buffer[i], action_buffer[i], rewards, next_obs_buffer[i],
                    [False] * (len(reward_lists[0]) - 1) + [True]
                )
                obs_buffer[i].clear()
                action_buffer[i].clear()
                next_obs_buffer[i].clear()

                new_transitions += len(reward_lists[0])
            if new_transitions > self.update_freq and len(rep_mem) > self.batch_size:
                update_times = new_transitions // self.update_freq
                for _ in range(update_times):
                    batch_data = rep_mem.sample(self.batch_size, device=model.device)
                    model.update(batch_data)   # update parameters of actor critic
                new_transitions = new_transitions % self.update_freq
                # print(len(rep_mem))


        # model.save(self.save_path)
        
        return rep_mem


    def train_model_new(self, model, itera, rep_mem):
        print("train_model_new")
        obv = []
        self.steps = 0

        obs = self.env.reset()
        print('Start to train SAC')
        obs_buffer = [[] for _ in range(self.env.num_envs)]
        action_buffer = [[] for _ in range(self.env.num_envs)]
        next_obs_buffer = [[] for _ in range(self.env.num_envs)]
        new_transitions = 0
        while self.steps < itera:
            # print(f"steps: {self.steps}")
            actions = model.make_decision(obs) # 20 (len of latent vector) * 4
            next_obs, _, dones, infos = self.env.step(actions)
            
            for i in range(len(next_obs)):
                if dones[i]:
                    next_obs[i] = infos[i]['terminal_observation']
            for i, (ob, action, next_ob) in enumerate(zip(obs, actions, next_obs)):
                obs_buffer[i].append(ob)
                action_buffer[i].append(action)
                next_obs_buffer[i].append(next_ob)

            del obs
            obs = next_obs
            self.steps += self.n_parallel
            for i, (done, info) in enumerate(zip(dones, infos)):
                if not done:
                    continue
                # print("enter the loop")
                reward_lists = []
                for key in info.keys():
                    if 'reward_list' not in key:
                        continue
                    reward_lists.append(info[key])
                rewards = []

                reward_weight = np.random.uniform(0, 1, 3)
                reward_weight = 3 * reward_weight / np.sum(reward_weight)
                for j in range(len(reward_lists[0])):
                    step_reward = 0
                    for w in range(len(reward_weight)):
                        step_reward += reward_lists[w][j] * reward_weight[w]
                    rewards.append(step_reward)
                rep_mem.add_batched(
                    obs_buffer[i], action_buffer[i], rewards, next_obs_buffer[i],
                    [False] * (len(reward_lists[0]) - 1) + [True]
                )
                obs_buffer[i].clear()
                action_buffer[i].clear()
                next_obs_buffer[i].clear()

                new_transitions += len(reward_lists[0])
            if new_transitions > self.update_freq and len(rep_mem) > self.batch_size:
                update_times = new_transitions // self.update_freq
                for _ in range(update_times):
                    batch_data = rep_mem.sample(self.batch_size, device=model.device)
                    model.update(batch_data)   # update parameters of actor critic
                new_transitions = new_transitions % self.update_freq
                # print(len(rep_mem))


        # model.save(self.save_path)
        
        return rep_mem

class SAC_Trainer:
    def __init__(
        self, env, step_budget, update_itv=500, update_repeats=100, update_start=None, batch_size=256,
        rep_mem=None, save_path = '.', check_points = None
    ):
        self.env = env
        self.n_parallel = env.num_envs
        self.step_budget = step_budget
        self.update_itv  = update_itv
        self.update_start = batch_size if update_start is None else update_start
        self.update_repeats = update_repeats
        self.batch_size = batch_size
        self.rep_mem = ReplayMem() if rep_mem is None else rep_mem
        self.steps = 0
        self.check_points = [] if not check_points else check_points
        self.check_points.sort(reverse=True)
        self.save_path = save_path
        pass

    def train(self, model):
        self.steps = 0
        update_wait = self.update_start
        obs = self.env.reset()
        print('Start to train SAC')
        while self.steps < self.step_budget:
            actions = model.make_decision(obs)
            next_obs, rewards, dones, infos = self.env.step(actions)
            self.rep_mem.add_batched(obs, actions, rewards, next_obs, dones)

            del obs
            obs = next_obs
            self.steps += self.n_parallel
            update_wait -= self.n_parallel
            if update_wait <= 0:
                for _ in range(self.update_repeats):
                    batch_data = self.rep_mem.sample(self.batch_size, device=model.device)
                    model.update(batch_data)
                update_wait = self.update_itv
            if len(self.check_points) and self.steps >= self.check_points[-1]:
                check_point_path = self.save_path + f'/model_at_{self.steps}'
                os.makedirs(check_point_path, exist_ok=True)
                model.save(check_point_path)
                self.check_points.pop()
                pass

        model.save(self.save_path)
        pass

