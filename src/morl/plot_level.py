import sys
import os
from torch import tensor
import pickle
import importlib
import copy

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(src_path)
from smb import *
from testing_segs import load_testing_segs
from src.gan.gan_use import *
from src.designer.train_designer import load_sac_model, load_sac_model_new, get_env
from sac_population import load_morl_model
from model_evaluation import morl_eval

if __name__=='__main__':
    load_exp = 16
    load_iter = 99
    seed = 12345
    eplen = 15
    path = f'../training_exp_mix/experiment{load_exp}/iteration{load_iter}/morl_pop/da'
    plot_path = '../MOPCGRL/src/morl/analyze/best_fb'
    best_fc_ind = 5
    best_fb_ind = 2
    knee_ind = 4
    device = 'cuda:0'
    archive = 'da'
    cfgs = pickle.load(open("cfgs.pkl","rb"))
    env = get_env(cfgs)
    play_style = cfgs.play_style
    sac_model = load_sac_model_new(cfgs, env)
    rfunc = (
            importlib.import_module('src.environment.rfuncs')
            .__getattribute__(f'{cfgs.rfunc_name}')
        )
    mario_proxy = MarioProxy() if rfunc.require_simlt else None
    loaded_model = load_morl_model(path, sac_model, load_iter, best_fb_ind, seed)
    count = 0
    nz = 20
    testing_segs = load_testing_segs()
    # segs = tensor(testing_segs[0:5]).view(-1, nz, 1, 1).to(device)
    np.random.seed(12345)
    sample_segs = np.random.randint(0, 100, size=20)
    generator = get_generator(device=device)
    for d in sample_segs:
        count += 1
        seg = tensor(testing_segs[d]).view(-1, nz, 1, 1).to(device)
        st_seg = copy.deepcopy(seg)
        level_segs = copy.deepcopy(st_seg)
        for i in range(eplen):
            obs_tensor = st_seg.reshape(1, -1)
            gen_seg = sac_model.make_decision(obs_tensor)
            # turn the generated segment to tensor
            next_seg = tensor(gen_seg).view(-1, nz, 1, 1).to(device)
            # concatenate new segment to level 
            # concat_seg = torch.cat((tensor(st_seg).view(-1, self.nz, 1, 1).to(self.device), next_seg), dim=0)
            concat_seg = torch.cat((st_seg.clone().detach().view(-1, nz, 1, 1).to(device), next_seg), dim=0)
            concat_re = concat_seg.reshape(1, -1)
            level_segs = torch.cat((level_segs, next_seg), dim=0)
            # get new obs
            st_seg = concat_re[:, -80:]
        level = process_levels(generator(level_segs), True)
        full_level = level_sum(level)
        # full_level.to_img('./src/morl/analyze/render2.png')
        w = MarioLevel.default_seg_width
        segs = [full_level[:, s: s + w] for s in range(0, full_level.w, w)]
        jagent = MarioJavaAgents.__getitem__(play_style)
        simlt_k = 4. if play_style == 'Runner' else 10.
        raw_simlt_res = mario_proxy.simulate_long(level_sum(segs), jagent, simlt_k)
        simlt_res = MarioProxy.get_seg_infos(raw_simlt_res)
        trace = mario_proxy.simulate_long(full_level, MarioJavaAgents.Runner)['trace']
        full_level.to_img_with_trace(trace, plot_path+f'/render_trace{d}.png', 5)
        np.save(plot_path+f'/trace{d}.npy', trace)