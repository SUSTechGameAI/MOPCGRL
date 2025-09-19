import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib.legend_handler import HandlerTuple

def load_egsac_test(path, start_id, end_id, start_step ,end_step):
    aver_fc = []
    aver_fb = []
    for run_id in range(start_id, end_id):
        iter_id = start_step
        while iter_id <= end_step:
            with open(path+f'/test_20w/model_aver_r{run_id}_i{iter_id}_g1.json', 'r', encoding='utf-8') as f:
                aver_json = json.load(f)
            aver_fc.append(aver_json[0]["Fun Content Average"])
            aver_fb.append(aver_json[0]["Fun Behaviour Average"])
            iter_id = iter_id + 10000
    aver_fc = np.array(aver_fc)
    aver_fb = np.array(aver_fb)
    return aver_fc, aver_fb

def load_mo_test(path, iteration, exp_num, archive, pop_size):
    aver_fc = []
    aver_fb = []
    for i in range(pop_size):
        with open(path+f'/training_exp/experiment{exp_num}/testing_{archive}/model_aver_iter{iteration}_ind{i}.json', 'r', encoding='utf-8') as f:
            aver_json = json.load(f)
        aver_fc.append(aver_json[0]["Fun Content Average"])
        aver_fb.append(aver_json[0]["Fun Behaviour Average"])

    aver_fc = np.array(aver_fc)
    aver_fb = np.array(aver_fb)
    return aver_fc, aver_fb

# load from interation to end_iteration
def load_mo_test2(path, iteration, end_iteration, exp_num, pop_size): 
    aver_fc = []
    aver_fb = []
    for iter in range(iteration, end_iteration, 10):
        model_fc = []
        model_fb = []
        for i in range(pop_size):
            with open(path+f'/training_exp/experiment{exp_num}/testing_{archive}/model_aver_iter{iter}_ind{i}.json', 'r', encoding='utf-8') as f:
                aver_json = json.load(f)    
            model_fc.append(aver_json[0]["Fun Content Average"])
            model_fb.append(aver_json[0]["Fun Behaviour Average"])
        aver_fc.append(np.mean(model_fc))
        aver_fb.append(np.mean(model_fb))

    aver_fc = np.array(aver_fc)
    aver_fb = np.array(aver_fb)
    return aver_fc, aver_fb

def load_pretrain():
    aver_fc = []
    aver_fb = []

    iter_id = 1000000
    while iter_id <= 1000000:
        for run_id in range(1, 6):
            for g in range(1, 6):
                if iter_id <= 790000:
                    with open(f'pretrain_evaluation/average/model_r{run_id}_i{iter_id}_g{g}.json', 'r', encoding='utf-8') as f2:
                        aver_json = json.load(f2)
                else:
                    with open(f'pretrain_evaluation/average_m/model_r{run_id}_i{iter_id}_g{g}.json', 'r', encoding='utf-8') as f2:
                        aver_json = json.load(f2)
                aver_fc.append(1-aver_json[0]["Fun Content Average"])
                aver_fb.append(1-aver_json[0]["Fun Behaviour Average"])

        iter_id = iter_id + 10000
    aver_fc = np.array(aver_fc)
    aver_fb = np.array(aver_fb)
    return aver_fc, aver_fb


def load_mo_train(path, iteration, exp_num, seed_num, archive):
    aver_fc = []
    aver_fb = []
    iter = 99
    while iter <= iteration:
        aver = np.loadtxt(path+f'/training_exp/experiment{exp_num}/iteration{iter}/iter_pop/{archive}/gen{iter}_test{seed_num}.txt')
        aver_fc.append(1-aver[:, 0])
        aver_fb.append(1-aver[:, 1])
        iter = iter + 1
    aver_fc = np.array(aver_fc)
    aver_fb = np.array(aver_fb)
    return aver_fc, aver_fb

def plot_all_test():
    aver_fc, aver_fb = load_mo_test2(path1, iteration, end_iteration, exp_num, pop_size)
    gen = np.arange(iteration, end_iteration, 10)
    plt.plot(gen, aver_fc, marker='o', linestyle='-', color='b', label='Game Level Diversity')
    plt.plot(gen, aver_fb, marker='s', linestyle='-', color='r', label='Gameplay Diversity')
    plt.xlabel('Generation', fontsize=14)
    plt.ylabel('Diversity Score', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.legend(fontsize=12)
    plt.savefig(path1+f'/training_exp/experiment{exp_num}/testing_{archive}/load{iteration}-{end_iteration}_ca_aver{seed_num}.png', bbox_inches='tight')
    

def plot_iter_test():
    plt.xlabel('game level divergence', fontsize=14)
    plt.ylabel('gameplay divergence', fontsize=14)
    plt.tick_params(labelsize=14)
    
    plt.legend()
    plt.savefig(path1+f'/training_exp/experiment{exp_num}/testing_{archive}/test_and_train_diversity_{iteration+1}_{partial_iter}.png', bbox_inches='tight')

# mark the area of each group
def add_ellipse(ax, x, y, facecolor='gray', alpha=0.2):
    if len(x) < 2:
        return
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * 2 * np.sqrt(vals)  # 2σ , covers 95%
    ellipse = Ellipse(xy=(np.mean(x), np.mean(y)),
                      width=width, height=height,
                      angle=theta, alpha=alpha,
                      facecolor=facecolor, edgecolor='none')
    ax.add_patch(ellipse)


pop_size = 10
path1 = '../training_data1'
path2 = '../training_data2'
exp_num = 12
iteration = 99
end_iteration = 100
seed_num = 12345
partial_iter = 5000
archive = 'da'

# egsac parameters
configs = [
    {
        'label': 'Mixed evolution warm-up',
        'color': 'black',
        'path_base': path1+'/model4pop',
        'path': path1,
        'archive': 'da',
        'exp_num': 20,
        'egsac_args': (8, 9, 200000, 290000)
    },
    {
        'label': '290k warm-up',
        'color': 'purple',
        'path_base': path1+'/model4pop',
        'path': path1,
        'archive': 'da',
        'exp_num': 32,
        'egsac_args': (21, 31, 290000, 290000)
    },
    {
        'label': '250k warm-up',
        'color': 'blue',
        'path_base': path1+'/model4pop',
        'path': path1, 
        'archive': 'da',
        'exp_num': 27,
        'egsac_args': (21, 31, 250000, 250000)
    },
     {
        'label': '200k warm-up',
        'color': 'green',
        'path_base': path1+'/model_pop',
        'path': path1,
        'archive': 'da',
        'exp_num': 2,
        'egsac_args': (21, 31, 200000, 200000)
    }
]

fig, ax = plt.subplots(figsize=(6, 5))  
legend_elements = []
for cfg in configs:
    fc, fb = load_mo_test(cfg['path'], iteration, cfg['exp_num'], cfg['archive'], pop_size)
    fc_init, fb_init = load_egsac_test(cfg['path_base'], *cfg['egsac_args'])

    ax.scatter(fc, fb, color=cfg['color'], marker='o', label=cfg['label'])
    ax.scatter(fc_init, fb_init, color=cfg['color'], marker='x')

    add_ellipse(ax, fc, fb, facecolor=cfg['color'], alpha=0.15)
    add_ellipse(ax, fc_init, fb_init, facecolor=cfg['color'], alpha=0.15)
    
    marker_o = Line2D([0], [0], color=cfg['color'], marker='o', linestyle='None')
    marker_x = Line2D([0], [0], color=cfg['color'], marker='x', linestyle='None')
    legend_elements.append(((marker_o, marker_x), cfg['label']))

ax.legend(
    handles=[item[0] for item in legend_elements],
    labels=[item[1] for item in legend_elements],
    handler_map={tuple: HandlerTuple(ndivide=None)},
    loc='best',
    fontsize=12
)
ax.grid(True)
ax.set_xlabel('Content diversity $f_c$', fontsize=14)
ax.set_ylabel('Gameplay diversity $f_g$', fontsize=14)
ax.tick_params(labelsize=14)
plt.tight_layout()
plt.savefig(f'{path1}/training_exp/pop_initial.pdf', bbox_inches='tight')

# plot_all_test()
# plot_iter_test()