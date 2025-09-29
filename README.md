# MOPCGRL: Multi-objective Procedural Content Generation via Reinforcement Learning

This repository provides an implementation of an MORL framework for the paper **Multi-Objective Procedural Content Generation via Reinforcement Learning (PCGRL)**. It combines **Soft Actor-Critic (SAC)** with **Multi-Objective Evolutionary Algorithms (MOEAs)** to evolve game content generators under multiple objectives and constraints.

Please consider citing this work if you use this repository:
```
@article{yuan2025multi-objective,
  author={Yi Yuan, Qingquan Zhang, Bo Yuan, Matthew Barthet, Ahmed Khalifa, Georgios N. Yannakakis, Huanhuan Chen and Jialin Liu},
  journal={Complex System Modeling and Simulation}, 
  title={MOPCGRL: Multi-objective Procedural Content Generation via Reinforcement Learning}, 
  year={2025},
  volume={},
  number={},
  pages={1-18},
  doi={}}
```

### Requirements

The packages and their versions of this code are listed in the `requirements.txt`. The environment configuration is based on the settings of [MFEDRL](https://github.com/SUSTechGameAI/MFEDRL), and the pretrained EGSAC models used here are also from that repository.

### Project Structure

The multi-objective training code is located in the `src/morl/` folder. Below are the key files and their descriptions:

- **train_morl.py**  
  Main framework for training multi-objective reinforcement learning (MORL) generator population. It initializes environments, algorithms, and handles the training loop.

- **train_morl_load.py**  
  Similar to `train_morl.py`, but designed to resume training from certain saved populations.

- **model_evaluation.py**    
  Contains evaluation routines for trained MORL models, including performance metrics and testing across multiple objectives.

- **sac_population.py**  
  The popualation definition for multi-objective learning.

- **c_taea.py**  
  Code for constrained multi-objective optimization, integrating evolutionary algorithm components (e.g., environmental selection operator, crossover operator, mutation operator).

- **plot_level.py**, **plot_origin_lvl.py**, **plot.py**  
  Visualization scripts for analyzing training results and generated levels.

- **run_commands.py**  
  Utility script for executing multiple training/evaluation commands, useful for large-scale experiments.

- **testing_segs.py**  
  Used for generating, storing and loading testing segments in the experiments.

#### Train an EGSAC model:
```bash
At the root path of this project> python train.py designer
```
#### Run an MORL experiment:
```bash
At the root path of this project> python src/morl/train_morl.py
```

### Example results

During training, each iteration saves population information and models automatically.
Example structure:
```bash
results/
└── experiment0/
    ├── ...
    ├── iteration1/
    │   ├── morl_pop/ca
    │   └── morl_pop/da
    ├── iteration0/
    │   ├── morl_pop/ca
    │   └── morl_pop/da
    └── exp_setting.txt
```