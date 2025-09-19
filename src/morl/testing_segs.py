import numpy as np
import json

def gen_testing_segs():
    test_size = 100
    nz = 20
    np.random.seed(2025)
    testing_segs = np.random.rand(test_size, 4, nz).astype(np.float32) * 2 - 1
    np.save("../MOPCGRL/testing_segs_100.npy", testing_segs)
    print(testing_segs)

def load_testing_segs():
    d_eval_total = np.load('../MOPCGRL/testing_segs_100.npy')
    return d_eval_total