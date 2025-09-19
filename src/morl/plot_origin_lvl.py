import sys
import os
from torch import tensor

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(src_path)
from smb import *

w = MarioLevel.default_seg_width
lvl = MarioLevel.from_file('../MOPCGRL/levels/original/mario-8-1.smblv')
segs = [lvl[:, s: s + w] for s in range(0, lvl.w, w)]