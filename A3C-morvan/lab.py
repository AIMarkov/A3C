import math, os
import torch.multiprocessing as mp
os.environ["OMP_NUM_THREADS"]='1'
c=os.environ["OMP_NUM_THREADS"]
print(c)
print(os.environ.keys())
print(mp.cpu_count())
print(mp.Value('i', 0))
print(mp.Value('d', 0.))
print(mp.Queue())