import subprocess
import time
import numpy as np

def make_normal_m_list(n, m, sigma=0.3, seed=None):
    """
    Return a length-n list of positive ints drawn ~ Normal(m, sigma*m)
    but rescaled so sum == n*m.

    Parameters
    ----------
    n : int          # number of clients
    m : int          # original homogeneous size
    sigma : float    # fractional std-dev; 0.3 → σ = 0.3·m
    seed : int|None  # for reproducibility
    """
    rng = np.random.default_rng(seed)

    # 1‒3  draw, clip at 1, round
    raw = rng.normal(loc=m, scale=sigma*m, size=n)
    raw = np.clip(raw, 1, None)
    raw = np.round(raw).astype(int)

    # 4  scale to correct total
    total = raw.sum()
    target = n * m
    scaled = (raw * (target / total)).round().astype(int)

    # 5  final tidy-up (difference is at most a handful of samples)
    diff = target - scaled.sum()
    for i in range(abs(diff)):
        idx = i % n
        scaled[idx] += int(np.sign(diff))

    assert scaled.sum() == target and (scaled > 0).all()
    return scaled.tolist()

f_values = [0]
m_values = [32]
alpha_values = [0.5]



for m in m_values:
    for alpha in alpha_values:
        for f in f_values:
            m_list = make_normal_m_list(20 - f, m, sigma=0.3, seed=42)
            try:
                command = f"python3 main_dist.py --dataset mnist --heterogeneity dirichlet_mnist --n 20 --m {m} --m_list {','.join(map(str,m_list))} --f {f} --T 100 --model cnn --lr 0.05 --attack SF --batch_size {m} --alpha {alpha} --nb_main_client 1 --nb_run 5 --nb_classes 10 --algo IPGDW"
                subprocess.run(command, shell=True)
            except Exception as e:
                print("Error with values", f,alpha, e)
            
