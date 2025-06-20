import subprocess
import time


f_values = [0, 3, 6, 9]
m_values = [32, 64]
alpha_values = [1000]


for m in m_values:
    for alpha in alpha_values:
        for f in f_values:
            try:
                command = f"python3 main.py --dataset mnist --heterogeneity homogeneous --n 20 --m {m} --f {f} --T 100 --model cnn --lr 0.05 --attack auto_FOE --batch_size {m} --alpha {alpha} --nb_main_client 1 --nb_run 5 --nb_classes 10"
                subprocess.run(command, shell=True)
            except Exception as e:
                print("Error with values", f,alpha, e)
            
